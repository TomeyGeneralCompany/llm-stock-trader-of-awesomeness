"""Shared semantic validation for inference and sealed analysis artifacts."""

import math
import re

EXPLANATION_INSTRUCTIONS = """explanation (concise English):
- observed_state: timeframe, position (flat/long), market_phase, macd_cross, macd_alignment (above/below/equal/unknown), evidence (list). Copy controller facts; alignment is not a new crossover.
- interpretation: explain current action and conflicting evidence; distinguish timeframes and inference.
- forecast: scenario, timeframe, assumptions (nonempty list), invalidation. Future crosses are not observations.
- conditional_actions: nonempty list of action (BUY/SELL/HOLD), trigger, timeframe, required_position (flat/long/any), prerequisite, reasoning. Future SELL requires long, after entry if flat.
- risk_assessment: evidence (list of exact controller keys e.g. current.close), rationale, downside. No guaranteed safety.
- execution: mode (now/wait), evidence (same key format), rationale. BUY/SELL only now on observed evidence; unmet triggers require HOLD/wait and conditional_actions."""



class DecisionError(ValueError):
    pass


def validate_decision(decision, context=None):
    if not isinstance(decision, dict) or decision.get("action") not in (
        "BUY",
        "SELL",
        "HOLD",
    ):
        raise DecisionError("A model decision was missing or invalid.")
    confidence = decision.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        raise DecisionError(
            "A model decision requires numeric confidence between 0 and 1."
        )
    if decision.get("risk_level") not in ("low", "medium", "high"):
        raise DecisionError("A model decision requires a valid risk_level.")
    placeholders = {
        "detailed decision reasoning",
        "description of expected outcome",
        "decision reasoning",
        "expected outcome",
        "reasoning",
        "",
    }
    for key in ("reasoning", "expected_outcome"):
        value = decision.get(key)
        if (
            not isinstance(value, str)
            or not value.strip()
            or value.strip().lower().rstrip(".") in placeholders
        ):
            raise DecisionError(
                "A model decision contains missing or placeholder " + key + "."
            )
    if context is not None:
        validate_explanation(decision, context)
    return decision


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise DecisionError("A model decision requires " + label + ".")


def _controller_evidence(keys, context, label):
    if not isinstance(keys, list) or not keys:
        raise DecisionError(label + " requires controller evidence keys.")
    for key in keys:
        if not isinstance(key, str) or not key.startswith(("current.", "technical_facts.", "trend.", "macd_")):
            raise DecisionError(label + " contains an invalid evidence key.")
        value = context
        for part in key.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        if value is None or value == "unknown" or (isinstance(value, float) and not math.isfinite(value)):
            raise DecisionError(label + " cites unavailable controller evidence: " + key)


def execution_action(decision):
    """Only explicit present-tense authorization produces a trading signal."""
    if not isinstance(decision, dict):
        return "HOLD"
    execution = decision.get("explanation", {}).get("execution", {})
    action = decision.get("action")
    if action in ("BUY", "SELL") and execution.get("mode") == "now":
        return action
    return "HOLD"


def validate_explanation(decision, context):
    """Validate explicit observations and conditional planning, not free prose claims."""
    explanation = decision.get("explanation")
    if not isinstance(explanation, dict):
        raise DecisionError("A model decision requires a structured explanation.")
    observed = explanation.get("observed_state")
    if not isinstance(observed, dict):
        raise DecisionError("Explanation requires observed_state.")
    for key in ("timeframe", "position", "market_phase", "macd_cross"):
        _text(observed.get(key), "observed_state." + key)
    facts = observed.get("evidence")
    if not isinstance(facts, list) or not facts:
        raise DecisionError("Observed state requires evidence.")
    for fact in facts:
        _text(fact, "observed evidence")
    expected_position = "long" if context["shares"] > 0 else "flat"
    expected_phase = (
        context["trend"].get("market_phase")
        or context["trend"].get("dominant_trend")
        or "unknown"
    )
    for key, expected in (
        ("position", expected_position),
        ("market_phase", expected_phase),
        ("macd_cross", context["macd_cross"]),
    ):
        if observed[key] != expected:
            raise DecisionError(
                f"Observed {key} must match controller value {expected!r}; describe alternate or anticipated states in interpretation/forecast with their timeframe."
            )
    if decision["action"] == "SELL" and expected_position == "flat":
        raise DecisionError(
            "Current SELL requires an existing long position; a future SELL belongs in conditional_actions with required_position long."
        )
    _text(explanation.get("interpretation"), "interpretation")
    alignment = context.get("macd_alignment", "unknown")
    if observed.get("macd_alignment") != alignment:
        raise DecisionError("Observed MACD alignment must match controller; alignment is not a crossover.")
    # Catch explicit claims of an already occurred cross, not anticipated scenarios.
    if context["macd_cross"] == "none":
        for claim in facts + [explanation["interpretation"], decision["reasoning"]]:
            if re.search(r"\b(?:bullish|bearish|golden|death) cross(?:over)? (?:has occurred|occurred|has happened|happened|is confirmed|was confirmed)\b|\bMACD (?:has crossed|crossed) (?:above|below)\b", claim, re.I) and not re.search(r"\b(?:if|may|might|could|would|not|no|until|when)\b", claim, re.I):
                raise DecisionError("No new MACD crossover is observed; distinguish alignment from a future cross.")
    risk = explanation.get("risk_assessment")
    if not isinstance(risk, dict):
        raise DecisionError("Explanation requires risk_assessment.")
    _controller_evidence(risk.get("evidence"), context, "Risk assessment")
    for key in ("rationale", "downside"):
        _text(risk.get(key), "risk_assessment." + key)
    for claim in (risk["rationale"], decision["reasoning"], explanation["interpretation"]):
        if re.search(r"\b(?:risk[- ]free|guaranteed (?:profit|return|safe)|zero risk|no downside)\b", claim, re.I) and not re.search(r"\b(?:not|never|no guarantee)\b", claim, re.I):
            raise DecisionError("Risk rationale must not assert guaranteed safety.")
    execution = explanation.get("execution")
    if not isinstance(execution, dict):
        raise DecisionError("Explanation requires execution intent.")
    expected_mode = "wait" if decision["action"] == "HOLD" else "now"
    if execution.get("mode") != expected_mode:
        raise DecisionError("Current action must match execution mode: use HOLD for unmet future triggers.")
    _controller_evidence(execution.get("evidence"), context, "Execution")
    _text(execution.get("rationale"), "execution.rationale")
    if expected_mode == "now":
        for claim in (execution["rationale"], decision["reasoning"], explanation["interpretation"]):
            if re.search(r"\b(?:wait (?:for|until)|only (?:buy|sell|enter|exit) (?:if|when|after)|(?:buy|sell|enter|exit) only (?:if|when|after)|(?:entry|exit) (?:requires|depends on) (?:a |the )?(?:future|confirmation|breakout))\b", claim, re.I):
                raise DecisionError("Current order cannot depend on an unmet trigger; use HOLD and conditional_actions.")
    forecast = explanation.get("forecast")
    if not isinstance(forecast, dict):
        raise DecisionError("Explanation requires a forecast object.")
    for key in ("scenario", "timeframe", "invalidation"):
        _text(forecast.get(key), "forecast." + key)
    assumptions = forecast.get("assumptions")
    if not isinstance(assumptions, list) or not assumptions:
        raise DecisionError("Forecast requires explicit assumptions.")
    for assumption in assumptions:
        _text(assumption, "forecast assumption")
    actions = explanation.get("conditional_actions")
    if not isinstance(actions, list) or not actions:
        raise DecisionError("Explanation requires conditional_actions.")
    for plan in actions:
        if not isinstance(plan, dict) or plan.get("action") not in (
            "BUY",
            "SELL",
            "HOLD",
        ):
            raise DecisionError("Conditional action must be BUY, SELL, or HOLD.")
        for key in ("trigger", "timeframe", "prerequisite", "reasoning"):
            _text(plan.get(key), "conditional action " + key)
        if plan.get("required_position") not in ("flat", "long", "any"):
            raise DecisionError(
                "Conditional action requires position flat, long, or any."
            )
        if plan["action"] == "SELL" and plan["required_position"] != "long":
            raise DecisionError(
                "Conditional SELL requires position long, including a prior entry when currently flat."
            )


def format_decision_reasoning(decision):
    """Keep the model's explanation readable in existing log and chart views."""
    text = decision.get("reasoning", "")
    explanation = decision.get("explanation")
    if not explanation:
        return text
    observed = explanation["observed_state"]
    forecast = explanation["forecast"]
    text += (
        f"\n\nObserved state ({observed['timeframe']}): "
        f"position={observed['position']}; phase={observed['market_phase']}; "
        f"observed MACD crossover={observed['macd_cross']}. "
        + " ".join(observed["evidence"])
    )
    text += "\n\nInterpretation: " + explanation["interpretation"]
    text += (
        f"\n\nForecast ({forecast['timeframe']}): {forecast['scenario']}"
        + "\nAssumptions: "
        + "; ".join(forecast["assumptions"])
        + "\nInvalidation: "
        + forecast["invalidation"]
    )
    if "risk_assessment" in explanation:
        risk = explanation["risk_assessment"]
        text += "\n\nRisk: " + risk["rationale"] + "\nDownside: " + risk["downside"] + "\nEvidence: " + ", ".join(risk["evidence"])
    if "execution" in explanation:
        execution = explanation["execution"]
        text += "\n\nExecution (" + execution["mode"] + "): " + execution["rationale"] + "\nEvidence: " + ", ".join(execution["evidence"])
    text += "\n\nConditional actions:"
    for plan in explanation["conditional_actions"]:
        text += (
            f"\n- {plan['action']} ({plan['timeframe']}): {plan['trigger']} "
            f"Required position: {plan['required_position']}. Prerequisite: {plan['prerequisite']} "
            f"Reason: {plan['reasoning']}"
        )
    return text
