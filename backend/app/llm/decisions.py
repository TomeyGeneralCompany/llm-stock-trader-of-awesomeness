"""Shared semantic validation for inference and sealed analysis artifacts."""

import math

EXPLANATION_INSTRUCTIONS = "- explanation: an object with four required sections below. Keep all reasoning concise, concrete and in English.\n  - observed_state: object with timeframe, position (flat/long), market_phase and macd_cross copied accurately from controller facts, plus evidence (list of specific observations). A MACD line above signal is not a newly observed crossover.\n  - interpretation: explain how observations support the current action, including conflicting evidence. If a broad uptrend contains shorter consolidation, name both timeframes and the evidence for each. Label inferences as interpretations.\n  - forecast: object with scenario, timeframe, assumptions (nonempty list), and invalidation. Label anticipated transitions/crossovers as future possibilities, never observed facts.\n  - conditional_actions: nonempty list of objects with action (BUY/SELL/HOLD), trigger, timeframe, required_position (flat/long/any), prerequisite, and reasoning. Explain what evidence changes the decision and the sequence of positions needed. A future SELL is allowed while currently flat when it explicitly requires a prior entry and a long position; it is not a current executable SELL. HOLD while flat means waiting."


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
    text += "\n\nConditional actions:"
    for plan in explanation["conditional_actions"]:
        text += (
            f"\n- {plan['action']} ({plan['timeframe']}): {plan['trigger']} "
            f"Required position: {plan['required_position']}. Prerequisite: {plan['prerequisite']} "
            f"Reason: {plan['reasoning']}"
        )
    return text
