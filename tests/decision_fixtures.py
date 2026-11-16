"""Structured explanation fixtures based on each request's controller snapshot."""

import copy
import json


def explained(decision, prompt):
    result = copy.deepcopy(decision)
    context = json.loads(
        prompt.split("Controller working state (authoritative):\n", 1)[1].split(
            "\n\nOwner question:", 1
        )[0]
    )
    result["explanation"] = {
        "observed_state": {
            "timeframe": context["observation_timeframe"],
            "position": "long" if context["shares"] > 0 else "flat",
            "market_phase": context["trend"].get("market_phase")
            or context["trend"].get("dominant_trend")
            or "unknown",
            "macd_cross": context["macd_cross"],
            "macd_alignment": context.get("macd_alignment", "unknown"),
            "evidence": ["Current close: " + str(context["current"]["close"])],
        },
        "interpretation": "The current evidence supports the selected action.",
        "risk_assessment": {
            "evidence": ["current.close"],
            "rationale": "The observed close informs price exposure, not certainty.",
            "downside": "Price could move against the position.",
        },
        "execution": {
            "mode": "wait" if result["action"] == "HOLD" else "now",
            "evidence": ["current.close"],
            "rationale": "The observed close supports the selected action.",
        },
        "forecast": {
            "scenario": "An upward breakout is possible, not yet observed.",
            "timeframe": "Next daily bars",
            "assumptions": ["Demand remains supportive."],
            "invalidation": "A confirmed downward reversal invalidates the scenario.",
        },
        "conditional_actions": [
            {
                "action": "BUY",
                "trigger": "Confirmed upward breakout with volume",
                "timeframe": "Next daily bars",
                "required_position": "flat",
                "prerequisite": "Remain flat until the trigger occurs.",
                "reasoning": "Confirmation would support entry.",
            },
            {
                "action": "SELL",
                "trigger": "Confirmed reversal after entry",
                "timeframe": "After a future entry",
                "required_position": "long",
                "prerequisite": "Execute the BUY first and hold shares.",
                "reasoning": "The reversal would invalidate the entry rationale.",
            },
        ],
    }
    return result
