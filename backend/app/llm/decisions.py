"""Shared semantic validation for inference and sealed analysis artifacts."""

import math


class DecisionError(ValueError):
    pass


def validate_decision(decision):
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
    return decision
