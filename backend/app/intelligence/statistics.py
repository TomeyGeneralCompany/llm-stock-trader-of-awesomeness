"""Deterministic event studies and forecast scoring; the LLM cannot invent metrics."""
import math
from datetime import datetime
import numpy as np
import pandas as pd


def event_study(asset, benchmark, event_time, pre=20, window=12):
    """Estimate alpha/beta strictly before the event; study the subsequent window."""
    if pre < 3 or window < 1:
        raise ValueError("Event study needs a pre-event sample and positive window.")
    if not asset or not benchmark:
        return {"status": "insufficient_data", "sample_size": 0}
    a = pd.DataFrame(asset).set_index("end").sort_index()
    m = pd.DataFrame(benchmark).set_index("end").sort_index()
    if a.empty or m.empty:
        return {"status": "insufficient_data", "sample_size": 0}
    joined = pd.concat([a.close.pct_change().rename("asset"), m.close.pct_change().rename("market")], axis=1).dropna()
    before = joined.loc[joined.index < event_time].tail(pre)
    after = joined.loc[joined.index >= event_time].head(window)
    if len(before) < pre or len(after) < window or float(before.market.var(ddof=1)) <= 0:
        return {"status": "insufficient_data", "sample_size": len(before), "event_observations": len(after)}
    beta = float(before.asset.cov(before.market) / before.market.var(ddof=1))
    alpha = float(before.asset.mean() - beta * before.market.mean())
    abnormal = after.asset - (alpha + beta * after.market)
    base_rows = a.loc[a.index < event_time]
    if base_rows.empty:
        return {"status": "insufficient_data", "sample_size": len(before)}
    base = float(base_rows.close.iloc[-1])
    observed = a.loc[a.index.isin(after.index)]
    volume_base = float(base_rows.volume.tail(pre).mean())
    responses = observed.close / base - 1
    volume_ratios = observed.volume / volume_base if volume_base > 0 else pd.Series(dtype=float)
    first = datetime.fromisoformat(str(after.index[0]))
    peak = datetime.fromisoformat(str(responses.abs().idxmax()))
    last = datetime.fromisoformat(str(after.index[-1]))
    origin = datetime.fromisoformat(str(event_time))
    movement = responses.abs()
    threshold = max(.001, float((before.asset-alpha-beta*before.market).std(ddof=1)))
    onset = responses.index[movement >= threshold]
    market_trend = "rising" if before.market.mean() > 0 else "falling" if before.market.mean() < 0 else "flat"
    result = {
        "status": "complete", "sample_size": len(before), "event_observations": len(after),
        "alpha": alpha, "beta": beta, "abnormal_returns": abnormal.tolist(),
        "cumulative_abnormal_return": float(abnormal.sum()),
        "fractional_return": float(responses.iloc[-1]),
        "favorable_excursion": float((observed.high / base - 1).max()) if "high" in observed else float(responses.max()),
        "unfavorable_excursion": float((observed.low / base - 1).min()) if "low" in observed else float(responses.min()),
        "peak_time": responses.abs().idxmax(),
        "onset_minutes": (datetime.fromisoformat(str(onset[0]))-origin).total_seconds()/60 if len(onset) else None,
        "peak_minutes": (peak-origin).total_seconds()/60,
        "observation_window_minutes": (last-first).total_seconds()/60,
        "response_duration_bars": int((movement >= threshold).sum()),
        "conditions": {"market_trend": market_trend, "pre_event_market_volatility": float(before.market.std(ddof=1)),
                       "session": observed.session.iloc[0] if "session" in observed else "unknown"},
        "abnormal_volume_peak": float(volume_ratios.max()) if len(volume_ratios) else None,
        "abnormal_volume_duration_bars": int((volume_ratios > 1).sum()) if len(volume_ratios) else None,
        "reversal": bool(responses.iloc[-1] * responses.loc[responses.abs().idxmax()] < 0),
        "uncertainty": {"method": "pre-event residual standard deviation",
                        "residual_std": float((before.asset-alpha-beta*before.market).std(ddof=1))},
        "causality": "An association, not proof of causation.",
    }
    return result


def aggregate(studies):
    usable = [s for s in studies if s.get("status") == "complete"]
    values = np.asarray([s["fractional_return"] for s in usable], dtype=float)
    if not len(values):
        return {"sample_size": 0, "status": "insufficient_data", "mean": None, "median": None,
                "positive_frequency": None, "negative_frequency": None, "uncertainty": None}
    error = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else None
    result = {"sample_size": len(values), "status": "complete", "mean": float(values.mean()),
            "median": float(np.median(values)), "positive_frequency": float((values > 0).mean()),
            "negative_frequency": float((values < 0).mean()),
            "mean_favorable_excursion": float(np.mean([s["favorable_excursion"] for s in usable])),
            "mean_unfavorable_excursion": float(np.mean([s["unfavorable_excursion"] for s in usable])),
            "reversal_frequency": float(np.mean([s["reversal"] for s in usable])),
            "uncertainty": {"method": "normal approximation, not causal certainty", "standard_error": error,
                            "mean_interval_95": [float(values.mean()-1.96*error), float(values.mean()+1.96*error)] if error is not None else None}}
    for field in ("peak_minutes", "onset_minutes", "response_duration_bars", "abnormal_volume_peak",
                  "abnormal_volume_duration_bars", "cumulative_abnormal_return"):
        numbers = [float(study[field]) for study in usable if study.get(field) is not None]
        result["mean_"+field] = float(np.mean(numbers)) if numbers else None
    groups = {}
    for study in usable:
        label = study.get("conditions", {}).get("market_trend", "unknown")
        groups.setdefault(label, []).append(study["fractional_return"])
    result["condition_comparison"] = [
        {"market_trend": label, "sample_size": len(samples), "mean": float(np.mean(samples)),
         "median": float(np.median(samples)), "positive_frequency": float(np.mean(np.array(samples)>0))}
        for label, samples in groups.items()]
    return result


def score_forecast(prediction, actual):
    expected = prediction["expected_return"]
    direction = prediction["direction"]
    correct = (actual > 0) - (actual < 0) == direction
    up_probability = prediction["confidence"] if direction > 0 else 1-prediction["confidence"] if direction < 0 else .5
    return {"actual_return": actual, "predicted_return": expected, "absolute_error": abs(actual-expected),
            "direction_correct": bool(correct), "brier": (up_probability-int(actual>0))**2,
            "error_classes": ([] if correct else ["directional"]) + (["magnitude"] if abs(actual-expected) > abs(expected) else []),
            "causal_interpretation": "Requires separate evidence-backed review.",
            "timing": "Evaluated on the first available observation at or after the forecast horizon.",
            "execution": "Forecast outcome is separate from any simulated trade outcome."}
