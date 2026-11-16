"""Regressions from real Qwen run 43931138082's contradictory evidence."""

import pytest
from app.ask_trader import sample_market
from app.llm.analysis.enhanced_technical_analyzer import EnhancedTechnicalAnalyzer
from app.utils.indicators import calculate_macd


def test_short_event_lookback_keeps_indicator_warmup():
    data = sample_market()
    context = EnhancedTechnicalAnalyzer().analyze_comprehensive_context(
        data, "2025-03-01", lookback_days=10
    )
    expected = round(data.close.tail(20).mean(), 2)
    assert context["moving_averages"]["ma_20"] == expected == 119.74
    assert context["moving_averages"]["ma_20"] != round(data.close.iloc[-1], 2)
    assert context["bollinger_analysis"]["bb_width"] > 0
    assert context["trend_analysis"]["adx_value"] > 0
    assert context["macd_analysis"]


@pytest.mark.parametrize("date_source", ["timestamp", "Date", "date-column"])
def test_requested_date_excludes_future_for_named_indexes_and_columns(date_source):
    data = sample_market()
    data.index.name = date_source
    if date_source == "date-column":
        data = data.rename_axis("Date").reset_index()
    date = "2025-02-10"
    context = EnhancedTechnicalAnalyzer().analyze_comprehensive_context(data, date, 10)
    historical = sample_market().loc[:date]
    assert context["moving_averages"]["ma_20"] == round(
        historical.close.tail(20).mean(), 2
    )
    assert context["date"] == date


def test_no_string_is_false_in_market_regime(monkeypatch):
    analyzer = EnhancedTechnicalAnalyzer()
    monkeypatch.setattr(
        analyzer, "_analyze_trend_strength", lambda *a: {"strong_trend": "no"}
    )
    monkeypatch.setattr(
        analyzer, "_analyze_volatility", lambda *a: {"is_high_volatility": "no"}
    )
    context = analyzer._classify_market_regime(None, None)
    assert context["market_regime"] == "sideways_stable"
    assert context["is_trending"] is False and context["is_volatile"] is False


def test_adaptive_macd_is_not_replaced_by_different_periods():
    data = calculate_macd(sample_market(), short_period=6, long_period=18)
    context = EnhancedTechnicalAnalyzer().analyze_comprehensive_context(
        data, "2025-03-01", 10
    )
    assert context["macd_analysis"]["macd_line"] == round(data.macd.iloc[-1], 4)
    assert context["macd_analysis"]["signal_line"] == round(
        data.macd_signal.iloc[-1], 4
    )


def test_missing_date_reports_unavailable_context():
    context = EnhancedTechnicalAnalyzer().analyze_comprehensive_context(
        sample_market(), "2030-01-01"
    )
    assert "error" in context


def test_volatility_percentile_compares_matching_units():
    data = sample_market()
    returns = data.close.pct_change().dropna()
    history = data.close.pct_change().rolling(20).std().dropna()
    expected = round((returns.std() > history).mean() * 100, 1)
    context = EnhancedTechnicalAnalyzer().analyze_comprehensive_context(
        data, "2025-03-01", 10
    )
    assert context["volatility_analysis"]["volatility_percentile"] == expected
