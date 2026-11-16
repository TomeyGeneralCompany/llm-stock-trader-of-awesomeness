"""
Trend Analysis Benchmark Database
Used for validating and improving the accuracy of trend identification algorithms
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Tuple

import pandas as pd


@dataclass
class TrendPhase:
    """Trend phase definition"""

    start_date: str
    end_date: str
    trend_type: str  # 'uptrend', 'downtrend', 'sideways', 'sideways_bullish', 'sideways_bearish'
    strength: str  # 'strong', 'moderate', 'weak'
    confidence: float  # 0.0 - 1.0 Manually labeled confidence
    description: str  # Descriptive explanation
    key_events: List[str] = None  # Key events or turning points


@dataclass
class BenchmarkCase:
    """Benchmark test case"""

    symbol: str
    period: str
    total_period: Tuple[str, str]  # (start_date, end_date)
    phases: List[TrendPhase]
    overall_trend: str
    complexity_level: str  # 'simple', 'moderate', 'complex'
    notes: str


class TrendBenchmarkDatabase:
    """Trend analysis benchmark database"""

    def __init__(self):
        self.benchmark_cases = self._initialize_benchmark_cases()

    def _initialize_benchmark_cases(self) -> Dict[str, BenchmarkCase]:
        """Initialize benchmark test cases"""
        cases = {}

        # 2330.TW Case - Complex multiphase trend
        tsmc_2024_2025 = BenchmarkCase(
            symbol="2330.TW",
            period="Past year (2024 year 10 month-2025 year 7 month)",
            total_period=("2024-10-01", "2025-07-30"),
            phases=[
                TrendPhase(
                    start_date="2024-10-20",
                    end_date="2024-11-14",
                    trend_type="sideways",
                    strength="moderate",
                    confidence=0.8,
                    description="Range-bound movement, Price fluctuates within a specific range, No clear direction",
                    key_events=["Sideways consolidation", "Awaiting breakout direction"],
                ),
                TrendPhase(
                    start_date="2025-02-20",
                    end_date="2025-04-20",
                    trend_type="downtrend",
                    strength="moderate",
                    confidence=0.85,
                    description="Clear downtrend, Progressively lower lows",
                    key_events=["Break below support", "Continued decline"],
                ),
                TrendPhase(
                    start_date="2025-04-23",
                    end_date="2025-07-23",
                    trend_type="uptrend",
                    strength="moderate",
                    confidence=0.9,
                    description="Uptrend, Progressively higher highs and lows",
                    key_events=["Break above resistance", "Continued rise"],
                ),
            ],
            overall_trend="complex_multi_phase",
            complexity_level="complex",
            notes="Typical multiphase trend transition example, Includes sideways movement->Decline->and a complete upward phase",
        )
        cases["TSMC_2024_2025"] = tsmc_2024_2025

        # TSLA Case - Downtrend + Two range-bound phases
        tsla_2025 = BenchmarkCase(
            symbol="TSLA",
            period="Past six months (2025 year 1 month-7 month)",
            total_period=("2025-01-22", "2025-07-28"),
            phases=[
                TrendPhase(
                    start_date="2025-01-22",
                    end_date="2025-03-11",
                    trend_type="downtrend",
                    strength="strong",
                    confidence=0.9,
                    description="Clear downward trend phase",
                    key_events=["Market correction", "Take profit"],
                ),
                TrendPhase(
                    start_date="2025-03-11",
                    end_date="2025-05-05",
                    trend_type="sideways",
                    strength="moderate",
                    confidence=0.85,
                    description="First range-bound phase, Price fluctuates within a range",
                    key_events=["Market stabilization", "Sideways consolidation"],
                ),
                TrendPhase(
                    start_date="2025-05-28",
                    end_date="2025-07-28",
                    trend_type="sideways",
                    strength="moderate",
                    confidence=0.8,
                    description="Second range-bound phase, Continued sideways consolidation",
                    key_events=["Continued consolidation", "Waiting for breakout"],
                ),
            ],
            overall_trend="complex_multi_phase",
            complexity_level="complex",
            notes="Complex case with a downtrend and two range-bound phases, Test range identification",
        )
        cases["TSLA_2025"] = tsla_2025

        # AAPL Case - Two-phase pattern from downtrend to sideways movement
        aapl_2025 = BenchmarkCase(
            symbol="AAPL",
            period="2025 year 2 month-7 month",
            total_period=("2025-02-01", "2025-07-28"),
            phases=[
                TrendPhase(
                    start_date="2025-02-25",
                    end_date="2025-04-08",
                    trend_type="downtrend",
                    strength="strong",
                    confidence=0.9,
                    description="Clearly defined downtrend phase, Price falls sharply from its peak",
                    key_events=["Market pullback", "Technical correction", "from $246.72 Falls to $172.19"],
                ),
                TrendPhase(
                    start_date="2025-04-09",
                    end_date="2025-07-28",
                    trend_type="sideways",
                    strength="moderate",
                    confidence=0.85,
                    description="Sideways consolidation phase, Price moves back and forth within a range",
                    key_events=["Sideways consolidation", "12.7% Range-bound movement", "Repeated support and resistance tests"],
                ),
            ],
            overall_trend="trend_to_sideways_transition",
            complexity_level="medium",
            notes="Typical transition from trend to range-bound movement, Tests the algorithm’s detection of trend transitions and range-bound movement",
        )
        cases["AAPL_2025"] = aapl_2025

        return cases

    def get_benchmark_case(self, case_id: str) -> BenchmarkCase:
        """Get a specific benchmark case"""
        return self.benchmark_cases.get(case_id)

    def add_benchmark_case(self, case_id: str, benchmark_case: BenchmarkCase):
        """Add new benchmark case"""
        self.benchmark_cases[case_id] = benchmark_case

    def get_all_cases(self) -> Dict[str, BenchmarkCase]:
        """Get all benchmark cases"""
        return self.benchmark_cases

    def evaluate_algorithm_performance(
        self, case_id: str, algorithm_result: Dict
    ) -> Dict[str, Any]:
        """Evaluate algorithm performance on a specific case"""
        benchmark = self.get_benchmark_case(case_id)
        if not benchmark:
            return {"error": f"Benchmark case not found: {case_id}"}

        evaluation = {
            "case_id": case_id,
            "benchmark_phases": len(benchmark.phases),
            "detected_phases": algorithm_result.get("detected_phases", 0),
            "overall_accuracy": 0.0,
            "phase_accuracy": [],
            "missed_transitions": [],
            "false_positives": [],
        }

        # Specific evaluation logic can be implemented here
        # Compare detected trend phases with manually labeled benchmarks

        return evaluation


# Global instance
trend_benchmark_db = TrendBenchmarkDatabase()


def get_benchmark_database() -> TrendBenchmarkDatabase:
    """Get benchmark database instance"""
    return trend_benchmark_db


def create_test_case_from_data(
    symbol: str, market_data: List[Dict], manual_phases: List[Dict]
) -> BenchmarkCase:
    """Create test case from market data and manual labels"""
    phases = []
    for phase_data in manual_phases:
        phase = TrendPhase(
            start_date=phase_data["start_date"],
            end_date=phase_data["end_date"],
            trend_type=phase_data["trend_type"],
            strength=phase_data.get("strength", "moderate"),
            confidence=phase_data.get("confidence", 0.8),
            description=phase_data.get("description", ""),
            key_events=phase_data.get("key_events", []),
        )
        phases.append(phase)

    # Determine overall complexity
    complexity = "simple"
    if len(phases) > 2:
        complexity = "moderate"
    if len(phases) > 3 or any(p.trend_type.startswith("sideways") for p in phases):
        complexity = "complex"

    return BenchmarkCase(
        symbol=symbol,
        period=f"{phases[0].start_date} to {phases[-1].end_date}",
        total_period=(phases[0].start_date, phases[-1].end_date),
        phases=phases,
        overall_trend="multi_phase" if len(phases) > 1 else phases[0].trend_type,
        complexity_level=complexity,
        notes=f"Automatically generated test case, Containing {len(phases)} trend phases",
    )
