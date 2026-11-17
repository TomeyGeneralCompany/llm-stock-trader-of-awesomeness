"""
LLM modules package initialization
"""

from . import analysis
from .analysis import EnhancedTrendAnalyzer, EnhancedTrendResult, TrendAnalysisResult
from .client import LLMClientConfig, get_llm_client
from .strategies import (
    ParameterSpec,
    ParameterType,
    SignalType,
    StrategyConfig,
    TradingSignal,
    TradingStrategy,
    get_available_strategies,
)

__all__ = [
    # Client utilities
    "get_llm_client",
    "LLMClientConfig",
    # Analysis Tools
    "EnhancedTrendAnalyzer",
    "EnhancedTrendResult",
    "TrendAnalysisResult",
    # Strategy Framework
    "ParameterSpec",
    "ParameterType",
    "SignalType",
    "StrategyConfig",
    "TradingSignal",
    "TradingStrategy",
    "get_available_strategies",
    # Sub-modules
    "analysis",
]
