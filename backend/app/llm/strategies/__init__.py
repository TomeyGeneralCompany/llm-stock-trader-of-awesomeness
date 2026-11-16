"""
LLM Strategies Module
Based on LLM trading strategy - Domain-separated architecture

This module implements the refactored LLM Trading strategy, Uses domain separation to split the original monolithic class into specialized modules: 

Core modules:
- LLMSmartStrategy: Main strategy class (Refactored version)
- LLMDecisionEngine: LLM Decision engine
- RiskManager: Risk manager
- PositionManager: Position manager
- StockCharacteristicsAnalyzer: Stock characteristics analyzer
- TradingEventDetector: Trading event detector
- PerformanceTracker: Performance tracker

Before and after refactoring:
- Before refactoring: Single class (~2900 lines, 30+method)
- After refactoring: 8 specialized modules (~500 lines/Module, Clear responsibilities)
"""

from .base import (
    ParameterSpec,
    ParameterType,
    SignalType,
    StrategyConfig,
    TradingSignal,
    TradingStrategy,
    get_available_strategies,
)

# New refactored modules (Temporarily commented out, because data_types has been removed)
# from .data_types import (
#     # Decision-related
#     DecisionContext,
#     LLMDecision,
#     PerformanceMetrics,
#     PnLInsights,
#     PositionMetrics,
#     StockCharacteristics,
#     # Strategy state
#     StrategyState,
#     TechnicalParameters,
#     # Core data types
#     TradingEvent,
#     TradingSignalRequest,
# )
# from .llm_decision_engine import LLMDecisionEngine
# from .llm_smart_strategy import LLMSmartStrategy
# Original strategy (Backward compatibility)
from .llm_strategy import LLMSmartStrategy as LLMStrategyLegacy

# from .performance_tracker import PerformanceTracker
# from .position_manager import PositionManager
# from .risk_manager import RiskManager
# from .stock_characteristics_analyzer import StockCharacteristicsAnalyzer
# from .trading_event_detector import TradingEventDetector

__all__ = [
    # Base Classes
    "ParameterSpec",
    "ParameterType",
    "SignalType",
    "StrategyConfig",
    "TradingSignal",
    "TradingStrategy",
    "get_available_strategies",
    # Original strategy (Backward compatibility)
    "LLMStrategyLegacy",
    # New main strategy class (Temporarily commented out, because the module was removed)
    # "LLMSmartStrategy",
    # Core modules (Temporarily commented out, because the module was removed)
    # "LLMDecisionEngine",
    # "RiskManager",
    # "PositionManager",
    # "StockCharacteristicsAnalyzer",
    # "TradingEventDetector",
    # "PerformanceTracker",
    # Data types (Temporarily commented out, because the module was removed)
    # "TradingEvent",
    # "StockCharacteristics",
    # "TechnicalParameters",
    # "PositionMetrics",
    # "PnLInsights",
    # "PerformanceMetrics",
    # "DecisionContext",
    # "LLMDecision",
    # "TradingSignalRequest",
    # "StrategyState",
    # Helper functions
    "print_architecture_info",
    "get_module_info",
]

# Version information
__version__ = "2.0.0"
__author__ = "LLM Agent Trader Team"
__description__ = "Refactored LLM Trading Strategy with Domain Separation Architecture"

# Architecture description
ARCHITECTURE_INFO = """
Domain-separated architecture (Domain Separation Architecture):

📊 LLMSmartStrategy (Main controller)
├── 🤖 LLMDecisionEngine (LLM Decision engine)
│   ├── prompt Build
│   ├── LLM Call
│   └── Response parsing
├── ⚡ RiskManager (Risk manager)  
│   ├── Risk assessment
│   ├── P&L insights
│   └── Decision validation
├── 💼 PositionManager (Position manager)
│   ├── Position tracking
│   ├── Trade execution
│   └── P&L calculation
├── 📈 StockCharacteristicsAnalyzer (Stock characteristics analyzer)
│   ├── Volatility analysis
│   ├── Trend consistency
│   └── MACD Effectiveness
├── 🔍 TradingEventDetector (Trading event detector)
│   ├── MACD Signal
│   ├── Moving-average crossover
│   ├── Bollinger Band breakouts
│   └── Price breakout
└── 📊 PerformanceTracker (Performance tracker)
    ├── Trade record
    ├── Performance calculation
    └── Report generation

📋 data_types (Shared data structures)
├── DTOs and data classes
├── Type definitions
└── Interface standards
"""


def print_architecture_info():
    """Print architecture information"""
    print(ARCHITECTURE_INFO)


def get_module_info():
    """Get module information"""
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "modules": len(__all__),
        "architecture": "Domain Separation Architecture",
    }
