"""
Backtesting engine for the Sentinel trading bot.

Provides historical data loading, event-driven trade simulation with full
RiskGuardian integration, performance metrics, and pre-built strategy
callbacks for testing without Grok API calls.

Usage::

    from backtesting import (
        BacktestMetrics,
        BacktestResult,
        BacktestSimulator,
        HistoricalDataLoader,
        simple_rsi_strategy,
    )

    loader = HistoricalDataLoader()
    data = loader.generate_synthetic("BTC", days=90, volatility=0.02)

    sim = BacktestSimulator(initial_capital=10_000)
    result = sim.run(data, simple_rsi_strategy)

    metrics = BacktestMetrics.calculate_all(result)
    BacktestMetrics.print_report(metrics)
"""

from backtesting.data_loader import HistoricalDataLoader
from backtesting.metrics import BacktestMetrics, BacktestResult
from backtesting.simulator import BacktestSimulator
from backtesting.strategies import (
    mean_reversion_strategy,
    momentum_strategy,
    simple_rsi_strategy,
)
from backtesting.walk_forward import WalkForwardBacktester, WalkForwardResult

__all__ = [
    "BacktestMetrics",
    "BacktestResult",
    "BacktestSimulator",
    "HistoricalDataLoader",
    "WalkForwardBacktester",
    "WalkForwardResult",
    "mean_reversion_strategy",
    "momentum_strategy",
    "simple_rsi_strategy",
]
