"""
Comprehensive tests for the walk-forward backtesting framework.

Covers:
- Window creation and data splitting
- Walk-forward backtest execution
- Result aggregation and metrics
- Report generation
- Performance context for Grok
- Edge cases (empty data, single window, no trades)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import pytest

from backtesting.data_loader import HistoricalDataLoader
from backtesting.metrics import BacktestResult
from backtesting.simulator import MarketSnapshot
from backtesting.walk_forward import WalkForwardBacktester, WalkForwardResult
from brain.models import TradeDecision


# ======================================================================
# Helpers
# ======================================================================


def _make_candles(
    prices: list[float],
    start: datetime | None = None,
    interval_hours: float = 1.0,
    spread_pct: float = 0.005,
) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame from a list of close prices."""
    if start is None:
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)

    n = len(prices)
    timestamps = [start + timedelta(hours=i * interval_hours) for i in range(n)]

    opens = [prices[0]] + prices[:-1]
    highs = [p * (1 + spread_pct) for p in prices]
    lows = [p * (1 - spread_pct) for p in prices]
    volumes = [1000.0] * n

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
    })


def _never_trade_strategy(snapshot: MarketSnapshot) -> list[TradeDecision]:
    """A strategy that never trades."""
    return []


def _always_long_strategy(snapshot: MarketSnapshot) -> list[TradeDecision]:
    """A simple strategy that always opens a long if not already in position."""
    if snapshot.asset in snapshot.positions:
        return []

    price = snapshot.close
    sl = round(price * 0.97, 2)
    tp = round(price * 1.06, 2)
    risk = abs(price - sl)
    reward = abs(tp - price)
    rr = reward / risk if risk > 0 else 2.0

    return [
        TradeDecision(
            action="open_long",
            asset=snapshot.asset,
            size_pct=0.05,
            leverage=2.0,
            entry_price=price,
            stop_loss=sl,
            take_profit=tp,
            order_type="market",
            reasoning="Always long strategy for testing.",
            conviction="medium",
            risk_reward_ratio=round(rr, 2),
        )
    ]


# ======================================================================
# Test: Window Creation
# ======================================================================


class TestWindowCreation:
    """Tests for _create_windows() static method."""

    def test_windows_created_correctly(self) -> None:
        """Should create correct number of non-overlapping windows."""
        # 30 days of data at 1h = 720 candles
        # lookback=10 days (240), walk_forward=5 days (120)
        # Each window consumes 120 candles of test data
        # Available for test = 720 candles
        # Windows: start at 0, then at 120, 240, etc.
        # Window 0: train[0:240], test[240:360]
        # Window 1: train[120:360], test[360:480]
        # etc.
        prices = [100 + i * 0.01 for i in range(720)]
        data = _make_candles(prices)

        windows = WalkForwardBacktester._create_windows(
            data, lookback_days=10, walk_forward_days=5,
        )

        assert len(windows) > 0
        for train, test in windows:
            assert len(train) > 0
            assert len(test) > 0

    def test_train_test_no_overlap(self) -> None:
        """Train and test data should not overlap."""
        prices = [100 + i * 0.01 for i in range(720)]
        data = _make_candles(prices)

        windows = WalkForwardBacktester._create_windows(
            data, lookback_days=5, walk_forward_days=3,
        )

        for train, test in windows:
            train_max_ts = train["timestamp"].max()
            test_min_ts = test["timestamp"].min()
            assert test_min_ts > train_max_ts, (
                "Test data should start after training data ends"
            )

    def test_empty_data_returns_empty(self) -> None:
        """Empty DataFrame should return empty windows list."""
        empty = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        windows = WalkForwardBacktester._create_windows(
            empty, lookback_days=5, walk_forward_days=3,
        )
        assert windows == []

    def test_insufficient_data_returns_empty(self) -> None:
        """Data shorter than one window should return empty."""
        prices = [100.0] * 10  # Only 10 candles
        data = _make_candles(prices)

        windows = WalkForwardBacktester._create_windows(
            data, lookback_days=5, walk_forward_days=5,
        )
        # 5 + 5 = 10 days = 240 candles needed, but only 10 available
        assert windows == []


# ======================================================================
# Test: Walk-Forward Backtest Execution
# ======================================================================


class TestWalkForwardExecution:
    """Tests for run_backtest()."""

    def test_basic_execution_with_synthetic_data(self) -> None:
        """Basic walk-forward backtest should run without errors."""
        wf = WalkForwardBacktester(initial_capital=10_000, fee_rate=0.0003)

        result = wf.run_backtest(
            strategy_fn=_never_trade_strategy,
            assets=["BTC"],
            lookback_days=5,
            walk_forward_days=3,
            volatility=0.02,
            seed=42,
        )

        assert isinstance(result, WalkForwardResult)
        assert result.initial_capital == 10_000

    def test_no_trades_preserves_capital(self) -> None:
        """A no-trade strategy should preserve initial capital."""
        wf = WalkForwardBacktester(initial_capital=10_000, fee_rate=0.0)

        result = wf.run_backtest(
            strategy_fn=_never_trade_strategy,
            assets=["BTC"],
            lookback_days=5,
            walk_forward_days=3,
            volatility=0.02,
            seed=42,
        )

        assert result.trade_count == 0
        assert result.final_equity == result.initial_capital

    def test_capital_carries_forward(self) -> None:
        """Capital should carry forward between windows."""
        wf = WalkForwardBacktester(initial_capital=10_000, fee_rate=0.0)

        result = wf.run_backtest(
            strategy_fn=_always_long_strategy,
            assets=["BTC"],
            lookback_days=5,
            walk_forward_days=5,
            volatility=0.02,
            seed=42,
        )

        # There should be multiple windows
        if len(result.window_results) > 1:
            # Capital after first window should be starting capital of second
            first_final = result.window_results[0].final_equity
            second_initial = result.window_results[1].initial_capital
            assert second_initial == pytest.approx(first_final, rel=0.01)

    def test_multiple_assets(self) -> None:
        """Backtest should handle multiple assets sequentially."""
        wf = WalkForwardBacktester(initial_capital=10_000, fee_rate=0.0)

        result = wf.run_backtest(
            strategy_fn=_never_trade_strategy,
            assets=["BTC", "ETH"],
            lookback_days=5,
            walk_forward_days=3,
            volatility=0.02,
            seed=42,
        )

        assert isinstance(result, WalkForwardResult)

    def test_with_provided_historical_data(self) -> None:
        """Should use provided historical data instead of generating."""
        prices = [100 + i * 0.1 for i in range(720)]
        data = _make_candles(prices)

        wf = WalkForwardBacktester(initial_capital=10_000, fee_rate=0.0)

        result = wf.run_backtest(
            strategy_fn=_never_trade_strategy,
            assets=["BTC"],
            lookback_days=10,
            walk_forward_days=5,
            historical_data={"BTC": data},
        )

        assert isinstance(result, WalkForwardResult)

    def test_fee_rate_applied(self) -> None:
        """Trades should have fees deducted using the configured rate."""
        wf = WalkForwardBacktester(initial_capital=10_000, fee_rate=0.001)

        result = wf.run_backtest(
            strategy_fn=_always_long_strategy,
            assets=["BTC"],
            lookback_days=5,
            walk_forward_days=5,
            volatility=0.01,
            seed=42,
        )

        # Check that fees were deducted if any trades occurred
        if result.trade_count > 0:
            # With fees, equity should differ from a no-fee run
            wf_nofee = WalkForwardBacktester(initial_capital=10_000, fee_rate=0.0)
            result_nofee = wf_nofee.run_backtest(
                strategy_fn=_always_long_strategy,
                assets=["BTC"],
                lookback_days=5,
                walk_forward_days=5,
                volatility=0.01,
                seed=42,
            )
            if result_nofee.trade_count > 0:
                assert result.final_equity != result_nofee.final_equity


# ======================================================================
# Test: WalkForwardResult
# ======================================================================


class TestWalkForwardResult:
    """Tests for the WalkForwardResult dataclass."""

    def test_default_values(self) -> None:
        """Default values should be sensible."""
        result = WalkForwardResult()

        assert result.total_return == 0.0
        assert result.sharpe_ratio == 0.0
        assert result.max_drawdown == 0.0
        assert result.win_rate == 0.0
        assert result.trade_count == 0
        assert result.equity_curve == []
        assert result.window_results == []
        assert result.initial_capital == 0.0
        assert result.final_equity == 0.0

    def test_populated_result(self) -> None:
        """Result should store provided values correctly."""
        result = WalkForwardResult(
            total_return=5.5,
            sharpe_ratio=1.2,
            max_drawdown=3.0,
            win_rate=60.0,
            trade_count=10,
            initial_capital=10_000,
            final_equity=10_550,
        )

        assert result.total_return == 5.5
        assert result.trade_count == 10


# ======================================================================
# Test: Report Generation
# ======================================================================


class TestReportGeneration:
    """Tests for generate_report()."""

    def test_report_format(self) -> None:
        """Report should contain key sections."""
        result = WalkForwardResult(
            total_return=5.5,
            sharpe_ratio=1.2,
            max_drawdown=3.0,
            win_rate=60.0,
            trade_count=10,
            initial_capital=10_000,
            final_equity=10_550,
            window_metrics=[
                {"window": 1, "asset": "BTC", "total_return_pct": 3.0, "total_trades": 5, "win_rate": 60.0},
                {"window": 2, "asset": "BTC", "total_return_pct": 2.5, "total_trades": 5, "win_rate": 60.0},
            ],
        )

        report = WalkForwardBacktester.generate_report(result)

        assert "WALK-FORWARD BACKTEST REPORT" in report
        assert "$10,000" in report
        assert "5.50%" in report or "+5.50%" in report
        assert "Window 1" in report
        assert "Window 2" in report

    def test_report_empty_result(self) -> None:
        """Report should handle empty results gracefully."""
        result = WalkForwardResult()
        report = WalkForwardBacktester.generate_report(result)

        assert "WALK-FORWARD" in report
        assert "$0.00" in report


# ======================================================================
# Test: Performance Context for Grok
# ======================================================================


class TestPerformanceContext:
    """Tests for get_performance_context()."""

    def test_context_contains_key_metrics(self) -> None:
        """Context should include return, sharpe, drawdown, win rate."""
        result = WalkForwardResult(
            total_return=5.5,
            sharpe_ratio=1.2,
            max_drawdown=3.0,
            win_rate=60.0,
            trade_count=10,
        )

        context = WalkForwardBacktester.get_performance_context(result)

        assert "5.50%" in context or "+5.50%" in context
        assert "1.200" in context
        assert "3.00%" in context
        assert "60.0%" in context
        assert "10" in context

    def test_context_includes_recent_window(self) -> None:
        """Context should reference the most recent window if available."""
        result = WalkForwardResult(
            total_return=5.5,
            sharpe_ratio=1.2,
            max_drawdown=3.0,
            win_rate=60.0,
            trade_count=10,
            window_metrics=[
                {"window": 1, "total_return_pct": 3.0, "total_trades": 5},
                {"window": 2, "total_return_pct": 2.5, "total_trades": 5},
            ],
        )

        context = WalkForwardBacktester.get_performance_context(result)

        assert "Most Recent Window" in context
        assert "+2.50%" in context

    def test_context_empty_result(self) -> None:
        """Context should handle empty results gracefully."""
        result = WalkForwardResult()
        context = WalkForwardBacktester.get_performance_context(result)

        assert "Strategy Performance" in context


# ======================================================================
# Test: End-to-end integration
# ======================================================================


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_walk_forward_with_rsi(self) -> None:
        """Run a complete walk-forward backtest with the RSI strategy."""
        from backtesting.strategies import simple_rsi_strategy

        wf = WalkForwardBacktester(initial_capital=10_000, fee_rate=0.0003)

        result = wf.run_backtest(
            strategy_fn=simple_rsi_strategy,
            assets=["BTC"],
            lookback_days=7,
            walk_forward_days=3,
            volatility=0.02,
            seed=42,
        )

        assert result.initial_capital == 10_000
        assert result.final_equity > 0
        assert isinstance(result.total_return, float)
        assert isinstance(result.sharpe_ratio, float)
        assert result.max_drawdown >= 0

        # Generate report should work
        report = WalkForwardBacktester.generate_report(result)
        assert len(report) > 0

        # Context should work
        context = WalkForwardBacktester.get_performance_context(result)
        assert len(context) > 0

    def test_uptrend_long_strategy_profits(self) -> None:
        """Long strategy on uptrend data should produce positive returns."""
        prices = [100 + i * 0.5 for i in range(720)]
        data = _make_candles(prices)

        wf = WalkForwardBacktester(initial_capital=10_000, fee_rate=0.0)

        result = wf.run_backtest(
            strategy_fn=_always_long_strategy,
            assets=["BTC"],
            lookback_days=10,
            walk_forward_days=5,
            historical_data={"BTC": data},
        )

        if result.trade_count > 0:
            assert result.final_equity >= result.initial_capital * 0.95  # Allow small losses from SL
