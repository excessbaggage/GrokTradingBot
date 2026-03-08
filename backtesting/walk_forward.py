"""
Walk-forward backtesting framework.

Extends the existing ``BacktestSimulator`` with a walk-forward
optimisation methodology: the historical period is split into
sequential train/test windows, the strategy is evaluated on each
test window in order, and results are aggregated into a single
``WalkForwardResult``.

This avoids overfitting to a single historical period and provides
a more realistic estimate of out-of-sample performance.

Usage::

    from backtesting.walk_forward import WalkForwardBacktester, WalkForwardResult

    wf = WalkForwardBacktester(initial_capital=10_000)
    result = wf.run_backtest(
        strategy_fn=simple_rsi_strategy,
        assets=["BTC"],
        lookback_days=30,
        walk_forward_days=7,
    )
    print(wf.generate_report(result))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd
from loguru import logger

from backtesting.data_loader import HistoricalDataLoader
from backtesting.metrics import BacktestMetrics, BacktestResult
from backtesting.simulator import BacktestSimulator, MarketSnapshot
from brain.models import TradeDecision
from config.trading_config import STARTING_CAPITAL


# ======================================================================
# Result dataclass
# ======================================================================


@dataclass
class WalkForwardResult:
    """Aggregated results from a walk-forward backtest.

    Attributes:
        total_return: Overall percentage return across all windows.
        sharpe_ratio: Annualised Sharpe ratio from the combined equity curve.
        max_drawdown: Maximum drawdown percentage across the full period.
        win_rate: Percentage of profitable trades.
        trade_count: Total number of trades executed.
        equity_curve: Combined equity curve across all walk-forward windows.
        window_results: Per-window ``BacktestResult`` list for drill-down.
        window_metrics: Per-window metrics dicts.
        initial_capital: Starting capital.
        final_equity: Ending equity after all windows.
    """

    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    trade_count: int = 0
    equity_curve: list[dict[str, Any]] = field(default_factory=list)
    window_results: list[BacktestResult] = field(default_factory=list)
    window_metrics: list[dict[str, Any]] = field(default_factory=list)
    initial_capital: float = 0.0
    final_equity: float = 0.0


# ======================================================================
# Walk-forward backtester
# ======================================================================

# Type alias for the strategy callback.
# Matches the ``DecisionCallback`` from ``simulator.py``.
StrategyCallback = Callable[[MarketSnapshot], list[TradeDecision]]


class WalkForwardBacktester:
    """Walk-forward backtesting engine.

    Divides historical data into sequential windows and runs the
    existing ``BacktestSimulator`` on each window's test portion.
    Capital carries forward between windows so the equity curve is
    continuous.

    Args:
        initial_capital: Starting portfolio value in USD.
            Defaults to ``STARTING_CAPITAL`` from trading config.
        fee_rate: Per-trade fee rate (default 0.03% = 0.0003 taker
            fee on Hyperliquid).
    """

    def __init__(
        self,
        initial_capital: float | None = None,
        fee_rate: float = 0.0003,
    ) -> None:
        self.initial_capital = initial_capital or STARTING_CAPITAL
        self.fee_rate = fee_rate

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_backtest(
        self,
        strategy_fn: StrategyCallback,
        assets: list[str],
        lookback_days: int = 30,
        walk_forward_days: int = 7,
        historical_data: dict[str, pd.DataFrame] | None = None,
        volatility: float = 0.02,
        seed: int | None = None,
    ) -> WalkForwardResult:
        """Execute a walk-forward backtest.

        For each asset, the total historical period is divided into
        sequential test windows of ``walk_forward_days``.  Each window
        is preceded by a ``lookback_days`` in-sample period that the
        strategy can use for indicator warm-up (handled internally by
        the simulator's lookback window).

        Capital carries forward between windows, so drawdown and
        compounding are realistic.

        Args:
            strategy_fn: Strategy callback with signature
                ``(MarketSnapshot) -> list[TradeDecision]``.
            assets: List of asset symbols to backtest (run sequentially).
            lookback_days: Number of days of history before each test
                window for indicator warm-up.
            walk_forward_days: Length of each test window in days.
            historical_data: Optional pre-loaded data per asset.  If
                not provided, synthetic data is generated.
            volatility: Volatility parameter for synthetic data generation.
            seed: Random seed for reproducible synthetic data.

        Returns:
            ``WalkForwardResult`` with aggregated metrics and per-window
            details.
        """
        total_days = lookback_days + walk_forward_days
        loader = HistoricalDataLoader()

        all_trades: list[dict[str, Any]] = []
        all_equity_curve: list[dict[str, Any]] = []
        window_results: list[BacktestResult] = []
        window_metrics_list: list[dict[str, Any]] = []
        current_capital = self.initial_capital

        for asset in assets:
            # Load or generate data
            if historical_data and asset in historical_data:
                data = historical_data[asset]
            else:
                data = loader.generate_synthetic(
                    asset=asset,
                    days=total_days,
                    volatility=volatility,
                    seed=seed,
                )

            if data.empty or len(data) < 48:  # Need at least 2 days of hourly data
                logger.warning(
                    "Insufficient data for {asset}, skipping", asset=asset,
                )
                continue

            # Divide data into walk-forward windows
            windows = self._create_windows(
                data, lookback_days, walk_forward_days,
            )

            for window_idx, (train_data, test_data) in enumerate(windows):
                if test_data.empty or len(test_data) < 2:
                    continue

                # Combine train + test for the simulator (it uses
                # lookback_window internally for indicator calculation)
                combined = pd.concat(
                    [train_data, test_data], ignore_index=True,
                )
                combined = combined.sort_values("timestamp").reset_index(drop=True)

                sim = BacktestSimulator(
                    initial_capital=current_capital,
                    fee_rate=self.fee_rate,
                    lookback_window=min(len(train_data), 50),
                )

                result = sim.run(combined, strategy_fn, asset=asset)

                # Collect results
                window_results.append(result)
                all_trades.extend(result.trades)
                all_equity_curve.extend(result.equity_curve)

                metrics = BacktestMetrics.calculate_all(result)
                window_metrics_list.append({
                    "window": window_idx + 1,
                    "asset": asset,
                    **metrics,
                })

                # Carry forward capital
                current_capital = result.final_equity

                logger.debug(
                    "Walk-forward window {w} | {asset} | "
                    "return={ret:+.2f}% | equity={eq:.2f}",
                    w=window_idx + 1,
                    asset=asset,
                    ret=metrics.get("total_return_pct", 0),
                    eq=result.final_equity,
                )

        # Compute aggregate metrics
        total_return = (
            ((current_capital - self.initial_capital) / self.initial_capital) * 100
            if self.initial_capital > 0
            else 0.0
        )
        sharpe = BacktestMetrics.sharpe_ratio(all_equity_curve)
        max_dd = BacktestMetrics.max_drawdown_pct(all_equity_curve)
        win_rate = BacktestMetrics.win_rate(all_trades)

        result = WalkForwardResult(
            total_return=round(total_return, 4),
            sharpe_ratio=round(sharpe, 4),
            max_drawdown=round(max_dd, 4),
            win_rate=round(win_rate, 2),
            trade_count=len(all_trades),
            equity_curve=all_equity_curve,
            window_results=window_results,
            window_metrics=window_metrics_list,
            initial_capital=self.initial_capital,
            final_equity=round(current_capital, 2),
        )

        logger.info(
            "Walk-forward backtest complete | windows={w} | trades={t} | "
            "return={r:+.2f}% | sharpe={s:.3f} | max_dd={dd:.2f}%",
            w=len(window_results),
            t=len(all_trades),
            r=total_return,
            s=sharpe,
            dd=max_dd,
        )

        return result

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    @staticmethod
    def generate_report(result: WalkForwardResult) -> str:
        """Generate a formatted text report from walk-forward results.

        Args:
            result: A ``WalkForwardResult`` from ``run_backtest()``.

        Returns:
            Multi-line formatted string summarising the backtest.
        """
        lines = [
            "=" * 60,
            "  WALK-FORWARD BACKTEST REPORT",
            "=" * 60,
            "",
            f"  Initial Capital:     ${result.initial_capital:,.2f}",
            f"  Final Equity:        ${result.final_equity:,.2f}",
            f"  Total Return:        {result.total_return:+.2f}%",
            "",
            f"  Sharpe Ratio:        {result.sharpe_ratio:.3f}",
            f"  Max Drawdown:        {result.max_drawdown:.2f}%",
            f"  Win Rate:            {result.win_rate:.1f}%",
            f"  Total Trades:        {result.trade_count}",
            f"  Walk-Forward Windows: {len(result.window_results)}",
            "",
        ]

        # Per-window breakdown
        if result.window_metrics:
            lines.append("  --- Per-Window Results ---")
            for wm in result.window_metrics:
                lines.append(
                    f"  Window {wm.get('window', '?')} ({wm.get('asset', '?')}): "
                    f"return={wm.get('total_return_pct', 0):+.2f}%, "
                    f"trades={wm.get('total_trades', 0)}, "
                    f"win_rate={wm.get('win_rate', 0):.1f}%"
                )
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Context summary for Grok
    # ------------------------------------------------------------------

    @staticmethod
    def get_performance_context(result: WalkForwardResult) -> str:
        """Generate a concise performance context block for the Grok prompt.

        This is designed to be appended to the context builder output
        so Grok has awareness of recent strategy performance.

        Args:
            result: Walk-forward backtest result.

        Returns:
            Formatted string for inclusion in the context prompt.
        """
        lines = [
            "### Strategy Performance (Walk-Forward Backtest)",
            f"- Total Return: {result.total_return:+.2f}%",
            f"- Sharpe Ratio: {result.sharpe_ratio:.3f}",
            f"- Max Drawdown: {result.max_drawdown:.2f}%",
            f"- Win Rate: {result.win_rate:.1f}%",
            f"- Total Trades: {result.trade_count}",
        ]

        # Add per-window summary if available
        if result.window_metrics:
            recent = result.window_metrics[-1]
            lines.append(
                f"- Most Recent Window: "
                f"{recent.get('total_return_pct', 0):+.2f}% return, "
                f"{recent.get('total_trades', 0)} trades"
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_windows(
        data: pd.DataFrame,
        lookback_days: int,
        walk_forward_days: int,
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Split data into (train, test) window pairs.

        Each window has:
        - train: ``lookback_days`` worth of data for indicator warm-up
        - test: ``walk_forward_days`` worth of data for evaluation

        Windows are sequential and non-overlapping on the test portion.

        Args:
            data: Full historical DataFrame sorted by timestamp.
            lookback_days: Days of training data per window.
            walk_forward_days: Days of test data per window.

        Returns:
            List of ``(train_df, test_df)`` tuples.
        """
        data = data.sort_values("timestamp").reset_index(drop=True)

        if data.empty:
            return []

        # Estimate candles per day from the data
        if len(data) >= 2:
            ts0 = data["timestamp"].iloc[0]
            ts1 = data["timestamp"].iloc[1]
            if isinstance(ts0, str):
                ts0 = datetime.fromisoformat(ts0)
            if isinstance(ts1, str):
                ts1 = datetime.fromisoformat(ts1)
            interval_hours = max((ts1 - ts0).total_seconds() / 3600, 0.01)
            candles_per_day = 24.0 / interval_hours
        else:
            candles_per_day = 24.0  # Default to 1h candles

        lookback_candles = int(lookback_days * candles_per_day)
        test_candles = int(walk_forward_days * candles_per_day)

        windows: list[tuple[pd.DataFrame, pd.DataFrame]] = []
        start_idx = 0

        while start_idx + lookback_candles + test_candles <= len(data):
            train_end = start_idx + lookback_candles
            test_end = train_end + test_candles

            train = data.iloc[start_idx:train_end].copy().reset_index(drop=True)
            test = data.iloc[train_end:test_end].copy().reset_index(drop=True)

            windows.append((train, test))

            # Advance by test_candles (non-overlapping test windows)
            start_idx += test_candles

        return windows
