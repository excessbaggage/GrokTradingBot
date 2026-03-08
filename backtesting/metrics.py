"""
Backtest metrics and result containers.

Provides the ``BacktestResult`` dataclass for storing raw simulation output
and the ``BacktestMetrics`` class for computing, printing, and exporting
comprehensive performance statistics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


@dataclass
class BacktestResult:
    """Container for raw backtest simulation output.

    Attributes:
        trades: Every simulated trade with entry/exit/pnl details.
        equity_curve: Timestamp + equity at each simulation step.
        initial_capital: Starting portfolio value.
        final_equity: Ending portfolio value after all trades.
    """

    trades: list[dict[str, Any]] = field(default_factory=list)
    equity_curve: list[dict[str, Any]] = field(default_factory=list)
    initial_capital: float = 0.0
    final_equity: float = 0.0


class BacktestMetrics:
    """Computes comprehensive performance metrics from backtest results.

    All methods are static -- no instance state needed. Call
    ``calculate_all()`` for the full metrics dictionary, or use
    individual methods for specific calculations.
    """

    # ------------------------------------------------------------------
    # Aggregate calculation
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_all(result: BacktestResult) -> dict[str, Any]:
        """Compute all performance metrics from a backtest result.

        Args:
            result: A ``BacktestResult`` from ``BacktestSimulator.run()``.

        Returns:
            Dictionary containing every metric described in this class.
        """
        trades = result.trades
        equity_curve = result.equity_curve

        metrics: dict[str, Any] = {
            "initial_capital": result.initial_capital,
            "final_equity": result.final_equity,
            "total_return_pct": BacktestMetrics.total_return_pct(result),
            "sharpe_ratio": BacktestMetrics.sharpe_ratio(equity_curve),
            "sortino_ratio": BacktestMetrics.sortino_ratio(equity_curve),
            "max_drawdown_pct": BacktestMetrics.max_drawdown_pct(equity_curve),
            "max_drawdown_duration": BacktestMetrics.max_drawdown_duration(equity_curve),
            "win_rate": BacktestMetrics.win_rate(trades),
            "profit_factor": BacktestMetrics.profit_factor(trades),
            "avg_rr_achieved": BacktestMetrics.avg_rr_achieved(trades),
            "trades_per_day": BacktestMetrics.trades_per_day(trades, equity_curve),
            "total_trades": len(trades),
            "best_trade": BacktestMetrics.best_trade(trades),
            "worst_trade": BacktestMetrics.worst_trade(trades),
            "avg_hold_duration": BacktestMetrics.avg_hold_duration(trades),
            "pnl_by_asset": BacktestMetrics.pnl_by_asset(trades),
            "pnl_by_hour": BacktestMetrics.pnl_by_hour(trades),
            "consecutive_wins": BacktestMetrics.consecutive_wins(trades),
            "consecutive_losses": BacktestMetrics.consecutive_losses(trades),
            "total_fees": sum(t.get("fees", 0.0) for t in trades),
        }
        return metrics

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    @staticmethod
    def total_return_pct(result: BacktestResult) -> float:
        """Total percentage return over the backtest period."""
        if result.initial_capital <= 0:
            return 0.0
        return ((result.final_equity - result.initial_capital) / result.initial_capital) * 100

    @staticmethod
    def sharpe_ratio(equity_curve: list[dict[str, Any]], periods_per_year: int = 365 * 24) -> float:
        """Annualised Sharpe ratio from the equity curve.

        Assumes hourly equity snapshots by default (365 * 24 = 8760
        periods per year for crypto markets).

        Args:
            equity_curve: List of dicts with ``"equity"`` key.
            periods_per_year: Number of data points per year for
                              annualisation.

        Returns:
            Sharpe ratio (float). Returns 0.0 if insufficient data.
        """
        if len(equity_curve) < 2:
            return 0.0

        equities = np.array([e["equity"] for e in equity_curve], dtype=float)
        returns = np.diff(equities) / equities[:-1]

        if len(returns) == 0:
            return 0.0

        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)

        if std_ret == 0:
            return 0.0

        return float((mean_ret / std_ret) * np.sqrt(periods_per_year))

    @staticmethod
    def sortino_ratio(equity_curve: list[dict[str, Any]], periods_per_year: int = 365 * 24) -> float:
        """Annualised Sortino ratio (penalises downside volatility only).

        Args:
            equity_curve: List of dicts with ``"equity"`` key.
            periods_per_year: Annualisation factor.

        Returns:
            Sortino ratio (float). Returns 0.0 if insufficient data.
        """
        if len(equity_curve) < 2:
            return 0.0

        equities = np.array([e["equity"] for e in equity_curve], dtype=float)
        returns = np.diff(equities) / equities[:-1]

        if len(returns) == 0:
            return 0.0

        mean_ret = np.mean(returns)
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0:
            # No downside -- infinite Sortino; cap at a large number
            return 100.0 if mean_ret > 0 else 0.0

        downside_std = np.std(downside_returns, ddof=1)
        if downside_std == 0:
            return 0.0

        return float((mean_ret / downside_std) * np.sqrt(periods_per_year))

    @staticmethod
    def max_drawdown_pct(equity_curve: list[dict[str, Any]]) -> float:
        """Maximum percentage drawdown from peak equity.

        Returns:
            Maximum drawdown as a positive percentage (e.g. 15.0 for 15%).
        """
        if not equity_curve:
            return 0.0

        equities = np.array([e["equity"] for e in equity_curve], dtype=float)
        peak = np.maximum.accumulate(equities)
        drawdowns = (peak - equities) / peak * 100

        return float(np.max(drawdowns))

    @staticmethod
    def max_drawdown_duration(equity_curve: list[dict[str, Any]]) -> str:
        """Duration of the longest drawdown period.

        Returns:
            Human-readable duration string (e.g. ``"3 days 4 hours"``),
            or ``"0"`` if no drawdown occurred.
        """
        if len(equity_curve) < 2:
            return "0"

        equities = np.array([e["equity"] for e in equity_curve], dtype=float)
        timestamps = [e.get("timestamp") for e in equity_curve]

        peak = equities[0]
        dd_start_idx = 0
        max_dd_duration = timedelta(0)

        in_drawdown = False

        for i in range(len(equities)):
            if equities[i] >= peak:
                if in_drawdown and timestamps[dd_start_idx] and timestamps[i]:
                    t_start = _parse_timestamp(timestamps[dd_start_idx])
                    t_end = _parse_timestamp(timestamps[i])
                    if t_start and t_end:
                        duration = t_end - t_start
                        if duration > max_dd_duration:
                            max_dd_duration = duration
                peak = equities[i]
                in_drawdown = False
            else:
                if not in_drawdown:
                    dd_start_idx = i
                    in_drawdown = True

        # Handle case where backtest ends in drawdown
        if in_drawdown and timestamps[dd_start_idx] and timestamps[-1]:
            t_start = _parse_timestamp(timestamps[dd_start_idx])
            t_end = _parse_timestamp(timestamps[-1])
            if t_start and t_end:
                duration = t_end - t_start
                if duration > max_dd_duration:
                    max_dd_duration = duration

        if max_dd_duration.total_seconds() == 0:
            return "0"

        days = max_dd_duration.days
        hours = max_dd_duration.seconds // 3600
        parts = []
        if days > 0:
            parts.append(f"{days} day{'s' if days != 1 else ''}")
        if hours > 0:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        return " ".join(parts) if parts else "< 1 hour"

    @staticmethod
    def win_rate(trades: list[dict[str, Any]]) -> float:
        """Percentage of trades that were profitable.

        Returns:
            Win rate as a percentage (e.g. 65.0 for 65%).
        """
        if not trades:
            return 0.0
        wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
        return (wins / len(trades)) * 100

    @staticmethod
    def profit_factor(trades: list[dict[str, Any]]) -> float:
        """Ratio of gross profit to gross loss.

        Returns:
            Profit factor (float). Returns infinity-cap of 999.0 if no
            losing trades.
        """
        gross_profit = sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0)
        gross_loss = abs(sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) < 0))

        if gross_loss == 0:
            return 999.0 if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @staticmethod
    def avg_rr_achieved(trades: list[dict[str, Any]]) -> float:
        """Average risk/reward ratio achieved across all trades.

        Computed as average winner size / average loser size.

        Returns:
            Average R:R ratio (float). Returns 0.0 if no winners or losers.
        """
        winners = [t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0]
        losers = [abs(t.get("pnl", 0)) for t in trades if t.get("pnl", 0) < 0]

        if not winners or not losers:
            return 0.0

        avg_win = sum(winners) / len(winners)
        avg_loss = sum(losers) / len(losers)

        if avg_loss == 0:
            return 0.0

        return avg_win / avg_loss

    @staticmethod
    def trades_per_day(
        trades: list[dict[str, Any]],
        equity_curve: list[dict[str, Any]],
    ) -> float:
        """Average number of trades per calendar day.

        Returns:
            Trades per day (float).
        """
        if not trades or len(equity_curve) < 2:
            return 0.0

        first_ts = _parse_timestamp(equity_curve[0].get("timestamp"))
        last_ts = _parse_timestamp(equity_curve[-1].get("timestamp"))

        if not first_ts or not last_ts:
            return 0.0

        days = max((last_ts - first_ts).total_seconds() / 86400, 1)
        return len(trades) / days

    @staticmethod
    def best_trade(trades: list[dict[str, Any]]) -> dict[str, Any]:
        """Return the trade with the highest PnL.

        Returns:
            Trade dict, or empty dict if no trades.
        """
        if not trades:
            return {}
        return max(trades, key=lambda t: t.get("pnl", 0))

    @staticmethod
    def worst_trade(trades: list[dict[str, Any]]) -> dict[str, Any]:
        """Return the trade with the lowest PnL.

        Returns:
            Trade dict, or empty dict if no trades.
        """
        if not trades:
            return {}
        return min(trades, key=lambda t: t.get("pnl", 0))

    @staticmethod
    def avg_hold_duration(trades: list[dict[str, Any]]) -> str:
        """Average holding duration across all trades.

        Returns:
            Human-readable duration string.
        """
        durations: list[float] = []
        for t in trades:
            entry_ts = _parse_timestamp(t.get("entry_time"))
            exit_ts = _parse_timestamp(t.get("exit_time"))
            if entry_ts and exit_ts:
                durations.append((exit_ts - entry_ts).total_seconds())

        if not durations:
            return "N/A"

        avg_seconds = sum(durations) / len(durations)
        hours = int(avg_seconds // 3600)
        minutes = int((avg_seconds % 3600) // 60)

        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        parts.append(f"{minutes}m")
        return " ".join(parts)

    @staticmethod
    def pnl_by_asset(trades: list[dict[str, Any]]) -> dict[str, float]:
        """Total PnL breakdown by asset symbol.

        Returns:
            Dict mapping asset names to total PnL.
        """
        breakdown: dict[str, float] = {}
        for t in trades:
            asset = t.get("asset", "UNKNOWN")
            breakdown[asset] = breakdown.get(asset, 0.0) + t.get("pnl", 0.0)
        return {k: round(v, 2) for k, v in breakdown.items()}

    @staticmethod
    def pnl_by_hour(trades: list[dict[str, Any]]) -> dict[int, float]:
        """Total PnL breakdown by hour of day (UTC).

        Returns:
            Dict mapping hour (0-23) to total PnL for that hour.
        """
        breakdown: dict[int, float] = {h: 0.0 for h in range(24)}
        for t in trades:
            ts = _parse_timestamp(t.get("entry_time"))
            if ts:
                hour = ts.hour
                breakdown[hour] += t.get("pnl", 0.0)
        return {k: round(v, 2) for k, v in breakdown.items()}

    @staticmethod
    def consecutive_wins(trades: list[dict[str, Any]]) -> int:
        """Maximum consecutive winning trades."""
        return _max_consecutive(trades, winning=True)

    @staticmethod
    def consecutive_losses(trades: list[dict[str, Any]]) -> int:
        """Maximum consecutive losing trades."""
        return _max_consecutive(trades, winning=False)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def print_report(metrics: dict[str, Any]) -> None:
        """Print a formatted performance report to the terminal.

        Args:
            metrics: The output of ``calculate_all()``.
        """
        print("\n" + "=" * 70)
        print("  BACKTEST PERFORMANCE REPORT")
        print("=" * 70)

        print(f"\n  Initial Capital:      ${metrics.get('initial_capital', 0):,.2f}")
        print(f"  Final Equity:         ${metrics.get('final_equity', 0):,.2f}")
        print(f"  Total Return:         {metrics.get('total_return_pct', 0):+.2f}%")
        print(f"  Total Fees:           ${metrics.get('total_fees', 0):,.2f}")

        print(f"\n  {'--- Risk Metrics ---':^40}")
        print(f"  Sharpe Ratio:         {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"  Sortino Ratio:        {metrics.get('sortino_ratio', 0):.3f}")
        print(f"  Max Drawdown:         {metrics.get('max_drawdown_pct', 0):.2f}%")
        print(f"  Max DD Duration:      {metrics.get('max_drawdown_duration', 'N/A')}")

        print(f"\n  {'--- Trade Stats ---':^40}")
        print(f"  Total Trades:         {metrics.get('total_trades', 0)}")
        print(f"  Trades/Day:           {metrics.get('trades_per_day', 0):.1f}")
        print(f"  Win Rate:             {metrics.get('win_rate', 0):.1f}%")
        print(f"  Profit Factor:        {metrics.get('profit_factor', 0):.2f}")
        print(f"  Avg R:R Achieved:     {metrics.get('avg_rr_achieved', 0):.2f}")
        print(f"  Avg Hold Duration:    {metrics.get('avg_hold_duration', 'N/A')}")
        print(f"  Max Consec. Wins:     {metrics.get('consecutive_wins', 0)}")
        print(f"  Max Consec. Losses:   {metrics.get('consecutive_losses', 0)}")

        best = metrics.get("best_trade", {})
        worst = metrics.get("worst_trade", {})
        if best:
            print(f"\n  Best Trade:           ${best.get('pnl', 0):+,.2f} ({best.get('asset', '?')})")
        if worst:
            print(f"  Worst Trade:          ${worst.get('pnl', 0):+,.2f} ({worst.get('asset', '?')})")

        pnl_asset = metrics.get("pnl_by_asset", {})
        if pnl_asset:
            print(f"\n  {'--- PnL by Asset ---':^40}")
            for asset, pnl in sorted(pnl_asset.items()):
                print(f"  {asset:>6}:  ${pnl:+,.2f}")

        print("\n" + "=" * 70 + "\n")

    @staticmethod
    def export_report(metrics: dict[str, Any], filepath: str | Path) -> None:
        """Save the full metrics dictionary as a JSON file.

        Args:
            metrics: The output of ``calculate_all()``.
            filepath: Destination file path.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Make metrics JSON-serializable
        serializable = _make_serializable(metrics)

        with open(filepath, "w") as f:
            json.dump(serializable, f, indent=2, default=str)

        logger.info("Backtest report exported to {path}", path=str(filepath))


# ======================================================================
# Module-level helpers
# ======================================================================


def _parse_timestamp(ts: Any) -> datetime | None:
    """Parse a timestamp from various formats into a timezone-aware datetime."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=None)  # keep naive for delta calculations
        return ts
    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts)
            return dt
        except ValueError:
            return None
    return None


def _max_consecutive(trades: list[dict[str, Any]], winning: bool) -> int:
    """Count the maximum consecutive winning or losing trades."""
    if not trades:
        return 0

    max_streak = 0
    current_streak = 0

    for t in trades:
        pnl = t.get("pnl", 0)
        is_match = (pnl > 0) if winning else (pnl < 0)

        if is_match:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    return max_streak


def _make_serializable(obj: Any) -> Any:
    """Recursively convert non-JSON-serializable objects."""
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
