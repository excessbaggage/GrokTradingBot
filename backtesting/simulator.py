"""
Event-driven backtest simulator for the Grok Trader trading bot.

Steps through historical candles chronologically, calls a user-provided
decision callback to generate ``TradeDecision`` objects, validates them
through the real ``RiskGuardian``, and simulates fills at realistic
prices (next candle's open -- no look-ahead bias).

SL/TP checking uses each candle's high/low with a direction heuristic
when both levels are hit within the same candle.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd
from loguru import logger

from backtesting.metrics import BacktestResult
from brain.models import TradeDecision
from config.risk_config import RISK_PARAMS
from execution.risk_guardian import RiskGuardian


# ======================================================================
# Internal position tracking
# ======================================================================


@dataclass
class _SimulatedPosition:
    """Tracks an open position inside the simulator."""

    asset: str
    side: str  # "long" or "short"
    entry_price: float
    size_usd: float  # Notional USD value at entry
    size_pct: float  # Fraction of equity at entry
    leverage: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    fees_paid: float = 0.0


# ======================================================================
# Market snapshot (what the strategy callback receives)
# ======================================================================


@dataclass
class MarketSnapshot:
    """A point-in-time market view passed to the decision callback.

    Contains the current candle plus a lookback window of recent candles
    so strategies can compute indicators (RSI, ATR, etc.).

    Attributes:
        asset: The asset symbol (e.g. ``"BTC"``).
        timestamp: The current candle's timestamp.
        open: Current candle open price.
        high: Current candle high.
        low: Current candle low.
        close: Current candle close.
        volume: Current candle volume.
        candles: DataFrame of the most recent candles (including current).
        equity: Current portfolio equity.
        positions: Dict of currently open positions by asset.
    """

    asset: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    candles: pd.DataFrame
    equity: float
    positions: dict[str, dict[str, Any]] = field(default_factory=dict)


# Type alias for the decision callback
DecisionCallback = Callable[[MarketSnapshot], list[TradeDecision]]


class BacktestSimulator:
    """Event-driven backtesting engine with full RiskGuardian integration.

    Steps through historical candles one by one, invokes a decision
    callback for each candle, validates decisions through the real
    ``RiskGuardian``, and tracks positions with SL/TP simulation.

    Usage::

        sim = BacktestSimulator(initial_capital=10_000)
        result = sim.run(historical_data, my_strategy)
        # result is a BacktestResult with trades + equity curve

    Args:
        initial_capital: Starting portfolio value in USD.
        risk_params: Override risk parameters (defaults to ``RISK_PARAMS``).
        fee_rate: Fee rate per trade side (default 0.035% taker).
        lookback_window: Number of candles to include in the snapshot
                         for indicator computation.
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        risk_params: dict[str, Any] | None = None,
        fee_rate: float = 0.00035,
        lookback_window: int = 50,
    ) -> None:
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.lookback_window = lookback_window

        # Build risk params -- relax time-between-trades and daily-trade-count
        # for backtesting (these checks are meaningless in simulated time).
        self._risk_params = dict(risk_params or RISK_PARAMS)
        self._risk_params["min_time_between_trades_minutes"] = 0
        self._risk_params["max_trades_per_day"] = 999_999

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(
        self,
        historical_data: pd.DataFrame,
        decisions_callback: DecisionCallback,
        asset: str = "BTC",
    ) -> BacktestResult:
        """Execute the backtest over the provided historical data.

        Args:
            historical_data: DataFrame with columns: timestamp, open,
                             high, low, close, volume.
            decisions_callback: A function that receives a
                                ``MarketSnapshot`` and returns a list
                                of ``TradeDecision`` objects.
            asset: The asset symbol being backtested.

        Returns:
            A ``BacktestResult`` containing all trades and the equity
            curve.
        """
        if historical_data.empty or len(historical_data) < 2:
            logger.warning("Insufficient data for backtest (need at least 2 candles)")
            return BacktestResult(
                initial_capital=self.initial_capital,
                final_equity=self.initial_capital,
            )

        # Initialise state
        equity = self.initial_capital
        peak_equity = equity
        positions: dict[str, _SimulatedPosition] = {}
        completed_trades: list[dict[str, Any]] = []
        equity_curve: list[dict[str, Any]] = []
        daily_pnl: float = 0.0
        weekly_pnl: float = 0.0
        current_day: int | None = None
        current_week: int | None = None

        # Create an in-memory DB and RiskGuardian for validation
        db_conn = self._create_backtest_db()
        guardian = RiskGuardian(risk_params=self._risk_params)

        candles = historical_data.sort_values("timestamp").reset_index(drop=True)

        logger.info(
            "Starting backtest | capital={cap} | candles={n} | asset={asset}",
            cap=self.initial_capital,
            n=len(candles),
            asset=asset,
        )

        for i in range(len(candles)):
            row = candles.iloc[i]
            ts = row["timestamp"]
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)

            # Reset daily/weekly PnL tracking
            if hasattr(ts, "day"):
                day = ts.timetuple().tm_yday
                week = ts.isocalendar()[1]
                if current_day is not None and day != current_day:
                    daily_pnl = 0.0
                if current_week is not None and week != current_week:
                    weekly_pnl = 0.0
                current_day = day
                current_week = week

            # ----------------------------------------------------------
            # Step 1: Check SL/TP against current candle for open positions
            # ----------------------------------------------------------
            closed_this_candle = self._check_sl_tp(
                positions=positions,
                candle=row,
                timestamp=ts,
                equity=equity,
            )

            for closed_trade in closed_this_candle:
                completed_trades.append(closed_trade)
                pnl = closed_trade.get("pnl", 0.0)
                # Use pre-PnL equity for percentage calculation
                daily_pnl += pnl / equity if equity > 0 else 0
                weekly_pnl += pnl / equity if equity > 0 else 0
                equity += pnl
                peak_equity = max(peak_equity, equity)

                # Remove from positions dict
                closed_asset = closed_trade.get("asset", "")
                if closed_asset in positions:
                    del positions[closed_asset]

            # ----------------------------------------------------------
            # Step 2: Build market snapshot and call decision callback
            # ----------------------------------------------------------
            lookback_start = max(0, i - self.lookback_window + 1)
            lookback_df = candles.iloc[lookback_start : i + 1].copy()

            # Build positions summary for the snapshot
            pos_summary = {}
            for a, pos in positions.items():
                pos_summary[a] = {
                    "side": pos.side,
                    "entry_price": pos.entry_price,
                    "size_pct": pos.size_pct,
                    "stop_loss": pos.stop_loss,
                    "take_profit": pos.take_profit,
                }

            snapshot = MarketSnapshot(
                asset=asset,
                timestamp=ts,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                candles=lookback_df,
                equity=equity,
                positions=pos_summary,
            )

            try:
                decisions = decisions_callback(snapshot)
            except Exception as exc:
                logger.warning(
                    "Decision callback failed at {ts}: {err}", ts=ts, err=exc,
                )
                decisions = []

            # ----------------------------------------------------------
            # Step 3: Validate decisions through RiskGuardian and fill
            # ----------------------------------------------------------
            if decisions and i + 1 < len(candles):
                # Fill price = next candle's open (no look-ahead bias)
                fill_price = float(candles.iloc[i + 1]["open"])

                portfolio_state = {
                    "equity": equity,
                    "peak_equity": peak_equity,
                    "daily_pnl_pct": daily_pnl,
                    "weekly_pnl_pct": weekly_pnl,
                    "total_exposure_pct": self._total_exposure(positions, equity),
                }

                for decision in decisions:
                    # Skip non-actionable decisions
                    if decision.action in ("hold", "no_trade"):
                        continue

                    # Handle close actions
                    if decision.action == "close":
                        if decision.asset in positions:
                            pos = positions[decision.asset]
                            closed_trade = self._close_position(
                                pos, fill_price, ts, "manual_close",
                            )
                            completed_trades.append(closed_trade)
                            pnl = closed_trade["pnl"]
                            equity += pnl
                            daily_pnl += pnl / equity if equity > 0 else 0
                            weekly_pnl += pnl / equity if equity > 0 else 0
                            peak_equity = max(peak_equity, equity)
                            del positions[decision.asset]
                        continue

                    # Skip if we already have a position in this asset
                    if decision.asset in positions:
                        continue

                    # Override entry_price with the realistic fill price
                    decision_with_fill = decision.model_copy(
                        update={"entry_price": fill_price}
                    )

                    # Recompute SL/TP relative to fill price if they seem
                    # unreasonable (e.g. strategy set them based on signal
                    # candle close, not fill price). Keep the percentage
                    # distances from the original decision.
                    if decision.entry_price and decision.entry_price > 0:
                        sl_dist = abs(decision.stop_loss - decision.entry_price) / decision.entry_price
                        tp_dist = abs(decision.take_profit - decision.entry_price) / decision.entry_price
                    else:
                        sl_dist = abs(decision.stop_loss - fill_price) / fill_price if fill_price > 0 else 0.03
                        tp_dist = abs(decision.take_profit - fill_price) / fill_price if fill_price > 0 else 0.06

                    is_long = decision.action == "open_long"
                    if is_long:
                        new_sl = fill_price * (1 - sl_dist)
                        new_tp = fill_price * (1 + tp_dist)
                    else:
                        new_sl = fill_price * (1 + sl_dist)
                        new_tp = fill_price * (1 - tp_dist)

                    decision_with_fill = decision_with_fill.model_copy(
                        update={
                            "stop_loss": round(new_sl, 2),
                            "take_profit": round(new_tp, 2),
                        }
                    )

                    # Recompute R:R for the adjusted prices
                    risk = abs(fill_price - decision_with_fill.stop_loss)
                    reward = abs(decision_with_fill.take_profit - fill_price)
                    rr = reward / risk if risk > 0 else 0
                    decision_with_fill = decision_with_fill.model_copy(
                        update={"risk_reward_ratio": round(rr, 2)}
                    )

                    # Update portfolio state for exposure calculation
                    portfolio_state["total_exposure_pct"] = self._total_exposure(
                        positions, equity,
                    )

                    # Validate through RiskGuardian
                    validation = guardian.validate(
                        decision_with_fill, portfolio_state, db_conn,
                    )

                    if not validation.approved:
                        logger.debug(
                            "Backtest: decision rejected | {asset} {action} | {reason}",
                            asset=decision.asset,
                            action=decision.action,
                            reason=validation.reason,
                        )
                        continue

                    # Open the position
                    notional = decision_with_fill.size_pct * equity * decision_with_fill.leverage
                    entry_fee = notional * self.fee_rate
                    equity -= entry_fee

                    positions[decision.asset] = _SimulatedPosition(
                        asset=decision.asset,
                        side="long" if is_long else "short",
                        entry_price=fill_price,
                        size_usd=notional,
                        size_pct=decision_with_fill.size_pct,
                        leverage=decision_with_fill.leverage,
                        stop_loss=decision_with_fill.stop_loss,
                        take_profit=decision_with_fill.take_profit,
                        entry_time=ts,
                        fees_paid=entry_fee,
                    )

                    # Record in the backtest DB so time-between-trades works
                    self._record_trade(db_conn, decision.asset, ts, positions[decision.asset])

            # ----------------------------------------------------------
            # Step 4: Record equity curve
            # ----------------------------------------------------------
            # Include unrealised P&L from open positions
            unrealised = self._unrealised_pnl(positions, float(row["close"]))
            equity_curve.append(
                {
                    "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                    "equity": round(equity + unrealised, 2),
                    "open_positions": len(positions),
                }
            )

        # ----------------------------------------------------------
        # Force-close any remaining positions at last candle's close
        # ----------------------------------------------------------
        if positions:
            last_row = candles.iloc[-1]
            last_close = float(last_row["close"])
            last_ts = last_row["timestamp"]
            if isinstance(last_ts, str):
                last_ts = datetime.fromisoformat(last_ts)

            for asset_key in list(positions.keys()):
                pos = positions[asset_key]
                closed_trade = self._close_position(pos, last_close, last_ts, "end_of_backtest")
                completed_trades.append(closed_trade)
                equity += closed_trade["pnl"]

            positions.clear()

        db_conn.close()

        result = BacktestResult(
            trades=completed_trades,
            equity_curve=equity_curve,
            initial_capital=self.initial_capital,
            final_equity=round(equity, 2),
        )

        logger.info(
            "Backtest complete | trades={n} | return={ret:.2f}% | "
            "final_equity={eq:.2f}",
            n=len(completed_trades),
            ret=((equity - self.initial_capital) / self.initial_capital) * 100,
            eq=equity,
        )

        return result

    # ------------------------------------------------------------------
    # SL/TP checking
    # ------------------------------------------------------------------

    def _check_sl_tp(
        self,
        positions: dict[str, _SimulatedPosition],
        candle: Any,
        timestamp: datetime,
        equity: float,
    ) -> list[dict[str, Any]]:
        """Check if any open positions hit their stop-loss or take-profit.

        For each candle, checks high/low against SL/TP levels. When both
        are hit within the same candle, uses a direction heuristic: if
        the candle is bullish (close > open), assume lows were hit first;
        if bearish, highs were hit first.

        Args:
            positions: Currently open positions.
            candle: Current candle row from the DataFrame.
            timestamp: Current candle timestamp.
            equity: Current portfolio equity.

        Returns:
            List of closed trade dicts.
        """
        closed: list[dict[str, Any]] = []
        high = float(candle["high"])
        low = float(candle["low"])
        candle_open = float(candle["open"])
        candle_close = float(candle["close"])
        is_bullish = candle_close >= candle_open

        for asset, pos in list(positions.items()):
            sl_hit = False
            tp_hit = False

            if pos.side == "long":
                sl_hit = low <= pos.stop_loss
                tp_hit = high >= pos.take_profit
            else:  # short
                sl_hit = high >= pos.stop_loss
                tp_hit = low <= pos.take_profit

            if sl_hit and tp_hit:
                # Both hit in same candle -- use direction heuristic
                if pos.side == "long":
                    # Bullish candle: price likely went down first (SL) then up (TP)
                    # Bearish candle: price likely went up first (TP) then down (SL)
                    if is_bullish:
                        # Assume SL hit first (worst case for long)
                        exit_price = pos.stop_loss
                        exit_reason = "stop_loss"
                    else:
                        # Assume TP hit first (favorable for long before reversal)
                        exit_price = pos.take_profit
                        exit_reason = "take_profit"
                else:  # short
                    if is_bullish:
                        # Bullish: price went down first (TP for short) then up
                        exit_price = pos.take_profit
                        exit_reason = "take_profit"
                    else:
                        # Bearish: price went up first (SL for short) then down
                        exit_price = pos.stop_loss
                        exit_reason = "stop_loss"
            elif sl_hit:
                exit_price = pos.stop_loss
                exit_reason = "stop_loss"
            elif tp_hit:
                exit_price = pos.take_profit
                exit_reason = "take_profit"
            else:
                continue

            closed_trade = self._close_position(pos, exit_price, timestamp, exit_reason)
            closed.append(closed_trade)

        return closed

    # ------------------------------------------------------------------
    # Position closing
    # ------------------------------------------------------------------

    def _close_position(
        self,
        pos: _SimulatedPosition,
        exit_price: float,
        timestamp: datetime,
        exit_reason: str,
    ) -> dict[str, Any]:
        """Close a simulated position and compute PnL.

        Args:
            pos: The position to close.
            exit_price: Price at which the position is closed.
            timestamp: Time of the close.
            exit_reason: Why the position was closed (stop_loss,
                         take_profit, manual_close, end_of_backtest).

        Returns:
            Dict with full trade details including PnL.
        """
        # Compute raw PnL
        if pos.side == "long":
            price_pnl = (exit_price - pos.entry_price) / pos.entry_price
        else:
            price_pnl = (pos.entry_price - exit_price) / pos.entry_price

        # PnL in USD terms (applies leverage)
        raw_pnl = price_pnl * pos.size_usd

        # Exit fee
        exit_notional = pos.size_usd * (1 + price_pnl)
        exit_fee = abs(exit_notional) * self.fee_rate
        total_fees = pos.fees_paid + exit_fee

        net_pnl = raw_pnl - exit_fee

        return {
            "asset": pos.asset,
            "side": pos.side,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "size_usd": pos.size_usd,
            "size_pct": pos.size_pct,
            "leverage": pos.leverage,
            "pnl": round(net_pnl, 2),
            "pnl_pct": round(price_pnl * 100, 4),
            "fees": round(total_fees, 2),
            "exit_reason": exit_reason,
            "entry_time": pos.entry_time.isoformat() if hasattr(pos.entry_time, "isoformat") else str(pos.entry_time),
            "exit_time": timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _total_exposure(
        self,
        positions: dict[str, _SimulatedPosition],
        equity: float,
    ) -> float:
        """Calculate total portfolio exposure as a fraction of equity."""
        if equity <= 0:
            return 0.0
        total = sum(p.size_pct for p in positions.values())
        return total

    def _unrealised_pnl(
        self,
        positions: dict[str, _SimulatedPosition],
        current_price: float,
    ) -> float:
        """Calculate total unrealised PnL across all open positions."""
        total = 0.0
        for pos in positions.values():
            if pos.side == "long":
                pnl_pct = (current_price - pos.entry_price) / pos.entry_price
            else:
                pnl_pct = (pos.entry_price - current_price) / pos.entry_price
            total += pnl_pct * pos.size_usd
        return total

    @staticmethod
    def _create_backtest_db() -> sqlite3.Connection:
        """Create an in-memory SQLite DB with the trades table.

        This is used by the RiskGuardian for time-between-trades and
        daily-trade-count checks.
        """
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset TEXT NOT NULL,
                action TEXT NOT NULL,
                side TEXT,
                size_pct REAL,
                leverage REAL,
                entry_price REAL,
                exit_price REAL,
                stop_loss REAL,
                take_profit REAL,
                status TEXT DEFAULT 'open',
                opened_at TEXT NOT NULL,
                closed_at TEXT,
                realized_pnl REAL DEFAULT 0.0
            )
            """
        )
        conn.commit()
        return conn

    @staticmethod
    def _record_trade(
        conn: sqlite3.Connection,
        asset: str,
        timestamp: datetime,
        position: _SimulatedPosition,
    ) -> None:
        """Record a trade in the backtest DB for RiskGuardian queries."""
        ts_str = timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp)
        action = f"open_{position.side}"
        conn.execute(
            """
            INSERT INTO trades (asset, action, side, size_pct, leverage,
                                entry_price, stop_loss, take_profit, status, opened_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)
            """,
            (
                asset, action, position.side, position.size_pct,
                position.leverage, position.entry_price,
                position.stop_loss, position.take_profit, ts_str,
            ),
        )
        conn.commit()
