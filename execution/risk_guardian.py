"""
Risk Guardian -- the FINAL AUTHORITY on every trade decision.

This module enforces hard-coded risk limits that the AI (Grok) CANNOT bypass.
Every trade decision must pass through the Risk Guardian before execution.
If any single check fails, the trade is rejected outright.

The Risk Guardian is intentionally conservative.  It protects real capital
by enforcing position sizing, drawdown limits, and a kill switch that
halts all trading when risk thresholds are breached.

Design principles:
    - Fail closed: any ambiguity results in rejection.
    - No runtime modification of limits (they come from risk_config.py).
    - Every rejection is logged with a clear reason.
    - The kill switch, once activated, requires manual reset.

Check order (14 sequential checks -- ALL must pass):
    1.  kill_switch           -- global halt
    2.  daily_loss_limit      -- daily P&L gate
    3.  weekly_loss_limit     -- weekly P&L gate
    4.  total_drawdown        -- peak drawdown -> auto kill switch
    5.  position_size         -- per-position cap
    6.  total_exposure        -- portfolio-wide exposure cap
    7.  leverage              -- hard leverage cap
    8.  stop_loss_exists      -- SL must be present
    9.  stop_loss_distance    -- SL distance from entry
    10. take_profit_exists    -- TP must be present
    11. risk_reward_ratio     -- minimum R:R
    12. time_between_trades   -- cooldown timer
    13. daily_trade_count     -- max trades per day
    14. correlation_risk      -- reject correlated positions
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger

from brain.models import RiskValidationResult, TradeDecision
from config.risk_config import RISK_PARAMS
from data.correlation_risk import CorrelationRiskManager


class RiskGuardian:
    """Validates every trade decision against hard-coded risk parameters.

    The Risk Guardian runs a gauntlet of 13 sequential checks.  A trade
    must pass ALL checks to be approved.  The checks are ordered so that
    the cheapest / most-likely-to-reject run first.

    Attributes:
        params: A frozen copy of the risk parameter dictionary.
        _kill_switch: Internal flag -- once True, ALL trades are rejected
                      until manually reset.
        _kill_switch_reason: Human-readable reason the kill switch was tripped.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, risk_params: dict[str, Any] | None = None) -> None:
        """Initialise the Risk Guardian.

        Args:
            risk_params: Risk parameter dictionary.  Falls back to the
                         global ``RISK_PARAMS`` from ``config.risk_config``
                         if not supplied.
        """
        # Freeze a copy so nothing can mutate our limits at runtime.
        self.params: dict[str, Any] = dict(risk_params or RISK_PARAMS)

        # Kill switch state
        self._kill_switch: bool = bool(self.params.get("kill_switch_enabled", False))
        self._kill_switch_reason: str = (
            "Manually enabled in config" if self._kill_switch else ""
        )

        logger.info(
            "RiskGuardian initialised | kill_switch={ks} | max_pos={mp}% | "
            "max_exposure={me}% | max_leverage={ml}x | max_daily_loss={mdl}%",
            ks=self._kill_switch,
            mp=self.params["max_position_size_pct"] * 100,
            me=self.params["max_total_exposure_pct"] * 100,
            ml=self.params["max_leverage"],
            mdl=self.params["max_daily_loss_pct"] * 100,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        decision: TradeDecision,
        portfolio_state: dict[str, Any],
        db_connection: Any,
        market_data: dict[str, dict[str, Any]] | None = None,
        open_positions: list[str] | None = None,
    ) -> RiskValidationResult:
        """Run the full risk-check gauntlet on a proposed trade decision.

        Args:
            decision: The ``TradeDecision`` produced by Grok.
            portfolio_state: Current portfolio snapshot.  Expected keys:
                ``equity``          -- current account equity (float)
                ``peak_equity``     -- highest equity ever recorded (float)
                ``daily_pnl_pct``   -- today's P&L as a decimal (negative = loss)
                ``weekly_pnl_pct``  -- this week's P&L as a decimal
                ``total_exposure_pct`` -- sum of open position sizes / equity
            db_connection: SQLite connection for querying trade history.
            market_data: Optional market data dict for correlation checks.
                If not provided, correlation check uses group fallback only.
            open_positions: Optional list of currently open position asset
                symbols for correlation checks.

        Returns:
            ``RiskValidationResult`` with ``approved=True`` if the trade
            passes all checks, otherwise ``approved=False`` with a
            human-readable ``reason``.
        """
        # Non-actionable decisions are always approved -- nothing to risk.
        if decision.action in ("hold", "no_trade"):
            return RiskValidationResult(
                approved=True,
                reason="Non-actionable decision; no risk check required.",
            )

        # Close / adjust_stop reduce risk -- always allowed through the
        # gauntlet (we still want to be able to exit positions even when
        # the kill switch is active).
        if decision.action in ("close", "adjust_stop"):
            return RiskValidationResult(
                approved=True,
                reason="Close/adjust actions always permitted.",
            )

        # Store context for check #14 (correlation risk)
        self._current_market_data = market_data
        self._current_open_positions = open_positions

        # Run every check in strict order.
        checks = [
            self._check_kill_switch,
            self._check_daily_loss_limit,
            self._check_weekly_loss_limit,
            self._check_total_drawdown,
            self._check_position_size,
            self._check_total_exposure,
            self._check_leverage,
            self._check_stop_loss_exists,
            self._check_stop_loss_distance,
            self._check_take_profit_exists,
            self._check_risk_reward_ratio,
            self._check_time_between_trades,
            self._check_daily_trade_count,
            self._check_correlation_risk,
        ]

        for check_fn in checks:
            result = check_fn(decision, portfolio_state, db_connection)
            if not result.approved:
                logger.warning(
                    "RISK REJECTED | asset={asset} action={action} | {reason}",
                    asset=decision.asset,
                    action=decision.action,
                    reason=result.reason,
                )
                return result

        logger.info(
            "RISK APPROVED | asset={asset} action={action} size={size}% lev={lev}x",
            asset=decision.asset,
            action=decision.action,
            size=round(decision.size_pct * 100, 2),
            lev=decision.leverage,
        )
        return RiskValidationResult(approved=True, reason="All 14 risk checks passed.")

    # ------------------------------------------------------------------
    # Kill switch helpers
    # ------------------------------------------------------------------

    def kill_switch_active(self) -> bool:
        """Return whether the kill switch is currently engaged."""
        return self._kill_switch

    def activate_kill_switch(self, reason: str) -> None:
        """Activate the kill switch.  ALL new trades will be rejected.

        Args:
            reason: Why the kill switch was activated (logged and stored).
        """
        self._kill_switch = True
        self._kill_switch_reason = reason
        logger.critical(
            "KILL SWITCH ACTIVATED | reason={reason}", reason=reason,
        )

    def deactivate_kill_switch(self) -> None:
        """Manually deactivate the kill switch after human review.

        This is the ONLY way to resume trading after a kill-switch event.
        """
        logger.warning("KILL SWITCH DEACTIVATED by manual override")
        self._kill_switch = False
        self._kill_switch_reason = ""

    # ------------------------------------------------------------------
    # Risk status summary
    # ------------------------------------------------------------------

    def calculate_risk_status(
        self,
        portfolio_state: dict[str, Any],
        db_connection: Any,
    ) -> dict[str, Any]:
        """Compute a snapshot of remaining risk headroom.

        Args:
            portfolio_state: Current portfolio snapshot (same schema as ``validate``).
            db_connection: SQLite connection for querying trade history.

        Returns:
            Dictionary with remaining daily/weekly loss budget, drawdown
            from peak, trade count, and kill switch state.
        """
        equity = portfolio_state.get("equity", 0.0)
        peak = portfolio_state.get("peak_equity", equity)

        daily_pnl_pct = portfolio_state.get("daily_pnl_pct", 0.0)
        weekly_pnl_pct = portfolio_state.get("weekly_pnl_pct", 0.0)
        daily_loss = abs(min(daily_pnl_pct, 0.0))
        weekly_loss = abs(min(weekly_pnl_pct, 0.0))
        drawdown_pct = (peak - equity) / peak if peak > 0 else 0.0

        trades_today = self._count_trades_today(db_connection)

        # Compute absolute USD P&L values for the context builder
        daily_pnl_usd = daily_pnl_pct * equity if equity > 0 else 0.0
        weekly_pnl_usd = weekly_pnl_pct * equity if equity > 0 else 0.0

        return {
            "daily_loss_remaining": round(
                self.params["max_daily_loss_pct"] - daily_loss, 6,
            ),
            "weekly_loss_remaining": round(
                self.params["max_weekly_loss_pct"] - weekly_loss, 6,
            ),
            "drawdown_from_peak": round(drawdown_pct, 6),
            "trades_today": trades_today,
            "max_trades_per_day": self.params["max_trades_per_day"],
            "total_exposure_pct": portfolio_state.get("total_exposure_pct", 0.0),
            "max_total_exposure_pct": self.params["max_total_exposure_pct"],
            "kill_switch_active": self._kill_switch,
            "kill_switch_reason": self._kill_switch_reason,
            # Keys consumed by context_builder.py
            "daily_pnl": round(daily_pnl_usd, 2),
            "weekly_pnl": round(weekly_pnl_usd, 2),
            "consecutive_losses": portfolio_state.get("consecutive_losses", 0),
        }

    # ------------------------------------------------------------------
    # Individual risk checks (private)
    # ------------------------------------------------------------------

    def _check_kill_switch(
        self,
        decision: TradeDecision,
        portfolio_state: dict[str, Any],
        db_connection: Any,
    ) -> RiskValidationResult:
        """Check 1: If the kill switch is engaged, reject everything."""
        if self._kill_switch:
            return RiskValidationResult(
                approved=False,
                reason=(
                    f"Kill switch is active: {self._kill_switch_reason}. "
                    f"All trading halted."
                ),
            )
        return RiskValidationResult(approved=True)

    def _check_daily_loss_limit(
        self,
        decision: TradeDecision,
        portfolio_state: dict[str, Any],
        db_connection: Any,
    ) -> RiskValidationResult:
        """Check 2: Reject if daily losses have reached the threshold."""
        daily_pnl_pct = portfolio_state.get("daily_pnl_pct", 0.0)
        max_daily = self.params["max_daily_loss_pct"]

        # daily_pnl_pct is negative when losing money.
        if daily_pnl_pct <= -max_daily:
            return RiskValidationResult(
                approved=False,
                reason=(
                    f"Daily loss limit reached: {daily_pnl_pct:.2%} loss "
                    f"(limit: -{max_daily:.2%}). Trading halted until next day."
                ),
            )
        return RiskValidationResult(approved=True)

    def _check_weekly_loss_limit(
        self,
        decision: TradeDecision,
        portfolio_state: dict[str, Any],
        db_connection: Any,
    ) -> RiskValidationResult:
        """Check 3: Reject if weekly losses have reached the threshold."""
        weekly_pnl_pct = portfolio_state.get("weekly_pnl_pct", 0.0)
        max_weekly = self.params["max_weekly_loss_pct"]

        if weekly_pnl_pct <= -max_weekly:
            return RiskValidationResult(
                approved=False,
                reason=(
                    f"Weekly loss limit reached: {weekly_pnl_pct:.2%} loss "
                    f"(limit: -{max_weekly:.2%}). Trading halted for 48 hours."
                ),
            )
        return RiskValidationResult(approved=True)

    def _check_total_drawdown(
        self,
        decision: TradeDecision,
        portfolio_state: dict[str, Any],
        db_connection: Any,
    ) -> RiskValidationResult:
        """Check 4: If drawdown from peak exceeds limit, activate kill switch."""
        equity = portfolio_state.get("equity", 0.0)
        peak = portfolio_state.get("peak_equity", equity)

        if peak <= 0:
            return RiskValidationResult(approved=True)

        drawdown_pct = (peak - equity) / peak
        max_drawdown = self.params["max_total_drawdown_pct"]

        if drawdown_pct >= max_drawdown:
            reason = (
                f"Total drawdown {drawdown_pct:.2%} has breached the maximum "
                f"allowed {max_drawdown:.2%}. Equity: {equity:.2f}, "
                f"Peak: {peak:.2f}. KILL SWITCH ACTIVATED."
            )
            self.activate_kill_switch(reason)
            return RiskValidationResult(approved=False, reason=reason)
        return RiskValidationResult(approved=True)

    def _check_position_size(
        self,
        decision: TradeDecision,
        portfolio_state: dict[str, Any],
        db_connection: Any,
    ) -> RiskValidationResult:
        """Check 5: Single position size must not exceed the cap."""
        max_size = self.params["max_position_size_pct"]

        if decision.size_pct > max_size:
            return RiskValidationResult(
                approved=False,
                reason=(
                    f"Position size {decision.size_pct:.2%} exceeds maximum "
                    f"allowed {max_size:.2%}."
                ),
            )
        return RiskValidationResult(approved=True)

    def _check_total_exposure(
        self,
        decision: TradeDecision,
        portfolio_state: dict[str, Any],
        db_connection: Any,
    ) -> RiskValidationResult:
        """Check 6: Total exposure (existing + proposed) must stay below cap."""
        current_exposure = portfolio_state.get("total_exposure_pct", 0.0)
        max_exposure = self.params["max_total_exposure_pct"]

        new_total = current_exposure + decision.size_pct
        if new_total > max_exposure:
            return RiskValidationResult(
                approved=False,
                reason=(
                    f"Total exposure would be {new_total:.2%} "
                    f"(current: {current_exposure:.2%} + new: {decision.size_pct:.2%}), "
                    f"exceeding limit of {max_exposure:.2%}."
                ),
            )
        return RiskValidationResult(approved=True)

    def _check_leverage(
        self,
        decision: TradeDecision,
        portfolio_state: dict[str, Any],
        db_connection: Any,
    ) -> RiskValidationResult:
        """Check 7: Leverage must not exceed the hard cap."""
        max_lev = self.params["max_leverage"]

        if decision.leverage > max_lev:
            return RiskValidationResult(
                approved=False,
                reason=(
                    f"Leverage {decision.leverage}x exceeds maximum "
                    f"allowed {max_lev}x."
                ),
            )
        return RiskValidationResult(approved=True)

    def _check_stop_loss_exists(
        self,
        decision: TradeDecision,
        portfolio_state: dict[str, Any],
        db_connection: Any,
    ) -> RiskValidationResult:
        """Check 8: A stop-loss price must be defined."""
        if not self.params.get("require_stop_loss", True):
            return RiskValidationResult(approved=True)

        if decision.stop_loss is None or decision.stop_loss == 0:
            return RiskValidationResult(
                approved=False,
                reason="Stop-loss is required but was not provided (None or 0).",
            )
        return RiskValidationResult(approved=True)

    def _check_stop_loss_distance(
        self,
        decision: TradeDecision,
        portfolio_state: dict[str, Any],
        db_connection: Any,
    ) -> RiskValidationResult:
        """Check 9: Stop-loss distance from entry must be within bounds."""
        entry = decision.entry_price
        stop = decision.stop_loss

        # For market orders, entry_price may be None until fill.  We skip
        # the distance check in that case; it will be validated post-fill.
        if entry is None or entry <= 0 or stop is None or stop <= 0:
            return RiskValidationResult(approved=True)

        max_distance = self.params["max_stop_loss_distance_pct"]
        distance_pct = abs(entry - stop) / entry

        if distance_pct > max_distance:
            return RiskValidationResult(
                approved=False,
                reason=(
                    f"Stop-loss distance {distance_pct:.2%} from entry exceeds "
                    f"maximum allowed {max_distance:.2%}. "
                    f"Entry: {entry}, Stop: {stop}."
                ),
            )
        return RiskValidationResult(approved=True)

    def _check_take_profit_exists(
        self,
        decision: TradeDecision,
        portfolio_state: dict[str, Any],
        db_connection: Any,
    ) -> RiskValidationResult:
        """Check 10: A take-profit price must be defined."""
        if not self.params.get("require_take_profit", True):
            return RiskValidationResult(approved=True)

        if decision.take_profit is None or decision.take_profit == 0:
            return RiskValidationResult(
                approved=False,
                reason="Take-profit is required but was not provided (None or 0).",
            )
        return RiskValidationResult(approved=True)

    def _check_risk_reward_ratio(
        self,
        decision: TradeDecision,
        portfolio_state: dict[str, Any],
        db_connection: Any,
    ) -> RiskValidationResult:
        """Check 11: Risk/reward ratio must meet the minimum.

        ALWAYS compute R:R from entry/stop/TP prices when available.
        Only fall back to the AI-supplied value when prices are missing
        (e.g. market orders where entry_price is unknown until fill).
        """
        min_rr = self.params["min_risk_reward_ratio"]

        # Primary: compute from actual prices (the source of truth).
        entry = decision.entry_price
        stop = decision.stop_loss
        tp = decision.take_profit

        if (
            entry is not None and entry > 0
            and stop is not None and stop > 0
            and tp is not None and tp > 0
        ):
            # Validate TP is on the correct side of entry
            is_long = "long" in decision.action
            if is_long and tp <= entry:
                return RiskValidationResult(
                    approved=False,
                    reason=(
                        f"Take-profit ({tp}) is at or below entry ({entry}) "
                        f"for a long trade — TP must be above entry."
                    ),
                )
            if not is_long and tp >= entry:
                return RiskValidationResult(
                    approved=False,
                    reason=(
                        f"Take-profit ({tp}) is at or above entry ({entry}) "
                        f"for a short trade — TP must be below entry."
                    ),
                )

            risk = abs(entry - stop)
            reward = abs(tp - entry)
            if risk == 0:
                return RiskValidationResult(
                    approved=False,
                    reason="Risk is zero (entry == stop); cannot compute R:R.",
                )
            ratio = reward / risk
            if ratio < min_rr:
                return RiskValidationResult(
                    approved=False,
                    reason=(
                        f"Computed risk/reward ratio {ratio:.2f} "
                        f"is below minimum {min_rr:.2f}."
                    ),
                )
            return RiskValidationResult(approved=True)

        # Fallback: use the AI-supplied ratio when prices are unavailable.
        if decision.risk_reward_ratio is not None and decision.risk_reward_ratio > 0:
            if decision.risk_reward_ratio < min_rr:
                return RiskValidationResult(
                    approved=False,
                    reason=(
                        f"Risk/reward ratio {decision.risk_reward_ratio:.2f} "
                        f"is below minimum {min_rr:.2f}."
                    ),
                )
            return RiskValidationResult(approved=True)

        return RiskValidationResult(approved=True)

    def _check_time_between_trades(
        self,
        decision: TradeDecision,
        portfolio_state: dict[str, Any],
        db_connection: Any,
    ) -> RiskValidationResult:
        """Check 12: Enforce a minimum cooldown between consecutive trades."""
        min_minutes = self.params["min_time_between_trades_minutes"]

        try:
            cursor = db_connection.execute(
                """
                SELECT MAX(opened_at) AS last_opened FROM trades
                WHERE asset = ? AND status IN ('open', 'closed')
                """,
                (decision.asset,),
            )
            row = cursor.fetchone()
            if row and row["last_opened"]:
                val = row["last_opened"]
                last_trade_time = val if isinstance(val, datetime) else datetime.fromisoformat(val)
                now = datetime.now(timezone.utc)
                if last_trade_time.tzinfo is None:
                    last_trade_time = last_trade_time.replace(tzinfo=timezone.utc)

                elapsed = now - last_trade_time
                required = timedelta(minutes=min_minutes)

                if elapsed < required:
                    remaining = required - elapsed
                    return RiskValidationResult(
                        approved=False,
                        reason=(
                            f"Minimum time between trades not met for {decision.asset}. "
                            f"Last trade: {elapsed.total_seconds() / 60:.1f} min ago, "
                            f"required: {min_minutes} min. "
                            f"Wait {remaining.total_seconds() / 60:.1f} more minutes."
                        ),
                    )
        except Exception as exc:
            # Fail-CLOSED: if we cannot verify the time gap, reject the trade
            # to prevent rapid-fire trading during DB outages.
            logger.error(
                "Could not query last trade time for {asset}: {err} — failing closed",
                asset=decision.asset,
                err=str(exc),
            )
            return RiskValidationResult(
                approved=False,
                reason=(
                    f"Database error while checking time between trades for "
                    f"{decision.asset}. Rejecting trade to fail safely."
                ),
            )

        return RiskValidationResult(approved=True)

    def _check_daily_trade_count(
        self,
        decision: TradeDecision,
        portfolio_state: dict[str, Any],
        db_connection: Any,
    ) -> RiskValidationResult:
        """Check 13: Reject if the daily trade count cap has been reached."""
        max_trades = self.params["max_trades_per_day"]
        trades_today = self._count_trades_today(db_connection)

        if trades_today >= max_trades:
            return RiskValidationResult(
                approved=False,
                reason=(
                    f"Daily trade limit reached: {trades_today}/{max_trades} "
                    f"trades today. No more trades until the next UTC day."
                ),
            )
        return RiskValidationResult(approved=True)

    def _check_correlation_risk(
        self,
        decision: TradeDecision,
        portfolio_state: dict[str, Any],
        db_connection: Any,
    ) -> RiskValidationResult:
        """Check 14: Reject if new asset is highly correlated with open positions."""
        market_data = getattr(self, "_current_market_data", None)
        open_positions = getattr(self, "_current_open_positions", None)

        if not open_positions:
            return RiskValidationResult(approved=True)

        # Use the threshold from params if configured, otherwise default 0.75
        threshold = self.params.get("correlation_threshold", 0.75)

        is_allowed, reason = CorrelationRiskManager.check_correlation_risk(
            new_asset=decision.asset,
            open_positions=open_positions,
            market_data=market_data or {},
            threshold=threshold,
        )

        if not is_allowed:
            return RiskValidationResult(approved=False, reason=reason)

        return RiskValidationResult(approved=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _count_trades_today(self, db_connection: Any) -> int:
        """Count the number of trades opened since midnight UTC today.

        Args:
            db_connection: Active database connection.

        Returns:
            Integer count of today's trades (0 if the table is missing).
        """
        today_start = (
            datetime.now(timezone.utc)
            .replace(hour=0, minute=0, second=0, microsecond=0)
            .isoformat()
        )

        try:
            cursor = db_connection.execute(
                "SELECT COUNT(*) AS cnt FROM trades WHERE opened_at >= ?",
                (today_start,),
            )
            row = cursor.fetchone()
            return row["cnt"] if row else 0
        except Exception as exc:
            # Fail-CLOSED: return a high count so the daily limit check
            # rejects new trades when the DB is unreachable.
            logger.error(
                "Could not count today's trades: {err} — failing closed (returning 999)",
                err=str(exc),
            )
            return 999
