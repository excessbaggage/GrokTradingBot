"""
Comprehensive tests for the Risk Guardian -- the MOST CRITICAL component.

The Risk Guardian is the final authority on all trading decisions. These tests
verify that every hard-coded risk limit is enforced correctly, that the kill
switch works, and that well-formed trades pass all validation.

All tests use in-memory SQLite databases and dict portfolio states.
No real API calls, no database files on disk, no network.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from brain.models import RiskValidationResult, TradeDecision
from config.risk_config import RISK_PARAMS
from execution.risk_guardian import RiskGuardian


# ======================================================================
# 1. Kill Switch
# ======================================================================


class TestKillSwitch:
    """When the kill switch is active, ALL opening trades must be rejected."""

    def test_kill_switch_blocks_all_trades(
        self,
        kill_switch_guardian: RiskGuardian,
        valid_long_decision: TradeDecision,
        healthy_portfolio: dict[str, Any],
        empty_db: sqlite3.Connection,
    ) -> None:
        """When kill switch is on, ALL open decisions are rejected."""
        result = kill_switch_guardian.validate(
            valid_long_decision, healthy_portfolio, empty_db
        )
        assert result.approved is False
        assert "kill switch" in result.reason.lower() or "Kill switch" in result.reason

    def test_kill_switch_blocks_short(
        self,
        kill_switch_guardian: RiskGuardian,
        valid_short_decision: TradeDecision,
        healthy_portfolio: dict[str, Any],
        empty_db: sqlite3.Connection,
    ) -> None:
        """Kill switch blocks shorts too -- it's ALL trades."""
        result = kill_switch_guardian.validate(
            valid_short_decision, healthy_portfolio, empty_db
        )
        assert result.approved is False

    def test_kill_switch_allows_close(
        self,
        kill_switch_guardian: RiskGuardian,
        close_decision: TradeDecision,
        healthy_portfolio: dict[str, Any],
        empty_db: sqlite3.Connection,
    ) -> None:
        """Close actions should ALWAYS be allowed, even with kill switch on."""
        result = kill_switch_guardian.validate(
            close_decision, healthy_portfolio, empty_db
        )
        assert result.approved is True


# ======================================================================
# 2. Daily Loss Limit
# ======================================================================


class TestDailyLossLimit:
    """When daily loss >= 10%, all opening trades must be rejected."""

    def test_daily_loss_limit_blocks_trades(
        self,
        risk_guardian: RiskGuardian,
        valid_long_decision: TradeDecision,
        daily_loss_portfolio: dict[str, Any],
        empty_db: sqlite3.Connection,
    ) -> None:
        """Daily loss at -10% triggers rejection."""
        result = risk_guardian.validate(
            valid_long_decision, daily_loss_portfolio, empty_db
        )
        assert result.approved is False
        assert "daily" in result.reason.lower() or "Daily" in result.reason

    def test_daily_loss_below_limit_allows_trades(
        self,
        risk_guardian: RiskGuardian,
        valid_long_decision: TradeDecision,
        empty_db: sqlite3.Connection,
    ) -> None:
        """Daily loss at -3% (below 5% limit) should allow trades."""
        portfolio: dict[str, Any] = {
            "equity": 9_700.0,
            "peak_equity": 10_000.0,
            "daily_pnl_pct": -0.03,
            "weekly_pnl_pct": -0.03,
            "total_exposure_pct": 0.0,
        }
        result = risk_guardian.validate(valid_long_decision, portfolio, empty_db)
        # Should not be rejected for daily loss
        if not result.approved:
            assert "daily" not in result.reason.lower()


# ======================================================================
# 3. Weekly Loss Limit
# ======================================================================


class TestWeeklyLossLimit:
    """When weekly loss >= 20%, all opening trades must be rejected."""

    def test_weekly_loss_limit_blocks_trades(
        self,
        risk_guardian: RiskGuardian,
        valid_long_decision: TradeDecision,
        weekly_loss_portfolio: dict[str, Any],
        empty_db: sqlite3.Connection,
    ) -> None:
        """Weekly loss at -20% triggers rejection."""
        result = risk_guardian.validate(
            valid_long_decision, weekly_loss_portfolio, empty_db
        )
        assert result.approved is False
        assert "weekly" in result.reason.lower() or "Weekly" in result.reason


# ======================================================================
# 4. Total Drawdown / Kill Switch Activation
# ======================================================================


class TestTotalDrawdown:
    """When drawdown >= 20%, the kill switch activates automatically."""

    def test_total_drawdown_activates_kill_switch(
        self,
        risk_guardian: RiskGuardian,
        valid_long_decision: TradeDecision,
        drawdown_portfolio: dict[str, Any],
        empty_db: sqlite3.Connection,
    ) -> None:
        """20% drawdown from peak should activate kill switch and reject."""
        assert risk_guardian.kill_switch_active() is False

        result = risk_guardian.validate(
            valid_long_decision, drawdown_portfolio, empty_db
        )

        assert result.approved is False
        assert "drawdown" in result.reason.lower() or "kill switch" in result.reason.lower()
        # Kill switch should now be active
        assert risk_guardian.kill_switch_active() is True

    def test_subsequent_trades_blocked_after_drawdown_kill_switch(
        self,
        risk_guardian: RiskGuardian,
        valid_long_decision: TradeDecision,
        drawdown_portfolio: dict[str, Any],
        healthy_portfolio: dict[str, Any],
        empty_db: sqlite3.Connection,
    ) -> None:
        """Once kill switch activates from drawdown, trades stay blocked."""
        # Trigger kill switch via drawdown
        risk_guardian.validate(valid_long_decision, drawdown_portfolio, empty_db)

        # Even with a "healthy" portfolio, kill switch should stay on
        result = risk_guardian.validate(
            valid_long_decision, healthy_portfolio, empty_db
        )
        assert result.approved is False


# ======================================================================
# 5. Position Size Limit
# ======================================================================


class TestPositionSizeLimit:
    """Reject positions > 15% of portfolio."""

    def test_position_size_limit(
        self,
        risk_guardian: RiskGuardian,
        healthy_portfolio: dict[str, Any],
        empty_db: sqlite3.Connection,
    ) -> None:
        """Position size > 25% should be rejected."""
        oversized = TradeDecision.model_construct(
            action="open_long",
            asset="BTC",
            size_pct=0.30,  # 30% -- over the 25% limit
            leverage=2.0,
            entry_price=65000.0,
            stop_loss=63000.0,
            take_profit=70000.0,
            order_type="limit",
            reasoning="Test oversized position.",
            conviction="high",
            risk_reward_ratio=2.5,
        )

        result = risk_guardian.validate(oversized, healthy_portfolio, empty_db)
        assert result.approved is False
        assert "position size" in result.reason.lower() or "size" in result.reason.lower()

    def test_position_at_exact_limit_passes(
        self,
        risk_guardian: RiskGuardian,
        healthy_portfolio: dict[str, Any],
        empty_db: sqlite3.Connection,
    ) -> None:
        """Position size at exactly 25% should pass (not strictly greater)."""
        at_limit = TradeDecision(
            action="open_long",
            asset="BTC",
            size_pct=0.25,
            leverage=2.0,
            entry_price=65000.0,
            stop_loss=63000.0,
            take_profit=70000.0,
            order_type="limit",
            reasoning="At limit.",
            conviction="high",
            risk_reward_ratio=2.5,
        )

        result = risk_guardian.validate(at_limit, healthy_portfolio, empty_db)
        assert result.approved is True


# ======================================================================
# 6. Total Exposure Limit
# ======================================================================


class TestTotalExposureLimit:
    """Reject if total exposure would exceed 85%."""

    def test_total_exposure_limit(
        self,
        risk_guardian: RiskGuardian,
        high_exposure_portfolio: dict[str, Any],
        empty_db: sqlite3.Connection,
    ) -> None:
        """Adding 10% to existing 80% exposure = 90% > 85% limit."""
        decision = TradeDecision(
            action="open_long",
            asset="ETH",
            size_pct=0.10,
            leverage=2.0,
            entry_price=3500.0,
            stop_loss=3400.0,
            take_profit=3700.0,
            order_type="market",
            reasoning="Test exposure limit.",
            conviction="high",
            risk_reward_ratio=2.0,
        )

        result = risk_guardian.validate(
            decision, high_exposure_portfolio, empty_db
        )
        assert result.approved is False
        assert "exposure" in result.reason.lower()

    def test_exposure_within_limit_passes(
        self,
        risk_guardian: RiskGuardian,
        empty_db: sqlite3.Connection,
    ) -> None:
        """Adding 5% to existing 75% = 80% under limit, should pass."""
        portfolio: dict[str, Any] = {
            "equity": 10_000.0,
            "peak_equity": 10_000.0,
            "daily_pnl_pct": 0.0,
            "weekly_pnl_pct": 0.0,
            "total_exposure_pct": 0.75,
        }
        decision = TradeDecision(
            action="open_long",
            asset="ETH",
            size_pct=0.05,
            leverage=2.0,
            entry_price=3500.0,
            stop_loss=3400.0,
            take_profit=3700.0,
            order_type="market",
            reasoning="Under exposure limit.",
            conviction="high",
            risk_reward_ratio=2.0,
        )

        result = risk_guardian.validate(
            decision, portfolio, empty_db
        )
        assert result.approved is True


# ======================================================================
# 7. Leverage Limit
# ======================================================================


class TestLeverageLimit:
    """Reject leverage > 3x."""

    def test_leverage_limit(
        self,
        risk_guardian: RiskGuardian,
        healthy_portfolio: dict[str, Any],
        empty_db: sqlite3.Connection,
    ) -> None:
        """Leverage of 5x should be rejected (limit is 3x)."""
        # Use model_construct to bypass Pydantic's le=3.0 constraint
        over_leveraged = TradeDecision.model_construct(
            action="open_long",
            asset="BTC",
            size_pct=0.05,
            leverage=5.0,  # Over the 3x limit
            entry_price=65000.0,
            stop_loss=63000.0,
            take_profit=70000.0,
            order_type="limit",
            reasoning="Over-leveraged test.",
            conviction="high",
            risk_reward_ratio=2.5,
        )

        result = risk_guardian.validate(
            over_leveraged, healthy_portfolio, empty_db
        )
        assert result.approved is False
        assert "leverage" in result.reason.lower() or "Leverage" in result.reason

    def test_leverage_at_limit_passes(
        self,
        risk_guardian: RiskGuardian,
        healthy_portfolio: dict[str, Any],
        empty_db: sqlite3.Connection,
    ) -> None:
        """Leverage at exactly 3x should pass."""
        at_limit = TradeDecision(
            action="open_long",
            asset="BTC",
            size_pct=0.05,
            leverage=3.0,
            entry_price=65000.0,
            stop_loss=63000.0,
            take_profit=70000.0,
            order_type="limit",
            reasoning="At leverage limit.",
            conviction="high",
            risk_reward_ratio=2.5,
        )

        result = risk_guardian.validate(at_limit, healthy_portfolio, empty_db)
        assert result.approved is True


# ======================================================================
# 8. Stop-Loss Required
# ======================================================================


class TestStopLossRequired:
    """Every trade MUST have a stop-loss."""

    def test_stop_loss_required(
        self,
        risk_guardian: RiskGuardian,
        healthy_portfolio: dict[str, Any],
        empty_db: sqlite3.Connection,
    ) -> None:
        """A trade with stop_loss=0 should be rejected."""
        no_stop = TradeDecision.model_construct(
            action="open_long",
            asset="BTC",
            size_pct=0.05,
            leverage=2.0,
            entry_price=65000.0,
            stop_loss=0.0,  # Missing / zero stop loss
            take_profit=70000.0,
            order_type="limit",
            reasoning="No stop loss set.",
            conviction="high",
            risk_reward_ratio=2.5,
        )

        result = risk_guardian.validate(no_stop, healthy_portfolio, empty_db)
        assert result.approved is False
        assert "stop" in result.reason.lower()


# ======================================================================
# 9. Stop-Loss Distance
# ======================================================================


class TestStopLossDistance:
    """Stop-loss can't be more than 5% from entry price."""

    def test_stop_loss_distance(
        self,
        risk_guardian: RiskGuardian,
        healthy_portfolio: dict[str, Any],
        empty_db: sqlite3.Connection,
    ) -> None:
        """Stop-loss at ~8% from entry should be rejected (limit is 5%)."""
        wide_stop = TradeDecision.model_construct(
            action="open_long",
            asset="BTC",
            size_pct=0.05,
            leverage=2.0,
            entry_price=65000.0,
            stop_loss=59800.0,  # ~8% below entry
            take_profit=72000.0,
            order_type="limit",
            reasoning="Wide stop loss.",
            conviction="high",
            risk_reward_ratio=2.5,
        )

        result = risk_guardian.validate(wide_stop, healthy_portfolio, empty_db)
        assert result.approved is False
        assert "stop" in result.reason.lower() and "distance" in result.reason.lower()

    def test_stop_loss_within_distance_passes(
        self,
        risk_guardian: RiskGuardian,
        healthy_portfolio: dict[str, Any],
        empty_db: sqlite3.Connection,
    ) -> None:
        """Stop-loss at ~3% from entry should pass."""
        tight_stop = TradeDecision(
            action="open_long",
            asset="BTC",
            size_pct=0.05,
            leverage=2.0,
            entry_price=65000.0,
            stop_loss=63050.0,  # ~3% below entry
            take_profit=70000.0,
            order_type="limit",
            reasoning="Tight stop loss.",
            conviction="high",
            risk_reward_ratio=2.5,
        )

        result = risk_guardian.validate(tight_stop, healthy_portfolio, empty_db)
        assert result.approved is True


# ======================================================================
# 10. Take-Profit Required
# ======================================================================


class TestTakeProfitRequired:
    """Every trade MUST have a take-profit target."""

    def test_take_profit_required(
        self,
        risk_guardian: RiskGuardian,
        healthy_portfolio: dict[str, Any],
        empty_db: sqlite3.Connection,
    ) -> None:
        """A trade with take_profit=0 should be rejected."""
        no_tp = TradeDecision.model_construct(
            action="open_long",
            asset="BTC",
            size_pct=0.05,
            leverage=2.0,
            entry_price=65000.0,
            stop_loss=63000.0,
            take_profit=0.0,  # Missing / zero take profit
            order_type="limit",
            reasoning="No take profit.",
            conviction="high",
            risk_reward_ratio=2.5,
        )

        result = risk_guardian.validate(no_tp, healthy_portfolio, empty_db)
        assert result.approved is False
        assert "take" in result.reason.lower() or "profit" in result.reason.lower()


# ======================================================================
# 11. Risk/Reward Ratio
# ======================================================================


class TestRiskRewardRatio:
    """Risk/reward ratio must be >= 1.2."""

    def test_risk_reward_ratio(
        self,
        risk_guardian: RiskGuardian,
        healthy_portfolio: dict[str, Any],
        empty_db: sqlite3.Connection,
    ) -> None:
        """R:R of 1.0 should be rejected (minimum is 1.2)."""
        bad_rr = TradeDecision(
            action="open_long",
            asset="BTC",
            size_pct=0.05,
            leverage=2.0,
            entry_price=65000.0,
            stop_loss=63000.0,
            take_profit=67000.0,
            order_type="limit",
            reasoning="Bad risk/reward.",
            conviction="medium",
            risk_reward_ratio=1.0,  # Below 1.2 minimum
        )

        result = risk_guardian.validate(bad_rr, healthy_portfolio, empty_db)
        assert result.approved is False
        assert "risk" in result.reason.lower() and "reward" in result.reason.lower()

    def test_risk_reward_at_minimum_passes(
        self,
        risk_guardian: RiskGuardian,
        healthy_portfolio: dict[str, Any],
        empty_db: sqlite3.Connection,
    ) -> None:
        """R:R of exactly 1.2 should pass."""
        ok_rr = TradeDecision(
            action="open_long",
            asset="BTC",
            size_pct=0.05,
            leverage=2.0,
            entry_price=65000.0,
            stop_loss=63000.0,
            take_profit=68000.0,
            order_type="limit",
            reasoning="Acceptable risk/reward.",
            conviction="medium",
            risk_reward_ratio=1.2,
        )

        result = risk_guardian.validate(ok_rr, healthy_portfolio, empty_db)
        assert result.approved is True


# ======================================================================
# 12. Time Between Trades
# ======================================================================


class TestTimeBetweenTrades:
    """Must wait >= 5 minutes between trades."""

    def test_time_between_trades(
        self,
        risk_guardian: RiskGuardian,
        valid_long_decision: TradeDecision,
        healthy_portfolio: dict[str, Any],
        recent_trade_db: sqlite3.Connection,
    ) -> None:
        """Trade only 2 minutes after last trade should be rejected."""
        result = risk_guardian.validate(
            valid_long_decision, healthy_portfolio, recent_trade_db
        )
        assert result.approved is False
        assert "min" in result.reason.lower()

    def test_sufficient_time_passes(
        self,
        risk_guardian: RiskGuardian,
        valid_long_decision: TradeDecision,
        healthy_portfolio: dict[str, Any],
        old_trade_db: sqlite3.Connection,
    ) -> None:
        """Trade 2 hours after last trade should pass the time check."""
        result = risk_guardian.validate(
            valid_long_decision, healthy_portfolio, old_trade_db
        )
        assert result.approved is True


# ======================================================================
# 13. Daily Trade Count
# ======================================================================


class TestDailyTradeCount:
    """Maximum 50 trades per day."""

    def test_daily_trade_count(
        self,
        risk_guardian: RiskGuardian,
        valid_long_decision: TradeDecision,
        healthy_portfolio: dict[str, Any],
        maxed_trades_db: sqlite3.Connection,
    ) -> None:
        """With 50 trades already done today, next trade should be rejected."""
        result = risk_guardian.validate(
            valid_long_decision, healthy_portfolio, maxed_trades_db
        )
        assert result.approved is False
        assert "trade" in result.reason.lower() and "limit" in result.reason.lower()


# ======================================================================
# 14. Valid Trade Passes All Checks
# ======================================================================


class TestValidTradePassesAllChecks:
    """A well-formed trade with a healthy portfolio should pass everything."""

    def test_valid_trade_passes_all_checks(
        self,
        risk_guardian: RiskGuardian,
        valid_long_decision: TradeDecision,
        healthy_portfolio: dict[str, Any],
        empty_db: sqlite3.Connection,
    ) -> None:
        """The canonical valid long decision should pass all checks."""
        result = risk_guardian.validate(
            valid_long_decision, healthy_portfolio, empty_db
        )
        assert result.approved is True

    def test_valid_short_passes_all_checks(
        self,
        risk_guardian: RiskGuardian,
        valid_short_decision: TradeDecision,
        healthy_portfolio: dict[str, Any],
        empty_db: sqlite3.Connection,
    ) -> None:
        """The canonical valid short decision should also pass."""
        result = risk_guardian.validate(
            valid_short_decision, healthy_portfolio, empty_db
        )
        assert result.approved is True


# ======================================================================
# 15. Close Action Always Allowed
# ======================================================================


class TestCloseActionAlwaysAllowed:
    """Close positions should bypass most checks."""

    def test_close_action_always_allowed(
        self,
        risk_guardian: RiskGuardian,
        close_decision: TradeDecision,
        daily_loss_portfolio: dict[str, Any],
        maxed_trades_db: sqlite3.Connection,
    ) -> None:
        """Close should pass even with daily loss limit hit and max trades reached."""
        result = risk_guardian.validate(
            close_decision, daily_loss_portfolio, maxed_trades_db
        )
        assert result.approved is True

    def test_close_with_kill_switch(
        self,
        kill_switch_guardian: RiskGuardian,
        close_decision: TradeDecision,
        drawdown_portfolio: dict[str, Any],
        maxed_trades_db: sqlite3.Connection,
    ) -> None:
        """Close should pass even when kill switch is active."""
        result = kill_switch_guardian.validate(
            close_decision, drawdown_portfolio, maxed_trades_db
        )
        assert result.approved is True

    def test_hold_always_allowed(
        self,
        kill_switch_guardian: RiskGuardian,
        hold_decision: TradeDecision,
        drawdown_portfolio: dict[str, Any],
        maxed_trades_db: sqlite3.Connection,
    ) -> None:
        """Hold (informational) should pass even under adverse conditions."""
        result = kill_switch_guardian.validate(
            hold_decision, drawdown_portfolio, maxed_trades_db
        )
        assert result.approved is True

    def test_no_trade_always_allowed(
        self,
        kill_switch_guardian: RiskGuardian,
        no_trade_decision: TradeDecision,
        drawdown_portfolio: dict[str, Any],
        maxed_trades_db: sqlite3.Connection,
    ) -> None:
        """no_trade (informational) should pass even under adverse conditions."""
        result = kill_switch_guardian.validate(
            no_trade_decision, drawdown_portfolio, maxed_trades_db
        )
        assert result.approved is True
