"""
Comprehensive tests for the enhanced Notifier.

Tests cover all four enhancement tiers:
  1. Rich trade notifications (portfolio context, emoji, formatting, risk score)
  2. Error escalation (severity levels, deduplication, info batching)
  3. Enhanced daily summary (grades, streaks, weekly comparison)
  4. Heartbeat monitoring (bot online/offline, stuck detection)

All tests mock Discord/Telegram HTTP endpoints -- no real network calls.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from execution.notifications import (
    Notifier,
    SEVERITY_CRITICAL,
    SEVERITY_INFO,
    SEVERITY_WARNING,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def silent_notifier() -> Notifier:
    """A Notifier with no channels configured (silent mode)."""
    return Notifier(
        discord_webhook_url="",
        telegram_bot_token="",
        telegram_chat_id="",
    )


@pytest.fixture
def discord_notifier() -> Notifier:
    """A Notifier with only Discord configured."""
    return Notifier(
        discord_webhook_url="https://discord.test/webhook",
        telegram_bot_token="",
        telegram_chat_id="",
    )


@pytest.fixture
def telegram_notifier() -> Notifier:
    """A Notifier with only Telegram configured."""
    return Notifier(
        discord_webhook_url="",
        telegram_bot_token="test_bot_token",
        telegram_chat_id="12345",
    )


@pytest.fixture
def dual_notifier() -> Notifier:
    """A Notifier with both Discord and Telegram configured."""
    return Notifier(
        discord_webhook_url="https://discord.test/webhook",
        telegram_bot_token="test_bot_token",
        telegram_chat_id="12345",
    )


@pytest.fixture
def mock_decision() -> MagicMock:
    """A mock TradeDecision with all expected attributes."""
    d = MagicMock()
    d.action = "open_long"
    d.asset = "BTC"
    d.size_pct = 0.05
    d.leverage = 2.0
    d.entry_price = 65000.0
    d.stop_loss = 63000.0
    d.take_profit = 70000.0
    d.risk_reward_ratio = 2.5
    d.order_type = "limit"
    d.reasoning = "Strong support at 63k with bullish divergence."
    d.conviction = "high"
    d.rsi = None
    d.regime = None
    return d


@pytest.fixture
def mock_order_result() -> dict[str, Any]:
    """A sample order result dict."""
    return {
        "live": False,
        "side": "buy",
        "asset": "BTC",
        "fill_price": 65100.0,
        "order_id": "test-order-123",
        "fees": 3.25,
        "status": "filled",
    }


@pytest.fixture
def sample_portfolio() -> dict[str, Any]:
    """A sample portfolio state dict."""
    return {
        "equity": 10000.0,
        "total_equity": 10000.0,
        "total_exposure_pct": 0.15,
        "open_positions": 2,
    }


@pytest.fixture
def sample_summary() -> dict[str, Any]:
    """A basic daily summary data dict."""
    return {
        "date": "2026-03-08",
        "daily_pnl_pct": 0.025,
        "equity": 10250.0,
        "peak_equity": 10500.0,
        "trades_today": 5,
        "wins": 3,
        "losses": 2,
        "win_rate": 0.6,
        "open_positions": 1,
        "total_exposure_pct": 0.10,
    }


# ======================================================================
# 1. Rich Trade Notifications
# ======================================================================


class TestRichTradeNotifications:
    """Verify enhanced trade alerts contain portfolio context and formatting."""

    @patch("execution.notifications.requests.post")
    def test_trade_alert_contains_emoji(
        self,
        mock_post: MagicMock,
        discord_notifier: Notifier,
        mock_decision: MagicMock,
        mock_order_result: dict,
    ) -> None:
        """Trade alert should include direction emoji."""
        mock_post.return_value = MagicMock(status_code=204)
        discord_notifier.send_trade_alert(mock_decision, mock_order_result)

        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        content = payload["content"]

        # Should contain the chart increasing emoji for open_long
        assert "\U0001F4C8" in content

    @patch("execution.notifications.requests.post")
    def test_trade_alert_short_emoji(
        self,
        mock_post: MagicMock,
        discord_notifier: Notifier,
        mock_decision: MagicMock,
        mock_order_result: dict,
    ) -> None:
        """Short trade should show chart decreasing emoji."""
        mock_post.return_value = MagicMock(status_code=204)
        mock_decision.action = "open_short"
        discord_notifier.send_trade_alert(mock_decision, mock_order_result)

        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        content = payload["content"]
        assert "\U0001F4C9" in content

    @patch("execution.notifications.requests.post")
    def test_trade_alert_close_emoji(
        self,
        mock_post: MagicMock,
        discord_notifier: Notifier,
        mock_decision: MagicMock,
        mock_order_result: dict,
    ) -> None:
        """Close trade should show counterclockwise arrows emoji."""
        mock_post.return_value = MagicMock(status_code=204)
        mock_decision.action = "close"
        discord_notifier.send_trade_alert(mock_decision, mock_order_result)

        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        content = payload["content"]
        assert "\U0001F504" in content

    @patch("execution.notifications.requests.post")
    def test_trade_alert_formatted_prices(
        self,
        mock_post: MagicMock,
        discord_notifier: Notifier,
        mock_decision: MagicMock,
        mock_order_result: dict,
    ) -> None:
        """Prices should be formatted with commas and dollar signs."""
        mock_post.return_value = MagicMock(status_code=204)
        discord_notifier.send_trade_alert(mock_decision, mock_order_result)

        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        content = payload["content"]

        # Fill price should be formatted as $65,100.00
        assert "$65,100.00" in content
        # Stop loss should be formatted
        assert "$63,000.00" in content

    @patch("execution.notifications.requests.post")
    def test_trade_alert_with_portfolio_context(
        self,
        mock_post: MagicMock,
        discord_notifier: Notifier,
        mock_decision: MagicMock,
        mock_order_result: dict,
        sample_portfolio: dict,
    ) -> None:
        """Trade alert with portfolio should include equity and exposure."""
        mock_post.return_value = MagicMock(status_code=204)
        discord_notifier.send_trade_alert(
            mock_decision, mock_order_result, portfolio=sample_portfolio
        )

        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        content = payload["content"]

        assert "$10,000.00" in content  # equity
        assert "15.0%" in content  # exposure
        assert "Positions: 2" in content

    @patch("execution.notifications.requests.post")
    def test_trade_alert_without_portfolio(
        self,
        mock_post: MagicMock,
        discord_notifier: Notifier,
        mock_decision: MagicMock,
        mock_order_result: dict,
    ) -> None:
        """Trade alert without portfolio should still work (backward compat)."""
        mock_post.return_value = MagicMock(status_code=204)
        discord_notifier.send_trade_alert(mock_decision, mock_order_result)

        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        content = payload["content"]

        # Should NOT contain portfolio section
        assert "--- Portfolio ---" not in content
        # But should still contain trade details
        assert "BTC" in content

    @patch("execution.notifications.requests.post")
    def test_trade_alert_with_rsi_and_regime(
        self,
        mock_post: MagicMock,
        discord_notifier: Notifier,
        mock_decision: MagicMock,
        mock_order_result: dict,
    ) -> None:
        """Trade alert should include RSI and regime when available."""
        mock_post.return_value = MagicMock(status_code=204)
        mock_order_result["rsi"] = 35.2
        mock_order_result["regime"] = "trending_bullish"

        discord_notifier.send_trade_alert(mock_decision, mock_order_result)

        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        content = payload["content"]

        assert "35.2" in content
        assert "trending_bullish" in content

    @patch("execution.notifications.requests.post")
    def test_trade_alert_risk_score(
        self,
        mock_post: MagicMock,
        discord_notifier: Notifier,
        mock_decision: MagicMock,
        mock_order_result: dict,
    ) -> None:
        """Trade alert should include a risk score."""
        mock_post.return_value = MagicMock(status_code=204)
        discord_notifier.send_trade_alert(mock_decision, mock_order_result)

        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        content = payload["content"]

        # Risk = 0.05 * (|65100 - 63000| / 65100) * 2 * 100 = ~0.32%
        assert "Risk:" in content
        assert "% of capital" in content


# ======================================================================
# 2. Error Escalation
# ======================================================================


class TestErrorEscalation:
    """Verify severity-based routing, deduplication, and batching."""

    @patch("execution.notifications.requests.post")
    def test_critical_error_sends_to_all_channels(
        self,
        mock_post: MagicMock,
        dual_notifier: Notifier,
    ) -> None:
        """Critical errors should be sent to all configured channels."""
        mock_post.return_value = MagicMock(status_code=204)
        dual_notifier.send_error_alert("Exchange connection lost", severity=SEVERITY_CRITICAL)

        # Should have called both Discord and Telegram
        assert mock_post.call_count == 2

    @patch("execution.notifications.requests.post")
    def test_critical_error_contains_severity_label(
        self,
        mock_post: MagicMock,
        discord_notifier: Notifier,
    ) -> None:
        """Critical error message should contain the CRITICAL label."""
        mock_post.return_value = MagicMock(status_code=204)
        discord_notifier.send_error_alert("SL/TP failure", severity=SEVERITY_CRITICAL)

        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        content = payload["content"]
        assert "CRITICAL" in content

    @patch("execution.notifications.requests.post")
    def test_warning_deduplication(
        self,
        mock_post: MagicMock,
        discord_notifier: Notifier,
    ) -> None:
        """Same warning sent twice within an hour should only broadcast once."""
        mock_post.return_value = MagicMock(status_code=204)

        discord_notifier.send_error_alert("API rate limited", severity=SEVERITY_WARNING)
        discord_notifier.send_error_alert("API rate limited", severity=SEVERITY_WARNING)

        # Should only be sent once (deduplicated)
        assert mock_post.call_count == 1

    @patch("execution.notifications.requests.post")
    def test_different_warnings_not_deduplicated(
        self,
        mock_post: MagicMock,
        discord_notifier: Notifier,
    ) -> None:
        """Different warnings should both be sent."""
        mock_post.return_value = MagicMock(status_code=204)

        discord_notifier.send_error_alert("API rate limited", severity=SEVERITY_WARNING)
        discord_notifier.send_error_alert("Data parsing error", severity=SEVERITY_WARNING)

        assert mock_post.call_count == 2

    @patch("execution.notifications.requests.post")
    def test_warning_resends_after_window(
        self,
        mock_post: MagicMock,
        discord_notifier: Notifier,
    ) -> None:
        """Same warning should resend after dedup window expires."""
        mock_post.return_value = MagicMock(status_code=204)

        discord_notifier.send_error_alert("API rate limited", severity=SEVERITY_WARNING)
        assert mock_post.call_count == 1

        # Simulate window expiry by modifying the tracking dict
        for key in discord_notifier._warning_sent:
            discord_notifier._warning_sent[key] -= 3700  # Move back past 1hr window

        discord_notifier.send_error_alert("API rate limited", severity=SEVERITY_WARNING)
        assert mock_post.call_count == 2

    @patch("execution.notifications.requests.post")
    def test_info_errors_batched_not_sent(
        self,
        mock_post: MagicMock,
        discord_notifier: Notifier,
    ) -> None:
        """Info-level errors should NOT be broadcast immediately."""
        mock_post.return_value = MagicMock(status_code=204)

        discord_notifier.send_error_alert("Minor cache miss", severity=SEVERITY_INFO)
        discord_notifier.send_error_alert("Stale data refreshed", severity=SEVERITY_INFO)

        # No HTTP calls should be made
        assert mock_post.call_count == 0

    def test_info_errors_stored_for_summary(
        self,
        silent_notifier: Notifier,
    ) -> None:
        """Info-level errors should be stored and accessible."""
        silent_notifier.send_error_alert("Minor cache miss", severity=SEVERITY_INFO)
        silent_notifier.send_error_alert("Stale data refreshed", severity=SEVERITY_INFO)

        pending = silent_notifier.get_pending_info_errors()
        assert len(pending) == 2
        assert pending[0]["message"] == "Minor cache miss"
        assert pending[1]["message"] == "Stale data refreshed"

    @patch("execution.notifications.requests.post")
    def test_info_errors_included_in_daily_summary(
        self,
        mock_post: MagicMock,
        discord_notifier: Notifier,
        sample_summary: dict,
    ) -> None:
        """Info errors should appear in the daily summary then be cleared."""
        mock_post.return_value = MagicMock(status_code=204)

        # Batch some info errors
        discord_notifier.send_error_alert("Cache miss 1", severity=SEVERITY_INFO)
        discord_notifier.send_error_alert("Cache miss 2", severity=SEVERITY_INFO)

        # Send daily summary
        discord_notifier.send_daily_summary(sample_summary)

        # The summary message should contain the info notices
        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        content = payload["content"]
        assert "Info Notices" in content
        assert "Cache miss 1" in content

        # Info errors should be cleared after summary
        assert len(discord_notifier.get_pending_info_errors()) == 0

    @patch("execution.notifications.requests.post")
    def test_default_severity_is_warning(
        self,
        mock_post: MagicMock,
        discord_notifier: Notifier,
    ) -> None:
        """Calling send_error_alert without severity should default to warning."""
        mock_post.return_value = MagicMock(status_code=204)
        discord_notifier.send_error_alert("Some error")

        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        content = payload["content"]
        assert "WARNING" in content

    @patch("execution.notifications.requests.post")
    def test_backward_compat_error_alert(
        self,
        mock_post: MagicMock,
        discord_notifier: Notifier,
    ) -> None:
        """send_error_alert with just message should work (backward compat)."""
        mock_post.return_value = MagicMock(status_code=204)
        # Old-style call: just a message string
        discord_notifier.send_error_alert("Something went wrong")
        assert mock_post.call_count == 1


# ======================================================================
# 3. Enhanced Daily Summary
# ======================================================================


class TestEnhancedDailySummary:
    """Verify the daily summary includes grades, streaks, and weekly comparison."""

    @patch("execution.notifications.requests.post")
    def test_summary_contains_grade(
        self,
        mock_post: MagicMock,
        discord_notifier: Notifier,
        sample_summary: dict,
    ) -> None:
        """Daily summary should include a performance grade."""
        mock_post.return_value = MagicMock(status_code=204)
        discord_notifier.send_daily_summary(sample_summary)

        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        content = payload["content"]
        assert "Grade:" in content

    @patch("execution.notifications.requests.post")
    def test_summary_grade_a(
        self,
        mock_post: MagicMock,
        discord_notifier: Notifier,
    ) -> None:
        """Win rate 70% with <5% drawdown should get grade A."""
        mock_post.return_value = MagicMock(status_code=204)
        summary = {
            "date": "2026-03-08",
            "daily_pnl_pct": 0.03,
            "equity": 10300.0,
            "peak_equity": 10500.0,
            "trades_today": 10,
            "wins": 7,
            "losses": 3,
            "win_rate": 0.7,
            "open_positions": 1,
            "total_exposure_pct": 0.10,
        }
        discord_notifier.send_daily_summary(summary)

        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        content = payload["content"]
        assert "Grade: A" in content

    @patch("execution.notifications.requests.post")
    def test_summary_grade_na_no_trades(
        self,
        mock_post: MagicMock,
        discord_notifier: Notifier,
    ) -> None:
        """No trades today should result in grade N/A."""
        mock_post.return_value = MagicMock(status_code=204)
        summary = {
            "date": "2026-03-08",
            "daily_pnl_pct": 0.0,
            "equity": 10000.0,
            "peak_equity": 10000.0,
            "trades_today": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "open_positions": 0,
            "total_exposure_pct": 0.0,
        }
        discord_notifier.send_daily_summary(summary)

        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        content = payload["content"]
        assert "Grade: N/A" in content

    @patch("execution.notifications.requests.post")
    def test_summary_with_weekly_change(
        self,
        mock_post: MagicMock,
        discord_notifier: Notifier,
        sample_summary: dict,
    ) -> None:
        """Summary should include weekly equity change when provided."""
        mock_post.return_value = MagicMock(status_code=204)
        sample_summary["weekly_equity_change_pct"] = 0.05
        discord_notifier.send_daily_summary(sample_summary)

        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        content = payload["content"]
        assert "Weekly Change:" in content
        assert "+5.00%" in content

    @patch("execution.notifications.requests.post")
    def test_summary_with_streak(
        self,
        mock_post: MagicMock,
        discord_notifier: Notifier,
        sample_summary: dict,
    ) -> None:
        """Summary should show current streak when provided."""
        mock_post.return_value = MagicMock(status_code=204)
        sample_summary["current_streak"] = 3
        discord_notifier.send_daily_summary(sample_summary)

        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        content = payload["content"]
        assert "Streak:" in content
        assert "3W" in content

    @patch("execution.notifications.requests.post")
    def test_summary_with_losing_streak(
        self,
        mock_post: MagicMock,
        discord_notifier: Notifier,
        sample_summary: dict,
    ) -> None:
        """Negative streak should display as losses."""
        mock_post.return_value = MagicMock(status_code=204)
        sample_summary["current_streak"] = -2
        discord_notifier.send_daily_summary(sample_summary)

        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        content = payload["content"]
        assert "2L" in content

    @patch("execution.notifications.requests.post")
    def test_summary_with_best_worst_trade(
        self,
        mock_post: MagicMock,
        discord_notifier: Notifier,
        sample_summary: dict,
    ) -> None:
        """Summary should show best and worst trade P&L."""
        mock_post.return_value = MagicMock(status_code=204)
        sample_summary["best_trade_pnl"] = 0.035
        sample_summary["worst_trade_pnl"] = -0.015
        discord_notifier.send_daily_summary(sample_summary)

        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        content = payload["content"]
        assert "Best Trade:" in content
        assert "+3.50%" in content
        assert "Worst Trade:" in content
        assert "-1.50%" in content

    @patch("execution.notifications.requests.post")
    def test_summary_with_avg_rr(
        self,
        mock_post: MagicMock,
        discord_notifier: Notifier,
        sample_summary: dict,
    ) -> None:
        """Summary should include average R:R when provided."""
        mock_post.return_value = MagicMock(status_code=204)
        sample_summary["avg_rr"] = 2.35
        discord_notifier.send_daily_summary(sample_summary)

        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        content = payload["content"]
        assert "Avg R:R:" in content
        assert "2.35" in content

    @patch("execution.notifications.requests.post")
    def test_summary_formatted_equity(
        self,
        mock_post: MagicMock,
        discord_notifier: Notifier,
        sample_summary: dict,
    ) -> None:
        """Equity in summary should be formatted with dollar sign and commas."""
        mock_post.return_value = MagicMock(status_code=204)
        discord_notifier.send_daily_summary(sample_summary)

        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        content = payload["content"]
        assert "$10,250.00" in content


# ======================================================================
# 4. Heartbeat Monitoring
# ======================================================================


class TestHeartbeatMonitoring:
    """Verify heartbeat tracking, bot online/offline notifications."""

    @patch("execution.notifications.requests.post")
    def test_bot_online_notification(
        self,
        mock_post: MagicMock,
        discord_notifier: Notifier,
    ) -> None:
        """Bot online should send startup config details."""
        mock_post.return_value = MagicMock(status_code=204)
        discord_notifier.send_bot_online(
            mode="PAPER",
            assets=["BTC", "ETH", "SOL"],
            cycle_interval=15,
        )

        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        content = payload["content"]
        assert "BOT ONLINE" in content
        assert "PAPER" in content
        assert "BTC" in content
        assert "15 min" in content

    @patch("execution.notifications.requests.post")
    def test_bot_offline_notification(
        self,
        mock_post: MagicMock,
        discord_notifier: Notifier,
    ) -> None:
        """Bot offline should send shutdown message."""
        mock_post.return_value = MagicMock(status_code=204)
        discord_notifier.send_bot_offline()

        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        content = payload["content"]
        assert "BOT OFFLINE" in content

    def test_record_heartbeat_updates_timestamp(
        self,
        silent_notifier: Notifier,
    ) -> None:
        """record_heartbeat should update the last heartbeat time."""
        old_hb = silent_notifier._last_heartbeat
        time.sleep(0.01)  # Small delay to ensure different timestamp
        silent_notifier.record_heartbeat()
        assert silent_notifier._last_heartbeat > old_hb

    @patch("execution.notifications.requests.post")
    def test_heartbeat_no_alert_when_healthy(
        self,
        mock_post: MagicMock,
        discord_notifier: Notifier,
    ) -> None:
        """No alert should fire if heartbeat is recent enough."""
        mock_post.return_value = MagicMock(status_code=204)
        discord_notifier.record_heartbeat()
        discord_notifier.check_heartbeat(expected_interval_minutes=15)

        # Should not have sent any alert
        assert mock_post.call_count == 0

    @patch("execution.notifications.requests.post")
    def test_heartbeat_alert_when_stuck(
        self,
        mock_post: MagicMock,
        discord_notifier: Notifier,
    ) -> None:
        """Alert should fire if no heartbeat for >2x expected interval."""
        mock_post.return_value = MagicMock(status_code=204)

        # Simulate stuck bot: set heartbeat far in the past
        discord_notifier._last_heartbeat = time.monotonic() - 2000  # ~33 min ago

        discord_notifier.check_heartbeat(expected_interval_minutes=15)

        # Should have sent a critical alert
        assert mock_post.call_count == 1
        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        content = payload["content"]
        assert "stuck" in content.lower() or "CRITICAL" in content


# ======================================================================
# 5. Formatting Helpers
# ======================================================================


class TestFormattingHelpers:
    """Verify number formatting and helper methods."""

    def test_fmt_price_large(self) -> None:
        """Large prices should get commas and 2 decimal places."""
        assert Notifier._fmt_price(65100.0) == "$65,100.00"

    def test_fmt_price_small(self) -> None:
        """Small prices (< $1) should get 4 decimal places."""
        assert Notifier._fmt_price(0.1234) == "$0.1234"

    def test_fmt_price_very_small(self) -> None:
        """Very small prices (< $0.01) should get 6 decimal places."""
        assert Notifier._fmt_price(0.000012) == "$0.000012"

    def test_fmt_price_none(self) -> None:
        """None should return N/A."""
        assert Notifier._fmt_price(None) == "N/A"

    def test_fmt_dollar(self) -> None:
        """Dollar amounts should be formatted with commas."""
        assert Notifier._fmt_dollar(12345.67) == "$12,345.67"

    def test_fmt_dollar_none(self) -> None:
        """None dollar amounts should return N/A."""
        assert Notifier._fmt_dollar(None) == "N/A"

    def test_direction_emoji_long(self) -> None:
        """open_long action should return chart increasing emoji."""
        assert Notifier._get_direction_emoji("open_long") == "\U0001F4C8"

    def test_direction_emoji_short(self) -> None:
        """open_short action should return chart decreasing emoji."""
        assert Notifier._get_direction_emoji("open_short") == "\U0001F4C9"

    def test_direction_emoji_close(self) -> None:
        """close action should return counterclockwise arrows emoji."""
        assert Notifier._get_direction_emoji("close") == "\U0001F504"

    def test_direction_emoji_unknown(self) -> None:
        """Unknown actions should return the bar chart fallback."""
        assert Notifier._get_direction_emoji("something") == "\U0001F4CA"

    def test_compute_risk_score(self) -> None:
        """Risk score should be calculated correctly."""
        decision = MagicMock()
        decision.size_pct = 0.05
        decision.leverage = 2.0
        decision.stop_loss = 63000.0
        decision.entry_price = 65000.0

        result = Notifier._compute_risk_score(decision, 65000.0)
        # Risk = 0.05 * (|65000-63000|/65000) * 2 * 100 = 0.05 * 0.03077 * 2 * 100 = 0.31%
        assert "% of capital" in result
        # Parse the number
        risk_val = float(result.split("%")[0])
        assert 0.2 < risk_val < 0.4

    def test_compute_risk_score_no_entry(self) -> None:
        """Risk score with missing entry should return N/A."""
        decision = MagicMock()
        decision.size_pct = 0.05
        decision.leverage = 2.0
        decision.stop_loss = 63000.0
        decision.entry_price = 0.0

        assert Notifier._compute_risk_score(decision, None) == "N/A"


# ======================================================================
# 6. Performance Grade
# ======================================================================


class TestPerformanceGrade:
    """Verify letter grade computation."""

    def test_grade_a(self) -> None:
        """60%+ win rate with <5% drawdown should be A."""
        assert Notifier._compute_performance_grade(0.65, 3.0, 10) == "A"

    def test_grade_b(self) -> None:
        """50%+ win rate with <10% drawdown should be B."""
        assert Notifier._compute_performance_grade(0.55, 7.0, 10) == "B"

    def test_grade_c(self) -> None:
        """40%+ win rate with <15% drawdown should be C."""
        assert Notifier._compute_performance_grade(0.45, 12.0, 10) == "C"

    def test_grade_d(self) -> None:
        """30%+ win rate or <20% drawdown should be D."""
        assert Notifier._compute_performance_grade(0.35, 18.0, 10) == "D"

    def test_grade_f(self) -> None:
        """<30% win rate with >=20% drawdown should be F."""
        assert Notifier._compute_performance_grade(0.20, 25.0, 10) == "F"

    def test_grade_na_no_trades(self) -> None:
        """No trades should return N/A."""
        assert Notifier._compute_performance_grade(0.0, 0.0, 0) == "N/A"


# ======================================================================
# 7. Hold Duration (existing, kept for regression)
# ======================================================================


class TestHoldDuration:
    """Verify hold duration computation still works correctly."""

    def test_normal_duration(self) -> None:
        """Standard duration should be calculated correctly."""
        result = Notifier._compute_hold_duration(
            "2026-03-07T10:00:00+00:00",
            "2026-03-07T12:35:00+00:00",
        )
        assert result == "2h 35m"

    def test_duration_with_days(self) -> None:
        """Multi-day durations should include days."""
        result = Notifier._compute_hold_duration(
            "2026-03-05T10:00:00+00:00",
            "2026-03-07T12:35:00+00:00",
        )
        assert "2d" in result

    def test_duration_na_empty(self) -> None:
        """Empty timestamps should return N/A."""
        assert Notifier._compute_hold_duration("", "") == "N/A"

    def test_duration_na_invalid(self) -> None:
        """Invalid timestamps should return N/A."""
        assert Notifier._compute_hold_duration("bad", "data") == "N/A"


# ======================================================================
# 8. Broadcasting
# ======================================================================


class TestBroadcasting:
    """Verify channel routing logic."""

    @patch("execution.notifications.requests.post")
    def test_broadcast_discord_only(
        self,
        mock_post: MagicMock,
        discord_notifier: Notifier,
    ) -> None:
        """With only Discord configured, only Discord should be called."""
        mock_post.return_value = MagicMock(status_code=204)
        discord_notifier._broadcast("test message")
        assert mock_post.call_count == 1
        url_called = mock_post.call_args[0][0] if mock_post.call_args[0] else mock_post.call_args.kwargs.get("url", "")
        # For positional args, the URL is the first positional argument
        assert "discord" in str(mock_post.call_args).lower()

    @patch("execution.notifications.requests.post")
    def test_broadcast_telegram_only(
        self,
        mock_post: MagicMock,
        telegram_notifier: Notifier,
    ) -> None:
        """With only Telegram configured, only Telegram should be called."""
        mock_post.return_value = MagicMock(status_code=200)
        telegram_notifier._broadcast("test message")
        assert mock_post.call_count == 1
        assert "telegram" in str(mock_post.call_args).lower()

    @patch("execution.notifications.requests.post")
    def test_broadcast_both_channels(
        self,
        mock_post: MagicMock,
        dual_notifier: Notifier,
    ) -> None:
        """With both channels, both should be called."""
        mock_post.return_value = MagicMock(status_code=204)
        dual_notifier._broadcast("test message")
        assert mock_post.call_count == 2

    @patch("execution.notifications.requests.post")
    def test_broadcast_silent_mode(
        self,
        mock_post: MagicMock,
        silent_notifier: Notifier,
    ) -> None:
        """With no channels, nothing should be called."""
        silent_notifier._broadcast("test message")
        assert mock_post.call_count == 0

    @patch("execution.notifications.requests.post")
    def test_broadcast_all_critical(
        self,
        mock_post: MagicMock,
        dual_notifier: Notifier,
    ) -> None:
        """_broadcast_all should try all configured channels."""
        mock_post.return_value = MagicMock(status_code=204)
        dual_notifier._broadcast_all("critical message")
        assert mock_post.call_count == 2


# ======================================================================
# 9. Constructor Backward Compatibility
# ======================================================================


class TestBackwardCompatibility:
    """Ensure constructor signature is unchanged."""

    def test_default_constructor(self) -> None:
        """Notifier() with no args should not raise."""
        n = Notifier()
        assert n.discord_url == ""

    def test_keyword_constructor(self) -> None:
        """Notifier with keyword args should work."""
        n = Notifier(
            discord_webhook_url="https://test.discord/hook",
            telegram_bot_token="tok",
            telegram_chat_id="chat",
        )
        assert n.discord_url == "https://test.discord/hook"
        assert n.telegram_token == "tok"
        assert n.telegram_chat_id == "chat"

    def test_send_trade_alert_backward_compat(self) -> None:
        """send_trade_alert without portfolio arg should work."""
        n = Notifier()
        decision = MagicMock()
        decision.action = "open_long"
        decision.asset = "ETH"
        decision.size_pct = 0.03
        decision.leverage = 1.0
        decision.stop_loss = 3400.0
        decision.take_profit = 3700.0
        decision.risk_reward_ratio = 2.0
        decision.order_type = "market"
        decision.reasoning = "Test"
        decision.rsi = None
        decision.regime = None

        order_result = {
            "live": False,
            "side": "buy",
            "fill_price": 3500.0,
            "order_id": "test",
            "fees": 1.0,
        }

        # Should not raise even without portfolio
        n.send_trade_alert(decision, order_result)

    def test_send_error_alert_backward_compat(self) -> None:
        """send_error_alert with just message should work (no severity)."""
        n = Notifier()
        # Old-style: just a message
        n.send_error_alert("Something went wrong")
