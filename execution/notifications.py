"""
Notifications -- send alerts via Discord webhooks and Telegram bots.

All notification methods are non-blocking and fail-silent: if no webhook
URL or bot token is configured, the call is a no-op.  Network errors are
logged but never propagate up to the caller.

Supported channels:
    - Discord (via webhook POST)
    - Telegram (via Bot API sendMessage)

Message formatting uses plain-text with simple separators for
cross-platform readability (Discord markdown, Telegram MarkdownV2 can
be fragile with special characters).

Enhanced features:
    - Rich trade alerts with portfolio context, emoji indicators, RSI/regime
    - Error severity levels (critical, warning, info) with deduplication
    - Enhanced daily summaries with performance grades and streaks
    - Heartbeat monitoring with bot online/offline notifications
"""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import requests
from loguru import logger

from config.trading_config import (
    DISCORD_WEBHOOK_URL,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
)


# Error severity constants
SEVERITY_CRITICAL = "critical"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"


class Notifier:
    """Sends trading alerts and summaries to Discord and/or Telegram.

    Both channels are optional.  If neither is configured the Notifier
    still works -- it just logs messages locally and returns.

    Attributes:
        discord_url: Discord webhook URL (empty string = disabled).
        telegram_token: Telegram Bot API token (empty string = disabled).
        telegram_chat_id: Telegram chat/group ID to send to.
    """

    # Discord webhook rate-limit safety
    _DISCORD_TIMEOUT_SECONDS: int = 10
    _TELEGRAM_TIMEOUT_SECONDS: int = 10

    # Warning deduplication window (seconds)
    _WARNING_DEDUP_WINDOW: int = 3600  # 1 hour

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        discord_webhook_url: str | None = None,
        telegram_bot_token: str | None = None,
        telegram_chat_id: str | None = None,
    ) -> None:
        """Initialise the Notifier.

        Falls back to values from ``config.trading_config`` when arguments
        are not provided.

        Args:
            discord_webhook_url: Discord webhook URL.
            telegram_bot_token: Telegram Bot API token.
            telegram_chat_id: Telegram chat/group ID.
        """
        self.discord_url: str = discord_webhook_url or DISCORD_WEBHOOK_URL or ""
        self.telegram_token: str = telegram_bot_token or TELEGRAM_BOT_TOKEN or ""
        self.telegram_chat_id: str = telegram_chat_id or TELEGRAM_CHAT_ID or ""

        # Warning deduplication: maps error type -> last sent timestamp
        self._warning_sent: dict[str, float] = {}

        # Info error batch: collects info-level errors for daily summary
        self._info_errors: list[dict[str, Any]] = []

        # Heartbeat tracking
        self._last_heartbeat: float = time.monotonic()

        channels: list[str] = []
        if self.discord_url:
            channels.append("Discord")
        if self.telegram_token and self.telegram_chat_id:
            channels.append("Telegram")

        if channels:
            logger.info(
                "Notifier initialised | channels={ch}", ch=", ".join(channels),
            )
        else:
            logger.info("Notifier initialised | no channels configured (silent mode)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_trade_alert(
        self,
        decision: Any,
        order_result: dict[str, Any],
        portfolio: dict[str, Any] | None = None,
    ) -> None:
        """Send a rich alert when a new trade is opened.

        Args:
            decision: The ``TradeDecision`` that was executed.
            order_result: The order result dict from ``OrderManager.place_order``.
            portfolio: Optional portfolio state dict for enriched context.
                Expected keys: ``equity``, ``total_exposure_pct``,
                ``open_positions``.
        """
        mode = "LIVE" if order_result.get("live", False) else "PAPER"
        action = getattr(decision, "action", order_result.get("side", "unknown"))
        asset = getattr(decision, "asset", order_result.get("asset", "?"))

        # Emoji indicators for direction
        direction_emoji = self._get_direction_emoji(action)

        # Format prices cleanly
        fill_price = order_result.get("fill_price")
        fill_str = self._fmt_price(fill_price) if fill_price is not None else "N/A"
        sl_val = getattr(decision, "stop_loss", None)
        sl_str = self._fmt_price(sl_val) if sl_val else "N/A"
        tp_val = getattr(decision, "take_profit", None)
        tp_str = self._fmt_price(tp_val) if tp_val else "N/A"
        fees_val = order_result.get("fees", 0.0)
        fees_str = self._fmt_price(fees_val)

        # Risk score: how much capital is at risk in this trade
        risk_score = self._compute_risk_score(decision, fill_price)

        lines = [
            f"{direction_emoji} NEW TRADE ({mode}) {direction_emoji}",
            f"Asset:     {asset}",
            f"Action:    {action.upper()}",
            f"Size:      {getattr(decision, 'size_pct', 0) * 100:.1f}% of portfolio",
            f"Leverage:  {getattr(decision, 'leverage', 1)}x",
            f"Entry:     {fill_str}",
            f"Stop Loss: {sl_str}",
            f"Target:    {tp_str}",
            f"R:R Ratio: {getattr(decision, 'risk_reward_ratio', 'N/A')}",
            f"Type:      {getattr(decision, 'order_type', 'N/A')}",
            f"Order ID:  {order_result.get('order_id', 'N/A')}",
            f"Fees:      {fees_str}",
            f"Risk:      {risk_score}",
        ]

        # RSI and regime from decision/order_result context
        rsi = order_result.get("rsi") or getattr(decision, "rsi", None)
        regime = order_result.get("regime") or getattr(decision, "regime", None)
        if rsi is not None:
            lines.append(f"RSI:       {rsi:.1f}")
        if regime is not None:
            lines.append(f"Regime:    {regime}")

        # Portfolio context
        if portfolio is not None:
            equity = portfolio.get("equity", portfolio.get("total_equity", 0.0))
            exposure_pct = portfolio.get("total_exposure_pct", 0.0)
            open_pos = portfolio.get("open_positions", 0)
            # If open_positions is a list, count it
            if isinstance(open_pos, list):
                open_pos = len(open_pos)
            lines.append("")
            lines.append("--- Portfolio ---")
            lines.append(f"Equity:    {self._fmt_dollar(equity)}")
            lines.append(f"Exposure:  {exposure_pct * 100:.1f}%")
            lines.append(f"Positions: {open_pos}")

        lines.append("")
        lines.append(f"Reasoning: {getattr(decision, 'reasoning', 'N/A')}")

        message = "\n".join(lines)
        self._broadcast(message)

    def send_trade_closed(self, trade_data: dict[str, Any]) -> None:
        """Send an alert when a trade is closed.

        Args:
            trade_data: Dict with trade closure details.  Expected keys:
                ``asset``, ``side``, ``entry_price``, ``exit_price``,
                ``realized_pnl_pct``, ``close_reason``, ``opened_at``,
                ``closed_at``.
        """
        asset = trade_data.get("asset", "?")
        pnl = trade_data.get("realized_pnl_pct", 0.0)
        pnl_emoji = "WIN" if pnl >= 0 else "LOSS"

        # Calculate hold duration
        opened = trade_data.get("opened_at", "")
        closed = trade_data.get("closed_at", "")
        hold_duration = self._compute_hold_duration(opened, closed)

        lines = [
            f"--- TRADE CLOSED ({pnl_emoji}) ---",
            f"Asset:      {asset}",
            f"Side:       {trade_data.get('side', 'N/A')}",
            f"Entry:      {trade_data.get('entry_price', 'N/A')}",
            f"Exit:       {trade_data.get('exit_price', 'N/A')}",
            f"P&L:        {pnl * 100:+.2f}%",
            f"Reason:     {trade_data.get('close_reason', 'N/A')}",
            f"Duration:   {hold_duration}",
        ]
        message = "\n".join(lines)
        self._broadcast(message)

    def send_daily_summary(self, summary_data: dict[str, Any]) -> None:
        """Send an enhanced daily performance summary.

        Args:
            summary_data: Dict with daily summary data.  Expected keys:
                ``date``, ``daily_pnl_pct``, ``equity``, ``peak_equity``,
                ``trades_today``, ``wins``, ``losses``,
                ``win_rate``, ``open_positions``, ``total_exposure_pct``.

                Enhanced optional keys:
                ``avg_rr``, ``best_trade_pnl``, ``worst_trade_pnl``,
                ``current_streak``, ``weekly_equity_change_pct``.
        """
        pnl = summary_data.get("daily_pnl_pct", 0.0)
        equity = summary_data.get("equity", 0.0)
        peak = summary_data.get("peak_equity", equity)
        drawdown = ((peak - equity) / peak * 100) if peak > 0 else 0.0

        wins = summary_data.get("wins", 0)
        losses = summary_data.get("losses", 0)
        trades_today = summary_data.get("trades_today", 0)
        win_rate = summary_data.get("win_rate", 0)

        # Performance grade
        grade = self._compute_performance_grade(win_rate, drawdown, trades_today)

        lines = [
            "=============================",
            f"   DAILY REPORT  |  Grade: {grade}",
            "=============================",
            f"Date:           {summary_data.get('date', 'N/A')}",
            "",
            "--- Equity ---",
            f"Equity:         {self._fmt_dollar(equity)}",
            f"Peak Equity:    {self._fmt_dollar(peak)}",
            f"Daily P&L:      {pnl * 100:+.2f}%",
            f"Drawdown:       {drawdown:.2f}%",
        ]

        # Weekly equity change comparison
        weekly_change = summary_data.get("weekly_equity_change_pct")
        if weekly_change is not None:
            lines.append(f"Weekly Change:  {weekly_change * 100:+.2f}%")

        lines.append("")
        lines.append("--- Trades ---")
        lines.append(f"Trades Today:   {trades_today}")
        lines.append(f"Wins:           {wins}")
        lines.append(f"Losses:         {losses}")
        lines.append(f"Win Rate:       {win_rate * 100:.1f}%")

        # Average R:R
        avg_rr = summary_data.get("avg_rr")
        if avg_rr is not None:
            lines.append(f"Avg R:R:        {avg_rr:.2f}")

        # Best / Worst trade
        best_trade = summary_data.get("best_trade_pnl")
        worst_trade = summary_data.get("worst_trade_pnl")
        if best_trade is not None:
            lines.append(f"Best Trade:     {best_trade * 100:+.2f}%")
        if worst_trade is not None:
            lines.append(f"Worst Trade:    {worst_trade * 100:+.2f}%")

        # Current streak
        current_streak = summary_data.get("current_streak")
        if current_streak is not None:
            if current_streak > 0:
                streak_str = f"{current_streak}W"
            elif current_streak < 0:
                streak_str = f"{abs(current_streak)}L"
            else:
                streak_str = "None"
            lines.append(f"Streak:         {streak_str}")

        lines.append("")
        lines.append("--- Positions ---")
        lines.append(f"Open Positions: {summary_data.get('open_positions', 0)}")
        lines.append(f"Exposure:       {summary_data.get('total_exposure_pct', 0) * 100:.1f}%")

        # Append batched info errors if any
        if self._info_errors:
            lines.append("")
            lines.append(f"--- Info Notices ({len(self._info_errors)}) ---")
            for err in self._info_errors[-10:]:  # Last 10
                lines.append(f"  {err.get('time', '?')}: {err.get('message', '?')}")
            if len(self._info_errors) > 10:
                lines.append(f"  ... and {len(self._info_errors) - 10} more")
            self._info_errors.clear()

        lines.append("=============================")

        message = "\n".join(lines)
        self._broadcast(message)

    def send_risk_alert(
        self,
        alert_type: str,
        details: str,
    ) -> None:
        """Send a risk management alert.

        Args:
            alert_type: Category of alert (e.g. ``"kill_switch"``,
                        ``"daily_limit"``, ``"drawdown_warning"``).
            details: Human-readable description of the risk event.
        """
        lines = [
            "!!! RISK ALERT !!!",
            f"Type:    {alert_type.upper()}",
            f"Details: {details}",
            f"Time:    {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        ]
        message = "\n".join(lines)
        self._broadcast(message)

    def send_error_alert(
        self,
        error_message: str,
        severity: str = SEVERITY_WARNING,
    ) -> None:
        """Send an error notification with severity-based routing.

        Severity levels:
            - ``"critical"``: Sent to ALL channels immediately regardless of
              channel configuration. Used for SL/TP failure, exchange
              connection loss, equity corruption.
            - ``"warning"``: Sent normally but deduplicated -- the same error
              type is sent at most once per hour.
            - ``"info"``: Batched and included in the next daily summary.

        Args:
            error_message: Description of the error.
            severity: One of ``"critical"``, ``"warning"``, ``"info"``.
        """
        now_ts = time.monotonic()
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        if severity == SEVERITY_INFO:
            # Batch for daily summary
            self._info_errors.append({
                "message": error_message,
                "time": now_utc,
            })
            logger.info("Info error batched for daily summary: {msg}", msg=error_message[:80])
            return

        if severity == SEVERITY_WARNING:
            # Deduplicate: only send once per error type per hour
            error_key = self._error_dedup_key(error_message)
            last_sent = self._warning_sent.get(error_key)
            if last_sent is not None and (now_ts - last_sent) < self._WARNING_DEDUP_WINDOW:
                logger.debug(
                    "Warning suppressed (sent recently): {msg}",
                    msg=error_message[:80],
                )
                return
            self._warning_sent[error_key] = now_ts

            # Periodic cleanup: remove stale dedup entries
            if len(self._warning_sent) > 100:
                cutoff = now_ts - self._WARNING_DEDUP_WINDOW
                self._warning_sent = {
                    k: v for k, v in self._warning_sent.items() if v > cutoff
                }

        severity_label = severity.upper()
        lines = [
            f"*** ERROR [{severity_label}] ***",
            f"Error:  {error_message}",
            f"Time:   {now_utc}",
        ]
        message = "\n".join(lines)

        if severity == SEVERITY_CRITICAL:
            # Critical: send to ALL channels regardless
            self._broadcast_all(message)
        else:
            self._broadcast(message)

    # ------------------------------------------------------------------
    # Heartbeat & Lifecycle
    # ------------------------------------------------------------------

    def record_heartbeat(self) -> None:
        """Record that a cycle has started. Call at the top of each cycle."""
        self._last_heartbeat = time.monotonic()

    def check_heartbeat(self, expected_interval_minutes: int) -> None:
        """Check if the bot may be stuck and send an alert if so.

        Sends a warning if more than 2x the expected cycle interval has
        elapsed since the last recorded heartbeat.

        Args:
            expected_interval_minutes: The configured cycle interval.
        """
        elapsed = time.monotonic() - self._last_heartbeat
        threshold = expected_interval_minutes * 2 * 60  # 2x interval in seconds

        if elapsed > threshold:
            elapsed_min = elapsed / 60
            self.send_error_alert(
                f"Bot may be stuck! No cycle completed in {elapsed_min:.0f} min "
                f"(expected every {expected_interval_minutes} min)",
                severity=SEVERITY_CRITICAL,
            )

    def send_bot_online(
        self,
        mode: str,
        assets: list[str],
        cycle_interval: int,
    ) -> None:
        """Send a startup notification with key configuration.

        Args:
            mode: ``"LIVE"`` or ``"PAPER"``.
            assets: List of asset symbols being traded.
            cycle_interval: Cycle interval in minutes.
        """
        lines = [
            ">>> BOT ONLINE <<<",
            f"Mode:     {mode}",
            f"Assets:   {', '.join(assets)}",
            f"Interval: {cycle_interval} min",
            f"Time:     {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        ]
        message = "\n".join(lines)
        self._broadcast(message)

    def send_bot_offline(self) -> None:
        """Send a shutdown notification."""
        lines = [
            "<<< BOT OFFLINE >>>",
            f"Time:   {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "Grok Trader has been shut down.",
        ]
        message = "\n".join(lines)
        self._broadcast(message)

    def get_pending_info_errors(self) -> list[dict[str, Any]]:
        """Return the current list of batched info errors (for testing).

        Returns:
            Copy of the pending info errors list.
        """
        return list(self._info_errors)

    # ------------------------------------------------------------------
    # Internal: broadcasting
    # ------------------------------------------------------------------

    def _broadcast(self, message: str) -> None:
        """Send a message to all configured channels.

        Args:
            message: The plain-text message to send.
        """
        logger.debug("Notification: {msg}", msg=message[:120])

        if self.discord_url:
            self._send_discord(message)

        if self.telegram_token and self.telegram_chat_id:
            self._send_telegram(message)

    def _broadcast_all(self, message: str) -> None:
        """Send a message to ALL channels, even if some are not configured.

        Used for critical errors -- tries every channel and logs failures,
        but never raises. Falls back to ``_broadcast`` if channels are
        configured normally.

        Args:
            message: The plain-text message to send.
        """
        logger.warning("CRITICAL broadcast: {msg}", msg=message[:120])

        # Always try Discord
        if self.discord_url:
            self._send_discord(message)

        # Always try Telegram
        if self.telegram_token and self.telegram_chat_id:
            self._send_telegram(message)

        # If neither channel is configured, at least log prominently
        if not self.discord_url and not (self.telegram_token and self.telegram_chat_id):
            logger.error("CRITICAL ALERT (no channels): {msg}", msg=message)

    def _send_discord(self, message: str) -> None:
        """Post a message to a Discord webhook.

        Discord webhooks accept a JSON payload with a ``content`` field.
        Messages longer than 2000 characters are truncated.

        Args:
            message: The plain-text message.
        """
        # Discord has a 2000-character limit per message
        if len(message) > 1990:
            message = message[:1990] + "\n..."

        payload = {"content": f"```\n{message}\n```"}

        try:
            response = requests.post(
                self.discord_url,
                json=payload,
                timeout=self._DISCORD_TIMEOUT_SECONDS,
            )
            if response.status_code not in (200, 204):
                logger.warning(
                    "Discord webhook returned {code}: {body}",
                    code=response.status_code,
                    body=response.text[:200],
                )
        except requests.exceptions.Timeout:
            logger.warning("Discord webhook timed out.")
        except requests.exceptions.RequestException as exc:
            logger.warning("Discord webhook failed: {err}", err=str(exc))

    def _send_telegram(self, message: str) -> None:
        """Send a message via the Telegram Bot API.

        Uses the ``sendMessage`` endpoint with plain text parsing.
        Messages longer than 4096 characters are truncated.

        Args:
            message: The plain-text message.
        """
        # Telegram has a 4096-character limit
        if len(message) > 4090:
            message = message[:4090] + "\n..."

        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        payload = {
            "chat_id": self.telegram_chat_id,
            "text": message,
            "disable_web_page_preview": True,
        }

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self._TELEGRAM_TIMEOUT_SECONDS,
            )
            if response.status_code != 200:
                logger.warning(
                    "Telegram API returned {code}: {body}",
                    code=response.status_code,
                    body=response.text[:200],
                )
        except requests.exceptions.Timeout:
            logger.warning("Telegram API timed out.")
        except requests.exceptions.RequestException as exc:
            logger.warning("Telegram API failed: {err}", err=str(exc))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_direction_emoji(action: str) -> str:
        """Return an emoji indicator for the trade direction.

        Args:
            action: The trade action string (open_long, open_short, close, etc.).

        Returns:
            Emoji string for the direction.
        """
        action_lower = action.lower() if action else ""
        if "long" in action_lower:
            return "\U0001F4C8"  # chart increasing
        elif "short" in action_lower:
            return "\U0001F4C9"  # chart decreasing
        elif "close" in action_lower:
            return "\U0001F504"  # counterclockwise arrows
        return "\U0001F4CA"  # bar chart (fallback)

    @staticmethod
    def _fmt_price(value: Any) -> str:
        """Format a price value cleanly with commas and 2 decimal places.

        Small values (< $1) get more decimal places for readability.

        Args:
            value: A numeric price value.

        Returns:
            Formatted string like ``"$1,234.56"`` or ``"$0.000012"``.
        """
        if value is None:
            return "N/A"
        try:
            v = float(value)
            if abs(v) < 0.01:
                return f"${v:.6f}"
            elif abs(v) < 1.0:
                return f"${v:.4f}"
            else:
                return f"${v:,.2f}"
        except (ValueError, TypeError):
            return str(value)

    @staticmethod
    def _fmt_dollar(value: Any) -> str:
        """Format a dollar amount with commas and 2 decimal places.

        Args:
            value: A numeric dollar amount.

        Returns:
            Formatted string like ``"$12,345.67"``.
        """
        if value is None:
            return "N/A"
        try:
            return f"${float(value):,.2f}"
        except (ValueError, TypeError):
            return str(value)

    @staticmethod
    def _compute_risk_score(decision: Any, fill_price: Any) -> str:
        """Compute how much capital is at risk in this trade.

        Risk = size_pct * |entry - stop_loss| / entry * leverage.
        Expressed as a percentage of total capital.

        Args:
            decision: The trade decision object.
            fill_price: The actual fill price (or entry estimate).

        Returns:
            Formatted risk score string like ``"1.5% of capital at risk"``.
        """
        try:
            size_pct = getattr(decision, "size_pct", 0.0)
            leverage = getattr(decision, "leverage", 1.0)
            stop_loss = getattr(decision, "stop_loss", 0.0)
            entry = float(fill_price) if fill_price else getattr(decision, "entry_price", 0.0)

            if not entry or entry == 0 or not stop_loss or stop_loss == 0:
                return "N/A"

            sl_distance_pct = abs(entry - stop_loss) / entry
            risk_pct = size_pct * sl_distance_pct * leverage * 100
            return f"{risk_pct:.2f}% of capital"
        except (TypeError, ValueError, ZeroDivisionError):
            return "N/A"

    @staticmethod
    def _compute_performance_grade(
        win_rate: float,
        drawdown: float,
        trades_today: int,
    ) -> str:
        """Compute a letter grade based on win rate and risk management.

        Grading rubric:
            A: Win rate >= 60% and drawdown < 5%
            B: Win rate >= 50% and drawdown < 10%
            C: Win rate >= 40% and drawdown < 15%
            D: Win rate >= 30% or drawdown < 20%
            F: Everything else

        Args:
            win_rate: Win rate as a decimal (0.0 to 1.0).
            drawdown: Current drawdown percentage (0.0 to 100.0).
            trades_today: Number of trades executed today.

        Returns:
            A letter grade string.
        """
        if trades_today == 0:
            return "N/A"

        if win_rate >= 0.60 and drawdown < 5.0:
            return "A"
        elif win_rate >= 0.50 and drawdown < 10.0:
            return "B"
        elif win_rate >= 0.40 and drawdown < 15.0:
            return "C"
        elif win_rate >= 0.30 or drawdown < 20.0:
            return "D"
        else:
            return "F"

    @staticmethod
    def _error_dedup_key(error_message: str) -> str:
        """Generate a deduplication key from an error message.

        Strips timestamps and numeric IDs to group similar errors.

        Args:
            error_message: The raw error message.

        Returns:
            A simplified string key for deduplication.
        """
        # Simple approach: take the first 60 characters (before dynamic data)
        return error_message[:60].strip()

    @staticmethod
    def _compute_hold_duration(opened: str, closed: str) -> str:
        """Compute a human-readable hold duration between two ISO timestamps.

        Args:
            opened: ISO-format timestamp of trade open.
            closed: ISO-format timestamp of trade close.

        Returns:
            String like ``"2h 35m"`` or ``"N/A"`` if timestamps are invalid.
        """
        if not opened or not closed:
            return "N/A"

        try:
            t_open = opened if isinstance(opened, datetime) else datetime.fromisoformat(opened)
            t_close = closed if isinstance(closed, datetime) else datetime.fromisoformat(closed)

            # Ensure both are tz-aware for subtraction
            if t_open.tzinfo is None:
                t_open = t_open.replace(tzinfo=timezone.utc)
            if t_close.tzinfo is None:
                t_close = t_close.replace(tzinfo=timezone.utc)

            delta = t_close - t_open
            total_seconds = int(delta.total_seconds())

            if total_seconds < 0:
                return "N/A"

            days = total_seconds // 86400
            hours = (total_seconds % 86400) // 3600
            minutes = (total_seconds % 3600) // 60

            parts: list[str] = []
            if days > 0:
                parts.append(f"{days}d")
            if hours > 0:
                parts.append(f"{hours}h")
            parts.append(f"{minutes}m")

            return " ".join(parts)
        except (ValueError, TypeError):
            return "N/A"
