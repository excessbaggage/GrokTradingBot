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
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import requests
from loguru import logger

from config.trading_config import (
    DISCORD_WEBHOOK_URL,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
)


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
    ) -> None:
        """Send an alert when a new trade is opened.

        Args:
            decision: The ``TradeDecision`` that was executed.
            order_result: The order result dict from ``OrderManager.place_order``.
        """
        mode = "LIVE" if order_result.get("live", False) else "PAPER"
        side = order_result.get("side", "unknown").upper()
        asset = getattr(decision, "asset", order_result.get("asset", "?"))

        lines = [
            f"--- NEW TRADE ({mode}) ---",
            f"Asset:     {asset}",
            f"Side:      {side}",
            f"Size:      {getattr(decision, 'size_pct', 0) * 100:.1f}% of portfolio",
            f"Leverage:  {getattr(decision, 'leverage', 1)}x",
            f"Entry:     {order_result.get('fill_price', 'N/A')}",
            f"Stop Loss: {getattr(decision, 'stop_loss', 'N/A')}",
            f"Target:    {getattr(decision, 'take_profit', 'N/A')}",
            f"R:R Ratio: {getattr(decision, 'risk_reward_ratio', 'N/A')}",
            f"Type:      {getattr(decision, 'order_type', 'N/A')}",
            f"Order ID:  {order_result.get('order_id', 'N/A')}",
            f"Fees:      {order_result.get('fees', 0.0)}",
            "",
            f"Reasoning: {getattr(decision, 'reasoning', 'N/A')}",
        ]
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
        """Send a daily performance summary.

        Args:
            summary_data: Dict with daily summary data.  Expected keys:
                ``date``, ``daily_pnl_pct``, ``equity``, ``peak_equity``,
                ``trades_today``, ``wins``, ``losses``,
                ``win_rate``, ``open_positions``, ``total_exposure_pct``.
        """
        pnl = summary_data.get("daily_pnl_pct", 0.0)
        equity = summary_data.get("equity", 0.0)
        peak = summary_data.get("peak_equity", equity)
        drawdown = ((peak - equity) / peak * 100) if peak > 0 else 0.0

        lines = [
            "=== DAILY SUMMARY ===",
            f"Date:           {summary_data.get('date', 'N/A')}",
            f"Daily P&L:      {pnl * 100:+.2f}%",
            f"Equity:         ${equity:,.2f}",
            f"Peak Equity:    ${peak:,.2f}",
            f"Drawdown:       {drawdown:.2f}%",
            "",
            f"Trades Today:   {summary_data.get('trades_today', 0)}",
            f"Wins:           {summary_data.get('wins', 0)}",
            f"Losses:         {summary_data.get('losses', 0)}",
            f"Win Rate:       {summary_data.get('win_rate', 0) * 100:.1f}%",
            "",
            f"Open Positions: {summary_data.get('open_positions', 0)}",
            f"Exposure:       {summary_data.get('total_exposure_pct', 0) * 100:.1f}%",
        ]
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

    def send_error_alert(self, error_message: str) -> None:
        """Send an error notification for operational issues.

        Args:
            error_message: Description of the error (API failure, parsing
                           error, etc.).
        """
        lines = [
            "*** ERROR ALERT ***",
            f"Error:  {error_message}",
            f"Time:   {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        ]
        message = "\n".join(lines)
        self._broadcast(message)

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
            t_open = datetime.fromisoformat(opened)
            t_close = datetime.fromisoformat(closed)

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
