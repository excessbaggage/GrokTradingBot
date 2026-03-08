"""
Formatting, math, and time utility functions used throughout the trading bot.

All helpers are pure functions with no side effects.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# FORMATTING HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def format_price(price: float, decimals: int = 2) -> str:
    """Format a price with appropriate decimal places and comma separators.

    Args:
        price: The price value.
        decimals: Number of decimal places (default 2).

    Returns:
        Formatted string like ``"67,432.10"``.
    """
    return f"{price:,.{decimals}f}"


def format_pct(value: float, decimals: int = 2, include_sign: bool = True) -> str:
    """Format a decimal ratio as a percentage string.

    Args:
        value: The ratio (e.g. 0.0523 for 5.23%).
        decimals: Number of decimal places.
        include_sign: Whether to prepend ``+`` for positive values.

    Returns:
        Formatted string like ``"+5.23%"`` or ``"-1.40%"``.
    """
    pct = value * 100
    sign = "+" if include_sign and pct > 0 else ""
    return f"{sign}{pct:.{decimals}f}%"


def format_usd(amount: float, decimals: int = 2) -> str:
    """Format a dollar amount with ``$`` sign and comma separators.

    Args:
        amount: The dollar amount.
        decimals: Number of decimal places.

    Returns:
        Formatted string like ``"$12,345.67"`` or ``"-$500.00"``.
    """
    if amount < 0:
        return f"-${abs(amount):,.{decimals}f}"
    return f"${amount:,.{decimals}f}"


# ═══════════════════════════════════════════════════════════════════════════
# MATH UTILITIES
# ═══════════════════════════════════════════════════════════════════════════


def calculate_pnl_pct(entry_price: float, exit_price: float, side: str) -> float:
    """Calculate the percentage profit/loss for a trade.

    Args:
        entry_price: The entry price.
        exit_price: The exit price.
        side: ``"long"`` or ``"short"``.

    Returns:
        P&L as a decimal ratio (e.g. 0.05 for +5%).

    Raises:
        ValueError: If *entry_price* is zero or *side* is invalid.
    """
    if entry_price == 0:
        raise ValueError("entry_price cannot be zero")

    side = side.lower()
    if side == "long":
        return (exit_price - entry_price) / entry_price
    elif side == "short":
        return (entry_price - exit_price) / entry_price
    else:
        raise ValueError(f"Invalid side: {side!r}. Must be 'long' or 'short'.")


def calculate_risk_reward_ratio(
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    side: str,
) -> float:
    """Calculate the risk-to-reward ratio for a proposed trade.

    Args:
        entry_price: Planned entry price.
        stop_loss: Stop-loss price.
        take_profit: Take-profit price.
        side: ``"long"`` or ``"short"``.

    Returns:
        The reward-to-risk ratio (e.g. 2.0 means 2:1 R:R).

    Raises:
        ValueError: If the risk is zero or inputs are inconsistent.
    """
    side = side.lower()

    if side == "long":
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
    elif side == "short":
        risk = abs(stop_loss - entry_price)
        reward = abs(entry_price - take_profit)
    else:
        raise ValueError(f"Invalid side: {side!r}. Must be 'long' or 'short'.")

    if risk == 0:
        raise ValueError("Risk (distance to stop-loss) cannot be zero.")

    return reward / risk


# ═══════════════════════════════════════════════════════════════════════════
# TIME UTILITIES
# ═══════════════════════════════════════════════════════════════════════════


def utc_now() -> datetime:
    """Return the current UTC datetime (timezone-aware).

    Returns:
        ``datetime`` with ``tzinfo=timezone.utc``.
    """
    return datetime.now(timezone.utc)


def is_within_minutes(dt: datetime | None, minutes: int) -> bool:
    """Check whether a datetime is within *minutes* of now (UTC).

    Args:
        dt: The datetime to check. If ``None``, returns ``False``.
        minutes: The threshold in minutes.

    Returns:
        ``True`` if *dt* is within the window, ``False`` otherwise.
    """
    if dt is None:
        return False

    # Make *dt* timezone-aware if naive
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    delta = utc_now() - dt
    return delta <= timedelta(minutes=minutes)


def time_since(dt: datetime | None) -> timedelta | None:
    """Return the elapsed time since *dt*.

    Args:
        dt: A datetime value. If ``None``, returns ``None``.

    Returns:
        A ``timedelta`` representing the elapsed time, or ``None``.
    """
    if dt is None:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return utc_now() - dt


# ═══════════════════════════════════════════════════════════════════════════
# CANDLE SUMMARY HELPER
# ═══════════════════════════════════════════════════════════════════════════


def summarize_candles(df: pd.DataFrame) -> str:
    """Produce a concise text summary of OHLCV candle data.

    The summary includes trend direction, key price levels, and average
    volume -- designed to be injected into a context prompt for Grok.

    Args:
        df: DataFrame with columns ``open``, ``high``, ``low``, ``close``,
            ``volume``.  Rows should be chronologically ordered (oldest
            first).

    Returns:
        A multi-line string summary suitable for the context prompt.
        Returns a fallback message if the DataFrame is empty or missing
        required columns.
    """
    required_cols = {"open", "high", "low", "close", "volume"}
    if df is None or df.empty or not required_cols.issubset(set(df.columns)):
        return "No candle data available."

    # Key price levels across the window
    period_high = df["high"].max()
    period_low = df["low"].min()
    latest_close = df["close"].iloc[-1]
    first_open = df["open"].iloc[0]

    # Trend direction
    pct_change = ((latest_close - first_open) / first_open) * 100 if first_open != 0 else 0.0
    if pct_change > 1.0:
        trend = "BULLISH"
    elif pct_change < -1.0:
        trend = "BEARISH"
    else:
        trend = "SIDEWAYS"

    # Average volume
    avg_volume = df["volume"].mean()

    # Recent momentum -- last 3 candles
    recent = df.tail(min(3, len(df)))
    recent_change = 0.0
    if len(recent) >= 2:
        r_open = recent["open"].iloc[0]
        r_close = recent["close"].iloc[-1]
        recent_change = ((r_close - r_open) / r_open) * 100 if r_open != 0 else 0.0

    # Volatility -- average true range proxy
    highs = df["high"].values
    lows = df["low"].values
    avg_range = float((pd.Series(highs) - pd.Series(lows)).mean())
    avg_range_pct = (avg_range / latest_close) * 100 if latest_close != 0 else 0.0

    lines = [
        f"Trend: {trend} ({pct_change:+.2f}% over {len(df)} candles)",
        f"Range: {format_price(period_low)} - {format_price(period_high)}",
        f"Latest close: {format_price(latest_close)}",
        f"Recent momentum (last {len(recent)} candles): {recent_change:+.2f}%",
        f"Avg volume: {avg_volume:,.0f}",
        f"Avg candle range: {avg_range_pct:.2f}%",
    ]
    return " | ".join(lines)
