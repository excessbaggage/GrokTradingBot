"""
Context prompt builder for the Grok trading agent.

Assembles market data, portfolio state, recent trades, and risk status
into the dynamic ``user`` message sent to Grok each cycle.  The format
matches Section 6 of the project specification.
"""

from __future__ import annotations

from typing import Any

from utils.helpers import (
    format_price,
    format_pct,
    format_usd,
    summarize_candles,
    utc_now,
)
from config.risk_config import RISK_PARAMS
from utils.logger import logger


def build_context_prompt(
    market_data: dict[str, dict[str, Any]],
    portfolio: dict[str, Any],
    recent_trades: list[dict[str, Any]],
    risk_status: dict[str, Any],
) -> str:
    """Assemble the dynamic context prompt for Grok.

    This is the ``user`` message content sent every cycle.  It contains
    all the data Grok needs to make trading decisions.

    Args:
        market_data: Nested dict keyed by asset symbol, as returned by
            ``MarketDataFetcher.fetch_all_market_data()``.
        portfolio: Dict with ``total_equity``, ``available_margin``,
            ``unrealized_pnl``, ``positions``, etc.
        recent_trades: List of recent trade dicts (newest first).
        risk_status: Dict with ``daily_pnl``, ``weekly_pnl``,
            ``drawdown_from_peak``, ``trades_today``,
            ``consecutive_losses``, etc.

    Returns:
        A fully formatted string ready to be passed as the Grok user
        message.
    """
    try:
        sections: list[str] = []
        timestamp = utc_now().isoformat()

        # ── HEADER ────────────────────────────────────────────────────
        sections.append(f"## CURRENT MARKET DATA (as of {timestamp})")

        # ── PER-ASSET MARKET DATA ─────────────────────────────────────
        for asset, data in market_data.items():
            sections.append(_build_asset_section(asset, data))

        # ── PORTFOLIO ─────────────────────────────────────────────────
        sections.append(_build_portfolio_section(portfolio, risk_status))

        # ── OPEN POSITIONS TABLE ──────────────────────────────────────
        sections.append(_build_positions_section(portfolio.get("positions", [])))

        # ── RECENT TRADES TABLE ───────────────────────────────────────
        sections.append(_build_recent_trades_section(recent_trades))

        # ── RISK STATUS ───────────────────────────────────────────────
        sections.append(
            _build_risk_status_section(risk_status, portfolio)
        )

        # ── TASK INSTRUCTION ──────────────────────────────────────────
        sections.append(_build_task_section())

        prompt = "\n\n".join(sections)
        logger.debug(
            "Context prompt built | length={ln} chars", ln=len(prompt)
        )
        return prompt

    except Exception as exc:
        logger.error("Failed to build context prompt: {err}", err=exc)
        # Return a minimal fallback so the cycle can still proceed
        return (
            "## CONTEXT BUILD ERROR\n"
            f"An error occurred assembling market data: {exc}\n"
            "Please respond with an empty decisions array and stay flat."
        )


# ═══════════════════════════════════════════════════════════════════════════
# PRIVATE SECTION BUILDERS
# ═══════════════════════════════════════════════════════════════════════════


def _build_asset_section(asset: str, data: dict[str, Any]) -> str:
    """Build the market-data block for a single asset."""
    price = data.get("price", 0)
    change_24h = data.get("24h_change_pct", 0)

    funding = data.get("funding", {})
    funding_current = funding.get("current_rate", 0)
    funding_7d = funding.get("avg_7d_rate", 0)

    oi = data.get("oi", {})
    current_oi = oi.get("current_oi", 0)
    oi_change = oi.get("oi_24h_change_pct", 0)

    # Candle summaries
    candles = data.get("candles", {})
    summary_4h = summarize_candles(candles.get("4h"))
    summary_1d = summarize_candles(candles.get("1d"))
    summary_1h = summarize_candles(candles.get("1h"))

    # Technical indicators (ATR, RSI, adaptive thresholds)
    tech = data.get("technicals", {})
    atr_14 = tech.get("atr_14", 0)
    atr_pct = tech.get("atr_pct", 0)
    rsi_14 = tech.get("rsi_14", 50)
    adaptive_ob = tech.get("adaptive_ob", 70)
    adaptive_os = tech.get("adaptive_os", 30)
    vol_regime = tech.get("volatility_regime", "normal")
    turtle_factor = tech.get("turtle_size_factor", 1.0)

    lines = [
        f"### {asset}-USD Perpetual",
        f"- Price: {format_usd(price)}",
        f"- 24h Change: {format_pct(change_24h)}",
        f"- 1h Candles Summary: {summary_1h}",
        f"- 4h Candles Summary: {summary_4h}",
        f"- Daily Candles Summary: {summary_1d}",
        f"- Funding Rate (current): {funding_current:.6f}% (8h)",
        f"- Funding Rate (avg 7d): {funding_7d:.6f}%",
        f"- Open Interest: {format_usd(current_oi)} ({format_pct(oi_change / 100 if oi_change else 0)} 24h change)",
        f"- **ATR(14)**: {format_usd(atr_14)} ({format_pct(atr_pct)} of price)",
        f"- **RSI(14)**: {rsi_14} (adaptive OB={adaptive_ob}, OS={adaptive_os})",
        f"- **Volatility Regime**: {vol_regime.upper()}",
        f"- **Turtle Size Factor**: {turtle_factor:.2f}x (higher = safer to size up, lower = reduce size)",
    ]
    return "\n".join(lines)


def _build_portfolio_section(
    portfolio: dict[str, Any],
    risk_status: dict[str, Any],
) -> str:
    """Build the portfolio overview block."""
    total_equity = portfolio.get("total_equity", 0)
    available_margin = portfolio.get("available_margin", 0)
    unrealized_pnl = portfolio.get("unrealized_pnl", 0)

    daily_pnl = risk_status.get("daily_pnl", 0)
    weekly_pnl = risk_status.get("weekly_pnl", 0)

    unrealized_pnl_pct = (
        unrealized_pnl / total_equity if total_equity else 0
    )
    daily_pnl_pct = daily_pnl / total_equity if total_equity else 0
    weekly_pnl_pct = weekly_pnl / total_equity if total_equity else 0

    lines = [
        "## YOUR CURRENT PORTFOLIO",
        f"- Total Equity: {format_usd(total_equity)}",
        f"- Available Margin: {format_usd(available_margin)}",
        f"- Unrealized P&L: {format_usd(unrealized_pnl)} ({format_pct(unrealized_pnl_pct)})",
        f"- Daily P&L: {format_usd(daily_pnl)} ({format_pct(daily_pnl_pct)})",
        f"- Weekly P&L: {format_usd(weekly_pnl)} ({format_pct(weekly_pnl_pct)})",
    ]
    return "\n".join(lines)


def _build_positions_section(positions: list[dict[str, Any]]) -> str:
    """Build the open positions table."""
    if not positions:
        return "### Open Positions\nNo open positions."

    header = "### Open Positions"
    table_header = "| Asset | Side | Size | Entry Price | Unrealized P&L | Leverage |"
    separator = "|-------|------|------|-------------|----------------|----------|"

    rows: list[str] = []
    for pos in positions:
        rows.append(
            f"| {pos.get('asset', 'N/A')} "
            f"| {pos.get('side', 'N/A')} "
            f"| {pos.get('size', 0):.4f} "
            f"| {format_usd(pos.get('entry_price', 0))} "
            f"| {format_usd(pos.get('unrealized_pnl', 0))} "
            f"| {pos.get('leverage', 1):.1f}x |"
        )

    return "\n".join([header, table_header, separator] + rows)


def _build_recent_trades_section(trades: list[dict[str, Any]]) -> str:
    """Build the recent trades table."""
    header = "### Recent Trades (Last 10)"

    if not trades:
        return f"{header}\nNo recent trades."

    table_header = "| Time | Asset | Side | Action | Entry | Exit | P&L | Status |"
    separator = "|------|-------|------|--------|-------|------|-----|--------|"

    rows: list[str] = []
    for t in trades[:10]:
        entry = format_usd(t.get("entry_price", 0)) if t.get("entry_price") else "N/A"
        exit_p = format_usd(t.get("exit_price", 0)) if t.get("exit_price") else "N/A"
        pnl_str = format_usd(t.get("pnl", 0)) if t.get("pnl") is not None else "N/A"
        ts = t.get("opened_at", t.get("timestamp", "N/A"))
        # Truncate timestamp to minutes for readability
        if isinstance(ts, str) and len(ts) > 16:
            ts = ts[:16]

        rows.append(
            f"| {ts} "
            f"| {t.get('asset', 'N/A')} "
            f"| {t.get('side', 'N/A')} "
            f"| {t.get('action', 'N/A')} "
            f"| {entry} "
            f"| {exit_p} "
            f"| {pnl_str} "
            f"| {t.get('status', 'N/A')} |"
        )

    return "\n".join([header, table_header, separator] + rows)


def _build_risk_status_section(
    risk_status: dict[str, Any],
    portfolio: dict[str, Any],
) -> str:
    """Build the risk status block."""
    total_equity = portfolio.get("total_equity", 0)

    daily_pnl = risk_status.get("daily_pnl", 0)
    weekly_pnl = risk_status.get("weekly_pnl", 0)
    drawdown = risk_status.get("drawdown_from_peak", 0)
    trades_today = risk_status.get("trades_today", 0)
    consecutive_losses = risk_status.get("consecutive_losses", 0)
    max_trades = RISK_PARAMS["max_trades_per_day"]

    # Compute remaining loss budgets
    daily_limit = RISK_PARAMS["max_daily_loss_pct"] * total_equity
    weekly_limit = RISK_PARAMS["max_weekly_loss_pct"] * total_equity
    daily_remaining = daily_limit + daily_pnl  # daily_pnl is negative if losing
    weekly_remaining = weekly_limit + weekly_pnl

    lines = [
        "## RISK STATUS",
        f"- Daily loss limit remaining: {format_usd(max(daily_remaining, 0))}",
        f"- Weekly loss limit remaining: {format_usd(max(weekly_remaining, 0))}",
        f"- Drawdown from peak: {format_pct(drawdown)}",
        f"- Trades today: {trades_today}/{max_trades}",
        f"",
        f"### Recent Losing Streak Count: {consecutive_losses}",
    ]
    return "\n".join(lines)


def _build_task_section() -> str:
    """Build the task instruction footer."""
    return (
        "## YOUR TASK\n"
        "Analyze the above data. Provide your market analysis, portfolio "
        "assessment, and trading decisions in the required JSON format. "
        "If no high-conviction setups exist, return an empty decisions "
        "array. DO NOT FORCE TRADES."
    )
