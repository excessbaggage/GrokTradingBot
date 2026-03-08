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
from data.correlation_risk import CorrelationRiskManager
from data.liquidation_estimator import LiquidationEstimator, LiquidationHeatmap
from utils.logger import logger


def build_context_prompt(
    market_data: dict[str, dict[str, Any]],
    portfolio: dict[str, Any],
    recent_trades: list[dict[str, Any]],
    risk_status: dict[str, Any],
    performance_summary: str = "",
    liquidation_data: dict[str, Any] | None = None,
    regime_data: dict[str, Any] | None = None,
    sentiment_data: dict[str, Any] | None = None,
    backtest_context: str = "",
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
        performance_summary: Pre-formatted performance analytics text
            from ``TradePerformanceAnalyzer.generate_performance_summary()``.
            If empty, the section is omitted.
        liquidation_data: Optional dict keyed by asset symbol, each value
            being a ``LiquidationHeatmap`` instance.  When provided,
            liquidation cluster estimates are appended to each asset
            section.
        regime_data: Optional dict keyed by asset symbol mapping to
            ``RegimeState`` objects from ``RegimeDetector.detect()``.
            When provided, regime classification and strategy advice are
            included in each asset's section.
        sentiment_data: Optional dict keyed by asset symbol mapping to
            ``SentimentData`` objects from ``XSentimentFetcher``.
            When provided, X/Twitter sentiment scores are included in
            each asset's section.
        backtest_context: Optional pre-formatted strategy performance
            text from ``WalkForwardBacktester.get_performance_context()``.
            When provided, recent backtest insights are included.

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
            asset_liq = (
                liquidation_data.get(asset) if liquidation_data else None
            )
            asset_regime = (
                regime_data.get(asset) if regime_data else None
            )
            asset_sentiment = (
                sentiment_data.get(asset) if sentiment_data is not None else None
            )
            sections.append(
                _build_asset_section(
                    asset, data, asset_liq, asset_regime, asset_sentiment
                )
            )

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

        # ── PERFORMANCE ANALYTICS ──────────────────────────────────────
        if performance_summary:
            sections.append(_build_performance_section(performance_summary))

        # ── CORRELATION AWARENESS ─────────────────────────────────────
        open_position_assets = [
            p.get("asset", "") for p in portfolio.get("positions", [])
        ]
        correlation_section = _build_correlation_section(
            market_data, open_position_assets,
        )
        if correlation_section:
            sections.append(correlation_section)

        # ── STRATEGY PERFORMANCE (BACKTEST) ──────────────────────────
        if backtest_context:
            sections.append(
                "## STRATEGY PERFORMANCE INSIGHTS\n" + backtest_context
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


def _build_asset_section(
    asset: str,
    data: dict[str, Any],
    liquidation: Any = None,
    regime: Any = None,
    sentiment: Any = None,
) -> str:
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

    # ── Liquidation Heatmap (optional) ────────────────────────────
    if liquidation is not None:
        estimator = LiquidationEstimator()
        liq_text = estimator.format_for_context(liquidation)
        lines.append(liq_text)

    # ── Market Regime (optional) ─────────────────────────────────
    if regime is not None:
        lines.extend([
            f"- **Market Regime**: {regime.regime.value.upper()} (confidence: {regime.confidence:.0%})",
            f"- **Regime Advice**: {regime.strategy_recommendation['description']}",
            f"- **ADX(14)**: {regime.adx:.1f} (+DI={regime.plus_di:.1f}, -DI={regime.minus_di:.1f})",
            f"- **Choppiness Index**: {regime.choppiness_index:.1f}",
            f"- **BB Width Percentile**: {regime.bb_width_percentile:.0f}%",
        ])

    # ── X Sentiment (optional) ─────────────────────────────────
    if sentiment is not None and hasattr(sentiment, "score"):
        topics_str = ", ".join(sentiment.key_topics[:3]) if sentiment.key_topics else "N/A"
        lines.extend([
            f"- **X Sentiment Score**: {sentiment.score:+.2f} (-1.0 bearish to +1.0 bullish)",
            f"- **X Sentiment Momentum**: {sentiment.momentum.upper()}",
            f"- **X Discussion Volume**: {sentiment.volume.upper()}",
            f"- **Key X Topics**: {topics_str}",
        ])

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


def _build_performance_section(performance_summary: str) -> str:
    """Build the performance analytics block from pre-formatted summary text."""
    return (
        "## YOUR PERFORMANCE ANALYTICS (Last 30 Days)\n"
        f"{performance_summary}"
    )


def _build_correlation_section(
    market_data: dict[str, dict[str, Any]],
    open_positions: list[str],
) -> str:
    """Build the correlation awareness section.

    Uses ``CorrelationRiskManager`` to identify highly correlated
    asset pairs and warn Grok about concentration risk.

    Args:
        market_data: Full market data dict.
        open_positions: List of currently open position asset symbols.

    Returns:
        Formatted string, or empty string if no meaningful data.
    """
    try:
        summary = CorrelationRiskManager.get_correlation_summary(
            market_data=market_data,
            open_positions=open_positions,
            threshold=0.70,
        )
        if summary and "No highly correlated" not in summary:
            return f"## CORRELATION AWARENESS\n{summary}"
        # Include even the "no correlations" message when positions are open
        if open_positions:
            return f"## CORRELATION AWARENESS\n{summary}"
        return ""
    except Exception as exc:
        logger.debug(
            "Could not build correlation section: {err}", err=exc,
        )
        return ""


def _build_task_section() -> str:
    """Build the task instruction footer."""
    return (
        "## YOUR TASK\n"
        "Analyze the above data. Provide your market analysis, portfolio "
        "assessment, and trading decisions in the required JSON format. "
        "If no high-conviction setups exist, return an empty decisions "
        "array. DO NOT FORCE TRADES."
    )
