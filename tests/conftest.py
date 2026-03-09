"""
Shared pytest fixtures for the GrokTradingBot test suite.

Provides reusable mock objects for portfolio state dicts, in-memory
SQLite databases, trade decisions, and complete Grok response payloads.
All fixtures are designed to work without network access or API keys.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from brain.models import (
    GrokResponse,
    RiskValidationResult,
    TradeDecision,
)
from config.risk_config import RISK_PARAMS
from execution.risk_guardian import RiskGuardian


# ======================================================================
# In-memory SQLite helpers
# ======================================================================


def _create_trades_table(conn: sqlite3.Connection) -> None:
    """Create the trades table used by RiskGuardian for history queries."""
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


def _insert_trade(
    conn: sqlite3.Connection,
    asset: str = "BTC",
    action: str = "open_long",
    opened_at: str | None = None,
    status: str = "open",
) -> None:
    """Insert a single trade record into the in-memory DB."""
    if opened_at is None:
        opened_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO trades (asset, action, side, size_pct, leverage,
                            entry_price, stop_loss, take_profit, status, opened_at)
        VALUES (?, ?, 'long', 0.05, 2.0, 65000.0, 63000.0, 70000.0, ?, ?)
        """,
        (asset, action, status, opened_at),
    )
    conn.commit()


# ======================================================================
# Fixtures: In-memory SQLite DB connections
# ======================================================================


@pytest.fixture
def empty_db() -> sqlite3.Connection:
    """An in-memory SQLite DB with the trades table but no rows."""
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    _create_trades_table(conn)
    yield conn
    conn.close()


@pytest.fixture
def recent_trade_db() -> sqlite3.Connection:
    """DB with one trade opened 2 minutes ago (too recent for 5-min gap)."""
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    _create_trades_table(conn)
    two_min_ago = (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()
    _insert_trade(conn, asset="BTC", opened_at=two_min_ago)
    yield conn
    conn.close()


@pytest.fixture
def old_trade_db() -> sqlite3.Connection:
    """DB with one trade opened 2 hours ago (well within time limits)."""
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    _create_trades_table(conn)
    two_hours_ago = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    _insert_trade(conn, asset="BTC", opened_at=two_hours_ago)
    yield conn
    conn.close()


@pytest.fixture
def maxed_trades_db() -> sqlite3.Connection:
    """DB with 50 trades today (at the daily limit of 50)."""
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    _create_trades_table(conn)
    # All 50 trades must land within today's UTC day so the daily count
    # query always finds them, even when the test runs near midnight UTC.
    # The latest trade is placed 6 minutes ago to clear the cooldown check.
    now = datetime.now(timezone.utc)
    today_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    latest_trade_time = now - timedelta(minutes=6)
    # Space trades evenly between midnight and 6 minutes ago
    span_seconds = (latest_trade_time - today_midnight).total_seconds()
    for i in range(50):
        offset = span_seconds * i / 49 if span_seconds > 0 else 0
        opened_at = (today_midnight + timedelta(seconds=offset)).isoformat()
        _insert_trade(conn, asset="BTC", opened_at=opened_at)
    yield conn
    conn.close()


# ======================================================================
# Fixtures: Portfolio State dicts
# ======================================================================


@pytest.fixture
def healthy_portfolio() -> dict[str, Any]:
    """A portfolio in good health: no losses, no existing exposure."""
    return {
        "equity": 10_000.0,
        "peak_equity": 10_000.0,
        "daily_pnl_pct": 0.0,
        "weekly_pnl_pct": 0.0,
        "total_exposure_pct": 0.0,
    }


@pytest.fixture
def daily_loss_portfolio() -> dict[str, Any]:
    """A portfolio that has hit the 10% daily loss limit."""
    return {
        "equity": 9_000.0,
        "peak_equity": 10_000.0,
        "daily_pnl_pct": -0.10,
        "weekly_pnl_pct": -0.10,
        "total_exposure_pct": 0.0,
    }


@pytest.fixture
def weekly_loss_portfolio() -> dict[str, Any]:
    """A portfolio that has hit the 20% weekly loss limit."""
    return {
        "equity": 8_000.0,
        "peak_equity": 10_000.0,
        "daily_pnl_pct": -0.05,
        "weekly_pnl_pct": -0.20,
        "total_exposure_pct": 0.0,
    }


@pytest.fixture
def drawdown_portfolio() -> dict[str, Any]:
    """A portfolio at 35% total drawdown from peak (triggers kill switch)."""
    return {
        "equity": 6_500.0,
        "peak_equity": 10_000.0,
        "daily_pnl_pct": -0.05,
        "weekly_pnl_pct": -0.10,
        "total_exposure_pct": 0.0,
    }


@pytest.fixture
def high_exposure_portfolio() -> dict[str, Any]:
    """A portfolio with 80% existing exposure (close to 85% limit)."""
    return {
        "equity": 10_000.0,
        "peak_equity": 10_000.0,
        "daily_pnl_pct": 0.0,
        "weekly_pnl_pct": 0.0,
        "total_exposure_pct": 0.80,
    }


# ======================================================================
# Fixtures: Risk Guardian
# ======================================================================


@pytest.fixture
def risk_guardian() -> RiskGuardian:
    """Standard Risk Guardian with default config."""
    return RiskGuardian(risk_params=RISK_PARAMS.copy())


@pytest.fixture
def kill_switch_guardian() -> RiskGuardian:
    """Risk Guardian with kill switch already active."""
    params = RISK_PARAMS.copy()
    params["kill_switch_enabled"] = True
    return RiskGuardian(risk_params=params)


# ======================================================================
# Fixtures: Trade Decisions
# ======================================================================


@pytest.fixture
def valid_long_decision() -> TradeDecision:
    """A well-formed open_long decision that should pass all risk checks."""
    return TradeDecision(
        action="open_long",
        asset="BTC",
        size_pct=0.05,
        leverage=2.0,
        entry_price=65000.0,
        stop_loss=63000.0,
        take_profit=70000.0,
        order_type="limit",
        reasoning="Strong support at 63k, bullish divergence on 4h.",
        conviction="high",
        risk_reward_ratio=2.5,
    )


@pytest.fixture
def valid_short_decision() -> TradeDecision:
    """A well-formed open_short decision."""
    return TradeDecision(
        action="open_short",
        asset="ETH",
        size_pct=0.03,
        leverage=2.0,
        entry_price=3500.0,
        stop_loss=3600.0,
        take_profit=3200.0,
        order_type="market",
        reasoning="Rejection at resistance, funding rate extreme.",
        conviction="medium",
        risk_reward_ratio=3.0,
    )


@pytest.fixture
def close_decision() -> TradeDecision:
    """A close-position decision (should bypass most checks)."""
    return TradeDecision(
        action="close",
        asset="BTC",
        size_pct=0.05,
        leverage=1.0,
        entry_price=None,
        stop_loss=0.0,
        take_profit=0.0,
        order_type="market",
        reasoning="Taking profit at resistance.",
        conviction="high",
        risk_reward_ratio=0.0,
    )


@pytest.fixture
def hold_decision() -> TradeDecision:
    """A hold decision (informational only)."""
    return TradeDecision(
        action="hold",
        asset="SOL",
        size_pct=0.0,
        leverage=1.0,
        entry_price=None,
        stop_loss=150.0,
        take_profit=200.0,
        order_type="market",
        reasoning="Maintaining current position.",
        conviction="medium",
        risk_reward_ratio=2.0,
    )


@pytest.fixture
def no_trade_decision() -> TradeDecision:
    """A no_trade decision (informational only)."""
    return TradeDecision(
        action="no_trade",
        asset="BTC",
        size_pct=0.0,
        leverage=1.0,
        entry_price=None,
        stop_loss=60000.0,
        take_profit=70000.0,
        order_type="market",
        reasoning="No high-conviction setups.",
        conviction="medium",
        risk_reward_ratio=2.0,
    )


# ======================================================================
# Fixtures: Sample Grok Response JSON strings
# ======================================================================


def _default_asset_analysis(
    bias: str = "neutral",
    conviction: str = "low",
    support: float = 100.0,
    resistance: float = 120.0,
    summary: str = "Range-bound. No clear setup.",
) -> dict:
    """Build a neutral AssetAnalysis dict for filler assets in test data."""
    return {
        "bias": bias,
        "conviction": conviction,
        "key_levels": {"support": support, "resistance": resistance},
        "sentiment_read": "Mixed sentiment, no strong signal.",
        "funding_rate_signal": "Neutral funding rate.",
        "summary": summary,
    }


def _build_sample_grok_response_dict(
    decisions: list[dict] | None = None,
) -> dict:
    """Build a complete GrokResponse-compatible dict with optional decisions."""
    if decisions is None:
        decisions = [
            {
                "action": "open_long",
                "asset": "BTC",
                "size_pct": 0.05,
                "leverage": 2.0,
                "entry_price": 65000.0,
                "stop_loss": 63000.0,
                "take_profit": 70000.0,
                "order_type": "limit",
                "reasoning": "Strong support at 63k with bullish divergence on 4h RSI.",
                "conviction": "high",
                "risk_reward_ratio": 2.5,
            }
        ]

    # Detailed analyses for original 3 assets
    market_analysis = {
        "btc": {
            "bias": "long",
            "conviction": "high",
            "key_levels": {"support": 63000.0, "resistance": 70000.0},
            "sentiment_read": "Cautiously optimistic, retail not yet overlevered.",
            "funding_rate_signal": "Neutral, 0.01% 8h -- no extremes.",
            "summary": "BTC consolidating above 64k support. Bullish structure intact.",
        },
        "eth": {
            "bias": "neutral",
            "conviction": "low",
            "key_levels": {"support": 3200.0, "resistance": 3600.0},
            "sentiment_read": "Mixed. ETH/BTC ratio declining.",
            "funding_rate_signal": "Slightly negative -- shorts paying longs.",
            "summary": "ETH range-bound between 3200-3600. No clear direction.",
        },
        "sol": {
            "bias": "short",
            "conviction": "medium",
            "key_levels": {"support": 140.0, "resistance": 165.0},
            "sentiment_read": "Bearish social sentiment after failed breakout.",
            "funding_rate_signal": "Elevated positive -- crowded long.",
            "summary": "SOL rejected at 165 resistance. Funding suggests overlevered longs.",
        },
    }

    # Add neutral defaults for remaining ASSET_UNIVERSE members
    from config.trading_config import ASSET_UNIVERSE

    _extra_prices = {
        "DOGE": (0.12, 0.18), "AVAX": (30.0, 40.0), "LINK": (15.0, 22.0),
        "ARB": (1.0, 1.5), "OP": (2.0, 3.0), "SUI": (1.5, 2.2), "APT": (7.0, 11.0),
        "PEPE": (0.000010, 0.000015), "SHIB": (0.000020, 0.000030),
        "WIF": (2.0, 3.0), "BONK": (0.000025, 0.000040),
        "FLOKI": (0.00015, 0.00025), "TRUMP": (12.0, 20.0), "PENGU": (0.010, 0.020),
    }
    for asset in ASSET_UNIVERSE:
        key = asset.lower()
        if key not in market_analysis:
            sup, res = _extra_prices.get(asset, (100.0, 120.0))
            market_analysis[key] = _default_asset_analysis(support=sup, resistance=res)

    return {
        "timestamp": "2026-03-07T12:00:00Z",
        "market_analysis": market_analysis,
        "portfolio_assessment": {
            "current_risk_level": "low",
            "recent_performance_note": "Last 3 trades profitable, equity at ATH.",
            "suggested_exposure_adjustment": "maintain",
        },
        "decisions": decisions,
        "overall_stance": "Cautiously bullish on BTC, waiting for confirmation on altcoins.",
        "next_review_suggestion_minutes": 15,
    }


@pytest.fixture
def sample_grok_response_dict() -> dict:
    """A complete valid GrokResponse as a Python dict."""
    return _build_sample_grok_response_dict()


@pytest.fixture
def sample_grok_response_json() -> str:
    """A complete valid GrokResponse as a JSON string."""
    return json.dumps(_build_sample_grok_response_dict())


@pytest.fixture
def sample_grok_response_json_with_markdown() -> str:
    """A valid GrokResponse wrapped in markdown code fences."""
    raw = json.dumps(_build_sample_grok_response_dict(), indent=2)
    return f"```json\n{raw}\n```"


@pytest.fixture
def sample_grok_response_empty_decisions_json() -> str:
    """A valid GrokResponse with no decisions (bot is staying flat)."""
    return json.dumps(_build_sample_grok_response_dict(decisions=[]))


@pytest.fixture
def sample_grok_response_mixed_decisions_json() -> str:
    """A GrokResponse with a mix of actionable and non-actionable decisions."""
    decisions = [
        {
            "action": "open_long",
            "asset": "BTC",
            "size_pct": 0.05,
            "leverage": 2.0,
            "entry_price": 65000.0,
            "stop_loss": 63000.0,
            "take_profit": 70000.0,
            "order_type": "limit",
            "reasoning": "Bullish setup.",
            "conviction": "high",
            "risk_reward_ratio": 2.5,
        },
        {
            "action": "no_trade",
            "asset": "ETH",
            "size_pct": 0.0,
            "leverage": 1.0,
            "entry_price": None,
            "stop_loss": 3200.0,
            "take_profit": 3600.0,
            "order_type": "market",
            "reasoning": "No setup.",
            "conviction": "medium",
            "risk_reward_ratio": 2.0,
        },
        {
            "action": "hold",
            "asset": "SOL",
            "size_pct": 0.0,
            "leverage": 1.0,
            "entry_price": None,
            "stop_loss": 140.0,
            "take_profit": 165.0,
            "order_type": "market",
            "reasoning": "Holding existing position.",
            "conviction": "medium",
            "risk_reward_ratio": 2.0,
        },
    ]
    return json.dumps(_build_sample_grok_response_dict(decisions=decisions))
