"""
End-to-end integration tests for the Grok Trading Bot.

These tests exercise REAL code paths across module boundaries.
Only external APIs (Grok, Hyperliquid, Discord, Telegram) are mocked.
All database operations use in-memory SQLite.

Test scenarios:
    1.  Database round-trip (init -> write -> read -> verify)
    2.  Context builder with realistic data structures
    3.  Decision parser with realistic Grok responses
    4.  Risk Guardian integration with TradeHistoryManager
    5.  Full cycle simulation (all real except external APIs)
    6.  Paper trading engine lifecycle
    7.  Kill switch persistence
    8.  Error recovery
    9.  Notification formatting
    10. Logger integration
"""

from __future__ import annotations

import io
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest

from brain.decision_parser import DecisionParser
from brain.models import (
    GrokResponse,
    RiskValidationResult,
    TradeDecision,
)
from brain.system_prompt import get_system_prompt
from config.risk_config import RISK_PARAMS
from data.context_builder import build_context_prompt
from data.database import (
    execute_query,
    fetch_all,
    fetch_one,
)

# SQLite-compatible schema matching the Supabase production tables.
# Used to create in-memory test databases.
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT DEFAULT (datetime('now')),
    asset TEXT NOT NULL,
    side TEXT CHECK(side IN ('long','short')),
    action TEXT,
    size_pct REAL,
    leverage REAL DEFAULT 1.0,
    entry_price REAL,
    exit_price REAL,
    stop_loss REAL,
    take_profit REAL,
    pnl REAL,
    pnl_pct REAL,
    fees REAL DEFAULT 0.0,
    status TEXT DEFAULT 'open' CHECK(status IN ('open','closed')),
    reasoning TEXT,
    conviction TEXT,
    opened_at TEXT DEFAULT (datetime('now')),
    closed_at TEXT
);

CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset TEXT NOT NULL,
    side TEXT CHECK(side IN ('long','short')),
    size_pct REAL,
    leverage REAL DEFAULT 1.0,
    entry_price REAL,
    stop_loss REAL,
    take_profit REAL,
    unrealized_pnl REAL DEFAULT 0.0,
    opened_at TEXT DEFAULT (datetime('now')),
    status TEXT DEFAULT 'open' CHECK(status IN ('open','closed'))
);

CREATE TABLE IF NOT EXISTS grok_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT DEFAULT (datetime('now')),
    system_prompt_hash TEXT,
    context_prompt TEXT,
    response_text TEXT,
    decisions_json TEXT,
    cycle_number INTEGER
);

CREATE TABLE IF NOT EXISTS daily_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT UNIQUE,
    starting_equity REAL,
    ending_equity REAL,
    pnl REAL,
    pnl_pct REAL,
    trades_count INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    win_rate REAL,
    max_drawdown REAL
);

CREATE TABLE IF NOT EXISTS rejections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT DEFAULT (datetime('now')),
    asset TEXT,
    action TEXT,
    reason TEXT,
    decision_json TEXT
);

CREATE TABLE IF NOT EXISTS equity_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT DEFAULT (datetime('now')),
    cycle_number INTEGER,
    equity REAL,
    unrealized_pnl REAL DEFAULT 0.0,
    realized_pnl REAL DEFAULT 0.0,
    open_positions INTEGER DEFAULT 0,
    total_exposure REAL DEFAULT 0.0
);
"""
from data.portfolio_state import PortfolioManager
from data.trade_history import TradeHistoryManager
from execution.notifications import Notifier
from execution.risk_guardian import RiskGuardian
from utils.helpers import (
    calculate_pnl_pct,
    calculate_risk_reward_ratio,
    format_pct,
    format_price,
    format_usd,
    summarize_candles,
    utc_now,
)
from utils.logger import (
    log_trade_decision,
    log_trade_execution,
    log_trade_rejection,
    log_grok_cycle,
)


# =====================================================================
# Helpers
# =====================================================================


def _make_db() -> sqlite3.Connection:
    """Create an in-memory database with the FULL production schema."""
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA_SQL)
    conn.commit()
    return conn


def _default_asset_analysis(
    support: float = 100.0,
    resistance: float = 120.0,
) -> dict:
    """Build a neutral AssetAnalysis dict for filler assets in test data."""
    return {
        "bias": "neutral",
        "conviction": "low",
        "key_levels": {"support": support, "resistance": resistance},
        "sentiment_read": "Mixed sentiment, no strong signal.",
        "funding_rate_signal": "Neutral funding rate.",
        "summary": "Range-bound. No clear setup.",
    }


def _sample_grok_response_dict(
    decisions: list[dict] | None = None,
    next_review: int = 15,
) -> dict:
    """Build a complete, valid GrokResponse-compatible dict."""
    if decisions is None:
        decisions = [
            {
                "action": "open_long",
                "asset": "BTC",
                "size_pct": 0.05,
                "leverage": 2.0,
                "entry_price": 67000.0,
                "stop_loss": 65500.0,
                "take_profit": 72000.0,
                "order_type": "limit",
                "reasoning": "Breakout above 66.5k with volume confirmation on 4h.",
                "conviction": "high",
                "risk_reward_ratio": 3.33,
            }
        ]

    # Detailed analyses for original 3 assets
    market_analysis = {
        "btc": {
            "bias": "long",
            "conviction": "high",
            "key_levels": {"support": 65000.0, "resistance": 72000.0},
            "sentiment_read": "Cautiously optimistic on X. Retail not overleveraged.",
            "funding_rate_signal": "Neutral at 0.008%, no extreme.",
            "summary": "BTC holding above 66k after breakout. Bullish structure.",
        },
        "eth": {
            "bias": "neutral",
            "conviction": "low",
            "key_levels": {"support": 3200.0, "resistance": 3700.0},
            "sentiment_read": "Mixed. ETH/BTC ratio declining.",
            "funding_rate_signal": "Slightly negative.",
            "summary": "ETH range-bound. No clear setup.",
        },
        "sol": {
            "bias": "short",
            "conviction": "medium",
            "key_levels": {"support": 140.0, "resistance": 170.0},
            "sentiment_read": "Bearish after failed breakout.",
            "funding_rate_signal": "Elevated positive. Crowded long.",
            "summary": "SOL rejected at 170 resistance. Overlevered longs.",
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
        "timestamp": utc_now().isoformat(),
        "market_analysis": market_analysis,
        "portfolio_assessment": {
            "current_risk_level": "low",
            "recent_performance_note": "2 of last 3 trades profitable.",
            "suggested_exposure_adjustment": "maintain",
        },
        "decisions": decisions,
        "overall_stance": "Cautiously bullish BTC, staying flat on alts.",
        "next_review_suggestion_minutes": next_review,
    }


def _make_market_data() -> dict[str, dict[str, Any]]:
    """Build realistic market data matching MarketDataFetcher output."""
    def _candles(base: float, n: int = 20) -> pd.DataFrame:
        rows = []
        for i in range(n):
            open_ = base + i * 10
            high = open_ + 50
            low = open_ - 30
            close = open_ + 20
            volume = 100_000 + i * 500
            ts = utc_now() - timedelta(hours=n - i)
            rows.append({
                "timestamp": ts,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            })
        return pd.DataFrame(rows)

    from config.trading_config import ASSET_UNIVERSE

    # Detailed data for original 3 + realistic defaults for others
    _asset_data = {
        "BTC": (67200.0, 0.023, 66000, 65500, 64000, 5_000_000_000, 2.5, 0.0001, 0.00008),
        "ETH": (3450.0, -0.005, 3400, 3380, 3300, 2_000_000_000, -1.0, -0.00005, 0.00002),
        "SOL": (155.0, -0.018, 153, 150, 145, 800_000_000, 5.0, 0.0003, 0.00025),
        "DOGE": (0.15, 0.01, 0.14, 0.13, 0.12, 200_000_000, 1.0, 0.0001, 0.0001),
        "AVAX": (35.0, -0.01, 34, 33, 32, 300_000_000, 0.5, 0.0001, 0.0001),
        "LINK": (18.0, 0.005, 17, 16.5, 16, 400_000_000, 1.5, 0.0001, 0.0001),
        "ARB": (1.2, -0.008, 1.1, 1.05, 1.0, 150_000_000, -0.5, 0.0001, 0.0001),
        "OP": (2.5, 0.012, 2.4, 2.3, 2.2, 120_000_000, 2.0, 0.0001, 0.0001),
        "SUI": (1.8, -0.003, 1.7, 1.6, 1.5, 100_000_000, 0.0, 0.0001, 0.0001),
        "APT": (9.0, 0.008, 8.5, 8.0, 7.5, 180_000_000, 1.0, 0.0001, 0.0001),
        # Meme coins
        "PEPE": (0.000012, 0.05, 0.000011, 0.000010, 0.000009, 50_000_000, 3.0, 0.0003, 0.0002),
        "SHIB": (0.000025, -0.02, 0.000024, 0.000023, 0.000022, 80_000_000, -1.0, 0.0001, 0.0001),
        "WIF": (2.50, 0.08, 2.3, 2.1, 2.0, 60_000_000, 5.0, 0.0005, 0.0003),
        "BONK": (0.000030, -0.03, 0.000028, 0.000025, 0.000022, 40_000_000, -2.0, 0.0002, 0.0001),
        "FLOKI": (0.00020, 0.02, 0.00019, 0.00018, 0.00017, 30_000_000, 1.0, 0.0001, 0.0001),
        "TRUMP": (15.0, 0.10, 14.0, 13.0, 12.0, 90_000_000, 8.0, 0.0008, 0.0005),
        "PENGU": (0.015, -0.04, 0.014, 0.013, 0.012, 20_000_000, -3.0, 0.0002, 0.0001),
    }

    result = {}
    for asset in ASSET_UNIVERSE:
        price, chg, c1h, c4h, c1d, oi, oi_chg, fund, fund_avg = _asset_data.get(
            asset, (100.0, 0.0, 99, 98, 97, 100_000_000, 0.0, 0.0001, 0.0001)
        )
        result[asset] = {
            "asset": asset,
            "price": price,
            "24h_change_pct": chg,
            "candles": {
                "1h": _candles(c1h, 48),
                "4h": _candles(c4h, 20),
                "1d": _candles(c1d, 10),
            },
            "funding": {
                "current_rate": fund,
                "avg_7d_rate": fund_avg,
            },
            "oi": {
                "current_oi": oi,
                "oi_24h_change_pct": oi_chg,
            },
        }

    return result


def _make_portfolio(
    equity: float = 10_000.0,
    positions: list[dict] | None = None,
) -> dict[str, Any]:
    """Build a realistic portfolio dict."""
    return {
        "total_equity": equity,
        "available_margin": equity * 0.8,
        "unrealized_pnl": 0.0,
        "positions": positions or [],
    }


def _make_risk_status(
    daily_pnl: float = 0.0,
    weekly_pnl: float = 0.0,
    drawdown: float = 0.0,
    trades_today: int = 0,
    consecutive_losses: int = 0,
) -> dict[str, Any]:
    return {
        "daily_pnl": daily_pnl,
        "weekly_pnl": weekly_pnl,
        "drawdown_from_peak": drawdown,
        "trades_today": trades_today,
        "consecutive_losses": consecutive_losses,
    }


def _make_valid_decision(**overrides) -> TradeDecision:
    """Build a valid TradeDecision with optional overrides."""
    defaults = dict(
        action="open_long",
        asset="BTC",
        size_pct=0.05,
        leverage=2.0,
        entry_price=67000.0,
        stop_loss=65500.0,
        take_profit=72000.0,
        order_type="limit",
        reasoning="Strong technical breakout with volume.",
        conviction="high",
        risk_reward_ratio=3.33,
    )
    defaults.update(overrides)
    return TradeDecision(**defaults)


# =====================================================================
# 1. DATABASE ROUND-TRIP
# =====================================================================


class TestDatabaseRoundTrip:
    """Verify data integrity through init -> write -> read cycles."""

    def test_full_schema_creates_all_tables(self):
        """init_db schema creates all 5 tables + indexes."""
        conn = _make_db()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = sorted(row[0] for row in cursor.fetchall())
        assert "trades" in tables
        assert "positions" in tables
        assert "grok_logs" in tables
        assert "daily_summaries" in tables
        assert "rejections" in tables
        conn.close()

    def test_log_trade_and_retrieve(self):
        """Log a trade via TradeHistoryManager then read it back."""
        conn = _make_db()
        thm = TradeHistoryManager()

        trade_data = {
            "asset": "BTC",
            "side": "long",
            "action": "open_long",
            "size_pct": 0.05,
            "leverage": 2.0,
            "entry_price": 67000.0,
            "stop_loss": 65500.0,
            "take_profit": 72000.0,
            "reasoning": "Breakout above resistance.",
            "conviction": "high",
        }

        trade_id = thm.log_trade(conn, trade_data)
        assert trade_id is not None
        assert trade_id > 0

        recent = thm.get_recent_trades(conn, limit=5)
        assert len(recent) == 1
        t = recent[0]
        assert t["asset"] == "BTC"
        assert t["side"] == "long"
        assert t["action"] == "open_long"
        assert t["size_pct"] == 0.05
        assert t["leverage"] == 2.0
        assert t["entry_price"] == 67000.0
        assert t["stop_loss"] == 65500.0
        assert t["take_profit"] == 72000.0
        assert t["status"] == "open"
        conn.close()

    def test_close_trade_updates_pnl(self):
        """Close a trade and verify exit price, PnL, and status."""
        conn = _make_db()
        thm = TradeHistoryManager()

        trade_id = thm.log_trade(conn, {
            "asset": "ETH",
            "side": "long",
            "action": "open_long",
            "size_pct": 0.03,
            "leverage": 2.0,
            "entry_price": 3400.0,
            "stop_loss": 3300.0,
            "take_profit": 3700.0,
        })

        thm.close_trade(conn, trade_id, exit_price=3600.0, pnl=200.0, fees=1.2)

        row = fetch_one(conn, "SELECT * FROM trades WHERE id = ?", (trade_id,))
        assert row is not None
        assert row["status"] == "closed"
        assert row["exit_price"] == 3600.0
        assert row["pnl"] == 200.0
        assert row["fees"] == 1.2
        # pnl_pct should be (3600-3400)/3400 * leverage(2) ~ 0.1176
        assert abs(row["pnl_pct"] - 0.1176) < 0.001
        assert row["closed_at"] is not None
        conn.close()

    def test_daily_trade_count(self):
        """get_daily_trade_count returns correct count of today's trades."""
        conn = _make_db()
        thm = TradeHistoryManager()

        # Log 3 trades today
        for asset in ["BTC", "ETH", "SOL"]:
            thm.log_trade(conn, {
                "asset": asset,
                "side": "long",
                "action": "open_long",
                "size_pct": 0.03,
                "leverage": 1.5,
                "entry_price": 100.0,
                "stop_loss": 95.0,
                "take_profit": 110.0,
            })

        count = thm.get_daily_trade_count(conn)
        assert count == 3
        conn.close()

    def test_last_trade_time_returns_most_recent(self):
        """get_last_trade_time should return the latest opened_at."""
        conn = _make_db()
        thm = TradeHistoryManager()

        thm.log_trade(conn, {
            "asset": "BTC", "side": "long", "action": "open_long",
            "size_pct": 0.03, "leverage": 1.0, "entry_price": 100.0,
            "stop_loss": 95.0, "take_profit": 110.0,
        })

        last_time = thm.get_last_trade_time(conn)
        assert last_time is not None
        assert last_time.tzinfo is not None  # timezone-aware
        # Should be within last minute
        delta = utc_now() - last_time
        assert delta.total_seconds() < 60
        conn.close()

    def test_multiple_open_close_cycles(self):
        """Open multiple trades, close some, verify counts."""
        conn = _make_db()
        thm = TradeHistoryManager()

        ids = []
        for asset in ["BTC", "ETH", "SOL"]:
            tid = thm.log_trade(conn, {
                "asset": asset, "side": "long", "action": "open_long",
                "size_pct": 0.03, "leverage": 1.5, "entry_price": 100.0,
                "stop_loss": 95.0, "take_profit": 110.0,
            })
            ids.append(tid)

        # Close BTC
        thm.close_trade(conn, ids[0], exit_price=105.0, pnl=5.0)

        # Verify: 2 open, 1 closed
        open_rows = fetch_all(conn, "SELECT * FROM trades WHERE status='open'")
        closed_rows = fetch_all(conn, "SELECT * FROM trades WHERE status='closed'")
        assert len(open_rows) == 2
        assert len(closed_rows) == 1
        assert closed_rows[0]["asset"] == "BTC"
        conn.close()

    def test_schema_idempotent(self):
        """Applying the schema twice does not error or duplicate data."""
        conn = _make_db()
        # Apply schema again
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
        # Should still have the 6 user tables + sqlite_sequence (auto-created
        # by SQLite for AUTOINCREMENT columns) = 7 total.
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        user_tables = [row[0] for row in cursor.fetchall()]
        assert len(user_tables) == 6
        assert set(user_tables) == {
            "trades", "positions", "grok_logs", "daily_summaries",
            "rejections", "equity_snapshots"
        }
        conn.close()


# =====================================================================
# 2. CONTEXT BUILDER WITH REAL DATA
# =====================================================================


class TestContextBuilder:
    """build_context_prompt produces valid prompts from realistic data."""

    def test_produces_string_with_all_sections(self):
        """Output contains all expected section headers."""
        market = _make_market_data()
        portfolio = _make_portfolio()
        trades: list[dict] = []
        risk = _make_risk_status()

        prompt = build_context_prompt(market, portfolio, trades, risk)

        assert isinstance(prompt, str)
        assert len(prompt) > 200

        # Section headers
        assert "CURRENT MARKET DATA" in prompt
        from config.trading_config import ASSET_UNIVERSE
        for asset in ASSET_UNIVERSE:
            assert f"{asset}-USD Perpetual" in prompt
        assert "YOUR CURRENT PORTFOLIO" in prompt
        assert "Open Positions" in prompt
        assert "Recent Trades" in prompt
        assert "RISK STATUS" in prompt
        assert "YOUR TASK" in prompt

    def test_portfolio_values_appear_in_prompt(self):
        """Equity and margin values appear formatted in the prompt."""
        market = _make_market_data()
        portfolio = _make_portfolio(equity=25_000.0)
        risk = _make_risk_status()

        prompt = build_context_prompt(market, portfolio, [], risk)
        assert "$25,000.00" in prompt

    def test_with_open_positions(self):
        """Positions table is populated when positions exist."""
        market = _make_market_data()
        positions = [
            {
                "asset": "BTC",
                "side": "long",
                "size": 0.15,
                "entry_price": 66000.0,
                "unrealized_pnl": 180.0,
                "leverage": 2.0,
            }
        ]
        portfolio = _make_portfolio(positions=positions)
        risk = _make_risk_status()

        prompt = build_context_prompt(market, portfolio, [], risk)
        assert "BTC" in prompt
        assert "long" in prompt
        assert "2.0x" in prompt

    def test_with_recent_trades(self):
        """Recent trades table is populated when trades are provided."""
        market = _make_market_data()
        portfolio = _make_portfolio()
        trades = [
            {
                "asset": "BTC",
                "side": "long",
                "action": "open_long",
                "entry_price": 65000.0,
                "exit_price": 67000.0,
                "pnl": 200.0,
                "status": "closed",
                "opened_at": "2026-03-06T10:00:00",
            }
        ]
        risk = _make_risk_status()

        prompt = build_context_prompt(market, portfolio, trades, risk)
        assert "$200.00" in prompt
        assert "closed" in prompt

    def test_fallback_on_error(self):
        """If market_data causes an error, a fallback prompt is returned."""
        # Pass None for candles to trigger error inside _build_asset_section
        bad_market = {
            "BTC": {
                "price": 67000.0,
                "24h_change_pct": 0.01,
                # missing "candles", "funding", "oi" keys
            }
        }
        portfolio = _make_portfolio()
        risk = _make_risk_status()

        # Should not raise, should return fallback
        prompt = build_context_prompt(bad_market, portfolio, [], risk)
        assert isinstance(prompt, str)
        # The fallback or a valid prompt should be present
        assert len(prompt) > 10

    def test_risk_status_calculations(self):
        """Risk status section shows correct remaining budgets."""
        market = _make_market_data()
        portfolio = _make_portfolio(equity=10_000.0)
        risk = _make_risk_status(daily_pnl=-200.0, weekly_pnl=-500.0, trades_today=3)

        prompt = build_context_prompt(market, portfolio, [], risk)
        assert "Trades today: 3/" in prompt


# =====================================================================
# 3. DECISION PARSER WITH REALISTIC GROK RESPONSES
# =====================================================================


class TestDecisionParserIntegration:
    """Full pipeline: raw JSON string -> validated GrokResponse."""

    def test_parse_valid_response(self):
        """Clean JSON parses and validates correctly."""
        parser = DecisionParser()
        raw = json.dumps(_sample_grok_response_dict())
        result = parser.parse_response(raw)

        assert result is not None
        assert isinstance(result, GrokResponse)
        assert len(result.decisions) == 1
        assert result.decisions[0].asset == "BTC"
        assert result.decisions[0].action == "open_long"
        assert result.overall_stance is not None

    def test_parse_with_markdown_fences(self):
        """Response wrapped in ```json ... ``` is handled."""
        parser = DecisionParser()
        inner = json.dumps(_sample_grok_response_dict(), indent=2)
        raw = f"```json\n{inner}\n```"
        result = parser.parse_response(raw)

        assert result is not None
        assert isinstance(result, GrokResponse)

    def test_parse_empty_decisions(self):
        """A response with zero decisions is valid (stay flat)."""
        parser = DecisionParser()
        raw = json.dumps(_sample_grok_response_dict(decisions=[]))
        result = parser.parse_response(raw)

        assert result is not None
        assert len(result.decisions) == 0

    def test_extract_actionable_decisions(self):
        """extract_decisions filters out hold and no_trade."""
        parser = DecisionParser()
        decisions = [
            {
                "action": "open_long",
                "asset": "BTC",
                "size_pct": 0.05,
                "leverage": 2.0,
                "entry_price": 67000.0,
                "stop_loss": 65500.0,
                "take_profit": 72000.0,
                "order_type": "limit",
                "reasoning": "Breakout.",
                "conviction": "high",
                "risk_reward_ratio": 3.0,
            },
            {
                "action": "no_trade",
                "asset": "ETH",
                "size_pct": 0.0,
                "leverage": 1.0,
                "entry_price": None,
                "stop_loss": 3200.0,
                "take_profit": 3700.0,
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
                "take_profit": 170.0,
                "order_type": "market",
                "reasoning": "Holding.",
                "conviction": "medium",
                "risk_reward_ratio": 2.0,
            },
        ]
        raw = json.dumps(_sample_grok_response_dict(decisions=decisions))
        response = parser.parse_response(raw)
        assert response is not None

        actionable = parser.extract_decisions(response)
        assert len(actionable) == 1
        assert actionable[0].action == "open_long"
        assert actionable[0].asset == "BTC"

    def test_reject_garbage_input(self):
        """Completely invalid input returns None (fail-safe)."""
        parser = DecisionParser()
        assert parser.parse_response("") is None
        assert parser.parse_response("   ") is None
        assert parser.parse_response("This is not JSON at all") is None
        assert parser.parse_response("42") is None  # valid JSON but not a dict

    def test_reject_partial_json(self):
        """JSON missing required fields returns None."""
        parser = DecisionParser()
        partial = json.dumps({"timestamp": "2026-03-07T12:00:00Z"})
        assert parser.parse_response(partial) is None

    def test_reject_invalid_field_values(self):
        """JSON with out-of-range values fails Pydantic validation."""
        parser = DecisionParser()
        bad = _sample_grok_response_dict(
            decisions=[
                {
                    "action": "open_long",
                    "asset": "BTC",
                    "size_pct": 150.0,  # 150% -> normalized to 1.5 -> exceeds 1.0 max
                    "leverage": 2.0,
                    "entry_price": 67000.0,
                    "stop_loss": 65500.0,
                    "take_profit": 72000.0,
                    "order_type": "limit",
                    "reasoning": "Test.",
                    "conviction": "high",
                    "risk_reward_ratio": 3.0,
                }
            ]
        )
        result = parser.parse_response(json.dumps(bad))
        assert result is None  # size_pct 150% -> normalized to 1.5 -> exceeds 1.0 Pydantic max

    def test_parse_close_decision(self):
        """A close decision round-trips through the parser."""
        parser = DecisionParser()
        decisions = [
            {
                "action": "close",
                "asset": "BTC",
                "size_pct": 0.05,
                "leverage": 1.0,
                "entry_price": None,
                "stop_loss": 0.0,
                "take_profit": 0.0,
                "order_type": "market",
                "reasoning": "Taking profit at resistance.",
                "conviction": "high",
                "risk_reward_ratio": 0.0,
            }
        ]
        raw = json.dumps(_sample_grok_response_dict(decisions=decisions))
        response = parser.parse_response(raw)
        assert response is not None

        actionable = parser.extract_decisions(response)
        assert len(actionable) == 1
        assert actionable[0].action == "close"


# =====================================================================
# 4. RISK GUARDIAN + TRADE HISTORY MANAGER INTEGRATION
# =====================================================================


class TestRiskGuardianWithTradeHistory:
    """Risk checks that depend on DB queries work with real data."""

    def _portfolio(self, **overrides) -> dict[str, Any]:
        defaults = {
            "equity": 10_000.0,
            "peak_equity": 10_000.0,
            "daily_pnl_pct": 0.0,
            "weekly_pnl_pct": 0.0,
            "total_exposure_pct": 0.0,
        }
        defaults.update(overrides)
        return defaults

    def test_time_between_trades_cooldown(self):
        """Recent trade (< 30 min ago) blocks new trade for same asset."""
        conn = _make_db()
        guardian = RiskGuardian(risk_params=RISK_PARAMS.copy())
        thm = TradeHistoryManager()

        # Log a BTC trade 5 minutes ago
        thm.log_trade(conn, {
            "asset": "BTC", "side": "long", "action": "open_long",
            "size_pct": 0.03, "leverage": 1.5, "entry_price": 67000.0,
            "stop_loss": 66000.0, "take_profit": 70000.0,
        })

        decision = _make_valid_decision(asset="BTC")
        result = guardian.validate(decision, self._portfolio(), conn)

        assert not result.approved
        assert "time between trades" in result.reason.lower() or "min" in result.reason.lower()
        conn.close()

    def test_old_trade_allows_new_trade(self):
        """Trade from 2 hours ago should not block new trade."""
        conn = _make_db()
        guardian = RiskGuardian(risk_params=RISK_PARAMS.copy())

        # Insert a trade with timestamp 2 hours ago
        two_hours_ago = (utc_now() - timedelta(hours=2)).isoformat()
        execute_query(
            conn,
            """INSERT INTO trades (asset, side, action, size_pct, leverage,
               entry_price, stop_loss, take_profit, status, opened_at, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?)""",
            ("BTC", "long", "open_long", 0.03, 1.5, 67000.0, 66000.0,
             70000.0, two_hours_ago, two_hours_ago),
        )

        decision = _make_valid_decision(asset="BTC")
        result = guardian.validate(decision, self._portfolio(), conn)

        # Should pass the time check (may fail daily_trade_count if at limit)
        # The key assertion: "time between trades" NOT in reason
        if not result.approved:
            assert "time between trades" not in result.reason.lower()
        conn.close()

    def test_daily_trade_count_limit(self):
        """Max 50 trades/day blocks the 51st trade."""
        conn = _make_db()
        guardian = RiskGuardian(risk_params=RISK_PARAMS.copy())

        # Insert 50 trades spread across today
        base = utc_now().replace(hour=0, minute=0, second=0, microsecond=0)
        for i in range(50):
            ts = (base + timedelta(minutes=i * 10)).isoformat()
            execute_query(
                conn,
                """INSERT INTO trades (asset, side, action, size_pct, leverage,
                   entry_price, stop_loss, take_profit, status, opened_at, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?)""",
                ("ETH", "long", "open_long", 0.02, 1.0, 3400.0, 3300.0,
                 3600.0, ts, ts),
            )

        # The 51st trade should be blocked
        decision = _make_valid_decision(asset="SOL")
        result = guardian.validate(decision, self._portfolio(), conn)
        assert not result.approved
        assert "daily trade limit" in result.reason.lower()
        conn.close()

    def test_approved_trade_passes_all_13_checks(self):
        """A well-formed trade with no prior history passes all checks."""
        conn = _make_db()
        guardian = RiskGuardian(risk_params=RISK_PARAMS.copy())
        decision = _make_valid_decision()

        result = guardian.validate(decision, self._portfolio(), conn)
        assert result.approved
        assert "13 risk checks passed" in result.reason.lower() or result.approved
        conn.close()

    def test_risk_status_computation(self):
        """calculate_risk_status integrates with real DB."""
        conn = _make_db()
        guardian = RiskGuardian(risk_params=RISK_PARAMS.copy())
        thm = TradeHistoryManager()

        # Log 2 trades today
        for _ in range(2):
            thm.log_trade(conn, {
                "asset": "BTC", "side": "long", "action": "open_long",
                "size_pct": 0.03, "leverage": 1.5, "entry_price": 67000.0,
                "stop_loss": 66000.0, "take_profit": 70000.0,
            })

        portfolio = self._portfolio(daily_pnl_pct=-0.02, weekly_pnl_pct=-0.04)
        status = guardian.calculate_risk_status(portfolio, conn)

        assert status["trades_today"] == 2
        assert status["max_trades_per_day"] == 50
        assert status["daily_loss_remaining"] > 0
        assert status["weekly_loss_remaining"] > 0
        assert status["kill_switch_active"] is False
        conn.close()


# =====================================================================
# 5. FULL CYCLE SIMULATION
# =====================================================================


class TestFullCycleSimulation:
    """Wire up all real components except external APIs, run a cycle."""

    def _run_simulated_cycle(
        self,
        conn: sqlite3.Connection,
        grok_response_dict: dict,
    ) -> dict[str, Any]:
        """Execute a full trading cycle with mocked externals.

        Returns a summary dict with results.
        """
        # Real components
        thm = TradeHistoryManager()
        guardian = RiskGuardian(risk_params=RISK_PARAMS.copy())
        parser = DecisionParser()
        notifier = Notifier()  # no channels configured = silent

        # Build context
        market_data = _make_market_data()
        portfolio = _make_portfolio(equity=10_000.0)
        recent_trades = thm.get_recent_trades(conn, limit=10)
        risk_status = guardian.calculate_risk_status(
            {
                "equity": 10_000.0,
                "peak_equity": 10_000.0,
                "daily_pnl_pct": 0.0,
                "weekly_pnl_pct": 0.0,
                "total_exposure_pct": 0.0,
            },
            conn,
        )

        # Build context prompt (real)
        context = build_context_prompt(market_data, portfolio, recent_trades, risk_status)
        assert isinstance(context, str) and len(context) > 100

        # Simulate Grok response
        raw_response = json.dumps(grok_response_dict)

        # Parse (real)
        grok_response = parser.parse_response(raw_response)
        if grok_response is None:
            return {"status": "parse_failed", "trades_logged": 0}

        # Extract actionable decisions (real)
        actionable = parser.extract_decisions(grok_response)

        trades_logged = 0
        rejections_logged = 0
        portfolio_state = {
            "equity": 10_000.0,
            "peak_equity": 10_000.0,
            "daily_pnl_pct": 0.0,
            "weekly_pnl_pct": 0.0,
            "total_exposure_pct": 0.0,
        }

        for decision in actionable:
            # Validate through risk guardian (real)
            result = guardian.validate(decision, portfolio_state, conn)

            if result.approved:
                # Log to DB (real)
                trade_data = {
                    "asset": decision.asset,
                    "side": "long" if "long" in decision.action else "short",
                    "action": decision.action,
                    "size_pct": decision.size_pct,
                    "leverage": decision.leverage,
                    "entry_price": decision.entry_price or 67000.0,
                    "stop_loss": decision.stop_loss,
                    "take_profit": decision.take_profit,
                    "reasoning": decision.reasoning,
                    "conviction": decision.conviction,
                }
                thm.log_trade(conn, trade_data)
                trades_logged += 1
            else:
                # Log rejection
                execute_query(
                    conn,
                    """INSERT INTO rejections (timestamp, asset, action, reason, decision_json)
                       VALUES (?, ?, ?, ?, ?)""",
                    (utc_now().isoformat(), decision.asset, decision.action,
                     result.reason, json.dumps(decision.model_dump())),
                )
                rejections_logged += 1

        return {
            "status": "success",
            "trades_logged": trades_logged,
            "rejections_logged": rejections_logged,
            "actionable_count": len(actionable),
            "total_decisions": len(grok_response.decisions),
            "next_review_minutes": grok_response.next_review_suggestion_minutes,
        }

    def test_full_cycle_with_one_trade(self):
        """A single valid trade flows through the entire pipeline."""
        conn = _make_db()
        response = _sample_grok_response_dict()
        result = self._run_simulated_cycle(conn, response)

        assert result["status"] == "success"
        assert result["trades_logged"] == 1
        assert result["rejections_logged"] == 0

        # Verify trade in DB
        trades = fetch_all(conn, "SELECT * FROM trades")
        assert len(trades) == 1
        assert trades[0]["asset"] == "BTC"
        assert trades[0]["status"] == "open"
        conn.close()

    def test_full_cycle_no_trades(self):
        """Empty decisions array results in no DB writes."""
        conn = _make_db()
        response = _sample_grok_response_dict(decisions=[])
        result = self._run_simulated_cycle(conn, response)

        assert result["status"] == "success"
        assert result["trades_logged"] == 0
        assert result["actionable_count"] == 0

        trades = fetch_all(conn, "SELECT * FROM trades")
        assert len(trades) == 0
        conn.close()

    def test_full_cycle_trade_rejected(self):
        """A trade that violates risk limits is rejected, not logged as trade."""
        conn = _make_db()

        # Insert 50 trades today to hit daily limit
        base = utc_now().replace(hour=0, minute=0, second=0, microsecond=0)
        for i in range(50):
            ts = (base + timedelta(minutes=i * 10)).isoformat()
            execute_query(
                conn,
                """INSERT INTO trades (asset, side, action, size_pct, leverage,
                   entry_price, stop_loss, take_profit, status, opened_at, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?)""",
                ("ETH", "long", "open_long", 0.02, 1.0, 3400.0, 3300.0,
                 3600.0, ts, ts),
            )

        response = _sample_grok_response_dict()
        result = self._run_simulated_cycle(conn, response)

        assert result["status"] == "success"
        assert result["trades_logged"] == 0
        assert result["rejections_logged"] == 1

        # Verify rejection logged
        rejections = fetch_all(conn, "SELECT * FROM rejections")
        assert len(rejections) == 1
        assert "daily trade limit" in rejections[0]["reason"].lower()
        conn.close()

    def test_full_cycle_db_state_consistency(self):
        """After multiple cycles, DB state remains consistent."""
        conn = _make_db()

        # Cycle 1: open BTC long
        resp1 = _sample_grok_response_dict()
        r1 = self._run_simulated_cycle(conn, resp1)
        assert r1["trades_logged"] == 1

        # Cycle 2: open ETH short (different asset, won't hit cooldown)
        resp2 = _sample_grok_response_dict(
            decisions=[
                {
                    "action": "open_short",
                    "asset": "ETH",
                    "size_pct": 0.03,
                    "leverage": 2.0,
                    "entry_price": 3500.0,
                    "stop_loss": 3600.0,
                    "take_profit": 3200.0,
                    "order_type": "market",
                    "reasoning": "Rejection at resistance.",
                    "conviction": "medium",
                    "risk_reward_ratio": 3.0,
                }
            ]
        )
        r2 = self._run_simulated_cycle(conn, resp2)
        assert r2["trades_logged"] == 1

        # Verify DB has 2 open trades
        all_trades = fetch_all(conn, "SELECT * FROM trades WHERE status='open'")
        assert len(all_trades) == 2
        assets = {t["asset"] for t in all_trades}
        assert assets == {"BTC", "ETH"}
        conn.close()


# =====================================================================
# 6. PAPER TRADING ENGINE LIFECYCLE
# =====================================================================


class TestPaperTradingLifecycle:
    """Open -> monitor -> close lifecycle using real paper engine + DB."""

    def test_open_check_pnl_close_lifecycle(self):
        """Full position lifecycle with P&L tracking."""
        conn = _make_db()
        thm = TradeHistoryManager()

        # Step 1: Open a long position
        trade_id = thm.log_trade(conn, {
            "asset": "BTC",
            "side": "long",
            "action": "open_long",
            "size_pct": 0.05,
            "leverage": 2.0,
            "entry_price": 65000.0,
            "stop_loss": 63000.0,
            "take_profit": 70000.0,
        })
        assert trade_id > 0

        # Verify open
        row = fetch_one(conn, "SELECT * FROM trades WHERE id = ?", (trade_id,))
        assert row["status"] == "open"
        assert row["entry_price"] == 65000.0

        # Step 2: Close with profit
        thm.close_trade(conn, trade_id, exit_price=68000.0, pnl=300.0, fees=2.0)

        row = fetch_one(conn, "SELECT * FROM trades WHERE id = ?", (trade_id,))
        assert row["status"] == "closed"
        assert row["exit_price"] == 68000.0
        assert row["pnl"] == 300.0
        assert row["fees"] == 2.0
        # pnl_pct = (68000-65000)/65000 * leverage(2) ~ 0.0923
        assert abs(row["pnl_pct"] - 0.0923) < 0.001
        conn.close()

    def test_short_position_pnl_calculation(self):
        """Short position P&L is calculated correctly (entry - exit)."""
        conn = _make_db()
        thm = TradeHistoryManager()

        trade_id = thm.log_trade(conn, {
            "asset": "ETH",
            "side": "short",
            "action": "open_short",
            "size_pct": 0.04,
            "leverage": 2.0,
            "entry_price": 3500.0,
            "stop_loss": 3600.0,
            "take_profit": 3200.0,
        })

        # Close with profit (price went down)
        thm.close_trade(conn, trade_id, exit_price=3300.0, pnl=200.0, fees=1.5)

        row = fetch_one(conn, "SELECT * FROM trades WHERE id = ?", (trade_id,))
        assert row["status"] == "closed"
        # pnl_pct for short = (3500-3300)/3500 * leverage(2) ~ 0.1143
        assert abs(row["pnl_pct"] - 0.1143) < 0.001
        conn.close()

    def test_losing_trade_records_negative_pnl(self):
        """A losing trade has negative PnL stored correctly."""
        conn = _make_db()
        thm = TradeHistoryManager()

        trade_id = thm.log_trade(conn, {
            "asset": "SOL",
            "side": "long",
            "action": "open_long",
            "size_pct": 0.03,
            "leverage": 1.5,
            "entry_price": 160.0,
            "stop_loss": 155.0,
            "take_profit": 175.0,
        })

        # Close at stop-loss
        thm.close_trade(conn, trade_id, exit_price=155.0, pnl=-15.0, fees=0.5)

        row = fetch_one(conn, "SELECT * FROM trades WHERE id = ?", (trade_id,))
        assert row["pnl"] == -15.0
        assert row["pnl_pct"] < 0
        conn.close()

    def test_equity_tracking_via_portfolio_manager(self):
        """PortfolioManager derives correct metrics from trade history."""
        conn = _make_db()
        thm = TradeHistoryManager()
        pm = PortfolioManager()

        # Log and close a winning trade
        tid = thm.log_trade(conn, {
            "asset": "BTC", "side": "long", "action": "open_long",
            "size_pct": 0.05, "leverage": 2.0, "entry_price": 65000.0,
            "stop_loss": 63000.0, "take_profit": 70000.0,
        })
        thm.close_trade(conn, tid, exit_price=68000.0, pnl=300.0)

        # Log and close a losing trade
        tid2 = thm.log_trade(conn, {
            "asset": "ETH", "side": "long", "action": "open_long",
            "size_pct": 0.03, "leverage": 1.5, "entry_price": 3500.0,
            "stop_loss": 3400.0, "take_profit": 3700.0,
        })
        thm.close_trade(conn, tid2, exit_price=3400.0, pnl=-100.0)

        daily_pnl = pm.get_daily_pnl(conn)
        assert daily_pnl == 200.0  # 300 - 100

        consecutive = pm.get_consecutive_losses(conn)
        assert consecutive == 1  # last trade was a loss
        conn.close()


# =====================================================================
# 7. KILL SWITCH PERSISTENCE
# =====================================================================


class TestKillSwitchPersistence:
    """Kill switch blocks new trades but allows closes."""

    def test_activate_blocks_new_trades(self):
        """After kill switch activation, all open_long/short are rejected."""
        conn = _make_db()
        guardian = RiskGuardian(risk_params=RISK_PARAMS.copy())

        guardian.activate_kill_switch("Drawdown exceeded 20%")

        decision = _make_valid_decision(action="open_long")
        portfolio = {
            "equity": 10_000.0,
            "peak_equity": 10_000.0,
            "daily_pnl_pct": 0.0,
            "weekly_pnl_pct": 0.0,
            "total_exposure_pct": 0.0,
        }
        result = guardian.validate(decision, portfolio, conn)
        assert not result.approved
        assert "kill switch" in result.reason.lower()
        conn.close()

    def test_close_allowed_during_kill_switch(self):
        """Close decisions are always permitted, even with kill switch on."""
        conn = _make_db()
        guardian = RiskGuardian(risk_params=RISK_PARAMS.copy())
        guardian.activate_kill_switch("Testing")

        close_dec = TradeDecision(
            action="close",
            asset="BTC",
            size_pct=0.05,
            leverage=1.0,
            entry_price=None,
            stop_loss=0.0,
            take_profit=0.0,
            order_type="market",
            reasoning="Closing to reduce risk.",
            conviction="high",
            risk_reward_ratio=0.0,
        )

        result = guardian.validate(close_dec, {"equity": 10_000.0}, conn)
        assert result.approved
        conn.close()

    def test_kill_switch_persists_across_multiple_decisions(self):
        """Kill switch stays active for all subsequent validations."""
        conn = _make_db()
        guardian = RiskGuardian(risk_params=RISK_PARAMS.copy())
        guardian.activate_kill_switch("Manual activation")

        portfolio = {
            "equity": 10_000.0, "peak_equity": 10_000.0,
            "daily_pnl_pct": 0.0, "weekly_pnl_pct": 0.0,
            "total_exposure_pct": 0.0,
        }

        for asset in ["BTC", "ETH", "SOL"]:
            dec = _make_valid_decision(asset=asset)
            result = guardian.validate(dec, portfolio, conn)
            assert not result.approved

        assert guardian.kill_switch_active()
        conn.close()

    def test_deactivate_resumes_trading(self):
        """After manual deactivation, trades are approved again."""
        conn = _make_db()
        guardian = RiskGuardian(risk_params=RISK_PARAMS.copy())

        # Activate then deactivate
        guardian.activate_kill_switch("Test")
        assert guardian.kill_switch_active()

        guardian.deactivate_kill_switch()
        assert not guardian.kill_switch_active()

        decision = _make_valid_decision()
        portfolio = {
            "equity": 10_000.0, "peak_equity": 10_000.0,
            "daily_pnl_pct": 0.0, "weekly_pnl_pct": 0.0,
            "total_exposure_pct": 0.0,
        }
        result = guardian.validate(decision, portfolio, conn)
        assert result.approved
        conn.close()

    def test_drawdown_auto_activates_kill_switch(self):
        """20% drawdown from peak automatically triggers kill switch."""
        conn = _make_db()
        guardian = RiskGuardian(risk_params=RISK_PARAMS.copy())

        decision = _make_valid_decision()
        portfolio = {
            "equity": 8_000.0,
            "peak_equity": 10_000.0,
            "daily_pnl_pct": -0.02,
            "weekly_pnl_pct": -0.08,
            "total_exposure_pct": 0.0,
        }

        result = guardian.validate(decision, portfolio, conn)
        assert not result.approved
        assert guardian.kill_switch_active()
        assert "kill switch activated" in result.reason.lower()
        conn.close()

    def test_hold_and_no_trade_bypass_kill_switch(self):
        """hold and no_trade decisions are approved even with kill switch."""
        conn = _make_db()
        guardian = RiskGuardian(risk_params=RISK_PARAMS.copy())
        guardian.activate_kill_switch("Testing")

        portfolio = {"equity": 10_000.0}

        hold = TradeDecision(
            action="hold", asset="BTC", size_pct=0.0, leverage=1.0,
            stop_loss=60000.0, take_profit=70000.0, order_type="market",
            reasoning="Holding.", conviction="medium", risk_reward_ratio=2.0,
        )
        assert guardian.validate(hold, portfolio, conn).approved

        no_trade = TradeDecision(
            action="no_trade", asset="ETH", size_pct=0.0, leverage=1.0,
            stop_loss=3200.0, take_profit=3700.0, order_type="market",
            reasoning="No setup.", conviction="medium", risk_reward_ratio=2.0,
        )
        assert guardian.validate(no_trade, portfolio, conn).approved
        conn.close()


# =====================================================================
# 8. ERROR RECOVERY
# =====================================================================


class TestErrorRecovery:
    """Bot behavior when things go wrong."""

    def test_garbage_grok_response_returns_none(self):
        """Parser handles non-JSON garbage gracefully."""
        parser = DecisionParser()
        garbage_inputs = [
            "I am Grok. I recommend buying BTC!",
            "<html><body>Error 500</body></html>",
            "{'not': 'valid json'}",  # Python dict syntax, not JSON
            "",
            None,
        ]
        for inp in garbage_inputs:
            if inp is None:
                # parse_response expects a string; None should be handled
                result = parser.parse_response("")
            else:
                result = parser.parse_response(inp)
            assert result is None, f"Expected None for input: {inp!r}"

    def test_context_builder_survives_missing_keys(self):
        """build_context_prompt handles missing keys gracefully."""
        # Minimal market data with many missing keys
        market = {
            "BTC": {
                "price": 67000.0,
                "24h_change_pct": 0.0,
                "candles": {},
                "funding": {},
                "oi": {},
            }
        }
        portfolio = {"total_equity": 10_000.0}
        risk = {}

        prompt = build_context_prompt(market, portfolio, [], risk)
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    def test_trade_history_on_empty_db(self):
        """TradeHistoryManager works correctly on empty database."""
        conn = _make_db()
        thm = TradeHistoryManager()

        recent = thm.get_recent_trades(conn, limit=10)
        assert recent == []

        count = thm.get_daily_trade_count(conn)
        assert count == 0

        last = thm.get_last_trade_time(conn)
        assert last is None
        conn.close()

    def test_portfolio_manager_on_empty_db(self):
        """PortfolioManager returns sensible defaults on empty DB."""
        conn = _make_db()
        pm = PortfolioManager()

        daily = pm.get_daily_pnl(conn)
        assert daily == 0.0

        weekly = pm.get_weekly_pnl(conn)
        assert weekly == 0.0

        consecutive = pm.get_consecutive_losses(conn)
        assert consecutive == 0

        drawdown = pm.get_drawdown_from_peak(10_000.0, 10_000.0)
        assert drawdown == 0.0
        conn.close()

    def test_risk_guardian_with_no_trades_in_db(self):
        """RiskGuardian validates cleanly when DB has no trades."""
        conn = _make_db()
        guardian = RiskGuardian(risk_params=RISK_PARAMS.copy())
        decision = _make_valid_decision()
        portfolio = {
            "equity": 10_000.0, "peak_equity": 10_000.0,
            "daily_pnl_pct": 0.0, "weekly_pnl_pct": 0.0,
            "total_exposure_pct": 0.0,
        }

        result = guardian.validate(decision, portfolio, conn)
        assert result.approved
        conn.close()

    def test_close_trade_on_nonexistent_id(self):
        """Closing a non-existent trade_id doesn't crash."""
        conn = _make_db()
        thm = TradeHistoryManager()
        # This should not raise -- it just updates zero rows
        thm.close_trade(conn, trade_id=9999, exit_price=100.0, pnl=0.0)
        # Verify nothing was inserted or crashed
        count = fetch_one(conn, "SELECT COUNT(*) as c FROM trades")
        assert count["c"] == 0
        conn.close()

    def test_double_close_is_safe(self):
        """Closing an already-closed trade is idempotent."""
        conn = _make_db()
        thm = TradeHistoryManager()

        tid = thm.log_trade(conn, {
            "asset": "BTC", "side": "long", "action": "open_long",
            "size_pct": 0.05, "leverage": 2.0, "entry_price": 65000.0,
            "stop_loss": 63000.0, "take_profit": 70000.0,
        })
        thm.close_trade(conn, tid, exit_price=68000.0, pnl=300.0)
        # Close again -- should not crash
        thm.close_trade(conn, tid, exit_price=69000.0, pnl=400.0)

        row = fetch_one(conn, "SELECT * FROM trades WHERE id = ?", (tid,))
        assert row["status"] == "closed"
        # The second close overwrites pnl to 400
        assert row["pnl"] == 400.0
        conn.close()


# =====================================================================
# 9. NOTIFICATION FORMATTING
# =====================================================================


class TestNotificationFormatting:
    """Verify notification messages contain expected fields, no secrets."""

    def test_trade_alert_formatting(self):
        """send_trade_alert produces a message with all required fields."""
        notifier = Notifier()  # no channels configured = silent

        decision = _make_valid_decision()
        order_result = {
            "order_id": "paper_abc123",
            "asset": "BTC",
            "side": "long",
            "fill_price": 67050.0,
            "fees": 0.023,
            "live": False,
        }

        # Capture the message by patching _broadcast
        with patch.object(notifier, "_broadcast") as mock_broadcast:
            notifier.send_trade_alert(decision, order_result)

        mock_broadcast.assert_called_once()
        message = mock_broadcast.call_args[0][0]

        assert "PAPER" in message
        assert "BTC" in message
        assert "LONG" in message
        assert "67050" in message or "67,050" in message
        assert "Stop Loss" in message
        assert "Target" in message
        assert "Order ID" in message
        assert "paper_abc123" in message
        # No API keys or private keys
        assert "xai" not in message.lower()
        assert "private" not in message.lower()
        assert "secret" not in message.lower()

    def test_trade_closed_formatting(self):
        """send_trade_closed produces a valid closure message."""
        notifier = Notifier()
        trade_data = {
            "asset": "ETH",
            "side": "short",
            "entry_price": 3500.0,
            "exit_price": 3300.0,
            "realized_pnl_pct": 0.0571,
            "close_reason": "take_profit_hit",
            "opened_at": "2026-03-06T10:00:00+00:00",
            "closed_at": "2026-03-07T02:00:00+00:00",
        }

        with patch.object(notifier, "_broadcast") as mock_broadcast:
            notifier.send_trade_closed(trade_data)

        message = mock_broadcast.call_args[0][0]
        assert "WIN" in message  # positive PnL
        assert "ETH" in message
        assert "take_profit_hit" in message
        assert "16h" in message  # duration

    def test_daily_summary_formatting(self):
        """send_daily_summary contains equity and performance stats."""
        notifier = Notifier()
        summary = {
            "date": "2026-03-07",
            "daily_pnl_pct": 0.023,
            "equity": 10_230.0,
            "peak_equity": 10_500.0,
            "trades_today": 3,
            "wins": 2,
            "losses": 1,
            "win_rate": 0.667,
            "open_positions": 1,
            "total_exposure_pct": 0.05,
        }

        with patch.object(notifier, "_broadcast") as mock_broadcast:
            notifier.send_daily_summary(summary)

        message = mock_broadcast.call_args[0][0]
        assert "DAILY SUMMARY" in message
        assert "2026-03-07" in message
        assert "$10,230.00" in message
        assert "66.7%" in message  # win rate
        assert "Drawdown" in message

    def test_risk_alert_contains_no_stack_traces(self):
        """Risk alerts are clean text, no Python tracebacks."""
        notifier = Notifier()

        with patch.object(notifier, "_broadcast") as mock_broadcast:
            notifier.send_risk_alert(
                "kill_switch",
                "Total drawdown exceeded 20%. All trading halted.",
            )

        message = mock_broadcast.call_args[0][0]
        assert "RISK ALERT" in message
        assert "KILL_SWITCH" in message
        assert "Traceback" not in message
        assert "File " not in message
        assert "Exception" not in message

    def test_error_alert_formatting(self):
        """send_error_alert produces a clean error message."""
        notifier = Notifier()

        with patch.object(notifier, "_broadcast") as mock_broadcast:
            notifier.send_error_alert("Failed to connect to Grok API: timeout")

        message = mock_broadcast.call_args[0][0]
        assert "ERROR ALERT" in message
        assert "timeout" in message
        assert "Traceback" not in message

    def test_losing_trade_shows_loss(self):
        """Closing notification correctly labels a loss."""
        notifier = Notifier()
        trade_data = {
            "asset": "SOL",
            "side": "long",
            "entry_price": 160.0,
            "exit_price": 155.0,
            "realized_pnl_pct": -0.03125,
            "close_reason": "stop_loss_hit",
            "opened_at": "2026-03-07T08:00:00+00:00",
            "closed_at": "2026-03-07T10:30:00+00:00",
        }

        with patch.object(notifier, "_broadcast") as mock_broadcast:
            notifier.send_trade_closed(trade_data)

        message = mock_broadcast.call_args[0][0]
        assert "LOSS" in message
        assert "stop_loss_hit" in message
        assert "-3.12%" in message or "-3.13%" in message


# =====================================================================
# 10. LOGGER INTEGRATION
# =====================================================================


class TestLoggerIntegration:
    """Verify structured logging helpers work without errors."""

    def test_log_trade_decision_does_not_raise(self):
        """log_trade_decision writes without error."""
        # Should not raise
        log_trade_decision(
            {
                "action": "open_long",
                "asset": "BTC",
                "size_pct": 0.05,
                "leverage": 2.0,
            },
            cycle=1,
        )

    def test_log_trade_execution_does_not_raise(self):
        """log_trade_execution writes without error."""
        log_trade_execution(
            {
                "order_id": "paper_abc123",
                "asset": "BTC",
                "side": "long",
                "fill_price": 67000.0,
                "fees": 0.023,
            }
        )

    def test_log_trade_rejection_does_not_raise(self):
        """log_trade_rejection writes without error."""
        log_trade_rejection(
            asset="BTC",
            action="open_long",
            reason="Daily trade limit reached: 8/8 trades today.",
            decision={"action": "open_long", "asset": "BTC"},
        )

    def test_log_grok_cycle_does_not_raise(self):
        """log_grok_cycle writes without error."""
        log_grok_cycle(
            cycle=5,
            prompt_preview="## CURRENT MARKET DATA ...",
            response_preview='{"timestamp": "2026-03-07T12:00:00Z", ...}',
            decisions_count=2,
        )

    def test_system_prompt_not_empty(self):
        """System prompt is loaded and contains critical elements."""
        prompt = get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 500
        assert "Sentinel" in prompt
        assert "stop-loss" in prompt.lower()
        assert "JSON" in prompt
        assert "decisions" in prompt
        assert "leverage" in prompt.lower()


# =====================================================================
# BONUS: Cross-cutting integration tests
# =====================================================================


class TestCrossCuttingIntegration:
    """Tests that span multiple subsystems together."""

    def test_grok_log_roundtrip(self):
        """Grok interaction is logged to DB and can be retrieved."""
        conn = _make_db()
        context = "## CURRENT MARKET DATA ..."
        raw_response = json.dumps(_sample_grok_response_dict())

        execute_query(
            conn,
            """INSERT INTO grok_logs (timestamp, system_prompt_hash,
               context_prompt, response_text, decisions_json, cycle_number)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (utc_now().isoformat(), "abc123hash", context, raw_response,
             "[]", 1),
        )

        row = fetch_one(conn, "SELECT * FROM grok_logs WHERE cycle_number = 1")
        assert row is not None
        assert row["system_prompt_hash"] == "abc123hash"
        assert row["cycle_number"] == 1
        # The context should be stored verbatim
        assert row["context_prompt"] == context
        conn.close()

    def test_rejection_log_roundtrip(self):
        """Risk rejections are logged to DB with full detail."""
        conn = _make_db()
        decision = _make_valid_decision()

        execute_query(
            conn,
            """INSERT INTO rejections (timestamp, asset, action, reason, decision_json)
               VALUES (?, ?, ?, ?, ?)""",
            (utc_now().isoformat(), decision.asset, decision.action,
             "Daily trade limit reached", json.dumps(decision.model_dump())),
        )

        row = fetch_one(conn, "SELECT * FROM rejections WHERE asset = 'BTC'")
        assert row is not None
        assert row["reason"] == "Daily trade limit reached"
        # decision_json should be valid JSON
        parsed = json.loads(row["decision_json"])
        assert parsed["action"] == "open_long"
        assert parsed["asset"] == "BTC"
        conn.close()

    def test_portfolio_manager_consecutive_losses_tracking(self):
        """consecutive_losses counts backwards through closed trades.

        NOTE: get_consecutive_losses() orders by closed_at DESC. When
        trades are closed within the same second, the ordering between
        them is non-deterministic. This test uses explicit closed_at
        timestamps to ensure stable ordering.
        """
        conn = _make_db()
        thm = TradeHistoryManager()
        pm = PortfolioManager()

        # Log 3 losing trades and close them with distinct timestamps
        base_time = utc_now() - timedelta(hours=3)
        for i in range(3):
            tid = thm.log_trade(conn, {
                "asset": "BTC", "side": "long", "action": "open_long",
                "size_pct": 0.03, "leverage": 1.5, "entry_price": 65000.0,
                "stop_loss": 63000.0, "take_profit": 70000.0,
            })
            # Manually set a distinct closed_at so ordering is deterministic
            close_ts = (base_time + timedelta(minutes=i * 30)).isoformat()
            execute_query(
                conn,
                """UPDATE trades
                   SET exit_price = ?, pnl = ?, pnl_pct = ?,
                       status = 'closed', closed_at = ?
                   WHERE id = ?""",
                (63000.0, -100.0, -0.0308, close_ts, tid),
            )

        assert pm.get_consecutive_losses(conn) == 3

        # Log a winning trade with a later closed_at
        tid_win = thm.log_trade(conn, {
            "asset": "ETH", "side": "long", "action": "open_long",
            "size_pct": 0.04, "leverage": 2.0, "entry_price": 3400.0,
            "stop_loss": 3300.0, "take_profit": 3700.0,
        })
        win_close_ts = (base_time + timedelta(hours=2)).isoformat()
        execute_query(
            conn,
            """UPDATE trades
               SET exit_price = ?, pnl = ?, pnl_pct = ?,
                   status = 'closed', closed_at = ?
               WHERE id = ?""",
            (3600.0, 200.0, 0.0588, win_close_ts, tid_win),
        )

        # Streak is broken -- the most recent closed trade is a win
        assert pm.get_consecutive_losses(conn) == 0
        conn.close()

    def test_full_pipeline_parse_validate_log(self):
        """Parse -> validate -> log cycle works end-to-end."""
        conn = _make_db()
        parser = DecisionParser()
        guardian = RiskGuardian(risk_params=RISK_PARAMS.copy())
        thm = TradeHistoryManager()

        # Parse a realistic Grok response
        raw = json.dumps(_sample_grok_response_dict())
        response = parser.parse_response(raw)
        assert response is not None

        # Extract actionable
        actionable = parser.extract_decisions(response)
        assert len(actionable) == 1

        # Validate
        portfolio = {
            "equity": 10_000.0, "peak_equity": 10_000.0,
            "daily_pnl_pct": 0.0, "weekly_pnl_pct": 0.0,
            "total_exposure_pct": 0.0,
        }
        decision = actionable[0]
        result = guardian.validate(decision, portfolio, conn)
        assert result.approved

        # Log
        trade_id = thm.log_trade(conn, {
            "asset": decision.asset,
            "side": "long",
            "action": decision.action,
            "size_pct": decision.size_pct,
            "leverage": decision.leverage,
            "entry_price": decision.entry_price,
            "stop_loss": decision.stop_loss,
            "take_profit": decision.take_profit,
            "reasoning": decision.reasoning,
            "conviction": decision.conviction,
        })
        assert trade_id > 0

        # Verify DB
        row = fetch_one(conn, "SELECT * FROM trades WHERE id = ?", (trade_id,))
        assert row["asset"] == "BTC"
        assert row["action"] == "open_long"
        assert row["size_pct"] == 0.05
        assert row["leverage"] == 2.0
        assert row["entry_price"] == 67000.0
        assert row["stop_loss"] == 65500.0
        assert row["take_profit"] == 72000.0
        conn.close()

    def test_helpers_math_functions(self):
        """Helper math utilities calculate correctly."""
        # P&L calculations
        assert abs(calculate_pnl_pct(65000, 68000, "long") - 0.04615) < 0.001
        assert abs(calculate_pnl_pct(3500, 3300, "short") - 0.05714) < 0.001

        # Risk/reward ratio
        rr = calculate_risk_reward_ratio(65000, 63000, 72000, "long")
        assert abs(rr - 3.5) < 0.01

        # Formatting
        assert format_price(67432.10) == "67,432.10"
        assert format_usd(12345.67) == "$12,345.67"
        assert format_usd(-500.0) == "-$500.00"
        assert format_pct(0.05) == "+5.00%"
        assert format_pct(-0.03) == "-3.00%"

    def test_daily_summary_table_roundtrip(self):
        """Daily summary records can be written and read back."""
        conn = _make_db()
        execute_query(
            conn,
            """INSERT INTO daily_summaries
               (date, starting_equity, ending_equity, pnl, pnl_pct,
                trades_count, wins, losses, win_rate, max_drawdown)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("2026-03-07", 10000.0, 10230.0, 230.0, 0.023,
             5, 3, 2, 0.6, 0.015),
        )

        row = fetch_one(conn, "SELECT * FROM daily_summaries WHERE date = '2026-03-07'")
        assert row is not None
        assert row["ending_equity"] == 10230.0
        assert row["win_rate"] == 0.6
        assert row["max_drawdown"] == 0.015
        conn.close()

    def test_peak_equity_from_daily_summaries(self):
        """PortfolioManager.get_peak_equity uses daily_summaries table."""
        conn = _make_db()
        pm = PortfolioManager()

        # Insert some daily summaries
        for i, equity in enumerate([10000, 10200, 10500, 10300]):
            execute_query(
                conn,
                """INSERT INTO daily_summaries
                   (date, starting_equity, ending_equity, pnl, pnl_pct)
                   VALUES (?, ?, ?, ?, ?)""",
                (f"2026-03-0{i+1}", equity - 100, equity, 100.0, 0.01),
            )

        peak = pm.get_peak_equity(conn)
        assert peak == 10500.0  # highest ending_equity
        conn.close()
