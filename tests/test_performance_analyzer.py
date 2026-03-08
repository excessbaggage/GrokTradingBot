"""
Comprehensive tests for the Trade Performance Analyzer.

Verifies all analysis methods produce correct calculations from known
trade data.  Uses in-memory SQLite databases with the real schema
from ``data.database``.  No network access or API keys needed.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from data.performance_analyzer import TradePerformanceAnalyzer
from data.context_builder import build_context_prompt


# ══════════════════════════════════════════════════════════════════════
# SCHEMA & HELPERS
# ══════════════════════════════════════════════════════════════════════

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL DEFAULT (datetime('now')),
    asset           TEXT    NOT NULL,
    side            TEXT    NOT NULL CHECK (side IN ('long', 'short')),
    action          TEXT    NOT NULL,
    size_pct        REAL    NOT NULL,
    leverage        REAL    NOT NULL DEFAULT 1.0,
    entry_price     REAL,
    exit_price      REAL,
    stop_loss       REAL,
    take_profit     REAL,
    pnl             REAL,
    pnl_pct         REAL,
    fees            REAL    DEFAULT 0.0,
    status          TEXT    NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed')),
    reasoning       TEXT,
    conviction      TEXT,
    opened_at       TEXT    NOT NULL DEFAULT (datetime('now')),
    closed_at       TEXT
);
"""


def _utc_iso(days_ago: int = 0, hours_ago: int = 0) -> str:
    """Return an ISO timestamp N days/hours ago in UTC."""
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago, hours=hours_ago)
    return dt.isoformat()


def _make_db() -> sqlite3.Connection:
    """Create an in-memory SQLite DB with the trades table."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def _insert_trade(
    conn: sqlite3.Connection,
    asset: str = "BTC",
    side: str = "long",
    action: str = "open_long",
    size_pct: float = 0.05,
    leverage: float = 2.0,
    entry_price: float = 65000.0,
    exit_price: float | None = None,
    stop_loss: float = 63000.0,
    take_profit: float = 70000.0,
    pnl: float | None = None,
    pnl_pct: float | None = None,
    status: str = "open",
    opened_at: str | None = None,
    closed_at: str | None = None,
) -> int:
    """Insert a trade and return its row ID."""
    if opened_at is None:
        opened_at = _utc_iso()
    now = opened_at

    cursor = conn.execute(
        """
        INSERT INTO trades
            (timestamp, asset, side, action, size_pct, leverage,
             entry_price, exit_price, stop_loss, take_profit,
             pnl, pnl_pct, status, opened_at, closed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            now, asset, side, action, size_pct, leverage,
            entry_price, exit_price, stop_loss, take_profit,
            pnl, pnl_pct, status, opened_at, closed_at,
        ),
    )
    conn.commit()
    return cursor.lastrowid


# ══════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture
def analyzer() -> TradePerformanceAnalyzer:
    return TradePerformanceAnalyzer(db_path=":memory:")


@pytest.fixture
def empty_db() -> sqlite3.Connection:
    conn = _make_db()
    yield conn
    conn.close()


@pytest.fixture
def mixed_db() -> sqlite3.Connection:
    """DB with a realistic mix of winners and losers across assets and sides."""
    conn = _make_db()

    # --- BTC long winners ---
    _insert_trade(
        conn, asset="BTC", side="long", action="open_long",
        size_pct=0.05, entry_price=65000, exit_price=67000,
        stop_loss=63000, take_profit=70000,
        pnl=200, pnl_pct=0.0308, status="closed",
        opened_at=_utc_iso(days_ago=5, hours_ago=10),
        closed_at=_utc_iso(days_ago=5, hours_ago=4),
    )
    _insert_trade(
        conn, asset="BTC", side="long", action="open_long",
        size_pct=0.06, entry_price=66000, exit_price=68000,
        stop_loss=64000, take_profit=71000,
        pnl=300, pnl_pct=0.0303, status="closed",
        opened_at=_utc_iso(days_ago=4, hours_ago=8),
        closed_at=_utc_iso(days_ago=4, hours_ago=2),
    )

    # --- BTC long loser ---
    _insert_trade(
        conn, asset="BTC", side="long", action="open_long",
        size_pct=0.04, entry_price=67000, exit_price=65000,
        stop_loss=64000, take_profit=72000,
        pnl=-150, pnl_pct=-0.0299, status="closed",
        opened_at=_utc_iso(days_ago=3, hours_ago=6),
        closed_at=_utc_iso(days_ago=3, hours_ago=1),
    )

    # --- ETH short winner ---
    _insert_trade(
        conn, asset="ETH", side="short", action="open_short",
        size_pct=0.04, entry_price=3500, exit_price=3300,
        stop_loss=3650, take_profit=3200,
        pnl=180, pnl_pct=0.0571, status="closed",
        opened_at=_utc_iso(days_ago=2, hours_ago=12),
        closed_at=_utc_iso(days_ago=2, hours_ago=6),
    )

    # --- ETH short loser ---
    _insert_trade(
        conn, asset="ETH", side="short", action="open_short",
        size_pct=0.03, entry_price=3400, exit_price=3550,
        stop_loss=3550, take_profit=3100,
        pnl=-120, pnl_pct=-0.0441, status="closed",
        opened_at=_utc_iso(days_ago=1, hours_ago=10),
        closed_at=_utc_iso(days_ago=1, hours_ago=4),
    )

    # --- SOL long loser ---
    _insert_trade(
        conn, asset="SOL", side="long", action="open_long",
        size_pct=0.03, entry_price=170, exit_price=160,
        stop_loss=155, take_profit=200,
        pnl=-80, pnl_pct=-0.0588, status="closed",
        opened_at=_utc_iso(days_ago=1, hours_ago=8),
        closed_at=_utc_iso(days_ago=1, hours_ago=3),
    )

    # --- SOL long loser (another) ---
    _insert_trade(
        conn, asset="SOL", side="long", action="open_long",
        size_pct=0.02, entry_price=165, exit_price=155,
        stop_loss=150, take_profit=190,
        pnl=-70, pnl_pct=-0.0606, status="closed",
        opened_at=_utc_iso(hours_ago=6),
        closed_at=_utc_iso(hours_ago=2),
    )

    # --- An OPEN trade (should be excluded from closed analysis) ---
    _insert_trade(
        conn, asset="BTC", side="long", action="open_long",
        size_pct=0.05, entry_price=69000,
        stop_loss=67000, take_profit=74000,
        status="open",
        opened_at=_utc_iso(hours_ago=1),
    )

    yield conn
    conn.close()


@pytest.fixture
def all_winners_db() -> sqlite3.Connection:
    """DB with only winning trades."""
    conn = _make_db()
    for i in range(5):
        _insert_trade(
            conn, asset="BTC", side="long", action="open_long",
            size_pct=0.05, entry_price=60000 + i * 1000,
            exit_price=62000 + i * 1000,
            stop_loss=58000 + i * 1000, take_profit=65000 + i * 1000,
            pnl=200, pnl_pct=0.033, status="closed",
            opened_at=_utc_iso(days_ago=10 - i, hours_ago=5),
            closed_at=_utc_iso(days_ago=10 - i, hours_ago=1),
        )
    yield conn
    conn.close()


@pytest.fixture
def all_losers_db() -> sqlite3.Connection:
    """DB with only losing trades."""
    conn = _make_db()
    for i in range(4):
        _insert_trade(
            conn, asset="ETH", side="short", action="open_short",
            size_pct=0.04, entry_price=3500,
            exit_price=3600,
            stop_loss=3600, take_profit=3200,
            pnl=-100, pnl_pct=-0.0286, status="closed",
            opened_at=_utc_iso(days_ago=5 - i, hours_ago=3),
            closed_at=_utc_iso(days_ago=5 - i, hours_ago=1),
        )
    yield conn
    conn.close()


@pytest.fixture
def single_trade_db() -> sqlite3.Connection:
    """DB with exactly one closed trade."""
    conn = _make_db()
    _insert_trade(
        conn, asset="BTC", side="long", action="open_long",
        size_pct=0.05, entry_price=65000, exit_price=67000,
        stop_loss=63000, take_profit=70000,
        pnl=200, pnl_pct=0.0308, status="closed",
        opened_at=_utc_iso(days_ago=1, hours_ago=10),
        closed_at=_utc_iso(days_ago=1, hours_ago=4),
    )
    yield conn
    conn.close()


# ══════════════════════════════════════════════════════════════════════
# TESTS: get_strategy_performance
# ══════════════════════════════════════════════════════════════════════


class TestStrategyPerformance:
    """Verify per-strategy (long vs short) aggregation."""

    def test_mixed_strategies(
        self, analyzer: TradePerformanceAnalyzer, mixed_db: sqlite3.Connection,
    ) -> None:
        result = analyzer.get_strategy_performance(mixed_db)

        # Longs: 2 winners (BTC +200, +300), 2 losers (BTC -150, SOL -80, -70)
        longs = result["long"]
        assert longs["count"] == 5  # 3 BTC + 2 SOL
        assert longs["total_pnl"] == pytest.approx(200 + 300 - 150 - 80 - 70, abs=0.01)

        # Shorts: 1 winner (ETH +180), 1 loser (ETH -120)
        shorts = result["short"]
        assert shorts["count"] == 2
        assert shorts["total_pnl"] == pytest.approx(180 - 120, abs=0.01)
        assert shorts["win_rate"] == pytest.approx(0.5, abs=0.01)

    def test_empty_db(
        self, analyzer: TradePerformanceAnalyzer, empty_db: sqlite3.Connection,
    ) -> None:
        result = analyzer.get_strategy_performance(empty_db)
        assert result["long"]["count"] == 0
        assert result["short"]["count"] == 0
        assert result["best_strategy"] is None

    def test_all_one_side(
        self, analyzer: TradePerformanceAnalyzer, all_winners_db: sqlite3.Connection,
    ) -> None:
        result = analyzer.get_strategy_performance(all_winners_db)
        assert result["long"]["count"] == 5
        assert result["long"]["win_rate"] == 1.0
        assert result["short"]["count"] == 0
        assert result["best_strategy"] == "long"


# ══════════════════════════════════════════════════════════════════════
# TESTS: get_asset_performance
# ══════════════════════════════════════════════════════════════════════


class TestAssetPerformance:
    """Verify per-asset performance breakdown."""

    def test_mixed_assets(
        self, analyzer: TradePerformanceAnalyzer, mixed_db: sqlite3.Connection,
    ) -> None:
        result = analyzer.get_asset_performance(mixed_db)

        # BTC: 2 wins, 1 loss = 66.7% win rate
        btc = result["BTC"]
        assert btc["count"] == 3
        assert btc["win_rate"] == pytest.approx(2 / 3, abs=0.01)

        # SOL: 0 wins, 2 losses = 0% win rate
        sol = result["SOL"]
        assert sol["count"] == 2
        assert sol["win_rate"] == 0.0

        assert result["best_asset"] == "BTC"
        assert result["worst_asset"] == "SOL"

    def test_hold_duration_calculated(
        self, analyzer: TradePerformanceAnalyzer, mixed_db: sqlite3.Connection,
    ) -> None:
        result = analyzer.get_asset_performance(mixed_db)
        # All test trades have known durations (opened_at to closed_at)
        for asset in ("BTC", "ETH", "SOL"):
            if asset in result and result[asset]["count"] > 0:
                assert result[asset]["avg_hold_hours"] > 0

    def test_empty_db(
        self, analyzer: TradePerformanceAnalyzer, empty_db: sqlite3.Connection,
    ) -> None:
        result = analyzer.get_asset_performance(empty_db)
        assert result["best_asset"] is None
        assert result["worst_asset"] is None


# ══════════════════════════════════════════════════════════════════════
# TESTS: get_time_performance
# ══════════════════════════════════════════════════════════════════════


class TestTimePerformance:
    """Verify time-based grouping (hour of day, day of week)."""

    def test_has_hour_entries(
        self, analyzer: TradePerformanceAnalyzer, mixed_db: sqlite3.Connection,
    ) -> None:
        result = analyzer.get_time_performance(mixed_db)
        assert len(result["by_hour"]) > 0

    def test_has_day_entries(
        self, analyzer: TradePerformanceAnalyzer, mixed_db: sqlite3.Connection,
    ) -> None:
        result = analyzer.get_time_performance(mixed_db)
        assert len(result["by_day"]) > 0

    def test_empty_db(
        self, analyzer: TradePerformanceAnalyzer, empty_db: sqlite3.Connection,
    ) -> None:
        result = analyzer.get_time_performance(empty_db)
        assert result["by_hour"] == {}
        assert result["by_day"] == {}
        assert result["best_hour"] is None
        assert result["worst_hour"] is None


# ══════════════════════════════════════════════════════════════════════
# TESTS: get_rr_accuracy
# ══════════════════════════════════════════════════════════════════════


class TestRRAccuracy:
    """Verify R:R prediction vs actual outcome analysis."""

    def test_mixed_rr(
        self, analyzer: TradePerformanceAnalyzer, mixed_db: sqlite3.Connection,
    ) -> None:
        result = analyzer.get_rr_accuracy(mixed_db)
        assert result["count"] > 0
        assert result["avg_predicted_rr"] > 0
        # Actual R:R can be negative (from losers), so just check it exists
        assert "avg_actual_rr" in result
        assert result["bias"] in (
            "accurate", "over-optimistic", "significantly_over-optimistic",
        )

    def test_all_winners_rr(
        self, analyzer: TradePerformanceAnalyzer, all_winners_db: sqlite3.Connection,
    ) -> None:
        result = analyzer.get_rr_accuracy(all_winners_db)
        # All winners: actual RR should be positive
        assert result["avg_actual_rr"] > 0
        assert result["count"] == 5

    def test_empty_db(
        self, analyzer: TradePerformanceAnalyzer, empty_db: sqlite3.Connection,
    ) -> None:
        result = analyzer.get_rr_accuracy(empty_db)
        assert result["count"] == 0
        assert result["bias"] == "insufficient_data"


# ══════════════════════════════════════════════════════════════════════
# TESTS: get_streak_analysis
# ══════════════════════════════════════════════════════════════════════


class TestStreakAnalysis:
    """Verify win/loss streak calculations."""

    def test_all_winners_streak(
        self, analyzer: TradePerformanceAnalyzer, all_winners_db: sqlite3.Connection,
    ) -> None:
        result = analyzer.get_streak_analysis(all_winners_db)
        assert result["current_streak"] == 5
        assert result["longest_win_streak"] == 5
        assert result["longest_loss_streak"] == 0
        assert result["total_trades"] == 5

    def test_all_losers_streak(
        self, analyzer: TradePerformanceAnalyzer, all_losers_db: sqlite3.Connection,
    ) -> None:
        result = analyzer.get_streak_analysis(all_losers_db)
        assert result["current_streak"] == -4
        assert result["longest_win_streak"] == 0
        assert result["longest_loss_streak"] == 4

    def test_mixed_streak(
        self, analyzer: TradePerformanceAnalyzer, mixed_db: sqlite3.Connection,
    ) -> None:
        result = analyzer.get_streak_analysis(mixed_db)
        # The last two trades in mixed_db are SOL losers
        assert result["current_streak"] < 0
        assert result["total_trades"] == 7  # 7 closed trades

    def test_empty_db(
        self, analyzer: TradePerformanceAnalyzer, empty_db: sqlite3.Connection,
    ) -> None:
        result = analyzer.get_streak_analysis(empty_db)
        assert result["current_streak"] == 0
        assert result["total_trades"] == 0

    def test_single_trade_streak(
        self, analyzer: TradePerformanceAnalyzer, single_trade_db: sqlite3.Connection,
    ) -> None:
        result = analyzer.get_streak_analysis(single_trade_db)
        assert result["current_streak"] == 1  # single winner
        assert result["longest_win_streak"] == 1
        assert result["total_trades"] == 1


# ══════════════════════════════════════════════════════════════════════
# TESTS: get_sizing_analysis
# ══════════════════════════════════════════════════════════════════════


class TestSizingAnalysis:
    """Verify position sizing comparison for winners vs losers."""

    def test_mixed_sizing(
        self, analyzer: TradePerformanceAnalyzer, mixed_db: sqlite3.Connection,
    ) -> None:
        result = analyzer.get_sizing_analysis(mixed_db)
        assert result["winners_count"] > 0
        assert result["losers_count"] > 0
        assert result["avg_winner_size"] > 0
        assert result["avg_loser_size"] > 0
        assert result["sizing_quality"] in ("good", "neutral", "bad")

    def test_all_winners_sizing(
        self, analyzer: TradePerformanceAnalyzer, all_winners_db: sqlite3.Connection,
    ) -> None:
        result = analyzer.get_sizing_analysis(all_winners_db)
        assert result["winners_count"] == 5
        assert result["losers_count"] == 0
        assert result["sizing_quality"] == "insufficient_data"

    def test_empty_db(
        self, analyzer: TradePerformanceAnalyzer, empty_db: sqlite3.Connection,
    ) -> None:
        result = analyzer.get_sizing_analysis(empty_db)
        assert result["winners_count"] == 0
        assert result["losers_count"] == 0

    def test_sizing_detects_bad_pattern(
        self, analyzer: TradePerformanceAnalyzer,
    ) -> None:
        """When losers are larger than winners, sizing_quality should be 'bad'."""
        conn = _make_db()
        # Winners with small size
        for i in range(3):
            _insert_trade(
                conn, asset="BTC", side="long", size_pct=0.02,
                entry_price=60000, exit_price=62000,
                stop_loss=58000, take_profit=65000,
                pnl=100, pnl_pct=0.033, status="closed",
                opened_at=_utc_iso(days_ago=5 - i, hours_ago=5),
                closed_at=_utc_iso(days_ago=5 - i, hours_ago=1),
            )
        # Losers with large size
        for i in range(3):
            _insert_trade(
                conn, asset="BTC", side="long", size_pct=0.08,
                entry_price=60000, exit_price=58000,
                stop_loss=58000, take_profit=65000,
                pnl=-200, pnl_pct=-0.033, status="closed",
                opened_at=_utc_iso(days_ago=3 - i, hours_ago=5),
                closed_at=_utc_iso(days_ago=3 - i, hours_ago=1),
            )

        result = analyzer.get_sizing_analysis(conn)
        assert result["sizing_quality"] == "bad"
        assert result["avg_loser_size"] > result["avg_winner_size"]
        conn.close()


# ══════════════════════════════════════════════════════════════════════
# TESTS: generate_performance_summary
# ══════════════════════════════════════════════════════════════════════


class TestPerformanceSummary:
    """Verify the formatted summary text generation."""

    def test_summary_with_data(
        self, analyzer: TradePerformanceAnalyzer, mixed_db: sqlite3.Connection,
    ) -> None:
        summary = analyzer.generate_performance_summary(mixed_db)
        assert isinstance(summary, str)
        assert len(summary) > 50
        assert "Strategy Performance" in summary
        assert "Asset Performance" in summary
        assert "Streak Status" in summary

    def test_summary_no_trades(
        self, analyzer: TradePerformanceAnalyzer, empty_db: sqlite3.Connection,
    ) -> None:
        summary = analyzer.generate_performance_summary(empty_db)
        assert "No closed trades" in summary

    def test_summary_single_trade(
        self, analyzer: TradePerformanceAnalyzer, single_trade_db: sqlite3.Connection,
    ) -> None:
        summary = analyzer.generate_performance_summary(single_trade_db)
        assert isinstance(summary, str)
        assert len(summary) > 20
        assert "BTC" in summary

    def test_summary_under_500_words(
        self, analyzer: TradePerformanceAnalyzer, mixed_db: sqlite3.Connection,
    ) -> None:
        summary = analyzer.generate_performance_summary(mixed_db)
        word_count = len(summary.split())
        assert word_count <= 500

    def test_summary_includes_actionable_advice_for_bad_asset(
        self, analyzer: TradePerformanceAnalyzer,
    ) -> None:
        """When an asset has very poor win rate, advice should mention it."""
        conn = _make_db()
        # SOL: 0 wins, 4 losses
        for i in range(4):
            _insert_trade(
                conn, asset="SOL", side="long", size_pct=0.03,
                entry_price=170, exit_price=160,
                stop_loss=155, take_profit=200,
                pnl=-50, pnl_pct=-0.06, status="closed",
                opened_at=_utc_iso(days_ago=5 - i, hours_ago=5),
                closed_at=_utc_iso(days_ago=5 - i, hours_ago=1),
            )
        # BTC: 3 wins for contrast
        for i in range(3):
            _insert_trade(
                conn, asset="BTC", side="long", size_pct=0.05,
                entry_price=65000, exit_price=67000,
                stop_loss=63000, take_profit=70000,
                pnl=200, pnl_pct=0.03, status="closed",
                opened_at=_utc_iso(days_ago=10 - i, hours_ago=3),
                closed_at=_utc_iso(days_ago=10 - i, hours_ago=1),
            )

        summary = analyzer.generate_performance_summary(conn)
        assert "SOL" in summary
        assert "0%" in summary or "avoiding" in summary.lower() or "reducing" in summary.lower()
        conn.close()


# ══════════════════════════════════════════════════════════════════════
# TESTS: Context Builder Integration
# ══════════════════════════════════════════════════════════════════════


class TestContextBuilderIntegration:
    """Verify that context_builder properly includes performance data."""

    def test_performance_section_included(self) -> None:
        """When performance_summary is provided, it appears in the prompt."""
        summary = "**Strategy Performance:** Long win rate 70%"
        prompt = build_context_prompt(
            market_data={},
            portfolio={"total_equity": 10000, "available_margin": 5000,
                       "unrealized_pnl": 0, "positions": []},
            recent_trades=[],
            risk_status={"daily_pnl": 0, "weekly_pnl": 0,
                         "drawdown_from_peak": 0, "trades_today": 0,
                         "consecutive_losses": 0},
            performance_summary=summary,
        )
        assert "PERFORMANCE ANALYTICS" in prompt
        assert "Long win rate 70%" in prompt

    def test_performance_section_omitted_when_empty(self) -> None:
        """When performance_summary is empty, the section is not in the prompt."""
        prompt = build_context_prompt(
            market_data={},
            portfolio={"total_equity": 10000, "available_margin": 5000,
                       "unrealized_pnl": 0, "positions": []},
            recent_trades=[],
            risk_status={"daily_pnl": 0, "weekly_pnl": 0,
                         "drawdown_from_peak": 0, "trades_today": 0,
                         "consecutive_losses": 0},
            performance_summary="",
        )
        assert "PERFORMANCE ANALYTICS" not in prompt

    def test_performance_section_between_risk_and_task(self) -> None:
        """Performance section should appear between RISK STATUS and YOUR TASK."""
        summary = "Test analytics content here"
        prompt = build_context_prompt(
            market_data={},
            portfolio={"total_equity": 10000, "available_margin": 5000,
                       "unrealized_pnl": 0, "positions": []},
            recent_trades=[],
            risk_status={"daily_pnl": 0, "weekly_pnl": 0,
                         "drawdown_from_peak": 0, "trades_today": 0,
                         "consecutive_losses": 0},
            performance_summary=summary,
        )
        risk_pos = prompt.index("RISK STATUS")
        perf_pos = prompt.index("PERFORMANCE ANALYTICS")
        task_pos = prompt.index("YOUR TASK")
        assert risk_pos < perf_pos < task_pos


# ══════════════════════════════════════════════════════════════════════
# TESTS: Edge Cases
# ══════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases: NULL values, open-only trades, old data outside lookback."""

    def test_open_trades_excluded(
        self, analyzer: TradePerformanceAnalyzer,
    ) -> None:
        """Open trades should not appear in closed-trade analysis."""
        conn = _make_db()
        _insert_trade(
            conn, asset="BTC", side="long", status="open",
            opened_at=_utc_iso(hours_ago=2),
        )
        result = analyzer.get_strategy_performance(conn)
        assert result["long"]["count"] == 0
        conn.close()

    def test_old_trades_outside_lookback(
        self, analyzer: TradePerformanceAnalyzer,
    ) -> None:
        """Trades older than lookback_days should be excluded."""
        conn = _make_db()
        _insert_trade(
            conn, asset="BTC", side="long", size_pct=0.05,
            entry_price=60000, exit_price=62000,
            stop_loss=58000, take_profit=65000,
            pnl=200, pnl_pct=0.033, status="closed",
            opened_at=_utc_iso(days_ago=60),
            closed_at=_utc_iso(days_ago=59),
        )

        result = analyzer.get_strategy_performance(conn, lookback_days=30)
        assert result["long"]["count"] == 0

        # But with a larger lookback, it should appear
        result = analyzer.get_strategy_performance(conn, lookback_days=90)
        assert result["long"]["count"] == 1
        conn.close()

    def test_null_pnl_excluded(
        self, analyzer: TradePerformanceAnalyzer,
    ) -> None:
        """Closed trades with NULL pnl should be excluded from analysis."""
        conn = _make_db()
        _insert_trade(
            conn, asset="BTC", side="long", status="closed",
            pnl=None, pnl_pct=None,
            opened_at=_utc_iso(days_ago=2),
            closed_at=_utc_iso(days_ago=1),
        )
        result = analyzer.get_strategy_performance(conn)
        assert result["long"]["count"] == 0
        conn.close()

    def test_zero_entry_price_excluded_from_rr(
        self, analyzer: TradePerformanceAnalyzer,
    ) -> None:
        """Trades with 0 entry_price should be excluded from R:R analysis."""
        conn = _make_db()
        _insert_trade(
            conn, asset="BTC", side="long", entry_price=0,
            exit_price=65000, stop_loss=0, take_profit=0,
            pnl=100, pnl_pct=0.01, status="closed",
            opened_at=_utc_iso(days_ago=2),
            closed_at=_utc_iso(days_ago=1),
        )
        result = analyzer.get_rr_accuracy(conn)
        assert result["count"] == 0
        conn.close()
