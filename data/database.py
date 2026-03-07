"""
SQLite database initialization and helper functions.

All persistent state lives here -- trades, positions, Grok interaction logs,
daily summaries, and risk-guardian rejections.  The schema is created
automatically on first run via :func:`init_db`.
"""

import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Generator

from utils.logger import logger
from config.trading_config import DB_PATH

# ═══════════════════════════════════════════════════════════════════════════
# SCHEMA DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

_SCHEMA_SQL = """
-- -----------------------------------------------------------------------
-- trades: full history of every opened/closed trade
-- -----------------------------------------------------------------------
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

-- -----------------------------------------------------------------------
-- positions: live / historical position snapshots
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS positions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    asset           TEXT    NOT NULL,
    side            TEXT    NOT NULL CHECK (side IN ('long', 'short')),
    size_pct        REAL    NOT NULL,
    leverage        REAL    NOT NULL DEFAULT 1.0,
    entry_price     REAL    NOT NULL,
    stop_loss       REAL,
    take_profit     REAL,
    unrealized_pnl  REAL    DEFAULT 0.0,
    opened_at       TEXT    NOT NULL DEFAULT (datetime('now')),
    status          TEXT    NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed'))
);

-- -----------------------------------------------------------------------
-- grok_logs: every prompt/response pair for auditability
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS grok_logs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           TEXT    NOT NULL DEFAULT (datetime('now')),
    system_prompt_hash  TEXT,
    context_prompt      TEXT    NOT NULL,
    response_text       TEXT    NOT NULL,
    decisions_json      TEXT,
    cycle_number        INTEGER
);

-- -----------------------------------------------------------------------
-- daily_summaries: aggregated end-of-day performance
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS daily_summaries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT    NOT NULL UNIQUE,
    starting_equity REAL,
    ending_equity   REAL,
    pnl             REAL,
    pnl_pct         REAL,
    trades_count    INTEGER DEFAULT 0,
    wins            INTEGER DEFAULT 0,
    losses          INTEGER DEFAULT 0,
    win_rate        REAL,
    max_drawdown    REAL
);

-- -----------------------------------------------------------------------
-- rejections: trades suggested by Grok but blocked by the risk guardian
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS rejections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL DEFAULT (datetime('now')),
    asset           TEXT    NOT NULL,
    action          TEXT    NOT NULL,
    reason          TEXT    NOT NULL,
    decision_json   TEXT
);

-- -----------------------------------------------------------------------
-- Indexes for common queries
-- -----------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_trades_status     ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_asset      ON trades(asset);
CREATE INDEX IF NOT EXISTS idx_trades_opened_at  ON trades(opened_at);
CREATE INDEX IF NOT EXISTS idx_trades_closed_at  ON trades(closed_at);
CREATE INDEX IF NOT EXISTS idx_positions_status  ON positions(status);
CREATE INDEX IF NOT EXISTS idx_positions_asset   ON positions(asset);
CREATE INDEX IF NOT EXISTS idx_grok_logs_cycle   ON grok_logs(cycle_number);
CREATE INDEX IF NOT EXISTS idx_rejections_ts     ON rejections(timestamp);
"""


# ═══════════════════════════════════════════════════════════════════════════
# DATABASE HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def get_db_connection(db_path: str | None = None) -> sqlite3.Connection:
    """Open a new SQLite connection with sensible defaults.

    Args:
        db_path: Path to the database file.  Defaults to the project-wide
                 ``DB_PATH`` from ``trading_config``.

    Returns:
        A ``sqlite3.Connection`` with row-factory set to ``sqlite3.Row``
        so that results can be accessed by column name.
    """
    path = db_path or DB_PATH

    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")      # better concurrency
    conn.execute("PRAGMA foreign_keys=ON")        # enforce FK constraints
    conn.execute("PRAGMA busy_timeout=5000")      # wait up to 5s on lock
    return conn


@contextmanager
def db_session(db_path: str | None = None) -> Generator[sqlite3.Connection, None, None]:
    """Context manager that yields a connection and auto-commits/rollbacks.

    Usage::

        with db_session() as conn:
            execute_query(conn, "INSERT INTO ...", (...))
    """
    conn = get_db_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str | None = None) -> None:
    """Create all tables and indexes if they don't already exist.

    Safe to call multiple times -- every statement uses
    ``CREATE TABLE IF NOT EXISTS``.

    Args:
        db_path: Optional override for the database file location.
    """
    conn = get_db_connection(db_path)
    try:
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
        logger.info("Database initialized at {path}", path=db_path or DB_PATH)
    except sqlite3.Error as exc:
        logger.error("Failed to initialize database: {err}", err=exc)
        raise
    finally:
        conn.close()


def execute_query(
    conn: sqlite3.Connection,
    query: str,
    params: tuple[Any, ...] | dict[str, Any] = (),
) -> sqlite3.Cursor:
    """Execute a single SQL statement and return the cursor.

    Args:
        conn: An open database connection.
        query: The SQL statement (may contain ``?`` or ``:name`` placeholders).
        params: Positional or named parameters to bind.

    Returns:
        The ``sqlite3.Cursor`` after execution.

    Raises:
        sqlite3.Error: Propagated after logging.
    """
    try:
        cursor = conn.execute(query, params)
        conn.commit()
        return cursor
    except sqlite3.Error as exc:
        logger.error(
            "SQL error: {err} | query={q} | params={p}",
            err=exc,
            q=query[:200],
            p=params,
        )
        raise


def fetch_one(
    conn: sqlite3.Connection,
    query: str,
    params: tuple[Any, ...] | dict[str, Any] = (),
) -> sqlite3.Row | None:
    """Execute a query and return the first row, or ``None``.

    Args:
        conn: An open database connection.
        query: The SQL query.
        params: Bind parameters.

    Returns:
        A ``sqlite3.Row`` or ``None`` if no rows match.
    """
    try:
        cursor = conn.execute(query, params)
        return cursor.fetchone()
    except sqlite3.Error as exc:
        logger.error(
            "SQL fetch_one error: {err} | query={q}",
            err=exc,
            q=query[:200],
        )
        raise


def fetch_all(
    conn: sqlite3.Connection,
    query: str,
    params: tuple[Any, ...] | dict[str, Any] = (),
) -> list[sqlite3.Row]:
    """Execute a query and return all matching rows.

    Args:
        conn: An open database connection.
        query: The SQL query.
        params: Bind parameters.

    Returns:
        A list of ``sqlite3.Row`` objects (may be empty).
    """
    try:
        cursor = conn.execute(query, params)
        return cursor.fetchall()
    except sqlite3.Error as exc:
        logger.error(
            "SQL fetch_all error: {err} | query={q}",
            err=exc,
            q=query[:200],
        )
        raise
