"""
PostgreSQL database helpers (Supabase).

Provides a thin wrapper around psycopg2 that preserves the same
interface used by the rest of the codebase (execute_query, fetch_one,
fetch_all, db_session).  The PgConnectionWrapper class translates
SQLite conventions (? placeholders, conn.execute()) to PostgreSQL.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Generator

import psycopg2
import psycopg2.extras

from utils.logger import logger
from config.trading_config import DATABASE_URL


# ═══════════════════════════════════════════════════════════════════════════
# PLACEHOLDER CONVERSION
# ═══════════════════════════════════════════════════════════════════════════

def _convert_placeholders(query: str) -> str:
    """Convert SQLite ``?`` positional placeholders to PostgreSQL ``%s``.

    This is safe because parameterised queries never contain literal ``?``
    characters in the SQL text itself — those values live in the params
    tuple.
    """
    return query.replace("?", "%s")


# ═══════════════════════════════════════════════════════════════════════════
# CONNECTION WRAPPER
# ═══════════════════════════════════════════════════════════════════════════

class PgConnectionWrapper:
    """Wraps a psycopg2 connection to provide an sqlite3-like interface.

    Allows existing code that calls ``db.execute(sql, params)`` to work
    unchanged against PostgreSQL.  Placeholder conversion and
    RealDictCursor creation happen transparently.
    """

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    # --- sqlite3-compatible interface ---

    def execute(self, query: str, params: tuple | dict = ()) -> Any:
        """Execute *query* and return a cursor (like ``sqlite3.Connection.execute``)."""
        pg_query = _convert_placeholders(query)
        cursor = self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute(pg_query, params)
        return cursor

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        self._conn.close()

    # --- pass-through for anything else ---

    def cursor(self, *args: Any, **kwargs: Any) -> Any:
        return self._conn.cursor(*args, **kwargs)

    @property
    def closed(self) -> bool:
        return self._conn.closed  # type: ignore[no-any-return]


# ═══════════════════════════════════════════════════════════════════════════
# DATABASE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def get_db_connection() -> PgConnectionWrapper:
    """Open a new PostgreSQL connection to Supabase.

    Returns a :class:`PgConnectionWrapper` so callers can use the same
    ``db.execute(sql, params)`` pattern they used with SQLite.
    """
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not set. Add your Supabase connection string "
            "to the .env file."
        )
    raw = psycopg2.connect(DATABASE_URL)
    return PgConnectionWrapper(raw)


@contextmanager
def db_session() -> Generator[PgConnectionWrapper, None, None]:
    """Context manager that yields a connection and auto-commits/rollbacks.

    Usage::

        with db_session() as conn:
            execute_query(conn, "INSERT INTO ...", (...))
    """
    conn = get_db_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str | None = None) -> None:
    """No-op — schema is managed via Supabase migrations.

    Kept for backwards compatibility with ``main.py`` which calls
    ``init_db()`` at startup.
    """
    logger.info("Database initialized (Supabase PostgreSQL)")


def execute_query(
    conn: PgConnectionWrapper,
    query: str,
    params: tuple[Any, ...] | dict[str, Any] = (),
) -> Any:
    """Execute a single SQL statement and return the cursor.

    Args:
        conn: An open database connection (PgConnectionWrapper).
        query: The SQL statement (may contain ``?`` placeholders).
        params: Positional or named parameters to bind.

    Returns:
        A psycopg2 cursor after execution.
    """
    pg_query = _convert_placeholders(query)
    try:
        cursor = conn._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute(pg_query, params)
        conn.commit()
        return cursor
    except Exception as exc:
        logger.error(
            "SQL error: {err} | query={q} | params={p}",
            err=exc,
            q=query[:200],
            p=params,
        )
        raise


def fetch_one(
    conn: PgConnectionWrapper,
    query: str,
    params: tuple[Any, ...] | dict[str, Any] = (),
) -> dict[str, Any] | None:
    """Execute a query and return the first row, or ``None``.

    Returns a dict (via RealDictCursor) so callers can use
    ``row["column_name"]`` access.
    """
    pg_query = _convert_placeholders(query)
    try:
        cursor = conn._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute(pg_query, params)
        return cursor.fetchone()
    except Exception as exc:
        logger.error(
            "SQL fetch_one error: {err} | query={q}",
            err=exc,
            q=query[:200],
        )
        raise


def fetch_all(
    conn: PgConnectionWrapper,
    query: str,
    params: tuple[Any, ...] | dict[str, Any] = (),
) -> list[dict[str, Any]]:
    """Execute a query and return all matching rows.

    Returns a list of dicts (via RealDictCursor).
    """
    pg_query = _convert_placeholders(query)
    try:
        cursor = conn._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute(pg_query, params)
        return cursor.fetchall()
    except Exception as exc:
        logger.error(
            "SQL fetch_all error: {err} | query={q}",
            err=exc,
            q=query[:200],
        )
        raise
