"""
Trade history management -- CRUD operations on the ``trades`` table.

Every trade opened or closed by the bot is recorded here for
auditability and performance tracking.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from utils.logger import logger
from utils.helpers import utc_now
from data.database import execute_query, fetch_one, fetch_all


class TradeHistoryManager:
    """Provides convenience methods for logging and querying trades.

    All methods are static -- no instance state is needed.  Pass an
    open database connection to each method.

    Usage::

        thm = TradeHistoryManager()
        trade_id = thm.log_trade(db, trade_data)
        thm.close_trade(db, trade_id, exit_price=68000, pnl=250, fees=2.5)
        recent = thm.get_recent_trades(db, limit=10)
    """

    @staticmethod
    def log_trade(db: Any, trade_data: dict[str, Any]) -> int:
        """Insert a new trade record into the database.

        Args:
            db: An open database connection.
            trade_data: Dict with trade details. Expected keys:

                - ``asset`` (str): e.g. ``"BTC"``
                - ``side`` (str): ``"long"`` or ``"short"``
                - ``action`` (str): e.g. ``"open_long"``, ``"open_short"``
                - ``size_pct`` (float): Position size as portfolio %
                - ``leverage`` (float): Leverage multiplier
                - ``entry_price`` (float): Entry price
                - ``stop_loss`` (float): Stop-loss price
                - ``take_profit`` (float): Take-profit price
                - ``reasoning`` (str, optional): Grok's reasoning
                - ``conviction`` (str, optional): Conviction level

        Returns:
            The ``id`` of the newly inserted trade row.
        """
        now = utc_now().isoformat()
        cursor = execute_query(
            db,
            """
            INSERT INTO trades
                (timestamp, asset, side, action, size_pct, leverage,
                 entry_price, stop_loss, take_profit, fees, status,
                 reasoning, conviction, opened_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?)
            RETURNING id
            """,
            (
                now,
                trade_data.get("asset", ""),
                trade_data.get("side", ""),
                trade_data.get("action", ""),
                float(trade_data.get("size_pct", 0)),
                float(trade_data.get("leverage", 1)),
                float(trade_data.get("entry_price", 0)),
                float(trade_data.get("stop_loss", 0)),
                float(trade_data.get("take_profit", 0)),
                float(trade_data.get("fees", 0)),
                trade_data.get("reasoning", ""),
                trade_data.get("conviction", ""),
                now,
            ),
        )
        row = cursor.fetchone()
        trade_id = row["id"] if row else None
        logger.info(
            "Trade logged | id={tid} asset={asset} side={side} action={act} "
            "size={sz}% entry={entry}",
            tid=trade_id,
            asset=trade_data.get("asset"),
            side=trade_data.get("side"),
            act=trade_data.get("action"),
            sz=trade_data.get("size_pct"),
            entry=trade_data.get("entry_price"),
        )
        return trade_id  # type: ignore[return-value]

    @staticmethod
    def close_trade(
        db: Any,
        trade_id: int,
        exit_price: float,
        pnl: float,
        fees: float = 0.0,
    ) -> None:
        """Close an existing trade by recording the exit price and P&L.

        Args:
            db: An open database connection.
            trade_id: The ``id`` of the trade to close.
            exit_price: The price at which the position was exited.
            pnl: Realized profit or loss in USD.
            fees: Trading fees incurred on the close.
        """
        now = utc_now().isoformat()

        # Retrieve entry_price and leverage to calculate pnl_pct
        row = fetch_one(
            db,
            "SELECT entry_price, side, leverage FROM trades WHERE id = ?",
            (trade_id,),
        )
        pnl_pct = 0.0
        if row and float(row["entry_price"]) != 0:
            entry = float(row["entry_price"])
            side = row["side"]
            leverage = float(row["leverage"]) if row["leverage"] else 1.0
            if side == "long":
                pnl_pct = (exit_price - entry) / entry
            else:
                pnl_pct = (entry - exit_price) / entry
            # Account P&L = spot return * leverage for perp futures
            pnl_pct *= leverage

        execute_query(
            db,
            """
            UPDATE trades
            SET exit_price = ?,
                pnl        = ?,
                pnl_pct    = ?,
                fees       = COALESCE(fees, 0) + ?,
                status     = 'closed',
                closed_at  = ?
            WHERE id = ?
            """,
            (exit_price, pnl, pnl_pct, fees, now, trade_id),
        )
        logger.info(
            "Trade closed | id={tid} exit={exit} pnl={pnl:.2f} "
            "pnl_pct={pct:.4f} fees={fees:.2f}",
            tid=trade_id,
            exit=exit_price,
            pnl=pnl,
            pct=pnl_pct,
            fees=fees,
        )

    @staticmethod
    def get_recent_trades(
        db: Any,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Fetch the most recent trades (open or closed).

        Args:
            db: An open database connection.
            limit: Maximum number of trades to return.

        Returns:
            List of trade dicts, newest first.
        """
        rows = fetch_all(
            db,
            """
            SELECT *
            FROM trades
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in rows]

    @staticmethod
    def get_trades_today(db: Any) -> list[dict[str, Any]]:
        """Fetch all trades opened today (UTC).

        Args:
            db: An open database connection.

        Returns:
            List of trade dicts opened today, newest first.
        """
        today_str = utc_now().strftime("%Y-%m-%d")
        rows = fetch_all(
            db,
            """
            SELECT *
            FROM trades
            WHERE DATE(opened_at) = ?
            ORDER BY timestamp DESC
            """,
            (today_str,),
        )
        return [dict(row) for row in rows]

    @staticmethod
    def get_last_trade_time(db: Any) -> datetime | None:
        """Return the timestamp of the most recently opened trade.

        Args:
            db: An open database connection.

        Returns:
            A timezone-aware ``datetime`` or ``None`` if no trades exist.
        """
        row = fetch_one(
            db,
            """
            SELECT opened_at
            FROM trades
            ORDER BY opened_at DESC
            LIMIT 1
            """,
        )
        if row and row["opened_at"]:
            try:
                val = row["opened_at"]
                dt = val if isinstance(val, datetime) else datetime.fromisoformat(val)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except (ValueError, TypeError):
                return None
        return None

    @staticmethod
    def get_daily_trade_count(db: Any) -> int:
        """Count how many trades have been opened today (UTC).

        Args:
            db: An open database connection.

        Returns:
            Integer count of today's trades.
        """
        today_str = utc_now().strftime("%Y-%m-%d")
        row = fetch_one(
            db,
            """
            SELECT COUNT(*) AS cnt
            FROM trades
            WHERE DATE(opened_at) = ?
            """,
            (today_str,),
        )
        return int(row["cnt"]) if row else 0
