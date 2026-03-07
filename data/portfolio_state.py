"""
Portfolio state management.

Tracks total equity, available margin, open positions, unrealized P&L,
and historical performance metrics.  Synchronizes the local SQLite
database with the live exchange state.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

from utils.logger import logger
from utils.helpers import utc_now
from data.database import execute_query, fetch_one, fetch_all


class PortfolioManager:
    """Manages portfolio state and performance metrics.

    All methods that accept a *db* parameter expect an open
    ``sqlite3.Connection`` (with ``row_factory = sqlite3.Row``).

    Usage::

        pm = PortfolioManager()
        state = pm.fetch_portfolio_from_exchange(exchange_client)
        dd = pm.get_drawdown_from_peak(state["total_equity"], peak)
    """

    # -------------------------------------------------------------------
    # Exchange interaction
    # -------------------------------------------------------------------

    def fetch_portfolio_from_exchange(
        self,
        exchange_client: Any,
        wallet_address: str | None = None,
    ) -> dict[str, Any]:
        """Fetch the current portfolio state from Hyperliquid.

        Args:
            exchange_client: An instance of the Hyperliquid ``Info`` client
                that exposes ``user_state(address)``-style methods.
            wallet_address: The wallet address to query.  Falls back to
                ``HYPERLIQUID_WALLET_ADDRESS`` from config if not provided.

        Returns:
            Dict with keys ``total_equity``, ``available_margin``,
            ``unrealized_pnl``, ``positions`` (list of position dicts).
        """
        try:
            from config.trading_config import (
                HYPERLIQUID_WALLET_ADDRESS,
                LIVE_TRADING,
                STARTING_CAPITAL,
            )
            address = wallet_address or HYPERLIQUID_WALLET_ADDRESS
            if not address:
                if not LIVE_TRADING:
                    # Paper mode — compute equity from trades table
                    return self._compute_paper_equity()
                logger.error("No wallet address configured for portfolio fetch")
                return {
                    "total_equity": 0.0,
                    "available_margin": 0.0,
                    "unrealized_pnl": 0.0,
                    "positions": [],
                    "margin_used": 0.0,
                }
            user_state = exchange_client.user_state(address)

            # Parse margin summary
            margin = user_state.get("marginSummary", {})
            total_equity = float(margin.get("accountValue", 0))
            total_margin_used = float(margin.get("totalMarginUsed", 0))
            available_margin = total_equity - total_margin_used

            # Parse positions
            raw_positions = user_state.get("assetPositions", [])
            positions: list[dict[str, Any]] = []
            unrealized_pnl = 0.0

            for pos_wrapper in raw_positions:
                pos = pos_wrapper.get("position", pos_wrapper)

                size = float(pos.get("szi", 0))
                if size == 0:
                    continue  # skip empty positions

                entry_px = float(pos.get("entryPx", 0))
                unrealized = float(pos.get("unrealizedPnl", 0))
                unrealized_pnl += unrealized

                positions.append(
                    {
                        "asset": pos.get("coin", ""),
                        "side": "long" if size > 0 else "short",
                        "size": abs(size),
                        "entry_price": entry_px,
                        "unrealized_pnl": unrealized,
                        "leverage": float(pos.get("leverage", {}).get("value", 1)),
                        "liquidation_price": float(pos.get("liquidationPx", 0) or 0),
                        "margin_used": float(pos.get("marginUsed", 0)),
                    }
                )

            result = {
                "total_equity": total_equity,
                "available_margin": available_margin,
                "unrealized_pnl": unrealized_pnl,
                "positions": positions,
                "margin_used": total_margin_used,
            }
            logger.info(
                "Portfolio fetched | equity={eq:.2f} margin_avail={ma:.2f} "
                "positions={pc}",
                eq=total_equity,
                ma=available_margin,
                pc=len(positions),
            )
            return result

        except Exception as exc:
            logger.error(
                "Failed to fetch portfolio from exchange: {err}", err=exc
            )
            return {
                "total_equity": 0.0,
                "available_margin": 0.0,
                "unrealized_pnl": 0.0,
                "positions": [],
                "margin_used": 0.0,
            }

    # -------------------------------------------------------------------
    # Paper mode equity computation
    # -------------------------------------------------------------------

    def _compute_paper_equity(self) -> dict[str, Any]:
        """Compute portfolio state from trades table in paper mode.

        Sums unrealized P&L from all open trades (using pnl_pct updated
        by PositionManager.manage_open_positions) and realized P&L from
        all closed trades.  This gives the dashboard and Grok an accurate
        view of the simulated equity rather than a static starting capital.

        Returns:
            Dict matching the same schema as fetch_portfolio_from_exchange.
        """
        from config.trading_config import STARTING_CAPITAL
        from data.database import get_db_connection

        try:
            db = get_db_connection()

            # Unrealized P&L from open trades:
            #   Each trade's $ impact = pnl_pct * size_pct * STARTING_CAPITAL * leverage
            open_rows = db.execute(
                """SELECT asset, side, size_pct, leverage, entry_price,
                          stop_loss, take_profit, pnl_pct
                   FROM trades WHERE status = 'open'"""
            ).fetchall()

            unrealized_pnl = 0.0
            margin_used = 0.0
            positions: list[dict[str, Any]] = []

            for row in open_rows:
                pnl_pct = float(row["pnl_pct"] or 0.0)
                size_pct = float(row["size_pct"])
                leverage = float(row["leverage"])
                trade_unrealized = pnl_pct * size_pct * STARTING_CAPITAL * leverage
                unrealized_pnl += trade_unrealized
                trade_margin = size_pct * STARTING_CAPITAL
                margin_used += trade_margin
                positions.append({
                    "asset": row["asset"],
                    "side": row["side"],
                    "size": size_pct,
                    "entry_price": float(row["entry_price"] or 0),
                    "unrealized_pnl": trade_unrealized,
                    "leverage": leverage,
                    "liquidation_price": 0.0,
                    "margin_used": trade_margin,
                })

            # Realized P&L from closed trades
            realized_row = db.execute(
                "SELECT COALESCE(SUM(pnl), 0) AS realized FROM trades WHERE status = 'closed'"
            ).fetchone()
            realized_pnl = float(realized_row["realized"]) if realized_row else 0.0

            # Total fees
            fee_row = db.execute(
                "SELECT COALESCE(SUM(fees), 0) AS total_fees FROM trades"
            ).fetchone()
            total_fees = float(fee_row["total_fees"]) if fee_row else 0.0

            total_equity = STARTING_CAPITAL + unrealized_pnl + realized_pnl - total_fees
            available_margin = total_equity - margin_used

            db.close()

            logger.debug(
                "Paper equity computed | equity={eq:.2f} unrealized={up:.2f} "
                "realized={rp:.2f} fees={f:.2f} positions={pc}",
                eq=total_equity, up=unrealized_pnl, rp=realized_pnl,
                f=total_fees, pc=len(positions),
            )
            return {
                "total_equity": total_equity,
                "available_margin": available_margin,
                "unrealized_pnl": unrealized_pnl,
                "positions": positions,
                "margin_used": margin_used,
            }

        except Exception as exc:
            logger.error("Paper equity computation failed: {err}", err=exc)
            from config.trading_config import STARTING_CAPITAL as SC
            return {
                "total_equity": SC,
                "available_margin": SC,
                "unrealized_pnl": 0.0,
                "positions": [],
                "margin_used": 0.0,
            }

    # -------------------------------------------------------------------
    # Unrealized P&L
    # -------------------------------------------------------------------

    @staticmethod
    def calculate_unrealized_pnl(positions: list[dict[str, Any]]) -> float:
        """Sum up unrealized P&L across all open positions.

        Args:
            positions: List of position dicts, each with an
                ``unrealized_pnl`` key.

        Returns:
            Total unrealized P&L as a float.
        """
        return sum(float(p.get("unrealized_pnl", 0)) for p in positions)

    # -------------------------------------------------------------------
    # Historical P&L from the database
    # -------------------------------------------------------------------

    @staticmethod
    def get_daily_pnl(db: sqlite3.Connection) -> float:
        """Calculate total realized P&L from today's closed trades.

        Args:
            db: An open database connection.

        Returns:
            Today's realized P&L in USD.
        """
        today_str = utc_now().strftime("%Y-%m-%d")
        row = fetch_one(
            db,
            """
            SELECT COALESCE(SUM(pnl), 0) AS total_pnl
            FROM trades
            WHERE status = 'closed'
              AND date(closed_at) = ?
            """,
            (today_str,),
        )
        return float(row["total_pnl"]) if row else 0.0

    @staticmethod
    def get_weekly_pnl(db: sqlite3.Connection) -> float:
        """Calculate total realized P&L from this week's closed trades.

        Uses a rolling 7-day window from the current UTC time.

        Args:
            db: An open database connection.

        Returns:
            This week's realized P&L in USD.
        """
        week_start = (utc_now() - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
        row = fetch_one(
            db,
            """
            SELECT COALESCE(SUM(pnl), 0) AS total_pnl
            FROM trades
            WHERE status = 'closed'
              AND closed_at >= ?
            """,
            (week_start,),
        )
        return float(row["total_pnl"]) if row else 0.0

    @staticmethod
    def get_peak_equity(db: sqlite3.Connection) -> float:
        """Return the highest recorded ending equity from daily summaries.

        Falls back to ``STARTING_CAPITAL`` if no summaries exist yet.

        Args:
            db: An open database connection.

        Returns:
            Peak equity value.
        """
        from config.trading_config import STARTING_CAPITAL

        row = fetch_one(
            db,
            """
            SELECT COALESCE(MAX(ending_equity), 0) AS peak
            FROM daily_summaries
            """,
        )
        peak = float(row["peak"]) if row and row["peak"] else 0.0
        return max(peak, STARTING_CAPITAL)

    @staticmethod
    def get_drawdown_from_peak(
        current_equity: float, peak_equity: float
    ) -> float:
        """Calculate the drawdown from peak equity as a decimal ratio.

        Args:
            current_equity: Current total equity.
            peak_equity: Historical peak equity.

        Returns:
            Drawdown as a positive decimal (e.g. 0.05 means 5% drawdown).
            Returns 0.0 if peak is zero or current >= peak.
        """
        if peak_equity <= 0 or current_equity >= peak_equity:
            return 0.0
        return (peak_equity - current_equity) / peak_equity

    @staticmethod
    def get_consecutive_losses(db: sqlite3.Connection) -> int:
        """Count the current streak of consecutive losing trades.

        Counts backwards from the most recent closed trade until a
        winning trade (``pnl >= 0``) is found.

        Args:
            db: An open database connection.

        Returns:
            Number of consecutive losses (0 if the last trade was a win).
        """
        rows = fetch_all(
            db,
            """
            SELECT pnl
            FROM trades
            WHERE status = 'closed'
            ORDER BY closed_at DESC
            LIMIT 50
            """,
        )
        streak = 0
        for row in rows:
            pnl = row["pnl"]
            if pnl is None:
                continue  # Skip trades closed by sync with no dollar P&L
            if float(pnl) < 0:
                streak += 1
            else:
                break
        return streak

    # -------------------------------------------------------------------
    # Position synchronization
    # -------------------------------------------------------------------

    def sync_positions_with_exchange(
        self,
        exchange_client: Any,
        db: sqlite3.Connection,
    ) -> None:
        """Synchronize the database ``positions`` table with the exchange.

        Positions that exist on the exchange but not in the DB are
        inserted.  Positions in the DB that no longer exist on the
        exchange are marked ``closed``.  Existing entries are updated
        with the latest unrealized P&L.

        Args:
            exchange_client: The Hyperliquid client.
            db: An open database connection.
        """
        try:
            portfolio = self.fetch_portfolio_from_exchange(exchange_client)
            exchange_positions = portfolio["positions"]

            # Build a lookup of exchange positions by asset
            exchange_map: dict[str, dict[str, Any]] = {
                p["asset"].upper(): p for p in exchange_positions
            }

            # Fetch DB open positions
            db_positions = fetch_all(
                db,
                "SELECT * FROM positions WHERE status = 'open'",
            )

            db_asset_map: dict[str, sqlite3.Row] = {}
            for db_pos in db_positions:
                db_asset_map[db_pos["asset"].upper()] = db_pos

            # --- Mark DB positions that are gone from the exchange as closed
            for asset_key, db_pos in db_asset_map.items():
                if asset_key not in exchange_map:
                    execute_query(
                        db,
                        "UPDATE positions SET status = 'closed' WHERE id = ?",
                        (db_pos["id"],),
                    )
                    logger.info(
                        "Position closed (exchange sync): {asset}",
                        asset=asset_key,
                    )

            # --- Update or insert exchange positions
            for asset_key, ex_pos in exchange_map.items():
                if asset_key in db_asset_map:
                    # Update unrealized P&L
                    execute_query(
                        db,
                        """
                        UPDATE positions
                        SET unrealized_pnl = ?,
                            leverage = ?
                        WHERE id = ?
                        """,
                        (
                            ex_pos["unrealized_pnl"],
                            ex_pos["leverage"],
                            db_asset_map[asset_key]["id"],
                        ),
                    )
                else:
                    # New position found on exchange -- insert it
                    execute_query(
                        db,
                        """
                        INSERT INTO positions
                            (asset, side, size_pct, leverage, entry_price,
                             unrealized_pnl, opened_at, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, 'open')
                        """,
                        (
                            ex_pos["asset"],
                            ex_pos["side"],
                            0.0,  # size_pct unknown from exchange alone
                            ex_pos["leverage"],
                            ex_pos["entry_price"],
                            ex_pos["unrealized_pnl"],
                            utc_now().isoformat(),
                        ),
                    )
                    logger.info(
                        "New position inserted from exchange: {asset} {side}",
                        asset=ex_pos["asset"],
                        side=ex_pos["side"],
                    )

            logger.info("Position sync complete")

        except Exception as exc:
            logger.error("Position sync failed: {err}", err=exc)
