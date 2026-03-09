"""
Position Manager -- monitors and reconciles open positions.

Responsibilities:
    - Periodically check open positions against exchange state.
    - Detect when stop-loss or take-profit orders have been filled.
    - Update the local database with current unrealized P&L.
    - Reconcile DB records with actual exchange positions (crash recovery).
    - Compute total portfolio exposure for risk checks.

Works in both live and paper modes.  In paper mode the position state
comes from ``OrderManager.paper_positions`` rather than the exchange API.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from loguru import logger

from config.trading_config import (
    HYPERLIQUID_WALLET_ADDRESS,
    LIVE_TRADING,
)
from config.risk_config import RISK_PARAMS
from execution.order_manager import OrderManager


class PositionManager:
    """Monitors open positions and keeps the local DB in sync with exchange state.

    The PositionManager does NOT make trading decisions -- it only
    observes, records, and reconciles.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, order_manager: OrderManager | None = None) -> None:
        """Initialise the Position Manager.

        Args:
            order_manager: An ``OrderManager`` instance.  Used to access
                           the Info client (live mode) or paper positions
                           (paper mode).  If ``None``, a new one is created.
        """
        self._om: OrderManager = order_manager or OrderManager()
        logger.info(
            "PositionManager initialised | mode={mode}",
            mode="LIVE" if LIVE_TRADING else "PAPER",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def manage_open_positions(
        self,
        db_connection: Any,
    ) -> list[dict[str, Any]]:
        """Check all open positions and update the database.

        For each position:
            1. Fetch current mark price.
            2. Compute unrealized P&L.
            3. Check if stop-loss or take-profit was hit on-exchange.
            4. Update the DB record accordingly.

        Args:
            db_connection: Active SQLite connection.

        Returns:
            List of position status dicts with current state.
        """
        positions = self._fetch_exchange_positions()
        results: list[dict[str, Any]] = []

        for pos in positions:
            asset = pos.get("asset", "")
            entry_price = pos.get("entry_price", 0.0)
            side = pos.get("side", "long")
            size = pos.get("size", 0.0)

            # Fetch current mark price
            mark_price = self._get_mark_price(asset)
            if mark_price is None or mark_price <= 0:
                logger.warning(
                    "Cannot get mark price for {asset}; skipping P&L update.",
                    asset=asset,
                )
                results.append({**pos, "status": "price_unavailable"})
                continue

            # Compute unrealized P&L
            if side == "long":
                unrealized_pnl_pct = (
                    (mark_price - entry_price) / entry_price
                    if entry_price > 0 else 0.0
                )
            else:
                unrealized_pnl_pct = (
                    (entry_price - mark_price) / entry_price
                    if entry_price > 0 else 0.0
                )

            # Update DB with current unrealized P&L (pnl_pct column in canonical schema)
            try:
                db_connection.execute(
                    """
                    UPDATE trades
                    SET pnl_pct = ?
                    WHERE asset = ? AND status = 'open'
                    """,
                    (
                        round(unrealized_pnl_pct, 6),
                        asset,
                    ),
                )
                db_connection.commit()
            except Exception as exc:
                logger.error(
                    "Failed to update P&L for {asset}: {err}",
                    asset=asset, err=exc,
                )

            pos_status = {
                "asset": asset,
                "side": side,
                "size": size,
                "entry_price": entry_price,
                "mark_price": mark_price,
                "unrealized_pnl_pct": round(unrealized_pnl_pct, 6),
                "status": "open",
            }

            results.append(pos_status)

        # Check for positions that were closed on the exchange
        # (e.g. stop-loss hit) but still show as open in the DB.
        self._detect_closed_positions(positions, db_connection)

        # Check for stale positions exceeding max holding period
        # (ai-trader OR-logic pattern: close if held > N hours)
        if RISK_PARAMS.get("stale_position_check", True):
            self._check_holding_period(db_connection)

        logger.debug(
            "Position check complete | {n} open positions",
            n=len(results),
        )
        return results

    def sync_positions(
        self,
        db_connection: Any,
    ) -> dict[str, Any]:
        """Reconcile exchange positions with DB records (crash recovery).

        Compares the exchange's view of positions with the local DB and
        corrects any discrepancies.  This is critical after a bot restart
        or crash.

        Args:
            db_connection: Active SQLite connection.

        Returns:
            Dict summarising the sync: ``positions_added``, ``positions_closed``,
            ``positions_unchanged``.
        """
        exchange_positions = self._fetch_exchange_positions()
        exchange_assets = {p["asset"] for p in exchange_positions}

        # Get DB open positions
        db_open: dict[str, dict[str, Any]] = {}
        try:
            cursor = db_connection.execute(
                "SELECT asset, side, size_pct, entry_price FROM trades WHERE status = 'open'"
            )
            for row in cursor.fetchall():
                db_open[row["asset"]] = {
                    "asset": row["asset"],
                    "side": row["side"],
                    "size_pct": row["size_pct"],
                    "entry_price": row["entry_price"],
                }
        except Exception as exc:
            logger.error("Failed to read DB positions: {err}", err=exc)

        db_assets = set(db_open.keys())

        added = 0
        closed = 0
        unchanged = 0

        # Positions on exchange but not in DB -> add them
        for pos in exchange_positions:
            asset = pos["asset"]
            if asset not in db_assets:
                try:
                    db_connection.execute(
                        """
                        INSERT INTO trades
                            (asset, side, action, size_pct, entry_price, leverage,
                             stop_loss, take_profit, status, opened_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)
                        """,
                        (
                            asset,
                            pos.get("side", "long"),
                            f"open_{pos.get('side', 'long')}",
                            pos.get("size", 0.0),
                            pos.get("entry_price", 0.0),
                            pos.get("leverage", 1.0),
                            pos.get("stop_loss", 0.0),
                            pos.get("take_profit", 0.0),
                            datetime.now(timezone.utc).isoformat(),
                        ),
                    )
                    db_connection.commit()
                    added += 1
                    logger.info(
                        "SYNC: Added missing position for {asset}", asset=asset,
                    )
                except Exception as exc:
                    logger.error(
                        "SYNC: Failed to add {asset}: {err}",
                        asset=asset, err=exc,
                    )
            else:
                unchanged += 1

        # Positions in DB but not on exchange -> mark as closed
        for asset in db_assets - exchange_assets:
            try:
                mark_price = self._get_mark_price(asset) or 0.0
                entry_price = db_open[asset].get("entry_price", 0.0)
                side = db_open[asset].get("side", "long")

                if side == "long":
                    pnl = (
                        (mark_price - entry_price) / entry_price
                        if entry_price > 0 else 0.0
                    )
                else:
                    pnl = (
                        (entry_price - mark_price) / entry_price
                        if entry_price > 0 else 0.0
                    )

                # Compute dollar P&L for the pnl column
                size_pct_f = float(db_open[asset].get("size_pct", 0))
                leverage_f = float(db_open[asset].get("leverage", 1))
                from config.trading_config import STARTING_CAPITAL
                dollar_pnl = round(pnl * size_pct_f * STARTING_CAPITAL * leverage_f, 4)

                db_connection.execute(
                    """
                    UPDATE trades
                    SET status     = 'closed',
                        exit_price = ?,
                        pnl        = ?,
                        pnl_pct    = ?,
                        closed_at  = ?,
                        reasoning  = COALESCE(reasoning, '') || ' [closed: exchange_sync]'
                    WHERE asset = ? AND status = 'open'
                    """,
                    (
                        mark_price,
                        dollar_pnl,
                        round(pnl, 6),
                        datetime.now(timezone.utc).isoformat(),
                        asset,
                    ),
                )
                db_connection.commit()
                closed += 1
                logger.info(
                    "SYNC: Closed stale DB position for {asset} | pnl={pnl}%",
                    asset=asset,
                    pnl=round(pnl * 100, 2),
                )
            except Exception as exc:
                logger.error(
                    "SYNC: Failed to close {asset}: {err}",
                    asset=asset, err=exc,
                )

        summary = {
            "positions_added": added,
            "positions_closed": closed,
            "positions_unchanged": unchanged,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.info(
            "SYNC COMPLETE | added={a} closed={c} unchanged={u}",
            a=added, c=closed, u=unchanged,
        )
        return summary

    def get_total_exposure(self, db_connection: Any) -> float:
        """Compute total exposure as a fraction of portfolio equity.

        Sums the ``size_pct`` of all open positions in the database.

        Args:
            db_connection: Active SQLite connection.

        Returns:
            Total exposure as a decimal (e.g. 0.25 = 25%).
        """
        try:
            cursor = db_connection.execute(
                "SELECT COALESCE(SUM(size_pct), 0.0) AS total FROM trades WHERE status = 'open'"
            )
            row = cursor.fetchone()
            return float(row["total"]) if row else 0.0
        except Exception as exc:
            logger.error("Failed to compute total exposure: {err}", err=exc)
            return 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_exchange_positions(self) -> list[dict[str, Any]]:
        """Fetch open positions from the exchange or paper state.

        Returns:
            List of normalised position dicts with keys:
            ``asset``, ``side``, ``size``, ``entry_price``, ``leverage``,
            ``unrealized_pnl``, ``stop_loss``, ``take_profit``.
        """
        if not LIVE_TRADING:
            return self._fetch_paper_positions()
        return self._fetch_live_positions()

    def _fetch_live_positions(self) -> list[dict[str, Any]]:
        """Fetch live positions from the Hyperliquid exchange."""
        if self._om.info is None:
            logger.warning("Info client unavailable; cannot fetch live positions.")
            return []

        try:
            address = HYPERLIQUID_WALLET_ADDRESS
            user_state = self._om.info.user_state(address)
            raw_positions = user_state.get("assetPositions", [])

            positions: list[dict[str, Any]] = []
            for raw in raw_positions:
                pos = raw.get("position", {})
                szi = float(pos.get("szi", 0))
                if szi == 0:
                    continue  # No active position

                positions.append({
                    "asset": pos.get("coin", ""),
                    "side": "long" if szi > 0 else "short",
                    "size": abs(szi),
                    "entry_price": float(pos.get("entryPx", 0)),
                    "leverage": float(pos.get("leverage", {}).get("value", 1)),
                    "unrealized_pnl": float(pos.get("unrealizedPnl", 0)),
                    "liquidation_price": float(pos.get("liquidationPx", 0)),
                })

            return positions

        except Exception as exc:
            logger.error("Failed to fetch live positions: {err}", err=exc)
            return []

    def _fetch_paper_positions(self) -> list[dict[str, Any]]:
        """Fetch positions from the paper trading state."""
        positions: list[dict[str, Any]] = []
        for asset, pos in self._om.paper_positions.items():
            positions.append({
                "asset": asset,
                "side": pos.get("side", "long"),
                "size": pos.get("size_pct", 0.0),
                "entry_price": pos.get("entry_price", 0.0),
                "leverage": pos.get("leverage", 1.0),
                "unrealized_pnl": pos.get("unrealized_pnl", 0.0),
                "stop_loss": pos.get("stop_loss", 0.0),
                "take_profit": pos.get("take_profit", 0.0),
            })
        return positions

    def _get_mark_price(self, asset: str) -> float | None:
        """Fetch the current mark/mid price for an asset.

        Delegates to the OrderManager's Info client.

        Args:
            asset: Asset symbol.

        Returns:
            Float price or None.
        """
        if self._om.info is None:
            return None

        try:
            all_mids = self._om.info.all_mids()
            price_str = all_mids.get(asset)
            if price_str is not None:
                return float(price_str)
        except Exception as exc:
            logger.warning(
                "Failed to fetch mark price for {asset}: {err}",
                asset=asset, err=exc,
            )
        return None

    def _detect_closed_positions(
        self,
        exchange_positions: list[dict[str, Any]],
        db_connection: Any,
    ) -> None:
        """Detect positions that were closed on-exchange (e.g. stop hit).

        If a position exists in the DB as 'open' but is absent from the
        exchange, it was likely closed by a trigger order.  Mark it as
        closed in the DB with the appropriate reason.

        Args:
            exchange_positions: Currently open exchange positions.
            db_connection: Active SQLite connection.
        """
        exchange_assets = {p["asset"] for p in exchange_positions}

        try:
            cursor = db_connection.execute(
                "SELECT asset, side, entry_price, stop_loss, take_profit, size_pct, leverage "
                "FROM trades WHERE status = 'open'"
            )
            db_open_rows = cursor.fetchall()
        except Exception as exc:
            logger.error(
                "Failed to query open trades for closure detection: {err}",
                err=exc,
            )
            return

        for row in db_open_rows:
            asset = row["asset"]
            side = row["side"]
            entry_price = row["entry_price"]
            stop_loss = row["stop_loss"]
            take_profit = row["take_profit"]
            size_pct = row["size_pct"]
            leverage = row["leverage"]

            if asset in exchange_assets:
                continue  # Still open on exchange

            # Position closed on exchange -- determine reason
            mark_price = self._get_mark_price(asset) or 0.0

            if side == "long":
                pnl = (
                    (mark_price - entry_price) / entry_price
                    if entry_price > 0 else 0.0
                )
                if stop_loss and mark_price <= stop_loss:
                    reason = "stop_loss_hit"
                elif take_profit and mark_price >= take_profit:
                    reason = "take_profit_hit"
                else:
                    reason = "closed_on_exchange"
            else:
                pnl = (
                    (entry_price - mark_price) / entry_price
                    if entry_price > 0 else 0.0
                )
                if stop_loss and mark_price >= stop_loss:
                    reason = "stop_loss_hit"
                elif take_profit and mark_price <= take_profit:
                    reason = "take_profit_hit"
                else:
                    reason = "closed_on_exchange"

            # Compute dollar P&L for the pnl column
            size_pct_f = float(size_pct or 0)
            leverage_f = float(leverage or 1)
            from config.trading_config import STARTING_CAPITAL
            dollar_pnl = round(pnl * size_pct_f * STARTING_CAPITAL * leverage_f, 4)

            try:
                db_connection.execute(
                    """
                    UPDATE trades
                    SET status     = 'closed',
                        exit_price = ?,
                        pnl        = ?,
                        pnl_pct    = ?,
                        closed_at  = ?,
                        reasoning  = COALESCE(reasoning, '') || ' [closed: ' || ? || ']'
                    WHERE asset = ? AND status = 'open'
                    """,
                    (
                        mark_price,
                        dollar_pnl,
                        round(pnl, 6),
                        datetime.now(timezone.utc).isoformat(),
                        reason,
                        asset,
                    ),
                )
                db_connection.commit()

                # Clean up paper state to prevent DB/memory desync
                if not LIVE_TRADING:
                    self._cleanup_paper_position(asset)

                logger.info(
                    "Position auto-closed | {asset} reason={reason} pnl={pnl}% (${dpnl})",
                    asset=asset,
                    reason=reason,
                    pnl=round(pnl * 100, 2),
                    dpnl=dollar_pnl,
                )
            except Exception as exc:
                logger.error(
                    "Failed to mark {asset} as closed: {err}",
                    asset=asset, err=exc,
                )

    def _check_holding_period(
        self,
        db_connection: Any,
    ) -> None:
        """Force-close positions that exceed the maximum holding period.

        Uses the OR-logic pattern from ai-trader: a position is closed if
        ``held > max_holding_period_hours``, regardless of P&L.  This
        prevents overnight exposure and stale positions from accumulating.

        Args:
            db_connection: Active SQLite connection.
        """
        max_hours = RISK_PARAMS.get("max_holding_period_hours", 8)

        try:
            cursor = db_connection.execute(
                "SELECT asset, side, entry_price, size_pct, leverage, opened_at "
                "FROM trades WHERE status = 'open'"
            )
            open_rows = cursor.fetchall()
        except Exception as exc:
            logger.error(
                "Failed to query trades for holding period check: {err}",
                err=exc,
            )
            return

        now = datetime.now(timezone.utc)

        for row in open_rows:
            asset = row["asset"]
            side = row["side"]
            entry_price = row["entry_price"]
            size_pct = row["size_pct"]
            leverage = row["leverage"]
            opened_at_val = row["opened_at"]

            if not opened_at_val:
                continue

            # Parse the opened_at timestamp (psycopg2 returns datetime objects)
            try:
                if isinstance(opened_at_val, datetime):
                    opened_at = opened_at_val
                else:
                    opened_at = datetime.fromisoformat(opened_at_val)
                if opened_at.tzinfo is None:
                    opened_at = opened_at.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue

            hours_held = (now - opened_at).total_seconds() / 3600

            if hours_held < max_hours:
                continue

            # Position exceeded max holding period — force close

            # In live mode, actually close the position on the exchange first
            if LIVE_TRADING:
                try:
                    close_result = self._om.close_position(asset)
                    if close_result.get("status") == "error":
                        logger.error(
                            "TIME-EXIT: Failed to close {asset} on exchange: {err}",
                            asset=asset, err=close_result.get("error", "unknown"),
                        )
                        continue  # Don't mark DB closed if exchange close failed
                    logger.info(
                        "TIME-EXIT: Closed {asset} on exchange",
                        asset=asset,
                    )
                except Exception as exc:
                    logger.error(
                        "TIME-EXIT: Exchange close failed for {asset}: {err}",
                        asset=asset, err=exc,
                    )
                    continue  # Don't mark DB closed if exchange close failed

            mark_price = self._get_mark_price(asset) or 0.0

            if side == "long":
                pnl_pct = (
                    (mark_price - entry_price) / entry_price
                    if entry_price and entry_price > 0 else 0.0
                )
            else:
                pnl_pct = (
                    (entry_price - mark_price) / entry_price
                    if entry_price and entry_price > 0 else 0.0
                )

            # Compute dollar P&L for the pnl column
            size_pct_f = float(size_pct or 0)
            leverage_f = float(leverage or 1)
            from config.trading_config import STARTING_CAPITAL
            dollar_pnl = round(pnl_pct * size_pct_f * STARTING_CAPITAL * leverage_f, 4)

            try:
                db_connection.execute(
                    """
                    UPDATE trades
                    SET status     = 'closed',
                        exit_price = ?,
                        pnl        = ?,
                        pnl_pct    = ?,
                        closed_at  = ?,
                        reasoning  = COALESCE(reasoning, '') || ' [closed: max_holding_period_exceeded]'
                    WHERE asset = ? AND status = 'open'
                    """,
                    (
                        mark_price,
                        dollar_pnl,
                        round(pnl_pct, 6),
                        now.isoformat(),
                        asset,
                    ),
                )
                db_connection.commit()

                # Clean up paper state to prevent DB/memory desync
                if not LIVE_TRADING:
                    self._cleanup_paper_position(asset)

                logger.warning(
                    "TIME-EXIT: {asset} held {hrs:.1f}h > {max}h limit | "
                    "pnl={pnl}% (${dpnl}) | exit_price={px}",
                    asset=asset,
                    hrs=hours_held,
                    max=max_hours,
                    pnl=round(pnl_pct * 100, 2),
                    dpnl=dollar_pnl,
                    px=mark_price,
                )
            except Exception as exc:
                logger.error(
                    "Failed to time-exit {asset}: {err}",
                    asset=asset, err=exc,
                )

    def _cleanup_paper_position(self, asset: str) -> None:
        """Remove a position from paper state after it's been closed in the DB.

        Prevents the desync where the DB marks a position as closed but the
        in-memory paper state still holds it as open.
        """
        from execution.order_manager import _paper_state

        if asset in _paper_state.positions:
            del _paper_state.positions[asset]
            # Cancel associated SL/TP orders
            for oid, order in list(_paper_state.orders.items()):
                if order.get("asset") == asset and order.get("status") == "open":
                    _paper_state.orders[oid]["status"] = "cancelled"
            logger.debug(
                "Cleaned up paper state for {asset} after DB close",
                asset=asset,
            )

    # NOTE: _ensure_trades_table was removed.  The canonical schema lives in
    # data/database.py and is created by init_db() at startup.  Having a
    # second CREATE TABLE with a different column set caused silent schema
    # divergence and runtime OperationalErrors.
