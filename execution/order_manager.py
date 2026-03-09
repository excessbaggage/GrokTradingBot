"""
Order Manager -- translates TradeDecisions into Hyperliquid API calls.

Supports both LIVE trading (real orders on Hyperliquid mainnet/testnet)
and PAPER trading (simulated fills tracked in-memory).  The mode is
determined by the ``LIVE_TRADING`` flag in ``config.trading_config``.

In paper mode every order is "filled" instantly at the current mark price
with a configurable simulated fee.  This allows full end-to-end testing
without risking capital.

Key integration points:
    - ``hyperliquid-python-sdk`` for exchange interaction
    - ``eth_account`` for wallet signing
    - ``config.trading_config`` for API URLs, keys, and the live/paper flag
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from brain.models import TradeDecision
from config.trading_config import (
    HYPERLIQUID_PRIVATE_KEY,
    HYPERLIQUID_WALLET_ADDRESS,
    LIVE_TRADING,
    get_hyperliquid_url,
)


# ---------------------------------------------------------------------------
# Paper-trading state (module-level so it persists across calls)
# ---------------------------------------------------------------------------

class _PaperState:
    """In-memory state for paper trading simulation."""

    def __init__(self) -> None:
        self.positions: dict[str, dict[str, Any]] = {}  # asset -> position info
        self.orders: dict[str, dict[str, Any]] = {}     # order_id -> order info
        self.trade_history: list[dict[str, Any]] = []
        self.simulated_fee_rate: float = 0.00035  # 3.5 bps taker fee

    def reset(self) -> None:
        """Clear all paper-trading state."""
        self.positions.clear()
        self.orders.clear()
        self.trade_history.clear()


_paper_state = _PaperState()


class OrderManager:
    """Translates TradeDecisions into exchange orders or paper fills.

    In live mode the manager uses the ``hyperliquid`` Python SDK to place
    orders on-chain.  In paper mode it simulates fills locally.

    Attributes:
        live: Whether we are executing real orders.
        exchange: Hyperliquid ``Exchange`` instance (live mode only).
        info: Hyperliquid ``Info`` instance for market data.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        """Initialise the OrderManager.

        In live mode, the Hyperliquid SDK ``Exchange`` and ``Info`` objects
        are created from the configured private key and API URL.  In paper
        mode only the ``Info`` object is needed (for mark prices).
        """
        self.live: bool = LIVE_TRADING
        self.exchange: Any = None
        self.info: Any = None

        base_url = get_hyperliquid_url()

        try:
            from hyperliquid.info import Info
            # Pass empty spot_meta — we only trade perps and testnet
            # spot metadata is often incomplete (causes IndexError).
            self.info = Info(
                base_url=base_url,
                skip_ws=True,
                spot_meta={"universe": [], "tokens": []},
            )
        except ImportError:
            logger.warning(
                "hyperliquid SDK not installed -- Info client unavailable. "
                "Paper mode will use placeholder prices.",
            )
        except Exception as exc:
            logger.error("Failed to initialise Hyperliquid Info: {err}", err=exc)

        if self.live:
            if not HYPERLIQUID_PRIVATE_KEY:
                raise ValueError(
                    "LIVE_TRADING is True but HYPERLIQUID_PRIVATE_KEY is not set."
                )
            try:
                from eth_account import Account
                from hyperliquid.exchange import Exchange

                wallet = Account.from_key(HYPERLIQUID_PRIVATE_KEY)
                self.exchange = Exchange(
                    wallet=wallet,
                    base_url=base_url,
                )
                logger.info(
                    "OrderManager initialised in LIVE mode | url={url}",
                    url=base_url,
                )
            except ImportError:
                raise ImportError(
                    "LIVE_TRADING requires `hyperliquid` and `eth_account` packages."
                )
        else:
            logger.info(
                "OrderManager initialised in PAPER mode | url={url}",
                url=base_url,
            )

    # ------------------------------------------------------------------
    # Place order
    # ------------------------------------------------------------------

    def place_order(
        self,
        decision: TradeDecision,
        portfolio_equity: float = 0.0,
    ) -> dict[str, Any]:
        """Translate a TradeDecision into an order and execute it.

        For ``open_long`` / ``open_short``:
            - Places the primary entry order (market or limit).
            - Places a stop-loss trigger order.
            - Places a take-profit trigger order.

        For ``close``:
            - Closes the position for the given asset.

        Args:
            decision: Validated ``TradeDecision`` from the brain layer.
            portfolio_equity: Current total equity in USD, used to convert
                ``size_pct`` into a concrete coin quantity.

        Returns:
            Dict with keys: ``order_id``, ``asset``, ``side``, ``size_pct``,
            ``fill_price``, ``status``, ``fees``, ``timestamp``,
            ``stop_order_id``, ``tp_order_id``.
        """
        # Route "close" actions to close_position instead of opening a new order
        if decision.action == "close":
            return self.close_position(decision.asset)

        # Route "adjust_stop" — update the stop-loss, don't open a new position
        if decision.action == "adjust_stop":
            return self._adjust_stop_loss(decision)

        if self.live:
            return self._place_live_order(decision, portfolio_equity)
        return self._place_paper_order(decision, portfolio_equity)

    # ------------------------------------------------------------------
    # Cancel order
    # ------------------------------------------------------------------

    def cancel_order(self, order_id: str, asset: str | None = None) -> bool:
        """Cancel an open order by ID.

        Args:
            order_id: The order identifier.
            asset: Asset symbol (required for live mode).

        Returns:
            True if the cancellation succeeded, False otherwise.
        """
        if self.live:
            return self._cancel_live_order(order_id, asset)
        return self._cancel_paper_order(order_id)

    # ------------------------------------------------------------------
    # Get open orders
    # ------------------------------------------------------------------

    def get_open_orders(self, asset: str | None = None) -> list[dict[str, Any]]:
        """Retrieve currently open orders.

        Args:
            asset: If provided, filter orders for this asset only.

        Returns:
            List of order dicts.
        """
        if self.live:
            return self._get_live_open_orders(asset)
        return self._get_paper_open_orders(asset)

    # ------------------------------------------------------------------
    # Close position
    # ------------------------------------------------------------------

    def close_position(
        self,
        asset: str,
        size: float | None = None,
    ) -> dict[str, Any]:
        """Close a full or partial position for the given asset.

        Args:
            asset: The asset symbol (e.g. ``"BTC"``).
            size: Position size to close.  If ``None``, close 100%.

        Returns:
            Order result dict.
        """
        if self.live:
            return self._close_live_position(asset, size)
        return self._close_paper_position(asset, size)

    # ------------------------------------------------------------------
    # Adjust stop-loss
    # ------------------------------------------------------------------

    def _adjust_stop_loss(self, decision: TradeDecision) -> dict[str, Any]:
        """Update the stop-loss for an existing position.

        In paper mode, updates the in-memory paper state.
        In live mode, cancels the old SL trigger order and places a new one.

        Args:
            decision: A TradeDecision with action ``"adjust_stop"`` and
                      a new ``stop_loss`` price.

        Returns:
            Order result dict.
        """
        asset = decision.asset
        new_sl = decision.stop_loss

        if not self.live:
            # Paper mode: update in-memory state
            if asset not in _paper_state.positions:
                return self._error_result(asset, f"No paper position for {asset} to adjust SL.")

            old_sl = _paper_state.positions[asset].get("stop_loss", 0)
            _paper_state.positions[asset]["stop_loss"] = new_sl

            # Cancel old SL order and create new one
            for oid, order in list(_paper_state.orders.items()):
                if (
                    order.get("asset") == asset
                    and order.get("type") == "stop_loss"
                    and order.get("status") == "open"
                ):
                    _paper_state.orders[oid]["status"] = "cancelled"

            new_sl_oid = f"paper_sl_{uuid.uuid4().hex[:8]}"
            pos = _paper_state.positions[asset]
            is_long = pos["side"] == "long"
            _paper_state.orders[new_sl_oid] = {
                "order_id": new_sl_oid,
                "asset": asset,
                "type": "stop_loss",
                "trigger_price": new_sl,
                "side": "sell" if is_long else "buy",
                "size_pct": pos["size_pct"],
                "status": "open",
            }

            logger.info(
                "PAPER SL ADJUSTED | {asset} old_sl={old} new_sl={new}",
                asset=asset, old=old_sl, new=new_sl,
            )
            return {
                "order_id": new_sl_oid,
                "asset": asset,
                "side": "adjust_stop",
                "size_pct": 0.0,
                "fill_price": 0.0,
                "status": "filled",
                "fees": 0.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "live": False,
            }
        else:
            # Live mode: cancel existing SL and place a new one
            try:
                # Cancel existing SL orders for this asset
                open_orders = self.get_open_orders(asset)
                for order in open_orders:
                    # Trigger orders have orderType containing "trigger"
                    if order.get("orderType") in ("Stop Market", "trigger"):
                        oid = str(order.get("oid", ""))
                        if oid:
                            self.cancel_order(oid, asset)

                # Get current position to determine size and direction
                address = HYPERLIQUID_WALLET_ADDRESS
                positions = self.info.user_state(address).get("assetPositions", [])
                target = None
                for pos in positions:
                    pos_info = pos.get("position", {})
                    if pos_info.get("coin") == asset:
                        target = pos_info
                        break

                if target is None:
                    return self._error_result(asset, f"No live position for {asset} to adjust SL.")

                pos_size = abs(float(target.get("szi", 0)))
                is_long = float(target.get("szi", 0)) > 0

                new_sl_oid = self._place_trigger_order_with_retry(
                    asset=asset,
                    is_buy=not is_long,
                    sz=pos_size,
                    trigger_price=new_sl,
                    tpsl="sl",
                    label="adjusted-stop-loss",
                )

                logger.info(
                    "LIVE SL ADJUSTED | {asset} new_sl={sl} oid={oid}",
                    asset=asset, sl=new_sl, oid=new_sl_oid,
                )
                return {
                    "order_id": new_sl_oid,
                    "asset": asset,
                    "side": "adjust_stop",
                    "size_pct": 0.0,
                    "fill_price": 0.0,
                    "status": "filled" if new_sl_oid else "error",
                    "fees": 0.0,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "live": True,
                }
            except Exception as exc:
                logger.error(
                    "LIVE SL ADJUST FAILED | {asset} | {err}",
                    asset=asset, err=exc,
                )
                return self._error_result(asset, str(exc))

    # ------------------------------------------------------------------
    # Live order methods
    # ------------------------------------------------------------------

    def _place_live_order(
        self,
        decision: TradeDecision,
        portfolio_equity: float = 0.0,
    ) -> dict[str, Any]:
        """Place a real order via the Hyperliquid SDK."""
        if self.exchange is None:
            raise RuntimeError("Exchange client is not initialised.")

        asset = decision.asset
        is_buy = decision.action == "open_long"
        order_type: dict[str, Any]

        try:
            # Fetch current mark price for sizing
            mark_price = self._get_mark_price(asset)
            if mark_price is None or mark_price <= 0:
                return self._error_result(asset, "Could not fetch mark price.")

            # Convert size_pct (fraction of portfolio) into coin quantity:
            #   notional_usd = size_pct * equity * leverage
            #   coin_qty     = notional_usd / mark_price
            if portfolio_equity <= 0:
                return self._error_result(asset, "Portfolio equity unavailable for sizing.")

            notional_usd = decision.size_pct * portfolio_equity * decision.leverage
            sz = notional_usd / mark_price
            if sz <= 0:
                return self._error_result(asset, f"Computed order size is zero (equity={portfolio_equity}).")

            if decision.order_type == "market":
                order_type = {"limit": {"tif": "Ioc"}}
                # For market orders we use a limit IOC with aggressive price
                slippage = 0.005  # 0.5% slippage tolerance
                px = mark_price * (1 + slippage) if is_buy else mark_price * (1 - slippage)
            else:
                # Limit order
                px = decision.entry_price if decision.entry_price else mark_price
                order_type = {"limit": {"tif": "Gtc"}}

            # Place entry order
            result = self.exchange.order(
                asset=asset,
                is_buy=is_buy,
                sz=sz,
                limit_px=px,
                order_type=order_type,
                reduce_only=False,
            )

            entry_order_id = self._extract_order_id(result)
            fill_price = px  # Approximate; actual fill may differ

            # Verify the entry order was actually filled on the exchange
            fill_info = self._verify_fill(entry_order_id, asset)
            if fill_info["status"] == "timeout":
                logger.warning(
                    "Entry order timed out for {asset} (oid={oid}). Order cancelled.",
                    asset=asset, oid=entry_order_id,
                )
                return self._error_result(asset, f"Entry order timed out and was cancelled (oid={entry_order_id}).")
            elif fill_info["status"] == "cancelled":
                logger.warning(
                    "Entry order was rejected/cancelled for {asset} (oid={oid}).",
                    asset=asset, oid=entry_order_id,
                )
                return self._error_result(asset, f"Entry order was rejected by exchange (oid={entry_order_id}).")
            elif fill_info["status"] == "partial":
                # Partial fill -- adjust size and log
                actual_sz = fill_info["filled_size"] if fill_info["filled_size"] > 0 else sz
                logger.warning(
                    "Partial fill for {asset}: requested={req} filled={filled}. "
                    "Adjusting protective orders to filled size.",
                    asset=asset, req=sz, filled=actual_sz,
                )
                sz = actual_sz
                if fill_info["avg_price"] > 0:
                    fill_price = fill_info["avg_price"]
            else:
                # Fully filled
                if fill_info["avg_price"] > 0:
                    fill_price = fill_info["avg_price"]

            # Place protective orders (SL/TP) with retry logic
            stop_order_id, tp_order_id = self._place_protective_orders(
                asset=asset,
                is_buy=is_buy,
                sz=sz,
                stop_loss=decision.stop_loss,
                take_profit=decision.take_profit,
            )

            # Safety check: if protective orders failed, close the position
            sl_required = decision.stop_loss and decision.stop_loss > 0
            tp_required = decision.take_profit and decision.take_profit > 0
            if (sl_required and not stop_order_id) or (tp_required and not tp_order_id):
                logger.critical(
                    "SAFETY ROLLBACK: Protective orders failed for {asset}. "
                    "SL placed={sl_ok} TP placed={tp_ok}. "
                    "Closing position to prevent unprotected exposure.",
                    asset=asset,
                    sl_ok=bool(stop_order_id),
                    tp_ok=bool(tp_order_id),
                )
                try:
                    self._close_live_position(asset)
                    logger.warning(
                        "Position closed for {asset} after SL/TP failure (safety rollback).",
                        asset=asset,
                    )
                except Exception as close_exc:
                    logger.critical(
                        "CRITICAL: Failed to close unprotected position for {asset}: {err}. "
                        "MANUAL INTERVENTION REQUIRED.",
                        asset=asset, err=close_exc,
                    )
                return self._error_result(
                    asset,
                    f"Protective orders failed; position rolled back for safety. "
                    f"SL={'ok' if stop_order_id else 'FAILED'} "
                    f"TP={'ok' if tp_order_id else 'FAILED'}",
                )

            order_result = {
                "order_id": entry_order_id,
                "asset": asset,
                "side": "long" if is_buy else "short",
                "size_pct": decision.size_pct,
                "leverage": decision.leverage,
                "fill_price": fill_price,
                "stop_loss": decision.stop_loss,
                "take_profit": decision.take_profit,
                "status": "filled",
                "fees": 0.0,  # Actual fees come from exchange
                "stop_order_id": stop_order_id,
                "tp_order_id": tp_order_id,
                "order_type": decision.order_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "live": True,
                "raw_result": result,
            }

            logger.info(
                "LIVE ORDER PLACED | {asset} {side} @ {price} | SL={sl} TP={tp}",
                asset=asset,
                side=order_result["side"],
                price=fill_price,
                sl=decision.stop_loss,
                tp=decision.take_profit,
            )
            return order_result

        except Exception as exc:
            logger.error(
                "LIVE ORDER FAILED | {asset} | {err}",
                asset=asset, err=exc,
            )
            return self._error_result(asset, str(exc))

    def _cancel_live_order(self, order_id: str, asset: str | None = None) -> bool:
        """Cancel a live order on Hyperliquid."""
        if self.exchange is None:
            logger.error("Cannot cancel: exchange client not initialised.")
            return False

        try:
            self.exchange.cancel(asset=asset or "", oid=int(order_id))
            logger.info("LIVE ORDER CANCELLED | oid={oid}", oid=order_id)
            return True
        except Exception as exc:
            logger.error(
                "LIVE CANCEL FAILED | oid={oid} | {err}",
                oid=order_id, err=exc,
            )
            return False

    def _get_live_open_orders(self, asset: str | None = None) -> list[dict[str, Any]]:
        """Fetch open orders from the exchange."""
        if self.info is None:
            return []

        try:
            address = HYPERLIQUID_WALLET_ADDRESS
            open_orders = self.info.open_orders(address)
            if asset:
                open_orders = [o for o in open_orders if o.get("coin") == asset]
            return open_orders
        except Exception as exc:
            logger.error("Failed to fetch open orders: {err}", err=exc)
            return []

    def _close_live_position(
        self,
        asset: str,
        size: float | None = None,
    ) -> dict[str, Any]:
        """Close a live position on Hyperliquid."""
        if self.exchange is None or self.info is None:
            return self._error_result(asset, "Exchange/Info client not initialised.")

        try:
            address = HYPERLIQUID_WALLET_ADDRESS
            positions = self.info.user_state(address).get("assetPositions", [])

            target = None
            for pos in positions:
                pos_info = pos.get("position", {})
                if pos_info.get("coin") == asset:
                    target = pos_info
                    break

            if target is None:
                return self._error_result(asset, f"No open position for {asset}.")

            pos_size = abs(float(target.get("szi", 0)))
            is_long = float(target.get("szi", 0)) > 0
            close_size = size if size is not None else pos_size

            mark_price = self._get_mark_price(asset) or 0.0
            slippage = 0.005
            px = mark_price * (1 - slippage) if is_long else mark_price * (1 + slippage)

            result = self.exchange.order(
                asset=asset,
                is_buy=not is_long,
                sz=close_size,
                limit_px=px,
                order_type={"limit": {"tif": "Ioc"}},
                reduce_only=True,
            )

            order_id = self._extract_order_id(result)

            logger.info(
                "LIVE POSITION CLOSED | {asset} size={sz} @ ~{px}",
                asset=asset, sz=close_size, px=px,
            )
            return {
                "order_id": order_id,
                "asset": asset,
                "side": "close_long" if is_long else "close_short",
                "size": close_size,
                "fill_price": px,
                "status": "filled",
                "fees": 0.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "live": True,
                "raw_result": result,
            }

        except Exception as exc:
            logger.error(
                "LIVE CLOSE FAILED | {asset} | {err}",
                asset=asset, err=exc,
            )
            return self._error_result(asset, str(exc))

    # ------------------------------------------------------------------
    # Paper order methods
    # ------------------------------------------------------------------

    def _place_paper_order(
        self,
        decision: TradeDecision,
        portfolio_equity: float = 0.0,
    ) -> dict[str, Any]:
        """Simulate an order fill in paper mode."""
        asset = decision.asset
        is_buy = decision.action == "open_long"
        order_id = f"paper_{uuid.uuid4().hex[:12]}"

        # Get a price from the exchange if possible, otherwise use entry_price
        mark_price = self._get_mark_price(asset)
        if mark_price is None or mark_price <= 0:
            mark_price = decision.entry_price or 0.0

        if decision.order_type == "market":
            fill_price = mark_price
        else:
            fill_price = decision.entry_price if decision.entry_price else mark_price

        # Compute notional value for fee simulation:
        #   notional_usd = size_pct * equity * leverage
        if portfolio_equity > 0:
            equity = portfolio_equity
        else:
            from config.trading_config import STARTING_CAPITAL
            equity = STARTING_CAPITAL
            logger.warning(
                "Paper order using STARTING_CAPITAL fallback (${sc:.2f}) "
                "because portfolio_equity={pe}",
                sc=STARTING_CAPITAL, pe=portfolio_equity,
            )
        notional = decision.size_pct * equity * decision.leverage
        fees = notional * _paper_state.simulated_fee_rate

        # Update paper positions — warn and clean up if overwriting an existing one
        side_str = "long" if is_buy else "short"
        if asset in _paper_state.positions:
            old_pos = _paper_state.positions[asset]
            logger.warning(
                "PAPER: Overwriting existing {asset} position ({side} @ {entry}) "
                "with new {new_side} @ {new_entry}",
                asset=asset,
                side=old_pos.get("side"),
                entry=old_pos.get("entry_price"),
                new_side=side_str,
                new_entry=fill_price,
            )
            # Close the old position in-memory and mark the DB trade closed
            try:
                self._close_paper_position(asset)
            except Exception as close_exc:
                logger.error(
                    "PAPER: Failed to close old {asset} position before overwrite: {err}",
                    asset=asset, err=close_exc,
                )
            # Also close the DB trade record for this asset
            try:
                from data.database import get_db_connection
                _db = get_db_connection()
                try:
                    _db.execute(
                        """UPDATE trades
                           SET status = 'closed',
                               exit_price = ?,
                               pnl_pct = 0,
                               closed_at = ?,
                               reasoning = COALESCE(reasoning, '') || ' [closed: overwritten by new position]'
                           WHERE asset = ? AND status = 'open'""",
                        (fill_price, datetime.now(timezone.utc).isoformat(), asset),
                    )
                    _db.commit()
                finally:
                    _db.close()
            except Exception as db_exc:
                logger.error(
                    "PAPER: Failed to close old DB trade for {asset}: {err}",
                    asset=asset, err=db_exc,
                )
            # Cancel any dangling SL/TP orders from the old position
            for oid, order in list(_paper_state.orders.items()):
                if order.get("asset") == asset and order.get("status") == "open":
                    _paper_state.orders[oid]["status"] = "cancelled"

        _paper_state.positions[asset] = {
            "asset": asset,
            "side": side_str,
            "size_pct": decision.size_pct,
            "entry_price": fill_price,
            "leverage": decision.leverage,
            "stop_loss": decision.stop_loss,
            "take_profit": decision.take_profit,
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "unrealized_pnl": 0.0,
        }

        # Create stop-loss and take-profit "orders"
        stop_order_id = f"paper_sl_{uuid.uuid4().hex[:8]}"
        tp_order_id = f"paper_tp_{uuid.uuid4().hex[:8]}"

        _paper_state.orders[stop_order_id] = {
            "order_id": stop_order_id,
            "asset": asset,
            "type": "stop_loss",
            "trigger_price": decision.stop_loss,
            "side": "sell" if is_buy else "buy",
            "size_pct": decision.size_pct,
            "status": "open",
        }
        _paper_state.orders[tp_order_id] = {
            "order_id": tp_order_id,
            "asset": asset,
            "type": "take_profit",
            "trigger_price": decision.take_profit,
            "side": "sell" if is_buy else "buy",
            "size_pct": decision.size_pct,
            "status": "open",
        }

        order_result = {
            "order_id": order_id,
            "asset": asset,
            "side": side_str,
            "size_pct": decision.size_pct,
            "leverage": decision.leverage,
            "fill_price": fill_price,
            "stop_loss": decision.stop_loss,
            "take_profit": decision.take_profit,
            "status": "filled",
            "fees": round(fees, 6),
            "stop_order_id": stop_order_id,
            "tp_order_id": tp_order_id,
            "order_type": decision.order_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "live": False,
        }

        _paper_state.trade_history.append(order_result)

        logger.info(
            "PAPER ORDER FILLED | {asset} {side} @ {price} | "
            "SL={sl} TP={tp} | fees={fees}",
            asset=asset,
            side=side_str,
            price=fill_price,
            sl=decision.stop_loss,
            tp=decision.take_profit,
            fees=round(fees, 6),
        )
        return order_result

    def _cancel_paper_order(self, order_id: str) -> bool:
        """Cancel a simulated paper order."""
        if order_id in _paper_state.orders:
            _paper_state.orders[order_id]["status"] = "cancelled"
            logger.info("PAPER ORDER CANCELLED | oid={oid}", oid=order_id)
            return True
        logger.warning("PAPER CANCEL: order {oid} not found", oid=order_id)
        return False

    def _get_paper_open_orders(self, asset: str | None = None) -> list[dict[str, Any]]:
        """Return open paper orders, optionally filtered by asset."""
        result = []
        for order in _paper_state.orders.values():
            if order.get("status") != "open":
                continue
            if asset and order.get("asset") != asset:
                continue
            result.append(order)
        return result

    def _close_paper_position(
        self,
        asset: str,
        size: float | None = None,
    ) -> dict[str, Any]:
        """Close a simulated paper position."""
        if asset not in _paper_state.positions:
            return self._error_result(asset, f"No paper position for {asset}.")

        pos = _paper_state.positions[asset]
        mark_price = self._get_mark_price(asset) or pos["entry_price"]

        # Calculate P&L (including leverage for perp futures)
        entry = pos["entry_price"]
        is_long = pos["side"] == "long"
        leverage = pos.get("leverage", 1.0)

        if is_long:
            pnl_pct = (mark_price - entry) / entry * leverage if entry > 0 else 0.0
        else:
            pnl_pct = (entry - mark_price) / entry * leverage if entry > 0 else 0.0

        close_result = {
            "order_id": f"paper_close_{uuid.uuid4().hex[:8]}",
            "asset": asset,
            "side": f"close_{pos['side']}",
            "size_pct": size or pos["size_pct"],
            "entry_price": entry,
            "exit_price": mark_price,
            "pnl_pct": round(pnl_pct, 6),
            "status": "filled",
            "fees": 0.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "live": False,
        }

        # Remove the position and cancel associated orders
        del _paper_state.positions[asset]
        for oid, order in list(_paper_state.orders.items()):
            if order.get("asset") == asset and order.get("status") == "open":
                _paper_state.orders[oid]["status"] = "cancelled"

        _paper_state.trade_history.append(close_result)

        logger.info(
            "PAPER POSITION CLOSED | {asset} | entry={entry} exit={exit} pnl={pnl}%",
            asset=asset,
            entry=entry,
            exit=mark_price,
            pnl=round(pnl_pct * 100, 2),
        )
        return close_result

    # ------------------------------------------------------------------
    # Protective order placement (SL/TP with retry)
    # ------------------------------------------------------------------

    def _place_protective_orders(
        self,
        asset: str,
        is_buy: bool,
        sz: float,
        stop_loss: float | None,
        take_profit: float | None,
        max_retries: int = 3,
    ) -> tuple[str, str]:
        """Place stop-loss and take-profit orders with exponential backoff retry.

        If either order fails after all retries, returns an empty string for
        that order ID.  The caller is responsible for deciding whether to roll
        back the position.

        Args:
            asset: Asset symbol (e.g. ``"BTC"``).
            is_buy: True if the entry was a buy (long).
            sz: Position size in coins.
            stop_loss: Stop-loss trigger price, or None/0 to skip.
            take_profit: Take-profit trigger price, or None/0 to skip.
            max_retries: Maximum retry attempts per order (default 3).

        Returns:
            Tuple of (stop_order_id, tp_order_id).  Empty string means
            the order could not be placed.
        """
        stop_order_id = ""
        tp_order_id = ""

        if stop_loss and stop_loss > 0:
            stop_order_id = self._place_trigger_order_with_retry(
                asset=asset,
                is_buy=not is_buy,
                sz=sz,
                trigger_price=stop_loss,
                tpsl="sl",
                label="stop-loss",
                max_retries=max_retries,
            )

        if take_profit and take_profit > 0:
            tp_order_id = self._place_trigger_order_with_retry(
                asset=asset,
                is_buy=not is_buy,
                sz=sz,
                trigger_price=take_profit,
                tpsl="tp",
                label="take-profit",
                max_retries=max_retries,
            )

        return stop_order_id, tp_order_id

    def _place_trigger_order_with_retry(
        self,
        asset: str,
        is_buy: bool,
        sz: float,
        trigger_price: float,
        tpsl: str,
        label: str,
        max_retries: int = 3,
    ) -> str:
        """Place a single trigger order (SL or TP) with exponential backoff.

        Args:
            asset: Asset symbol.
            is_buy: Direction of the protective order (opposite of entry).
            sz: Size in coins.
            trigger_price: Trigger price for the order.
            tpsl: ``"sl"`` or ``"tp"``.
            label: Human-readable label for logging (e.g. ``"stop-loss"``).
            max_retries: Maximum number of attempts.

        Returns:
            Order ID string, or empty string on failure.
        """
        if self.exchange is None:
            logger.error("Cannot place {label}: exchange client not initialised.", label=label)
            return ""

        for attempt in range(1, max_retries + 1):
            try:
                result = self.exchange.order(
                    asset=asset,
                    is_buy=is_buy,
                    sz=sz,
                    limit_px=trigger_price,
                    order_type={
                        "trigger": {
                            "triggerPx": str(trigger_price),
                            "isMarket": True,
                            "tpsl": tpsl,
                        }
                    },
                    reduce_only=True,
                )
                order_id = self._extract_order_id(result)
                if attempt > 1:
                    logger.info(
                        "{label} placed on retry {attempt} for {asset}",
                        label=label, attempt=attempt, asset=asset,
                    )
                return order_id

            except Exception as exc:
                if attempt < max_retries:
                    backoff = 2 ** (attempt - 1)  # 1s, 2s, 4s
                    logger.warning(
                        "Failed to place {label} for {asset} (attempt {attempt}/{max}): "
                        "{err}. Retrying in {backoff}s...",
                        label=label, asset=asset, attempt=attempt,
                        max=max_retries, err=exc, backoff=backoff,
                    )
                    time.sleep(backoff)
                else:
                    logger.critical(
                        "CRITICAL: All {max} attempts to place {label} for {asset} "
                        "failed. Last error: {err}",
                        max=max_retries, label=label, asset=asset, err=exc,
                    )

        return ""

    # ------------------------------------------------------------------
    # Order fill verification (live mode)
    # ------------------------------------------------------------------

    def _verify_fill(
        self,
        order_id: str,
        asset: str,
        timeout_seconds: float = 30.0,
        poll_interval: float = 2.0,
    ) -> dict[str, Any]:
        """Verify an order's fill status by querying the exchange.

        Polls the exchange for the actual order status until the order is
        filled, partially filled, or the timeout expires.

        Args:
            order_id: The order ID to verify.
            asset: Asset symbol for the order.
            timeout_seconds: Max time to wait for fill confirmation.
            poll_interval: Seconds between status checks.

        Returns:
            Dict with keys:
                ``status``: ``"filled"``, ``"partial"``, ``"cancelled"``, ``"timeout"``
                ``filled_size``: Actual filled quantity (0 if not filled).
                ``avg_price``: Average fill price (0 if not filled).
                ``order_id``: The original order ID.
        """
        if self.info is None:
            logger.warning("Info client unavailable; cannot verify fill for oid={oid}", oid=order_id)
            return {"status": "unverified", "filled_size": 0.0, "avg_price": 0.0, "order_id": order_id}

        address = HYPERLIQUID_WALLET_ADDRESS
        start_time = time.time()

        while (time.time() - start_time) < timeout_seconds:
            try:
                # Check if the order is still in open orders
                open_orders = self.info.open_orders(address)
                still_open = any(
                    str(o.get("oid", "")) == order_id for o in open_orders
                )

                if not still_open:
                    # Order is no longer open -- check order status for fill details
                    try:
                        order_status = self.info.query_order_by_oid(address, int(order_id))
                        if isinstance(order_status, dict):
                            status = order_status.get("status", "")
                            filled_sz = float(order_status.get("szFilled", 0))
                            total_sz = float(order_status.get("sz", 0))
                            avg_px = float(order_status.get("avgPx", 0))

                            if status == "filled" or (filled_sz > 0 and filled_sz >= total_sz):
                                return {
                                    "status": "filled",
                                    "filled_size": filled_sz,
                                    "avg_price": avg_px,
                                    "order_id": order_id,
                                }
                            elif filled_sz > 0 and filled_sz < total_sz:
                                return {
                                    "status": "partial",
                                    "filled_size": filled_sz,
                                    "avg_price": avg_px,
                                    "order_id": order_id,
                                }
                            else:
                                return {
                                    "status": "cancelled",
                                    "filled_size": 0.0,
                                    "avg_price": 0.0,
                                    "order_id": order_id,
                                }
                    except Exception:
                        # query_order_by_oid may not be available; assume filled
                        # if the order disappeared from open orders
                        return {
                            "status": "filled",
                            "filled_size": 0.0,
                            "avg_price": 0.0,
                            "order_id": order_id,
                        }

            except Exception as exc:
                logger.warning(
                    "Error verifying fill for oid={oid}: {err}",
                    oid=order_id, err=exc,
                )

            time.sleep(poll_interval)

        # Timeout -- order still open
        logger.warning(
            "Fill verification timed out for oid={oid} after {timeout}s. "
            "Attempting to cancel.",
            oid=order_id, timeout=timeout_seconds,
        )

        # Try to cancel the unfilled order
        try:
            self.cancel_order(order_id, asset)
        except Exception as cancel_exc:
            logger.error(
                "Failed to cancel timed-out order oid={oid}: {err}",
                oid=order_id, err=cancel_exc,
            )

        return {
            "status": "timeout",
            "filled_size": 0.0,
            "avg_price": 0.0,
            "order_id": order_id,
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _get_mark_price(self, asset: str) -> float | None:
        """Fetch the current mark/mid price for an asset.

        Args:
            asset: Asset symbol (e.g. ``"BTC"``).

        Returns:
            Mark price as float, or None if unavailable.
        """
        if self.info is None:
            return None

        try:
            all_mids = self.info.all_mids()
            price_str = all_mids.get(asset)
            if price_str is not None:
                return float(price_str)
        except Exception as exc:
            logger.warning(
                "Failed to fetch mark price for {asset}: {err}",
                asset=asset, err=exc,
            )
        return None

    @staticmethod
    def _extract_order_id(result: Any) -> str:
        """Extract the order ID from a Hyperliquid SDK response.

        The SDK returns varying response shapes; this helper handles
        the common cases gracefully.

        Args:
            result: Raw response from ``exchange.order()``.

        Returns:
            Order ID as a string, or ``"unknown"`` if extraction fails.
        """
        try:
            if isinstance(result, dict):
                # Common pattern: {"status": "ok", "response": {"type": "order", "data": {"statuses": [{"resting": {"oid": 123}}]}}}
                statuses = (
                    result.get("response", {})
                    .get("data", {})
                    .get("statuses", [])
                )
                if statuses:
                    first = statuses[0]
                    if isinstance(first, dict):
                        for key in ("resting", "filled"):
                            if key in first and "oid" in first[key]:
                                return str(first[key]["oid"])
                # Fallback: top-level oid
                if "oid" in result:
                    return str(result["oid"])
            return str(result)
        except Exception:
            return "unknown"

    @staticmethod
    def _error_result(asset: str, error_msg: str) -> dict[str, Any]:
        """Build a standardised error result dict.

        Args:
            asset: The asset symbol.
            error_msg: Human-readable error description.

        Returns:
            Dict with ``status="error"`` and the error message.
        """
        return {
            "order_id": "",
            "asset": asset,
            "side": "",
            "size_pct": 0.0,
            "fill_price": 0.0,
            "status": "error",
            "error": error_msg,
            "fees": 0.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "live": LIVE_TRADING,
        }

    # ------------------------------------------------------------------
    # Paper state access (for testing / position manager)
    # ------------------------------------------------------------------

    @property
    def paper_positions(self) -> dict[str, dict[str, Any]]:
        """Read-only access to simulated paper positions."""
        return dict(_paper_state.positions)

    @property
    def paper_orders(self) -> dict[str, dict[str, Any]]:
        """Read-only access to simulated paper orders."""
        return dict(_paper_state.orders)

    @property
    def paper_trade_history(self) -> list[dict[str, Any]]:
        """Read-only access to simulated trade history."""
        return list(_paper_state.trade_history)

    def reset_paper_state(self) -> None:
        """Clear all paper-trading state (useful for tests)."""
        _paper_state.reset()
        logger.info("Paper trading state reset.")
