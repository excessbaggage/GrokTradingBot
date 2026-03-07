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

            # Place stop-loss order (trigger order)
            stop_order_id = ""
            if decision.stop_loss and decision.stop_loss > 0:
                try:
                    stop_result = self.exchange.order(
                        asset=asset,
                        is_buy=not is_buy,  # Opposite side to close
                        sz=sz,
                        limit_px=decision.stop_loss,
                        order_type={
                            "trigger": {
                                "triggerPx": str(decision.stop_loss),
                                "isMarket": True,
                                "tpsl": "sl",
                            }
                        },
                        reduce_only=True,
                    )
                    stop_order_id = self._extract_order_id(stop_result)
                except Exception as exc:
                    logger.error(
                        "Failed to place stop-loss for {asset}: {err}",
                        asset=asset, err=exc,
                    )

            # Place take-profit order (trigger order)
            tp_order_id = ""
            if decision.take_profit and decision.take_profit > 0:
                try:
                    tp_result = self.exchange.order(
                        asset=asset,
                        is_buy=not is_buy,
                        sz=sz,
                        limit_px=decision.take_profit,
                        order_type={
                            "trigger": {
                                "triggerPx": str(decision.take_profit),
                                "isMarket": True,
                                "tpsl": "tp",
                            }
                        },
                        reduce_only=True,
                    )
                    tp_order_id = self._extract_order_id(tp_result)
                except Exception as exc:
                    logger.error(
                        "Failed to place take-profit for {asset}: {err}",
                        asset=asset, err=exc,
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
        equity = portfolio_equity if portfolio_equity > 0 else 10_000.0  # fallback for tests
        notional = decision.size_pct * equity * decision.leverage
        fees = notional * _paper_state.simulated_fee_rate

        # Update paper positions
        side_str = "long" if is_buy else "short"
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

        # Calculate P&L
        entry = pos["entry_price"]
        is_long = pos["side"] == "long"

        if is_long:
            pnl_pct = (mark_price - entry) / entry if entry > 0 else 0.0
        else:
            pnl_pct = (entry - mark_price) / entry if entry > 0 else 0.0

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
