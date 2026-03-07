"""
Market data fetcher for Hyperliquid perpetual futures.

Retrieves OHLCV candles, funding rates, open interest, and order book
snapshots via the Hyperliquid Python SDK.  Includes retry logic via
``tenacity`` and switches between testnet/mainnet based on
``trading_config.LIVE_TRADING``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from hyperliquid.info import Info as InfoAPI
from hyperliquid.utils import constants as hl_constants

from config.trading_config import (
    CANDLE_INTERVALS,
    CANDLE_LOOKBACK,
    LIVE_TRADING,
)
from utils.logger import logger


# ═══════════════════════════════════════════════════════════════════════════
# RETRY DECORATOR
# ═══════════════════════════════════════════════════════════════════════════

_api_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, Exception)),
    reraise=True,
)


class MarketDataFetcher:
    """Fetches market data from Hyperliquid's Info API.

    Supports both testnet and mainnet, toggled by the ``LIVE_TRADING``
    config flag.

    Usage::

        fetcher = MarketDataFetcher()
        ohlcv = fetcher.fetch_ohlcv("BTC", "1h", limit=48)
        all_data = fetcher.fetch_all_market_data(["BTC", "ETH", "SOL"])
    """

    def __init__(self) -> None:
        base_url = (
            hl_constants.MAINNET_API_URL
            if LIVE_TRADING
            else hl_constants.TESTNET_API_URL
        )
        self._info = InfoAPI(base_url=base_url, skip_ws=True)
        logger.info(
            "MarketDataFetcher initialized | live={live} url={url}",
            live=LIVE_TRADING,
            url=base_url,
        )

    # -------------------------------------------------------------------
    # OHLCV
    # -------------------------------------------------------------------

    @_api_retry
    def fetch_ohlcv(
        self,
        asset: str,
        interval: str = "1h",
        limit: int = 48,
    ) -> pd.DataFrame:
        """Fetch OHLCV candle data for an asset.

        Args:
            asset: Symbol, e.g. ``"BTC"``.
            interval: Candle interval string (``"1h"``, ``"4h"``, ``"1d"``).
            limit: Maximum number of candles to retrieve.

        Returns:
            DataFrame with columns ``timestamp``, ``open``, ``high``,
            ``low``, ``close``, ``volume`` sorted by time ascending.
        """
        try:
            snapshot = self._info.candles_snapshot(
                coin=asset,
                interval=interval,
                n_candles=limit,
            )

            if not snapshot:
                logger.warning(
                    "Empty OHLCV response for {asset} {interval}",
                    asset=asset,
                    interval=interval,
                )
                return pd.DataFrame(
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )

            rows: list[dict[str, Any]] = []
            for candle in snapshot:
                rows.append(
                    {
                        "timestamp": datetime.fromtimestamp(
                            candle["t"] / 1000, tz=timezone.utc
                        ),
                        "open": float(candle["o"]),
                        "high": float(candle["h"]),
                        "low": float(candle["l"]),
                        "close": float(candle["c"]),
                        "volume": float(candle["v"]),
                    }
                )

            df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
            logger.debug(
                "Fetched {n} candles for {asset} {interval}",
                n=len(df),
                asset=asset,
                interval=interval,
            )
            return df

        except Exception as exc:
            logger.error(
                "Failed to fetch OHLCV for {asset} {interval}: {err}",
                asset=asset,
                interval=interval,
                err=exc,
            )
            raise

    # -------------------------------------------------------------------
    # FUNDING RATE
    # -------------------------------------------------------------------

    @_api_retry
    def fetch_funding_rate(self, asset: str) -> dict[str, Any]:
        """Fetch current and 7-day average funding rate for an asset.

        Args:
            asset: Symbol, e.g. ``"BTC"``.

        Returns:
            Dict with keys ``current_rate``, ``avg_7d_rate``,
            ``premium``, ``asset``.
        """
        try:
            meta = self._info.meta()
            asset_meta = None
            for u in meta.get("universe", []):
                if u.get("name", "").upper() == asset.upper():
                    asset_meta = u
                    break

            # Fetch current funding from the clearinghouse state
            ctx_list = self._info.meta_and_asset_ctxs()
            asset_ctx = None
            if isinstance(ctx_list, list) and len(ctx_list) >= 2:
                asset_ctxs = ctx_list[1]
                universe = ctx_list[0].get("universe", [])
                for i, u in enumerate(universe):
                    if u.get("name", "").upper() == asset.upper() and i < len(asset_ctxs):
                        asset_ctx = asset_ctxs[i]
                        break

            current_rate = float(asset_ctx.get("funding", 0)) if asset_ctx else 0.0
            premium = float(asset_ctx.get("premium", 0)) if asset_ctx else 0.0

            # Fetch historical funding for 7d average
            try:
                funding_history = self._info.funding_history(
                    coin=asset,
                    startTime=int(
                        (datetime.now(timezone.utc).timestamp() - 7 * 86400) * 1000
                    ),
                )
                if funding_history:
                    rates = [float(f.get("fundingRate", 0)) for f in funding_history]
                    avg_7d = sum(rates) / len(rates) if rates else 0.0
                else:
                    avg_7d = current_rate
            except Exception:
                avg_7d = current_rate

            result = {
                "asset": asset,
                "current_rate": current_rate,
                "avg_7d_rate": avg_7d,
                "premium": premium,
            }
            logger.debug("Funding rate for {asset}: {r}", asset=asset, r=result)
            return result

        except Exception as exc:
            logger.error(
                "Failed to fetch funding rate for {asset}: {err}",
                asset=asset,
                err=exc,
            )
            return {
                "asset": asset,
                "current_rate": 0.0,
                "avg_7d_rate": 0.0,
                "premium": 0.0,
            }

    # -------------------------------------------------------------------
    # OPEN INTEREST
    # -------------------------------------------------------------------

    @_api_retry
    def fetch_open_interest(self, asset: str) -> dict[str, Any]:
        """Fetch open interest data for an asset.

        Args:
            asset: Symbol, e.g. ``"BTC"``.

        Returns:
            Dict with keys ``asset``, ``current_oi``, ``oi_24h_change_pct``.
        """
        try:
            ctx_list = self._info.meta_and_asset_ctxs()
            asset_ctx = None
            if isinstance(ctx_list, list) and len(ctx_list) >= 2:
                universe = ctx_list[0].get("universe", [])
                asset_ctxs = ctx_list[1]
                for i, u in enumerate(universe):
                    if u.get("name", "").upper() == asset.upper() and i < len(asset_ctxs):
                        asset_ctx = asset_ctxs[i]
                        break

            current_oi = float(asset_ctx.get("openInterest", 0)) if asset_ctx else 0.0
            day_ntl_vlm = float(asset_ctx.get("dayNtlVlm", 0)) if asset_ctx else 0.0
            prev_day_px = float(asset_ctx.get("prevDayPx", 0)) if asset_ctx else 0.0

            # Estimate 24h OI change (the SDK doesn't provide historical OI
            # directly, so we store 0 and let the context builder note the
            # limitation)
            oi_24h_change_pct = 0.0

            result = {
                "asset": asset,
                "current_oi": current_oi,
                "oi_24h_change_pct": oi_24h_change_pct,
                "day_ntl_vlm": day_ntl_vlm,
                "prev_day_px": prev_day_px,
            }
            logger.debug("Open interest for {asset}: {r}", asset=asset, r=result)
            return result

        except Exception as exc:
            logger.error(
                "Failed to fetch OI for {asset}: {err}",
                asset=asset,
                err=exc,
            )
            return {
                "asset": asset,
                "current_oi": 0.0,
                "oi_24h_change_pct": 0.0,
                "day_ntl_vlm": 0.0,
                "prev_day_px": 0.0,
            }

    # -------------------------------------------------------------------
    # ORDER BOOK
    # -------------------------------------------------------------------

    @_api_retry
    def fetch_order_book(self, asset: str, depth: int = 10) -> dict[str, Any]:
        """Fetch the current order book snapshot.

        Args:
            asset: Symbol, e.g. ``"BTC"``.
            depth: Number of price levels per side.

        Returns:
            Dict with ``bids`` and ``asks`` lists, each item being
            ``{"price": float, "size": float}``, plus ``spread`` and
            ``mid_price``.
        """
        try:
            book = self._info.l2_snapshot(coin=asset)

            bids_raw = book.get("levels", [[]])[0] if book.get("levels") else []
            asks_raw = book.get("levels", [[], []])[1] if book.get("levels") and len(book["levels"]) > 1 else []

            bids = [
                {"price": float(b["px"]), "size": float(b["sz"])}
                for b in bids_raw[:depth]
            ]
            asks = [
                {"price": float(a["px"]), "size": float(a["sz"])}
                for a in asks_raw[:depth]
            ]

            best_bid = bids[0]["price"] if bids else 0.0
            best_ask = asks[0]["price"] if asks else 0.0
            spread = best_ask - best_bid if best_bid and best_ask else 0.0
            mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0.0

            result = {
                "asset": asset,
                "bids": bids,
                "asks": asks,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread": spread,
                "mid_price": mid_price,
            }
            logger.debug(
                "Order book for {asset}: bid={bid} ask={ask} spread={sp}",
                asset=asset,
                bid=best_bid,
                ask=best_ask,
                sp=spread,
            )
            return result

        except Exception as exc:
            logger.error(
                "Failed to fetch order book for {asset}: {err}",
                asset=asset,
                err=exc,
            )
            return {
                "asset": asset,
                "bids": [],
                "asks": [],
                "best_bid": 0.0,
                "best_ask": 0.0,
                "spread": 0.0,
                "mid_price": 0.0,
            }

    # -------------------------------------------------------------------
    # AGGREGATE: all data for multiple assets
    # -------------------------------------------------------------------

    def fetch_all_market_data(
        self,
        assets: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Fetch all market data for the given assets.

        Gathers OHLCV candles (for every configured interval), funding
        rates, open interest, and order book snapshots.

        Args:
            assets: List of asset symbols.  Defaults to
                    ``trading_config.ASSET_UNIVERSE``.

        Returns:
            Nested dict keyed by asset symbol, each containing
            ``candles`` (keyed by interval), ``funding``, ``oi``,
            ``order_book``, and ``price`` (latest close from 1h candles).
        """
        from config.trading_config import ASSET_UNIVERSE

        if assets is None:
            assets = ASSET_UNIVERSE

        result: dict[str, dict[str, Any]] = {}

        for asset in assets:
            logger.info("Fetching market data for {asset}...", asset=asset)

            asset_data: dict[str, Any] = {"asset": asset, "candles": {}}

            # -- OHLCV candles for each configured interval
            for interval in CANDLE_INTERVALS:
                lookback = CANDLE_LOOKBACK.get(interval, 20)
                try:
                    df = self.fetch_ohlcv(asset, interval=interval, limit=lookback)
                    asset_data["candles"][interval] = df
                except Exception as exc:
                    logger.warning(
                        "Candle fetch failed for {asset} {interval}: {err}",
                        asset=asset,
                        interval=interval,
                        err=exc,
                    )
                    asset_data["candles"][interval] = pd.DataFrame(
                        columns=["timestamp", "open", "high", "low", "close", "volume"]
                    )

            # -- Latest price from the most granular candle data
            hourly = asset_data["candles"].get("1h")
            if hourly is not None and not hourly.empty:
                asset_data["price"] = float(hourly["close"].iloc[-1])
                first_close = float(hourly["close"].iloc[0])
                if first_close != 0:
                    asset_data["24h_change_pct"] = (
                        (asset_data["price"] - first_close) / first_close
                    )
                else:
                    asset_data["24h_change_pct"] = 0.0
            else:
                asset_data["price"] = 0.0
                asset_data["24h_change_pct"] = 0.0

            # -- Funding rate
            asset_data["funding"] = self.fetch_funding_rate(asset)

            # -- Open interest
            asset_data["oi"] = self.fetch_open_interest(asset)

            # -- Order book
            asset_data["order_book"] = self.fetch_order_book(asset, depth=10)

            result[asset] = asset_data

        logger.info(
            "Market data fetch complete for {n} assets", n=len(result)
        )
        return result
