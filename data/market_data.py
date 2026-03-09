"""
Market data fetcher for Hyperliquid perpetual futures.

Retrieves OHLCV candles, funding rates, open interest, and order book
snapshots via the Hyperliquid Python SDK.  Includes retry logic via
``tenacity`` and switches between testnet/mainnet based on
``trading_config.LIVE_TRADING``.
"""

from __future__ import annotations

import time
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
# INTERVAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════

# Mapping from interval string to duration in seconds
_INTERVAL_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


# ═══════════════════════════════════════════════════════════════════════════
# RETRY DECORATOR
# ═══════════════════════════════════════════════════════════════════════════

_api_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
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
        # Pass empty spot_meta to skip spot token initialization — we only
        # trade perpetual futures, and the testnet spot metadata is often
        # incomplete which causes IndexError inside the SDK.
        self._info = InfoAPI(
            base_url=base_url,
            skip_ws=True,
            spot_meta={"universe": [], "tokens": []},
        )
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
            # SDK v0.22+ uses positional (name, interval, startTime, endTime).
            # Compute startTime from the desired number of candles.
            interval_secs = _INTERVAL_SECONDS.get(interval, 3600)
            end_ms = int(time.time() * 1000)
            start_ms = end_ms - (limit * interval_secs * 1000)

            snapshot = self._info.candles_snapshot(
                asset,          # name (positional)
                interval,       # interval
                start_ms,       # startTime (epoch ms)
                end_ms,         # endTime (epoch ms)
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
                    asset,
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
            book = self._info.l2_snapshot(asset)

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
    # TECHNICAL INDICATORS (computed from OHLCV, no extra API calls)
    # -------------------------------------------------------------------

    @staticmethod
    def compute_technical_indicators(
        candles_1h: pd.DataFrame,
        candles_4h: pd.DataFrame,
    ) -> dict[str, Any]:
        """Compute ATR, RSI, and adaptive RSI thresholds from candle data.

        Inspired by the AlphaRSIPro strategy from ai-trader:
        - ATR (Average True Range) for volatility-based position sizing
        - RSI with adaptive overbought/oversold thresholds that widen in
          high-volatility environments and tighten in low-volatility ones.

        Args:
            candles_1h: 1-hour OHLCV DataFrame.
            candles_4h: 4-hour OHLCV DataFrame.

        Returns:
            Dict with ``atr_14``, ``atr_pct``, ``rsi_14``,
            ``adaptive_ob`` (overbought threshold),
            ``adaptive_os`` (oversold threshold),
            ``volatility_regime``, ``turtle_size_factor``.
        """
        result: dict[str, Any] = {
            "atr_14": 0.0,
            "atr_pct": 0.0,
            "rsi_14": 50.0,
            "adaptive_ob": 70.0,
            "adaptive_os": 30.0,
            "volatility_regime": "normal",
            "turtle_size_factor": 1.0,
        }

        # --- ATR-14 from 1h candles ---
        if candles_1h is not None and len(candles_1h) >= 15:
            df = candles_1h.copy()
            high = df["high"]
            low = df["low"]
            close = df["close"]

            # True Range
            prev_close = close.shift(1)
            tr = pd.concat([
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ], axis=1).max(axis=1)

            atr_14 = tr.rolling(14).mean().iloc[-1]
            latest_close = close.iloc[-1]

            if pd.notna(atr_14) and latest_close > 0:
                result["atr_14"] = round(float(atr_14), 4)
                result["atr_pct"] = round(float(atr_14 / latest_close), 6)

                # Volatility ratio = ATR(14) / SMA(ATR(14), 50)
                # (ai-trader AlphaRSIPro pattern)
                atr_series = tr.rolling(14).mean()
                atr_sma_50 = atr_series.rolling(min(50, len(atr_series))).mean().iloc[-1]

                if pd.notna(atr_sma_50) and atr_sma_50 > 0:
                    vol_ratio = float(atr_14 / atr_sma_50)
                    adjustment = (vol_ratio - 1) * 20  # Scale factor from ai-trader

                    # Adaptive RSI thresholds
                    ob = round(min(70 + adjustment, 85), 1)
                    os_ = round(max(30 - adjustment, 15), 1)
                    # Guard against threshold inversion in extreme regimes
                    if ob <= os_:
                        ob, os_ = 70.0, 30.0  # Reset to defaults
                    result["adaptive_ob"] = ob
                    result["adaptive_os"] = os_

                    if vol_ratio > 1.3:
                        result["volatility_regime"] = "high"
                    elif vol_ratio < 0.7:
                        result["volatility_regime"] = "low"
                    else:
                        result["volatility_regime"] = "normal"

                # Turtle position sizing factor: 1 / (ATR_pct * 100)
                # Higher ATR = smaller suggested position
                atr_pct = result["atr_pct"]
                if atr_pct > 0:
                    result["turtle_size_factor"] = round(
                        min(0.01 / atr_pct, 3.0), 2  # Cap at 3x
                    )

        # --- RSI-14 from 1h candles ---
        if candles_1h is not None and len(candles_1h) >= 15:
            close = candles_1h["close"]
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = (-delta.clip(upper=0))

            avg_gain = gain.rolling(14).mean().iloc[-1]
            avg_loss = loss.rolling(14).mean().iloc[-1]

            if pd.notna(avg_gain) and pd.notna(avg_loss) and avg_loss > 0:
                rs = avg_gain / avg_loss
                result["rsi_14"] = round(100 - (100 / (1 + rs)), 1)
            elif pd.notna(avg_gain) and (pd.isna(avg_loss) or avg_loss == 0):
                # If avg_gain is also zero (flat market), RSI is neutral (50)
                # Only set to 100 when there are actual gains with zero losses
                result["rsi_14"] = 100.0 if avg_gain > 0 else 50.0

        return result

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

            # -- Technical indicators (ATR, RSI, adaptive thresholds)
            asset_data["technicals"] = self.compute_technical_indicators(
                candles_1h=asset_data["candles"].get("1h"),
                candles_4h=asset_data["candles"].get("4h"),
            )

            result[asset] = asset_data

        logger.info(
            "Market data fetch complete for {n} assets", n=len(result)
        )
        return result
