"""
Historical data loader for backtesting.

Supports loading from the Hyperliquid API (via MarketDataFetcher patterns),
CSV files, and synthetic random-walk generation for quick testing.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


class HistoricalDataLoader:
    """Loads and manages historical OHLCV data for backtesting.

    Provides multiple data sources: live Hyperliquid API fetch, CSV file
    import/export, and synthetic data generation.

    Usage::

        loader = HistoricalDataLoader()

        # Synthetic data (no API keys needed)
        data = loader.generate_synthetic("BTC", days=90, volatility=0.02)

        # Save / reload from CSV
        loader.save_to_csv(data, "btc_90d.csv")
        data = loader.load_from_csv("btc_90d.csv")
    """

    # Expected columns in all DataFrames produced by this loader.
    REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

    # ------------------------------------------------------------------
    # Hyperliquid API loader
    # ------------------------------------------------------------------

    def load_from_hyperliquid(
        self,
        asset: str,
        interval: str = "1h",
        start_date: datetime | str | None = None,
        end_date: datetime | str | None = None,
    ) -> pd.DataFrame:
        """Fetch historical candles from the Hyperliquid API.

        Uses the same ``MarketDataFetcher`` patterns as the live bot.
        Requires network access but no API key (Hyperliquid's public info
        endpoint is unauthenticated).

        Args:
            asset: Symbol, e.g. ``"BTC"``, ``"ETH"``, ``"SOL"``.
            interval: Candle interval (``"1h"``, ``"4h"``, ``"1d"``).
            start_date: Start of the historical range (inclusive).
                        Accepts ``datetime`` or ISO-8601 string.
            end_date: End of the historical range (inclusive).
                      Defaults to now.

        Returns:
            DataFrame with columns: timestamp, open, high, low, close,
            volume -- sorted by timestamp ascending.
        """
        # Lazy import to avoid initialising the SDK at module level.
        from data.market_data import MarketDataFetcher, _INTERVAL_SECONDS

        fetcher = MarketDataFetcher()

        # Parse dates
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)

        if start_date is None:
            start_date = datetime.now(timezone.utc) - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now(timezone.utc)

        # Ensure timezone-aware
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        # Calculate how many candles we need
        interval_secs = _INTERVAL_SECONDS.get(interval, 3600)
        total_seconds = (end_date - start_date).total_seconds()
        limit = int(total_seconds / interval_secs) + 1

        # The SDK limits candle requests; fetch in chunks if needed.
        max_chunk = 5000
        all_frames: list[pd.DataFrame] = []
        current_start = start_date

        while current_start < end_date:
            chunk_limit = min(max_chunk, limit)
            try:
                df = fetcher.fetch_ohlcv(
                    asset=asset,
                    interval=interval,
                    limit=chunk_limit,
                )
                if df.empty:
                    break
                all_frames.append(df)

                # Advance start time past the last candle we received
                last_ts = df["timestamp"].max()
                current_start = last_ts + timedelta(seconds=interval_secs)
                limit -= len(df)

                if limit <= 0:
                    break
            except Exception as exc:
                logger.error(
                    "Failed to fetch chunk for {asset} {interval}: {err}",
                    asset=asset,
                    interval=interval,
                    err=exc,
                )
                break

        if not all_frames:
            logger.warning(
                "No data fetched for {asset} {interval}", asset=asset, interval=interval,
            )
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS)

        result = (
            pd.concat(all_frames, ignore_index=True)
            .drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        # Trim to requested range
        result = result[
            (result["timestamp"] >= start_date) & (result["timestamp"] <= end_date)
        ].reset_index(drop=True)

        logger.info(
            "Loaded {n} candles for {asset} {interval} from Hyperliquid",
            n=len(result),
            asset=asset,
            interval=interval,
        )
        return result

    # ------------------------------------------------------------------
    # CSV loader / saver
    # ------------------------------------------------------------------

    def load_from_csv(self, filepath: str | Path) -> pd.DataFrame:
        """Load OHLCV data from a CSV file.

        Expected columns: timestamp, open, high, low, close, volume.
        The timestamp column is parsed into timezone-aware UTC datetimes.

        Args:
            filepath: Path to the CSV file.

        Returns:
            DataFrame with standard columns sorted by timestamp.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If required columns are missing.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")

        df = pd.read_csv(filepath)

        # Validate columns
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {missing}. "
                f"Expected: {self.REQUIRED_COLUMNS}"
            )

        # Parse timestamps
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Ensure numeric types
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.sort_values("timestamp").reset_index(drop=True)

        logger.info("Loaded {n} candles from {path}", n=len(df), path=str(filepath))
        return df

    def save_to_csv(self, data: pd.DataFrame, filepath: str | Path) -> None:
        """Save OHLCV data to a CSV file for re-use.

        Args:
            data: DataFrame with standard OHLCV columns.
            filepath: Destination path. Parent directories are created
                      automatically.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert timestamps to ISO format for portability
        df = data.copy()
        if "timestamp" in df.columns:
            df["timestamp"] = df["timestamp"].apply(
                lambda ts: ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            )

        df.to_csv(filepath, index=False)
        logger.info("Saved {n} candles to {path}", n=len(df), path=str(filepath))

    # ------------------------------------------------------------------
    # Synthetic data generator
    # ------------------------------------------------------------------

    def generate_synthetic(
        self,
        asset: str = "BTC",
        days: int = 90,
        volatility: float = 0.02,
        start_price: float | None = None,
        interval: str = "1h",
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Generate synthetic OHLCV data using a geometric random walk.

        Useful for rapid testing without API access. The random walk
        produces realistic-looking price action with configurable
        volatility.

        Args:
            asset: Symbol name (stored in metadata, not in columns).
            days: Number of days of data to generate.
            volatility: Per-candle return standard deviation (e.g. 0.02
                        = 2% per candle). Controls how "noisy" the
                        price action looks.
            start_price: Opening price of the first candle. Defaults to
                         typical prices for known assets.
            interval: Candle interval for timestamp spacing.
            seed: Random seed for reproducibility.

        Returns:
            DataFrame with standard OHLCV columns.
        """
        if seed is not None:
            np.random.seed(seed)

        # Default start prices for known assets
        default_prices = {"BTC": 65000.0, "ETH": 3500.0, "SOL": 150.0}
        if start_price is None:
            start_price = default_prices.get(asset.upper(), 100.0)

        # Calculate number of candles
        interval_hours = {
            "1m": 1 / 60, "5m": 5 / 60, "15m": 0.25,
            "1h": 1, "4h": 4, "1d": 24,
        }
        hours_per_candle = interval_hours.get(interval, 1)
        n_candles = int((days * 24) / hours_per_candle)

        # Generate log-returns with slight positive drift (crypto tends up)
        drift = 0.0001  # Slight positive drift per candle
        returns = np.random.normal(drift, volatility, n_candles)

        # Build close prices via cumulative product
        close_prices = start_price * np.cumprod(1 + returns)

        # Generate OHLC from close prices with intra-candle noise
        opens = np.zeros(n_candles)
        highs = np.zeros(n_candles)
        lows = np.zeros(n_candles)
        volumes = np.zeros(n_candles)

        opens[0] = start_price
        for i in range(1, n_candles):
            opens[i] = close_prices[i - 1]

        for i in range(n_candles):
            # Intra-candle range: high/low deviate from open-close range
            candle_range = abs(close_prices[i] - opens[i])
            wick_size = max(candle_range * 0.3, start_price * volatility * 0.1)

            highs[i] = max(opens[i], close_prices[i]) + abs(np.random.normal(0, wick_size))
            lows[i] = min(opens[i], close_prices[i]) - abs(np.random.normal(0, wick_size))

            # Ensure low > 0
            lows[i] = max(lows[i], close_prices[i] * 0.9)

            # Volume: base volume with some randomness, higher on volatile candles
            base_vol = start_price * 1000
            vol_multiplier = 1 + abs(returns[i]) / volatility
            volumes[i] = abs(np.random.normal(base_vol * vol_multiplier, base_vol * 0.3))

        # Build timestamps
        start_time = datetime.now(timezone.utc) - timedelta(days=days)
        timestamps = [
            start_time + timedelta(hours=i * hours_per_candle) for i in range(n_candles)
        ]

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": np.round(opens, 2),
                "high": np.round(highs, 2),
                "low": np.round(lows, 2),
                "close": np.round(close_prices, 2),
                "volume": np.round(volumes, 2),
            }
        )

        logger.info(
            "Generated {n} synthetic candles for {asset} | "
            "start_price={sp} volatility={vol} days={d}",
            n=len(df),
            asset=asset,
            sp=start_price,
            vol=volatility,
            d=days,
        )
        return df
