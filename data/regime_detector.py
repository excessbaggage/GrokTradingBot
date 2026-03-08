"""
Market regime detection using ADX, Choppiness Index, and Bollinger Band width.

Classifies each asset into one of five regimes -- TRENDING_UP, TRENDING_DOWN,
RANGING, VOLATILE_EXPANSION, or MEAN_REVERTING -- and provides strategy
recommendations and confidence scores for each classification.

The detector runs on 1-hour OHLCV candle data and requires at least 30 rows
to produce meaningful results.  When insufficient data is available it
gracefully falls back to RANGING with low confidence.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from utils.logger import logger


# ======================================================================
# Regime Enum
# ======================================================================


class MarketRegime(enum.Enum):
    """The five detectable market regimes."""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE_EXPANSION = "volatile_expansion"
    MEAN_REVERTING = "mean_reverting"


# ======================================================================
# Strategy recommendations per regime
# ======================================================================

REGIME_STRATEGY_MAP: dict[str, dict[str, Any]] = {
    "trending_up": {
        "preferred_strategy": "momentum_long",
        "avoid": ["mean_reversion_short"],
        "position_size_multiplier": 1.2,
        "stop_distance_multiplier": 1.5,
        "description": "Strong uptrend -- favor momentum longs, trail stops wider",
    },
    "trending_down": {
        "preferred_strategy": "momentum_short",
        "avoid": ["mean_reversion_long"],
        "position_size_multiplier": 1.2,
        "stop_distance_multiplier": 1.5,
        "description": "Strong downtrend -- favor momentum shorts, trail stops wider",
    },
    "ranging": {
        "preferred_strategy": "mean_reversion",
        "avoid": ["momentum", "breakout"],
        "position_size_multiplier": 0.7,
        "stop_distance_multiplier": 0.8,
        "description": "Choppy/sideways -- fade extremes, reduce size, tighten stops",
    },
    "volatile_expansion": {
        "preferred_strategy": "breakout",
        "avoid": ["mean_reversion"],
        "position_size_multiplier": 0.6,
        "stop_distance_multiplier": 2.0,
        "description": "Volatility expanding -- breakout setups, smaller size, wider stops",
    },
    "mean_reverting": {
        "preferred_strategy": "mean_reversion",
        "avoid": ["momentum"],
        "position_size_multiplier": 0.8,
        "stop_distance_multiplier": 1.0,
        "description": "Overextended -- fade the move with tight stops",
    },
}


# ======================================================================
# RegimeState dataclass
# ======================================================================


@dataclass
class RegimeState:
    """Current regime detection result for a single asset."""

    asset: str
    regime: MarketRegime
    confidence: float  # 0.0 to 1.0
    adx: float
    plus_di: float
    minus_di: float
    choppiness_index: float
    bb_width: float
    bb_width_percentile: float  # Where current BB width sits vs last 100 candles
    strategy_recommendation: dict[str, Any]


# ======================================================================
# Minimum candle requirement
# ======================================================================

_MIN_CANDLES = 30


# ======================================================================
# RegimeDetector
# ======================================================================


class RegimeDetector:
    """Detect market regime from OHLCV candle data.

    The detector computes ADX(14), Choppiness Index(14), Bollinger Band
    Width(20, 2), RSI(14), and ATR(14) from 1-hour candles.  It then
    classifies the regime using a prioritised scoring system:

        1. Mean Reverting (overextended price vs SMA-50 + RSI extreme)
        2. Volatile Expansion (BB width percentile > 80)
        3. Trending Up (ADX > 25, +DI > -DI)
        4. Trending Down (ADX > 25, -DI > +DI)
        5. Ranging (Choppiness > 61.8 or ADX < 20)

    If none of the conditions are met, it defaults to RANGING at 0.5
    confidence.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        asset: str,
        candles_1h: pd.DataFrame,
        candles_4h: pd.DataFrame | None = None,
    ) -> RegimeState:
        """Analyse candles and classify the current market regime.

        Args:
            asset: Symbol, e.g. ``"BTC"``.
            candles_1h: 1-hour OHLCV DataFrame.  Must contain at least
                ``_MIN_CANDLES`` rows with columns ``open``, ``high``,
                ``low``, ``close``, ``volume``.
            candles_4h: Optional 4-hour OHLCV DataFrame (reserved for
                future multi-timeframe confirmation).

        Returns:
            A ``RegimeState`` with the detected regime, confidence score,
            indicator values, and strategy recommendation.
        """
        # ---- guard: insufficient data --------------------------------
        if candles_1h is None or len(candles_1h) < _MIN_CANDLES:
            logger.warning(
                "Insufficient candle data for {asset}: "
                "{n} rows (need {min})",
                asset=asset,
                n=0 if candles_1h is None else len(candles_1h),
                min=_MIN_CANDLES,
            )
            return self._fallback_state(asset)

        try:
            df = candles_1h.copy().reset_index(drop=True)

            # ---- Compute indicators ----------------------------------
            adx, plus_di, minus_di = self.compute_adx(df, period=14)
            choppiness = self.compute_choppiness_index(df, period=14)
            bb_width, bb_width_pctl = self.compute_bb_width(
                df, period=20, std_dev=2.0,
            )
            rsi = self._compute_rsi(df, period=14)
            atr_14 = self._compute_atr(df, period=14)
            price = float(df["close"].iloc[-1])

            # SMA-50 -- use available data if fewer than 50 candles
            sma_period = min(50, len(df))
            sma_50 = float(df["close"].rolling(sma_period).mean().iloc[-1])

            # ATR as percentage of price
            atr_pct = atr_14 / price if price > 0 else 0.0

            # ---- Classify --------------------------------------------
            regime, confidence = self._classify_regime(
                adx=adx,
                plus_di=plus_di,
                minus_di=minus_di,
                choppiness=choppiness,
                bb_width=bb_width,
                bb_width_pctl=bb_width_pctl,
                rsi=rsi,
                atr_pct=atr_pct,
                price=price,
                sma_50=sma_50,
                atr_14=atr_14,
            )

            strategy = REGIME_STRATEGY_MAP.get(
                regime.value,
                REGIME_STRATEGY_MAP["ranging"],
            )

            state = RegimeState(
                asset=asset,
                regime=regime,
                confidence=confidence,
                adx=round(adx, 2),
                plus_di=round(plus_di, 2),
                minus_di=round(minus_di, 2),
                choppiness_index=round(choppiness, 2),
                bb_width=round(bb_width, 4),
                bb_width_percentile=round(bb_width_pctl, 1),
                strategy_recommendation=strategy,
            )

            logger.debug(
                "Regime detected | asset={asset} regime={regime} "
                "confidence={conf:.0%} ADX={adx:.1f} CI={ci:.1f} "
                "BBW_pctl={bbp:.0f}%",
                asset=asset,
                regime=regime.value,
                conf=confidence,
                adx=adx,
                ci=choppiness,
                bbp=bb_width_pctl,
            )
            return state

        except Exception as exc:
            logger.error(
                "Regime detection failed for {asset}: {err}",
                asset=asset,
                err=exc,
            )
            return self._fallback_state(asset)

    # ------------------------------------------------------------------
    # Static indicator computations
    # ------------------------------------------------------------------

    @staticmethod
    def compute_adx(
        df: pd.DataFrame,
        period: int = 14,
    ) -> tuple[float, float, float]:
        """Compute ADX, +DI, and -DI using Wilder's smoothing method.

        Args:
            df: OHLCV DataFrame with ``high``, ``low``, ``close`` columns.
            period: Lookback period (default 14).

        Returns:
            Tuple of ``(adx, plus_di, minus_di)``.
        """
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        close = df["close"].values.astype(float)
        n = len(df)

        if n < period + 1:
            return (0.0, 0.0, 0.0)

        # True Range, +DM, -DM
        tr = np.zeros(n)
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)

        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)

            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]

            plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
            minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0

        # Wilder's smoothing (EMA with alpha = 1/period)
        alpha = 1.0 / period

        smoothed_tr = np.zeros(n)
        smoothed_plus_dm = np.zeros(n)
        smoothed_minus_dm = np.zeros(n)

        # Seed with the first ``period`` values
        smoothed_tr[period] = np.sum(tr[1 : period + 1])
        smoothed_plus_dm[period] = np.sum(plus_dm[1 : period + 1])
        smoothed_minus_dm[period] = np.sum(minus_dm[1 : period + 1])

        for i in range(period + 1, n):
            smoothed_tr[i] = smoothed_tr[i - 1] - (smoothed_tr[i - 1] / period) + tr[i]
            smoothed_plus_dm[i] = (
                smoothed_plus_dm[i - 1]
                - (smoothed_plus_dm[i - 1] / period)
                + plus_dm[i]
            )
            smoothed_minus_dm[i] = (
                smoothed_minus_dm[i - 1]
                - (smoothed_minus_dm[i - 1] / period)
                + minus_dm[i]
            )

        # +DI, -DI
        plus_di_arr = np.zeros(n)
        minus_di_arr = np.zeros(n)
        dx_arr = np.zeros(n)

        for i in range(period, n):
            if smoothed_tr[i] > 0:
                plus_di_arr[i] = 100.0 * smoothed_plus_dm[i] / smoothed_tr[i]
                minus_di_arr[i] = 100.0 * smoothed_minus_dm[i] / smoothed_tr[i]
            di_sum = plus_di_arr[i] + minus_di_arr[i]
            if di_sum > 0:
                dx_arr[i] = 100.0 * abs(plus_di_arr[i] - minus_di_arr[i]) / di_sum

        # ADX = Wilder-smoothed DX
        adx_arr = np.zeros(n)
        adx_start = 2 * period  # Need ``period`` DX values to seed ADX
        if adx_start < n:
            adx_arr[adx_start] = np.mean(dx_arr[period : adx_start + 1])
            for i in range(adx_start + 1, n):
                adx_arr[i] = (adx_arr[i - 1] * (period - 1) + dx_arr[i]) / period

        adx = float(adx_arr[-1])
        plus_di = float(plus_di_arr[-1])
        minus_di = float(minus_di_arr[-1])

        return (adx, plus_di, minus_di)

    @staticmethod
    def compute_choppiness_index(
        df: pd.DataFrame,
        period: int = 14,
    ) -> float:
        """Compute the Choppiness Index.

        Formula:
            CI = 100 * LOG10(SUM(ATR(1), n) / (HH - LL)) / LOG10(n)

        Values above 61.8 indicate a choppy/sideways market; values
        below 38.2 indicate a trending market.

        Args:
            df: OHLCV DataFrame with ``high``, ``low``, ``close`` columns.
            period: Lookback period (default 14).

        Returns:
            Choppiness Index value (float).
        """
        if len(df) < period + 1:
            return 50.0  # Neutral default

        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        close = df["close"].values.astype(float)

        # ATR(1) = True Range for each bar
        n = len(df)
        tr = np.zeros(n)
        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)

        # Use the last ``period`` bars
        tr_sum = float(np.sum(tr[-period:]))
        highest_high = float(np.max(high[-period:]))
        lowest_low = float(np.min(low[-period:]))

        hl_range = highest_high - lowest_low
        if hl_range <= 0:
            return 100.0  # Flat market = maximally choppy

        ci = 100.0 * np.log10(tr_sum / hl_range) / np.log10(period)
        return float(np.clip(ci, 0.0, 100.0))

    @staticmethod
    def compute_bb_width(
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> tuple[float, float]:
        """Compute Bollinger Band Width and its percentile rank.

        Width = (Upper Band - Lower Band) / SMA
        Percentile = rank of current width vs last 100 values.

        Args:
            df: OHLCV DataFrame with ``close`` column.
            period: SMA period (default 20).
            std_dev: Standard deviation multiplier (default 2.0).

        Returns:
            Tuple of ``(bb_width, bb_width_percentile)``.
        """
        close = df["close"]

        if len(close) < period:
            return (0.0, 50.0)

        sma = close.rolling(period).mean()
        rolling_std = close.rolling(period).std()

        upper = sma + std_dev * rolling_std
        lower = sma - std_dev * rolling_std

        # Width normalised by SMA
        width = (upper - lower) / sma
        width = width.dropna()

        if len(width) == 0:
            return (0.0, 50.0)

        current_width = float(width.iloc[-1])

        # Percentile rank vs last 100 values (or however many are available)
        lookback = min(100, len(width))
        recent = width.iloc[-lookback:]
        pctl = float((recent < current_width).sum() / len(recent) * 100.0)

        return (current_width, pctl)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_rsi(df: pd.DataFrame, period: int = 14) -> float:
        """Compute RSI from close prices.

        Args:
            df: OHLCV DataFrame with ``close`` column.
            period: Lookback period (default 14).

        Returns:
            RSI value (0-100).
        """
        close = df["close"]
        if len(close) < period + 1:
            return 50.0

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta.clip(upper=0))

        avg_gain = gain.rolling(period).mean().iloc[-1]
        avg_loss = loss.rolling(period).mean().iloc[-1]

        if pd.isna(avg_gain) or pd.isna(avg_loss):
            return 50.0
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0

        rs = avg_gain / avg_loss
        return float(100.0 - (100.0 / (1.0 + rs)))

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
        """Compute Average True Range.

        Args:
            df: OHLCV DataFrame with ``high``, ``low``, ``close`` columns.
            period: Lookback period (default 14).

        Returns:
            ATR value (float).
        """
        if len(df) < period + 1:
            return 0.0

        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)

        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.rolling(period).mean().iloc[-1]
        return float(atr) if pd.notna(atr) else 0.0

    def _classify_regime(
        self,
        adx: float,
        plus_di: float,
        minus_di: float,
        choppiness: float,
        bb_width: float,
        bb_width_pctl: float,
        rsi: float,
        atr_pct: float,
        price: float,
        sma_50: float,
        atr_14: float,
    ) -> tuple[MarketRegime, float]:
        """Score and classify regime using a prioritised rule set.

        Returns:
            Tuple of ``(MarketRegime, confidence)``.
        """
        # Priority 1: Mean Reverting -- overextended from the mean
        if atr_14 > 0 and abs(price - sma_50) > 2 * atr_14:
            if rsi > 75 or rsi < 25:
                confidence = min(abs(price - sma_50) / (3 * atr_14), 1.0)
                return MarketRegime.MEAN_REVERTING, round(confidence, 3)

        # Priority 2: Volatile Expansion -- BB width percentile > 80
        if bb_width_pctl > 80:
            confidence = min((bb_width_pctl - 60) / 40.0, 1.0)
            return MarketRegime.VOLATILE_EXPANSION, round(confidence, 3)

        # Priority 3: Trending Up -- ADX > 25 and +DI > -DI
        if adx > 25 and plus_di > minus_di:
            confidence = min((adx - 20) / 30.0, 1.0)
            return MarketRegime.TRENDING_UP, round(confidence, 3)

        # Priority 4: Trending Down -- ADX > 25 and -DI > +DI
        if adx > 25 and minus_di > plus_di:
            confidence = min((adx - 20) / 30.0, 1.0)
            return MarketRegime.TRENDING_DOWN, round(confidence, 3)

        # Priority 5: Ranging -- choppy or low ADX
        if choppiness > 61.8 or adx < 20:
            # Score blends choppiness overshoot and ADX weakness
            chop_score = max(choppiness - 50.0, 0.0)
            adx_score = max(20.0 - adx, 0.0)
            raw = max(chop_score, adx_score)
            confidence = min(raw / 30.0, 1.0)
            return MarketRegime.RANGING, round(confidence, 3)

        # Default: Ranging at moderate confidence
        return MarketRegime.RANGING, 0.5

    def _fallback_state(self, asset: str) -> RegimeState:
        """Return a safe RANGING state when detection cannot run."""
        return RegimeState(
            asset=asset,
            regime=MarketRegime.RANGING,
            confidence=0.1,
            adx=0.0,
            plus_di=0.0,
            minus_di=0.0,
            choppiness_index=50.0,
            bb_width=0.0,
            bb_width_percentile=50.0,
            strategy_recommendation=REGIME_STRATEGY_MAP["ranging"],
        )
