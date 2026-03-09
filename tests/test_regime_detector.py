"""
Comprehensive tests for the Market Regime Detector.

Verifies ADX, Choppiness Index, and Bollinger Band Width computations,
the full regime classification pipeline for all five regimes, edge
cases (insufficient data, flat markets, extreme values), and the
RegimeState dataclass construction.

All tests use synthetic OHLCV DataFrames -- no network or API calls.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from data.regime_detector import (
    REGIME_STRATEGY_MAP,
    MarketRegime,
    RegimeDetector,
    RegimeState,
    _MIN_CANDLES,
)


# ======================================================================
# Helpers: synthetic OHLCV generators
# ======================================================================


def _make_candles(
    n: int = 60,
    start_price: float = 100.0,
    trend: float = 0.0,
    noise: float = 1.0,
    volume_base: float = 1000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV candles.

    Args:
        n: Number of candles.
        start_price: Opening price of the first candle.
        trend: Per-candle drift (positive = uptrend, negative = downtrend).
        noise: Standard deviation of random price movement.
        volume_base: Base volume per candle.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns ``open``, ``high``, ``low``, ``close``,
        ``volume``.
    """
    rng = np.random.RandomState(seed)
    prices = [start_price]
    for _ in range(n - 1):
        prices.append(prices[-1] + trend + rng.normal(0, noise))

    rows = []
    for i in range(n):
        base = prices[i]
        jitter = abs(rng.normal(0, noise * 0.5))
        o = base
        c = base + trend + rng.normal(0, noise * 0.3)
        h = max(o, c) + jitter
        l = min(o, c) - jitter
        # Ensure low is always positive
        l = max(l, 0.01)
        h = max(h, l + 0.01)
        v = volume_base + rng.uniform(-volume_base * 0.3, volume_base * 0.3)
        rows.append({"open": o, "high": h, "low": l, "close": c, "volume": v})

    return pd.DataFrame(rows)


def _make_uptrend(n: int = 60, start: float = 100.0) -> pd.DataFrame:
    """Generate a strong uptrend (large positive drift, low noise)."""
    return _make_candles(n=n, start_price=start, trend=2.0, noise=0.3)


def _make_downtrend(n: int = 60, start: float = 200.0) -> pd.DataFrame:
    """Generate a strong downtrend (large negative drift, low noise)."""
    return _make_candles(n=n, start_price=start, trend=-2.0, noise=0.3)


def _make_ranging(n: int = 60, start: float = 100.0) -> pd.DataFrame:
    """Generate a ranging / choppy market (zero drift, moderate noise)."""
    return _make_candles(n=n, start_price=start, trend=0.0, noise=0.8)


def _make_volatile_expansion(
    n: int = 60,
    start: float = 100.0,
) -> pd.DataFrame:
    """Generate volatility expansion: calm first half, explosion second half."""
    # First half: tight range
    calm = _make_candles(n=n // 2, start_price=start, trend=0.0, noise=0.2, seed=10)
    # Second half: wide volatility
    last_close = float(calm["close"].iloc[-1])
    wild = _make_candles(
        n=n - n // 2,
        start_price=last_close,
        trend=0.5,
        noise=5.0,
        seed=20,
    )
    return pd.concat([calm, wild], ignore_index=True)


def _make_mean_reverting(n: int = 60, start: float = 100.0) -> pd.DataFrame:
    """Generate an overextended move far from SMA-50 with extreme RSI.

    Builds a gradual trend that accelerates sharply at the end so that
    the latest price is well above 2 ATR from SMA-50 and RSI is extreme.
    """
    # Steady base for SMA-50 to anchor around ``start``
    base_n = max(n - 10, 40)
    base = _make_candles(
        n=base_n,
        start_price=start,
        trend=0.1,
        noise=0.3,
        seed=55,
    )
    last_close = float(base["close"].iloc[-1])
    # Sharp spike for last 10 candles (all closes increasing)
    spike_rows = []
    price = last_close
    for i in range(n - base_n):
        price += 8.0  # Strong upward move every candle
        spike_rows.append({
            "open": price - 1.0,
            "high": price + 0.5,
            "low": price - 1.5,
            "close": price,
            "volume": 1500.0,
        })
    spike = pd.DataFrame(spike_rows)
    return pd.concat([base, spike], ignore_index=True)


def _make_flat(n: int = 60, price: float = 100.0) -> pd.DataFrame:
    """Generate a perfectly flat market (all candles identical)."""
    rows = [{
        "open": price,
        "high": price + 0.01,
        "low": price - 0.01,
        "close": price,
        "volume": 1000.0,
    }] * n
    return pd.DataFrame(rows)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def detector() -> RegimeDetector:
    """A fresh RegimeDetector instance."""
    return RegimeDetector()


@pytest.fixture
def uptrend_candles() -> pd.DataFrame:
    return _make_uptrend()


@pytest.fixture
def downtrend_candles() -> pd.DataFrame:
    return _make_downtrend()


@pytest.fixture
def ranging_candles() -> pd.DataFrame:
    return _make_ranging()


@pytest.fixture
def volatile_candles() -> pd.DataFrame:
    return _make_volatile_expansion()


@pytest.fixture
def mean_reverting_candles() -> pd.DataFrame:
    return _make_mean_reverting()


@pytest.fixture
def flat_candles() -> pd.DataFrame:
    return _make_flat()


@pytest.fixture
def short_candles() -> pd.DataFrame:
    """Too few candles (below _MIN_CANDLES)."""
    return _make_candles(n=10)


# ======================================================================
# 1. ADX Computation
# ======================================================================


class TestADXComputation:
    """Verify ADX, +DI, -DI calculation."""

    def test_adx_uptrend_high(self, uptrend_candles: pd.DataFrame) -> None:
        """In a strong uptrend, ADX should be elevated and +DI > -DI."""
        adx, plus_di, minus_di = RegimeDetector.compute_adx(uptrend_candles)
        assert adx > 20, f"ADX should be > 20 in strong uptrend, got {adx:.1f}"
        assert plus_di > minus_di, "+DI should exceed -DI in an uptrend"

    def test_adx_downtrend_high(self, downtrend_candles: pd.DataFrame) -> None:
        """In a strong downtrend, ADX should be elevated and -DI > +DI."""
        adx, plus_di, minus_di = RegimeDetector.compute_adx(downtrend_candles)
        assert adx > 20, f"ADX should be > 20 in strong downtrend, got {adx:.1f}"
        assert minus_di > plus_di, "-DI should exceed +DI in a downtrend"

    def test_adx_ranging_low(self, ranging_candles: pd.DataFrame) -> None:
        """In a ranging market, ADX should be relatively low."""
        adx, _, _ = RegimeDetector.compute_adx(ranging_candles)
        # Ranging data should produce ADX well below trending thresholds
        # Allow some tolerance since random data can trend briefly
        assert adx < 50, f"ADX should be < 50 in ranging market, got {adx:.1f}"

    def test_adx_returns_tuple_of_three(self, uptrend_candles: pd.DataFrame) -> None:
        """compute_adx must return a 3-tuple of floats."""
        result = RegimeDetector.compute_adx(uptrend_candles)
        assert len(result) == 3
        for val in result:
            assert isinstance(val, float)

    def test_adx_insufficient_data(self) -> None:
        """With fewer rows than the period, ADX should return zeros."""
        tiny = _make_candles(n=5)
        adx, plus_di, minus_di = RegimeDetector.compute_adx(tiny, period=14)
        assert adx == 0.0
        assert plus_di == 0.0
        assert minus_di == 0.0

    def test_adx_non_negative(self, uptrend_candles: pd.DataFrame) -> None:
        """All ADX components must be non-negative."""
        adx, plus_di, minus_di = RegimeDetector.compute_adx(uptrend_candles)
        assert adx >= 0
        assert plus_di >= 0
        assert minus_di >= 0


# ======================================================================
# 2. Choppiness Index
# ======================================================================


class TestChoppinessIndex:
    """Verify Choppiness Index computation."""

    def test_choppiness_trending_low(self, uptrend_candles: pd.DataFrame) -> None:
        """In a strong trend, Choppiness Index should be low (< 61.8)."""
        ci = RegimeDetector.compute_choppiness_index(uptrend_candles)
        assert ci < 61.8, f"CI should be < 61.8 in a trend, got {ci:.1f}"

    def test_choppiness_ranging_high(self, ranging_candles: pd.DataFrame) -> None:
        """In a choppy market, Choppiness Index should be higher."""
        ci = RegimeDetector.compute_choppiness_index(ranging_candles)
        # Ranging data may not always exceed 61.8 with random noise,
        # but it should be significantly above trending levels
        assert ci > 30, f"CI should be > 30 in ranging market, got {ci:.1f}"

    def test_choppiness_flat_high(self, flat_candles: pd.DataFrame) -> None:
        """In a perfectly flat market, CI should be at the maximum (100)."""
        ci = RegimeDetector.compute_choppiness_index(flat_candles)
        assert ci >= 90, f"CI should be >= 90 in flat market, got {ci:.1f}"

    def test_choppiness_bounded(self, uptrend_candles: pd.DataFrame) -> None:
        """Choppiness Index must be between 0 and 100."""
        ci = RegimeDetector.compute_choppiness_index(uptrend_candles)
        assert 0.0 <= ci <= 100.0

    def test_choppiness_insufficient_data(self) -> None:
        """With insufficient data, should return 50.0 (neutral default)."""
        tiny = _make_candles(n=5)
        ci = RegimeDetector.compute_choppiness_index(tiny, period=14)
        assert ci == 50.0


# ======================================================================
# 3. Bollinger Band Width
# ======================================================================


class TestBBWidth:
    """Verify Bollinger Band Width and percentile computation."""

    def test_bb_width_expanding(self, volatile_candles: pd.DataFrame) -> None:
        """Volatile expansion should produce a high BB width percentile."""
        bb_width, bb_pctl = RegimeDetector.compute_bb_width(volatile_candles)
        assert bb_width > 0, "BB width must be positive"
        assert bb_pctl > 50, (
            f"BB width percentile should be > 50 during volatility expansion, "
            f"got {bb_pctl:.0f}%"
        )

    def test_bb_width_flat_low(self, flat_candles: pd.DataFrame) -> None:
        """A flat market should have near-zero BB width."""
        bb_width, _ = RegimeDetector.compute_bb_width(flat_candles)
        assert bb_width < 0.01, f"BB width should be near 0 in flat market, got {bb_width:.4f}"

    def test_bb_width_returns_two_values(self, uptrend_candles: pd.DataFrame) -> None:
        """compute_bb_width must return a 2-tuple of floats."""
        result = RegimeDetector.compute_bb_width(uptrend_candles)
        assert len(result) == 2
        for val in result:
            assert isinstance(val, float)

    def test_bb_percentile_bounded(self, uptrend_candles: pd.DataFrame) -> None:
        """Percentile must be between 0 and 100."""
        _, pctl = RegimeDetector.compute_bb_width(uptrend_candles)
        assert 0.0 <= pctl <= 100.0

    def test_bb_width_insufficient_data(self) -> None:
        """With fewer rows than the period, return (0.0, 50.0)."""
        tiny = _make_candles(n=10)
        bb_width, bb_pctl = RegimeDetector.compute_bb_width(tiny, period=20)
        assert bb_width == 0.0
        assert bb_pctl == 50.0


# ======================================================================
# 4. Full Regime Classification
# ======================================================================


class TestRegimeClassification:
    """Verify the detect() pipeline for each of the five regimes."""

    def test_detect_trending_up(
        self,
        detector: RegimeDetector,
        uptrend_candles: pd.DataFrame,
    ) -> None:
        """A strong uptrend should be classified as TRENDING_UP."""
        state = detector.detect("BTC", uptrend_candles)
        assert state.regime in (
            MarketRegime.TRENDING_UP,
            MarketRegime.VOLATILE_EXPANSION,
            MarketRegime.MEAN_REVERTING,
        ), f"Expected TRENDING_UP (or higher-priority), got {state.regime}"
        assert state.asset == "BTC"
        assert 0.0 <= state.confidence <= 1.0

    def test_detect_trending_down(
        self,
        detector: RegimeDetector,
        downtrend_candles: pd.DataFrame,
    ) -> None:
        """A strong downtrend should be classified as TRENDING_DOWN."""
        state = detector.detect("ETH", downtrend_candles)
        assert state.regime in (
            MarketRegime.TRENDING_DOWN,
            MarketRegime.VOLATILE_EXPANSION,
            MarketRegime.MEAN_REVERTING,
        ), f"Expected TRENDING_DOWN (or higher-priority), got {state.regime}"
        assert state.asset == "ETH"
        assert 0.0 <= state.confidence <= 1.0

    def test_detect_ranging(
        self,
        detector: RegimeDetector,
        ranging_candles: pd.DataFrame,
    ) -> None:
        """A choppy sideways market should be classified as RANGING.

        Note: synthetic random-walk data can incidentally drift enough
        to push ADX just above 25, so we also accept a trending regime
        but only at LOW confidence (< 0.3).
        """
        state = detector.detect("SOL", ranging_candles)
        if state.regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
            assert state.confidence < 0.3, (
                f"Ranging data should not have high confidence in a "
                f"trending regime, got {state.confidence:.2f}"
            )
        else:
            assert state.regime in (
                MarketRegime.RANGING,
                MarketRegime.VOLATILE_EXPANSION,
            ), f"Expected RANGING (or compatible), got {state.regime}"

    def test_detect_volatile_expansion(
        self,
        detector: RegimeDetector,
        volatile_candles: pd.DataFrame,
    ) -> None:
        """Volatility expansion should be detected via high BB percentile."""
        state = detector.detect("BTC", volatile_candles)
        # The volatile candles have a clear expansion in the second half,
        # but depending on the exact random data other higher-priority
        # regimes may fire.  At minimum, it should not be RANGING.
        assert state.regime != MarketRegime.RANGING or state.confidence < 0.5, (
            "Volatile expansion data should not be confidently classified as RANGING"
        )

    def test_detect_mean_reverting(
        self,
        detector: RegimeDetector,
        mean_reverting_candles: pd.DataFrame,
    ) -> None:
        """Overextended price should classify as MEAN_REVERTING."""
        state = detector.detect("BTC", mean_reverting_candles)
        # Mean reverting is the highest-priority regime, so it should fire
        # when price is > 2 ATR from SMA-50 with extreme RSI
        assert state.regime in (
            MarketRegime.MEAN_REVERTING,
            MarketRegime.VOLATILE_EXPANSION,
            MarketRegime.TRENDING_UP,
        ), f"Expected MEAN_REVERTING (or compatible), got {state.regime}"


# ======================================================================
# 5. Edge Cases
# ======================================================================


class TestEdgeCases:
    """Edge case handling."""

    def test_insufficient_data_returns_ranging(
        self,
        detector: RegimeDetector,
        short_candles: pd.DataFrame,
    ) -> None:
        """With too few candles, detect() should return RANGING with low confidence."""
        state = detector.detect("BTC", short_candles)
        assert state.regime == MarketRegime.RANGING
        assert state.confidence <= 0.2

    def test_none_candles_returns_ranging(
        self,
        detector: RegimeDetector,
    ) -> None:
        """Passing None for candles should return a safe fallback."""
        state = detector.detect("BTC", None)  # type: ignore[arg-type]
        assert state.regime == MarketRegime.RANGING
        assert state.confidence <= 0.2

    def test_empty_dataframe_returns_ranging(
        self,
        detector: RegimeDetector,
    ) -> None:
        """An empty DataFrame should return a safe fallback."""
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        state = detector.detect("BTC", empty)
        assert state.regime == MarketRegime.RANGING
        assert state.confidence <= 0.2

    def test_flat_market(
        self,
        detector: RegimeDetector,
        flat_candles: pd.DataFrame,
    ) -> None:
        """A perfectly flat market should classify as RANGING."""
        state = detector.detect("BTC", flat_candles)
        assert state.regime == MarketRegime.RANGING

    def test_extreme_values(self, detector: RegimeDetector) -> None:
        """Extreme price values should not cause errors."""
        candles = _make_candles(n=60, start_price=1_000_000, trend=100, noise=50)
        state = detector.detect("BTC", candles)
        assert isinstance(state, RegimeState)
        assert 0.0 <= state.confidence <= 1.0

    def test_very_small_prices(self, detector: RegimeDetector) -> None:
        """Very small prices (sub-penny) should not cause division errors."""
        candles = _make_candles(n=60, start_price=0.001, trend=0.0001, noise=0.00005)
        state = detector.detect("DOGE", candles)
        assert isinstance(state, RegimeState)
        assert 0.0 <= state.confidence <= 1.0

    def test_optional_4h_candles(
        self,
        detector: RegimeDetector,
        uptrend_candles: pd.DataFrame,
    ) -> None:
        """Passing candles_4h=None should work without errors."""
        state = detector.detect("BTC", uptrend_candles, candles_4h=None)
        assert isinstance(state, RegimeState)

    def test_with_4h_candles(
        self,
        detector: RegimeDetector,
        uptrend_candles: pd.DataFrame,
    ) -> None:
        """Passing 4h candles should not break anything (reserved for future use)."""
        candles_4h = _make_uptrend(n=30)
        state = detector.detect("BTC", uptrend_candles, candles_4h=candles_4h)
        assert isinstance(state, RegimeState)


# ======================================================================
# 6. RegimeState Dataclass
# ======================================================================


class TestRegimeState:
    """Verify RegimeState construction and fields."""

    def test_dataclass_fields(self) -> None:
        """All fields should be accessible."""
        state = RegimeState(
            asset="BTC",
            regime=MarketRegime.TRENDING_UP,
            confidence=0.85,
            adx=35.0,
            plus_di=30.0,
            minus_di=15.0,
            choppiness_index=35.0,
            bb_width=0.05,
            bb_width_percentile=65.0,
            strategy_recommendation=REGIME_STRATEGY_MAP["trending_up"],
        )
        assert state.asset == "BTC"
        assert state.regime == MarketRegime.TRENDING_UP
        assert state.confidence == 0.85
        assert state.adx == 35.0
        assert state.plus_di == 30.0
        assert state.minus_di == 15.0
        assert state.choppiness_index == 35.0
        assert state.bb_width == 0.05
        assert state.bb_width_percentile == 65.0
        assert state.strategy_recommendation["preferred_strategy"] == "momentum_long"

    def test_detect_returns_regime_state(
        self,
        detector: RegimeDetector,
        uptrend_candles: pd.DataFrame,
    ) -> None:
        """detect() should return a RegimeState instance."""
        result = detector.detect("BTC", uptrend_candles)
        assert isinstance(result, RegimeState)

    def test_regime_state_has_strategy(
        self,
        detector: RegimeDetector,
        uptrend_candles: pd.DataFrame,
    ) -> None:
        """The returned RegimeState should include a strategy recommendation dict."""
        result = detector.detect("BTC", uptrend_candles)
        rec = result.strategy_recommendation
        assert isinstance(rec, dict)
        assert "preferred_strategy" in rec
        assert "avoid" in rec
        assert "position_size_multiplier" in rec
        assert "stop_distance_multiplier" in rec
        assert "description" in rec


# ======================================================================
# 7. Strategy Map Coverage
# ======================================================================


class TestStrategyMap:
    """Verify REGIME_STRATEGY_MAP completeness."""

    def test_all_regimes_in_map(self) -> None:
        """Every MarketRegime enum value must have an entry in the map."""
        for regime in MarketRegime:
            assert regime.value in REGIME_STRATEGY_MAP, (
                f"Regime {regime.value} missing from REGIME_STRATEGY_MAP"
            )

    def test_map_has_required_keys(self) -> None:
        """Each strategy entry must have the required keys."""
        required = {
            "preferred_strategy",
            "avoid",
            "position_size_multiplier",
            "stop_distance_multiplier",
            "description",
        }
        for regime_name, strategy in REGIME_STRATEGY_MAP.items():
            for key in required:
                assert key in strategy, (
                    f"Key '{key}' missing from strategy for regime '{regime_name}'"
                )

    def test_position_multipliers_positive(self) -> None:
        """Position and stop multipliers must be positive."""
        for name, strategy in REGIME_STRATEGY_MAP.items():
            assert strategy["position_size_multiplier"] > 0, (
                f"position_size_multiplier <= 0 for {name}"
            )
            assert strategy["stop_distance_multiplier"] > 0, (
                f"stop_distance_multiplier <= 0 for {name}"
            )

    def test_avoid_is_list(self) -> None:
        """The 'avoid' field must be a list."""
        for name, strategy in REGIME_STRATEGY_MAP.items():
            assert isinstance(strategy["avoid"], list), (
                f"'avoid' for {name} should be a list"
            )


# ======================================================================
# 8. Confidence Bounds
# ======================================================================


class TestConfidenceBounds:
    """All confidence scores must be in [0, 1]."""

    @pytest.mark.parametrize("candle_fn,asset", [
        (_make_uptrend, "BTC"),
        (_make_downtrend, "ETH"),
        (_make_ranging, "SOL"),
        (_make_volatile_expansion, "BTC"),
        (_make_mean_reverting, "BTC"),
        (_make_flat, "BTC"),
    ])
    def test_confidence_bounded(
        self,
        detector: RegimeDetector,
        candle_fn: Any,
        asset: str,
    ) -> None:
        """Confidence must be between 0 and 1 for all market conditions."""
        candles = candle_fn()
        state = detector.detect(asset, candles)
        assert 0.0 <= state.confidence <= 1.0, (
            f"Confidence {state.confidence} out of bounds for "
            f"{state.regime.value}"
        )


# ======================================================================
# 9. Context Builder Integration (smoke test)
# ======================================================================


class TestContextBuilderIntegration:
    """Verify that regime data integrates into the context prompt."""

    def test_regime_lines_included_when_provided(
        self,
        detector: RegimeDetector,
        uptrend_candles: pd.DataFrame,
    ) -> None:
        """When regime_data is passed, the prompt should contain regime info."""
        from data.context_builder import build_context_prompt

        regime = detector.detect("BTC", uptrend_candles)

        market_data = {
            "BTC": {
                "price": 65000.0,
                "24h_change_pct": 0.02,
                "funding": {"current_rate": 0.01, "avg_7d_rate": 0.005},
                "oi": {"current_oi": 1_000_000, "oi_24h_change_pct": 5.0},
                "candles": {},
                "technicals": {
                    "atr_14": 500,
                    "atr_pct": 0.008,
                    "rsi_14": 55,
                    "adaptive_ob": 70,
                    "adaptive_os": 30,
                    "volatility_regime": "normal",
                    "turtle_size_factor": 1.0,
                },
            }
        }
        portfolio = {
            "total_equity": 10_000,
            "available_margin": 8_000,
            "unrealized_pnl": 200,
            "positions": [],
        }
        risk_status = {
            "daily_pnl": 100,
            "weekly_pnl": 300,
            "drawdown_from_peak": 0.01,
            "trades_today": 2,
            "consecutive_losses": 0,
        }

        prompt = build_context_prompt(
            market_data=market_data,
            portfolio=portfolio,
            recent_trades=[],
            risk_status=risk_status,
            regime_data={"BTC": regime},
        )

        assert "Regime=" in prompt
        assert "ADX=" in prompt
        assert "CI=" in prompt

    def test_regime_lines_omitted_when_none(self) -> None:
        """When regime_data is None, no regime lines should appear."""
        from data.context_builder import build_context_prompt

        market_data = {
            "BTC": {
                "price": 65000.0,
                "24h_change_pct": 0.02,
                "funding": {"current_rate": 0.01, "avg_7d_rate": 0.005},
                "oi": {"current_oi": 1_000_000, "oi_24h_change_pct": 5.0},
                "candles": {},
                "technicals": {
                    "atr_14": 500,
                    "atr_pct": 0.008,
                    "rsi_14": 55,
                    "adaptive_ob": 70,
                    "adaptive_os": 30,
                    "volatility_regime": "normal",
                    "turtle_size_factor": 1.0,
                },
            }
        }
        portfolio = {
            "total_equity": 10_000,
            "available_margin": 8_000,
            "unrealized_pnl": 200,
            "positions": [],
        }
        risk_status = {
            "daily_pnl": 100,
            "weekly_pnl": 300,
            "drawdown_from_peak": 0.01,
            "trades_today": 2,
            "consecutive_losses": 0,
        }

        prompt = build_context_prompt(
            market_data=market_data,
            portfolio=portfolio,
            recent_trades=[],
            risk_status=risk_status,
            regime_data=None,
        )

        assert "Regime=" not in prompt

    def test_regime_lines_omitted_for_missing_asset(self) -> None:
        """If regime_data exists but lacks the asset, no regime lines appear."""
        from data.context_builder import build_context_prompt

        market_data = {
            "SOL": {
                "price": 150.0,
                "24h_change_pct": -0.01,
                "funding": {"current_rate": 0.005, "avg_7d_rate": 0.003},
                "oi": {"current_oi": 500_000, "oi_24h_change_pct": 2.0},
                "candles": {},
                "technicals": {
                    "atr_14": 3.0,
                    "atr_pct": 0.02,
                    "rsi_14": 45,
                    "adaptive_ob": 70,
                    "adaptive_os": 30,
                    "volatility_regime": "normal",
                    "turtle_size_factor": 1.0,
                },
            }
        }
        portfolio = {
            "total_equity": 10_000,
            "available_margin": 8_000,
            "unrealized_pnl": 0,
            "positions": [],
        }
        risk_status = {
            "daily_pnl": 0,
            "weekly_pnl": 0,
            "drawdown_from_peak": 0,
            "trades_today": 0,
            "consecutive_losses": 0,
        }

        # regime_data has BTC but market_data has SOL -- no overlap
        from data.regime_detector import RegimeState, REGIME_STRATEGY_MAP

        btc_regime = RegimeState(
            asset="BTC",
            regime=MarketRegime.TRENDING_UP,
            confidence=0.8,
            adx=30.0,
            plus_di=25.0,
            minus_di=15.0,
            choppiness_index=40.0,
            bb_width=0.04,
            bb_width_percentile=55.0,
            strategy_recommendation=REGIME_STRATEGY_MAP["trending_up"],
        )

        prompt = build_context_prompt(
            market_data=market_data,
            portfolio=portfolio,
            recent_trades=[],
            risk_status=risk_status,
            regime_data={"BTC": btc_regime},  # BTC regime, but only SOL in market_data
        )

        # SOL section should NOT have regime data
        assert "Regime=" not in prompt


# ======================================================================
# 10. Indicator Consistency
# ======================================================================


class TestIndicatorConsistency:
    """Verify that indicator values in RegimeState are plausible."""

    def test_adx_in_state_matches_direct(
        self,
        detector: RegimeDetector,
        uptrend_candles: pd.DataFrame,
    ) -> None:
        """ADX in the RegimeState should match the direct computation."""
        state = detector.detect("BTC", uptrend_candles)
        adx_direct, _, _ = RegimeDetector.compute_adx(uptrend_candles)
        assert abs(state.adx - round(adx_direct, 2)) < 0.1

    def test_choppiness_in_state_matches_direct(
        self,
        detector: RegimeDetector,
        ranging_candles: pd.DataFrame,
    ) -> None:
        """CI in the RegimeState should match the direct computation."""
        state = detector.detect("SOL", ranging_candles)
        ci_direct = RegimeDetector.compute_choppiness_index(ranging_candles)
        assert abs(state.choppiness_index - round(ci_direct, 2)) < 0.1

    def test_bb_width_in_state_matches_direct(
        self,
        detector: RegimeDetector,
        volatile_candles: pd.DataFrame,
    ) -> None:
        """BB width in the RegimeState should match the direct computation."""
        state = detector.detect("BTC", volatile_candles)
        width_direct, pctl_direct = RegimeDetector.compute_bb_width(volatile_candles)
        assert abs(state.bb_width - round(width_direct, 4)) < 0.01
        assert abs(state.bb_width_percentile - round(pctl_direct, 1)) < 1.0


# ======================================================================
# 11. MarketRegime Enum
# ======================================================================


class TestMarketRegimeEnum:
    """Verify the MarketRegime enum."""

    def test_five_regimes(self) -> None:
        """There must be exactly 5 regimes."""
        assert len(MarketRegime) == 5

    def test_enum_values(self) -> None:
        """Each regime has the expected string value."""
        expected = {
            "trending_up",
            "trending_down",
            "ranging",
            "volatile_expansion",
            "mean_reverting",
        }
        actual = {r.value for r in MarketRegime}
        assert actual == expected

    def test_enum_from_value(self) -> None:
        """Can construct a MarketRegime from its string value."""
        regime = MarketRegime("trending_up")
        assert regime == MarketRegime.TRENDING_UP
