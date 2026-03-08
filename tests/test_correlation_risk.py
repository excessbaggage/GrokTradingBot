"""
Comprehensive tests for the correlation-aware risk management module.

Covers:
- Pairwise Pearson correlation calculation
- Correlation risk check (data-driven and group fallback)
- Integration with RiskGuardian as check #14
- Correlation summary generation for context builder
- Edge cases (empty data, single asset, constant prices)
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pytest

from brain.models import RiskValidationResult, TradeDecision
from config.risk_config import RISK_PARAMS
from data.correlation_risk import (
    GROUP_CORRELATIONS,
    CorrelationRiskManager,
)
from execution.risk_guardian import RiskGuardian


# ======================================================================
# Helpers
# ======================================================================


def _make_market_data(
    assets: dict[str, list[float]],
) -> dict[str, dict[str, Any]]:
    """Build a market_data dict from per-asset close price lists.

    Each asset gets a ``candles -> 1h`` entry with dicts like
    ``{"c": <price>}``.
    """
    market_data: dict[str, dict[str, Any]] = {}
    for asset, prices in assets.items():
        candles_1h = [{"c": p} for p in prices]
        market_data[asset] = {
            "candles": {"1h": candles_1h},
            "price": prices[-1] if prices else 0,
        }
    return market_data


def _make_correlated_prices(
    base: list[float],
    correlation: float,
    seed: int = 42,
) -> list[float]:
    """Generate a price series with a target correlation to `base`.

    Uses a linear combination of the base series and random noise
    to achieve approximately the desired Pearson correlation.
    """
    rng = np.random.RandomState(seed)
    n = len(base)
    base_arr = np.array(base, dtype=float)
    noise = rng.normal(0, np.std(base_arr) * 0.5, n)

    # Weight base vs noise to get approximately target correlation
    weight = max(0.0, min(1.0, abs(correlation)))
    combined = weight * base_arr + (1 - weight) * noise

    # Shift to positive values
    combined = combined - combined.min() + 100
    return combined.tolist()


def _create_empty_db() -> sqlite3.Connection:
    """Create an in-memory SQLite DB with the trades table."""
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset TEXT NOT NULL,
            action TEXT NOT NULL,
            side TEXT,
            size_pct REAL,
            leverage REAL,
            entry_price REAL,
            exit_price REAL,
            stop_loss REAL,
            take_profit REAL,
            status TEXT DEFAULT 'open',
            opened_at TEXT NOT NULL,
            closed_at TEXT,
            realized_pnl REAL DEFAULT 0.0
        )
        """
    )
    conn.commit()
    return conn


# ======================================================================
# Test: Correlation Matrix Calculation
# ======================================================================


class TestCorrelationMatrix:
    """Tests for calculate_correlation_matrix()."""

    def test_identical_series_correlation_is_one(self) -> None:
        """Two identical price series should have correlation of 1.0."""
        prices = [100 + i * 0.5 for i in range(50)]
        market_data = _make_market_data({"BTC": prices, "ETH": prices})

        matrix = CorrelationRiskManager.calculate_correlation_matrix(market_data)

        assert "BTC" in matrix
        assert "ETH" in matrix
        assert matrix["BTC"]["ETH"] == pytest.approx(1.0, abs=0.001)
        assert matrix["ETH"]["BTC"] == pytest.approx(1.0, abs=0.001)

    def test_inversely_correlated_series(self) -> None:
        """One rising and one falling series should have negative correlation."""
        up = [100 + i for i in range(50)]
        down = [200 - i for i in range(50)]
        market_data = _make_market_data({"BTC": up, "ETH": down})

        matrix = CorrelationRiskManager.calculate_correlation_matrix(market_data)

        assert matrix["BTC"]["ETH"] < -0.9

    def test_self_correlation_is_one(self) -> None:
        """Self-correlation should always be 1.0."""
        prices = [100 + i * 0.5 for i in range(30)]
        market_data = _make_market_data({"BTC": prices})

        matrix = CorrelationRiskManager.calculate_correlation_matrix(market_data)

        assert matrix["BTC"]["BTC"] == 1.0

    def test_insufficient_data_excluded(self) -> None:
        """Pairs with fewer than MIN_DATA_POINTS should be excluded."""
        short_prices = [100 + i for i in range(5)]
        long_prices = [200 + i for i in range(50)]
        market_data = _make_market_data(
            {"BTC": short_prices, "ETH": long_prices}
        )

        matrix = CorrelationRiskManager.calculate_correlation_matrix(market_data)

        # BTC-ETH pair should not be in the matrix (too few shared points)
        assert "ETH" not in matrix.get("BTC", {})

    def test_empty_market_data(self) -> None:
        """Empty market data should return empty matrix."""
        matrix = CorrelationRiskManager.calculate_correlation_matrix({})
        assert matrix == {}

    def test_multiple_assets_matrix(self) -> None:
        """Matrix should contain pairwise correlations for all assets."""
        np.random.seed(42)
        base = [100 + i * 0.3 for i in range(50)]
        market_data = _make_market_data({
            "BTC": base,
            "ETH": _make_correlated_prices(base, 0.9, seed=1),
            "SOL": _make_correlated_prices(base, 0.5, seed=2),
        })

        matrix = CorrelationRiskManager.calculate_correlation_matrix(market_data)

        # All three assets should be in the matrix
        assert len(matrix) == 3
        for asset in ["BTC", "ETH", "SOL"]:
            assert asset in matrix

    def test_symmetry(self) -> None:
        """Correlation matrix should be symmetric."""
        base = [100 + i * 0.5 for i in range(50)]
        corr = _make_correlated_prices(base, 0.8)
        market_data = _make_market_data({"BTC": base, "ETH": corr})

        matrix = CorrelationRiskManager.calculate_correlation_matrix(market_data)

        assert matrix["BTC"]["ETH"] == matrix["ETH"]["BTC"]

    def test_constant_prices_excluded(self) -> None:
        """Constant price series (zero std) should be excluded from correlations."""
        flat = [100.0] * 50
        rising = [100 + i * 0.5 for i in range(50)]
        market_data = _make_market_data({"BTC": flat, "ETH": rising})

        matrix = CorrelationRiskManager.calculate_correlation_matrix(market_data)

        # Flat series should not produce correlation with rising
        assert "ETH" not in matrix.get("BTC", {})

    def test_list_format_candles(self) -> None:
        """Should handle candle data in list-of-lists format."""
        prices_btc = [100 + i * 0.5 for i in range(30)]
        prices_eth = [200 + i * 0.5 for i in range(30)]

        market_data = {
            "BTC": {
                "candles": {
                    "1h": [[i, 100, 101, 99, p, 1000] for i, p in enumerate(prices_btc)]
                }
            },
            "ETH": {
                "candles": {
                    "1h": [[i, 200, 201, 199, p, 1000] for i, p in enumerate(prices_eth)]
                }
            },
        }

        matrix = CorrelationRiskManager.calculate_correlation_matrix(market_data)

        assert "BTC" in matrix
        assert "ETH" in matrix
        assert matrix["BTC"]["ETH"] == pytest.approx(1.0, abs=0.01)


# ======================================================================
# Test: Correlation Risk Check
# ======================================================================


class TestCorrelationRiskCheck:
    """Tests for check_correlation_risk()."""

    def test_no_open_positions_always_allowed(self) -> None:
        """With no open positions, any new asset should be allowed."""
        allowed, reason = CorrelationRiskManager.check_correlation_risk(
            new_asset="BTC",
            open_positions=[],
            market_data={},
        )
        assert allowed is True

    def test_correlated_positions_rejected_data_driven(self) -> None:
        """Highly correlated positions should be rejected when data is available."""
        prices = [100 + i * 0.5 for i in range(50)]
        market_data = _make_market_data({"BTC": prices, "ETH": prices})

        allowed, reason = CorrelationRiskManager.check_correlation_risk(
            new_asset="ETH",
            open_positions=["BTC"],
            market_data=market_data,
            threshold=0.75,
        )
        assert allowed is False
        assert "correlation" in reason.lower()

    def test_uncorrelated_positions_allowed(self) -> None:
        """Uncorrelated positions should be allowed."""
        np.random.seed(42)
        btc = [100 + i * 0.5 for i in range(50)]
        # Random noise with no correlation to BTC
        doge = list(np.random.normal(0.15, 0.01, 50))
        market_data = _make_market_data({"BTC": btc, "DOGE": doge})

        allowed, reason = CorrelationRiskManager.check_correlation_risk(
            new_asset="DOGE",
            open_positions=["BTC"],
            market_data=market_data,
            threshold=0.75,
        )
        assert allowed is True

    def test_group_fallback_rejects_meme_cluster(self) -> None:
        """Group fallback should reject known correlated meme coins."""
        allowed, reason = CorrelationRiskManager.check_correlation_risk(
            new_asset="SHIB",
            open_positions=["DOGE"],
            market_data={},  # No candle data -- triggers fallback
        )
        assert allowed is False
        assert "group fallback" in reason.lower()

    def test_group_fallback_rejects_l2_cluster(self) -> None:
        """Group fallback should reject known correlated L2 tokens."""
        allowed, reason = CorrelationRiskManager.check_correlation_risk(
            new_asset="OP",
            open_positions=["ARB"],
            market_data={},
        )
        assert allowed is False

    def test_group_fallback_allows_unrelated(self) -> None:
        """Group fallback should allow unrelated assets."""
        allowed, reason = CorrelationRiskManager.check_correlation_risk(
            new_asset="LINK",
            open_positions=["DOGE"],
            market_data={},
        )
        assert allowed is True

    def test_threshold_boundary(self) -> None:
        """Correlation at exactly the threshold should be allowed (> not >=)."""
        # Create two series with correlation of about 0.75
        base = [100 + i * 0.5 for i in range(50)]
        corr75 = _make_correlated_prices(base, 0.75, seed=10)
        market_data = _make_market_data({"BTC": base, "ETH": corr75})

        matrix = CorrelationRiskManager.calculate_correlation_matrix(market_data)
        actual_corr = abs(matrix.get("BTC", {}).get("ETH", 0))

        # The threshold check is strictly greater than
        allowed, reason = CorrelationRiskManager.check_correlation_risk(
            new_asset="ETH",
            open_positions=["BTC"],
            market_data=market_data,
            threshold=actual_corr,  # At exactly the threshold
        )
        # At exactly the threshold, should be allowed (> not >=)
        assert allowed is True

    def test_case_insensitive_assets(self) -> None:
        """Asset names should be case-insensitive."""
        allowed, reason = CorrelationRiskManager.check_correlation_risk(
            new_asset="shib",
            open_positions=["doge"],
            market_data={},
        )
        assert allowed is False

    def test_multiple_open_positions_any_correlated_rejects(self) -> None:
        """If ANY open position is correlated, the trade is rejected."""
        allowed, reason = CorrelationRiskManager.check_correlation_risk(
            new_asset="BONK",
            open_positions=["BTC", "DOGE"],  # DOGE-BONK are correlated
            market_data={},
        )
        assert allowed is False
        assert "DOGE" in reason


# ======================================================================
# Test: GROUP_CORRELATIONS structure
# ======================================================================


class TestGroupCorrelations:
    """Tests for the GROUP_CORRELATIONS lookup table."""

    def test_symmetry(self) -> None:
        """If A lists B, then B must list A."""
        for asset, correlated in GROUP_CORRELATIONS.items():
            for other in correlated:
                assert asset in GROUP_CORRELATIONS.get(other, []), (
                    f"{other} should list {asset} as correlated"
                )

    def test_known_pairs_present(self) -> None:
        """Known high-correlation pairs should be in the table."""
        assert "ETH" in GROUP_CORRELATIONS["BTC"]
        assert "BTC" in GROUP_CORRELATIONS["ETH"]
        assert "SHIB" in GROUP_CORRELATIONS["DOGE"]
        assert "OP" in GROUP_CORRELATIONS["ARB"]
        assert "AVAX" in GROUP_CORRELATIONS["SOL"]


# ======================================================================
# Test: RiskGuardian Integration (Check #14)
# ======================================================================


class TestRiskGuardianCorrelation:
    """Tests for correlation check integration in RiskGuardian."""

    def _make_guardian(self) -> RiskGuardian:
        """Create a RiskGuardian with default params."""
        return RiskGuardian(risk_params=RISK_PARAMS.copy())

    def _make_decision(self, asset: str = "SHIB") -> TradeDecision:
        """Create a valid trade decision for the given asset."""
        return TradeDecision(
            action="open_long",
            asset=asset,
            size_pct=0.05,
            leverage=2.0,
            entry_price=0.00003,
            stop_loss=0.000029,
            take_profit=0.000033,
            order_type="market",
            reasoning="Test correlation check.",
            conviction="medium",
            risk_reward_ratio=3.0,
        )

    def test_correlated_position_rejected_by_guardian(self) -> None:
        """RiskGuardian should reject trades correlated with open positions."""
        guardian = self._make_guardian()
        db = _create_empty_db()
        portfolio = {
            "equity": 10_000.0,
            "peak_equity": 10_000.0,
            "daily_pnl_pct": 0.0,
            "weekly_pnl_pct": 0.0,
            "total_exposure_pct": 0.05,
        }

        decision = self._make_decision("SHIB")
        result = guardian.validate(
            decision,
            portfolio,
            db,
            market_data={},
            open_positions=["DOGE"],
        )

        assert result.approved is False
        assert "correlation" in result.reason.lower()
        db.close()

    def test_uncorrelated_position_allowed_by_guardian(self) -> None:
        """RiskGuardian should allow uncorrelated trades."""
        guardian = self._make_guardian()
        db = _create_empty_db()
        portfolio = {
            "equity": 10_000.0,
            "peak_equity": 10_000.0,
            "daily_pnl_pct": 0.0,
            "weekly_pnl_pct": 0.0,
            "total_exposure_pct": 0.0,
        }

        decision = TradeDecision(
            action="open_long",
            asset="BTC",
            size_pct=0.05,
            leverage=2.0,
            entry_price=65000.0,
            stop_loss=63000.0,
            take_profit=70000.0,
            order_type="limit",
            reasoning="Test uncorrelated.",
            conviction="high",
            risk_reward_ratio=2.5,
        )

        result = guardian.validate(
            decision,
            portfolio,
            db,
            market_data={},
            open_positions=["LINK"],
        )

        assert result.approved is True
        db.close()

    def test_no_market_data_no_positions_passes(self) -> None:
        """With no open positions, correlation check should pass."""
        guardian = self._make_guardian()
        db = _create_empty_db()
        portfolio = {
            "equity": 10_000.0,
            "peak_equity": 10_000.0,
            "daily_pnl_pct": 0.0,
            "weekly_pnl_pct": 0.0,
            "total_exposure_pct": 0.0,
        }

        decision = TradeDecision(
            action="open_long",
            asset="BTC",
            size_pct=0.05,
            leverage=2.0,
            entry_price=65000.0,
            stop_loss=63000.0,
            take_profit=70000.0,
            order_type="limit",
            reasoning="No open positions test.",
            conviction="high",
            risk_reward_ratio=2.5,
        )

        result = guardian.validate(
            decision,
            portfolio,
            db,
            market_data=None,
            open_positions=None,
        )

        assert result.approved is True
        db.close()

    def test_backward_compatible_validate_without_new_args(self) -> None:
        """Validate should still work without the new optional arguments."""
        guardian = self._make_guardian()
        db = _create_empty_db()
        portfolio = {
            "equity": 10_000.0,
            "peak_equity": 10_000.0,
            "daily_pnl_pct": 0.0,
            "weekly_pnl_pct": 0.0,
            "total_exposure_pct": 0.0,
        }

        decision = TradeDecision(
            action="open_long",
            asset="BTC",
            size_pct=0.05,
            leverage=2.0,
            entry_price=65000.0,
            stop_loss=63000.0,
            take_profit=70000.0,
            order_type="limit",
            reasoning="Backward compat test.",
            conviction="high",
            risk_reward_ratio=2.5,
        )

        # Call without market_data and open_positions
        result = guardian.validate(decision, portfolio, db)
        assert result.approved is True
        db.close()


# ======================================================================
# Test: Correlation Summary
# ======================================================================


class TestCorrelationSummary:
    """Tests for get_correlation_summary()."""

    def test_summary_with_correlated_data(self) -> None:
        """Summary should list highly correlated pairs."""
        prices = [100 + i * 0.5 for i in range(50)]
        market_data = _make_market_data({"BTC": prices, "ETH": prices})

        summary = CorrelationRiskManager.get_correlation_summary(
            market_data=market_data,
            open_positions=["BTC"],
            threshold=0.70,
        )

        assert "BTC" in summary
        assert "ETH" in summary
        assert "OPEN POSITION" in summary

    def test_summary_empty_data_uses_groups(self) -> None:
        """With no market data, should use group correlations."""
        summary = CorrelationRiskManager.get_correlation_summary(
            market_data={},
            open_positions=["DOGE"],
        )

        assert "Correlation" in summary

    def test_summary_no_positions(self) -> None:
        """Summary should work with no open positions."""
        summary = CorrelationRiskManager.get_correlation_summary(
            market_data={},
            open_positions=[],
        )

        assert "Correlation" in summary
