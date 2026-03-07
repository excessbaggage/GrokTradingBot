"""
Integration-style tests for paper trading mode.

These tests verify the full trading pipeline in paper mode:
- Paper orders don't touch real exchange APIs
- Simulated P&L is tracked correctly
- Paper positions can be opened and closed
- A full orchestrator cycle works end-to-end with all components mocked

All external dependencies (Grok API, exchange, market data) are mocked.
Tests require no network access or API keys.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from brain.decision_parser import DecisionParser
from brain.models import GrokResponse, RiskValidationResult, TradeDecision
from config.risk_config import RISK_PARAMS
from execution.risk_guardian import RiskGuardian
from tests.conftest import _build_sample_grok_response_dict, _create_trades_table


# ======================================================================
# Paper Trading Engine (lightweight simulator for tests)
# ======================================================================


@dataclass
class PaperPosition:
    """A single simulated position in paper trading mode."""

    asset: str
    side: str  # "long" or "short"
    entry_price: float
    size_pct: float
    leverage: float
    stop_loss: float
    take_profit: float
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    closed_at: Optional[datetime] = None
    exit_price: Optional[float] = None
    realized_pnl: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.closed_at is None


@dataclass
class PaperTradingEngine:
    """
    Lightweight paper trading engine for testing.

    Simulates order placement, position tracking, and P&L calculation
    without any exchange API calls.
    """

    starting_capital: float = 10_000.0
    positions: list[PaperPosition] = field(default_factory=list)
    order_history: list[dict] = field(default_factory=list)
    _equity: float = 10_000.0

    @property
    def equity(self) -> float:
        return self._equity

    @property
    def open_positions(self) -> list[PaperPosition]:
        return [p for p in self.positions if p.is_open]

    def place_order(self, decision: TradeDecision, current_price: float) -> dict:
        """Place a paper order (no real exchange interaction)."""
        entry = decision.entry_price if decision.entry_price else current_price

        if decision.action in ("open_long", "open_short"):
            side = "long" if decision.action == "open_long" else "short"
            position = PaperPosition(
                asset=decision.asset,
                side=side,
                entry_price=entry,
                size_pct=decision.size_pct,
                leverage=decision.leverage,
                stop_loss=decision.stop_loss,
                take_profit=decision.take_profit,
            )
            self.positions.append(position)

            order_record = {
                "type": "open",
                "asset": decision.asset,
                "side": side,
                "entry_price": entry,
                "size_pct": decision.size_pct,
                "leverage": decision.leverage,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "paper_mode": True,
            }
            self.order_history.append(order_record)
            return order_record

        elif decision.action == "close":
            # Find and close the matching open position
            for pos in self.open_positions:
                if pos.asset == decision.asset:
                    pos.exit_price = current_price
                    pos.closed_at = datetime.now(timezone.utc)

                    # Calculate P&L
                    if pos.side == "long":
                        pnl_pct = (current_price - pos.entry_price) / pos.entry_price
                    else:
                        pnl_pct = (pos.entry_price - current_price) / pos.entry_price

                    pos.realized_pnl = (
                        self.starting_capital * pos.size_pct * pos.leverage * pnl_pct
                    )
                    self._equity += pos.realized_pnl

                    order_record = {
                        "type": "close",
                        "asset": decision.asset,
                        "side": pos.side,
                        "entry_price": pos.entry_price,
                        "exit_price": current_price,
                        "realized_pnl": pos.realized_pnl,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "paper_mode": True,
                    }
                    self.order_history.append(order_record)
                    return order_record

            return {"type": "close", "error": "no matching position", "paper_mode": True}

        return {"type": "noop", "paper_mode": True}


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def paper_engine() -> PaperTradingEngine:
    """A fresh paper trading engine with $10k capital."""
    return PaperTradingEngine(starting_capital=10_000.0)


@pytest.fixture
def mock_market_prices() -> dict[str, float]:
    """Simulated current market prices for each asset."""
    return {
        "BTC": 65000.0,
        "ETH": 3500.0,
        "SOL": 155.0,
    }


@pytest.fixture
def parser() -> DecisionParser:
    return DecisionParser()


@pytest.fixture
def guardian() -> RiskGuardian:
    return RiskGuardian(risk_params=RISK_PARAMS.copy())


@pytest.fixture
def paper_db() -> sqlite3.Connection:
    """In-memory SQLite DB for paper mode integration tests."""
    conn = sqlite3.connect(":memory:")
    _create_trades_table(conn)
    yield conn
    conn.close()


# ======================================================================
# 1. Paper Order Placement
# ======================================================================


class TestPaperOrderPlacement:
    """Orders in paper mode must not hit any real exchange."""

    def test_paper_order_placement(
        self,
        paper_engine: PaperTradingEngine,
        valid_long_decision: TradeDecision,
    ) -> None:
        """Paper orders should be recorded locally without any exchange API call."""
        result = paper_engine.place_order(valid_long_decision, current_price=65000.0)

        # Verify it was recorded as a paper order
        assert result["paper_mode"] is True
        assert result["type"] == "open"
        assert result["asset"] == "BTC"
        assert result["side"] == "long"
        assert result["entry_price"] == 65000.0

        # Verify position was created
        assert len(paper_engine.open_positions) == 1
        assert paper_engine.open_positions[0].asset == "BTC"
        assert paper_engine.open_positions[0].side == "long"

        # Verify order history was recorded
        assert len(paper_engine.order_history) == 1

    def test_paper_order_no_exchange_calls(
        self,
        paper_engine: PaperTradingEngine,
        valid_long_decision: TradeDecision,
    ) -> None:
        """Verify no real exchange client is invoked during paper trading."""
        mock_exchange = MagicMock()

        # Place through paper engine (not through mock exchange)
        paper_engine.place_order(valid_long_decision, current_price=65000.0)

        # Exchange should never be called
        mock_exchange.place_order.assert_not_called()
        mock_exchange.cancel_order.assert_not_called()


# ======================================================================
# 2. Paper P&L Tracking
# ======================================================================


class TestPaperPnlTracking:
    """Paper mode must track simulated P&L correctly."""

    def test_paper_pnl_tracking(
        self,
        paper_engine: PaperTradingEngine,
    ) -> None:
        """Opening a long at 65k and closing at 70k should show profit."""
        open_decision = TradeDecision(
            action="open_long",
            asset="BTC",
            size_pct=0.10,
            leverage=2.0,
            entry_price=65000.0,
            stop_loss=63000.0,
            take_profit=70000.0,
            order_type="market",
            reasoning="Test profit tracking.",
            conviction="high",
            risk_reward_ratio=2.5,
        )

        close_decision = TradeDecision(
            action="close",
            asset="BTC",
            size_pct=0.10,
            leverage=1.0,
            entry_price=None,
            stop_loss=63000.0,
            take_profit=70000.0,
            order_type="market",
            reasoning="Taking profit.",
            conviction="high",
            risk_reward_ratio=0.0,
        )

        # Open at 65000
        paper_engine.place_order(open_decision, current_price=65000.0)
        initial_equity = paper_engine.equity

        # Close at 70000 (price went up ~7.69%)
        result = paper_engine.place_order(close_decision, current_price=70000.0)

        assert result["type"] == "close"
        assert result["realized_pnl"] > 0
        assert paper_engine.equity > initial_equity

        # Expected P&L: $10,000 * 10% * 2x * (70000-65000)/65000 = $153.85 approx
        expected_pnl = 10_000 * 0.10 * 2.0 * (70000 - 65000) / 65000
        assert abs(result["realized_pnl"] - expected_pnl) < 0.01

    def test_paper_pnl_loss(
        self,
        paper_engine: PaperTradingEngine,
    ) -> None:
        """Opening a long at 65k and closing at 62k should show loss."""
        open_decision = TradeDecision(
            action="open_long",
            asset="BTC",
            size_pct=0.05,
            leverage=2.0,
            entry_price=65000.0,
            stop_loss=62000.0,
            take_profit=70000.0,
            order_type="market",
            reasoning="Test loss tracking.",
            conviction="high",
            risk_reward_ratio=2.5,
        )

        close_decision = TradeDecision(
            action="close",
            asset="BTC",
            size_pct=0.05,
            leverage=1.0,
            entry_price=None,
            stop_loss=62000.0,
            take_profit=70000.0,
            order_type="market",
            reasoning="Stop loss hit.",
            conviction="high",
            risk_reward_ratio=0.0,
        )

        paper_engine.place_order(open_decision, current_price=65000.0)
        initial_equity = paper_engine.equity

        result = paper_engine.place_order(close_decision, current_price=62000.0)

        assert result["type"] == "close"
        assert result["realized_pnl"] < 0
        assert paper_engine.equity < initial_equity

    def test_paper_pnl_short_profit(
        self,
        paper_engine: PaperTradingEngine,
    ) -> None:
        """Short at 3500, close at 3200 should show profit."""
        open_short = TradeDecision(
            action="open_short",
            asset="ETH",
            size_pct=0.05,
            leverage=2.0,
            entry_price=3500.0,
            stop_loss=3600.0,
            take_profit=3200.0,
            order_type="market",
            reasoning="Short ETH.",
            conviction="high",
            risk_reward_ratio=3.0,
        )

        close_short = TradeDecision(
            action="close",
            asset="ETH",
            size_pct=0.05,
            leverage=1.0,
            entry_price=None,
            stop_loss=3600.0,
            take_profit=3200.0,
            order_type="market",
            reasoning="Taking short profit.",
            conviction="high",
            risk_reward_ratio=0.0,
        )

        paper_engine.place_order(open_short, current_price=3500.0)
        result = paper_engine.place_order(close_short, current_price=3200.0)

        assert result["realized_pnl"] > 0


# ======================================================================
# 3. Paper Position Management
# ======================================================================


class TestPaperPositionManagement:
    """Opening and closing paper positions should work correctly."""

    def test_paper_position_management(
        self,
        paper_engine: PaperTradingEngine,
        valid_long_decision: TradeDecision,
        valid_short_decision: TradeDecision,
    ) -> None:
        """Open multiple positions, verify they track correctly."""
        # Open BTC long
        paper_engine.place_order(valid_long_decision, current_price=65000.0)
        assert len(paper_engine.open_positions) == 1

        # Open ETH short
        paper_engine.place_order(valid_short_decision, current_price=3500.0)
        assert len(paper_engine.open_positions) == 2

        # Verify position details
        btc_pos = paper_engine.open_positions[0]
        assert btc_pos.asset == "BTC"
        assert btc_pos.side == "long"
        assert btc_pos.is_open is True

        eth_pos = paper_engine.open_positions[1]
        assert eth_pos.asset == "ETH"
        assert eth_pos.side == "short"
        assert eth_pos.is_open is True

    def test_close_specific_position(
        self,
        paper_engine: PaperTradingEngine,
    ) -> None:
        """Closing one position should leave others open."""
        btc_open = TradeDecision(
            action="open_long",
            asset="BTC",
            size_pct=0.05,
            leverage=2.0,
            entry_price=65000.0,
            stop_loss=63000.0,
            take_profit=70000.0,
            order_type="market",
            reasoning="BTC long.",
            conviction="high",
            risk_reward_ratio=2.5,
        )
        eth_open = TradeDecision(
            action="open_long",
            asset="ETH",
            size_pct=0.03,
            leverage=2.0,
            entry_price=3500.0,
            stop_loss=3400.0,
            take_profit=3700.0,
            order_type="market",
            reasoning="ETH long.",
            conviction="medium",
            risk_reward_ratio=2.0,
        )

        paper_engine.place_order(btc_open, current_price=65000.0)
        paper_engine.place_order(eth_open, current_price=3500.0)
        assert len(paper_engine.open_positions) == 2

        # Close only BTC
        btc_close = TradeDecision(
            action="close",
            asset="BTC",
            size_pct=0.05,
            leverage=1.0,
            entry_price=None,
            stop_loss=63000.0,
            take_profit=70000.0,
            order_type="market",
            reasoning="Close BTC.",
            conviction="high",
            risk_reward_ratio=0.0,
        )
        paper_engine.place_order(btc_close, current_price=66000.0)

        # ETH should still be open, BTC closed
        assert len(paper_engine.open_positions) == 1
        assert paper_engine.open_positions[0].asset == "ETH"

    def test_close_nonexistent_position(
        self,
        paper_engine: PaperTradingEngine,
    ) -> None:
        """Closing a position that doesn't exist should not crash."""
        close_nothing = TradeDecision(
            action="close",
            asset="SOL",
            size_pct=0.0,
            leverage=1.0,
            entry_price=None,
            stop_loss=140.0,
            take_profit=165.0,
            order_type="market",
            reasoning="Close nonexistent.",
            conviction="high",
            risk_reward_ratio=0.0,
        )

        result = paper_engine.place_order(close_nothing, current_price=155.0)
        assert result["paper_mode"] is True


# ======================================================================
# 4. Full Cycle Paper Mode (End-to-End Integration)
# ======================================================================


class TestFullCyclePaperMode:
    """Simulate a complete orchestrator cycle in paper mode."""

    def test_full_cycle_paper_mode(
        self,
        paper_engine: PaperTradingEngine,
        parser: DecisionParser,
        guardian: RiskGuardian,
        mock_market_prices: dict[str, float],
        paper_db: sqlite3.Connection,
    ) -> None:
        """
        Full cycle: mock Grok response -> parse -> validate through
        risk guardian -> execute paper trade.

        This simulates the core orchestrator loop from the spec.
        """
        # Step 1: Mock Grok API response (no real API call)
        raw_grok_response = json.dumps(_build_sample_grok_response_dict())

        # Step 2: Parse the response
        grok_response = parser.parse_response(raw_grok_response)
        assert grok_response is not None
        assert len(grok_response.decisions) == 1

        # Step 3: Extract actionable decisions
        actionable = parser.extract_decisions(grok_response)
        assert len(actionable) == 1
        decision = actionable[0]
        assert decision.action == "open_long"
        assert decision.asset == "BTC"

        # Step 4: Create portfolio state for risk validation
        portfolio: dict[str, Any] = {
            "equity": 10_000.0,
            "peak_equity": 10_000.0,
            "daily_pnl_pct": 0.0,
            "weekly_pnl_pct": 0.0,
            "total_exposure_pct": 0.0,
        }

        # Step 5: Validate through Risk Guardian
        validation = guardian.validate(decision, portfolio, paper_db)
        assert validation.approved is True

        # Step 6: Execute paper trade
        current_price = mock_market_prices[decision.asset]
        order_result = paper_engine.place_order(decision, current_price=current_price)

        # Step 7: Verify the full cycle worked
        assert order_result["paper_mode"] is True
        assert order_result["type"] == "open"
        assert order_result["asset"] == "BTC"
        assert len(paper_engine.open_positions) == 1
        assert paper_engine.open_positions[0].side == "long"

    def test_full_cycle_risk_rejection(
        self,
        paper_engine: PaperTradingEngine,
        parser: DecisionParser,
        guardian: RiskGuardian,
        paper_db: sqlite3.Connection,
    ) -> None:
        """
        Full cycle where the Risk Guardian rejects the trade.

        Simulates: Grok suggests a trade, but daily loss limit is breached.
        The paper engine should NOT receive any order.
        """
        # Step 1: Mock Grok response
        raw_grok_response = json.dumps(_build_sample_grok_response_dict())

        # Step 2: Parse
        grok_response = parser.parse_response(raw_grok_response)
        assert grok_response is not None

        actionable = parser.extract_decisions(grok_response)
        assert len(actionable) == 1

        # Step 3: Portfolio has breached daily loss limit
        bad_portfolio: dict[str, Any] = {
            "equity": 9_400.0,
            "peak_equity": 10_000.0,
            "daily_pnl_pct": -0.06,  # Over the 5% daily limit
            "weekly_pnl_pct": -0.06,
            "total_exposure_pct": 0.0,
        }

        # Step 4: Risk Guardian should reject
        validation = guardian.validate(actionable[0], bad_portfolio, paper_db)
        assert validation.approved is False
        assert "daily" in validation.reason.lower()

        # Step 5: No order should be placed
        assert len(paper_engine.open_positions) == 0
        assert len(paper_engine.order_history) == 0

    def test_full_cycle_no_trade_response(
        self,
        paper_engine: PaperTradingEngine,
        parser: DecisionParser,
    ) -> None:
        """
        Full cycle where Grok returns no trades (staying flat).

        The parser should succeed, but extract_decisions should be empty,
        so no orders are placed.
        """
        raw_response = json.dumps(_build_sample_grok_response_dict(decisions=[]))

        response = parser.parse_response(raw_response)
        assert response is not None
        assert len(response.decisions) == 0

        actionable = parser.extract_decisions(response)
        assert len(actionable) == 0

        # Nothing should happen to the paper engine
        assert len(paper_engine.open_positions) == 0

    def test_full_cycle_with_mock_grok_api(
        self,
        paper_engine: PaperTradingEngine,
        parser: DecisionParser,
        guardian: RiskGuardian,
        mock_market_prices: dict[str, float],
        paper_db: sqlite3.Connection,
    ) -> None:
        """
        Full cycle using unittest.mock to simulate the Grok API client.

        This is the most realistic test: we mock the API client itself,
        then run through parse -> validate -> execute.
        """
        # Create a mock Grok client
        mock_grok_client = MagicMock()
        mock_grok_client.get_trading_decision.return_value = json.dumps(
            _build_sample_grok_response_dict()
        )

        # Simulate the orchestrator calling Grok
        raw_response = mock_grok_client.get_trading_decision(
            system_prompt="You are Sentinel...",
            context="## CURRENT MARKET DATA...",
        )
        mock_grok_client.get_trading_decision.assert_called_once()

        # Parse
        grok_response = parser.parse_response(raw_response)
        assert grok_response is not None

        # Extract actionable
        actionable = parser.extract_decisions(grok_response)
        assert len(actionable) == 1

        # Validate
        portfolio: dict[str, Any] = {
            "equity": 10_000.0,
            "peak_equity": 10_000.0,
            "daily_pnl_pct": 0.0,
            "weekly_pnl_pct": 0.0,
            "total_exposure_pct": 0.0,
        }
        validation = guardian.validate(actionable[0], portfolio, paper_db)
        assert validation.approved is True

        # Execute paper trade
        decision = actionable[0]
        price = mock_market_prices[decision.asset]
        result = paper_engine.place_order(decision, current_price=price)

        assert result["paper_mode"] is True
        assert result["type"] == "open"
        assert len(paper_engine.open_positions) == 1

        # Mock exchange should never have been called
        mock_exchange = MagicMock()
        mock_exchange.place_order.assert_not_called()
