"""
Pydantic v2 models for structured data throughout the brain layer.

These models define the schema for Grok's JSON responses, trade decisions,
market analysis, and risk validation results. They serve as the contract
between the AI's output and the bot's execution logic.

All models use Pydantic v2 BaseModel with strict validation to ensure
that untrusted AI responses are safely parsed and constrained.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class KeyLevels(BaseModel):
    """Support and resistance price levels identified by the trading agent."""

    support: float
    resistance: float


class AssetAnalysis(BaseModel):
    """Per-asset market analysis produced by Grok for a single trading pair."""

    bias: Literal["long", "short", "neutral"]
    conviction: Literal["none", "low", "medium", "high"]
    key_levels: KeyLevels
    sentiment_read: str
    funding_rate_signal: str
    summary: str


class MarketAnalysis(BaseModel):
    """Aggregated market analysis across the entire asset universe."""

    btc: AssetAnalysis
    eth: AssetAnalysis
    sol: AssetAnalysis


class PortfolioAssessment(BaseModel):
    """Grok's assessment of the current portfolio state and suggested adjustments."""

    current_risk_level: Literal["low", "moderate", "elevated", "high"]
    recent_performance_note: str
    suggested_exposure_adjustment: Literal["increase", "maintain", "decrease"]


class TradeDecision(BaseModel):
    """
    A single trade decision from the AI agent.

    Pydantic validates structure and type bounds. The Risk Guardian in
    execution/risk_guardian.py enforces the actual business limits
    (max position size, max leverage, etc.). Pydantic limits here are
    intentionally generous to avoid rejecting entire API responses when
    Grok slightly exceeds our target ranges.
    """

    action: Literal["open_long", "open_short", "close", "adjust_stop", "hold", "no_trade"]
    asset: Literal["BTC", "ETH", "SOL"]
    size_pct: float = Field(ge=0.0, le=1.0)  # Risk Guardian enforces actual limit
    leverage: float = Field(ge=1.0, le=3.0)
    entry_price: Optional[float] = None
    stop_loss: float
    take_profit: float
    order_type: Literal["market", "limit"]
    reasoning: str
    conviction: Literal["medium", "high"]
    risk_reward_ratio: float = Field(ge=0.0)


class GrokResponse(BaseModel):
    """
    The complete structured response from Grok on each trading cycle.

    This is the top-level model that the decision parser validates
    the raw AI JSON output against.
    """

    timestamp: str
    market_analysis: MarketAnalysis
    portfolio_assessment: PortfolioAssessment
    decisions: list[TradeDecision]
    overall_stance: str
    next_review_suggestion_minutes: int = Field(ge=5, le=1440)  # Up to 24h; main.py clamps to MIN_CYCLE_INTERVAL


class RiskValidationResult(BaseModel):
    """
    Result of the Risk Guardian's validation of a single TradeDecision.

    If the decision is rejected, `reason` explains why.
    If the decision is modified (e.g., size reduced), `modified_decision`
    contains the adjusted version that passed validation.
    """

    approved: bool
    reason: str = ""
    modified_decision: Optional[TradeDecision] = None
