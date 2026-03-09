"""
Pydantic v2 models for structured data throughout the brain layer.

These models define the schema for Grok's JSON responses, trade decisions,
market analysis, and risk validation results. They serve as the contract
between the AI's output and the bot's execution logic.

All models use Pydantic v2 BaseModel with strict validation to ensure
that untrusted AI responses are safely parsed and constrained.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class KeyLevels(BaseModel):
    """Support and resistance price levels identified by the trading agent."""

    support: float
    resistance: float


class AssetAnalysis(BaseModel):
    """Per-asset market analysis produced by Grok for a single trading pair."""

    bias: str  # "long", "short", "neutral"
    conviction: str  # "none", "low", "medium", "high"
    key_levels: KeyLevels
    sentiment_read: Optional[str] = None  # Removed from compact prompt to save tokens
    funding_rate_signal: Optional[str] = None  # Removed from compact prompt to save tokens
    entry_conditions_met: Optional[int] = None  # 0-5 N-of-M conditions met
    summary: str


class MarketAnalysis(BaseModel):
    """Aggregated market analysis across the asset universe.

    Accepts dynamic asset keys (e.g., btc, eth, doge, ...) each mapping
    to an AssetAnalysis. The asset set is driven by ASSET_UNIVERSE in config.

    Grok returns: ``{"btc": {...}, "eth": {...}, ...}``
    The model_validator re-packs those into the ``assets`` dict.
    Dot-notation access is preserved via ``__getattr__``.
    """

    assets: dict[str, AssetAnalysis] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def collect_asset_keys(cls, data: Any) -> Any:
        """Collect top-level asset keys into the ``assets`` dict."""
        if not isinstance(data, dict):
            return data

        # Already in the normalised format
        if "assets" in data and isinstance(data["assets"], dict):
            return data

        from config.trading_config import ASSET_UNIVERSE

        known = {a.lower() for a in ASSET_UNIVERSE}
        assets_dict: dict[str, Any] = {}
        remaining: dict[str, Any] = {}

        for key, val in data.items():
            if key.lower() in known:
                assets_dict[key.lower()] = val
            else:
                remaining[key] = val

        remaining["assets"] = assets_dict
        return remaining

    def __getattr__(self, name: str) -> AssetAnalysis:
        """Allow dot-notation: ``market_analysis.btc`` -> ``assets['btc']``."""
        if name != "assets" and name in self.assets:
            return self.assets[name]
        raise AttributeError(f"MarketAnalysis has no asset '{name}'")


class PortfolioAssessment(BaseModel):
    """Grok's assessment of the current portfolio state and suggested adjustments."""

    current_risk_level: str  # "low", "moderate", "elevated", "high"
    recent_performance_note: Optional[str] = None  # Removed from compact prompt
    suggested_exposure_adjustment: str  # "increase", "maintain", "decrease"


class TradeDecision(BaseModel):
    """
    A single trade decision from the AI agent.

    Pydantic validates structure and type bounds. The Risk Guardian in
    execution/risk_guardian.py enforces the actual business limits
    (max position size, max leverage, etc.). Pydantic limits here are
    intentionally generous to avoid rejecting entire API responses when
    Grok slightly exceeds our target ranges.
    """

    action: str  # "open_long", "open_short", "close", "adjust_stop", "hold", "no_trade"
    asset: str = Field(description="Asset symbol from ASSET_UNIVERSE")
    size_pct: float = Field(ge=0.0, le=1.0)  # Risk Guardian enforces actual limit
    leverage: float = Field(ge=1.0, le=3.0)
    entry_price: Optional[float] = None
    stop_loss: float = Field(ge=0)
    take_profit: float = Field(ge=0)
    order_type: str  # "market", "limit"
    reasoning: str
    conviction: str  # "medium", "high"
    risk_reward_ratio: float = Field(ge=0.0)

    @field_validator("asset")
    @classmethod
    def validate_asset(cls, v: str) -> str:
        """Validate asset is in the configured ASSET_UNIVERSE."""
        from config.trading_config import ASSET_UNIVERSE

        v_upper = v.strip().upper()
        if v_upper not in ASSET_UNIVERSE:
            raise ValueError(
                f"Asset '{v}' not in ASSET_UNIVERSE: {ASSET_UNIVERSE}"
            )
        return v_upper


class GrokResponse(BaseModel):
    """
    The complete structured response from Grok on each trading cycle.

    This is the top-level model that the decision parser validates
    the raw AI JSON output against.
    """

    timestamp: Optional[str] = None  # Removed from compact prompt
    market_analysis: MarketAnalysis
    portfolio_assessment: PortfolioAssessment
    decisions: list[TradeDecision]
    overall_stance: str
    next_review_suggestion_minutes: Optional[int] = Field(default=None, ge=5, le=1440)


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
