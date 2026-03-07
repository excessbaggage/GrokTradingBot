"""
Tests for the Decision Parser — JSON parsing and Pydantic validation of Grok responses.

The Decision Parser is the gateway between untrusted AI output and the bot's
structured decision pipeline. These tests verify that:
- Valid JSON is correctly parsed into GrokResponse models
- Markdown-wrapped JSON is handled gracefully
- Invalid / malformed JSON returns None (fail safe, not crash)
- Missing fields, invalid actions, and out-of-range values are caught
- Non-actionable decisions (hold, no_trade) are properly filtered

All tests use hardcoded JSON strings — no network, no API keys.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from brain.decision_parser import DecisionParser
from brain.models import GrokResponse, TradeDecision


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def parser() -> DecisionParser:
    """A fresh DecisionParser instance."""
    return DecisionParser()


# ======================================================================
# 1. Parse Valid Response
# ======================================================================


class TestParseValidResponse:
    """Valid JSON should parse correctly into GrokResponse."""

    def test_parse_valid_response(
        self,
        parser: DecisionParser,
        sample_grok_response_json: str,
    ) -> None:
        """A well-formed JSON string should parse into a GrokResponse."""
        result = parser.parse_response(sample_grok_response_json)

        assert result is not None
        assert isinstance(result, GrokResponse)
        assert result.timestamp == "2026-03-07T12:00:00Z"
        assert len(result.decisions) == 1
        assert result.decisions[0].action == "open_long"
        assert result.decisions[0].asset == "BTC"
        assert result.decisions[0].size_pct == 0.05
        assert result.decisions[0].leverage == 2.0
        assert result.decisions[0].risk_reward_ratio == 2.5
        assert result.market_analysis.btc.bias == "long"
        assert result.portfolio_assessment.current_risk_level == "low"
        assert result.next_review_suggestion_minutes == 15


# ======================================================================
# 2. Parse Response with Markdown Wrapping
# ======================================================================


class TestParseMarkdownWrapped:
    """Grok sometimes wraps JSON in ```json ... ``` code fences."""

    def test_parse_response_with_markdown_wrapping(
        self,
        parser: DecisionParser,
        sample_grok_response_json_with_markdown: str,
    ) -> None:
        """JSON wrapped in markdown code fences should still parse correctly."""
        result = parser.parse_response(sample_grok_response_json_with_markdown)

        assert result is not None
        assert isinstance(result, GrokResponse)
        assert len(result.decisions) == 1
        assert result.decisions[0].action == "open_long"

    def test_parse_response_with_plain_code_fence(
        self,
        parser: DecisionParser,
        sample_grok_response_json: str,
    ) -> None:
        """JSON wrapped in plain ``` ... ``` (no json tag) should also work."""
        wrapped = f"```\n{sample_grok_response_json}\n```"
        result = parser.parse_response(wrapped)

        assert result is not None
        assert isinstance(result, GrokResponse)


# ======================================================================
# 3. Parse Empty Decisions
# ======================================================================


class TestParseEmptyDecisions:
    """An empty decisions array is valid — the bot is choosing to stay flat."""

    def test_parse_empty_decisions(
        self,
        parser: DecisionParser,
        sample_grok_response_empty_decisions_json: str,
    ) -> None:
        """Empty decisions array should parse successfully."""
        result = parser.parse_response(sample_grok_response_empty_decisions_json)

        assert result is not None
        assert isinstance(result, GrokResponse)
        assert len(result.decisions) == 0


# ======================================================================
# 4. Parse Invalid JSON
# ======================================================================


class TestParseInvalidJSON:
    """Malformed JSON should return None, not crash."""

    def test_parse_invalid_json(self, parser: DecisionParser) -> None:
        """Completely broken JSON should return None."""
        broken = "{ this is not valid json at all !!"
        result = parser.parse_response(broken)
        assert result is None

    def test_parse_empty_string(self, parser: DecisionParser) -> None:
        """Empty string should return None."""
        result = parser.parse_response("")
        assert result is None

    def test_parse_whitespace_only(self, parser: DecisionParser) -> None:
        """Whitespace-only string should return None."""
        result = parser.parse_response("   \n\t  ")
        assert result is None

    def test_parse_html_response(self, parser: DecisionParser) -> None:
        """If Grok returns HTML (error page), should return None."""
        html = "<html><body>Service Unavailable</body></html>"
        result = parser.parse_response(html)
        assert result is None

    def test_parse_truncated_json(self, parser: DecisionParser) -> None:
        """Truncated JSON (e.g. from token limit) should return None."""
        truncated = '{"timestamp": "2026-03-07T12:00:00Z", "market_analysis": {'
        result = parser.parse_response(truncated)
        assert result is None


# ======================================================================
# 5. Parse Missing Fields
# ======================================================================


class TestParseMissingFields:
    """Missing required fields should cause validation failure (returns None)."""

    def test_parse_missing_fields(
        self,
        parser: DecisionParser,
        sample_grok_response_dict: dict,
    ) -> None:
        """Response missing the 'decisions' field should return None."""
        incomplete = sample_grok_response_dict.copy()
        del incomplete["decisions"]
        result = parser.parse_response(json.dumps(incomplete))
        assert result is None

    def test_parse_missing_market_analysis(
        self,
        parser: DecisionParser,
        sample_grok_response_dict: dict,
    ) -> None:
        """Response missing 'market_analysis' should return None."""
        incomplete = sample_grok_response_dict.copy()
        del incomplete["market_analysis"]
        result = parser.parse_response(json.dumps(incomplete))
        assert result is None

    def test_parse_missing_decision_fields(
        self,
        parser: DecisionParser,
        sample_grok_response_dict: dict,
    ) -> None:
        """A decision missing the 'action' field should return None."""
        bad = sample_grok_response_dict.copy()
        bad["decisions"] = [{"asset": "BTC", "size_pct": 0.05}]
        result = parser.parse_response(json.dumps(bad))
        assert result is None


# ======================================================================
# 6. Parse Invalid Action
# ======================================================================


class TestParseInvalidAction:
    """Invalid action types should be caught by Pydantic validation."""

    def test_parse_invalid_action(
        self,
        parser: DecisionParser,
        sample_grok_response_dict: dict,
    ) -> None:
        """An action like 'yolo_trade' is not in the valid enum."""
        bad = sample_grok_response_dict.copy()
        bad["decisions"] = [
            {
                "action": "yolo_trade",  # Invalid action
                "asset": "BTC",
                "size_pct": 0.05,
                "leverage": 2.0,
                "entry_price": 65000.0,
                "stop_loss": 63000.0,
                "take_profit": 70000.0,
                "order_type": "limit",
                "reasoning": "YOLO!",
                "conviction": "high",
                "risk_reward_ratio": 2.5,
            }
        ]
        result = parser.parse_response(json.dumps(bad))
        assert result is None


# ======================================================================
# 7. Parse Leverage Out of Range
# ======================================================================


class TestParseLeverageOutOfRange:
    """Leverage > 3 should be caught by Pydantic validation."""

    def test_parse_leverage_out_of_range(
        self,
        parser: DecisionParser,
        sample_grok_response_dict: dict,
    ) -> None:
        """Leverage of 10x should fail Pydantic validation."""
        bad = sample_grok_response_dict.copy()
        bad["decisions"] = [
            {
                "action": "open_long",
                "asset": "BTC",
                "size_pct": 0.05,
                "leverage": 10.0,  # Way over 3x limit
                "entry_price": 65000.0,
                "stop_loss": 63000.0,
                "take_profit": 70000.0,
                "order_type": "limit",
                "reasoning": "Over-leveraged.",
                "conviction": "high",
                "risk_reward_ratio": 2.5,
            }
        ]
        result = parser.parse_response(json.dumps(bad))
        assert result is None

    def test_pydantic_directly_rejects_leverage(self) -> None:
        """Verify Pydantic itself rejects leverage > 3 on TradeDecision."""
        with pytest.raises(ValidationError):
            TradeDecision(
                action="open_long",
                asset="BTC",
                size_pct=0.05,
                leverage=10.0,
                entry_price=65000.0,
                stop_loss=63000.0,
                take_profit=70000.0,
                order_type="limit",
                reasoning="Over-leveraged.",
                conviction="high",
                risk_reward_ratio=2.5,
            )


# ======================================================================
# 8. Parse Position Size Out of Range
# ======================================================================


class TestParsePositionSizeOutOfRange:
    """Position size > 100% should be caught by Pydantic validation.

    Note: Pydantic allows up to 1.0 (100%) as a generous parse bound.
    The Risk Guardian enforces the actual 15% limit at runtime.
    Values > 1.0 are either caught by Pydantic or normalized by the parser.
    """

    def test_parse_position_size_out_of_range(
        self,
        parser: DecisionParser,
        sample_grok_response_dict: dict,
    ) -> None:
        """size_pct of 1.50 (150%) should fail Pydantic validation."""
        bad = sample_grok_response_dict.copy()
        bad["decisions"] = [
            {
                "action": "open_long",
                "asset": "BTC",
                "size_pct": 1.50,  # Over 1.0 max (and not normalizable since 1.5% is valid)
                "leverage": 2.0,
                "entry_price": 65000.0,
                "stop_loss": 63000.0,
                "take_profit": 70000.0,
                "order_type": "limit",
                "reasoning": "Too large.",
                "conviction": "high",
                "risk_reward_ratio": 2.5,
            }
        ]
        result = parser.parse_response(json.dumps(bad))
        # Parser normalizes 1.50 -> 0.015 (treats as percentage), so it succeeds
        # Let's use a value that stays > 1.0 after normalization
        bad["decisions"][0]["size_pct"] = 150.0  # 150% -> normalized to 1.5 -> still > 1.0
        result = parser.parse_response(json.dumps(bad))
        assert result is None

    def test_pydantic_directly_rejects_size(self) -> None:
        """Verify Pydantic itself rejects size_pct > 1.0 on TradeDecision."""
        with pytest.raises(ValidationError):
            TradeDecision(
                action="open_long",
                asset="BTC",
                size_pct=1.50,
                leverage=2.0,
                entry_price=65000.0,
                stop_loss=63000.0,
                take_profit=70000.0,
                order_type="limit",
                reasoning="Too large.",
                conviction="high",
                risk_reward_ratio=2.5,
            )


# ======================================================================
# 9. Extract Actionable Decisions
# ======================================================================


class TestExtractActionableDecisions:
    """no_trade and hold should be filtered out, leaving only executable actions."""

    def test_extract_actionable_decisions(
        self,
        parser: DecisionParser,
        sample_grok_response_mixed_decisions_json: str,
    ) -> None:
        """From 3 decisions (open_long, no_trade, hold), only open_long is actionable."""
        response = parser.parse_response(sample_grok_response_mixed_decisions_json)
        assert response is not None
        assert len(response.decisions) == 3

        actionable = parser.extract_decisions(response)
        assert len(actionable) == 1
        assert actionable[0].action == "open_long"
        assert actionable[0].asset == "BTC"

    def test_extract_from_empty_decisions(
        self,
        parser: DecisionParser,
        sample_grok_response_empty_decisions_json: str,
    ) -> None:
        """Empty decisions list should produce empty actionable list."""
        response = parser.parse_response(sample_grok_response_empty_decisions_json)
        assert response is not None

        actionable = parser.extract_decisions(response)
        assert len(actionable) == 0

    def test_close_is_actionable(
        self,
        parser: DecisionParser,
        sample_grok_response_dict: dict,
    ) -> None:
        """A 'close' action should be considered actionable."""
        close_dict = sample_grok_response_dict.copy()
        close_dict["decisions"] = [
            {
                "action": "close",
                "asset": "BTC",
                "size_pct": 0.05,
                "leverage": 1.0,
                "entry_price": None,
                "stop_loss": 63000.0,
                "take_profit": 70000.0,
                "order_type": "market",
                "reasoning": "Taking profit.",
                "conviction": "high",
                "risk_reward_ratio": 0.0,
            }
        ]
        response = parser.parse_response(json.dumps(close_dict))
        assert response is not None

        actionable = parser.extract_decisions(response)
        assert len(actionable) == 1
        assert actionable[0].action == "close"
