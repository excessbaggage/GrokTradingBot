"""
Tests for the X Sentiment Fetcher — live X/Twitter sentiment via xAI API.

All tests use mocked API responses — no network, no API keys.
Verifies:
- Successful sentiment parsing from well-formed JSON
- Graceful fallback on API errors (returns empty dict)
- Score clamping to [-1.0, 1.0] range
- Invalid enum values fall back to defaults
- Markdown code fence stripping
- Caching behavior (reuse within TTL, refresh after)
- Context builder integration with sentiment data
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from data.x_sentiment import SentimentData, XSentimentFetcher, _clamp, _validate_enum


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def mock_openai_response():
    """Build a mock OpenAI API response with sentiment data."""
    def _build(content: str):
        mock_choice = MagicMock()
        mock_choice.message.content = content
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
        )
        return mock_response
    return _build


@pytest.fixture
def sample_sentiment_json() -> str:
    """Well-formed sentiment JSON for BTC and PEPE."""
    return json.dumps({
        "assets": {
            "BTC": {
                "score": 0.65,
                "momentum": "bullish",
                "volume": "high",
                "key_topics": ["whale accumulation", "ETF inflows", "halving"],
                "raw_summary": "Strong bullish sentiment on X.",
            },
            "PEPE": {
                "score": -0.30,
                "momentum": "bearish",
                "volume": "normal",
                "key_topics": ["whale dump", "fading hype"],
                "raw_summary": "Bearish sentiment, whales selling.",
            },
        }
    })


# ======================================================================
# SentimentData dataclass
# ======================================================================


class TestSentimentData:
    """SentimentData should have sensible defaults."""

    def test_defaults(self) -> None:
        s = SentimentData()
        assert s.score == 0.0
        assert s.momentum == "neutral"
        assert s.volume == "normal"
        assert s.key_topics == []
        assert s.raw_summary == ""

    def test_custom_values(self) -> None:
        s = SentimentData(
            score=0.8,
            momentum="bullish",
            volume="high",
            key_topics=["moon", "pump"],
            raw_summary="Very bullish.",
        )
        assert s.score == 0.8
        assert s.momentum == "bullish"
        assert len(s.key_topics) == 2


# ======================================================================
# Helper functions
# ======================================================================


class TestHelpers:
    """Test _clamp and _validate_enum utility functions."""

    def test_clamp_within_range(self) -> None:
        assert _clamp(0.5, -1.0, 1.0) == 0.5

    def test_clamp_below_min(self) -> None:
        assert _clamp(-2.0, -1.0, 1.0) == -1.0

    def test_clamp_above_max(self) -> None:
        assert _clamp(3.0, -1.0, 1.0) == 1.0

    def test_validate_enum_valid(self) -> None:
        assert _validate_enum("bullish", {"bullish", "bearish", "neutral"}, "neutral") == "bullish"

    def test_validate_enum_case_insensitive(self) -> None:
        assert _validate_enum("BULLISH", {"bullish", "bearish", "neutral"}, "neutral") == "bullish"

    def test_validate_enum_invalid(self) -> None:
        assert _validate_enum("yolo", {"bullish", "bearish", "neutral"}, "neutral") == "neutral"


# ======================================================================
# XSentimentFetcher — parsing
# ======================================================================


class TestSentimentParsing:
    """Test parsing of API responses into SentimentData."""

    @patch("data.x_sentiment.OpenAI")
    def test_parse_valid_response(
        self, mock_openai_cls, sample_sentiment_json, mock_openai_response
    ) -> None:
        """Well-formed JSON should parse into SentimentData objects."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response(
            sample_sentiment_json
        )
        mock_openai_cls.return_value = mock_client

        fetcher = XSentimentFetcher(api_key="test-key")
        result = fetcher.fetch_sentiment(["BTC", "PEPE"])

        assert "BTC" in result
        assert "PEPE" in result
        assert result["BTC"].score == 0.65
        assert result["BTC"].momentum == "bullish"
        assert result["BTC"].volume == "high"
        assert len(result["BTC"].key_topics) == 3
        assert result["PEPE"].score == -0.30
        assert result["PEPE"].momentum == "bearish"

    @patch("data.x_sentiment.OpenAI")
    def test_parse_markdown_wrapped(
        self, mock_openai_cls, sample_sentiment_json, mock_openai_response
    ) -> None:
        """JSON wrapped in markdown code fences should still parse."""
        wrapped = f"```json\n{sample_sentiment_json}\n```"
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response(wrapped)
        mock_openai_cls.return_value = mock_client

        fetcher = XSentimentFetcher(api_key="test-key")
        result = fetcher.fetch_sentiment(["BTC"])

        assert "BTC" in result
        assert result["BTC"].score == 0.65

    @patch("data.x_sentiment.OpenAI")
    def test_score_clamping(self, mock_openai_cls, mock_openai_response) -> None:
        """Scores outside [-1, 1] should be clamped."""
        extreme_json = json.dumps({
            "assets": {
                "BTC": {"score": 5.0, "momentum": "bullish", "volume": "high"},
                "ETH": {"score": -3.0, "momentum": "bearish", "volume": "low"},
            }
        })
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response(extreme_json)
        mock_openai_cls.return_value = mock_client

        fetcher = XSentimentFetcher(api_key="test-key")
        result = fetcher.fetch_sentiment(["BTC", "ETH"])

        assert result["BTC"].score == 1.0
        assert result["ETH"].score == -1.0

    @patch("data.x_sentiment.OpenAI")
    def test_invalid_enum_fallback(self, mock_openai_cls, mock_openai_response) -> None:
        """Invalid momentum/volume values should fall back to defaults."""
        bad_enums = json.dumps({
            "assets": {
                "BTC": {
                    "score": 0.5,
                    "momentum": "moon_soon",
                    "volume": "extreme",
                },
            }
        })
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response(bad_enums)
        mock_openai_cls.return_value = mock_client

        fetcher = XSentimentFetcher(api_key="test-key")
        result = fetcher.fetch_sentiment(["BTC"])

        assert result["BTC"].momentum == "neutral"  # fallback
        assert result["BTC"].volume == "normal"  # fallback

    @patch("data.x_sentiment.OpenAI")
    def test_missing_asset_in_response(
        self, mock_openai_cls, mock_openai_response
    ) -> None:
        """Assets missing from the API response should not appear in result."""
        partial = json.dumps({
            "assets": {
                "BTC": {"score": 0.5, "momentum": "bullish", "volume": "high"},
            }
        })
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response(partial)
        mock_openai_cls.return_value = mock_client

        fetcher = XSentimentFetcher(api_key="test-key")
        result = fetcher.fetch_sentiment(["BTC", "ETH", "SOL"])

        assert "BTC" in result
        assert "ETH" not in result
        assert "SOL" not in result

    @patch("data.x_sentiment.OpenAI")
    def test_key_topics_truncated(self, mock_openai_cls, mock_openai_response) -> None:
        """Key topics should be truncated to 5 max."""
        many_topics = json.dumps({
            "assets": {
                "BTC": {
                    "score": 0.5,
                    "momentum": "bullish",
                    "volume": "high",
                    "key_topics": ["a", "b", "c", "d", "e", "f", "g"],
                },
            }
        })
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response(many_topics)
        mock_openai_cls.return_value = mock_client

        fetcher = XSentimentFetcher(api_key="test-key")
        result = fetcher.fetch_sentiment(["BTC"])

        assert len(result["BTC"].key_topics) == 5


# ======================================================================
# XSentimentFetcher — error handling
# ======================================================================


class TestSentimentErrorHandling:
    """Errors should return empty dict, not crash."""

    @patch("data.x_sentiment.OpenAI")
    def test_empty_response(self, mock_openai_cls, mock_openai_response) -> None:
        """Empty API response should return empty dict."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response("")
        mock_openai_cls.return_value = mock_client

        fetcher = XSentimentFetcher(api_key="test-key")
        result = fetcher.fetch_sentiment(["BTC"])
        assert result == {}

    @patch("data.x_sentiment.OpenAI")
    def test_invalid_json(self, mock_openai_cls, mock_openai_response) -> None:
        """Invalid JSON should return empty dict."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response(
            "not valid json at all"
        )
        mock_openai_cls.return_value = mock_client

        fetcher = XSentimentFetcher(api_key="test-key")
        result = fetcher.fetch_sentiment(["BTC"])
        assert result == {}

    @patch("data.x_sentiment.OpenAI")
    def test_api_exception(self, mock_openai_cls) -> None:
        """API exception should return empty dict (graceful fallback)."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API down")
        mock_openai_cls.return_value = mock_client

        fetcher = XSentimentFetcher(api_key="test-key")
        result = fetcher.fetch_sentiment(["BTC"])
        assert result == {}

    def test_empty_assets_list(self) -> None:
        """Empty assets list should return empty dict without API call."""
        with patch("data.x_sentiment.OpenAI"):
            fetcher = XSentimentFetcher(api_key="test-key")
            result = fetcher.fetch_sentiment([])
            assert result == {}

    def test_missing_api_key(self) -> None:
        """Missing API key should raise ValueError."""
        with patch("data.x_sentiment.OpenAI"):
            with pytest.raises(ValueError, match="api_key must be provided"):
                XSentimentFetcher(api_key="")


# ======================================================================
# XSentimentFetcher — caching
# ======================================================================


class TestSentimentCaching:
    """Cache should reuse results within TTL."""

    @patch("data.x_sentiment.OpenAI")
    def test_cache_hit(
        self, mock_openai_cls, sample_sentiment_json, mock_openai_response
    ) -> None:
        """Second call within TTL should reuse cached data."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response(
            sample_sentiment_json
        )
        mock_openai_cls.return_value = mock_client

        fetcher = XSentimentFetcher(api_key="test-key", cache_minutes=5)

        # First call — hits API
        result1 = fetcher.fetch_sentiment(["BTC"])
        assert mock_client.chat.completions.create.call_count == 1

        # Second call — should use cache
        result2 = fetcher.fetch_sentiment(["BTC"])
        assert mock_client.chat.completions.create.call_count == 1  # No new call
        assert result1["BTC"].score == result2["BTC"].score

    @patch("data.x_sentiment.OpenAI")
    def test_cache_expiry(
        self, mock_openai_cls, sample_sentiment_json, mock_openai_response
    ) -> None:
        """After cache TTL expires, should make a new API call."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response(
            sample_sentiment_json
        )
        mock_openai_cls.return_value = mock_client

        fetcher = XSentimentFetcher(api_key="test-key", cache_minutes=0)  # Instant expiry

        # First call
        fetcher.fetch_sentiment(["BTC"])
        assert mock_client.chat.completions.create.call_count == 1

        # Force cache expiry by setting timestamp to past
        fetcher._cache = (time.time() - 600, fetcher._cache[1])

        # Second call — cache expired, hits API again
        fetcher.fetch_sentiment(["BTC"])
        assert mock_client.chat.completions.create.call_count == 2


# ======================================================================
# Context builder integration
# ======================================================================


class TestContextBuilderIntegration:
    """Verify sentiment data renders correctly in the context prompt."""

    def test_sentiment_renders_in_context(self) -> None:
        """Context builder should include sentiment lines when data provided."""
        from data.context_builder import build_context_prompt
        from config.trading_config import ASSET_UNIVERSE

        # Minimal market data for one asset
        market_data = {
            "BTC": {
                "price": 65000.0,
                "24h_change_pct": 0.02,
                "candles": {"1h": None, "4h": None, "1d": None},
                "funding": {"current_rate": 0.0001, "avg_7d_rate": 0.00008},
                "oi": {"current_oi": 5_000_000_000, "oi_24h_change_pct": 2.5},
            }
        }
        portfolio = {
            "total_equity": 10000,
            "available_margin": 9000,
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
        sentiment = {
            "BTC": SentimentData(
                score=0.72,
                momentum="bullish",
                volume="high",
                key_topics=["ETF inflows", "whale accumulation"],
                raw_summary="Very bullish.",
            )
        }

        context = build_context_prompt(
            market_data=market_data,
            portfolio=portfolio,
            recent_trades=[],
            risk_status=risk_status,
            sentiment_data=sentiment,
        )

        assert "X Sentiment Score" in context
        assert "+0.72" in context
        assert "BULLISH" in context
        assert "HIGH" in context
        assert "ETF inflows" in context

    def test_no_sentiment_renders_clean(self) -> None:
        """Context builder should work fine with no sentiment data."""
        from data.context_builder import build_context_prompt

        market_data = {
            "BTC": {
                "price": 65000.0,
                "24h_change_pct": 0.02,
                "candles": {"1h": None, "4h": None, "1d": None},
                "funding": {"current_rate": 0.0001, "avg_7d_rate": 0.00008},
                "oi": {"current_oi": 5_000_000_000, "oi_24h_change_pct": 2.5},
            }
        }
        portfolio = {
            "total_equity": 10000,
            "available_margin": 9000,
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

        context = build_context_prompt(
            market_data=market_data,
            portfolio=portfolio,
            recent_trades=[],
            risk_status=risk_status,
        )

        assert "X Sentiment Score" not in context
        assert "BTC-USD Perpetual" in context  # Still renders normal data

    def test_empty_dict_sentiment_renders_clean(self) -> None:
        """Context builder should handle empty dict (all fetches failed) correctly."""
        from data.context_builder import build_context_prompt

        market_data = {
            "BTC": {
                "price": 65000.0,
                "24h_change_pct": 0.02,
                "candles": {"1h": None, "4h": None, "1d": None},
                "funding": {"current_rate": 0.0001, "avg_7d_rate": 0.00008},
                "oi": {"current_oi": 5_000_000_000, "oi_24h_change_pct": 2.5},
            }
        }
        portfolio = {
            "total_equity": 10000,
            "available_margin": 9000,
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

        # Pass empty dict (not None) — simulates all sentiment fetches failing
        context = build_context_prompt(
            market_data=market_data,
            portfolio=portfolio,
            recent_trades=[],
            risk_status=risk_status,
            sentiment_data={},
        )

        assert "X Sentiment Score" not in context
        assert "BTC-USD Perpetual" in context

    def test_negative_sentiment_renders_correctly(self) -> None:
        """Negative sentiment scores should render with minus sign."""
        from data.context_builder import build_context_prompt

        market_data = {
            "BTC": {
                "price": 65000.0,
                "24h_change_pct": -0.05,
                "candles": {"1h": None, "4h": None, "1d": None},
                "funding": {"current_rate": 0.0001, "avg_7d_rate": 0.00008},
                "oi": {"current_oi": 5_000_000_000, "oi_24h_change_pct": 2.5},
            }
        }
        portfolio = {
            "total_equity": 10000,
            "available_margin": 9000,
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
        sentiment = {
            "BTC": SentimentData(
                score=-0.45,
                momentum="bearish",
                volume="low",
                key_topics=["whale dump"],
                raw_summary="Bearish.",
            )
        }

        context = build_context_prompt(
            market_data=market_data,
            portfolio=portfolio,
            recent_trades=[],
            risk_status=risk_status,
            sentiment_data=sentiment,
        )

        assert "-0.45" in context
        assert "BEARISH" in context
        assert "LOW" in context


# ======================================================================
# Additional edge case tests
# ======================================================================


class TestSentimentEdgeCases:
    """Additional edge cases found during code review."""

    @patch("data.x_sentiment.OpenAI")
    def test_non_numeric_score_isolates_failure(
        self, mock_openai_cls, mock_openai_response
    ) -> None:
        """Non-numeric score on one asset should not poison other assets."""
        bad_data = json.dumps({
            "assets": {
                "BTC": {"score": "very_bullish", "momentum": "bullish", "volume": "high"},
                "ETH": {"score": 0.3, "momentum": "neutral", "volume": "normal"},
            }
        })
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response(bad_data)
        mock_openai_cls.return_value = mock_client

        fetcher = XSentimentFetcher(api_key="test-key")
        result = fetcher.fetch_sentiment(["BTC", "ETH"])

        # BTC should fail (non-numeric score), but ETH should still parse
        assert "BTC" not in result
        assert "ETH" in result
        assert result["ETH"].score == 0.3

    @patch("data.x_sentiment.OpenAI")
    def test_unwrapped_json_format(
        self, mock_openai_cls, mock_openai_response
    ) -> None:
        """JSON without 'assets' wrapper should still parse."""
        unwrapped = json.dumps({
            "BTC": {"score": 0.5, "momentum": "bullish", "volume": "high"},
        })
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response(unwrapped)
        mock_openai_cls.return_value = mock_client

        fetcher = XSentimentFetcher(api_key="test-key")
        result = fetcher.fetch_sentiment(["BTC"])

        assert "BTC" in result
        assert result["BTC"].score == 0.5

    @patch("data.x_sentiment.OpenAI")
    def test_empty_choices_returns_empty(self, mock_openai_cls) -> None:
        """API response with empty choices list should return empty dict."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = []  # Empty choices
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client

        fetcher = XSentimentFetcher(api_key="test-key")
        result = fetcher.fetch_sentiment(["BTC"])
        assert result == {}

    @patch("data.x_sentiment.OpenAI")
    def test_none_content_with_tool_calls(self, mock_openai_cls) -> None:
        """Response with tool_calls but no content should return empty dict."""
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_choice.message.tool_calls = [MagicMock()]  # Has tool calls
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client

        fetcher = XSentimentFetcher(api_key="test-key")
        result = fetcher.fetch_sentiment(["BTC"])
        assert result == {}

    @patch("data.x_sentiment.OpenAI")
    def test_raw_summary_truncated(self, mock_openai_cls, mock_openai_response) -> None:
        """Raw summary longer than 500 chars should be truncated."""
        long_summary = "A" * 600
        data = json.dumps({
            "assets": {
                "BTC": {
                    "score": 0.5,
                    "momentum": "bullish",
                    "volume": "high",
                    "raw_summary": long_summary,
                },
            }
        })
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response(data)
        mock_openai_cls.return_value = mock_client

        fetcher = XSentimentFetcher(api_key="test-key")
        result = fetcher.fetch_sentiment(["BTC"])

        assert len(result["BTC"].raw_summary) == 500
