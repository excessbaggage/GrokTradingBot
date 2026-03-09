"""
Live X/Twitter sentiment fetcher via xAI Responses API with built-in x_search.

Uses the xAI Responses API with the built-in ``x_search`` tool to fetch
real-time sentiment data from X for each asset in the trading universe.
The API auto-executes x_search server-side — no manual tool-call handling
is needed.

Features:
- Batches all assets into a single API call for efficiency
- Uses the Responses API with ``{"type": "x_search"}`` (built-in, server-side)
- Caches results to avoid redundant API calls within a configurable window
- Graceful fallback: returns empty dict on any failure (cycle continues without sentiment)
- Same retry logic as ``grok_client.py`` (3 attempts, exponential backoff)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger
from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from config.trading_config import (
    XAI_BASE_URL,
    X_SENTIMENT_MODEL,
    X_SENTIMENT_CACHE_MINUTES,
)


@dataclass
class SentimentData:
    """Structured sentiment data for a single asset."""

    score: float = 0.0  # -1.0 (extreme bearish) to +1.0 (extreme bullish)
    momentum: str = "neutral"  # "bullish", "bearish", "neutral"
    volume: str = "normal"  # "high", "normal", "low"
    key_topics: list[str] = field(default_factory=list)
    raw_summary: str = ""


# System prompt for the sentiment-gathering call
_SENTIMENT_SYSTEM_PROMPT = """\
Crypto sentiment analyst. Search X for each asset, return JSON only:
{"assets":{"BTC":{"score":0.45,"momentum":"bullish","volume":"high","key_topics":["topic1","topic2"]},...}}
Score: -1(panic) to +1(euphoria). Momentum: bullish/bearish/neutral. Volume: high/normal/low.
Key_topics: max 2 per asset. No raw_summary needed. Missing data → score=0, momentum=neutral, volume=low.\
"""


class XSentimentFetcher:
    """Fetch live X/Twitter sentiment for trading assets via xAI Responses API.

    Uses the xAI Responses API with the built-in ``x_search`` tool.
    The API auto-executes x_search server-side — no manual tool handling.
    Results are cached for ``X_SENTIMENT_CACHE_MINUTES`` to avoid
    redundant API calls.

    Args:
        api_key: The xAI API key (same key used for trading decisions).
        model: Model to use for sentiment analysis. Defaults to config value.
        cache_minutes: How long to cache results. Defaults to config value.
    """

    def __init__(
        self,
        api_key: str,
        model: str = X_SENTIMENT_MODEL,
        cache_minutes: int = X_SENTIMENT_CACHE_MINUTES,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must be provided and non-empty")

        self._model = model
        self._cache_minutes = cache_minutes
        self._client = OpenAI(
            api_key=api_key,
            base_url=XAI_BASE_URL,
            timeout=120.0,
        )

        # Simple in-memory cache: (timestamp, results)
        self._cache: tuple[float, dict[str, SentimentData]] | None = None

        logger.info(
            "XSentimentFetcher initialised | model={} | cache={}min",
            self._model,
            self._cache_minutes,
        )

    def fetch_sentiment(
        self, assets: list[str]
    ) -> dict[str, SentimentData]:
        """Fetch X sentiment for the given assets.

        Returns cached results if still fresh. On any failure, returns
        an empty dict so the trading cycle can continue without sentiment.

        Args:
            assets: List of asset symbols (e.g., ``["BTC", "ETH", "PEPE"]``).

        Returns:
            Dict mapping asset symbol to ``SentimentData``. May be empty
            on failure or if no assets are provided.
        """
        if not assets:
            return {}

        # Check cache freshness
        if self._cache is not None:
            cache_time, cached_data = self._cache
            age_minutes = (time.time() - cache_time) / 60
            if age_minutes < self._cache_minutes:
                logger.debug(
                    "Using cached X sentiment data | age={:.1f}min",
                    age_minutes,
                )
                return cached_data

        # Fetch fresh sentiment
        try:
            result = self._fetch_with_retry(assets)
            self._cache = (time.time(), result)
            logger.info(
                "X sentiment fetched | assets={} | scores={}",
                len(result),
                {k: f"{v.score:+.2f}" for k, v in result.items()},
            )
            return result
        except Exception as exc:
            logger.warning(
                "X sentiment fetch failed, continuing without sentiment | error={}",
                exc,
            )
            return {}

    @retry(
        retry=retry_if_exception_type(
            (APIConnectionError, APITimeoutError, RateLimitError)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
        reraise=True,
    )
    def _fetch_with_retry(
        self, assets: list[str]
    ) -> dict[str, SentimentData]:
        """Make the API call with retry logic.

        Uses the xAI Responses API with built-in x_search tool.
        The API auto-executes x_search server-side and returns the
        final structured response with sentiment data.

        Args:
            assets: List of asset symbols to analyze.

        Returns:
            Dict mapping asset symbol to SentimentData.
        """
        asset_list = ", ".join(assets)
        user_message = (
            f"Search X for recent sentiment on these crypto assets and "
            f"return your analysis as JSON: {asset_list}"
        )

        logger.debug(
            "Querying xAI for X sentiment | model={} | assets={}",
            self._model,
            asset_list,
        )

        # Use the Responses API with built-in x_search tool
        # The API auto-executes x_search server-side
        response = self._client.responses.create(
            model=self._model,
            instructions=_SENTIMENT_SYSTEM_PROMPT,
            input=[
                {"role": "user", "content": user_message},
            ],
            tools=[
                {"type": "x_search"},
            ],
            temperature=0.3,  # Low temp for consistent structured output
        )

        # Extract text content from the response
        raw_text = ""
        for item in response.output:
            if hasattr(item, "content") and isinstance(item.content, list):
                for block in item.content:
                    if hasattr(block, "text"):
                        raw_text += block.text
            elif hasattr(item, "text"):
                raw_text += item.text

        if not raw_text:
            logger.warning("xAI Responses API returned no text content")
            return {}

        if hasattr(response, "usage") and response.usage:
            logger.debug(
                "Sentiment API usage | input={} | output={} | total={}",
                response.usage.input_tokens,
                response.usage.output_tokens,
                response.usage.total_tokens,
            )

        return self._parse_sentiment_response(raw_text, assets)

    def _parse_sentiment_response(
        self, raw_text: str, assets: list[str]
    ) -> dict[str, SentimentData]:
        """Parse the structured JSON response into SentimentData objects.

        Handles common formatting issues (markdown fences, missing fields)
        and falls back to neutral sentiment for any unparseable asset.

        Args:
            raw_text: Raw text from the API response.
            assets: Expected asset list (for fallback generation).

        Returns:
            Dict mapping asset symbol to SentimentData.
        """
        if not raw_text.strip():
            logger.warning("Empty sentiment response from API")
            return {}

        # Strip markdown code fences if present
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            # Remove ```json ... ``` or ``` ... ```
            lines = cleaned.split("\n")
            # Drop first and last lines (fences)
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Failed to parse sentiment JSON | error={} | snippet={}",
                exc,
                cleaned[:200],
            )
            return {}

        # Extract the assets dict
        assets_data = data.get("assets", data)  # Handle both wrapped and unwrapped
        if not isinstance(assets_data, dict):
            logger.warning("Sentiment response is not a dict")
            return {}

        result: dict[str, SentimentData] = {}
        for asset in assets:
            # Try both upper and lower case keys
            asset_info = assets_data.get(asset) or assets_data.get(asset.lower())
            if not isinstance(asset_info, dict) or not asset_info:
                continue

            try:
                result[asset] = SentimentData(
                    score=_clamp(float(asset_info.get("score", 0.0)), -1.0, 1.0),
                    momentum=_validate_enum(
                        asset_info.get("momentum", "neutral"),
                        {"bullish", "bearish", "neutral"},
                        "neutral",
                    ),
                    volume=_validate_enum(
                        asset_info.get("volume", "normal"),
                        {"high", "normal", "low"},
                        "normal",
                    ),
                    key_topics=asset_info.get("key_topics", [])[:5],
                    raw_summary=str(asset_info.get("raw_summary", ""))[:500],
                )
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "Failed to parse sentiment for {asset}: {err}",
                    asset=asset,
                    err=exc,
                )
                continue

        return result


def _clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value to the given range."""
    return max(min_val, min(max_val, value))


def _validate_enum(value: str, valid: set[str], default: str) -> str:
    """Return the value if it's in the valid set, otherwise return default."""
    return value.lower() if value.lower() in valid else default
