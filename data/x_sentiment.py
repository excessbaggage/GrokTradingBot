"""
Live X/Twitter sentiment fetcher via xAI Agent Tools API.

Uses the xAI Grok API with the ``x_search`` tool to fetch real-time
sentiment data from X for each asset in the trading universe. The model
autonomously searches X, reads posts, and returns structured sentiment
scores.

Features:
- Batches all assets into a single API call for efficiency
- Uses ``grok-3-mini`` by default (fast, cheap — this is data gathering, not decision-making)
- Caches results to avoid redundant API calls within a configurable window
- Graceful fallback: returns empty dict on any failure (cycle continues without sentiment)
- Same retry logic as ``grok_client.py`` (3 attempts, exponential backoff)
"""

from __future__ import annotations

import json
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
You are a crypto market sentiment analyst. Your job is to search X (Twitter) \
for recent posts about the given cryptocurrency assets and produce a structured \
sentiment report.

For each asset, search X for recent discussion (last few hours). Analyze the \
tone, volume, and key themes of the conversation.

Return your analysis as a JSON object with this exact structure:
{
    "assets": {
        "BTC": {
            "score": 0.45,
            "momentum": "bullish",
            "volume": "high",
            "key_topics": ["whale accumulation", "ETF inflows"],
            "raw_summary": "Predominantly bullish sentiment with focus on institutional buying."
        },
        ...same structure for each asset...
    }
}

Score guidelines:
- -1.0 to -0.6: Extreme bearish (panic, capitulation talk, mass liquidations)
- -0.6 to -0.2: Bearish (concern, selling pressure, negative news)
- -0.2 to 0.2: Neutral (mixed, no clear direction)
- 0.2 to 0.6: Bullish (optimism, buying interest, positive catalysts)
- 0.6 to 1.0: Extreme bullish (euphoria, FOMO, parabolic calls)

Momentum: Overall direction of sentiment shift ("bullish" if improving, "bearish" if worsening, "neutral" if stable).
Volume: Discussion volume relative to normal ("high" if trending, "low" if quiet).
Key topics: 2-3 most discussed themes (e.g., "whale moves", "regulatory news", "exchange listing").

IMPORTANT: Only return the JSON object, nothing else. If you cannot find data for an asset, use score=0.0, momentum="neutral", volume="low".\
"""


class XSentimentFetcher:
    """Fetch live X/Twitter sentiment for trading assets via xAI API.

    Uses the xAI Agent Tools API with the ``x_search`` tool to let Grok
    autonomously search X, analyze posts, and return structured sentiment.
    Results are cached for ``X_SENTIMENT_CACHE_MINUTES`` to avoid
    redundant API calls.

    Args:
        api_key: The xAI API key (same key used for trading decisions).
        model: Model to use for sentiment analysis. Defaults to config value.
        cache_minutes: How long to cache results. Defaults to config value.
    """

    # Tool definition for xAI's native X search
    _X_SEARCH_TOOL: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "x_search",
            "description": "Search X (Twitter) for posts matching a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for X posts",
                    },
                },
                "required": ["query"],
            },
        },
    }

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
            timeout=60.0,
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
        before_sleep=before_sleep_log(logger, "WARNING"),  # type: ignore[arg-type]
        reraise=True,
    )
    def _fetch_with_retry(
        self, assets: list[str]
    ) -> dict[str, SentimentData]:
        """Make the API call with retry logic.

        This method handles the multi-turn agent tools flow:
        1. Send initial request with x_search tool definition
        2. If model requests tool calls, the xAI API handles them server-side
        3. Parse the final structured JSON response

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

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SENTIMENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            tools=[self._X_SEARCH_TOOL],
            max_tokens=4096,
            temperature=0.3,  # Low temp for consistent structured output
        )

        raw_text = response.choices[0].message.content or ""

        if response.usage:
            logger.debug(
                "Sentiment API usage | prompt={} | completion={} | total={}",
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
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

        return result


def _clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value to the given range."""
    return max(min_val, min(max_val, value))


def _validate_enum(value: str, valid: set[str], default: str) -> str:
    """Return the value if it's in the valid set, otherwise return default."""
    return value.lower() if value.lower() in valid else default
