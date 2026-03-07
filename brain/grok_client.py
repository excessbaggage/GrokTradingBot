"""
Grok API client for the trading bot's brain layer.

Wraps the xAI API (accessed via the OpenAI SDK with a custom base_url)
to send trading context to the Grok model and receive structured
trading decisions.

Features:
- Automatic retries with exponential backoff (via tenacity)
- Request timeout handling (30s per request)
- Full request/response logging for auditability
- Health check endpoint for connectivity verification
"""

from __future__ import annotations

from typing import Optional

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
    GROK_MAX_TOKENS,
    GROK_TEMPERATURE,
)


class GrokClient:
    """Client for communicating with the xAI Grok API.

    Uses the OpenAI Python SDK pointed at the xAI endpoint. Handles
    authentication, retries, timeouts, and structured logging of every
    interaction for auditability.

    Args:
        api_key: The xAI API key for authentication.
        model: The Grok model identifier (e.g., ``"grok-4"``).
        max_tokens: Maximum tokens in the response. Defaults to config value.
        temperature: Sampling temperature. Defaults to config value.
        timeout: Per-request timeout in seconds. Defaults to ``30``.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int = GROK_MAX_TOKENS,
        temperature: float = GROK_TEMPERATURE,
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must be provided and non-empty")

        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout

        self._client = OpenAI(
            api_key=api_key,
            base_url=XAI_BASE_URL,
            timeout=self._timeout,
        )

        logger.info(
            "GrokClient initialised | model={} | max_tokens={} | temperature={} | timeout={}s",
            self._model,
            self._max_tokens,
            self._temperature,
            self._timeout,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type((APIConnectionError, APITimeoutError, RateLimitError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=before_sleep_log(logger, "WARNING"),  # type: ignore[arg-type]
        reraise=True,
    )
    def get_trading_decision(self, system_prompt: str, context: str) -> str:
        """Send the system prompt and market context to Grok and return the raw response.

        The method sends a two-message conversation (system + user) to the
        Grok model and returns the assistant's text content verbatim.  The
        caller is responsible for parsing and validating the JSON.

        Args:
            system_prompt: The static Sentinel system prompt.
            context: The dynamic market-data / portfolio context for this cycle.

        Returns:
            The raw text of Grok's response.

        Raises:
            openai.APIConnectionError: If the API is unreachable after retries.
            openai.APITimeoutError: If every attempt times out.
            openai.RateLimitError: If rate-limited after retries.
            Exception: Any other unexpected API error (not retried).
        """
        logger.info("Sending trading context to Grok | model={}", self._model)
        logger.debug("System prompt length: {} chars", len(system_prompt))
        logger.debug("Context length: {} chars", len(context))

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context},
            ],
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )

        raw_text: str = response.choices[0].message.content or ""

        # Log token usage when available
        if response.usage:
            logger.info(
                "Grok response received | prompt_tokens={} | completion_tokens={} | total_tokens={}",
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                response.usage.total_tokens,
            )
        else:
            logger.info("Grok response received | token usage not reported")

        logger.debug("Raw Grok response ({} chars): {}", len(raw_text), raw_text[:500])

        return raw_text

    def health_check(self) -> bool:
        """Verify connectivity to the xAI API with a minimal request.

        Sends a trivial prompt and checks that a response is returned.
        This is useful at startup or in monitoring dashboards.

        Returns:
            ``True`` if the API responded successfully, ``False`` otherwise.
        """
        try:
            logger.info("Running Grok health check...")
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "user", "content": "ping"},
                ],
                max_tokens=8,
                temperature=0.0,
            )
            ok = bool(response.choices and response.choices[0].message.content)
            logger.info("Grok health check {} ", "PASSED" if ok else "FAILED (empty response)")
            return ok
        except Exception as exc:
            logger.error("Grok health check FAILED | error={}", exc)
            return False
