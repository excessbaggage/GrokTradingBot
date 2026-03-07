"""
Decision parser for Grok's raw JSON responses.

Responsible for transforming the raw text returned by the Grok API into
validated Pydantic models. Because Grok's output is untrusted input,
this module is written with extreme defensiveness:

- Strips markdown fences and surrounding whitespace.
- Attempts JSON parsing with informative error messages.
- Validates against the ``GrokResponse`` Pydantic schema.
- Never raises on malformed input -- returns ``None`` so the bot
  can safely skip the cycle (fail safe, not fail open).
"""

from __future__ import annotations

import json
import re
from typing import Optional

from loguru import logger
from pydantic import ValidationError

from brain.models import GrokResponse, TradeDecision


class DecisionParser:
    """Parse and validate Grok trading responses into structured models.

    This class is intentionally stateless -- every method is a pure
    transformation from input data to validated output (or ``None``
    on failure). All errors are logged but never propagated.
    """

    # Actions that represent an actual order to execute
    _ACTIONABLE_ACTIONS: frozenset[str] = frozenset({
        "open_long",
        "open_short",
        "close",
        "adjust_stop",
    })

    # Regex to strip markdown code fences (```json ... ``` or ``` ... ```)
    _CODE_FENCE_RE: re.Pattern[str] = re.compile(
        r"```(?:json)?\s*([\s\S]*?)\s*```",
        re.IGNORECASE,
    )

    def parse_response(self, raw_response: str) -> Optional[GrokResponse]:
        """Parse a raw Grok response string into a validated ``GrokResponse``.

        Processing pipeline:
        1. Strip surrounding whitespace.
        2. Remove markdown code fences if present.
        3. Parse as JSON.
        4. Validate against the ``GrokResponse`` Pydantic model.

        Args:
            raw_response: The raw text returned by ``GrokClient.get_trading_decision``.

        Returns:
            A validated ``GrokResponse`` instance, or ``None`` if parsing
            or validation failed at any stage.
        """
        if not raw_response or not raw_response.strip():
            logger.error("Received empty or whitespace-only response from Grok")
            return None

        # Step 1 -- strip outer whitespace
        cleaned = raw_response.strip()

        # Step 2 -- remove markdown code fences
        cleaned = self._strip_code_fences(cleaned)

        # Step 3 -- parse JSON
        parsed_json = self._safe_json_parse(cleaned)
        if parsed_json is None:
            return None

        # Step 4 -- validate against Pydantic model
        return self._validate_model(parsed_json)

    def extract_decisions(self, grok_response: GrokResponse) -> list[TradeDecision]:
        """Extract only actionable trade decisions from a validated response.

        Filters out ``"no_trade"`` and ``"hold"`` actions, returning only
        decisions that require order placement or position modification.

        Args:
            grok_response: A validated ``GrokResponse`` instance.

        Returns:
            A list of ``TradeDecision`` objects with actionable actions.
            May be empty if no trades are warranted.
        """
        actionable: list[TradeDecision] = [
            decision
            for decision in grok_response.decisions
            if decision.action in self._ACTIONABLE_ACTIONS
        ]

        logger.info(
            "Extracted {}/{} actionable decisions from Grok response",
            len(actionable),
            len(grok_response.decisions),
        )

        for decision in actionable:
            logger.debug(
                "Actionable decision: action={} asset={} size_pct={} leverage={} conviction={}",
                decision.action,
                decision.asset,
                decision.size_pct,
                decision.leverage,
                decision.conviction,
            )

        return actionable

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _strip_code_fences(self, text: str) -> str:
        """Remove markdown code fences wrapping a JSON payload.

        Handles both ```` ```json ... ``` ```` and ```` ``` ... ``` ````.
        If no fences are found the text is returned unchanged.

        Args:
            text: The potentially fence-wrapped text.

        Returns:
            The text with code fences removed.
        """
        match = self._CODE_FENCE_RE.search(text)
        if match:
            extracted = match.group(1).strip()
            logger.debug("Stripped markdown code fences from Grok response")
            return extracted
        return text

    def _safe_json_parse(self, text: str) -> Optional[dict]:
        """Attempt to parse a string as JSON, returning ``None`` on failure.

        Args:
            text: The JSON string to parse.

        Returns:
            The parsed dictionary, or ``None`` if the text is not valid JSON.
        """
        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                logger.error(
                    "Grok response parsed as JSON but is not a dict (got {})",
                    type(data).__name__,
                )
                return None
            return data
        except json.JSONDecodeError as exc:
            # Log a snippet of the problematic text for debugging
            snippet = text[:200] + ("..." if len(text) > 200 else "")
            logger.error(
                "Failed to parse Grok response as JSON | error={} | snippet={}",
                exc,
                snippet,
            )
            return None

    def _normalize_decisions(self, data: dict) -> dict:
        """Normalize decision values before Pydantic validation.

        Grok sometimes returns size_pct as a percentage (e.g. 8.0 for 8%)
        instead of a decimal fraction (0.08). This method detects and fixes
        that, as well as other common AI formatting issues.

        Args:
            data: The parsed JSON dictionary.

        Returns:
            The same dictionary with normalized decision values.
        """
        decisions = data.get("decisions", [])
        for d in decisions:
            if not isinstance(d, dict):
                continue

            # Fix size_pct: if >= 1.0, Grok likely returned a percentage.
            # A value of 1.0 means "100%" (not "1%") since max position
            # size is 25% — no valid trade uses 100% of capital as a decimal.
            size_pct = d.get("size_pct")
            if isinstance(size_pct, (int, float)) and size_pct >= 1.0:
                d["size_pct"] = size_pct / 100.0
                logger.debug(
                    "Normalized size_pct: {} -> {}",
                    size_pct, d["size_pct"],
                )

            # Fix risk_reward_ratio: ensure non-negative
            rr = d.get("risk_reward_ratio")
            if isinstance(rr, (int, float)) and rr < 0:
                d["risk_reward_ratio"] = abs(rr)

        return data

    def _validate_model(self, data: dict) -> Optional[GrokResponse]:
        """Validate a parsed dict against the ``GrokResponse`` Pydantic model.

        Args:
            data: The parsed JSON dictionary.

        Returns:
            A validated ``GrokResponse``, or ``None`` if validation failed.
        """
        # Normalize before validation to fix common Grok formatting issues
        data = self._normalize_decisions(data)

        try:
            response = GrokResponse.model_validate(data)
            logger.info(
                "Grok response validated successfully | decisions={} | stance={}",
                len(response.decisions),
                response.overall_stance[:80] if response.overall_stance else "(empty)",
            )
            return response
        except ValidationError as exc:
            logger.error(
                "Grok response failed Pydantic validation | errors={}",
                exc.error_count(),
            )
            for error in exc.errors():
                logger.error(
                    "  Validation error: loc={} msg={} type={}",
                    error.get("loc"),
                    error.get("msg"),
                    error.get("type"),
                )
            return None
