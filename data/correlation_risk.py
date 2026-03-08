"""
Correlation-aware risk management.

Prevents the bot from opening multiple positions in highly correlated
assets, which amplifies portfolio risk.  Computes pairwise Pearson
correlation from hourly close prices and maintains a fast-fallback
table of known high-correlation crypto pairs.

Integration point: called as check #14 in ``RiskGuardian.validate()``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger


# ======================================================================
# Known high-correlation groups (fallback when candle data is sparse)
# ======================================================================

GROUP_CORRELATIONS: dict[str, list[str]] = {
    # Layer-1 majors tend to move together
    "BTC": ["ETH"],
    "ETH": ["BTC"],
    # Meme-coin cluster
    "DOGE": ["SHIB", "FLOKI", "BONK", "PEPE", "WIF"],
    "SHIB": ["DOGE", "FLOKI", "BONK", "PEPE", "WIF"],
    "FLOKI": ["DOGE", "SHIB", "BONK", "PEPE", "WIF"],
    "BONK": ["DOGE", "SHIB", "FLOKI", "PEPE", "WIF"],
    "PEPE": ["DOGE", "SHIB", "FLOKI", "BONK", "WIF"],
    "WIF": ["DOGE", "SHIB", "FLOKI", "BONK", "PEPE"],
    # Alt-L1 cluster
    "SOL": ["AVAX", "SUI", "APT"],
    "AVAX": ["SOL", "SUI", "APT"],
    "SUI": ["SOL", "AVAX", "APT"],
    "APT": ["SOL", "AVAX", "SUI"],
    # L2 cluster
    "ARB": ["OP"],
    "OP": ["ARB"],
}


class CorrelationRiskManager:
    """Manages correlation-based risk checks for the trading bot.

    Provides two levels of correlation detection:

    1. **Data-driven**: Computes pairwise Pearson correlation from hourly
       close prices when sufficient candle data is available.
    2. **Fallback**: Uses the ``GROUP_CORRELATIONS`` lookup table for
       known high-correlation crypto pairs when candle data is missing
       or insufficient.

    Usage::

        crm = CorrelationRiskManager()
        allowed, reason = crm.check_correlation_risk(
            new_asset="DOGE",
            open_positions=["SHIB", "BTC"],
            market_data=market_data,
        )
        if not allowed:
            print(f"Trade rejected: {reason}")
    """

    # Minimum number of shared hourly close prices needed for a
    # meaningful correlation calculation.
    MIN_DATA_POINTS: int = 20

    # ------------------------------------------------------------------
    # Correlation matrix from candle data
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_correlation_matrix(
        market_data: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, float]]:
        """Compute pairwise Pearson correlation from hourly close prices.

        Args:
            market_data: Nested dict keyed by asset symbol.  Each value
                must contain ``"candles"`` -> ``"1h"`` -> list of candle
                dicts, where each candle has a ``"c"`` (close) field, or
                a list of lists ``[timestamp, open, high, low, close, volume]``.

        Returns:
            Nested dict ``{asset_a: {asset_b: correlation, ...}, ...}``
            containing pairwise Pearson correlations.  Only pairs with
            at least ``MIN_DATA_POINTS`` shared data points are included.
        """
        # Extract close-price arrays per asset
        close_arrays: dict[str, list[float]] = {}

        for asset, data in market_data.items():
            closes = CorrelationRiskManager._extract_hourly_closes(data)
            if closes:
                close_arrays[asset] = closes

        assets = sorted(close_arrays.keys())
        matrix: dict[str, dict[str, float]] = {a: {} for a in assets}

        for i, asset_a in enumerate(assets):
            for j, asset_b in enumerate(assets):
                if i == j:
                    matrix[asset_a][asset_b] = 1.0
                    continue
                if j < i:
                    # Symmetric -- reuse already computed value if it exists
                    if asset_a in matrix.get(asset_b, {}):
                        matrix[asset_a][asset_b] = matrix[asset_b][asset_a]
                    continue

                closes_a = close_arrays[asset_a]
                closes_b = close_arrays[asset_b]

                # Align arrays to the shortest length
                min_len = min(len(closes_a), len(closes_b))
                if min_len < CorrelationRiskManager.MIN_DATA_POINTS:
                    continue  # Not enough data for this pair

                arr_a = np.array(closes_a[-min_len:], dtype=float)
                arr_b = np.array(closes_b[-min_len:], dtype=float)

                # Guard against constant arrays (zero std)
                if np.std(arr_a) == 0 or np.std(arr_b) == 0:
                    continue

                corr = float(np.corrcoef(arr_a, arr_b)[0, 1])
                matrix[asset_a][asset_b] = round(corr, 4)
                matrix[asset_b][asset_a] = round(corr, 4)

        return matrix

    # ------------------------------------------------------------------
    # Correlation risk check
    # ------------------------------------------------------------------

    @staticmethod
    def check_correlation_risk(
        new_asset: str,
        open_positions: list[str],
        market_data: dict[str, dict[str, Any]],
        threshold: float = 0.75,
    ) -> tuple[bool, str]:
        """Check whether a new position would create excessive correlation risk.

        Args:
            new_asset: Symbol of the asset the bot wants to open.
            open_positions: List of asset symbols with currently open
                positions.
            market_data: Full market data dict (same format as the
                trading cycle's ``market_data``).
            threshold: Correlation coefficient above which a new
                position is rejected.  Default 0.75.

        Returns:
            ``(is_allowed, reason)`` tuple.  ``is_allowed`` is ``True``
            if the trade may proceed, ``False`` if it should be blocked.
            ``reason`` is a human-readable explanation.
        """
        if not open_positions:
            return True, "No open positions; correlation check passed."

        new_asset_upper = new_asset.strip().upper()
        open_upper = [p.strip().upper() for p in open_positions]

        # 1. Try data-driven correlation first
        try:
            matrix = CorrelationRiskManager.calculate_correlation_matrix(
                market_data
            )
            if new_asset_upper in matrix:
                for pos_asset in open_upper:
                    if pos_asset in matrix.get(new_asset_upper, {}):
                        corr = matrix[new_asset_upper][pos_asset]
                        if abs(corr) > threshold:
                            return False, (
                                f"Correlation risk: {new_asset_upper} has "
                                f"{corr:.2f} correlation with open position "
                                f"{pos_asset} (threshold: {threshold:.2f}). "
                                f"Opening both amplifies portfolio risk."
                            )
                # Data-driven check passed for all open positions
                logger.debug(
                    "Correlation check passed (data-driven) for {asset}",
                    asset=new_asset_upper,
                )
                return True, "Correlation check passed (data-driven)."
        except Exception as exc:
            logger.warning(
                "Data-driven correlation failed, falling back to groups: {err}",
                err=exc,
            )

        # 2. Fallback: use GROUP_CORRELATIONS lookup
        known_correlated = GROUP_CORRELATIONS.get(new_asset_upper, [])
        for pos_asset in open_upper:
            if pos_asset in known_correlated:
                return False, (
                    f"Correlation risk (group fallback): {new_asset_upper} "
                    f"belongs to the same high-correlation group as open "
                    f"position {pos_asset}. Opening both amplifies "
                    f"portfolio risk."
                )

        return True, "Correlation check passed."

    # ------------------------------------------------------------------
    # Summary for context builder
    # ------------------------------------------------------------------

    @staticmethod
    def get_correlation_summary(
        market_data: dict[str, dict[str, Any]],
        open_positions: list[str] | None = None,
        threshold: float = 0.70,
    ) -> str:
        """Generate a human-readable correlation summary for Grok context.

        Lists highly correlated asset pairs that the AI should be aware
        of when making trading decisions.

        Args:
            market_data: Full market data dict.
            open_positions: Currently open position symbols (highlighted
                in the output).
            threshold: Pairs with |correlation| above this are listed.

        Returns:
            Formatted string suitable for inclusion in the context prompt.
        """
        if open_positions is None:
            open_positions = []

        open_upper = {p.strip().upper() for p in open_positions}
        lines: list[str] = ["### Correlation Awareness"]

        try:
            matrix = CorrelationRiskManager.calculate_correlation_matrix(
                market_data
            )
            high_corr_pairs: list[tuple[str, str, float]] = []
            seen: set[tuple[str, str]] = set()

            for asset_a, correlations in matrix.items():
                for asset_b, corr in correlations.items():
                    if asset_a == asset_b:
                        continue
                    pair = tuple(sorted([asset_a, asset_b]))
                    if pair in seen:
                        continue
                    seen.add(pair)
                    if abs(corr) > threshold:
                        high_corr_pairs.append((asset_a, asset_b, corr))

            if high_corr_pairs:
                high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                for a, b, corr in high_corr_pairs:
                    marker = ""
                    if a in open_upper or b in open_upper:
                        marker = " **[OPEN POSITION]**"
                    lines.append(
                        f"- {a}-{b}: {corr:+.2f} correlation{marker}"
                    )
            else:
                lines.append("- No highly correlated pairs detected from data.")

        except Exception:
            # Fall back to group info
            lines.append("- Using known correlation groups (insufficient data):")
            reported: set[tuple[str, str]] = set()
            for asset, correlated in GROUP_CORRELATIONS.items():
                for other in correlated:
                    pair = tuple(sorted([asset, other]))
                    if pair not in reported:
                        reported.add(pair)
                        marker = ""
                        if asset in open_upper or other in open_upper:
                            marker = " **[OPEN POSITION]**"
                        lines.append(f"  - {asset}-{other}: known high correlation{marker}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_hourly_closes(
        asset_data: dict[str, Any],
    ) -> list[float]:
        """Extract hourly close prices from a market data entry.

        Handles multiple candle data formats:
        - List of dicts with ``"c"`` key (Hyperliquid SDK format)
        - List of lists ``[ts, o, h, l, c, v]``
        - Pandas-like records with ``"close"`` key

        Args:
            asset_data: Single-asset data dict from market_data.

        Returns:
            List of close prices (floats), possibly empty.
        """
        candles = asset_data.get("candles", {})

        # Try 1h candles first
        hourly = candles.get("1h", [])
        if not hourly:
            # No hourly data available
            return []

        closes: list[float] = []
        for candle in hourly:
            if isinstance(candle, dict):
                # Dict format: {"c": 65000, ...} or {"close": 65000, ...}
                close_val = candle.get("c") or candle.get("close")
                if close_val is not None:
                    try:
                        closes.append(float(close_val))
                    except (ValueError, TypeError):
                        continue
            elif isinstance(candle, (list, tuple)) and len(candle) >= 5:
                # List format: [timestamp, open, high, low, close, ...]
                try:
                    closes.append(float(candle[4]))
                except (ValueError, TypeError, IndexError):
                    continue

        return closes
