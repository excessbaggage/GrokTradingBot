"""Liquidation cluster estimation for perpetual futures.

Estimates where liquidation price clusters likely exist based on
open interest and leverage distribution modeling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from utils.logger import logger


# Common leverage tiers and their estimated OI distribution
# Based on typical exchange data: most OI is at lower leverage
LEVERAGE_DISTRIBUTION: dict[int, float] = {
    2: 0.30,   # 30% of OI at 2x
    3: 0.25,   # 25% at 3x
    5: 0.20,   # 20% at 5x
    10: 0.15,  # 15% at 10x
    20: 0.07,  # 7% at 20x
    50: 0.03,  # 3% at 50x
}

# Maintenance margin rates by leverage tier (Hyperliquid uses tiered rates)
MAINTENANCE_MARGIN: dict[int, float] = {
    2: 0.005,   # 0.5%
    3: 0.0067,  # 0.67%
    5: 0.01,    # 1%
    10: 0.02,   # 2%
    20: 0.025,  # 2.5%
    50: 0.03,   # 3%
}


@dataclass
class LiquidationCluster:
    """A cluster of estimated liquidation prices."""

    price_level: float
    side: str  # "long" or "short"
    estimated_oi_usd: float  # Estimated OI that would liquidate here
    leverage: int
    distance_from_current_pct: float  # How far from current price


@dataclass
class LiquidationHeatmap:
    """Complete liquidation heatmap for a single asset."""

    asset: str
    current_price: float
    total_oi_usd: float
    long_clusters: list[LiquidationCluster] = field(default_factory=list)
    short_clusters: list[LiquidationCluster] = field(default_factory=list)
    nearest_long_liq: float = 0.0  # Nearest long liquidation cluster below
    nearest_short_liq: float = 0.0  # Nearest short liquidation cluster above
    long_liq_density_5pct: float = 0.0  # OI that liquidates within 5% down
    short_liq_density_5pct: float = 0.0  # OI that liquidates within 5% up


class LiquidationEstimator:
    """Estimate liquidation price clusters from OI and leverage distribution."""

    def estimate(
        self,
        asset: str,
        current_price: float,
        total_oi_usd: float,
    ) -> LiquidationHeatmap:
        """Compute liquidation clusters for an asset.

        For each leverage tier:
        - Compute long liquidation price: entry * (1 - 1/leverage + maintenance_margin)
        - Compute short liquidation price: entry * (1 + 1/leverage - maintenance_margin)
        - Weight by estimated OI at that leverage tier

        Args:
            asset: Symbol (BTC, ETH, SOL).
            current_price: Current mark price.
            total_oi_usd: Total open interest in USD.

        Returns:
            LiquidationHeatmap with sorted clusters.
        """
        if current_price <= 0 or total_oi_usd <= 0:
            return LiquidationHeatmap(
                asset=asset,
                current_price=current_price,
                total_oi_usd=total_oi_usd,
            )

        heatmap = LiquidationHeatmap(
            asset=asset,
            current_price=current_price,
            total_oi_usd=total_oi_usd,
        )

        for leverage, oi_fraction in LEVERAGE_DISTRIBUTION.items():
            oi_at_leverage = total_oi_usd * oi_fraction
            maint = MAINTENANCE_MARGIN.get(leverage, 0.01)

            # Long liquidation price (below current)
            # Liq price = entry * (1 - (1/leverage) + maintenance_margin_rate)
            long_liq_price = current_price * (1 - (1 / leverage) + maint)
            long_distance = (current_price - long_liq_price) / current_price

            heatmap.long_clusters.append(LiquidationCluster(
                price_level=round(long_liq_price, 2),
                side="long",
                estimated_oi_usd=round(oi_at_leverage / 2, 2),  # Split OI 50/50 long/short
                leverage=leverage,
                distance_from_current_pct=round(long_distance, 4),
            ))

            # Short liquidation price (above current)
            # Liq price = entry * (1 + (1/leverage) - maintenance_margin_rate)
            short_liq_price = current_price * (1 + (1 / leverage) - maint)
            short_distance = (short_liq_price - current_price) / current_price

            heatmap.short_clusters.append(LiquidationCluster(
                price_level=round(short_liq_price, 2),
                side="short",
                estimated_oi_usd=round(oi_at_leverage / 2, 2),
                leverage=leverage,
                distance_from_current_pct=round(short_distance, 4),
            ))

        # Sort: long clusters by price descending (nearest first), short ascending
        heatmap.long_clusters.sort(key=lambda c: c.price_level, reverse=True)
        heatmap.short_clusters.sort(key=lambda c: c.price_level)

        # Summary metrics
        if heatmap.long_clusters:
            heatmap.nearest_long_liq = heatmap.long_clusters[0].price_level
        if heatmap.short_clusters:
            heatmap.nearest_short_liq = heatmap.short_clusters[0].price_level

        # OI within 5% of current price
        heatmap.long_liq_density_5pct = sum(
            c.estimated_oi_usd for c in heatmap.long_clusters
            if c.distance_from_current_pct <= 0.05
        )
        heatmap.short_liq_density_5pct = sum(
            c.estimated_oi_usd for c in heatmap.short_clusters
            if c.distance_from_current_pct <= 0.05
        )

        logger.debug(
            "Liquidation heatmap for {asset}: nearest_long_liq={nll}, nearest_short_liq={nsl}, "
            "long_5pct_density=${ld}, short_5pct_density=${sd}",
            asset=asset,
            nll=heatmap.nearest_long_liq,
            nsl=heatmap.nearest_short_liq,
            ld=heatmap.long_liq_density_5pct,
            sd=heatmap.short_liq_density_5pct,
        )

        return heatmap

    def format_for_context(self, heatmap: LiquidationHeatmap) -> str:
        """Format heatmap data as text lines for the context prompt.

        Returns a multi-line string showing:
        - Nearest long liquidation cluster
        - Nearest short liquidation cluster
        - OI density within 5% in each direction
        - Top 3 clusters in each direction
        """
        if not heatmap.long_clusters and not heatmap.short_clusters:
            return "- **Liquidation Heatmap**: No data available"

        lines = [
            f"- **Liquidation Heatmap** (estimated from OI=${heatmap.total_oi_usd:,.0f}):",
        ]

        if heatmap.long_clusters:
            lines.append(
                f"  - Nearest long liq cluster: ${heatmap.nearest_long_liq:,.2f} "
                f"({heatmap.long_clusters[0].distance_from_current_pct:.1%} below)"
            )
        if heatmap.short_clusters:
            lines.append(
                f"  - Nearest short liq cluster: ${heatmap.nearest_short_liq:,.2f} "
                f"({heatmap.short_clusters[0].distance_from_current_pct:.1%} above)"
            )

        lines.append(
            f"  - Long liq OI within 5% drop: ${heatmap.long_liq_density_5pct:,.0f}"
        )
        lines.append(
            f"  - Short liq OI within 5% rise: ${heatmap.short_liq_density_5pct:,.0f}"
        )

        # Top 3 each side
        if heatmap.long_clusters[:3]:
            lines.append("  - Key long liq levels: " + ", ".join(
                f"${c.price_level:,.0f} ({c.leverage}x, ${c.estimated_oi_usd:,.0f})"
                for c in heatmap.long_clusters[:3]
            ))
        if heatmap.short_clusters[:3]:
            lines.append("  - Key short liq levels: " + ", ".join(
                f"${c.price_level:,.0f} ({c.leverage}x, ${c.estimated_oi_usd:,.0f})"
                for c in heatmap.short_clusters[:3]
            ))

        return "\n".join(lines)
