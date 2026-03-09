"""
Trade performance analytics for the Grok Trader trading bot.

Mines the PostgreSQL trade history to produce actionable insights that
Grok can learn from -- strategy win rates, asset performance, timing
patterns, R:R accuracy, streaks, and sizing efficiency.

All methods are read-only (SELECT queries only).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from utils.logger import logger

from data.database import fetch_all, fetch_one


class TradePerformanceAnalyzer:
    """Analyzes closed trade history and produces learning insights for Grok.

    Usage::

        analyzer = TradePerformanceAnalyzer()
        db = get_db_connection()
        summary = analyzer.generate_performance_summary(db)
    """

    def __init__(self) -> None:
        pass

    # ══════════════════════════════════════════════════════════════════
    # CORE ANALYSIS METHODS
    # ══════════════════════════════════════════════════════════════════

    def get_strategy_performance(
        self,
        db: Any,
        lookback_days: int = 30,
    ) -> dict[str, Any]:
        """Group closed trades by action type and compute per-strategy metrics.

        Uses ``open_long`` vs ``open_short`` as a proxy for directional
        strategy classification.

        Returns:
            Dict with keys ``long`` and ``short``, each containing
            ``win_rate``, ``avg_pnl``, ``avg_pnl_pct``, ``count``,
            ``total_pnl``, plus a ``best_strategy`` key.
        """
        cutoff = self._lookback_cutoff(lookback_days)
        rows = fetch_all(
            db,
            """
            SELECT side, pnl, pnl_pct
            FROM trades
            WHERE status = 'closed'
              AND closed_at >= ?
              AND pnl IS NOT NULL
            ORDER BY closed_at ASC
            """,
            (cutoff,),
        )

        strategies: dict[str, list[dict]] = {"long": [], "short": []}
        for row in rows:
            side = row["side"] if row["side"] in ("long", "short") else "long"
            strategies[side].append({
                "pnl": float(row["pnl"]),
                "pnl_pct": float(row["pnl_pct"]) if row["pnl_pct"] is not None else 0.0,
            })

        result: dict[str, Any] = {}
        best_strategy = None
        best_win_rate = -1.0

        for strategy, trades in strategies.items():
            stats = self._compute_trade_stats(trades)
            result[strategy] = stats
            if stats["count"] > 0 and stats["win_rate"] > best_win_rate:
                best_win_rate = stats["win_rate"]
                best_strategy = strategy

        result["best_strategy"] = best_strategy
        return result

    def get_asset_performance(
        self,
        db: Any,
        lookback_days: int = 30,
    ) -> dict[str, Any]:
        """Per-asset breakdown of win rate, avg PnL, hold duration, and count.

        Returns:
            Dict keyed by asset symbol (BTC, ETH, SOL) with stats, plus
            ``best_asset`` and ``worst_asset`` keys.
        """
        cutoff = self._lookback_cutoff(lookback_days)
        rows = fetch_all(
            db,
            """
            SELECT asset, pnl, pnl_pct, opened_at, closed_at
            FROM trades
            WHERE status = 'closed'
              AND closed_at >= ?
              AND pnl IS NOT NULL
            ORDER BY closed_at ASC
            """,
            (cutoff,),
        )

        assets: dict[str, list[dict]] = {}
        for row in rows:
            asset = row["asset"]
            if asset not in assets:
                assets[asset] = []

            hold_hours = self._calc_hold_hours(row["opened_at"], row["closed_at"])
            assets[asset].append({
                "pnl": float(row["pnl"]),
                "pnl_pct": float(row["pnl_pct"]) if row["pnl_pct"] is not None else 0.0,
                "hold_hours": hold_hours,
            })

        result: dict[str, Any] = {}
        best_asset = None
        worst_asset = None
        best_wr = -1.0
        worst_wr = 2.0

        for asset, trades in assets.items():
            stats = self._compute_trade_stats(trades)
            hold_hours_list = [t["hold_hours"] for t in trades if t["hold_hours"] is not None]
            stats["avg_hold_hours"] = (
                round(sum(hold_hours_list) / len(hold_hours_list), 1)
                if hold_hours_list
                else 0.0
            )
            result[asset] = stats

            if stats["count"] >= 2:
                if stats["win_rate"] > best_wr:
                    best_wr = stats["win_rate"]
                    best_asset = asset
                if stats["win_rate"] < worst_wr:
                    worst_wr = stats["win_rate"]
                    worst_asset = asset

        result["best_asset"] = best_asset
        result["worst_asset"] = worst_asset
        return result

    def get_time_performance(
        self,
        db: Any,
        lookback_days: int = 30,
    ) -> dict[str, Any]:
        """Analyze trade performance by hour-of-day and day-of-week (UTC).

        Returns:
            Dict with ``by_hour`` (dict[int, stats]) and ``by_day``
            (dict[int, stats]) plus ``best_hour``, ``worst_hour``,
            ``best_day``, ``worst_day``.
        """
        cutoff = self._lookback_cutoff(lookback_days)
        rows = fetch_all(
            db,
            """
            SELECT opened_at, pnl, pnl_pct
            FROM trades
            WHERE status = 'closed'
              AND closed_at >= ?
              AND pnl IS NOT NULL
            ORDER BY closed_at ASC
            """,
            (cutoff,),
        )

        by_hour: dict[int, list[dict]] = {}
        by_day: dict[int, list[dict]] = {}

        for row in rows:
            dt = self._parse_iso(row["opened_at"])
            if dt is None:
                continue
            trade = {
                "pnl": float(row["pnl"]),
                "pnl_pct": float(row["pnl_pct"]) if row["pnl_pct"] is not None else 0.0,
            }
            hour = dt.hour
            day = dt.weekday()  # 0=Monday, 6=Sunday
            by_hour.setdefault(hour, []).append(trade)
            by_day.setdefault(day, []).append(trade)

        result: dict[str, Any] = {"by_hour": {}, "by_day": {}}

        # Process hours
        best_hour, worst_hour = None, None
        best_hour_wr, worst_hour_wr = -1.0, 2.0
        for hour, trades in sorted(by_hour.items()):
            stats = self._compute_trade_stats(trades)
            result["by_hour"][hour] = stats
            if stats["count"] >= 2:
                if stats["win_rate"] > best_hour_wr:
                    best_hour_wr = stats["win_rate"]
                    best_hour = hour
                if stats["win_rate"] < worst_hour_wr:
                    worst_hour_wr = stats["win_rate"]
                    worst_hour = hour

        # Process days
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        best_day, worst_day = None, None
        best_day_wr, worst_day_wr = -1.0, 2.0
        for day, trades in sorted(by_day.items()):
            stats = self._compute_trade_stats(trades)
            result["by_day"][day] = stats
            if stats["count"] >= 2:
                if stats["win_rate"] > best_day_wr:
                    best_day_wr = stats["win_rate"]
                    best_day = day_names[day]
                if stats["win_rate"] < worst_day_wr:
                    worst_day_wr = stats["win_rate"]
                    worst_day = day_names[day]

        result["best_hour"] = best_hour
        result["worst_hour"] = worst_hour
        result["best_day"] = best_day
        result["worst_day"] = worst_day
        return result

    def get_rr_accuracy(
        self,
        db: Any,
        lookback_days: int = 30,
    ) -> dict[str, Any]:
        """Compare predicted R:R (from SL/TP at entry) vs actual R:R achieved.

        Returns:
            Dict with ``avg_predicted_rr``, ``avg_actual_rr``,
            ``rr_accuracy_ratio``, ``bias`` (over-optimistic / under-optimistic
            / accurate), and ``count``.
        """
        cutoff = self._lookback_cutoff(lookback_days)
        rows = fetch_all(
            db,
            """
            SELECT side, entry_price, exit_price, stop_loss, take_profit
            FROM trades
            WHERE status = 'closed'
              AND closed_at >= ?
              AND pnl IS NOT NULL
              AND entry_price IS NOT NULL
              AND exit_price IS NOT NULL
              AND stop_loss IS NOT NULL
              AND take_profit IS NOT NULL
              AND entry_price > 0
              AND stop_loss > 0
              AND take_profit > 0
            """,
            (cutoff,),
        )

        predicted_rrs: list[float] = []
        actual_rrs: list[float] = []

        for row in rows:
            entry = float(row["entry_price"])
            exit_p = float(row["exit_price"])
            sl = float(row["stop_loss"])
            tp = float(row["take_profit"])
            side = row["side"]

            # Predicted R:R from planned SL/TP
            if side == "long":
                risk = abs(entry - sl)
                reward = abs(tp - entry)
                actual_move = exit_p - entry
            else:
                risk = abs(sl - entry)
                reward = abs(entry - tp)
                actual_move = entry - exit_p

            if risk == 0:
                continue

            predicted_rr = reward / risk
            actual_rr = actual_move / risk  # Can be negative for losses

            predicted_rrs.append(predicted_rr)
            actual_rrs.append(actual_rr)

        if not predicted_rrs:
            return {
                "avg_predicted_rr": 0.0,
                "avg_actual_rr": 0.0,
                "rr_accuracy_ratio": 0.0,
                "bias": "insufficient_data",
                "count": 0,
            }

        avg_predicted = sum(predicted_rrs) / len(predicted_rrs)
        avg_actual = sum(actual_rrs) / len(actual_rrs)
        accuracy_ratio = avg_actual / avg_predicted if avg_predicted != 0 else 0.0

        if accuracy_ratio > 0.9:
            bias = "accurate"
        elif accuracy_ratio > 0:
            bias = "over-optimistic"
        else:
            bias = "significantly_over-optimistic"

        return {
            "avg_predicted_rr": round(avg_predicted, 2),
            "avg_actual_rr": round(avg_actual, 2),
            "rr_accuracy_ratio": round(accuracy_ratio, 2),
            "bias": bias,
            "count": len(predicted_rrs),
        }

    def get_streak_analysis(self, db: Any) -> dict[str, Any]:
        """Analyze win/loss streaks across all trade history.

        Returns:
            Dict with ``current_streak`` (positive = wins, negative = losses),
            ``longest_win_streak``, ``longest_loss_streak``,
            ``avg_pnl_after_consecutive_losses``, ``total_trades``.
        """
        rows = fetch_all(
            db,
            """
            SELECT pnl
            FROM trades
            WHERE status = 'closed'
              AND pnl IS NOT NULL
            ORDER BY closed_at ASC
            """,
        )

        if not rows:
            return {
                "current_streak": 0,
                "longest_win_streak": 0,
                "longest_loss_streak": 0,
                "avg_pnl_after_consecutive_losses": 0.0,
                "total_trades": 0,
            }

        pnls = [float(r["pnl"]) for r in rows]

        # Current streak
        current_streak = 0
        for pnl in reversed(pnls):
            if pnl > 0:
                if current_streak < 0:
                    break
                current_streak += 1
            elif pnl < 0:
                if current_streak > 0:
                    break
                current_streak -= 1
            else:
                break  # breakeven, stop streak

        # Longest streaks
        longest_win = 0
        longest_loss = 0
        current_win_run = 0
        current_loss_run = 0

        for pnl in pnls:
            if pnl > 0:
                current_win_run += 1
                current_loss_run = 0
                longest_win = max(longest_win, current_win_run)
            elif pnl < 0:
                current_loss_run += 1
                current_win_run = 0
                longest_loss = max(longest_loss, current_loss_run)
            else:
                current_win_run = 0
                current_loss_run = 0

        # Average PnL after 2+ consecutive losses (recovery check)
        recovery_pnls: list[float] = []
        consecutive_losses = 0
        for i, pnl in enumerate(pnls):
            if pnl < 0:
                consecutive_losses += 1
            else:
                if consecutive_losses >= 2 and pnl > 0:
                    recovery_pnls.append(pnl)
                elif consecutive_losses >= 2 and pnl <= 0:
                    recovery_pnls.append(pnl)
                consecutive_losses = 0

        avg_recovery = (
            round(sum(recovery_pnls) / len(recovery_pnls), 2)
            if recovery_pnls
            else 0.0
        )

        return {
            "current_streak": current_streak,
            "longest_win_streak": longest_win,
            "longest_loss_streak": longest_loss,
            "avg_pnl_after_consecutive_losses": avg_recovery,
            "total_trades": len(pnls),
        }

    def get_sizing_analysis(
        self,
        db: Any,
        lookback_days: int = 30,
    ) -> dict[str, Any]:
        """Compare average position size for winners vs losers.

        Good sizing: bigger on winners, smaller on losers.
        Bad sizing: bigger on losers (doubling down into losses).

        Returns:
            Dict with ``avg_winner_size``, ``avg_loser_size``,
            ``sizing_quality`` (good / neutral / bad), ``count``.
        """
        cutoff = self._lookback_cutoff(lookback_days)
        rows = fetch_all(
            db,
            """
            SELECT size_pct, pnl
            FROM trades
            WHERE status = 'closed'
              AND closed_at >= ?
              AND pnl IS NOT NULL
              AND size_pct IS NOT NULL
            """,
            (cutoff,),
        )

        winner_sizes: list[float] = []
        loser_sizes: list[float] = []

        for row in rows:
            pnl = float(row["pnl"])
            size = float(row["size_pct"])
            if pnl > 0:
                winner_sizes.append(size)
            elif pnl < 0:
                loser_sizes.append(size)

        avg_winner = (
            round(sum(winner_sizes) / len(winner_sizes), 4)
            if winner_sizes
            else 0.0
        )
        avg_loser = (
            round(sum(loser_sizes) / len(loser_sizes), 4)
            if loser_sizes
            else 0.0
        )

        # Determine sizing quality
        if not winner_sizes or not loser_sizes:
            quality = "insufficient_data"
        elif avg_winner > avg_loser * 1.1:
            quality = "good"
        elif avg_loser > avg_winner * 1.1:
            quality = "bad"
        else:
            quality = "neutral"

        return {
            "avg_winner_size": avg_winner,
            "avg_loser_size": avg_loser,
            "sizing_quality": quality,
            "winners_count": len(winner_sizes),
            "losers_count": len(loser_sizes),
        }

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY GENERATOR
    # ══════════════════════════════════════════════════════════════════

    def generate_performance_summary(self, db: Any) -> str:
        """Produce a formatted text block of performance insights for Grok.

        Calls all analysis methods and assembles a concise summary
        (under 500 words) ready to inject into the context prompt.

        Args:
            db: An open database connection.

        Returns:
            A formatted string with performance analytics and actionable advice.
        """
        try:
            strategy = self.get_strategy_performance(db)
            asset = self.get_asset_performance(db)
            timing = self.get_time_performance(db)
            rr = self.get_rr_accuracy(db)
            streaks = self.get_streak_analysis(db)
            sizing = self.get_sizing_analysis(db)

            # Check if we have enough data
            total_trades = streaks.get("total_trades", 0)
            if total_trades == 0:
                return "No closed trades yet. Analytics will appear after first trades are completed."

            lines: list[str] = []

            # --- Strategy Performance ---
            lines.append("**Strategy Performance (30d):**")
            for side in ("long", "short"):
                s = strategy.get(side, {})
                if s.get("count", 0) > 0:
                    lines.append(
                        f"- {side.upper()}: {s['count']} trades, "
                        f"{s['win_rate']:.0%} win rate, "
                        f"avg PnL ${s['avg_pnl']:.2f}, "
                        f"total ${s['total_pnl']:.2f}"
                    )
            if strategy.get("best_strategy"):
                lines.append(f"- Best strategy: {strategy['best_strategy'].upper()}")

            # --- Asset Performance ---
            lines.append("")
            lines.append("**Asset Performance (30d):**")
            for symbol in sorted(asset.keys()):
                if symbol in ("best_asset", "worst_asset"):
                    continue
                a = asset[symbol]
                if a.get("count", 0) > 0:
                    lines.append(
                        f"- {symbol}: {a['count']} trades, "
                        f"{a['win_rate']:.0%} win rate, "
                        f"avg PnL ${a['avg_pnl']:.2f}, "
                        f"avg hold {a.get('avg_hold_hours', 0):.1f}h"
                    )
            if asset.get("best_asset"):
                lines.append(f"- Best asset: {asset['best_asset']}")
            if asset.get("worst_asset") and asset["worst_asset"] != asset.get("best_asset"):
                lines.append(f"- Worst asset: {asset['worst_asset']}")

            # --- R:R Accuracy ---
            if rr.get("count", 0) > 0:
                lines.append("")
                lines.append("**R:R Accuracy:**")
                lines.append(
                    f"- Predicted avg R:R: {rr['avg_predicted_rr']:.2f}, "
                    f"Actual avg R:R: {rr['avg_actual_rr']:.2f}"
                )
                lines.append(
                    f"- Accuracy ratio: {rr['rr_accuracy_ratio']:.2f} "
                    f"({rr['bias'].replace('_', ' ')})"
                )

            # --- Streaks ---
            lines.append("")
            lines.append("**Streak Status:**")
            cs = streaks["current_streak"]
            if cs > 0:
                lines.append(f"- Current: {cs} consecutive WIN(s)")
            elif cs < 0:
                lines.append(f"- Current: {abs(cs)} consecutive LOSS(es)")
            else:
                lines.append("- Current: No active streak")
            lines.append(
                f"- Historical: longest win streak {streaks['longest_win_streak']}, "
                f"longest loss streak {streaks['longest_loss_streak']}"
            )

            # --- Sizing ---
            if sizing.get("winners_count", 0) > 0 or sizing.get("losers_count", 0) > 0:
                lines.append("")
                lines.append("**Sizing Analysis:**")
                lines.append(
                    f"- Avg winner size: {sizing['avg_winner_size']:.2%}, "
                    f"avg loser size: {sizing['avg_loser_size']:.2%}"
                )
                q = sizing["sizing_quality"]
                if q == "good":
                    lines.append("- Sizing is GOOD: larger on winners, smaller on losers")
                elif q == "bad":
                    lines.append(
                        "- Sizing is POOR: larger on losers than winners. "
                        "REDUCE position sizes on lower-conviction setups."
                    )
                elif q == "neutral":
                    lines.append("- Sizing is neutral: similar sizes for winners and losers")

            # --- Actionable Advice ---
            advice = self._generate_advice(strategy, asset, rr, streaks, sizing)
            if advice:
                lines.append("")
                lines.append("**Actionable Insights:**")
                for tip in advice:
                    lines.append(f"- {tip}")

            summary = "\n".join(lines)

            # Safety: truncate if somehow over 500 words
            words = summary.split()
            if len(words) > 500:
                summary = " ".join(words[:490]) + "\n[...truncated for context window]"

            logger.debug(
                "Performance summary generated | {n} trades analyzed, {w} words",
                n=total_trades,
                w=len(summary.split()),
            )
            return summary

        except Exception as exc:
            logger.error("Failed to generate performance summary: {err}", err=exc)
            return "Performance analytics unavailable due to an error."

    # ══════════════════════════════════════════════════════════════════
    # PRIVATE HELPERS
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _lookback_cutoff(days: int) -> str:
        """Return ISO timestamp for N days ago."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return cutoff.isoformat()

    @staticmethod
    def _parse_iso(ts: Any) -> datetime | None:
        """Parse a timestamp (ISO string or datetime object), returning None on failure."""
        if not ts:
            return None
        try:
            if isinstance(ts, datetime):
                dt = ts
            else:
                dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _calc_hold_hours(opened_at: Any, closed_at: Any) -> float | None:
        """Calculate hold duration in hours between two timestamps."""
        if not opened_at or not closed_at:
            return None
        try:
            open_dt = opened_at if isinstance(opened_at, datetime) else datetime.fromisoformat(opened_at)
            close_dt = closed_at if isinstance(closed_at, datetime) else datetime.fromisoformat(closed_at)
            if open_dt.tzinfo is None:
                open_dt = open_dt.replace(tzinfo=timezone.utc)
            if close_dt.tzinfo is None:
                close_dt = close_dt.replace(tzinfo=timezone.utc)
            delta = close_dt - open_dt
            return round(delta.total_seconds() / 3600, 1)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _compute_trade_stats(trades: list[dict]) -> dict[str, Any]:
        """Compute common stats (win_rate, avg_pnl, etc.) from a list of trades."""
        if not trades:
            return {
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "avg_pnl_pct": 0.0,
                "total_pnl": 0.0,
                "count": 0,
            }

        pnls = [t["pnl"] for t in trades]
        pnl_pcts = [t.get("pnl_pct", 0.0) for t in trades]
        wins = sum(1 for p in pnls if p > 0)
        non_breakeven = sum(1 for p in pnls if p != 0)

        return {
            "win_rate": wins / non_breakeven if non_breakeven > 0 else 0.0,
            "avg_pnl": round(sum(pnls) / len(pnls), 2),
            "avg_pnl_pct": round(sum(pnl_pcts) / len(pnl_pcts), 4),
            "total_pnl": round(sum(pnls), 2),
            "count": len(pnls),
        }

    @staticmethod
    def _generate_advice(
        strategy: dict,
        asset: dict,
        rr: dict,
        streaks: dict,
        sizing: dict,
    ) -> list[str]:
        """Generate specific actionable advice from the analytics."""
        advice: list[str] = []

        # Strategy-based advice
        for side in ("long", "short"):
            s = strategy.get(side, {})
            other = "short" if side == "long" else "long"
            other_s = strategy.get(other, {})
            if (
                s.get("count", 0) >= 3
                and other_s.get("count", 0) >= 3
                and s.get("win_rate", 0) < 0.35
                and other_s.get("win_rate", 0) > 0.55
            ):
                advice.append(
                    f"Your {side} trades have only {s['win_rate']:.0%} win rate vs "
                    f"{other_s['win_rate']:.0%} for {other}s. "
                    f"Consider reducing {side} exposure."
                )

        # Asset-specific directional advice
        for symbol in sorted(asset.keys()):
            if symbol in ("best_asset", "worst_asset"):
                continue
            a = asset[symbol]
            if a.get("count", 0) >= 3 and a.get("win_rate", 0) < 0.30:
                advice.append(
                    f"{symbol} has only {a['win_rate']:.0%} win rate over "
                    f"{a['count']} trades. Consider avoiding {symbol} or "
                    f"reducing size."
                )

        # R:R advice
        if rr.get("count", 0) >= 3 and rr.get("bias") == "over-optimistic":
            advice.append(
                f"R:R predictions are over-optimistic (accuracy ratio "
                f"{rr['rr_accuracy_ratio']:.2f}). Consider tightening "
                f"take-profit targets."
            )

        # Streak-based advice
        if streaks.get("current_streak", 0) <= -3:
            advice.append(
                f"Currently on a {abs(streaks['current_streak'])}-trade "
                f"losing streak. Consider reducing position sizes until "
                f"the streak breaks."
            )

        # Sizing advice
        if sizing.get("sizing_quality") == "bad":
            advice.append(
                "Position sizing is inverted: larger on losers than winners. "
                "Only size up on HIGH conviction setups."
            )

        return advice
