"""
Grok Trading Bot — Main Orchestrator

Entry point for the autonomous perpetual futures trading bot.
Runs a scheduled loop that:
  1. Gathers market data and portfolio state
  2. Queries Grok for trading decisions
  3. Validates decisions through the Risk Guardian
  4. Executes approved trades on Hyperliquid
  5. Logs everything for auditability

Usage:
    python main.py
"""

import json as _json
import sys
import signal
import time
from datetime import datetime, timezone

from loguru import logger

from config.trading_config import (
    XAI_API_KEY,
    HYPERLIQUID_PRIVATE_KEY,
    HYPERLIQUID_WALLET_ADDRESS,
    LIVE_TRADING,
    CYCLE_INTERVAL_MINUTES,
    MIN_CYCLE_INTERVAL_MINUTES,
    MAX_CYCLE_INTERVAL_MINUTES,
    ASSET_UNIVERSE,
    GROK_MODEL,
    DISCORD_WEBHOOK_URL,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    DAILY_SUMMARY_HOUR_UTC,
    X_SENTIMENT_ENABLED,
)
from config.risk_config import RISK_PARAMS
from utils.logger import setup_logger
from data.database import init_db, get_db_connection
from data.market_data import MarketDataFetcher
from data.portfolio_state import PortfolioManager
from data.trade_history import TradeHistoryManager
from data.context_builder import build_context_prompt
from data.performance_analyzer import TradePerformanceAnalyzer
from data.regime_detector import RegimeDetector
from data.liquidation_estimator import LiquidationEstimator
from brain.grok_client import GrokClient
from brain.system_prompt import get_system_prompt
from brain.decision_parser import DecisionParser
from execution.risk_guardian import RiskGuardian
from execution.order_manager import OrderManager
from execution.position_manager import PositionManager
from execution.notifications import Notifier


# Global flag for graceful shutdown
_shutdown_requested = False


def signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM for graceful shutdown."""
    global _shutdown_requested
    logger.warning(f"Shutdown signal received ({signum}). Finishing current cycle...")
    _shutdown_requested = True


def validate_environment() -> bool:
    """Check that all required configuration is present before starting.

    In paper mode, only XAI_API_KEY is strictly required — the Hyperliquid
    SDK keys are only needed for live trading.
    """
    from config.trading_config import DATABASE_URL

    errors = []

    if not XAI_API_KEY:
        errors.append("XAI_API_KEY is not set")

    if not DATABASE_URL:
        errors.append("DATABASE_URL is not set (required for Supabase connection)")

    # Hyperliquid keys are only required in live mode
    if LIVE_TRADING:
        if not HYPERLIQUID_PRIVATE_KEY:
            errors.append("HYPERLIQUID_PRIVATE_KEY is not set (required for LIVE mode)")
        if not HYPERLIQUID_WALLET_ADDRESS:
            errors.append("HYPERLIQUID_WALLET_ADDRESS is not set (required for LIVE mode)")

    if errors:
        for err in errors:
            logger.error(f"Configuration error: {err}")
        return False

    return True


def run_cycle(
    cycle_number: int,
    grok: GrokClient,
    market_fetcher: MarketDataFetcher,
    portfolio_mgr: PortfolioManager,
    trade_history: TradeHistoryManager,
    risk_guardian: RiskGuardian,
    order_mgr: OrderManager,
    position_mgr: PositionManager,
    decision_parser: DecisionParser,
    notifier: Notifier,
    sentiment_fetcher: "XSentimentFetcher | None" = None,
) -> int:
    """
    Execute one full trading cycle.

    Returns the suggested minutes until next cycle.
    """
    db = get_db_connection()
    logger.info(f"=== CYCLE {cycle_number} START ===")
    _log_cycle_event(db, cycle_number, "cycle_start", f"Cycle {cycle_number} started")

    # Record heartbeat so the monitor knows we're alive
    notifier.record_heartbeat()

    try:
        # --- Step 1: Check kill switch ---
        if risk_guardian.kill_switch_active():
            logger.warning("Kill switch is ACTIVE. Skipping cycle.")
            return 5  # Check again in 5 minutes

        # --- Step 2: Gather market data ---
        logger.info("Fetching market data...")
        market_data = market_fetcher.fetch_all_market_data(ASSET_UNIVERSE)
        if not market_data:
            logger.error("Failed to fetch market data. Skipping cycle.")
            return CYCLE_INTERVAL_MINUTES

        # --- Step 3: Gather portfolio state ---
        logger.info("Fetching portfolio state...")
        portfolio = portfolio_mgr.fetch_portfolio_from_exchange(order_mgr.info)

        # Guard: skip cycle if equity is suspiciously low
        from config.trading_config import STARTING_CAPITAL
        equity_floor = STARTING_CAPITAL * 0.01  # 1% of starting capital
        if portfolio.get("total_equity", 0) < equity_floor:
            logger.warning(
                "Equity ${eq:.2f} is below safety floor ${floor:.2f} "
                "(1% of STARTING_CAPITAL). Skipping cycle to prevent "
                "corrupt position sizing.",
                eq=portfolio.get("total_equity", 0),
                floor=equity_floor,
            )
            _log_cycle_event(db, cycle_number, "equity_guard",
                f"Cycle skipped: equity ${portfolio.get('total_equity', 0):.2f} "
                f"below floor ${equity_floor:.2f}",
                severity="warn")
            return CYCLE_INTERVAL_MINUTES

        # Enrich portfolio with DB-derived metrics for risk checks
        portfolio["peak_equity"] = portfolio_mgr.get_peak_equity(db)
        portfolio["daily_pnl_pct"] = (
            portfolio_mgr.get_daily_pnl(db) / portfolio.get("total_equity", 1)
            if portfolio.get("total_equity", 0) > 0 else 0.0
        )
        portfolio["weekly_pnl_pct"] = (
            portfolio_mgr.get_weekly_pnl(db) / portfolio.get("total_equity", 1)
            if portfolio.get("total_equity", 0) > 0 else 0.0
        )
        portfolio["total_exposure_pct"] = position_mgr.get_total_exposure(db)
        portfolio["consecutive_losses"] = portfolio_mgr.get_consecutive_losses(db)
        portfolio["equity"] = portfolio.get("total_equity", 0)

        # --- Step 4: Get recent trades and risk status ---
        recent_trades = trade_history.get_recent_trades(db, limit=10)
        risk_status = risk_guardian.calculate_risk_status(portfolio, db)

        # --- Step 4b: Generate performance analytics ---
        performance_summary = ""
        try:
            analyzer = TradePerformanceAnalyzer()
            performance_summary = analyzer.generate_performance_summary(db)
        except Exception as e:
            logger.warning(f"Performance analytics generation failed: {e}")

        # --- Step 4c: Detect market regimes ---
        regime_data = {}
        try:
            detector = RegimeDetector()
            for asset, data in market_data.items():
                candles_1h = data.get("candles", {}).get("1h")
                candles_4h = data.get("candles", {}).get("4h")
                regime_data[asset] = detector.detect(asset, candles_1h, candles_4h)
        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")

        # --- Step 4d: Estimate liquidation clusters ---
        liquidation_data = {}
        try:
            estimator = LiquidationEstimator()
            for asset, data in market_data.items():
                price = data.get("price", 0)
                oi = data.get("oi", {}).get("current_oi", 0)
                if price > 0 and oi > 0:
                    liquidation_data[asset] = estimator.estimate(asset, price, oi)
        except Exception as e:
            logger.warning(f"Liquidation estimation failed: {e}")

        # --- Step 4e: Fetch X sentiment ---
        sentiment_data = {}
        if sentiment_fetcher is not None:
            try:
                sentiment_data = sentiment_fetcher.fetch_sentiment(
                    list(market_data.keys())
                )
                logger.info(f"X sentiment fetched for {len(sentiment_data)} assets")
                _log_cycle_event(db, cycle_number, "sentiment_fetched",
                    f"Sentiment fetched for {len(sentiment_data)} assets",
                    details={a: {"score": s.score, "momentum": s.momentum} for a, s in sentiment_data.items()})
            except Exception as e:
                logger.warning(f"X sentiment fetch failed: {e}")

        # --- Step 5: Build context prompt ---
        context = build_context_prompt(
            market_data=market_data,
            portfolio=portfolio,
            recent_trades=recent_trades,
            risk_status=risk_status,
            performance_summary=performance_summary,
            liquidation_data=liquidation_data,
            regime_data=regime_data,
            sentiment_data=sentiment_data,
        )

        # --- Step 6: Query Grok ---
        logger.info("Querying Grok for trading decision...")
        system_prompt = get_system_prompt()
        raw_response = grok.get_trading_decision(
            system_prompt=system_prompt,
            context=context,
        )

        if raw_response is None:
            logger.error("Grok returned no response. Skipping cycle.")
            return CYCLE_INTERVAL_MINUTES

        # --- Step 7: Parse Grok's response ---
        grok_response = decision_parser.parse_response(raw_response)
        if grok_response is None:
            logger.error("Failed to parse Grok response. Skipping cycle.")
            notifier.send_error_alert("Grok response parsing failed", severity="warning")
            return CYCLE_INTERVAL_MINUTES

        # --- Step 8: Log the full interaction ---
        _log_grok_interaction(db, system_prompt, context, raw_response, grok_response, cycle_number)

        logger.info(f"Grok stance: {grok_response.overall_stance}")
        logger.info(f"Decisions count: {len(grok_response.decisions)}")
        _log_cycle_event(db, cycle_number, "grok_decision",
            f"Grok: {grok_response.overall_stance} | {len(grok_response.decisions)} decisions",
            details={"stance": grok_response.overall_stance, "decision_count": len(grok_response.decisions)})

        # --- Step 9: Extract actionable decisions ---
        actionable = decision_parser.extract_decisions(grok_response)

        if not actionable:
            logger.info("No actionable trades this cycle. Staying flat.")
        else:
            # --- Step 10: Validate and execute each decision ---
            for decision in actionable:
                logger.info(
                    f"Evaluating: {decision.action} {decision.asset} "
                    f"size={decision.size_pct:.1%} leverage={decision.leverage}x"
                )

                validation = risk_guardian.validate(decision, portfolio, db)

                if validation.approved:
                    logger.info(f"APPROVED: {decision.action} {decision.asset}")
                    order_result = order_mgr.place_order(
                        decision,
                        portfolio_equity=portfolio.get("total_equity", 0.0),
                    )

                    if order_result and order_result.get("status") != "error":
                        # Determine side: for close actions, extract from order result
                        if decision.action == "close":
                            raw_side = order_result.get("side", "")
                            trade_side = raw_side.replace("close_", "") or "long"
                        else:
                            trade_side = "long" if "long" in decision.action else "short"

                        # Build trade_data dict combining decision + order result
                        trade_data = {
                            "asset": decision.asset,
                            "side": trade_side,
                            "action": decision.action,
                            "size_pct": decision.size_pct,
                            "leverage": decision.leverage,
                            "entry_price": order_result.get("fill_price", decision.entry_price),
                            "stop_loss": decision.stop_loss,
                            "take_profit": decision.take_profit,
                            "reasoning": decision.reasoning,
                            "conviction": decision.conviction,
                            "fees": order_result.get("fees", 0.0),
                        }
                        trade_history.log_trade(db, trade_data)
                        notifier.send_trade_alert(decision, order_result, portfolio=portfolio)
                        logger.info(f"Trade executed: {order_result}")
                        _log_cycle_event(db, cycle_number, "trade_opened",
                            f"{decision.action} {decision.asset} @ ${order_result.get('fill_price', 0):,.2f}",
                            severity="success", asset=decision.asset,
                            details={"action": decision.action, "size_pct": decision.size_pct, "leverage": decision.leverage})
                    else:
                        logger.error(f"Order placement failed for {decision.asset}")
                        notifier.send_error_alert(
                            f"Order placement failed: {decision.action} {decision.asset}",
                            severity="critical",
                        )
                else:
                    logger.warning(f"REJECTED: {decision.action} {decision.asset} — {validation.reason}")
                    _log_rejection(db, decision, validation.reason)
                    _log_cycle_event(db, cycle_number, "trade_rejected",
                        f"Rejected {decision.action} {decision.asset}: {validation.reason}",
                        severity="warn", asset=decision.asset,
                        details={"action": decision.action, "reason": validation.reason})

        # --- Step 11: Manage existing positions ---
        position_mgr.manage_open_positions(db)

        # --- Step 11b: Save equity snapshot for dashboard chart ---
        _save_equity_snapshot(db, cycle_number, portfolio)

        # --- Step 11c: Save market + regime + sentiment snapshots for dashboard ---
        _save_market_snapshots(db, cycle_number, market_data, regime_data, sentiment_data)
        _save_performance_cache(db, cycle_number)

        # --- Step 12: Daily summary ---
        now = datetime.now(timezone.utc)
        if now.hour == DAILY_SUMMARY_HOUR_UTC and now.minute < CYCLE_INTERVAL_MINUTES:
            summary = _generate_daily_summary(db, portfolio)
            notifier.send_daily_summary(summary)
            # Cleanup old dashboard data (keep 7 days)
            try:
                db.execute("DELETE FROM cycle_events WHERE timestamp < NOW() - INTERVAL '7 days'")
                db.execute("DELETE FROM market_snapshots WHERE timestamp < NOW() - INTERVAL '7 days'")
                db.execute("DELETE FROM performance_cache WHERE timestamp < NOW() - INTERVAL '48 hours'")
                db.commit()
            except Exception:
                pass  # Non-critical cleanup

        # --- Step 13: Determine next cycle timing ---
        suggested = grok_response.next_review_suggestion_minutes
        next_cycle = max(min(suggested, MAX_CYCLE_INTERVAL_MINUTES), MIN_CYCLE_INTERVAL_MINUTES)
        if suggested != next_cycle:
            logger.info(f"Grok suggested {suggested} min, clamped to {next_cycle} min")
        logger.info(f"=== CYCLE {cycle_number} COMPLETE === Next in {next_cycle} min")
        _log_cycle_event(db, cycle_number, "cycle_end",
            f"Cycle {cycle_number} complete. Next in {next_cycle} min",
            details={"next_cycle_minutes": next_cycle})
        return next_cycle

    except Exception as e:
        logger.exception(f"Cycle {cycle_number} error: {e}")
        notifier.send_error_alert(f"Cycle {cycle_number} error: {e}", severity="critical")
        return CYCLE_INTERVAL_MINUTES
    finally:
        db.close()


def _log_grok_interaction(db, system_prompt, context, raw_response, parsed_response, cycle_number):
    """Log the full Grok prompt/response pair for auditability."""
    import hashlib
    import json

    try:
        prompt_hash = hashlib.sha256(system_prompt.encode()).hexdigest()[:16]
        decisions_json = json.dumps(
            [d.model_dump() for d in parsed_response.decisions]
        )

        db.execute(
            """INSERT INTO grok_logs
               (timestamp, system_prompt_hash, context_prompt, response_text,
                decisions_json, cycle_number)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                prompt_hash,
                context,
                raw_response,
                decisions_json,
                cycle_number,
            ),
        )
        db.commit()
    except Exception as e:
        try:
            db.rollback()
        except Exception:
            pass
        logger.error(f"Failed to log Grok interaction: {e}")


def _log_rejection(db, decision, reason):
    """Log a rejected trade decision."""
    import json

    try:
        db.execute(
            """INSERT INTO rejections
               (timestamp, asset, action, reason, decision_json)
               VALUES (?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                decision.asset,
                decision.action,
                reason,
                json.dumps(decision.model_dump()),
            ),
        )
        db.commit()
    except Exception as e:
        try:
            db.rollback()
        except Exception:
            pass
        logger.error(f"Failed to log rejection: {e}")


def _save_equity_snapshot(db, cycle_number, portfolio):
    """Save a per-cycle equity snapshot for the dashboard chart.

    Args:
        db: Open PostgreSQL connection.
        cycle_number: Current cycle number.
        portfolio: Portfolio state dict with keys total_equity,
            unrealized_pnl, positions, margin_used.
    """
    try:
        from config.trading_config import STARTING_CAPITAL

        equity = portfolio.get("total_equity", STARTING_CAPITAL)
        unrealized = portfolio.get("unrealized_pnl", 0.0)

        # Realized P&L from closed trades
        realized_row = db.execute(
            "SELECT COALESCE(SUM(pnl), 0) AS realized FROM trades WHERE status = 'closed'"
        ).fetchone()
        realized = float(realized_row["realized"]) if realized_row else 0.0

        # Open position count and exposure
        open_count_row = db.execute(
            "SELECT COUNT(*) AS cnt, COALESCE(SUM(size_pct), 0) AS exp FROM trades WHERE status = 'open'"
        ).fetchone()
        open_positions = int(open_count_row["cnt"]) if open_count_row else 0
        total_exposure = float(open_count_row["exp"]) if open_count_row else 0.0

        db.execute(
            """INSERT INTO equity_snapshots
               (timestamp, cycle_number, equity, unrealized_pnl, realized_pnl,
                open_positions, total_exposure)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                cycle_number,
                round(equity, 2),
                round(unrealized, 4),
                round(realized, 4),
                open_positions,
                round(total_exposure, 4),
            ),
        )
        db.commit()
        logger.info(
            "Equity snapshot saved | cycle={c} equity=${eq:.2f} unrealized=${u:.2f}",
            c=cycle_number, eq=equity, u=unrealized,
        )
    except Exception as e:
        try:
            db.rollback()
        except Exception:
            pass
        logger.error(f"Failed to save equity snapshot: {e}")


def _log_cycle_event(db, cycle_number, event_type, summary, severity="info", asset=None, details=None):
    """Log an event to the cycle_events table for the dashboard activity feed."""
    try:
        db.execute(
            """INSERT INTO cycle_events
               (timestamp, cycle_number, event_type, severity, asset, summary, details_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                cycle_number, event_type, severity, asset, summary,
                _json.dumps(details) if details else None,
            ),
        )
        db.commit()
    except Exception as e:
        logger.error(f"Failed to log cycle event: {e}")
        try:
            db.rollback()
        except Exception:
            pass


def _save_market_snapshots(db, cycle_number, market_data, regime_data, sentiment_data):
    """Save per-asset market state (price, regime, sentiment) for the dashboard."""
    try:
        timestamp = datetime.now(timezone.utc).isoformat()
        for asset, data in market_data.items():
            tech = data.get("technicals", {})
            funding = data.get("funding", {})
            oi = data.get("oi", {})
            regime = regime_data.get(asset)
            sentiment = sentiment_data.get(asset) if sentiment_data else None

            # Cast numpy types to plain Python to avoid psycopg2 serialisation issues
            _f = lambda v, default=0: float(v) if v is not None else float(default)

            db.execute(
                """INSERT INTO market_snapshots
                   (timestamp, cycle_number, asset, price, change_24h_pct,
                    funding_rate, open_interest, rsi_14, atr_pct, volatility_regime,
                    market_regime, regime_confidence, adx, choppiness_index,
                    sentiment_score, sentiment_momentum, sentiment_volume, sentiment_topics)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    timestamp, cycle_number, asset,
                    _f(data.get("price", 0)),
                    _f(data.get("24h_change_pct", 0)),
                    _f(funding.get("current_rate", 0)),
                    _f(oi.get("current_oi", 0)),
                    _f(tech.get("rsi_14", 50)),
                    _f(tech.get("atr_pct", 0)),
                    str(tech.get("volatility_regime", "normal")),
                    regime.regime.value if regime else None,
                    float(regime.confidence) if regime and regime.confidence is not None else None,
                    float(regime.adx) if regime and regime.adx is not None else None,
                    float(regime.choppiness_index) if regime and regime.choppiness_index is not None else None,
                    float(sentiment.score) if sentiment and hasattr(sentiment, "score") else None,
                    str(sentiment.momentum) if sentiment and hasattr(sentiment, "momentum") else None,
                    str(sentiment.volume) if sentiment and hasattr(sentiment, "volume") else None,
                    ", ".join(sentiment.key_topics[:3]) if sentiment and hasattr(sentiment, "key_topics") and sentiment.key_topics else None,
                ),
            )
        db.commit()
    except Exception as e:
        logger.error(f"Failed to save market snapshots: {e}")
        try:
            db.rollback()
        except Exception:
            pass


def _save_performance_cache(db, cycle_number):
    """Compute and cache performance analytics for the dashboard."""
    try:
        analyzer = TradePerformanceAnalyzer()
        metrics = {
            "strategy": analyzer.get_strategy_performance(db),
            "asset": analyzer.get_asset_performance(db),
            "streaks": analyzer.get_streak_analysis(db),
        }
        db.execute(
            """INSERT INTO performance_cache (timestamp, cycle_number, metrics_json)
               VALUES (?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                cycle_number,
                _json.dumps(metrics, default=str),
            ),
        )
        db.commit()
    except Exception as e:
        try:
            db.rollback()
        except Exception:
            pass
        logger.error(f"Failed to save performance cache: {e}")


def _generate_daily_summary(db, portfolio) -> dict:
    """Generate an enhanced daily performance summary.

    Returns keys matching what ``Notifier.send_daily_summary()`` expects,
    plus enhanced fields for richer reporting:
    ``avg_rr``, ``best_trade_pnl``, ``worst_trade_pnl``,
    ``current_streak``, ``weekly_equity_change_pct``.
    """
    from data.trade_history import TradeHistoryManager

    th = TradeHistoryManager()
    trades_today_list = th.get_trades_today(db)

    wins = sum(1 for t in trades_today_list if t.get("pnl", 0) > 0)
    losses = sum(1 for t in trades_today_list if t.get("pnl", 0) < 0)
    total_pnl = sum(t.get("pnl", 0) for t in trades_today_list)
    equity = portfolio.get("total_equity", 0)
    daily_pnl_pct = total_pnl / equity if equity > 0 else 0.0

    summary = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "equity": equity,
        "peak_equity": portfolio.get("peak_equity", equity),
        "daily_pnl_pct": daily_pnl_pct,
        "trades_today": len(trades_today_list),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(trades_today_list) if trades_today_list else 0,
        "open_positions": len(portfolio.get("positions", [])),
        "total_exposure_pct": portfolio.get("total_exposure_pct", 0.0),
    }

    # --- Enhanced fields ---
    try:
        # Average R:R for today's closed trades
        rr_values = [
            t.get("risk_reward_ratio", 0)
            for t in trades_today_list
            if t.get("risk_reward_ratio") is not None
        ]
        if rr_values:
            summary["avg_rr"] = sum(rr_values) / len(rr_values)

        # Best and worst trade P&L
        pnl_pcts = [
            t.get("pnl_pct", 0)
            for t in trades_today_list
            if t.get("pnl_pct") is not None
        ]
        if pnl_pcts:
            summary["best_trade_pnl"] = max(pnl_pcts)
            summary["worst_trade_pnl"] = min(pnl_pcts)

        # Current streak from performance analyzer
        analyzer = TradePerformanceAnalyzer()
        streaks = analyzer.get_streak_analysis(db)
        summary["current_streak"] = streaks.get("current_streak", 0)

        # Weekly equity change
        weekly_pnl = portfolio.get("weekly_pnl_pct")
        if weekly_pnl is not None:
            summary["weekly_equity_change_pct"] = weekly_pnl
    except Exception as e:
        logger.warning(f"Enhanced summary fields failed: {e}")

    return summary


def main():
    """Main entry point — initialize components and run the trading loop."""
    # Setup logging
    setup_logger()

    logger.info("=" * 60)
    logger.info("  GROK TRADING BOT — SENTINEL")
    logger.info(f"  Mode: {'LIVE' if LIVE_TRADING else 'PAPER (Testnet)'}")
    logger.info(f"  Assets: {', '.join(ASSET_UNIVERSE)}")
    logger.info(f"  Model: {GROK_MODEL}")
    logger.info(f"  Cycle interval: {CYCLE_INTERVAL_MINUTES} min")
    logger.info("=" * 60)

    if LIVE_TRADING:
        logger.warning("!!! LIVE TRADING MODE — REAL CAPITAL AT RISK !!!")

    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed. Exiting.")
        sys.exit(1)

    # Initialize database
    init_db()
    logger.info("Database initialized.")

    # Initialize components
    grok = GrokClient(api_key=XAI_API_KEY, model=GROK_MODEL)
    market_fetcher = MarketDataFetcher()
    portfolio_mgr = PortfolioManager()
    trade_history = TradeHistoryManager()
    risk_guardian = RiskGuardian(risk_params=RISK_PARAMS)
    order_mgr = OrderManager()
    position_mgr = PositionManager(order_manager=order_mgr)
    decision_parser = DecisionParser()
    notifier = Notifier(
        discord_webhook_url=DISCORD_WEBHOOK_URL,
        telegram_bot_token=TELEGRAM_BOT_TOKEN,
        telegram_chat_id=TELEGRAM_CHAT_ID,
    )

    # Initialize X sentiment fetcher (persists across cycles for caching)
    sentiment_fetcher = None
    if X_SENTIMENT_ENABLED:
        try:
            from data.x_sentiment import XSentimentFetcher
            sentiment_fetcher = XSentimentFetcher(api_key=XAI_API_KEY)
        except Exception as e:
            logger.warning(f"X sentiment initialization failed: {e}")

    # Verify Grok API connectivity
    if grok.health_check():
        logger.info("Grok API connection verified.")
    else:
        logger.error("Cannot reach Grok API. Check XAI_API_KEY.")
        sys.exit(1)

    # Sync positions with exchange on startup (crash recovery)
    logger.info("Syncing positions with exchange...")
    startup_db = None
    try:
        startup_db = get_db_connection()
        position_mgr.sync_positions(startup_db)
    except Exception as e:
        logger.warning(f"Position sync on startup failed: {e}")
    finally:
        if startup_db is not None:
            startup_db.close()

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Notify startup
    notifier.send_bot_online(
        mode="LIVE" if LIVE_TRADING else "PAPER",
        assets=ASSET_UNIVERSE,
        cycle_interval=CYCLE_INTERVAL_MINUTES,
    )

    # --- Main Loop ---
    cycle_number = 0

    while not _shutdown_requested:
        cycle_number += 1

        next_cycle_minutes = run_cycle(
            cycle_number=cycle_number,
            grok=grok,
            market_fetcher=market_fetcher,
            portfolio_mgr=portfolio_mgr,
            trade_history=trade_history,
            risk_guardian=risk_guardian,
            order_mgr=order_mgr,
            position_mgr=position_mgr,
            decision_parser=decision_parser,
            notifier=notifier,
            sentiment_fetcher=sentiment_fetcher,
        )

        if _shutdown_requested:
            break

        # Sleep until next cycle (check shutdown flag every 10s)
        sleep_seconds = next_cycle_minutes * 60
        logger.info(f"Sleeping {next_cycle_minutes} minutes until next cycle...")
        elapsed = 0
        while elapsed < sleep_seconds and not _shutdown_requested:
            time.sleep(min(10, sleep_seconds - elapsed))
            elapsed += 10
            # Heartbeat check: alert if bot seems stuck
            notifier.check_heartbeat(next_cycle_minutes)

    # Graceful shutdown
    logger.info("Shutting down Sentinel gracefully...")
    notifier.send_bot_offline()
    logger.info("Sentinel stopped. Goodbye.")


if __name__ == "__main__":
    main()
