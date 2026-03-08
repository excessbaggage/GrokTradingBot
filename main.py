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
) -> int:
    """
    Execute one full trading cycle.

    Returns the suggested minutes until next cycle.
    """
    db = get_db_connection()
    logger.info(f"=== CYCLE {cycle_number} START ===")

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

        # --- Step 5: Build context prompt ---
        context = build_context_prompt(
            market_data=market_data,
            portfolio=portfolio,
            recent_trades=recent_trades,
            risk_status=risk_status,
            performance_summary=performance_summary,
            liquidation_data=liquidation_data,
            regime_data=regime_data,
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
            notifier.send_error_alert("Grok response parsing failed")
            return CYCLE_INTERVAL_MINUTES

        # --- Step 8: Log the full interaction ---
        _log_grok_interaction(db, system_prompt, context, raw_response, grok_response, cycle_number)

        logger.info(f"Grok stance: {grok_response.overall_stance}")
        logger.info(f"Decisions count: {len(grok_response.decisions)}")

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
                        notifier.send_trade_alert(decision, order_result)
                        logger.info(f"Trade executed: {order_result}")
                    else:
                        logger.error(f"Order placement failed for {decision.asset}")
                        notifier.send_error_alert(
                            f"Order placement failed: {decision.action} {decision.asset}"
                        )
                else:
                    logger.warning(f"REJECTED: {decision.action} {decision.asset} — {validation.reason}")
                    _log_rejection(db, decision, validation.reason)

        # --- Step 11: Manage existing positions ---
        position_mgr.manage_open_positions(db)

        # --- Step 11b: Save equity snapshot for dashboard chart ---
        _save_equity_snapshot(db, cycle_number, portfolio)

        # --- Step 12: Daily summary ---
        now = datetime.now(timezone.utc)
        if now.hour == DAILY_SUMMARY_HOUR_UTC and now.minute < CYCLE_INTERVAL_MINUTES:
            summary = _generate_daily_summary(db, portfolio)
            notifier.send_daily_summary(summary)

        # --- Step 13: Determine next cycle timing ---
        suggested = grok_response.next_review_suggestion_minutes
        next_cycle = max(min(suggested, MAX_CYCLE_INTERVAL_MINUTES), MIN_CYCLE_INTERVAL_MINUTES)
        if suggested != next_cycle:
            logger.info(f"Grok suggested {suggested} min, clamped to {next_cycle} min")
        logger.info(f"=== CYCLE {cycle_number} COMPLETE === Next in {next_cycle} min")
        return next_cycle

    except Exception as e:
        logger.exception(f"Cycle {cycle_number} error: {e}")
        notifier.send_error_alert(f"Cycle {cycle_number} error: {e}")
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
        logger.error(f"Failed to log rejection: {e}")


def _save_equity_snapshot(db, cycle_number, portfolio):
    """Save a per-cycle equity snapshot for the dashboard chart.

    Args:
        db: Open SQLite connection.
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
        logger.error(f"Failed to save equity snapshot: {e}")


def _generate_daily_summary(db, portfolio) -> dict:
    """Generate a daily performance summary.

    Returns keys matching what ``Notifier.send_daily_summary()`` expects:
    ``date``, ``daily_pnl_pct``, ``equity``, ``peak_equity``,
    ``trades_today``, ``wins``, ``losses``, ``win_rate``,
    ``open_positions``, ``total_exposure_pct``.
    """
    from data.trade_history import TradeHistoryManager

    th = TradeHistoryManager()
    trades_today_list = th.get_trades_today(db)

    wins = sum(1 for t in trades_today_list if t.get("pnl", 0) > 0)
    losses = sum(1 for t in trades_today_list if t.get("pnl", 0) < 0)
    total_pnl = sum(t.get("pnl", 0) for t in trades_today_list)
    equity = portfolio.get("total_equity", 0)
    daily_pnl_pct = total_pnl / equity if equity > 0 else 0.0

    return {
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
    notifier.send_risk_alert(
        "bot_started",
        f"Sentinel started in {'LIVE' if LIVE_TRADING else 'PAPER'} mode. "
        f"Assets: {', '.join(ASSET_UNIVERSE)}",
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

    # Graceful shutdown
    logger.info("Shutting down Sentinel gracefully...")
    notifier.send_risk_alert("bot_stopped", "Sentinel has been shut down.")
    logger.info("Sentinel stopped. Goodbye.")


if __name__ == "__main__":
    main()
