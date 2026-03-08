"""
Grok Trader — Web Monitoring Dashboard

A read-only Flask dashboard that connects to the Supabase PostgreSQL
database and presents real-time trading status, portfolio metrics,
positions, trade history, and Grok's analysis in a dark-themed
trading terminal UI.

Run alongside the bot as a separate process:
    python dashboard.py

Then open http://localhost:5050 in your browser.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

from flask import Flask, g, jsonify, render_template_string

from config.trading_config import (
    ASSET_UNIVERSE,
    LIVE_TRADING,
    STARTING_CAPITAL,
)
from data.database import get_db_connection

# ═══════════════════════════════════════════════════════════════════════════
# APP SETUP
# ═══════════════════════════════════════════════════════════════════════════

app = Flask(__name__)

DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "5050"))
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "127.0.0.1")


def get_ro_db():
    """Get a database connection, cached on Flask's request-scoped ``g``.

    The connection is automatically closed at the end of the request
    by :func:`close_db`.
    """
    if "db" not in g:
        g.db = get_db_connection()
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    """Close the database connection at the end of each request."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


def rows_to_dicts(rows: list) -> list[dict]:
    """Convert row objects to plain dicts for JSON serialization."""
    return [dict(row) for row in rows]


# ═══════════════════════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════════════════════


@app.route("/api/status")
def api_status():
    """Bot status: last cycle time, cycle count, trading mode."""
    try:
        db = get_ro_db()
        row = db.execute(
            "SELECT * FROM grok_logs ORDER BY id DESC LIMIT 1"
        ).fetchone()


        # Get last activity event
        event_row = db.execute(
            "SELECT timestamp, event_type, summary FROM cycle_events ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        last_event = {
            "timestamp": event_row["timestamp"] if event_row else None,
            "type": event_row["event_type"] if event_row else None,
            "summary": event_row["summary"] if event_row else None,
        } if event_row else None

        if row:
            return jsonify({
                "online": True,
                "last_cycle_time": row["timestamp"],
                "cycle_number": row["cycle_number"],
                "mode": "LIVE" if LIVE_TRADING else "PAPER",
                "assets": ASSET_UNIVERSE,
                "last_event": last_event,
            })
        return jsonify({
            "online": False,
            "last_cycle_time": None,
            "cycle_number": 0,
            "mode": "LIVE" if LIVE_TRADING else "PAPER",
            "assets": ASSET_UNIVERSE,
            "last_event": last_event,
        })
    except Exception as e:
        app.logger.error("API error: %s", e)
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/portfolio")
def api_portfolio():
    """Portfolio: equity, daily/weekly P&L, drawdown.

    In paper mode, computes equity from trades table using
    pnl_pct * size_pct * STARTING_CAPITAL * leverage for each open trade.
    Falls back to equity_snapshots or daily_summaries if available.
    """
    try:
        db = get_ro_db()

        # --- Compute current paper equity from open trades ---
        open_trades = db.execute(
            """SELECT size_pct, leverage, pnl_pct FROM trades WHERE status = 'open'"""
        ).fetchall()

        unrealized_pnl = 0.0
        for t in open_trades:
            pnl_pct = float(t["pnl_pct"] or 0.0)
            size_pct = float(t["size_pct"] or 0.0)
            leverage = float(t["leverage"] or 1.0)
            unrealized_pnl += pnl_pct * size_pct * STARTING_CAPITAL * leverage

        # Realized P&L from closed trades
        realized_row = db.execute(
            "SELECT COALESCE(SUM(pnl), 0) AS realized FROM trades WHERE status = 'closed'"
        ).fetchone()
        realized_pnl = float(realized_row["realized"]) if realized_row else 0.0

        # Total fees
        fee_row = db.execute(
            "SELECT COALESCE(SUM(fees), 0) AS total_fees FROM trades"
        ).fetchone()
        total_fees = float(fee_row["total_fees"]) if fee_row else 0.0

        equity = STARTING_CAPITAL + unrealized_pnl + realized_pnl - total_fees

        # --- Capital deployed in open positions ---
        invested = 0.0
        for t in open_trades:
            size_pct = float(t["size_pct"] or 0.0)
            invested += size_pct * equity
        uninvested = equity - invested

        # --- Daily realized P&L ---
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        daily_pnl_row = db.execute(
            """SELECT COALESCE(SUM(pnl), 0) AS total_pnl
               FROM trades WHERE status = 'closed' AND DATE(closed_at) = ?""",
            (today_str,),
        ).fetchone()

        # --- Weekly realized P&L ---
        week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        weekly_pnl_row = db.execute(
            """SELECT COALESCE(SUM(pnl), 0) AS total_pnl
               FROM trades WHERE status = 'closed' AND closed_at >= ?""",
            (week_ago,),
        ).fetchone()

        # Peak equity from snapshots
        peak_row = db.execute(
            "SELECT COALESCE(MAX(equity), 0) AS peak FROM equity_snapshots"
        ).fetchone()



        peak = max(float(peak_row["peak"]) if peak_row else 0, STARTING_CAPITAL, equity)
        drawdown = ((peak - equity) / peak * 100) if peak > 0 and equity < peak else 0.0
        daily_pnl = float(daily_pnl_row["total_pnl"]) if daily_pnl_row else 0.0
        weekly_pnl = float(weekly_pnl_row["total_pnl"]) if weekly_pnl_row else 0.0

        daily_pnl_pct = (daily_pnl / STARTING_CAPITAL * 100) if STARTING_CAPITAL > 0 else 0.0
        weekly_pnl_pct = (weekly_pnl / STARTING_CAPITAL * 100) if STARTING_CAPITAL > 0 else 0.0

        return jsonify({
            "equity": round(equity, 2),
            "starting_capital": STARTING_CAPITAL,
            "total_return": round(
                ((equity - STARTING_CAPITAL) / STARTING_CAPITAL * 100), 4
            ) if STARTING_CAPITAL > 0 else 0.0,
            "invested": round(invested, 2),
            "uninvested": round(uninvested, 2),
            "invested_pct": round((invested / equity * 100), 2) if equity > 0 else 0.0,
            "daily_pnl": round(daily_pnl, 2),
            "daily_pnl_pct": round(daily_pnl_pct, 4),
            "weekly_pnl": round(weekly_pnl, 2),
            "weekly_pnl_pct": round(weekly_pnl_pct, 4),
            "peak_equity": round(peak, 2),
            "drawdown_pct": round(drawdown, 4),
            "total_fees": round(total_fees, 2),
        })
    except Exception as e:
        app.logger.error("API error: %s", e)
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/risk")
def api_risk():
    """Risk metrics: exposure, consecutive losses, trade counts."""
    try:
        db = get_ro_db()

        # Open positions for exposure (read from trades, not positions table)
        positions = db.execute(
            "SELECT * FROM trades WHERE status = 'open'"
        ).fetchall()
        total_exposure = sum(float(p["size_pct"]) for p in positions)

        # Consecutive losses
        recent = db.execute(
            """SELECT pnl FROM trades WHERE status = 'closed'
               ORDER BY closed_at DESC LIMIT 50"""
        ).fetchall()
        consecutive_losses = 0
        for row in recent:
            pnl = row["pnl"]
            if pnl is None:
                continue  # Skip trades closed by sync with no dollar P&L
            if float(pnl) < 0:
                consecutive_losses += 1
            else:
                break

        # Today's trade count
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        trade_count_row = db.execute(
            """SELECT COUNT(*) AS cnt FROM trades
               WHERE DATE(opened_at) = ?""",
            (today_str,),
        ).fetchone()

        # Total rejection count
        rejection_count_row = db.execute(
            "SELECT COUNT(*) AS cnt FROM rejections"
        ).fetchone()

        # Win rate from all closed trades
        wr_row = db.execute(
            """SELECT
                   COUNT(*) AS total,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins
               FROM trades WHERE status = 'closed' AND pnl IS NOT NULL"""
        ).fetchone()
        total_closed = int(wr_row["total"]) if wr_row else 0
        total_wins = int(wr_row["wins"]) if wr_row else 0
        win_rate = (total_wins / total_closed * 100) if total_closed > 0 else 0.0

        return jsonify({
            "total_exposure_pct": total_exposure * 100,
            "open_position_count": len(positions),
            "consecutive_losses": consecutive_losses,
            "trades_today": trade_count_row["cnt"] if trade_count_row else 0,
            "total_rejections": rejection_count_row["cnt"] if rejection_count_row else 0,
            "win_rate": round(win_rate, 1),
            "total_closed_trades": total_closed,
            "total_wins": total_wins,
        })
    except Exception as e:
        app.logger.error("API error: %s", e)
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/positions")
def api_positions():
    """Open positions with unrealized P&L.

    Reads from the trades table (where paper trades are logged) rather
    than the positions table (which is unused in paper mode).  Computes
    unrealized P&L in USD from the stored pnl_pct.
    """
    try:
        db = get_ro_db()
        rows = db.execute(
            """SELECT asset, side, size_pct, leverage, entry_price,
                      stop_loss, take_profit, pnl_pct, opened_at
               FROM trades WHERE status = 'open'
               ORDER BY opened_at DESC"""
        ).fetchall()


        positions = []
        for row in rows:
            d = dict(row)
            pnl_pct = float(d.get("pnl_pct") or 0.0)
            size_pct = float(d.get("size_pct") or 0.0)
            leverage = float(d.get("leverage") or 1.0)
            unrealized_usd = pnl_pct * size_pct * STARTING_CAPITAL * leverage
            d["unrealized_pnl"] = round(unrealized_usd, 4)
            d["unrealized_pnl_pct"] = round(pnl_pct * 100, 4)
            positions.append(d)

        return jsonify({"positions": positions})
    except Exception as e:
        app.logger.error("API error: %s", e)
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/trades")
def api_trades():
    """Recent trade history (last 50)."""
    try:
        db = get_ro_db()
        rows = db.execute(
            """SELECT asset, side, action, size_pct, leverage,
                      entry_price, exit_price, stop_loss, take_profit,
                      pnl, pnl_pct, status, conviction, reasoning,
                      opened_at, closed_at
               FROM trades ORDER BY opened_at DESC LIMIT 50"""
        ).fetchall()

        return jsonify({"trades": rows_to_dicts(rows)})
    except Exception as e:
        app.logger.error("API error: %s", e)
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/analysis")
def api_analysis():
    """Latest Grok market analysis and stance."""
    try:
        db = get_ro_db()
        row = db.execute(
            """SELECT timestamp, response_text, decisions_json, cycle_number
               FROM grok_logs ORDER BY id DESC LIMIT 1"""
        ).fetchone()


        if not row:
            return jsonify({"analysis": None})

        # Try to parse the response text for the overall_stance
        response_text = row["response_text"] or ""
        decisions_json = row["decisions_json"] or "[]"

        # Extract structured data from Grok's response
        stance = ""
        market_analysis = {}
        try:
            parsed = json.loads(response_text)
            stance = parsed.get("overall_stance", "")
            market_analysis = parsed.get("market_analysis", {})
        except (json.JSONDecodeError, TypeError):
            # Response might not be valid JSON — just show raw text
            stance = response_text[:500] if response_text else "No analysis available"

        return jsonify({
            "analysis": {
                "timestamp": row["timestamp"],
                "cycle_number": row["cycle_number"],
                "overall_stance": stance,
                "market_analysis": market_analysis,
                "decisions": json.loads(decisions_json),
            }
        })
    except Exception as e:
        app.logger.error("API error: %s", e)
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/rejections")
def api_rejections():
    """Risk Guardian rejections (last 20)."""
    try:
        db = get_ro_db()
        rows = db.execute(
            """SELECT timestamp, asset, action, reason
               FROM rejections ORDER BY timestamp DESC LIMIT 20"""
        ).fetchall()

        return jsonify({"rejections": rows_to_dicts(rows)})
    except Exception as e:
        app.logger.error("API error: %s", e)
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/activity")
def api_activity():
    """Unified activity feed from cycle_events table."""
    try:
        db = get_ro_db()
        rows = db.execute(
            """SELECT timestamp, cycle_number, event_type, severity,
                      asset, summary, details_json
               FROM cycle_events
               ORDER BY timestamp DESC LIMIT 50"""
        ).fetchall()

        events = []
        for row in rows:
            d = dict(row)
            if d.get("details_json") and isinstance(d["details_json"], str):
                try:
                    d["details"] = json.loads(d["details_json"])
                except json.JSONDecodeError:
                    d["details"] = None
            elif isinstance(d.get("details_json"), dict):
                d["details"] = d["details_json"]
            else:
                d["details"] = None
            d.pop("details_json", None)
            events.append(d)

        return jsonify({"events": events})
    except Exception as e:
        app.logger.error("API error: %s", e)
        return jsonify({"events": []})


@app.route("/api/market")
def api_market():
    """Latest market snapshot per asset (prices, regime, sentiment)."""
    try:
        db = get_ro_db()
        latest = db.execute(
            "SELECT MAX(cycle_number) AS latest FROM market_snapshots"
        ).fetchone()

        if not latest or not latest["latest"]:
            return jsonify({"assets": [], "cycle_number": 0})

        cycle = latest["latest"]
        rows = db.execute(
            """SELECT asset, price, change_24h_pct, funding_rate, open_interest,
                      rsi_14, atr_pct, volatility_regime,
                      market_regime, regime_confidence, adx, choppiness_index,
                      sentiment_score, sentiment_momentum, sentiment_volume,
                      sentiment_topics, timestamp
               FROM market_snapshots
               WHERE cycle_number = ?
               ORDER BY asset""",
            (cycle,),
        ).fetchall()

        return jsonify({"assets": rows_to_dicts(rows), "cycle_number": cycle})
    except Exception as e:
        app.logger.error("API error: %s", e)
        return jsonify({"assets": [], "cycle_number": 0})


@app.route("/api/performance")
def api_performance():
    """Cached performance analytics from the latest cycle."""
    try:
        db = get_ro_db()
        row = db.execute(
            """SELECT metrics_json, cycle_number, timestamp
               FROM performance_cache
               ORDER BY timestamp DESC LIMIT 1"""
        ).fetchone()

        if not row:
            return jsonify({"performance": None})

        metrics = row["metrics_json"]
        if isinstance(metrics, str):
            metrics = json.loads(metrics)

        return jsonify({
            "performance": metrics,
            "cycle_number": row["cycle_number"],
            "timestamp": row["timestamp"],
        })
    except Exception as e:
        app.logger.error("API error: %s", e)
        return jsonify({"performance": None})


@app.route("/api/equity-chart")
def api_equity_chart():
    """Equity curve data for Chart.js.

    Reads from the per-cycle equity_snapshots table for granular
    intraday equity tracking.  Falls back to daily_summaries for
    historical data.
    """
    try:
        db = get_ro_db()

        # Try per-cycle snapshots first (populated every cycle by main.py)
        rows = db.execute(
            """SELECT timestamp, cycle_number, equity, unrealized_pnl,
                      realized_pnl, open_positions, total_exposure
               FROM equity_snapshots ORDER BY timestamp ASC"""
        ).fetchall()

        if rows:
            data = []
            for row in rows:
                data.append({
                    "date": row["timestamp"],
                    "ending_equity": row["equity"],
                    "pnl": float(row["unrealized_pnl"] or 0) + float(row["realized_pnl"] or 0),
                    "pnl_pct": round(
                        (float(row["equity"]) - STARTING_CAPITAL) / STARTING_CAPITAL * 100, 4
                    ) if STARTING_CAPITAL > 0 else 0.0,
                    "open_positions": row["open_positions"],
                    "cycle_number": row["cycle_number"],
                })
    
            return jsonify({"equity_curve": data})

        # Fall back to daily summaries
        rows = db.execute(
            """SELECT date, ending_equity, pnl, pnl_pct, max_drawdown
               FROM daily_summaries ORDER BY date ASC"""
        ).fetchall()


        data = rows_to_dicts(rows)

        # If nothing at all, return starting capital as the only point
        if not data:
            data = [{
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "ending_equity": STARTING_CAPITAL,
                "pnl": 0,
                "pnl_pct": 0,
                "max_drawdown": 0,
            }]

        return jsonify({"equity_curve": data})
    except Exception as e:
        app.logger.error("API error: %s", e)
        return jsonify({"error": "Internal server error"}), 500


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PAGE — EMBEDDED HTML TEMPLATE
# ═══════════════════════════════════════════════════════════════════════════

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Grok Trader — Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    colors: {
                        dark: {
                            bg: '#0f1117',
                            card: '#1a1d29',
                            border: '#2a2d3a',
                            text: '#e1e4eb',
                            muted: '#8b8fa3',
                        },
                        profit: '#22c55e',
                        loss: '#ef4444',
                        accent: '#6366f1',
                    }
                }
            }
        }
    </script>
    <style>
        body { background: #0f1117; font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', monospace; }
        .card { background: #1a1d29; border: 1px solid #2a2d3a; border-radius: 12px; }
        .pulse-green { animation: pulse-g 2s infinite; }
        .pulse-yellow { animation: pulse-y 2s infinite; }
        .pulse-red { animation: pulse-r 2s infinite; }
        @keyframes pulse-g { 0%, 100% { box-shadow: 0 0 0 0 rgba(34,197,94,0.4); } 50% { box-shadow: 0 0 0 6px rgba(34,197,94,0); } }
        @keyframes pulse-y { 0%, 100% { box-shadow: 0 0 0 0 rgba(234,179,8,0.4); } 50% { box-shadow: 0 0 0 6px rgba(234,179,8,0); } }
        @keyframes pulse-r { 0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.4); } 50% { box-shadow: 0 0 0 6px rgba(239,68,68,0); } }
        .status-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
        table { width: 100%; border-collapse: collapse; }
        th { text-align: left; padding: 8px 12px; color: #8b8fa3; font-weight: 500; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; border-bottom: 1px solid #2a2d3a; }
        td { padding: 8px 12px; color: #e1e4eb; font-size: 0.85rem; border-bottom: 1px solid #1f2233; }
        .scrollable { max-height: 320px; overflow-y: auto; }
        .scrollable::-webkit-scrollbar { width: 6px; }
        .scrollable::-webkit-scrollbar-track { background: #1a1d29; }
        .scrollable::-webkit-scrollbar-thumb { background: #2a2d3a; border-radius: 3px; }
    </style>
</head>
<body class="min-h-screen text-dark-text">

    <!-- HEADER -->
    <header class="border-b border-dark-border px-6 py-4 flex items-center justify-between">
        <div class="flex items-center gap-3">
            <div class="text-2xl font-bold tracking-tight">
                <span class="text-accent">GROK TRADER</span>
                <span class="text-dark-muted ml-2 text-sm font-normal">Crypto Dashboard</span>
            </div>
        </div>
        <div class="flex items-center gap-4">
            <div id="mode-badge" class="px-3 py-1 rounded-full text-xs font-medium bg-yellow-500/20 text-yellow-400">
                PAPER
            </div>
            <div class="flex items-center gap-2">
                <div id="status-dot" class="status-dot bg-gray-500"></div>
                <span id="status-text" class="text-dark-muted text-sm">Connecting...</span>
            </div>
            <div class="text-dark-muted text-xs" id="last-update"></div>
        </div>
    </header>

    <!-- MAIN CONTENT -->
    <main class="p-6 max-w-7xl mx-auto space-y-6">

        <!-- TOP STATS ROW -->
        <div class="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-8 gap-4">
            <div class="card p-4">
                <div class="text-dark-muted text-xs uppercase tracking-wide mb-1">Equity</div>
                <div id="equity" class="text-2xl font-bold">$0.00</div>
                <div id="total-return" class="text-sm text-dark-muted">+0.00%</div>
            </div>
            <div class="card p-4 border-accent/30">
                <div class="text-dark-muted text-xs uppercase tracking-wide mb-1">Win Rate</div>
                <div id="win-rate" class="text-2xl font-bold text-accent">0.0%</div>
                <div id="win-rate-detail" class="text-sm text-dark-muted">0/0 trades</div>
            </div>
            <div class="card p-4">
                <div class="text-dark-muted text-xs uppercase tracking-wide mb-1">Invested</div>
                <div id="invested" class="text-2xl font-bold text-accent">$0.00</div>
                <div id="invested-pct" class="text-sm text-dark-muted">0% deployed</div>
            </div>
            <div class="card p-4">
                <div class="text-dark-muted text-xs uppercase tracking-wide mb-1">Uninvested</div>
                <div id="uninvested" class="text-2xl font-bold text-dark-text">$0.00</div>
                <div id="uninvested-label" class="text-sm text-dark-muted">available</div>
            </div>
            <div class="card p-4">
                <div class="text-dark-muted text-xs uppercase tracking-wide mb-1">Daily P&L</div>
                <div id="daily-pnl" class="text-2xl font-bold">$0.00</div>
                <div id="daily-pnl-pct" class="text-sm text-dark-muted">+0.00%</div>
            </div>
            <div class="card p-4">
                <div class="text-dark-muted text-xs uppercase tracking-wide mb-1">Weekly P&L</div>
                <div id="weekly-pnl" class="text-2xl font-bold">$0.00</div>
                <div id="weekly-pnl-pct" class="text-sm text-dark-muted">+0.00%</div>
            </div>
            <div class="card p-4">
                <div class="text-dark-muted text-xs uppercase tracking-wide mb-1">Drawdown</div>
                <div id="drawdown" class="text-2xl font-bold text-dark-text">0.00%</div>
                <div id="peak-equity" class="text-sm text-dark-muted">Peak: $0.00</div>
            </div>
            <div class="card p-4 border-yellow-500/30">
                <div class="text-dark-muted text-xs uppercase tracking-wide mb-1">Total Fees</div>
                <div id="total-fees" class="text-2xl font-bold text-yellow-400">$0.00</div>
                <div id="total-fees-label" class="text-sm text-dark-muted">accrued trading fees</div>
            </div>
        </div>

        <!-- EQUITY CHART -->
        <div class="card p-4">
            <div class="text-dark-muted text-xs uppercase tracking-wide mb-3">Equity Curve</div>
            <div style="height: 250px;">
                <canvas id="equityChart"></canvas>
            </div>
        </div>

        <!-- RECENT TRADES + OPEN POSITIONS -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div class="card p-4">
                <div class="text-dark-muted text-xs uppercase tracking-wide mb-3">Recent Trades</div>
                <div class="scrollable">
                    <table>
                        <thead>
                            <tr>
                                <th>Time</th>
                                <th>Asset</th>
                                <th>Action</th>
                                <th>Side</th>
                                <th>Size</th>
                                <th>Entry</th>
                                <th>Exit</th>
                                <th>P&L</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody id="trades-table">
                            <tr><td colspan="9" class="text-center text-dark-muted py-8">No trades yet</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- OPEN POSITIONS -->
            <div class="card p-4">
                <div class="text-dark-muted text-xs uppercase tracking-wide mb-3">Open Positions</div>
                <div class="scrollable">
                    <table>
                        <thead>
                            <tr>
                                <th>Asset</th>
                                <th>Side</th>
                                <th>Size</th>
                                <th>Leverage</th>
                                <th>Entry</th>
                                <th>uPnL</th>
                            </tr>
                        </thead>
                        <tbody id="positions-table">
                            <tr><td colspan="6" class="text-center text-dark-muted py-8">No open positions</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- ACTIVITY FEED -->
        <div class="card p-4">
            <div class="flex items-center justify-between mb-3">
                <div class="text-dark-muted text-xs uppercase tracking-wide">Live Activity Feed</div>
                <div class="text-dark-muted text-xs" id="feed-status"></div>
            </div>
            <div class="scrollable" style="max-height: 360px;" id="activity-feed">
                <div class="text-dark-muted text-center py-8">Waiting for first cycle...</div>
            </div>
        </div>

        <!-- MARKET OVERVIEW -->
        <div class="card p-4">
            <div class="flex items-center justify-between mb-3">
                <div class="text-dark-muted text-xs uppercase tracking-wide">Market Overview</div>
                <div class="text-dark-muted text-xs" id="market-cycle"></div>
            </div>
            <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3" id="market-grid">
                <div class="text-dark-muted text-center py-8 col-span-full">Loading market data...</div>
            </div>
        </div>

        <!-- RISK METRICS ROW -->
        <div class="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div class="card p-3 text-center">
                <div class="text-dark-muted text-xs uppercase mb-1">Exposure</div>
                <div id="exposure" class="text-lg font-bold">0%</div>
            </div>
            <div class="card p-3 text-center">
                <div class="text-dark-muted text-xs uppercase mb-1">Open Positions</div>
                <div id="open-positions" class="text-lg font-bold">0</div>
            </div>
            <div class="card p-3 text-center">
                <div class="text-dark-muted text-xs uppercase mb-1">Loss Streak</div>
                <div id="loss-streak" class="text-lg font-bold">0</div>
            </div>
            <div class="card p-3 text-center">
                <div class="text-dark-muted text-xs uppercase mb-1">Trades Today</div>
                <div id="trades-today" class="text-lg font-bold">0</div>
            </div>
            <div class="card p-3 text-center">
                <div class="text-dark-muted text-xs uppercase mb-1">Rejections</div>
                <div id="total-rejections" class="text-lg font-bold text-loss">0</div>
            </div>
        </div>

        <!-- GROK ANALYSIS -->
        <div class="card p-4">
            <div class="text-dark-muted text-xs uppercase tracking-wide mb-3">Latest Grok Analysis</div>
            <div id="grok-analysis" class="text-sm space-y-3">
                <div class="text-dark-muted">Waiting for first cycle...</div>
            </div>
        </div>

        <!-- PERFORMANCE ANALYTICS -->
        <div class="card p-4">
            <div class="flex items-center gap-4 mb-3">
                <div class="text-dark-muted text-xs uppercase tracking-wide">Performance Analytics</div>
                <div class="flex gap-1">
                    <button onclick="showPerfTab('strategy')" id="tab-strategy" class="px-2 py-1 rounded text-xs bg-accent/20 text-accent">Strategy</button>
                    <button onclick="showPerfTab('assets')" id="tab-assets" class="px-2 py-1 rounded text-xs bg-dark-border text-dark-muted">Assets</button>
                    <button onclick="showPerfTab('streaks')" id="tab-streaks" class="px-2 py-1 rounded text-xs bg-dark-border text-dark-muted">Streaks</button>
                </div>
            </div>
            <div id="perf-strategy" class="perf-content"></div>
            <div id="perf-assets" class="perf-content hidden"></div>
            <div id="perf-streaks" class="perf-content hidden"></div>
            <div id="perf-empty" class="text-dark-muted text-center py-6 text-sm">No performance data yet</div>
        </div>

        <!-- REJECTIONS -->
        <div class="card p-4">
            <div class="text-dark-muted text-xs uppercase tracking-wide mb-3">Risk Guardian Rejections</div>
            <div class="scrollable">
                <table>
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Asset</th>
                            <th>Action</th>
                            <th>Reason</th>
                        </tr>
                    </thead>
                    <tbody id="rejections-table">
                        <tr><td colspan="4" class="text-center text-dark-muted py-8">No rejections</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

    </main>

    <!-- FOOTER -->
    <footer class="text-center text-dark-muted text-xs py-4 border-t border-dark-border">
        Grok Trader &mdash; AI-Powered Crypto Trading &mdash; Read-only dashboard (fast: 15s / slow: 30s)
    </footer>

    <script>
    // ─── GLOBALS ─────────────────────────────────────────────
    let equityChart = null;

    // ─── FORMATTERS ─────────────────────────────────────────
    function fmtUsd(val) {
        if (val == null) return '$0.00';
        const neg = val < 0;
        const s = Math.abs(val).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
        return neg ? `-$${s}` : `$${s}`;
    }

    function fmtPct(val, includeSign = true) {
        if (val == null) return '0.00%';
        const s = Math.abs(val).toFixed(2);
        if (includeSign && val > 0) return `+${s}%`;
        if (val < 0) return `-${s}%`;
        return `${s}%`;
    }

    function fmtPrice(val) {
        if (val == null || val === 0) return '-';
        return val.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
    }

    function fmtTime(ts) {
        if (!ts) return '-';
        try {
            const d = new Date(ts.includes('T') ? ts : ts + 'Z');
            return d.toLocaleString('en-US', {month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'});
        } catch { return ts; }
    }

    function escapeHtml(str) {
        if (str == null) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    function pnlColor(val) {
        if (val > 0) return 'text-profit';
        if (val < 0) return 'text-loss';
        return 'text-dark-muted';
    }

    function sideColor(side) {
        return side === 'long' ? 'text-profit' : 'text-loss';
    }

    // ─── DATA FETCHERS ──────────────────────────────────────

    async function fetchJson(url) {
        try {
            const resp = await fetch(url);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            return await resp.json();
        } catch (e) {
            console.error(`Fetch failed: ${url}`, e);
            return null;
        }
    }

    async function updateStatus() {
        const data = await fetchJson('/api/status');
        if (!data) return;

        const dot = document.getElementById('status-dot');
        const text = document.getElementById('status-text');
        const badge = document.getElementById('mode-badge');

        badge.textContent = data.mode;
        if (data.mode === 'LIVE') {
            badge.className = 'px-3 py-1 rounded-full text-xs font-medium bg-red-500/20 text-red-400';
        } else {
            badge.className = 'px-3 py-1 rounded-full text-xs font-medium bg-yellow-500/20 text-yellow-400';
        }

        if (data.online && data.last_cycle_time) {
            // Check if last cycle was within 30 minutes
            const lastCycle = new Date(data.last_cycle_time.includes('T') ? data.last_cycle_time : data.last_cycle_time + 'Z');
            const minutesAgo = (Date.now() - lastCycle.getTime()) / 60000;

            if (minutesAgo < 30) {
                dot.className = 'status-dot bg-profit pulse-green';
                text.textContent = `Cycle ${data.cycle_number} — ${Math.round(minutesAgo)}m ago`;
            } else {
                dot.className = 'status-dot bg-yellow-500 pulse-yellow';
                text.textContent = `Sleeping — Cycle ${data.cycle_number} was ${Math.round(minutesAgo)}m ago`;
            }
        } else {
            dot.className = 'status-dot bg-gray-500';
            text.textContent = 'No cycles recorded';
        }
    }

    async function updatePortfolio() {
        const data = await fetchJson('/api/portfolio');
        if (!data || data.error) return;

        document.getElementById('equity').textContent = fmtUsd(data.equity);
        const retEl = document.getElementById('total-return');
        retEl.textContent = fmtPct(data.total_return);
        retEl.className = `text-sm ${pnlColor(data.total_return)}`;

        document.getElementById('invested').textContent = fmtUsd(data.invested);
        document.getElementById('invested-pct').textContent = `${(data.invested_pct || 0).toFixed(1)}% deployed`;
        document.getElementById('uninvested').textContent = fmtUsd(data.uninvested);

        const dpEl = document.getElementById('daily-pnl');
        dpEl.textContent = fmtUsd(data.daily_pnl);
        dpEl.className = `text-2xl font-bold ${pnlColor(data.daily_pnl)}`;
        const dppEl = document.getElementById('daily-pnl-pct');
        dppEl.textContent = fmtPct(data.daily_pnl_pct);
        dppEl.className = `text-sm ${pnlColor(data.daily_pnl_pct)}`;

        const wpEl = document.getElementById('weekly-pnl');
        wpEl.textContent = fmtUsd(data.weekly_pnl);
        wpEl.className = `text-2xl font-bold ${pnlColor(data.weekly_pnl)}`;
        const wppEl = document.getElementById('weekly-pnl-pct');
        wppEl.textContent = fmtPct(data.weekly_pnl_pct);
        wppEl.className = `text-sm ${pnlColor(data.weekly_pnl_pct)}`;

        const ddEl = document.getElementById('drawdown');
        ddEl.textContent = fmtPct(data.drawdown_pct, false);
        ddEl.className = `text-2xl font-bold ${data.drawdown_pct > 10 ? 'text-loss' : data.drawdown_pct > 5 ? 'text-yellow-400' : 'text-dark-text'}`;
        document.getElementById('peak-equity').textContent = `Peak: ${fmtUsd(data.peak_equity)}`;

        // Total fees card
        document.getElementById('total-fees').textContent = fmtUsd(data.total_fees || 0);
    }

    async function updateRisk() {
        const data = await fetchJson('/api/risk');
        if (!data || data.error) return;

        document.getElementById('exposure').textContent = fmtPct(data.total_exposure_pct, false);
        document.getElementById('open-positions').textContent = data.open_position_count;

        const lsEl = document.getElementById('loss-streak');
        lsEl.textContent = data.consecutive_losses;
        lsEl.className = `text-lg font-bold ${data.consecutive_losses >= 3 ? 'text-loss' : ''}`;

        document.getElementById('trades-today').textContent = data.trades_today;
        document.getElementById('total-rejections').textContent = data.total_rejections;

        // Win rate card
        const wr = data.win_rate || 0;
        const wrEl = document.getElementById('win-rate');
        wrEl.textContent = fmtPct(wr, false);
        wrEl.className = `text-2xl font-bold ${wr >= 50 ? 'text-profit' : wr > 0 ? 'text-loss' : 'text-accent'}`;
        document.getElementById('win-rate-detail').textContent =
            `${data.total_wins || 0}/${data.total_closed_trades || 0} trades`;
    }

    async function updatePositions() {
        const data = await fetchJson('/api/positions');
        if (!data || data.error) return;

        const tbody = document.getElementById('positions-table');
        if (!data.positions || data.positions.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="text-center text-dark-muted py-8">No open positions</td></tr>';
            return;
        }

        tbody.innerHTML = data.positions.map(p => `
            <tr>
                <td class="font-medium">${escapeHtml(p.asset)}</td>
                <td class="${sideColor(p.side)} font-medium uppercase">${escapeHtml(p.side)}</td>
                <td>${(p.size_pct * 100).toFixed(1)}%</td>
                <td>${p.leverage}x</td>
                <td>${fmtPrice(p.entry_price)}</td>
                <td class="${pnlColor(p.unrealized_pnl)}">${fmtUsd(p.unrealized_pnl)}</td>
            </tr>
        `).join('');
    }

    async function updateTrades() {
        const data = await fetchJson('/api/trades');
        if (!data || data.error) return;

        const tbody = document.getElementById('trades-table');
        if (!data.trades || data.trades.length === 0) {
            tbody.innerHTML = '<tr><td colspan="9" class="text-center text-dark-muted py-8">No trades yet</td></tr>';
            return;
        }

        tbody.innerHTML = data.trades.map(t => `
            <tr>
                <td class="text-dark-muted text-xs">${fmtTime(t.opened_at)}</td>
                <td class="font-medium">${escapeHtml(t.asset)}</td>
                <td>${escapeHtml(t.action)}</td>
                <td class="${sideColor(t.side)} uppercase">${escapeHtml(t.side)}</td>
                <td>${(t.size_pct * 100).toFixed(1)}%</td>
                <td>${fmtPrice(t.entry_price)}</td>
                <td>${fmtPrice(t.exit_price)}</td>
                <td class="${pnlColor(t.pnl)}">${t.pnl != null ? fmtUsd(t.pnl) : '-'}</td>
                <td>
                    <span class="px-2 py-0.5 rounded text-xs ${t.status === 'open' ? 'bg-accent/20 text-accent' : 'bg-dark-border text-dark-muted'}">
                        ${escapeHtml(t.status)}
                    </span>
                </td>
            </tr>
        `).join('');
    }

    async function updateAnalysis() {
        const data = await fetchJson('/api/analysis');
        if (!data || data.error || !data.analysis) return;

        const container = document.getElementById('grok-analysis');
        const a = data.analysis;
        const decisions = a.decisions || [];

        let html = `
            <div class="flex items-center gap-2 mb-2">
                <span class="text-dark-muted text-xs">Cycle ${a.cycle_number} &mdash; ${fmtTime(a.timestamp)}</span>
            </div>
            <div class="bg-dark-bg rounded-lg p-3 mb-3">
                <div class="text-xs text-dark-muted uppercase mb-1">Overall Stance</div>
                <div class="text-sm leading-relaxed">${escapeHtml(typeof a.overall_stance === 'string' ? a.overall_stance : JSON.stringify(a.overall_stance))}</div>
            </div>
        `;

        // Market analysis per asset
        if (a.market_analysis && typeof a.market_analysis === 'object') {
            html += '<div class="grid grid-cols-1 gap-2">';
            for (const [asset, info] of Object.entries(a.market_analysis)) {
                if (typeof info === 'object' && info !== null) {
                    const biasColor = info.bias === 'long' ? 'text-profit' : info.bias === 'short' ? 'text-loss' : 'text-dark-muted';
                    html += `
                        <div class="bg-dark-bg rounded p-2 text-xs">
                            <span class="font-bold uppercase">${escapeHtml(asset)}</span>
                            <span class="${biasColor} ml-2">${escapeHtml(info.bias || '-')}</span>
                            <span class="text-dark-muted ml-2">${escapeHtml(info.conviction || '-')} conviction</span>
                            ${info.summary ? `<div class="text-dark-muted mt-1">${escapeHtml(info.summary)}</div>` : ''}
                        </div>
                    `;
                }
            }
            html += '</div>';
        }

        // Decisions
        if (decisions.length > 0) {
            html += '<div class="mt-3 text-xs text-dark-muted uppercase mb-1">Decisions</div>';
            decisions.forEach(d => {
                html += `
                    <div class="bg-dark-bg rounded p-2 text-xs mb-1">
                        <span class="font-medium">${escapeHtml(d.action)}</span>
                        <span class="ml-1">${escapeHtml(d.asset)}</span>
                        <span class="text-dark-muted ml-2">${escapeHtml(d.conviction)} conviction</span>
                    </div>
                `;
            });
        } else {
            html += '<div class="text-xs text-dark-muted mt-2">No actionable trades this cycle</div>';
        }

        container.innerHTML = html;
    }

    async function updateRejections() {
        const data = await fetchJson('/api/rejections');
        if (!data || data.error) return;

        const tbody = document.getElementById('rejections-table');
        if (!data.rejections || data.rejections.length === 0) {
            tbody.innerHTML = '<tr><td colspan="4" class="text-center text-dark-muted py-8">No rejections</td></tr>';
            return;
        }

        tbody.innerHTML = data.rejections.map(r => `
            <tr>
                <td class="text-dark-muted text-xs">${fmtTime(r.timestamp)}</td>
                <td class="font-medium">${escapeHtml(r.asset)}</td>
                <td>${escapeHtml(r.action)}</td>
                <td class="text-loss text-xs">${escapeHtml(r.reason)}</td>
            </tr>
        `).join('');
    }

    async function updateEquityChart() {
        const data = await fetchJson('/api/equity-chart');
        if (!data || data.error || !data.equity_curve) return;

        const labels = data.equity_curve.map(d => d.date);
        const values = data.equity_curve.map(d => d.ending_equity);

        const ctx = document.getElementById('equityChart').getContext('2d');

        if (equityChart) {
            equityChart.data.labels = labels;
            equityChart.data.datasets[0].data = values;
            equityChart.update('none');
            return;
        }

        equityChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Equity',
                    data: values,
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: values.length > 30 ? 0 : 3,
                    pointBackgroundColor: '#6366f1',
                    borderWidth: 2,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { intersect: false, mode: 'index' },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: '#1a1d29',
                        borderColor: '#2a2d3a',
                        borderWidth: 1,
                        titleColor: '#e1e4eb',
                        bodyColor: '#8b8fa3',
                        callbacks: {
                            label: (ctx) => `Equity: $${ctx.raw.toLocaleString('en-US', {minimumFractionDigits: 2})}`
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: '#1f2233' },
                        ticks: { color: '#8b8fa3', maxTicksLimit: 10 },
                    },
                    y: {
                        grid: { color: '#1f2233' },
                        ticks: {
                            color: '#8b8fa3',
                            callback: (v) => '$' + v.toLocaleString()
                        }
                    }
                }
            }
        });
    }

    // ─── HELPERS FOR NEW PANELS ─────────────────────────────

    function timeAgo(ts) {
        if (!ts) return '';
        try {
            const d = new Date(ts.includes('T') ? ts : ts + 'Z');
            const secs = Math.floor((Date.now() - d.getTime()) / 1000);
            if (secs < 60) return secs + 's ago';
            if (secs < 3600) return Math.floor(secs / 60) + 'm ago';
            if (secs < 86400) return Math.floor(secs / 3600) + 'h ago';
            return Math.floor(secs / 86400) + 'd ago';
        } catch { return ''; }
    }

    const EVENT_ICONS = {
        cycle_start: String.fromCodePoint(0x1F504), cycle_end: String.fromCodePoint(0x2705),
        trade_opened: String.fromCodePoint(0x1F4C8), trade_closed: String.fromCodePoint(0x1F4C9),
        trade_rejected: String.fromCodePoint(0x1F6AB), sentiment_fetched: String.fromCodePoint(0x1F4F0),
        grok_decision: String.fromCodePoint(0x1F9E0), error: String.fromCodePoint(0x26A0, 0xFE0F)
    };
    const SEVERITY_BORDER = {
        info: 'border-l-blue-500', success: 'border-l-green-500',
        warn: 'border-l-yellow-500', error: 'border-l-red-500'
    };

    // ─── ACTIVITY FEED ──────────────────────────────────────

    async function updateActivity() {
        const data = await fetchJson('/api/activity');
        if (!data || !data.events) return;

        const feed = document.getElementById('activity-feed');
        if (data.events.length === 0) {
            feed.textContent = 'Waiting for first cycle...';
            return;
        }
        document.getElementById('feed-status').textContent = data.events.length + ' events';

        // Build activity items using DOM methods (XSS-safe)
        while (feed.firstChild) feed.removeChild(feed.firstChild);
        data.events.forEach(ev => {
            const row = document.createElement('div');
            row.className = 'flex items-start gap-3 py-2 border-l-2 pl-3 mb-1 ' +
                (SEVERITY_BORDER[ev.severity] || 'border-l-blue-500');

            const icon = document.createElement('span');
            icon.className = 'text-sm mt-0.5';
            icon.textContent = EVENT_ICONS[ev.event_type] || String.fromCodePoint(0x25CF);

            const body = document.createElement('div');
            body.className = 'flex-1 min-w-0';

            const top = document.createElement('div');
            top.className = 'flex items-center gap-2';
            const summarySpan = document.createElement('span');
            summarySpan.className = 'text-sm truncate';
            summarySpan.textContent = ev.summary || '';
            const timeSpan = document.createElement('span');
            timeSpan.className = 'text-xs text-dark-muted whitespace-nowrap';
            timeSpan.textContent = timeAgo(ev.timestamp);
            top.appendChild(summarySpan);
            top.appendChild(timeSpan);

            const meta = document.createElement('div');
            meta.className = 'text-xs text-dark-muted';
            meta.textContent = 'Cycle ' + (ev.cycle_number || '-');
            if (ev.asset) meta.textContent += ' \u2022 ' + ev.asset;

            body.appendChild(top);
            body.appendChild(meta);
            row.appendChild(icon);
            row.appendChild(body);
            feed.appendChild(row);
        });
    }

    // ─── MARKET OVERVIEW ────────────────────────────────────

    function regimeBadge(regime) {
        if (!regime) return '';
        const colors = {
            'trending_up': 'bg-green-500/20 text-green-400',
            'trending_down': 'bg-red-500/20 text-red-400',
            'ranging': 'bg-blue-500/20 text-blue-400',
            'volatile_expansion': 'bg-yellow-500/20 text-yellow-400',
            'mean_reverting': 'bg-purple-500/20 text-purple-400'
        };
        return colors[regime] || 'bg-dark-border text-dark-muted';
    }

    function sentimentColor(score) {
        if (score == null) return '#8b8fa3';
        if (score > 0.3) return '#22c55e';
        if (score < -0.3) return '#ef4444';
        return '#eab308';
    }

    async function updateMarket() {
        const data = await fetchJson('/api/market');
        if (!data || !data.assets) return;

        const grid = document.getElementById('market-grid');
        const cycleEl = document.getElementById('market-cycle');
        cycleEl.textContent = data.cycle_number ? 'Cycle ' + data.cycle_number : '';

        if (data.assets.length === 0) {
            grid.textContent = 'Loading market data...';
            return;
        }

        // Build market cards using DOM methods (XSS-safe)
        while (grid.firstChild) grid.removeChild(grid.firstChild);
        data.assets.forEach(a => {
            const card = document.createElement('div');
            card.className = 'bg-dark-bg rounded-lg p-3';

            // Asset name row
            const nameRow = document.createElement('div');
            nameRow.className = 'flex items-center justify-between mb-2';
            const nameSpan = document.createElement('span');
            nameSpan.className = 'font-bold text-sm uppercase';
            nameSpan.textContent = a.asset || '';
            const changeSpan = document.createElement('span');
            const chg = parseFloat(a.change_24h_pct) || 0;
            changeSpan.className = 'text-xs font-medium ' + pnlColor(chg);
            changeSpan.textContent = fmtPct(chg);
            nameRow.appendChild(nameSpan);
            nameRow.appendChild(changeSpan);

            // Price
            const priceDiv = document.createElement('div');
            priceDiv.className = 'text-lg font-bold mb-2';
            priceDiv.textContent = '$' + fmtPrice(parseFloat(a.price) || 0);

            // Regime badge
            const regimeDiv = document.createElement('div');
            regimeDiv.className = 'mb-2';
            if (a.market_regime) {
                const badge = document.createElement('span');
                badge.className = 'px-2 py-0.5 rounded-full text-xs ' + regimeBadge(a.market_regime);
                badge.textContent = (a.market_regime || '').replace(/_/g, ' ');
                regimeDiv.appendChild(badge);
            }

            // Metrics row
            const metricsDiv = document.createElement('div');
            metricsDiv.className = 'grid grid-cols-2 gap-1 text-xs';

            const rsiLabel = document.createElement('span');
            rsiLabel.className = 'text-dark-muted';
            rsiLabel.textContent = 'RSI';
            const rsiVal = document.createElement('span');
            const rsi = parseFloat(a.rsi_14) || 50;
            rsiVal.className = rsi > 70 ? 'text-loss' : rsi < 30 ? 'text-profit' : '';
            rsiVal.textContent = rsi.toFixed(1);

            const sentLabel = document.createElement('span');
            sentLabel.className = 'text-dark-muted';
            sentLabel.textContent = 'Sent.';
            const sentVal = document.createElement('span');
            const sentScore = parseFloat(a.sentiment_score);
            sentVal.style.color = sentimentColor(isNaN(sentScore) ? null : sentScore);
            sentVal.textContent = isNaN(sentScore) ? '-' : sentScore.toFixed(2);

            metricsDiv.appendChild(rsiLabel);
            metricsDiv.appendChild(rsiVal);
            metricsDiv.appendChild(sentLabel);
            metricsDiv.appendChild(sentVal);

            card.appendChild(nameRow);
            card.appendChild(priceDiv);
            card.appendChild(regimeDiv);
            card.appendChild(metricsDiv);
            grid.appendChild(card);
        });
    }

    // ─── PERFORMANCE ANALYTICS ──────────────────────────────

    function showPerfTab(tab) {
        ['strategy', 'assets', 'streaks'].forEach(t => {
            const el = document.getElementById('perf-' + t);
            const btn = document.getElementById('tab-' + t);
            if (t === tab) {
                el.classList.remove('hidden');
                btn.className = 'px-2 py-1 rounded text-xs bg-accent/20 text-accent';
            } else {
                el.classList.add('hidden');
                btn.className = 'px-2 py-1 rounded text-xs bg-dark-border text-dark-muted';
            }
        });
    }

    async function updatePerformance() {
        const data = await fetchJson('/api/performance');
        if (!data || !data.performance) return;
        const p = data.performance;

        document.getElementById('perf-empty').classList.add('hidden');

        // Strategy tab
        const stratEl = document.getElementById('perf-strategy');
        while (stratEl.firstChild) stratEl.removeChild(stratEl.firstChild);
        if (p.strategy) {
            const strat = p.strategy;
            const long = strat.long || {};
            const short = strat.short || {};
            const totalTrades = (long.count || 0) + (short.count || 0);
            const totalPnl = (long.total_pnl || 0) + (short.total_pnl || 0);
            const overallWR = totalTrades > 0 ? ((long.count || 0) * (long.win_rate || 0) + (short.count || 0) * (short.win_rate || 0)) / totalTrades * 100 : 0;
            const grid = document.createElement('div');
            grid.className = 'grid grid-cols-2 md:grid-cols-4 gap-4';

            const items = [
                { label: 'Total Trades', value: totalTrades },
                { label: 'Win Rate', value: fmtPct(overallWR, false) },
                { label: 'Total P&L', value: fmtUsd(totalPnl), cls: totalPnl >= 0 ? 'text-profit' : 'text-loss' },
                { label: 'Best Strategy', value: (strat.best_strategy || '-').toUpperCase() },
                { label: 'Long Win %', value: fmtPct((long.win_rate || 0) * 100, false) },
                { label: 'Short Win %', value: fmtPct((short.win_rate || 0) * 100, false) },
                { label: 'Long Avg P&L', value: fmtUsd(long.avg_pnl || 0), cls: (long.avg_pnl || 0) >= 0 ? 'text-profit' : 'text-loss' },
                { label: 'Short Avg P&L', value: fmtUsd(short.avg_pnl || 0), cls: (short.avg_pnl || 0) >= 0 ? 'text-profit' : 'text-loss' },
            ];
            items.forEach(item => {
                const card = document.createElement('div');
                card.className = 'bg-dark-bg rounded p-3';
                const lbl = document.createElement('div');
                lbl.className = 'text-xs text-dark-muted uppercase mb-1';
                lbl.textContent = item.label;
                const val = document.createElement('div');
                val.className = 'text-lg font-bold ' + (item.cls || '');
                val.textContent = item.value;
                card.appendChild(lbl);
                card.appendChild(val);
                grid.appendChild(card);
            });
            stratEl.appendChild(grid);
        }

        // Assets tab
        const assetEl = document.getElementById('perf-assets');
        while (assetEl.firstChild) assetEl.removeChild(assetEl.firstChild);
        if (p.asset && typeof p.asset === 'object') {
            const tbl = document.createElement('table');
            const thead = document.createElement('thead');
            const headerRow = document.createElement('tr');
            ['Asset', 'Trades', 'Win Rate', 'Total P&L'].forEach(h => {
                const th = document.createElement('th');
                th.textContent = h;
                headerRow.appendChild(th);
            });
            thead.appendChild(headerRow);
            tbl.appendChild(thead);

            const tbody = document.createElement('tbody');
            Object.entries(p.asset).forEach(([asset, stats]) => {
                const tr = document.createElement('tr');
                const tdAsset = document.createElement('td');
                tdAsset.className = 'font-medium uppercase';
                tdAsset.textContent = asset;
                const tdTrades = document.createElement('td');
                tdTrades.textContent = stats.count || 0;
                const tdWR = document.createElement('td');
                tdWR.textContent = fmtPct((stats.win_rate || 0) * 100, false);
                const tdPnl = document.createElement('td');
                const pnl = stats.total_pnl || 0;
                tdPnl.className = pnlColor(pnl);
                tdPnl.textContent = fmtUsd(pnl);
                tr.appendChild(tdAsset);
                tr.appendChild(tdTrades);
                tr.appendChild(tdWR);
                tr.appendChild(tdPnl);
                tbody.appendChild(tr);
            });
            tbl.appendChild(tbody);
            assetEl.appendChild(tbl);
        }

        // Streaks tab
        const streakEl = document.getElementById('perf-streaks');
        while (streakEl.firstChild) streakEl.removeChild(streakEl.firstChild);
        if (p.streaks) {
            const sk = p.streaks;
            const grid = document.createElement('div');
            grid.className = 'grid grid-cols-2 md:grid-cols-3 gap-4';

            const items = [
                { label: 'Current Streak', value: sk.current_streak || 0,
                  cls: (sk.current_streak || 0) > 0 ? 'text-profit' : (sk.current_streak || 0) < 0 ? 'text-loss' : '' },
                { label: 'Best Win Streak', value: sk.longest_win_streak || 0, cls: 'text-profit' },
                { label: 'Worst Loss Streak', value: sk.longest_loss_streak || 0, cls: 'text-loss' },
            ];
            items.forEach(item => {
                const card = document.createElement('div');
                card.className = 'bg-dark-bg rounded p-3';
                const lbl = document.createElement('div');
                lbl.className = 'text-xs text-dark-muted uppercase mb-1';
                lbl.textContent = item.label;
                const val = document.createElement('div');
                val.className = 'text-lg font-bold ' + (item.cls || '');
                val.textContent = item.value;
                card.appendChild(lbl);
                card.appendChild(val);
                grid.appendChild(card);
            });
            streakEl.appendChild(grid);
        }
    }

    // ─── DUAL-SPEED POLLING ─────────────────────────────────

    const FAST_INTERVAL = 15000;  // 15s for real-time data
    const SLOW_INTERVAL = 30000;  // 30s for heavier queries

    async function refreshFast() {
        await Promise.all([
            updateStatus(),
            updateActivity(),
            updateMarket(),
            updatePositions(),
        ]);
        document.getElementById('last-update').textContent =
            'Updated: ' + new Date().toLocaleTimeString();
    }

    async function refreshSlow() {
        await Promise.all([
            updatePortfolio(),
            updateRisk(),
            updateTrades(),
            updateAnalysis(),
            updateRejections(),
            updateEquityChart(),
            updatePerformance(),
        ]);
    }

    async function refreshAll() {
        await Promise.all([refreshFast(), refreshSlow()]);
    }

    // Initial load + dual-speed auto-refresh
    refreshAll();
    setInterval(refreshFast, FAST_INTERVAL);
    setInterval(refreshSlow, SLOW_INTERVAL);
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    """Serve the main dashboard page."""
    return render_template_string(DASHBOARD_HTML)


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("")
    print("=" * 50)
    print("  GROK TRADER -- Crypto Dashboard")
    print(f"  http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
    print(f"  Database: Supabase PostgreSQL")
    print("  Mode: READ-ONLY")
    print("=" * 50)
    print("")
    app.run(
        host=DASHBOARD_HOST,
        port=DASHBOARD_PORT,
        debug=False,
    )
