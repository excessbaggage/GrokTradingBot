"""
Sentinel Trading Bot — Web Monitoring Dashboard

A read-only Flask dashboard that connects to the bot's SQLite database
and presents real-time trading status, portfolio metrics, positions,
trade history, and Grok's analysis in a dark-themed trading terminal UI.

Run alongside the bot as a separate process:
    python dashboard.py

Then open http://localhost:5050 in your browser.

The dashboard opens the database in read-only mode (?mode=ro), so it
can never interfere with the bot's writes.  SQLite WAL journal mode
(already enabled by the bot) allows concurrent readers without blocking.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone

from flask import Flask, jsonify, render_template_string

from config.trading_config import (
    ASSET_UNIVERSE,
    DB_PATH,
    LIVE_TRADING,
    STARTING_CAPITAL,
)

# ═══════════════════════════════════════════════════════════════════════════
# APP SETUP
# ═══════════════════════════════════════════════════════════════════════════

app = Flask(__name__)

DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "5050"))
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "127.0.0.1")


def get_ro_db() -> sqlite3.Connection:
    """Open a read-only connection to the bot's database."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict]:
    """Convert sqlite3.Row objects to plain dicts for JSON serialization."""
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
        db.close()

        if row:
            return jsonify({
                "online": True,
                "last_cycle_time": row["timestamp"],
                "cycle_number": row["cycle_number"],
                "mode": "LIVE" if LIVE_TRADING else "PAPER",
                "assets": ASSET_UNIVERSE,
            })
        return jsonify({
            "online": False,
            "last_cycle_time": None,
            "cycle_number": 0,
            "mode": "LIVE" if LIVE_TRADING else "PAPER",
            "assets": ASSET_UNIVERSE,
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

        # --- Daily realized P&L ---
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        daily_pnl_row = db.execute(
            """SELECT COALESCE(SUM(pnl), 0) AS total_pnl
               FROM trades WHERE status = 'closed' AND date(closed_at) = ?""",
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

        db.close()

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
            "daily_pnl": round(daily_pnl, 2),
            "daily_pnl_pct": round(daily_pnl_pct, 4),
            "weekly_pnl": round(weekly_pnl, 2),
            "weekly_pnl_pct": round(weekly_pnl_pct, 4),
            "peak_equity": round(peak, 2),
            "drawdown_pct": round(drawdown, 4),
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
               WHERE date(opened_at) = ?""",
            (today_str,),
        ).fetchone()

        # Total rejection count
        rejection_count_row = db.execute(
            "SELECT COUNT(*) AS cnt FROM rejections"
        ).fetchone()

        db.close()

        return jsonify({
            "total_exposure_pct": total_exposure * 100,
            "open_position_count": len(positions),
            "consecutive_losses": consecutive_losses,
            "trades_today": trade_count_row["cnt"] if trade_count_row else 0,
            "total_rejections": rejection_count_row["cnt"] if rejection_count_row else 0,
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
        db.close()

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
        db.close()
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
        db.close()

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
        db.close()
        return jsonify({"rejections": rows_to_dicts(rows)})
    except Exception as e:
        app.logger.error("API error: %s", e)
        return jsonify({"error": "Internal server error"}), 500


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
            db.close()
            return jsonify({"equity_curve": data})

        # Fall back to daily summaries
        rows = db.execute(
            """SELECT date, ending_equity, pnl, pnl_pct, max_drawdown
               FROM daily_summaries ORDER BY date ASC"""
        ).fetchall()
        db.close()

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
    <title>Sentinel — Trading Dashboard</title>
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
                <span class="text-accent">SENTINEL</span>
                <span class="text-dark-muted ml-2 text-sm font-normal">Trading Dashboard</span>
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
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div class="card p-4">
                <div class="text-dark-muted text-xs uppercase tracking-wide mb-1">Equity</div>
                <div id="equity" class="text-2xl font-bold">$0.00</div>
                <div id="total-return" class="text-sm text-dark-muted">+0.00%</div>
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
        </div>

        <!-- EQUITY CHART -->
        <div class="card p-4">
            <div class="text-dark-muted text-xs uppercase tracking-wide mb-3">Equity Curve</div>
            <div style="height: 250px;">
                <canvas id="equityChart"></canvas>
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

        <!-- POSITIONS + GROK ANALYSIS -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">

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

            <!-- GROK ANALYSIS -->
            <div class="card p-4">
                <div class="text-dark-muted text-xs uppercase tracking-wide mb-3">Latest Grok Analysis</div>
                <div id="grok-analysis" class="text-sm space-y-3">
                    <div class="text-dark-muted">Waiting for first cycle...</div>
                </div>
            </div>
        </div>

        <!-- RECENT TRADES -->
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
        Sentinel &mdash; Grok-Powered Autonomous Trader &mdash; Read-only dashboard (auto-refreshes every 30s)
    </footer>

    <script>
    // ─── GLOBALS ─────────────────────────────────────────────
    let equityChart = null;
    const REFRESH_INTERVAL = 30000; // 30 seconds

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

    // ─── REFRESH ALL ─────────────────────────────────────────

    async function refreshAll() {
        await Promise.all([
            updateStatus(),
            updatePortfolio(),
            updateRisk(),
            updatePositions(),
            updateTrades(),
            updateAnalysis(),
            updateRejections(),
            updateEquityChart(),
        ]);
        document.getElementById('last-update').textContent =
            'Updated: ' + new Date().toLocaleTimeString();
    }

    // Initial load + auto-refresh
    refreshAll();
    setInterval(refreshAll, REFRESH_INTERVAL);
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
    print("  SENTINEL -- Trading Dashboard")
    print(f"  http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
    print(f"  Database: {DB_PATH}")
    print("  Mode: READ-ONLY")
    print("=" * 50)
    print("")
    app.run(
        host=DASHBOARD_HOST,
        port=DASHBOARD_PORT,
        debug=False,
    )
