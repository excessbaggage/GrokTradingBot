# Grok Trader — Claude Code Project Context

## Project Overview
Autonomous perpetual futures trading bot powered by xAI Grok API. Trades 17 perpetual futures (10 majors/L1/L2 + 7 meme coins) on Hyperliquid DEX. Agent name: **Grok Trader** — active intraday trader. Asset universe is fully dynamic — edit `ASSET_UNIVERSE` in `config/trading_config.py` to add/remove assets. Live X/Twitter sentiment analysis via xAI Responses API with built-in x_search feeds real-time social signals into every trading decision.

- **Spec file**: `grok-trading-bot-spec.md` in project root
- **Language**: Python 3.11+
- **Mode**: Paper trading by default (`LIVE_TRADING=False`)
- **Database**: Supabase PostgreSQL via `DATABASE_URL` env var (9 tables: trades, positions, grok_logs, daily_summaries, rejections, equity_snapshots, cycle_events, market_snapshots, performance_cache)
- **GitHub**: `https://github.com/excessbaggage/GrokTradingBot.git`

## Architecture (4 layers)

```
Config:     config/risk_config.py (immutable risk params)
            config/trading_config.py (env-based settings)

Data:       data/database.py (PostgreSQL via psycopg2 + PgConnectionWrapper)
            data/market_data.py (Hyperliquid SDK + technical indicators)
            data/portfolio_state.py (equity computation)
            data/trade_history.py (trade CRUD — all @staticmethod)
            data/context_builder.py (assembles Grok prompt context)
            data/x_sentiment.py (live X/Twitter sentiment via xAI Responses API)
            data/performance_analyzer.py (strategy/asset/streak analysis)
            data/regime_detector.py (market regime classification)
            data/liquidation_estimator.py (liquidation heatmaps)

Brain:      brain/models.py (Pydantic v2 schemas — dynamic asset validation)
            brain/system_prompt.py (dynamic Grok system prompt from ASSET_UNIVERSE)
            brain/grok_client.py (OpenAI SDK + x.ai base_url, 120s timeout)
            brain/decision_parser.py (JSON parsing + validation)

Execution:  execution/risk_guardian.py (13 sequential checks)
            execution/order_manager.py (live + paper modes)
            execution/position_manager.py (position monitoring)
            execution/notifications.py (alerts)

Entry:      main.py (orchestrator loop)
Dashboard:  dashboard.py (Flask read-only web UI on port 5050)
```

## Key Design Decisions

- **AI never executes directly** — Grok proposes, Risk Guardian validates, OrderManager executes
- `OrderManager.__init__()` takes no args — reads config internally
- `PositionManager.__init__(order_manager=None)` — pass OrderManager instance
- `RiskGuardian.validate(decision, portfolio_state_dict, db_connection)` — portfolio is a dict: `{equity, peak_equity, daily_pnl_pct, weekly_pnl_pct, total_exposure_pct}`
- `TradeHistoryManager` methods are `@staticmethod` — pass db connection each time
- Paper mode simulates fills in-memory via `_paper_state` (PaperState dataclass in order_manager.py)
- **Dual state tracking**: Paper positions exist in BOTH the `trades` table AND in-memory `_paper_state` — all close paths must update both

## Running the Bot

```bash
# Start bot (runs perpetual cycle loop)
python3 main.py

# Start dashboard (Flask on port 5050)
python3 dashboard.py

# Both need .env with DATABASE_URL and XAI_API_KEY
```

### Deployment Notes (Mac Mini)
- Clone repo, install deps: `pip3 install -r requirements.txt`
- Copy `.env` file with `DATABASE_URL`, `XAI_API_KEY`
- Run both `main.py` and `dashboard.py` — use `tmux` or `screen` for persistence
- Dashboard reads from the same Supabase DB, so it works from any machine
- Bot writes logs to `logs/bot_YYYY-MM-DD.log` and `logs/errors_YYYY-MM-DD.log`
- Some Hyperliquid testnet assets fail (LINK, PEPE, SHIB, BONK, FLOKI) — this is expected, bot handles it gracefully

## Running Tests

```bash
python -m pytest tests/ -v --tb=short
```

- **477 tests** across multiple files, all pass in ~2s
- Tests use in-memory SQLite and mocks — **no API keys needed**

## Dependencies

```
hyperliquid-python-sdk, openai, pydantic>=2.0, loguru, tenacity, pandas, numpy, APScheduler, python-dotenv, flask, psycopg2-binary
```

## Environment Variables (.env)

```
XAI_API_KEY=           # Required: xAI/Grok API key (also used for X sentiment)
GROK_MODEL=grok-4-1-fast-reasoning  # Grok model for trading decisions
DATABASE_URL=          # Required: Supabase PostgreSQL connection string
LIVE_TRADING=False     # NEVER set True without reading the spec
STARTING_CAPITAL=10000 # Paper trading starting capital
CYCLE_INTERVAL_MINUTES=5  # Minutes between trading cycles (default 5, min 5, max 30)
HYPERLIQUID_WALLET_ADDRESS=  # Required for live mode only
HYPERLIQUID_PRIVATE_KEY=     # Required for live mode only
X_SENTIMENT_ENABLED=True     # Enable/disable live X sentiment fetching
X_SENTIMENT_MODEL=grok-4-1-fast-reasoning  # Model for sentiment analysis
```

## Current State (March 8, 2026)

### Model & API
- **Trading model**: `grok-4-1-fast-reasoning` (Chat Completions API via OpenAI SDK)
- **Sentiment model**: `grok-4-1-fast-reasoning` (Responses API with built-in `x_search` tool)
- **Cycle interval**: 5 minutes (lowered from 15 for faster data collection)
- **Equity**: ~$9,992 (started at $10,000)

### What's Working
- Full trading cycle: market data → X sentiment (17/17 assets) → Grok reasoning → risk check → paper execution
- Dashboard at port 5050 with: equity chart, top stats row (equity, win rate, invested, P&L, drawdown, fees), recent trades, open positions, activity feed, market overview, risk metrics, performance analytics, Grok analysis
- Position sync with exchange on startup (closes stale DB positions)
- Regime detection (trending up/down, mean reverting, volatile expansion)
- Liquidation heatmap estimation
- Performance analyzer with strategy/asset/streak breakdowns

### Known Issues
- Testnet assets LINK, PEPE, SHIB, BONK, FLOKI fail on Hyperliquid testnet (delisted) — bot logs warnings and continues
- Win rate is 0% with very small sample (need 50+ trades to be meaningful)
- No calibration analysis yet (measuring Grok prediction accuracy vs outcomes)

## Recent Work Log (March 7-8, 2026)

### Session 1 (March 7)
- Switched model from `grok-3-fast` to `grok-4-1-fast-reasoning`
- Rewrote `x_sentiment.py` from Chat Completions API to Responses API with `x_search` tool
- Fixed numpy float64 → PostgreSQL serialization in main.py
- Added `db.rollback()` to helper functions
- Updated tests for Responses API format
- All 477 tests passing

### Session 2 (March 8)
- Fixed dashboard unicode escape crash (`\u{XXXX}` JS syntax inside Python template → `String.fromCodePoint()`)
- Moved Recent Trades + Open Positions side-by-side below equity chart
- Ran 4-agent parallel code review (code-reviewer, silent-failure-hunter, code-explorer, comment-analyzer)
- Fixed 3 confirmed JS field name mismatches in performance analytics:
  - Strategy tab: flat `strat.total_trades` → nested `strat.long.count + strat.short.count`
  - Assets tab: `stats.win_rate_pct` → `stats.win_rate * 100`
  - Streaks tab: `sk.best_win_streak` → `sk.longest_win_streak`
- Added `db.rollback()` to `_log_grok_interaction`, `_log_rejection`, `_save_equity_snapshot`
- Fixed stale comments (grok_client timeout 30→120s, model example, SQLite→PostgreSQL docstring)
- Updated CLAUDE.md references from "Agent Tools API" to "Responses API"
- Added win rate card to dashboard top stats row
- Renamed bot from "Sentinel" to "Grok Trader"
- Lowered cycle interval from 15min to 5min for faster data collection
- All 477 tests passing

### Performance Analyzer Data Structures (for dashboard JS)
These are returned by `data/performance_analyzer.py` — dashboard JS must match:
- `_compute_trade_stats()` → `{"win_rate": 0.6, "avg_pnl": 50.0, "avg_pnl_pct": 0.05, "total_pnl": 500.0, "count": 10}`
- `get_strategy_performance()` → `{"long": {stats}, "short": {stats}, "best_strategy": "long"}`
- `get_asset_performance()` → `{"BTC": {stats}, "ETH": {stats}, "best_asset": "BTC", "worst_asset": "ETH"}`
- `get_streak_analysis()` → `{"current_streak": 3, "longest_win_streak": 5, "longest_loss_streak": 2, "total_trades": 15}`

## Next Steps

- [ ] Collect 50+ trades at 5-min intervals to get meaningful statistics
- [ ] Analyze why initial trades lost (directional read? stop loss too tight? timing?)
- [ ] Add Grok prediction calibration (conviction vs actual win rate)
- [ ] Consider Kelly criterion position sizing once sample size is sufficient
- [ ] Add trailing stop-loss support
- [ ] Phase 3: Switch to mainnet with small capital ($500-1000)
- [ ] WebSocket for real-time dashboard updates

## Known Remaining MEDIUM Issues (lower priority)

- `market_data.py` — Bollinger band period is hardcoded to 20; consider making configurable
- `order_manager.py` — SL/TP order check loop is O(n) per asset; fine for small portfolio
- `dashboard.py` — Equity chart query has no LIMIT; will slow down after thousands of cycles
- `main.py` — Daily summary can fire multiple times if cycle spans midnight
- Silent `except: pass` blocks in some rollback handlers (logged but swallowed)
