# Sentinel Trading Bot — Claude Code Project Context

## Project Overview
Autonomous perpetual futures trading bot powered by xAI Grok API. Trades BTC/ETH/SOL on Hyperliquid DEX. Agent name: **Sentinel** — conservative swing trader.

- **Spec file**: `grok-trading-bot-spec.md` in project root
- **Language**: Python 3.11+
- **Mode**: Paper trading by default (`LIVE_TRADING=False`)
- **Database**: SQLite at `db/trading_bot.db` (5 tables: trades, positions, grok_logs, daily_summaries, rejections + equity_snapshots)

## Architecture (4 layers)

```
Config:     config/risk_config.py (immutable risk params)
            config/trading_config.py (env-based settings)

Data:       data/database.py (SQLite schema + init)
            data/market_data.py (Hyperliquid SDK + technical indicators)
            data/portfolio_state.py (equity computation)
            data/trade_history.py (trade CRUD — all @staticmethod)
            data/context_builder.py (assembles Grok prompt context)

Brain:      brain/models.py (Pydantic v2 schemas)
            brain/system_prompt.py (Grok system prompt)
            brain/grok_client.py (OpenAI SDK + x.ai base_url)
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
- **Dual state tracking**: Paper positions exist in BOTH SQLite `trades` table AND in-memory `_paper_state` — all close paths must update both

## Running Tests

```bash
python -m pytest tests/ -v --tb=short
```

- **126 tests** across 3 files, all pass in <1s
- `test_risk_guardian.py` — 32 tests covering all 13 risk checks
- `test_decision_parser.py` — 20 tests for JSON parsing + Pydantic validation
- `test_integration.py` — 74 integration tests (full cycle simulation, DB roundtrip, error recovery, etc.)
- Tests use in-memory SQLite and mocks — **no API keys needed**

## Dependencies

```
hyperliquid-python-sdk, openai, pydantic>=2.0, loguru, tenacity, pandas, numpy, APScheduler, python-dotenv, flask
```

## Environment Variables (.env)

```
XAI_API_KEY=           # Required: xAI/Grok API key
GROK_MODEL=grok-3-mini # Grok model to use
LIVE_TRADING=False     # NEVER set True without reading the spec
STARTING_CAPITAL=1000  # Paper trading starting capital
CYCLE_INTERVAL_MINUTES=15
HYPERLIQUID_WALLET_ADDRESS=  # Required for live mode only
HYPERLIQUID_PRIVATE_KEY=     # Required for live mode only
```

## Recent Work (March 2026)

### Comprehensive Code Review & Bug Fix (commit bc1cbaf)
Three parallel review agents analyzed the entire codebase and found 33 issues (7 CRITICAL, 13 HIGH, 13 MEDIUM). All CRITICAL and HIGH issues were fixed:

**Core logic fixes:**
| Fix | File | What |
|-----|------|------|
| RSI flat market | `data/market_data.py` | Returns 50 (neutral) instead of 100 (overbought) when `avg_gain == avg_loss == 0` |
| Adaptive RSI inversion guard | `data/market_data.py` | Resets to defaults (70/30) if overbought threshold drops below oversold |
| Connection leak | `data/portfolio_state.py` | Added `finally: db.close()` to `_compute_paper_equity` |
| Paper position overwrite | `execution/order_manager.py` | Warns and cancels old SL/TP before overwriting existing position |
| DB/memory desync | `execution/position_manager.py` | `_check_holding_period` and `_detect_closed_positions` now update both DB and `_paper_state` |
| NULL pnl on close | `execution/position_manager.py` | Computes `pnl_pct * size_pct * STARTING_CAPITAL * leverage` on every trade closure |
| Equity snapshot | `main.py` | Fixed broken function signature (was expecting `portfolio_mgr`, now accepts `portfolio` dict) |
| Close action logging | `main.py` | Close trades now log actual side from order result instead of always "short" |

**Risk & validation fixes:**
| Fix | File | What |
|-----|------|------|
| TP direction check | `execution/risk_guardian.py` | Validates take-profit is above entry for longs, below for shorts (was using `abs()` masking wrong-side TP) |
| size_pct normalization | `brain/decision_parser.py` | Changed `> 1.0` to `>= 1.0` since exactly 1.0 always means "100%", not "1%" |

**Security fixes (dashboard.py):**
| Fix | What |
|-----|------|
| XSS prevention | Added `escapeHtml()` to all `innerHTML` insertions rendering LLM/DB content |
| Error disclosure | Replaced `str(e)` in API responses with generic "Internal server error", logged actual error server-side |
| NULL pnl crash | `/api/risk` consecutive losses loop now handles NULL pnl gracefully |
| Weekly P&L | Replaced placeholder `weekly_pnl_pct = daily_pnl_pct` with actual weekly realized P&L query |

## Next Steps (from spec — Phase 2+)

- [ ] Wire up testnet paper trading and run for 2 weeks
- [ ] Add daily summary duplicate guard (main.py — prevent re-sending if already sent today)
- [ ] Add trailing stop-loss support
- [ ] Add X/Twitter sentiment analysis via Grok
- [ ] Backtesting framework
- [ ] Web dashboard improvements (WebSocket for real-time updates)
- [ ] Phase 3: Switch to mainnet with small capital ($500-1000)

## Known Remaining MEDIUM Issues (lower priority)

- `market_data.py` — Bollinger band period is hardcoded to 20; consider making configurable
- `order_manager.py` — SL/TP order check loop is O(n) per asset; fine for small portfolio, may need indexing later
- `dashboard.py` — Equity chart query has no LIMIT; will slow down after thousands of cycles (add pagination)
- `main.py` — Daily summary can fire multiple times if cycle spans midnight (add sent-today flag)

## Git Workflow (for future changes)

```bash
# Create feature branch
git checkout -b fix/description-of-change

# Make changes, run tests
python -m pytest tests/ -v

# Commit and push
git add <files>
git commit -m "Fix: description"
git push -u origin fix/description-of-change

# Create PR via GitHub CLI
gh pr create --title "Fix: description" --body "Summary of changes"
```
