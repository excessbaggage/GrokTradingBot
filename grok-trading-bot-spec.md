# Grok-Powered Perpetual Trading Bot — Build Specification

> **Purpose:** This document is a complete specification for Claude Code (Opus 4.6) to build an autonomous perpetual futures trading bot powered by the xAI Grok API. The bot trades with real capital on Hyperliquid DEX. Every design decision prioritizes capital preservation first, alpha generation second.

---

## 1. Project Overview

Build a Python-based autonomous trading agent that:

- Uses the **xAI Grok API** (model: `grok-4` or `grok-4-1-fast-reasoning`) as its decision-making engine
- Executes **perpetual futures trades** on **Hyperliquid DEX** using their Python SDK
- Runs on a scheduled loop (configurable interval, default every 15 minutes)
- Maintains a persistent state/memory of positions, trade history, and market context
- Enforces hard-coded risk management guardrails that **cannot be overridden by the AI**
- Logs every decision, rationale, and trade for full auditability
- Sends notifications (Discord webhook or Telegram) on trades and daily P&L summaries

### Target Assets (Initial Universe)
- BTC-USD perpetual
- ETH-USD perpetual
- SOL-USD perpetual

Expand the universe later via configuration, not code changes.

---

## 2. Agent Profile & Persona

The Grok trading agent operates under a defined persona and set of constraints embedded in its system prompt. This is the "personality" of the bot.

### Agent Name: **Sentinel**

### Agent Philosophy
- **Style:** Swing trader with 4-hour to multi-day holding periods. NOT a scalper.
- **Edge:** Sentiment arbitrage via X/Twitter data (Grok's native advantage), combined with technical structure and macro awareness.
- **Temperament:** Patient, disciplined, contrarian when conviction is high. Willing to sit in cash. Flat is a position.
- **Risk Identity:** Conservative. The bot treats drawdown avoidance as more important than upside capture. Inspired by risk parity and trend-following principles, not YOLO momentum.

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR (Python)                 │
│                   (main loop / scheduler)                │
├─────────────┬──────────────┬────────────────────────────┤
│  DATA LAYER │  BRAIN LAYER │  EXECUTION LAYER           │
│             │              │                            │
│ • Market    │ • Grok API   │ • Hyperliquid SDK          │
│   data feed │   (decision  │   (order placement,        │
│ • Portfolio │    engine)   │    position mgmt)          │
│   state     │ • System     │ • Risk guardian            │
│ • Trade     │   prompt +   │   (hard limits, cannot     │
│   history   │   context    │    be overridden by AI)    │
│ • Logs      │   assembly   │ • Notification service     │
│             │              │                            │
└─────────────┴──────────────┴────────────────────────────┘
```

### Component Breakdown

#### 3.1 Data Layer (`data/`)
- **Market Data Module:** Fetch OHLCV candles (1h, 4h, 1d), order book snapshots, funding rates, and open interest from Hyperliquid's API.
- **Portfolio State Module:** Track current positions, unrealized P&L, available margin, entry prices, and position sizes. Persist to a local SQLite database.
- **Trade History Module:** Log every trade with timestamp, asset, side, size, price, fees, Grok's reasoning text, and the full prompt/response pair. Store in SQLite.
- **Context Builder Module:** Assemble the market data, portfolio state, and recent trade history into a structured context payload for Grok.

#### 3.2 Brain Layer (`brain/`)
- **Grok Client Module:** Wrapper around the xAI API (using OpenAI SDK with `base_url="https://api.x.ai/v1"`). Handles authentication, retries, structured output parsing.
- **System Prompt:** The static persona and rules (see Section 5 below). Never changes at runtime.
- **Context Prompt:** Dynamic market data and portfolio state injected each cycle.
- **Decision Parser:** Grok returns structured JSON (enforced via system prompt). Parse into a `TradeDecision` dataclass.

#### 3.3 Execution Layer (`execution/`)
- **Risk Guardian Module:** Hard-coded Python guardrails (see Section 4). Validates every trade decision BEFORE execution. Can reject or modify Grok's decisions. The AI never talks directly to the exchange.
- **Order Manager Module:** Translates validated decisions into Hyperliquid API calls. Handles order types (limit, market), stop-losses, take-profits.
- **Notification Module:** Posts trade alerts and daily summaries to Discord webhook or Telegram bot.

---

## 4. Risk Management Guardrails (Hard-Coded — NOT AI-Controlled)

These are Python-enforced limits. Grok can suggest whatever it wants — these gates cannot be bypassed.

```python
# risk_config.py — THESE VALUES ARE THE FINAL AUTHORITY

RISK_PARAMS = {
    # === Position Sizing ===
    "max_position_size_pct": 0.10,        # Max 10% of portfolio in any single position
    "max_total_exposure_pct": 0.30,        # Max 30% of portfolio across ALL positions combined
    "max_leverage": 3.0,                   # Hard cap at 3x leverage (conservative)
    
    # === Loss Limits ===
    "max_loss_per_trade_pct": 0.02,        # Stop-loss: max 2% portfolio loss per trade
    "max_daily_loss_pct": 0.05,            # If daily drawdown hits 5%, halt ALL trading until next day
    "max_weekly_loss_pct": 0.10,           # If weekly drawdown hits 10%, halt trading for 48 hours
    "max_total_drawdown_pct": 0.20,        # If total drawdown from peak hits 20%, KILL SWITCH — halt indefinitely, notify owner
    
    # === Trade Frequency ===
    "min_time_between_trades_minutes": 30,  # No rapid-fire trading
    "max_trades_per_day": 8,                # Cap daily trade count
    
    # === Mandatory Risk Controls ===
    "require_stop_loss": True,              # Every position MUST have a stop-loss
    "require_take_profit": True,            # Every position MUST have a take-profit
    "max_stop_loss_distance_pct": 0.05,     # Stop can't be more than 5% from entry
    "min_risk_reward_ratio": 1.5,           # Minimum 1.5:1 reward-to-risk
    
    # === Kill Switch ===
    "kill_switch_enabled": False,           # Manual override to halt all trading
}
```

### Risk Guardian Logic (Pseudocode)

```
FOR each TradeDecision from Grok:
    1. CHECK kill_switch — if enabled, reject ALL trades
    2. CHECK daily_loss_limit — if breached, reject and log
    3. CHECK weekly_loss_limit — if breached, reject and log
    4. CHECK total_drawdown — if breached, KILL SWITCH ON, notify owner
    5. VALIDATE position_size <= max_position_size_pct
    6. VALIDATE total_exposure + new_position <= max_total_exposure_pct
    7. VALIDATE leverage <= max_leverage
    8. VALIDATE stop_loss exists and is within max_stop_loss_distance_pct
    9. VALIDATE take_profit exists
    10. VALIDATE risk_reward_ratio >= min_risk_reward_ratio
    11. VALIDATE time_since_last_trade >= min_time_between_trades
    12. VALIDATE daily_trade_count < max_trades_per_day
    
    IF all checks pass → EXECUTE
    ELSE → LOG rejection reason, notify owner, skip trade
```

---

## 5. Grok System Prompt (The Agent's Brain)

This is the static system prompt sent to Grok on every API call. It defines WHO the agent is, HOW it thinks, and WHAT format it responds in.

```
You are Sentinel, an autonomous perpetual futures trading agent. You analyze market data, 
sentiment, and portfolio state to make disciplined trading decisions. You trade BTC, ETH, 
and SOL perpetual futures on Hyperliquid.

## YOUR TRADING PHILOSOPHY

1. CAPITAL PRESERVATION IS YOUR PRIMARY OBJECTIVE. You would rather miss a trade than 
   take a bad one. Being flat (no position) is a valid and often optimal decision.

2. You are a swing trader. Your ideal holding period is 4 hours to 5 days. You do NOT 
   scalp. You do NOT chase pumps. You wait for high-conviction setups.

3. Your edge comes from three sources:
   - SENTIMENT: You have unique awareness of social/X sentiment and narrative shifts. 
     Use this to identify when retail is overleveraged, when narratives are shifting, 
     or when fear/greed is at extremes.
   - TECHNICAL STRUCTURE: Support/resistance levels, trend direction on 4h and daily 
     timeframes, funding rate extremes, open interest shifts.
   - MACRO AWARENESS: Fed policy, dollar strength, risk-on/risk-off regime, correlation 
     with equities.

4. You are naturally contrarian. When everyone is euphoric, you look for shorts. When 
   everyone is terrified, you look for longs. But you ONLY trade against the crowd 
   when the data supports it — contrarian for its own sake is not a strategy.

5. You size positions based on conviction level:
   - LOW conviction: Skip the trade entirely
   - MEDIUM conviction: 3-5% of portfolio
   - HIGH conviction: 5-10% of portfolio
   - You NEVER go all-in. You NEVER exceed 10% on a single position.

6. You always define your stop-loss and take-profit BEFORE entering. Risk/reward must 
   be at minimum 1.5:1. If you can't find a clean stop level, you don't take the trade.

7. You track your own performance honestly. If you've had 3+ consecutive losing trades, 
   you reduce position sizes by 50% until you get a winner. If your weekly P&L is negative, 
   you become even more selective.

## YOUR RESPONSE FORMAT

You MUST respond with ONLY a JSON object. No markdown, no explanation outside the JSON. 
The orchestrator parses your response programmatically.

{
  "timestamp": "ISO 8601 timestamp",
  "market_analysis": {
    "btc": {
      "bias": "long|short|neutral",
      "conviction": "none|low|medium|high",
      "key_levels": {"support": float, "resistance": float},
      "sentiment_read": "string — your read on current X/social sentiment",
      "funding_rate_signal": "string — is funding elevated/depressed and what it implies",
      "summary": "string — 2-3 sentence analysis"
    },
    "eth": { ...same structure... },
    "sol": { ...same structure... }
  },
  "portfolio_assessment": {
    "current_risk_level": "low|moderate|elevated|high",
    "recent_performance_note": "string — brief assessment of recent trades",
    "suggested_exposure_adjustment": "increase|maintain|decrease"
  },
  "decisions": [
    {
      "action": "open_long|open_short|close|adjust_stop|hold|no_trade",
      "asset": "BTC|ETH|SOL",
      "size_pct": float,          // Percentage of portfolio (0.0 to 0.10)
      "leverage": float,          // 1.0 to 3.0
      "entry_price": float|null,  // null for market orders
      "stop_loss": float,
      "take_profit": float,
      "order_type": "market|limit",
      "reasoning": "string — 2-3 sentences explaining WHY this trade, what's the edge",
      "conviction": "medium|high",
      "risk_reward_ratio": float
    }
  ],
  "overall_stance": "string — one sentence: what is your overall market read right now?",
  "next_review_suggestion_minutes": int  // When should the orchestrator call you again?
}

If you have NO trades to make, return an empty "decisions" array and explain in 
"overall_stance" why you're staying flat. Staying flat is GOOD. Do not force trades.

## CRITICAL RULES

- NEVER suggest leverage above 3x
- NEVER suggest a position larger than 10% of portfolio
- NEVER suggest a trade without a stop-loss
- NEVER chase a move that has already happened
- If the data is unclear or conflicting, your answer is NO TRADE
- You are NOT trying to be right on every trade. You are trying to have a positive 
  expected value over hundreds of trades.
```

---

## 6. Context Prompt Template (Dynamic — Assembled Each Cycle)

This is what gets sent as the `user` message each cycle, populated with live data:

```
## CURRENT MARKET DATA (as of {timestamp})

### BTC-USD Perpetual
- Price: ${btc_price}
- 24h Change: {btc_24h_change}%
- 4h Candles (last 20): {btc_4h_candles_summary}
- Daily Candles (last 10): {btc_daily_candles_summary}
- Funding Rate (current): {btc_funding_rate}% (8h)
- Funding Rate (avg 7d): {btc_funding_7d_avg}%
- Open Interest: ${btc_oi} ({btc_oi_24h_change}% 24h change)
- Long/Short Ratio: {btc_ls_ratio}

### ETH-USD Perpetual
{...same structure...}

### SOL-USD Perpetual
{...same structure...}

## YOUR CURRENT PORTFOLIO
- Total Equity: ${total_equity}
- Available Margin: ${available_margin}
- Unrealized P&L: ${unrealized_pnl} ({unrealized_pnl_pct}%)
- Daily P&L: ${daily_pnl} ({daily_pnl_pct}%)
- Weekly P&L: ${weekly_pnl} ({weekly_pnl_pct}%)

### Open Positions
{position_table_or_none}

### Recent Trades (Last 10)
{recent_trades_table}

### Recent Losing Streak Count: {consecutive_losses}

## RISK STATUS
- Daily loss limit remaining: ${daily_loss_remaining}
- Weekly loss limit remaining: ${weekly_loss_remaining}
- Drawdown from peak: {drawdown_from_peak}%
- Trades today: {trades_today}/{max_trades_per_day}

## YOUR TASK
Analyze the above data. Provide your market analysis, portfolio assessment, and 
trading decisions in the required JSON format. If no high-conviction setups exist, 
return an empty decisions array. DO NOT FORCE TRADES.
```

---

## 7. Technology Stack & Dependencies

```
# requirements.txt
xai-sdk>=0.1.0              # or use openai SDK with base_url override
hyperliquid-python-sdk       # Hyperliquid DEX Python SDK
ccxt                         # Backup/alternative exchange abstraction
pandas                       # Data manipulation for candle analysis
numpy                        # Numerical operations
sqlite3                      # Built into Python — trade log database
python-dotenv                # Environment variable management
schedule                     # Lightweight task scheduler (or APScheduler)
requests                     # HTTP requests for notifications
loguru                       # Structured logging
pydantic                     # Data validation for trade decisions
tenacity                     # Retry logic for API calls
```

### Environment Variables Required
```
XAI_API_KEY=your_xai_api_key
HYPERLIQUID_PRIVATE_KEY=your_wallet_private_key
HYPERLIQUID_WALLET_ADDRESS=your_wallet_address
DISCORD_WEBHOOK_URL=your_discord_webhook (optional)
TELEGRAM_BOT_TOKEN=your_telegram_token (optional)
TELEGRAM_CHAT_ID=your_chat_id (optional)
STARTING_CAPITAL=10000
```

---

## 8. File Structure

```
grok-trading-bot/
├── main.py                    # Entry point — orchestrator loop
├── config/
│   ├── risk_config.py         # Hard-coded risk parameters (Section 4)
│   ├── trading_config.py      # Asset universe, intervals, API settings
│   └── .env                   # API keys (gitignored)
├── data/
│   ├── market_data.py         # Fetch OHLCV, funding, OI from Hyperliquid
│   ├── portfolio_state.py     # Track positions, equity, P&L
│   ├── trade_history.py       # SQLite trade log CRUD
│   ├── context_builder.py     # Assemble the dynamic prompt
│   └── database.py            # SQLite initialization and helpers
├── brain/
│   ├── grok_client.py         # xAI API wrapper
│   ├── system_prompt.py       # Static system prompt (Section 5)
│   ├── decision_parser.py     # Parse + validate Grok's JSON response
│   └── models.py              # Pydantic models: TradeDecision, MarketAnalysis, etc.
├── execution/
│   ├── risk_guardian.py        # Hard-coded risk checks (Section 4 logic)
│   ├── order_manager.py       # Hyperliquid order placement
│   ├── position_manager.py    # Monitor open positions, trailing stops
│   └── notifications.py       # Discord/Telegram alerts
├── utils/
│   ├── logger.py              # Loguru configuration
│   └── helpers.py             # Formatting, math utilities
├── tests/
│   ├── test_risk_guardian.py   # Unit tests for risk checks
│   ├── test_decision_parser.py # Unit tests for JSON parsing
│   └── test_paper_mode.py     # Paper trading integration test
├── logs/                       # Runtime logs directory
├── db/                         # SQLite database directory
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 9. Main Orchestrator Loop (Pseudocode)

```python
def main_loop():
    """Core trading loop — runs every CYCLE_INTERVAL minutes."""
    
    # 1. Initialize
    db = initialize_database()
    grok = GrokClient(api_key=XAI_API_KEY, model="grok-4")
    exchange = HyperliquidClient(private_key=HL_PRIVATE_KEY)
    risk = RiskGuardian(config=RISK_PARAMS)
    notifier = Notifier(discord_url=DISCORD_WEBHOOK_URL)
    
    while True:
        try:
            # 2. Check kill switch
            if risk.kill_switch_active():
                logger.warning("Kill switch active. Sleeping.")
                sleep(300)
                continue
            
            # 3. Gather data
            market_data = fetch_market_data(exchange, assets=ASSET_UNIVERSE)
            portfolio = fetch_portfolio_state(exchange, db)
            recent_trades = fetch_recent_trades(db, limit=10)
            risk_status = risk.calculate_risk_status(portfolio)
            
            # 4. Build context prompt
            context = build_context_prompt(market_data, portfolio, recent_trades, risk_status)
            
            # 5. Query Grok
            grok_response = grok.get_trading_decision(
                system_prompt=SYSTEM_PROMPT,
                context=context
            )
            
            # 6. Parse response
            decisions = parse_decisions(grok_response)
            
            # 7. Log the full prompt/response pair
            log_grok_interaction(db, context, grok_response)
            
            # 8. Validate each decision through Risk Guardian
            for decision in decisions:
                validation = risk.validate(decision, portfolio)
                
                if validation.approved:
                    # 9. Execute trade
                    order_result = exchange.place_order(decision)
                    log_trade(db, decision, order_result)
                    notifier.send_trade_alert(decision, order_result)
                else:
                    logger.info(f"Trade REJECTED: {validation.reason}")
                    log_rejection(db, decision, validation.reason)
            
            # 10. Check and manage existing positions
            manage_open_positions(exchange, risk, db)
            
            # 11. Daily summary check
            if is_daily_summary_time():
                summary = generate_daily_summary(db, portfolio)
                notifier.send_daily_summary(summary)
            
            # 12. Sleep until next cycle
            next_review = decisions.get("next_review_suggestion_minutes", 15)
            sleep(max(next_review, 5) * 60)  # Minimum 5 minutes between cycles
            
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            notifier.send_error_alert(str(e))
            sleep(60)
```

---

## 10. Hyperliquid Integration Notes

Hyperliquid is a decentralized perpetual futures exchange. Key implementation details:

- **Authentication:** Uses an Ethereum wallet private key to sign transactions. No traditional API key.
- **SDK:** `hyperliquid-python-sdk` — install from PyPI or GitHub.
- **Order Types:** Market, limit, stop-market, stop-limit, take-profit.
- **Leverage:** Set per-asset. Our bot caps at 3x but Hyperliquid allows up to 50x.
- **Funding:** 8-hour funding rate cycle. The bot should track this as a signal AND as a cost.
- **Fees:** Maker ~0.01%, Taker ~0.035%. The bot should factor fees into R:R calculations.
- **Testnet:** Hyperliquid has a testnet at `https://app.hyperliquid-testnet.xyz`. **USE TESTNET FIRST for at least 2 weeks of paper trading before going live.**

### Alternative: If the user prefers centralized exchanges
- **Bybit** via CCXT — well-documented perpetual futures API
- **Binance Futures** via CCXT — most liquid perpetuals
- Both can be swapped in by changing the execution layer only.

---

## 11. Implementation Phases

### Phase 1: Foundation (Week 1)
- [ ] Set up project structure and dependencies
- [ ] Implement Grok API client with retry logic
- [ ] Implement market data fetching from Hyperliquid
- [ ] Build SQLite database schema (trades, positions, grok_logs, daily_summaries)
- [ ] Implement context builder
- [ ] Build decision parser with Pydantic validation
- [ ] Implement Risk Guardian with ALL checks from Section 4

### Phase 2: Paper Trading (Weeks 2-3)
- [ ] Implement order manager (Hyperliquid testnet)
- [ ] Implement position manager (monitor, trailing stops)
- [ ] Wire up the full orchestrator loop
- [ ] Implement notification system
- [ ] Run on Hyperliquid TESTNET with simulated capital
- [ ] Log and review every Grok decision for quality
- [ ] Tune system prompt based on observed behavior

### Phase 3: Go Live (Week 4)
- [ ] Review 2 weeks of paper trading results
- [ ] ONLY proceed if paper trading shows positive expectancy
- [ ] Switch to Hyperliquid mainnet
- [ ] Start with SMALL capital ($500-$1,000)
- [ ] Monitor closely for first 48 hours
- [ ] Scale up only after 2+ weeks of live profitability

### Phase 4: Enhancements (Ongoing)
- [ ] Add Grok's built-in web search tool for real-time news
- [ ] Add X/Twitter sentiment scoring as explicit data input
- [ ] Implement trailing stop-loss logic
- [ ] Add portfolio heat map dashboard (simple web UI)
- [ ] Add backtesting module using historical data
- [ ] Explore Grok 4.20 Multi-Agent API when available

---

## 12. Key Design Principles for Claude Code

When building this system, follow these principles:

1. **The AI is NEVER trusted with execution.** Grok suggests. Python validates. Only validated trades execute. The Risk Guardian is the final authority.

2. **Every interaction with Grok is logged.** Full prompt, full response, timestamp, resulting action. This is your audit trail and your training data for prompt improvement.

3. **Fail safe, not fail open.** If Grok's response is malformed, unparseable, or times out — do NOTHING. Log the error and wait for the next cycle.

4. **Configuration over code.** Risk parameters, asset universe, cycle interval, notification preferences — all in config files or environment variables. Changing behavior should never require code changes.

5. **Testnet first, always.** The codebase should have a single `LIVE_TRADING = True/False` flag that gates real execution. Default is False.

6. **Idempotency.** If the bot crashes and restarts, it should recover gracefully by reading current positions from the exchange and trade history from the database.

7. **No state in memory.** Everything persists to SQLite. The bot should be restartable at any time without data loss.

---

## 13. Monitoring & Observability

The bot should produce these outputs:

- **Console logs** (via loguru): Every cycle's market read, decisions, and execution results
- **SQLite tables:**
  - `trades` — full trade history with P&L
  - `positions` — current and historical positions
  - `grok_logs` — every prompt/response pair
  - `daily_summaries` — aggregated daily performance
  - `rejections` — trades Grok suggested but Risk Guardian blocked
- **Discord/Telegram notifications:**
  - Trade opened (asset, side, size, entry, stop, target)
  - Trade closed (asset, P&L, hold duration)
  - Daily summary (P&L, win rate, open positions, equity curve)
  - Risk alerts (daily limit approached, kill switch triggered)
  - Errors (API failures, parsing errors)

---

## 14. Estimated API Costs

Based on xAI API pricing (~$0.20/M input, $0.50/M output):

- Each cycle sends ~2,000 input tokens (system prompt + context)
- Each cycle receives ~800 output tokens (JSON response)
- At 15-minute intervals = 96 cycles/day
- **Daily cost estimate: ~$0.06/day ($1.80/month)**
- Even at 5-minute intervals: ~$0.18/day ($5.40/month)

This is extremely cheap to operate.

---

## 15. Important Disclaimers

- **This is a speculative trading system.** Perpetual futures carry significant risk of loss, including total loss of capital.
- **Past performance (including Alpha Arena results) does not predict future results.**
- **Start with capital you can afford to lose entirely.**
- **The bot's risk guardrails are a safety net, not a guarantee.**
- **Monitor the bot actively, especially in the first weeks of live trading.**
- **Consult a financial advisor before deploying significant capital.**

---

## Appendix A: Quick-Start Checklist

1. Create xAI account → get API key → fund with credits
2. Create Ethereum wallet for Hyperliquid (MetaMask or similar)
3. Bridge USDC to Hyperliquid (Arbitrum network)
4. Export environment variables (Section 7)
5. Run `pip install -r requirements.txt`
6. Set `LIVE_TRADING = False` (paper mode)
7. Run `python main.py`
8. Monitor logs and Discord for 2 weeks
9. Review performance → decide whether to go live
10. If going live: set `LIVE_TRADING = True`, start with small capital

---

*Spec version: 1.0 | Created: March 2026 | For: Claude Code (Opus 4.6) implementation*
