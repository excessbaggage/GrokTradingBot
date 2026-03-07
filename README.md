# Sentinel - Grok-Powered Autonomous Trading Bot

An autonomous perpetual futures trading bot powered by [xAI's Grok API](https://x.ai/api). Sentinel is a conservative swing trader that trades BTC, ETH, and SOL on [Hyperliquid DEX](https://hyperliquid.xyz).

> **Grok proposes, Risk Guardian disposes.** The AI never executes trades directly -- every decision passes through 13 hard-coded risk checks before reaching the exchange.

## Architecture

```
main.py (orchestrator loop)
  |
  +-- config/         Configuration & environment
  |     trading_config.py   - API keys, intervals, asset universe
  |     risk_config.py      - Immutable risk parameters
  |
  +-- data/           Market data & state
  |     database.py         - SQLite schema & helpers
  |     market_data.py      - Hyperliquid candle/orderbook fetcher
  |     portfolio_state.py  - Equity, positions, P&L tracking
  |     trade_history.py    - Trade CRUD operations
  |     context_builder.py  - Builds the context prompt for Grok
  |
  +-- brain/          AI decision engine
  |     grok_client.py      - OpenAI SDK client (x.ai base_url)
  |     system_prompt.py    - Sentinel's personality & rules
  |     decision_parser.py  - JSON response -> Pydantic models
  |     models.py           - TradeDecision, GrokResponse schemas
  |
  +-- execution/      Trade execution & safety
  |     risk_guardian.py    - 13 sequential risk checks
  |     order_manager.py   - Live + paper order placement
  |     position_manager.py - Position monitoring & reconciliation
  |     notifications.py   - Discord/Telegram alerts
  |
  +-- utils/          Shared utilities
  |     logger.py          - Loguru setup (4 sinks)
  |     helpers.py         - UTC helpers
  |
  +-- tests/          Test suite (126 tests)
        test_risk_guardian.py   - 32 tests for all 13 risk checks
        test_decision_parser.py - 20 tests for JSON parsing
        test_paper_mode.py      - 12 paper mode integration tests
        test_integration.py     - 62 end-to-end integration tests
```

## How It Works

### The Trading Cycle

Every cycle (default: 15 minutes), Sentinel runs this loop:

1. **Kill switch check** -- If drawdown exceeds 20%, halt everything
2. **Fetch market data** -- OHLCV candles (1h, 4h, 1d), orderbook, funding rates
3. **Fetch portfolio state** -- Equity, open positions, unrealized P&L
4. **Get risk status** -- Daily/weekly P&L, drawdown, exposure metrics
5. **Build context prompt** -- Combine all data into a structured prompt
6. **Query Grok** -- Send context to Grok-4 for trading decisions
7. **Parse response** -- Extract structured JSON from Grok's response
8. **Validate each trade** -- Risk Guardian runs 13 checks per decision
9. **Execute approved trades** -- Place orders on Hyperliquid (or simulate)
10. **Manage positions** -- Update P&L, detect stop/TP fills
11. **Send notifications** -- Alert via Discord/Telegram
12. **Sleep** -- Wait for next cycle (Grok can suggest shorter intervals)

### Risk Guardian (13 Checks)

Every trade decision must pass ALL of these before execution:

| # | Check | Threshold |
|---|-------|-----------|
| 1 | Kill switch | Not active |
| 2 | Max position size | <= 20% of equity |
| 3 | Max leverage | <= 5x |
| 4 | Max total exposure | <= 60% of equity |
| 5 | Max correlated exposure | <= 40% same-direction |
| 6 | Risk/Reward ratio | >= 1.5:1 |
| 7 | Max daily loss | >= -5% daily |
| 8 | Max weekly loss | >= -10% weekly |
| 9 | Max drawdown | >= -20% from peak |
| 10 | Min time between trades | >= 5 minutes |
| 11 | Max daily trades | <= 10 per day |
| 12 | Consecutive loss cooldown | Back off after 3 losses |
| 13 | Volatility check | Price change within bounds |

### Paper vs Live Mode

| Feature | Paper Mode | Live Mode |
|---------|-----------|-----------|
| API key needed | XAI_API_KEY only | All keys required |
| Orders | Simulated in-memory | Real on Hyperliquid |
| Starting capital | $10,000 (configurable) | Your actual wallet |
| Exchange | None (simulated fills) | Hyperliquid testnet/mainnet |
| Risk | Zero | Real capital at risk |

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/guywholikestobreakthings/GrokTradingBot.git
cd GrokTradingBot
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example config/.env
```

Edit `config/.env` with your keys:

```env
# Required -- get yours at https://console.x.ai
XAI_API_KEY=your_xai_api_key_here

# Only needed for live trading
HYPERLIQUID_PRIVATE_KEY=your_wallet_private_key_here
HYPERLIQUID_WALLET_ADDRESS=your_wallet_address_here

# Optional -- for trade notifications
DISCORD_WEBHOOK_URL=your_discord_webhook_url_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here

# Trading config
STARTING_CAPITAL=10000
LIVE_TRADING=False
CYCLE_INTERVAL_MINUTES=15
GROK_MODEL=grok-4
```

### 3. Run in Paper Mode (Default)

```bash
python main.py
```

That's it. Sentinel will:
- Start in **paper trading mode** (no real money)
- Connect to the Grok API and verify connectivity
- Initialize the SQLite database
- Begin the trading cycle loop
- Log everything to `logs/` and the console

### 4. Run Tests

```bash
pytest tests/ -v
```

All 126 tests run in < 1 second with zero API calls (fully mocked).

### 5. Switch to Live Trading

> **Warning**: Live mode uses real capital. Start small ($500-1000).

1. Set up a [Hyperliquid](https://hyperliquid.xyz) wallet
2. Fund it with USDC
3. Update `config/.env`:
   ```env
   LIVE_TRADING=True
   HYPERLIQUID_PRIVATE_KEY=your_private_key
   HYPERLIQUID_WALLET_ADDRESS=your_wallet_address
   ```
4. Run: `python main.py`

## Web Dashboard

Sentinel includes a read-only web dashboard for monitoring the bot in real time.

### Start the Dashboard

```bash
# In a separate terminal (while the bot is running)
python dashboard.py
```

Then open **http://localhost:5050** in your browser.

### What You See

- **Equity & P&L** — Current equity, daily/weekly P&L with percentages
- **Equity curve** — Chart.js line chart of equity over time
- **Risk metrics** — Exposure, open positions, loss streak, trade count, rejections
- **Open positions** — Live table with unrealized P&L
- **Grok analysis** — Latest market analysis, stance, and trade decisions
- **Trade history** — Last 50 trades with entry/exit prices and P&L
- **Risk Guardian rejections** — Trades blocked by the 13 safety checks

The dashboard auto-refreshes every 30 seconds and connects to the bot's SQLite database in **read-only mode** — it cannot interfere with the bot.

### Configuration

| Env Variable | Default | Description |
|-------------|---------|-------------|
| `DASHBOARD_PORT` | `5050` | Port the dashboard runs on |
| `DASHBOARD_HOST` | `127.0.0.1` | Host to bind to (localhost only) |

## Graceful Shutdown

Press `Ctrl+C` to stop the bot. Sentinel will:
1. Finish the current cycle
2. Send a shutdown notification
3. Exit cleanly (no orphaned positions)

## Logs & Database

| Location | Contents |
|----------|----------|
| `logs/bot_YYYY-MM-DD.log` | Full human-readable logs |
| `logs/errors_YYYY-MM-DD.log` | Errors only |
| `logs/trades_YYYY-MM-DD.jsonl` | Structured JSON trade log |
| `db/trading_bot.db` | SQLite database (trades, positions, Grok logs) |

## Configuration Reference

### `config/trading_config.py`

| Setting | Default | Description |
|---------|---------|-------------|
| `LIVE_TRADING` | `False` | Paper mode by default |
| `STARTING_CAPITAL` | `10000` | Paper mode starting equity |
| `CYCLE_INTERVAL_MINUTES` | `15` | Time between trading cycles |
| `GROK_MODEL` | `grok-4` | xAI model to use |
| `ASSET_UNIVERSE` | `BTC, ETH, SOL` | Assets to trade |
| `GROK_TEMPERATURE` | `0.3` | Low for deterministic decisions |

### `config/risk_config.py`

All risk parameters are defined here and enforced by the Risk Guardian. See the file for the full list of thresholds.

## Tech Stack

- **AI**: xAI Grok-4 via OpenAI SDK
- **Exchange**: Hyperliquid DEX (perpetual futures)
- **Database**: SQLite
- **Dashboard**: Flask + Tailwind CSS + Chart.js
- **Validation**: Pydantic v2
- **Logging**: Loguru
- **Retries**: Tenacity
- **Data**: Pandas, NumPy

## License

MIT
