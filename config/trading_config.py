"""
Trading configuration — asset universe, intervals, API settings.

Modify these to change bot behavior without touching code.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from config/ directory or project root
_config_dir = Path(__file__).parent
load_dotenv(_config_dir / ".env")
load_dotenv()  # Also check project root as fallback

# === API Keys ===
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
HYPERLIQUID_PRIVATE_KEY = os.getenv("HYPERLIQUID_PRIVATE_KEY", "")
HYPERLIQUID_WALLET_ADDRESS = os.getenv("HYPERLIQUID_WALLET_ADDRESS", "")

# === Notifications ===
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# === Trading Settings ===
LIVE_TRADING = os.getenv("LIVE_TRADING", "False").lower() == "true"
STARTING_CAPITAL = float(os.getenv("STARTING_CAPITAL", "10000"))
CYCLE_INTERVAL_MINUTES = int(os.getenv("CYCLE_INTERVAL_MINUTES", "15"))
MIN_CYCLE_INTERVAL_MINUTES = 5  # Absolute minimum between cycles
MAX_CYCLE_INTERVAL_MINUTES = int(os.getenv("MAX_CYCLE_INTERVAL_MINUTES", "15"))  # Cap for active trading

# === Grok Model ===
GROK_MODEL = os.getenv("GROK_MODEL", "grok-4")
XAI_BASE_URL = "https://api.x.ai/v1"
GROK_MAX_TOKENS = 4096
GROK_TEMPERATURE = 0.5  # Moderate temperature for varied trade ideas (was 0.3 in conservative mode)

# === Asset Universe ===
ASSET_UNIVERSE = ["BTC", "ETH", "SOL", "DOGE", "AVAX", "LINK", "ARB", "OP", "SUI", "APT"]

# === Hyperliquid Settings ===
HYPERLIQUID_MAINNET_URL = "https://api.hyperliquid.xyz"
HYPERLIQUID_TESTNET_URL = "https://api.hyperliquid-testnet.xyz"

def get_hyperliquid_url() -> str:
    """Return the appropriate Hyperliquid API URL based on LIVE_TRADING flag."""
    return HYPERLIQUID_MAINNET_URL if LIVE_TRADING else HYPERLIQUID_TESTNET_URL

# === Candle Intervals for Data Fetching ===
CANDLE_INTERVALS = ["1h", "4h", "1d"]
CANDLE_LOOKBACK = {
    "1h": 48,   # Last 48 hourly candles
    "4h": 20,   # Last 20 4-hour candles
    "1d": 10,   # Last 10 daily candles
}

# === Database ===
DATABASE_URL = os.getenv("DATABASE_URL", "")

# === Logging ===
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
LOG_ROTATION = "10 MB"
LOG_RETENTION = "30 days"

# === Daily Summary ===
DAILY_SUMMARY_HOUR_UTC = 0  # Send daily summary at midnight UTC
