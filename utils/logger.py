"""
Loguru-based logging configuration for the trading bot.

Provides structured, colorized console output and rotated file logging.
Trade decisions are logged as structured JSON for auditability.
"""

from __future__ import annotations

import os
import sys
import json
from typing import Any

from loguru import logger

from config.trading_config import LOG_DIR, LOG_ROTATION, LOG_RETENTION

# ---------------------------------------------------------------------------
# Remove the default loguru handler so we can configure our own
# ---------------------------------------------------------------------------
logger.remove()

# ---------------------------------------------------------------------------
# Ensure log directory exists
# ---------------------------------------------------------------------------
os.makedirs(LOG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Console handler -- colorized, human-readable
# ---------------------------------------------------------------------------
_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

logger.add(
    sys.stderr,
    format=_CONSOLE_FORMAT,
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    colorize=True,
    backtrace=True,
    diagnose=False,  # SECURITY: diagnose=True dumps local vars (including API keys) in tracebacks
)

# ---------------------------------------------------------------------------
# General file handler -- rotated, plain-text
# ---------------------------------------------------------------------------
_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{name}:{function}:{line} | "
    "{message}"
)

logger.add(
    os.path.join(LOG_DIR, "bot_{time:YYYY-MM-DD}.log"),
    format=_FILE_FORMAT,
    level="DEBUG",
    rotation=LOG_ROTATION,
    retention=LOG_RETENTION,
    compression="gz",
    backtrace=True,
    diagnose=False,  # SECURITY: never dump variable values in file logs
    enqueue=True,  # thread-safe writes
)

# ---------------------------------------------------------------------------
# Structured JSON file handler -- for trade decisions only
# ---------------------------------------------------------------------------


def _json_serializer(message: Any) -> str:
    """Custom serializer that writes one JSON object per line."""
    record = message.record
    log_entry = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "module": record["name"],
        "function": record["function"],
        "line": record["line"],
        "message": record["message"],
    }
    # Merge any extra structured data attached via logger.bind()
    if record["extra"]:
        log_entry["extra"] = {
            k: v for k, v in record["extra"].items() if k != "_json_log"
        }
    return json.dumps(log_entry, default=str) + "\n"


logger.add(
    os.path.join(LOG_DIR, "trades_{time:YYYY-MM-DD}.jsonl"),
    format=_json_serializer,
    level="INFO",
    rotation=LOG_ROTATION,
    retention=LOG_RETENTION,
    compression="gz",
    filter=lambda record: record["extra"].get("_json_log", False),
    enqueue=True,
)

# ---------------------------------------------------------------------------
# Error file handler -- errors and above only
# ---------------------------------------------------------------------------
logger.add(
    os.path.join(LOG_DIR, "errors_{time:YYYY-MM-DD}.log"),
    format=_FILE_FORMAT,
    level="ERROR",
    rotation=LOG_ROTATION,
    retention=LOG_RETENTION,
    compression="gz",
    backtrace=True,
    diagnose=False,  # SECURITY: never dump variable values in error logs
    enqueue=True,
)


# ---------------------------------------------------------------------------
# Setup function (called by main.py — logging is configured at import time,
# this ensures the module is loaded and confirms initialization)
# ---------------------------------------------------------------------------


def setup_logger() -> None:
    """Initialize logging. Called by main.py at startup.

    The actual configuration happens at module import time above.
    This function exists as an explicit entry point and logs confirmation.
    """
    logger.info("Logger initialized — console + file + JSON sinks active")


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def log_trade_decision(decision_data: dict[str, Any], cycle: int | None = None) -> None:
    """Log a trade decision as structured JSON.

    Args:
        decision_data: The trade decision dictionary from Grok or the risk guardian.
        cycle: Optional cycle number for correlation.
    """
    bound = logger.bind(_json_log=True, cycle=cycle)
    bound.info("TRADE_DECISION | {data}", data=json.dumps(decision_data, default=str))


def log_trade_execution(trade_data: dict[str, Any]) -> None:
    """Log a trade execution as structured JSON.

    Args:
        trade_data: The executed trade details (asset, side, price, size, etc.).
    """
    bound = logger.bind(_json_log=True)
    bound.info("TRADE_EXECUTED | {data}", data=json.dumps(trade_data, default=str))


def log_trade_rejection(
    asset: str, action: str, reason: str, decision: dict[str, Any] | None = None
) -> None:
    """Log a rejected trade decision as structured JSON.

    Args:
        asset: The asset symbol (e.g. "BTC").
        action: The proposed action (e.g. "open_long").
        reason: Why the risk guardian rejected it.
        decision: The full decision dict, if available.
    """
    bound = logger.bind(_json_log=True)
    bound.warning(
        "TRADE_REJECTED | asset={asset} action={action} reason={reason}",
        asset=asset,
        action=action,
        reason=reason,
    )


def log_grok_cycle(
    cycle: int,
    prompt_preview: str,
    response_preview: str,
    decisions_count: int,
) -> None:
    """Log a summary of a Grok interaction cycle.

    Args:
        cycle: The cycle number.
        prompt_preview: A truncated preview of the context prompt.
        response_preview: A truncated preview of Grok's response.
        decisions_count: How many trade decisions were returned.
    """
    bound = logger.bind(_json_log=True, cycle=cycle)
    bound.info(
        "GROK_CYCLE | cycle={cycle} decisions={cnt} prompt_len={plen} response_len={rlen}",
        cycle=cycle,
        cnt=decisions_count,
        plen=len(prompt_preview),
        rlen=len(response_preview),
    )
