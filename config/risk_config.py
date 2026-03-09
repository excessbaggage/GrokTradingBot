"""
Hard-coded risk management parameters.

These values are the FINAL AUTHORITY on all trading decisions.
The AI (Grok) can suggest whatever it wants — these gates cannot be bypassed.
No runtime modification is allowed.

NOTE: Parameters tuned for active trading stress test (24-hour paper mode).
Original conservative values are preserved in comments for easy rollback.
"""

RISK_PARAMS = {
    # === Position Sizing ===
    "max_position_size_pct": 0.25,        # Max 25% of portfolio in any single position ($2,500)
    "max_total_exposure_pct": 0.85,        # Max 85% of portfolio across ALL positions ($8,500)
    "max_leverage": 3.0,                   # Hard cap at 3x leverage (non-negotiable)

    # === Loss Limits ===
    "max_loss_per_trade_pct": 0.03,        # Stop-loss: max 3% portfolio loss per trade ($300)
    "max_daily_loss_pct": 0.10,            # If daily drawdown hits 10%, halt ALL trading ($1,000)
    "max_weekly_loss_pct": 0.20,           # If weekly drawdown hits 20%, halt 48 hours ($2,000)
    "max_total_drawdown_pct": 0.35,        # If total drawdown hits 35%, KILL SWITCH ($3,500)

    # === Trade Frequency ===
    "min_time_between_trades_minutes": 5,  # Allow re-entry after 5 min (kept for 5-min cycles)
    "max_trades_per_day": 50,              # Allow up to 50 trades per day

    # === Mandatory Risk Controls ===
    "require_stop_loss": True,             # Every position MUST have a stop-loss
    "require_take_profit": True,           # Every position MUST have a take-profit
    "max_stop_loss_distance_pct": 0.06,    # Stop can be up to 6% from entry
    "min_risk_reward_ratio": 1.2,          # Minimum 1.2:1 reward-to-risk

    # === Time-Based Exits (from ai-trader pattern) ===
    "max_holding_period_hours": 12,        # Force-close positions held longer than 12 hours
    "stale_position_check": True,          # Enable time-based forced exit checks

    # === Kill Switch ===
    "kill_switch_enabled": False,          # Manual override to halt all trading
}
