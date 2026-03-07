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
    "max_position_size_pct": 0.15,        # Max 15% of portfolio in any single position  (was 0.10)
    "max_total_exposure_pct": 0.60,        # Max 60% of portfolio across ALL positions     (was 0.30)
    "max_leverage": 3.0,                   # Hard cap at 3x leverage (unchanged — non-negotiable)

    # === Loss Limits ===
    "max_loss_per_trade_pct": 0.02,        # Stop-loss: max 2% portfolio loss per trade (unchanged)
    "max_daily_loss_pct": 0.06,            # If daily drawdown hits 6%, halt ALL trading  (was 0.05)
    "max_weekly_loss_pct": 0.12,           # If weekly drawdown hits 12%, halt 48 hours   (was 0.10)
    "max_total_drawdown_pct": 0.20,        # If total drawdown hits 20%, KILL SWITCH (unchanged)

    # === Trade Frequency (loosened for active trading) ===
    "min_time_between_trades_minutes": 5,  # Allow rapid re-entry after 5 min              (was 30)
    "max_trades_per_day": 50,              # Allow up to 50 trades per day                  (was 8)

    # === Mandatory Risk Controls ===
    "require_stop_loss": True,             # Every position MUST have a stop-loss (unchanged)
    "require_take_profit": True,           # Every position MUST have a take-profit (unchanged)
    "max_stop_loss_distance_pct": 0.05,    # Stop can't be more than 5% from entry (unchanged)
    "min_risk_reward_ratio": 1.2,          # Minimum 1.2:1 reward-to-risk                  (was 1.5)

    # === Time-Based Exits (from ai-trader pattern) ===
    "max_holding_period_hours": 8,         # Force-close positions held longer than 8 hours
    "stale_position_check": True,          # Enable time-based forced exit checks

    # === Kill Switch ===
    "kill_switch_enabled": False,          # Manual override to halt all trading
}
