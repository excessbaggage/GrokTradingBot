"""
Hard-coded risk management parameters.

These values are the FINAL AUTHORITY on all trading decisions.
The AI (Grok) can suggest whatever it wants — these gates cannot be bypassed.
No runtime modification is allowed.
"""

RISK_PARAMS = {
    # === Position Sizing ===
    "max_position_size_pct": 0.10,       # Max 10% of portfolio in any single position
    "max_total_exposure_pct": 0.30,       # Max 30% of portfolio across ALL positions combined
    "max_leverage": 3.0,                  # Hard cap at 3x leverage

    # === Loss Limits ===
    "max_loss_per_trade_pct": 0.02,       # Stop-loss: max 2% portfolio loss per trade
    "max_daily_loss_pct": 0.05,           # If daily drawdown hits 5%, halt ALL trading until next day
    "max_weekly_loss_pct": 0.10,          # If weekly drawdown hits 10%, halt trading for 48 hours
    "max_total_drawdown_pct": 0.20,       # If total drawdown from peak hits 20%, KILL SWITCH

    # === Trade Frequency ===
    "min_time_between_trades_minutes": 30, # No rapid-fire trading
    "max_trades_per_day": 8,               # Cap daily trade count

    # === Mandatory Risk Controls ===
    "require_stop_loss": True,             # Every position MUST have a stop-loss
    "require_take_profit": True,           # Every position MUST have a take-profit
    "max_stop_loss_distance_pct": 0.05,    # Stop can't be more than 5% from entry
    "min_risk_reward_ratio": 1.5,          # Minimum 1.5:1 reward-to-risk

    # === Kill Switch ===
    "kill_switch_enabled": False,          # Manual override to halt all trading
}
