"""
Pre-built strategy callbacks for backtesting without Grok API calls.

Each function accepts a ``MarketSnapshot`` and returns a list of
``TradeDecision`` objects matching the ``brain/models.py`` schemas.
These are designed to plug directly into ``BacktestSimulator.run()``.

Usage::

    from backtesting import BacktestSimulator, simple_rsi_strategy

    sim = BacktestSimulator(initial_capital=10_000)
    result = sim.run(data, simple_rsi_strategy)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from brain.models import TradeDecision

if TYPE_CHECKING:
    from backtesting.simulator import MarketSnapshot


# ======================================================================
# Technical indicator helpers
# ======================================================================


def _compute_rsi(closes: pd.Series, period: int = 14) -> float:
    """Compute RSI from a series of close prices.

    Returns 50.0 (neutral) if insufficient data.
    """
    if len(closes) < period + 1:
        return 50.0

    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))

    avg_gain = gain.rolling(period).mean().iloc[-1]
    avg_loss = loss.rolling(period).mean().iloc[-1]

    if pd.isna(avg_gain) or pd.isna(avg_loss):
        return 50.0
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0

    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def _compute_atr(candles: pd.DataFrame, period: int = 14) -> float:
    """Compute Average True Range from a candle DataFrame.

    Returns 0.0 if insufficient data.
    """
    if len(candles) < period + 1:
        return 0.0

    high = candles["high"]
    low = candles["low"]
    close = candles["close"]

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr) if pd.notna(atr) else 0.0


def _compute_sma(closes: pd.Series, period: int) -> float:
    """Compute Simple Moving Average.

    Returns the last close if insufficient data.
    """
    if len(closes) < period:
        return float(closes.iloc[-1]) if len(closes) > 0 else 0.0
    return float(closes.rolling(period).mean().iloc[-1])


# ======================================================================
# Strategy 1: Simple RSI
# ======================================================================


def simple_rsi_strategy(snapshot: MarketSnapshot) -> list[TradeDecision]:
    """RSI mean-reversion strategy.

    - Buys when RSI-14 < 30 (oversold)
    - Sells when RSI-14 > 70 (overbought)
    - Uses ATR-based stop-loss and 2:1 risk-reward take-profit

    This is a classic "buy low, sell high" approach that works well in
    ranging markets but underperforms in strong trends.
    """
    candles = snapshot.candles
    if len(candles) < 16:
        return []

    closes = candles["close"]
    rsi = _compute_rsi(closes, period=14)
    atr = _compute_atr(candles, period=14)
    price = snapshot.close

    if atr <= 0:
        atr = price * 0.02  # Fallback: 2% of price

    # Skip if already in a position for this asset
    if snapshot.asset in snapshot.positions:
        return []

    decisions: list[TradeDecision] = []

    if rsi < 30:
        # Oversold -- go long
        sl = round(price - 1.5 * atr, 2)
        tp = round(price + 3.0 * atr, 2)  # 2:1 R:R
        risk = abs(price - sl)
        reward = abs(tp - price)
        rr = reward / risk if risk > 0 else 0

        decisions.append(
            TradeDecision(
                action="open_long",
                asset=snapshot.asset,
                size_pct=0.05,
                leverage=2.0,
                entry_price=price,
                stop_loss=sl,
                take_profit=tp,
                order_type="market",
                reasoning=f"RSI oversold at {rsi:.1f}. Opening long with ATR-based stops.",
                conviction="medium",
                risk_reward_ratio=round(rr, 2),
            )
        )

    elif rsi > 70:
        # Overbought -- go short
        sl = round(price + 1.5 * atr, 2)
        tp = round(price - 3.0 * atr, 2)  # 2:1 R:R
        risk = abs(sl - price)
        reward = abs(price - tp)
        rr = reward / risk if risk > 0 else 0

        decisions.append(
            TradeDecision(
                action="open_short",
                asset=snapshot.asset,
                size_pct=0.05,
                leverage=2.0,
                entry_price=price,
                stop_loss=sl,
                take_profit=tp,
                order_type="market",
                reasoning=f"RSI overbought at {rsi:.1f}. Opening short with ATR-based stops.",
                conviction="medium",
                risk_reward_ratio=round(rr, 2),
            )
        )

    return decisions


# ======================================================================
# Strategy 2: Momentum
# ======================================================================


def momentum_strategy(snapshot: MarketSnapshot) -> list[TradeDecision]:
    """Trend-following momentum strategy.

    - Goes long when price is above 20-period SMA and the SMA is rising
    - Goes short when price is below 20-period SMA and the SMA is falling
    - Uses ATR-based stops with trailing characteristics

    This strategy profits in trending markets but suffers in chop.
    """
    candles = snapshot.candles
    if len(candles) < 25:
        return []

    closes = candles["close"]
    price = snapshot.close
    atr = _compute_atr(candles, period=14)

    if atr <= 0:
        atr = price * 0.02

    # Skip if already in a position
    if snapshot.asset in snapshot.positions:
        return []

    # Compute SMA-20 at current and 5 candles ago
    sma_20_current = _compute_sma(closes, 20)
    sma_20_prev = _compute_sma(closes.iloc[:-5], 20) if len(closes) > 25 else sma_20_current

    sma_rising = sma_20_current > sma_20_prev
    sma_falling = sma_20_current < sma_20_prev

    decisions: list[TradeDecision] = []

    if price > sma_20_current and sma_rising:
        # Uptrend -- go long
        sl = round(price - 2.0 * atr, 2)
        tp = round(price + 4.0 * atr, 2)  # 2:1 R:R with wider stops
        risk = abs(price - sl)
        reward = abs(tp - price)
        rr = reward / risk if risk > 0 else 0

        decisions.append(
            TradeDecision(
                action="open_long",
                asset=snapshot.asset,
                size_pct=0.05,
                leverage=2.0,
                entry_price=price,
                stop_loss=sl,
                take_profit=tp,
                order_type="market",
                reasoning=(
                    f"Momentum long: price {price:.2f} above rising SMA-20 "
                    f"({sma_20_current:.2f}). ATR={atr:.2f}."
                ),
                conviction="medium",
                risk_reward_ratio=round(rr, 2),
            )
        )

    elif price < sma_20_current and sma_falling:
        # Downtrend -- go short
        sl = round(price + 2.0 * atr, 2)
        tp = round(price - 4.0 * atr, 2)  # 2:1 R:R
        risk = abs(sl - price)
        reward = abs(price - tp)
        rr = reward / risk if risk > 0 else 0

        decisions.append(
            TradeDecision(
                action="open_short",
                asset=snapshot.asset,
                size_pct=0.05,
                leverage=2.0,
                entry_price=price,
                stop_loss=sl,
                take_profit=tp,
                order_type="market",
                reasoning=(
                    f"Momentum short: price {price:.2f} below falling SMA-20 "
                    f"({sma_20_current:.2f}). ATR={atr:.2f}."
                ),
                conviction="medium",
                risk_reward_ratio=round(rr, 2),
            )
        )

    return decisions


# ======================================================================
# Strategy 3: Mean Reversion (Funding Rate)
# ======================================================================


def mean_reversion_strategy(snapshot: MarketSnapshot) -> list[TradeDecision]:
    """Mean-reversion strategy that fades extreme price deviations.

    - Goes long when price is more than 2 ATR below SMA-50
    - Goes short when price is more than 2 ATR above SMA-50
    - Tighter stops (1 ATR) since we expect reversion

    In a live context, this would also fade extreme funding rates, but
    since backtesting operates on OHLCV data only, we use price deviation
    from mean as a proxy for "extreme" conditions.
    """
    candles = snapshot.candles
    if len(candles) < 50:
        return []

    closes = candles["close"]
    price = snapshot.close
    atr = _compute_atr(candles, period=14)

    if atr <= 0:
        atr = price * 0.02

    # Skip if already in a position
    if snapshot.asset in snapshot.positions:
        return []

    sma_50 = _compute_sma(closes, 50)
    deviation = price - sma_50
    deviation_atr = abs(deviation) / atr if atr > 0 else 0

    decisions: list[TradeDecision] = []

    if deviation_atr > 2.0 and deviation < 0:
        # Price significantly below mean -- buy the dip
        sl = round(price - 1.0 * atr, 2)
        tp = round(sma_50, 2)  # Target mean reversion to SMA
        risk = abs(price - sl)
        reward = abs(tp - price)
        rr = reward / risk if risk > 0 else 0

        if rr >= 1.2:  # Only take trades with acceptable R:R
            decisions.append(
                TradeDecision(
                    action="open_long",
                    asset=snapshot.asset,
                    size_pct=0.04,
                    leverage=2.0,
                    entry_price=price,
                    stop_loss=sl,
                    take_profit=tp,
                    order_type="market",
                    reasoning=(
                        f"Mean reversion long: price {deviation_atr:.1f} ATR below "
                        f"SMA-50 ({sma_50:.2f}). Expecting reversion."
                    ),
                    conviction="medium",
                    risk_reward_ratio=round(rr, 2),
                )
            )

    elif deviation_atr > 2.0 and deviation > 0:
        # Price significantly above mean -- fade the rally
        sl = round(price + 1.0 * atr, 2)
        tp = round(sma_50, 2)  # Target mean reversion
        risk = abs(sl - price)
        reward = abs(price - tp)
        rr = reward / risk if risk > 0 else 0

        if rr >= 1.2:
            decisions.append(
                TradeDecision(
                    action="open_short",
                    asset=snapshot.asset,
                    size_pct=0.04,
                    leverage=2.0,
                    entry_price=price,
                    stop_loss=sl,
                    take_profit=tp,
                    order_type="market",
                    reasoning=(
                        f"Mean reversion short: price {deviation_atr:.1f} ATR above "
                        f"SMA-50 ({sma_50:.2f}). Expecting reversion."
                    ),
                    conviction="medium",
                    risk_reward_ratio=round(rr, 2),
                )
            )

    return decisions
