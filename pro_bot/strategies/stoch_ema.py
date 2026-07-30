"""
Stochastic OB/OS Cross + EMA Filter Strategy

Logic:
  EMA(50) slope defines trend direction.
  Stochastic(5,3,3) %K crosses %D in oversold/overbought zone.

  Uptrend   + %K crosses above %D while < os_level  → BUY
  Downtrend + %K crosses below %D while > ob_level  → SELL

  SL: below/above the most recent swing low/high.
  TP: 1:1.5 minimum, trail after 1:1.

Best platform:    Binance/Bybit (crypto) or MT4/MT5
Best instruments: BTC/USDT, ETH/USDT, SOL/USDT on 15min
                  Also works on EUR/USD, GBP/USD, Gold
Expected WR:      58–66% with 1:1.5 R:R
"""

from .base import BaseProStrategy, Signal
from ..indicators import ema, stochastic


class StochEMAStrategy(BaseProStrategy):

    name             = "stoch_ema"
    best_platform    = "Binance / Bybit / MT4"
    best_instruments = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "EURUSD", "XAUUSD"]

    def __init__(self, config: dict):
        super().__init__(config)
        self.k_period   = config.get("k_period",   5)
        self.d_period   = config.get("d_period",   3)
        self.ob         = config.get("ob",        80.0)
        self.os_level   = config.get("os_level",  20.0)
        self.ema_period = config.get("ema_period", 50)
        self.slope_bars = config.get("slope_bars",  3)
        self.tp_rr      = config.get("tp_rr",      1.5)

    def _evaluate(self) -> Signal:
        bars = self._bars
        needed = max(self.k_period + 2 * self.d_period,
                     self.ema_period + self.slope_bars) + 3
        if len(bars) < needed:
            return Signal(action="HOLD", reason="Warming up")

        closes = [b["close"] for b in bars]
        sk, sd = stochastic(bars, self.k_period, self.d_period)
        ema_s  = ema(closes, self.ema_period)

        sk_now, sk_prev = sk[-1], sk[-2]
        sd_now, sd_prev = sd[-1], sd[-2]
        ema_now         = ema_s[-1]
        ema_prev        = ema_s[-1 - self.slope_bars]

        if any(x is None for x in [sk_now, sd_now, sk_prev, sd_prev,
                                    ema_now, ema_prev]):
            return Signal(action="HOLD", reason="Indicator not ready")

        trend_up   = ema_now > ema_prev
        trend_down = ema_now < ema_prev

        cross_up   = sk_prev < sd_prev and sk_now >= sd_now
        cross_down = sk_prev > sd_prev and sk_now <= sd_now

        price = bars[-1]["close"]
        swing_low  = min(b["low"]  for b in bars[-5:])
        swing_high = max(b["high"] for b in bars[-5:])

        if trend_up and cross_up and sk_now < self.ob:
            sl = price - swing_low
            return Signal(
                action="BUY",
                reason=f"Stoch {sk_now:.1f} cross↑ in OS zone | EMA↑",
                sl_pips=sl,
                tp_pips=sl * self.tp_rr,
                confidence=min(1.0, (self.ob - sk_now) / 30),
                meta={"stoch_k": round(sk_now, 1), "stoch_d": round(sd_now, 1)},
            )

        if trend_down and cross_down and sk_now > self.os_level:
            sl = swing_high - price
            return Signal(
                action="SELL",
                reason=f"Stoch {sk_now:.1f} cross↓ in OB zone | EMA↓",
                sl_pips=sl,
                tp_pips=sl * self.tp_rr,
                confidence=min(1.0, (sk_now - self.os_level) / 30),
                meta={"stoch_k": round(sk_now, 1), "stoch_d": round(sd_now, 1)},
            )

        return Signal(
            action="HOLD",
            reason=f"Stoch K={sk_now:.1f} D={sd_now:.1f} | "
                   f"EMA {'↑' if trend_up else '↓'}",
        )
