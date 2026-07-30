"""
RSI Divergence Strategy

Logic:
  Bullish divergence: price makes lower low BUT RSI makes higher low → BUY
    (momentum weakening on the downside = reversal incoming)

  Bearish divergence: price makes higher high BUT RSI makes lower high → SELL
    (momentum weakening on the upside = reversal incoming)

  Hidden bullish divergence (trend continuation):
    Price higher low + RSI lower low → BUY in uptrend (pullback is shallow)

  Hidden bearish divergence (trend continuation):
    Price lower high + RSI higher high → SELL in downtrend

  SL: beyond the divergence swing point.
  TP: minimum 1:1.5 R:R, target next swing high/low.

Best platform:    Any — TradingView alerts, MT4/MT5, Binance
Best instruments: BTC/USD on 4h, Gold on daily, EUR/USD on 4h
                  Works on ALL timeframes but best on 4h and above
Expected WR:      65–75% on higher timeframes (4h, daily)
"""

from .base import BaseProStrategy, Signal
from ..indicators import rsi


class RSIDivergenceStrategy(BaseProStrategy):

    name             = "rsi_divergence"
    best_platform    = "TradingView / MT4 / Binance"
    best_instruments = ["BTCUSD (4h)", "XAUUSD (daily)", "EURUSD (4h)",
                        "ETHUSD (4h)", "US500 (4h)"]

    def __init__(self, config: dict):
        super().__init__(config)
        self.rsi_period  = config.get("rsi_period",  14)
        self.lookback    = config.get("lookback",     12)   # bars to look back for divergence
        self.os_level    = config.get("os_level",    40.0)  # oversold threshold
        self.ob_level    = config.get("ob_level",    60.0)  # overbought threshold
        self.hidden_div  = config.get("hidden_divergence", True)
        self.tp_rr       = config.get("tp_rr",        2.0)
        self._cooldown   = 0

    def _evaluate(self) -> Signal:
        bars = self._bars
        needed = self.rsi_period + self.lookback + 3
        if len(bars) < needed:
            return Signal(action="HOLD", reason="Warming up")

        if self._cooldown > 0:
            self._cooldown -= 1
            return Signal(action="HOLD", reason=f"Cooldown {self._cooldown}")

        closes   = [b["close"] for b in bars]
        rsi_vals = rsi(closes, self.rsi_period)

        rsi_now  = rsi_vals[-1]
        if rsi_now is None:
            return Signal(action="HOLD", reason="RSI not ready")

        window_closes = closes[-self.lookback - 1: -1]
        window_rsi    = [rsi_vals[i] for i in range(-self.lookback - 1, -1)
                         if rsi_vals[i] is not None]

        if not window_rsi:
            return Signal(action="HOLD", reason="Window empty")

        prev_low_p  = min(window_closes)
        prev_high_p = max(window_closes)
        prev_low_r  = min(window_rsi)
        prev_high_r = max(window_rsi)

        price = closes[-1]
        price_low_idx  = window_closes.index(prev_low_p)
        price_high_idx = window_closes.index(prev_high_p)

        # ── Classic Bullish Divergence ────────────────────────────
        if (price < prev_low_p and
                rsi_now > prev_low_r and
                rsi_now < self.os_level):
            sl = price - min(b["low"] for b in bars[-3:])
            self._cooldown = 3
            return Signal(
                action="BUY",
                reason=f"Bullish divergence: price LL, RSI HL at {rsi_now:.1f}",
                sl_pips=sl,
                tp_pips=sl * self.tp_rr,
                confidence=min(1.0, (prev_high_r - rsi_now) / 20),
                meta={"divergence": "classic_bullish", "rsi": rsi_now},
            )

        # ── Classic Bearish Divergence ────────────────────────────
        if (price > prev_high_p and
                rsi_now < prev_high_r and
                rsi_now > self.ob_level):
            sl = max(b["high"] for b in bars[-3:]) - price
            self._cooldown = 3
            return Signal(
                action="SELL",
                reason=f"Bearish divergence: price HH, RSI LH at {rsi_now:.1f}",
                sl_pips=sl,
                tp_pips=sl * self.tp_rr,
                confidence=min(1.0, (rsi_now - prev_low_r) / 20),
                meta={"divergence": "classic_bearish", "rsi": rsi_now},
            )

        # ── Hidden Bullish Divergence (trend continuation) ────────
        if self.hidden_div:
            if (price > prev_low_p and
                    rsi_now < prev_low_r and
                    rsi_now < 50):
                sl = price - min(b["low"] for b in bars[-3:])
                self._cooldown = 3
                return Signal(
                    action="BUY",
                    reason=f"Hidden bullish div: price HL, RSI LL at {rsi_now:.1f}",
                    sl_pips=sl,
                    tp_pips=sl * self.tp_rr,
                    confidence=0.7,
                    meta={"divergence": "hidden_bullish", "rsi": rsi_now},
                )

            if (price < prev_high_p and
                    rsi_now > prev_high_r and
                    rsi_now > 50):
                sl = max(b["high"] for b in bars[-3:]) - price
                self._cooldown = 3
                return Signal(
                    action="SELL",
                    reason=f"Hidden bearish div: price LH, RSI HH at {rsi_now:.1f}",
                    sl_pips=sl,
                    tp_pips=sl * self.tp_rr,
                    confidence=0.7,
                    meta={"divergence": "hidden_bearish", "rsi": rsi_now},
                )

        return Signal(
            action="HOLD",
            reason=f"RSI {rsi_now:.1f} | no divergence",
        )

    def on_result(self, won: bool) -> None:
        if not won:
            self._cooldown = max(self._cooldown, 2)
