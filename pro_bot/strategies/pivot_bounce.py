"""
Pivot Point Bounce Strategy

Logic:
  Calculate daily/weekly pivot levels from prior period OHLC:
    P  = (H + L + C) / 3
    R1 = 2P - L   |  S1 = 2P - H
    R2 = P + (H-L) |  S2 = P - (H-L)
    R3 = H + 2(P-L)|  S3 = L - 2(H-P)

  When price approaches a support (S1/S2) with RSI oversold → BUY
  When price approaches a resistance (R1/R2) with RSI overbought → SELL
  At the central pivot (P): use RSI direction.

  SL: beyond the next pivot level.
  TP: previous pivot level in the direction of the trade.

Best platform:    MT4/MT5, NinjaTrader, Sierra Chart (futures)
Best instruments: US500/SPX500, NAS100, Gold futures (GC), CME FX futures
                  Especially powerful during NY session intraday
Expected WR:      60–70% when levels align with RSI confirmation
"""

from .base import BaseProStrategy, Signal
from ..indicators import pivot_levels, rsi


class PivotBounceStrategy(BaseProStrategy):

    name             = "pivot_bounce"
    best_platform    = "MT4/MT5 / NinjaTrader"
    best_instruments = ["US500", "NAS100", "XAUUSD", "ES futures", "NQ futures"]

    def __init__(self, config: dict):
        super().__init__(config)
        self.rsi_period  = config.get("rsi_period",  14)
        self.rsi_os      = config.get("rsi_os",     40.0)
        self.tol_pct     = config.get("tol_pct",    0.001)  # % of price for pivot zone width
        self.tp_rr       = config.get("tp_rr",       1.5)
        self._daily_ohlc: dict | None = None   # prior day's OHLC
        self._pivots:     dict | None = None
        self._last_day   = -1

    def feed_daily_bar(self, bar: dict) -> None:
        """Feed completed daily bar to update pivot levels for next day."""
        self._daily_ohlc = bar
        self._pivots     = pivot_levels(bar)

    def _evaluate(self) -> Signal:
        bars = self._bars
        if len(bars) < self.rsi_period + 2:
            return Signal(action="HOLD", reason="Warming up")
        if self._pivots is None:
            return Signal(action="HOLD", reason="No pivot levels yet — feed daily bar")

        closes   = [b["close"] for b in bars]
        rsi_vals = rsi(closes, self.rsi_period)
        rsi_now  = rsi_vals[-1]

        if rsi_now is None:
            return Signal(action="HOLD", reason="RSI not ready")

        price  = closes[-1]
        ob     = 100.0 - self.rsi_os
        pivots = self._pivots

        nearest_level = None
        nearest_dist  = float("inf")
        for name, level in pivots.items():
            tol  = level * self.tol_pct
            dist = abs(price - level)
            if dist <= tol and dist < nearest_dist:
                nearest_dist  = dist
                nearest_level = name

        if nearest_level is None:
            return Signal(action="HOLD",
                          reason=f"Price {price:.5f} not near any pivot")

        level_price = pivots[nearest_level]

        # Support levels: S1, S2, S3 + central pivot when price coming from above
        if nearest_level in ("S1", "S2", "S3") and rsi_now < self.rsi_os:
            sl = price - pivots.get("S3", price - price * 0.003)
            tp = pivots.get("P", price + sl * self.tp_rr) - price
            return Signal(
                action="BUY",
                reason=f"Price at {nearest_level}={level_price:.5f} | RSI {rsi_now:.1f} OS",
                sl_pips=abs(sl),
                tp_pips=abs(tp),
                confidence=min(1.0, (self.rsi_os - rsi_now) / 15),
                meta={"pivot": nearest_level, "level": round(level_price, 5),
                      "all_pivots": {k: round(v, 5) for k, v in pivots.items()}},
            )

        # Resistance levels: R1, R2, R3
        if nearest_level in ("R1", "R2", "R3") and rsi_now > ob:
            sl = pivots.get("R3", price + price * 0.003) - price
            tp = price - pivots.get("P", price - sl * self.tp_rr)
            return Signal(
                action="SELL",
                reason=f"Price at {nearest_level}={level_price:.5f} | RSI {rsi_now:.1f} OB",
                sl_pips=abs(sl),
                tp_pips=abs(tp),
                confidence=min(1.0, (rsi_now - ob) / 15),
                meta={"pivot": nearest_level, "level": round(level_price, 5),
                      "all_pivots": {k: round(v, 5) for k, v in pivots.items()}},
            )

        # Central pivot — use RSI direction
        if nearest_level == "P":
            if rsi_now < self.rsi_os:
                sl = price - pivots.get("S1", price * 0.999)
                tp = pivots.get("R1", price * 1.002) - price
                return Signal(
                    action="BUY",
                    reason=f"Price at Pivot P={level_price:.5f} | RSI {rsi_now:.1f} OS",
                    sl_pips=abs(sl),
                    tp_pips=abs(tp),
                    meta={"pivot": "P", "rsi": rsi_now},
                )
            if rsi_now > ob:
                sl = pivots.get("R1", price * 1.001) - price
                tp = price - pivots.get("S1", price * 0.998)
                return Signal(
                    action="SELL",
                    reason=f"Price at Pivot P={level_price:.5f} | RSI {rsi_now:.1f} OB",
                    sl_pips=abs(sl),
                    tp_pips=abs(tp),
                    meta={"pivot": "P", "rsi": rsi_now},
                )

        return Signal(
            action="HOLD",
            reason=f"Near {nearest_level} but RSI {rsi_now:.1f} not confirming",
        )
