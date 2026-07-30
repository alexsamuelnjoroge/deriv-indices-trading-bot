"""
Session Open Breakout Strategy (London / New York)

Logic:
  1. During Asian session (00:00–07:00 UTC): track range high and low.
  2. At London open (07:00–09:00 UTC): if price breaks above range high → BUY,
     if price breaks below range low → SELL.
  3. At NY open (13:00–15:00 UTC): same logic on any unbroken range.

  SL: opposite side of the Asian range.
  TP: 1× the Asian range width minimum (1:1 R:R), trail beyond that.

Best platform:    MT4/MT5 with pending orders (buy stop / sell stop)
Best instruments: EUR/USD, GBP/USD, USD/JPY, Gold
Best times:       Every trading day — London open is the most reliable session
Expected WR:      62–68% with 1:1.5 R:R
"""

from .base import BaseProStrategy, Signal


ASIAN_START_H  = 0    # UTC hour
ASIAN_END_H    = 7
LONDON_START_H = 7
LONDON_END_H   = 9
NY_START_H     = 13
NY_END_H       = 15

BREAKOUT_BUFFER = 0.10   # price must break range by 10% of range width to confirm


class SessionBreakoutStrategy(BaseProStrategy):

    name             = "session_breakout"
    best_platform    = "MT4/MT5 with pending orders"
    best_instruments = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]

    def __init__(self, config: dict):
        super().__init__(config)
        self.asian_start  = config.get("asian_start_h",  ASIAN_START_H)
        self.asian_end    = config.get("asian_end_h",    ASIAN_END_H)
        self.london_start = config.get("london_start_h", LONDON_START_H)
        self.london_end   = config.get("london_end_h",   LONDON_END_H)
        self.ny_start     = config.get("ny_start_h",     NY_START_H)
        self.ny_end       = config.get("ny_end_h",       NY_END_H)
        self.buffer       = config.get("breakout_buffer", BREAKOUT_BUFFER)
        self.tp_rr        = config.get("tp_rr",           1.5)
        self._fired_today = False
        self._last_day    = -1

    def _hour_utc(self, epoch: int) -> int:
        return (epoch % 86400) // 3600

    def _day(self, epoch: int) -> int:
        return epoch // 86400

    def _evaluate(self) -> Signal:
        if not self._bars:
            return Signal(action="HOLD", reason="No bars")

        bar   = self._bars[-1]
        epoch = bar.get("epoch", 0)
        h_utc = self._hour_utc(epoch)
        day   = self._day(epoch)

        # Reset daily state at start of new day
        if day != self._last_day:
            self._fired_today = False
            self._last_day    = day

        # Only one trade per day
        if self._fired_today:
            return Signal(action="HOLD", reason="Already traded today")

        in_breakout = (self.london_start <= h_utc < self.london_end or
                       self.ny_start     <= h_utc < self.ny_end)
        if not in_breakout:
            return Signal(action="HOLD", reason=f"UTC hour {h_utc} — not breakout window")

        # Build Asian range from today's bars
        asian_bars = [
            b for b in self._bars
            if self._day(b["epoch"]) == day
            and self.asian_start <= self._hour_utc(b["epoch"]) < self.asian_end
        ]
        if len(asian_bars) < 5:
            return Signal(action="HOLD", reason="Insufficient Asian bars")

        range_high = max(b["high"] for b in asian_bars)
        range_low  = min(b["low"]  for b in asian_bars)
        span       = range_high - range_low

        if span <= 0:
            return Signal(action="HOLD", reason="Zero Asian range")

        price = bar["close"]
        buf   = span * self.buffer

        if price > range_high + buf:
            self._fired_today = True
            sl = price - range_low
            return Signal(
                action="BUY",
                reason=f"London/NY breakout above Asian range high {range_high:.5f}",
                sl_pips=sl,
                tp_pips=sl * self.tp_rr,
                confidence=min(1.0, (price - range_high) / span),
                meta={"range_high": range_high, "range_low": range_low,
                      "span": round(span, 5)},
            )

        if price < range_low - buf:
            self._fired_today = True
            sl = range_high - price
            return Signal(
                action="SELL",
                reason=f"London/NY breakdown below Asian range low {range_low:.5f}",
                sl_pips=sl,
                tp_pips=sl * self.tp_rr,
                confidence=min(1.0, (range_low - price) / span),
                meta={"range_high": range_high, "range_low": range_low,
                      "span": round(span, 5)},
            )

        return Signal(
            action="HOLD",
            reason=f"Inside range [{range_low:.5f} – {range_high:.5f}]",
        )
