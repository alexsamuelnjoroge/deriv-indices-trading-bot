"""Candle Body Velocity Exhaustion — accel_bars growing bodies followed by tiny exhaustion bar → counter-trend fade."""
from typing import Optional

from .research_base import ResearchDailyStrategy
from .base import Signal


class CBVEStrategy(ResearchDailyStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.accel_bars        = config.get("accel_bars", 2)
        self.exhaust_ratio     = config.get("exhaust_ratio", 0.25)
        self.avg_bars          = config.get("avg_bars", 5)
        self.min_accel_body_mult = config.get("min_accel_body_mult", 1.0)

    def _check_signal(self, atr, allow_long, allow_short) -> Optional[Signal]:
        bars = list(self._h1)
        A    = self.accel_bars
        needed = A + self.avg_bars + 2
        if len(bars) < needed:
            return None

        bar = bars[-1]

        # Rolling average body (excludes current bar)
        avg_window   = bars[-(self.avg_bars + 1):-1]
        avg_body     = sum(abs(b["close"] - b["open"]) for b in avg_window) / len(avg_window)
        if avg_body <= 0:
            return None

        # Current bar must be the exhaustion (tiny body)
        cur_body = abs(bar["close"] - bar["open"])
        if cur_body >= self.exhaust_ratio * avg_body:
            return None

        # Previous accel_bars bars: same direction + strictly growing bodies
        prev = bars[-(A + 1):-1]
        prev_bodies = [abs(b["close"] - b["open"]) for b in prev]

        # All prev bodies must be meaningful (not dojis)
        if any(pb < self.min_accel_body_mult * avg_body * 0.5 for pb in prev_bodies):
            return None

        # Strictly expanding
        if not all(prev_bodies[j] > prev_bodies[j - 1] for j in range(1, len(prev_bodies))):
            return None

        bull_count = sum(1 for b in prev if b["close"] > b["open"])
        bear_count = sum(1 for b in prev if b["close"] < b["open"])

        sl, tp = self._sl_tp(atr)

        if bull_count == A and allow_short:
            return Signal("SELL", sl_pips=sl, tp_pips=tp,
                          reason=f"CBVE sell bull exhaust avg_body={avg_body:.5f}")

        if bear_count == A and allow_long:
            return Signal("BUY", sl_pips=sl, tp_pips=tp,
                          reason=f"CBVE buy bear exhaust avg_body={avg_body:.5f}")

        return None
