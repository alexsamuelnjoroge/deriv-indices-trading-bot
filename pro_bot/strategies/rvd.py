"""Return Velocity Decay — ATR-normalized close-to-close returns shrink over consecutive bars → counter-trend fade."""
from typing import Optional

from .research_base import ResearchDailyStrategy
from .base import Signal


class RVDStrategy(ResearchDailyStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.min_streak = config.get("min_streak", 4)
        self.decay_bars = config.get("decay_bars", 2)
        self.min_ret    = config.get("min_ret", 0.1)

    def _check_signal(self, atr, allow_long, allow_short) -> Optional[Signal]:
        bars = list(self._h1)
        if len(bars) < self.min_streak + 5:
            return None

        # Build ATR-normalized returns for recent bars
        norm_rets = []
        for i in range(max(1, len(bars) - (self.min_streak + 4)), len(bars)):
            a = atr  # use current ATR as approximation
            if a and a > 0:
                r = (bars[i]["close"] - bars[i - 1]["close"]) / a
                norm_rets.append(r)
            else:
                norm_rets.append(None)

        if not norm_rets or norm_rets[-1] is None:
            return None

        r_i = norm_rets[-1]
        if abs(r_i) < self.min_ret:
            return None

        # Build same-direction streak from most recent bar backwards
        streak = [r_i]
        for r in reversed(norm_rets[:-1]):
            if r is None:
                break
            if r * r_i <= 0:  # different direction
                break
            if abs(r) < self.min_ret:
                break
            streak.append(r)

        if len(streak) < self.min_streak:
            return None
        if len(streak) < self.decay_bars:
            return None

        # streak[0] = current bar, streak[1] = prior bar, etc.
        # Check that each is strictly smaller in magnitude than the previous
        magnitudes = [abs(streak[k]) for k in range(self.decay_bars)]
        if not all(magnitudes[k] < magnitudes[k + 1] for k in range(self.decay_bars - 1)):
            return None

        sl, tp = self._sl_tp(atr)

        if r_i > 0 and allow_short:
            return Signal("SELL", sl_pips=sl, tp_pips=tp,
                          reason=f"RVD sell str={len(streak)} decay={self.decay_bars}")

        if r_i < 0 and allow_long:
            return Signal("BUY", sl_pips=sl, tp_pips=tp,
                          reason=f"RVD buy str={len(streak)} decay={self.decay_bars}")

        return None
