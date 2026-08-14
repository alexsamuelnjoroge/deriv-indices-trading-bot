"""Z-Score Mean Reversion — price extreme on rolling z-score → fade back to mean."""
import math
from typing import Optional

from .research_base import ResearchDailyStrategy
from .base import Signal


class ZSRStrategy(ResearchDailyStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.zscore_period   = config.get("zscore_period", 72)
        self.entry_threshold = config.get("entry_threshold", 1.5)

    def _check_signal(self, atr, allow_long, allow_short) -> Optional[Signal]:
        bars = list(self._h1)
        N = self.zscore_period
        if len(bars) < N + 1:
            return None

        closes = [b["close"] for b in bars[-N:]]
        mean_N = sum(closes) / N
        var_N  = sum((c - mean_N) ** 2 for c in closes) / N
        if var_N <= 0:
            return None
        z = (bars[-1]["close"] - mean_N) / math.sqrt(var_N)

        sl, tp = self._sl_tp(atr)

        if z >= self.entry_threshold and allow_short:
            return Signal("SELL", sl_pips=sl, tp_pips=tp,
                          reason=f"ZSR sell z={z:.2f}")
        if z <= -self.entry_threshold and allow_long:
            return Signal("BUY", sl_pips=sl, tp_pips=tp,
                          reason=f"ZSR buy z={z:.2f}")
        return None
