"""
Consecutive Rejection Wick (CRW).

Stateless _check_signal — no extra state beyond _h1 buffer.

Logic: examine the last n_wicks bars (before current). If all n_wicks bars have
an upper wick >= wick_ratio × bar-range → bulls repeatedly failed to hold highs
→ SELL counter-trend. Mirror for lower wicks → BUY.

Deployed config (ROBUST 4/4 on USDJPY):
  n_wicks=4, wick_ratio=0.35, RR=3.0, ATR×2.0, min_range_atr=0.4

MOSTLY OK configs (3/4): XAUUSD and GBPUSD at n_wicks=3, wick_ratio=0.55.
"""

from .research_base import ResearchDailyStrategy
from .base import Signal


class CRWStrategy(ResearchDailyStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.n_wicks       = config.get("n_wicks", 4)
        self.wick_ratio    = config.get("wick_ratio", 0.35)
        self.min_range_atr = config.get("min_range_atr", 0.4)

    def _check_signal(self, atr: float,
                      allow_long: bool, allow_short: bool):
        bars = list(self._h1)
        if len(bars) < self.n_wicks + 2:
            return None

        # The n_wicks bars immediately preceding the current bar
        sequence = bars[-(self.n_wicks + 1):-1]

        upper_rej = 0
        lower_rej = 0

        for bar in sequence:
            rng = bar["high"] - bar["low"]
            if rng < self.min_range_atr * atr:
                break  # tiny bar resets the sequence requirement
            upper_wick = bar["high"] - max(bar["open"], bar["close"])
            lower_wick = min(bar["open"], bar["close"]) - bar["low"]
            if upper_wick >= self.wick_ratio * rng:
                upper_rej += 1
            if lower_wick >= self.wick_ratio * rng:
                lower_rej += 1

        sl, tp = self._sl_tp(atr)

        if upper_rej == self.n_wicks and allow_short:
            return Signal("SELL", sl_pips=sl, tp_pips=tp,
                          reason="CRW_upper_rejection")
        if lower_rej == self.n_wicks and allow_long:
            return Signal("BUY", sl_pips=sl, tp_pips=tp,
                          reason="CRW_lower_rejection")
        return None
