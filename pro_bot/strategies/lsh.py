"""Liquidity Sweep — bar wicks beyond a key level then closes back inside → stop-hunt fade.

level_type = "pdh_pdl"  → uses previous day High/Low (XAUUSD best)
level_type = "asian"    → uses Asian session (22:00–06:59 UTC) High/Low (USDJPY best)
"""
from typing import Optional, Tuple

from .research_base import ResearchDailyStrategy
from .base import Signal

_ASIAN_START_H = 22   # UTC — previous day
_ASIAN_END_H   = 6    # UTC — current day (inclusive)
_LONDON_OPEN_H = 7    # UTC — earliest bar hour eligible to fire on Asian sweep


class LSHStrategy(ResearchDailyStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.max_sweep_atr = config.get("max_sweep_atr", 0.2)
        self.level_type    = config.get("level_type", "pdh_pdl")

    def _asian_hl(self, current_epoch: int) -> Optional[Tuple[float, float]]:
        """Asian session H/L from the 1H buffer for the current UTC day."""
        day_start  = (current_epoch // 86400) * 86400          # midnight of current day
        asian_start = day_start - 2 * 3600                      # 22:00 prev day
        asian_end   = day_start + (_ASIAN_END_H + 1) * 3600    # 07:00 current day

        asian_bars = [b for b in self._h1
                      if asian_start <= b.get("epoch", 0) < asian_end]
        if not asian_bars:
            return None
        return (max(b["high"] for b in asian_bars),
                min(b["low"]  for b in asian_bars))

    def _check_signal(self, atr, allow_long, allow_short) -> Optional[Signal]:
        bar   = list(self._h1)[-1]
        epoch = bar.get("epoch", 0)
        hour  = (epoch % 86400) // 3600

        if self.level_type == "asian":
            # Only fire after Asian session has closed (London open onwards)
            if hour < _LONDON_OPEN_H:
                return None
            levels = self._asian_hl(epoch)
        else:  # pdh_pdl
            levels = self._pdh_pdl()

        if levels is None:
            return None
        ref_h, ref_l = levels

        sl, tp = self._sl_tp(atr)

        # Wick above level, close back below → liquidity swept → SELL
        sweep_up = bar["high"] - ref_h
        if (bar["high"] > ref_h
                and bar["close"] < ref_h
                and 0 < sweep_up <= self.max_sweep_atr * atr
                and allow_short):
            return Signal("SELL", sl_pips=sl, tp_pips=tp,
                          reason=f"LSH sell sweep={sweep_up/atr:.3f}ATR above {self.level_type}")

        # Wick below level, close back above → liquidity swept → BUY
        sweep_dn = ref_l - bar["low"]
        if (bar["low"] < ref_l
                and bar["close"] > ref_l
                and 0 < sweep_dn <= self.max_sweep_atr * atr
                and allow_long):
            return Signal("BUY", sl_pips=sl, tp_pips=tp,
                          reason=f"LSH buy sweep={sweep_dn/atr:.3f}ATR below {self.level_type}")

        return None
