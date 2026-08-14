"""Asian Range Sweep — London bar wicks above/below Asian session boundary then closes back inside → stop-hunt fade."""
from typing import Optional, Tuple

from .research_base import ResearchDailyStrategy
from .base import Signal

_ASIAN_START_H = 22
_ASIAN_END_H   = 6
_LONDON_END_DEFAULT = 10   # last London hour inclusive (inclusive on hour)


class ARSStrategy(ResearchDailyStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.min_probe    = config.get("min_probe", 0.5)
        self.london_end_h = config.get("london_end_h", _LONDON_END_DEFAULT)

    def _asian_hl(self, current_epoch: int) -> Optional[Tuple[float, float]]:
        day_start   = (current_epoch // 86400) * 86400
        asian_start = day_start - 2 * 3600          # 22:00 prev day UTC
        asian_end   = day_start + (_ASIAN_END_H + 1) * 3600  # 07:00 current day

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

        # Fire only during early London session
        if not (7 <= hour <= self.london_end_h):
            return None

        levels = self._asian_hl(epoch)
        if levels is None:
            return None
        asian_high, asian_low = levels

        sl, tp = self._sl_tp(atr)

        probe_up = bar["high"] - asian_high
        if (probe_up >= self.min_probe * atr
                and bar["close"] < asian_high
                and allow_short):
            return Signal("SELL", sl_pips=sl, tp_pips=tp,
                          reason=f"ARS sell probe={probe_up/atr:.3f}ATR above Asian high")

        probe_dn = asian_low - bar["low"]
        if (probe_dn >= self.min_probe * atr
                and bar["close"] > asian_low
                and allow_long):
            return Signal("BUY", sl_pips=sl, tp_pips=tp,
                          reason=f"ARS buy probe={probe_dn/atr:.3f}ATR below Asian low")

        return None
