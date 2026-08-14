"""Structural High/Low Deviation — wick pierces rolling N-bar structural level, closes back inside → stop-hunt fade."""
from typing import Optional

from .research_base import ResearchDailyStrategy
from .base import Signal


class SHDStrategy(ResearchDailyStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.n_bars    = config.get("n_bars", 20)
        self.min_probe = config.get("min_probe", 0.3)

    def _check_signal(self, atr, allow_long, allow_short) -> Optional[Signal]:
        bars = list(self._h1)
        N    = self.n_bars
        if len(bars) < N + 1:
            return None

        bar         = bars[-1]
        lookback    = bars[-(N + 1):-1]
        struct_high = max(b["high"] for b in lookback)
        struct_low  = min(b["low"]  for b in lookback)

        sl, tp = self._sl_tp(atr)

        probe_up = bar["high"] - struct_high
        if (probe_up >= self.min_probe * atr
                and bar["close"] < struct_high
                and allow_short):
            return Signal("SELL", sl_pips=sl, tp_pips=tp,
                          reason=f"SHD sell probe={probe_up/atr:.3f}ATR above N{N} high")

        probe_dn = struct_low - bar["low"]
        if (probe_dn >= self.min_probe * atr
                and bar["close"] > struct_low
                and allow_long):
            return Signal("BUY", sl_pips=sl, tp_pips=tp,
                          reason=f"SHD buy probe={probe_dn/atr:.3f}ATR below N{N} low")

        return None
