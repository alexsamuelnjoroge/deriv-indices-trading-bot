"""Narrow Range Compression — N compressed bars define zone; false breakout fade or true breakout follow."""
from typing import Optional

from .research_base import ResearchDailyStrategy
from .base import Signal


class NRCStrategy(ResearchDailyStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.compress_n      = config.get("compress_n", 5)
        self.compress_thresh = config.get("compress_thresh", 0.8)
        self.signal_type     = config.get("signal_type", "fade")   # "fade" | "follow"

    def _check_signal(self, atr, allow_long, allow_short) -> Optional[Signal]:
        bars = list(self._h1)
        N    = self.compress_n
        if len(bars) < N + 1:
            return None

        zone_bars  = bars[-(N + 1):-1]
        comp_h     = max(b["high"] for b in zone_bars)
        comp_l     = min(b["low"]  for b in zone_bars)
        zone_range = comp_h - comp_l

        if zone_range > self.compress_thresh * atr:
            return None

        bar    = bars[-1]
        sl, tp = self._sl_tp(atr)

        if self.signal_type == "fade":
            # Wick above zone then closes back inside → false breakout → SELL
            if bar["high"] > comp_h and bar["close"] < comp_h and allow_short:
                return Signal("SELL", sl_pips=sl, tp_pips=tp,
                              reason=f"NRC fade sell zone={zone_range/atr:.2f}ATR")
            # Wick below zone then closes back inside → false breakout → BUY
            if bar["low"] < comp_l and bar["close"] > comp_l and allow_long:
                return Signal("BUY", sl_pips=sl, tp_pips=tp,
                              reason=f"NRC fade buy zone={zone_range/atr:.2f}ATR")

        else:  # follow
            # Close above zone → genuine breakout → BUY
            if bar["close"] > comp_h and allow_long:
                return Signal("BUY", sl_pips=sl, tp_pips=tp,
                              reason=f"NRC follow buy zone={zone_range/atr:.2f}ATR")
            # Close below zone → genuine breakout → SELL
            if bar["close"] < comp_l and allow_short:
                return Signal("SELL", sl_pips=sl, tp_pips=tp,
                              reason=f"NRC follow sell zone={zone_range/atr:.2f}ATR")

        return None
