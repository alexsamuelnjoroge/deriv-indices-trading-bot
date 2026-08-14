"""London-NY Divergence — fade large London session displacement at NY open."""
from typing import Optional

from .research_base import ResearchDailyStrategy
from .base import Signal

_LONDON_START = 7    # UTC hour London session opens
_LONDON_END   = 12   # UTC hour London tracking ends (inclusive)
_NY_START     = 13   # UTC hour NY window begins


class LNDStrategy(ResearchDailyStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.min_london_atr = config.get("min_london_atr", 1.5)
        self.ny_window      = config.get("ny_window", 2)   # hours at NY open to accept signal

        # Per-day tracking keyed by UTC day number (epoch // 86400)
        self._lon_open:  dict = {}   # day_key → open price at London start
        self._lon_disp:  dict = {}   # day_key → ATR-normalised displacement at London end
        self._fired:     dict = {}   # day_key → bool

    def _check_signal(self, atr, allow_long, allow_short) -> Optional[Signal]:
        bar   = list(self._h1)[-1]
        epoch = bar.get("epoch", 0)
        hour  = (epoch % 86400) // 3600
        dk    = epoch // 86400

        # During London: record open and track cumulative displacement
        if _LONDON_START <= hour <= _LONDON_END:
            if dk not in self._lon_open:
                self._lon_open[dk] = bar["open"]
            self._lon_disp[dk] = (bar["close"] - self._lon_open[dk]) / atr
            return None

        # NY window: fire if London displacement was large enough
        if _NY_START <= hour < _NY_START + self.ny_window:
            if self._fired.get(dk, False):
                return None
            disp = self._lon_disp.get(dk)
            if disp is None:
                return None

            sl, tp = self._sl_tp(atr)

            if disp >= self.min_london_atr and allow_short:
                self._fired[dk] = True
                return Signal("SELL", sl_pips=sl, tp_pips=tp,
                              reason=f"LND sell london={disp:.2f}ATR")
            if disp <= -self.min_london_atr and allow_long:
                self._fired[dk] = True
                return Signal("BUY", sl_pips=sl, tp_pips=tp,
                              reason=f"LND buy london={disp:.2f}ATR")

        return None
