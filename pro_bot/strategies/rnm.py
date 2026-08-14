"""Round Number Magnet — bar wicks to a round-number grid level then bounces away → follow bounce."""
from typing import Optional

from .research_base import ResearchDailyStrategy
from .base import Signal

# Round number grid spacing per symbol prefix
_GRIDS = {
    "XAUUSD": 50.0,
    "EURUSD": 0.0050,
    "GBPUSD": 0.0050,
    "USDJPY": 1.0,
}
_DEFAULT_GRID = 0.0050


def _nearest_round(price: float, grid: float) -> float:
    return round(round(price / grid) * grid, 10)


class RNMStrategy(ResearchDailyStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.touch_zone = config.get("touch_zone", 1.0)
        self.min_bounce = config.get("min_bounce", 0.4)
        self._symbol    = config.get("symbols", [""])[0] if isinstance(config.get("symbols"), list) else ""
        self._grid      = _GRIDS.get(self._symbol, _DEFAULT_GRID)
        self._level_last: dict = {}   # round_level → last bar index that fired

    def _check_signal(self, atr, allow_long, allow_short) -> Optional[Signal]:
        bars = list(self._h1)
        bar  = bars[-1]
        sl, tp = self._sl_tp(atr)
        zone   = self.touch_zone * atr
        bounce = self.min_bounce * atr

        # Find round number closest to bar's extremes
        for candidate in (bar["low"], bar["high"]):
            level = _nearest_round(candidate, self._grid)
            last  = self._level_last.get(level, -9999)
            if self._bar_idx - last < self.cooldown_bars:
                continue

            # Bar wicked into the zone around the round number
            if abs(candidate - level) > zone:
                continue

            dist_close = bar["close"] - level
            if dist_close >= bounce and allow_long:
                self._level_last[level] = self._bar_idx
                return Signal("BUY", sl_pips=sl, tp_pips=tp,
                              reason=f"RNM buy bounce {dist_close/atr:.3f}ATR above {level}")

            if -dist_close >= bounce and allow_short:
                self._level_last[level] = self._bar_idx
                return Signal("SELL", sl_pips=sl, tp_pips=tp,
                              reason=f"RNM sell bounce {-dist_close/atr:.3f}ATR below {level}")

        return None
