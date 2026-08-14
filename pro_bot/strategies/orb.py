"""Opening Range Breakout — London 07:00 bar defines equilibrium; subsequent bar closes beyond it with momentum → follow."""
from typing import Optional

from .research_base import ResearchDailyStrategy
from .base import Signal


class ORBStrategy(ResearchDailyStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.min_orb_atr = config.get("min_orb_atr", 0.5)
        self.max_orb_atr = config.get("max_orb_atr", 3.0)
        self.min_break   = config.get("min_break", 0.2)
        self.max_entry_h = config.get("max_entry_h", 10)

        self._orb_map: dict = {}   # day_midnight → (orb_high, orb_low, orb_atr)

    def _record_orb(self, bar: dict, atr: float):
        epoch = bar.get("epoch", 0)
        h     = (epoch % 86400) // 3600
        if h == 7 and atr and atr > 0:
            day = (epoch // 86400) * 86400
            if day not in self._orb_map:
                self._orb_map[day] = (bar["high"], bar["low"], atr)

    def feed(self, bar: dict):
        # Record ORB before evaluating
        bars  = list(self._h1)
        bars.append(bar)
        # Compute running ATR for the 07:00 bar
        if len(bars) >= 15:
            trs = []
            for i in range(max(1, len(bars) - 14), len(bars)):
                h2, l2, cp = bars[i]["high"], bars[i]["low"], bars[i - 1]["close"]
                trs.append(max(h2 - l2, abs(h2 - cp), abs(l2 - cp)))
            atr_now = sum(trs) / len(trs) if trs else 0
        else:
            atr_now = 0
        self._record_orb(bar, atr_now)
        return super().feed(bar)

    def _seed_h1(self, bars: list) -> None:
        for i, bar in enumerate(bars):
            if i >= 14:
                trs = []
                for j in range(i - 13, i + 1):
                    h2, l2, cp = bars[j]["high"], bars[j]["low"], bars[j - 1]["close"]
                    trs.append(max(h2 - l2, abs(h2 - cp), abs(l2 - cp)))
                atr_s = sum(trs) / len(trs) if trs else 0
                self._record_orb(bar, atr_s)
            self._h1.append(bar)
            self._bar_idx += 1

    def _check_signal(self, atr, allow_long, allow_short) -> Optional[Signal]:
        bar   = list(self._h1)[-1]
        epoch = bar.get("epoch", 0)
        h     = (epoch % 86400) // 3600

        # Only fire after ORB bar (h=7), during h=8..max_entry_h
        if not (8 <= h <= self.max_entry_h):
            return None

        day = (epoch // 86400) * 86400
        orb = self._orb_map.get(day)
        if orb is None:
            return None

        orb_high, orb_low, orb_atr = orb
        orb_size = orb_high - orb_low

        if orb_size < self.min_orb_atr * orb_atr:
            return None
        if orb_size > self.max_orb_atr * orb_atr:
            return None

        sl, tp = self._sl_tp(atr)

        if (bar["close"] > orb_high
                and bar["close"] - orb_high >= self.min_break * atr
                and bar["close"] > bar["open"]
                and allow_long):
            return Signal("BUY", sl_pips=sl, tp_pips=tp,
                          reason=f"ORB buy break={bar['close'] - orb_high:.5f}")

        if (bar["close"] < orb_low
                and orb_low - bar["close"] >= self.min_break * atr
                and bar["close"] < bar["open"]
                and allow_short):
            return Signal("SELL", sl_pips=sl, tp_pips=tp,
                          reason=f"ORB sell break={orb_low - bar['close']:.5f}")

        return None
