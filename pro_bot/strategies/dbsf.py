"""Daily Bar Streak Filter — N consecutive same-direction daily closes → fade at London open."""
from typing import Optional

from .research_base import ResearchDailyStrategy
from .base import Signal


class DBSFStrategy(ResearchDailyStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.min_streak = config.get("min_streak", 2)
        self.entry_hour = config.get("entry_hour", 7)   # UTC hour to begin watching

        self._today_key   = None
        self._fired_today = False

    def feed(self, bar: dict) -> Signal:
        day_key = str(bar.get("epoch", 0) // 86400)
        if day_key != self._today_key:
            self._today_key   = day_key
            self._fired_today = False
        return super().feed(bar)

    def _check_signal(self, atr, allow_long, allow_short) -> Optional[Signal]:
        if self._fired_today:
            return None

        bar      = list(self._h1)[-1]
        bar_hour = (bar.get("epoch", 0) % 86400) // 3600

        if bar_hour < self.entry_hour or bar_hour > self.entry_hour + 1:
            return None

        d1 = list(self._d1)
        if len(d1) < self.min_streak + 1:
            return None

        streak    = 0
        direction = None
        for i in range(len(d1) - 1, 0, -1):
            move = d1[i]["close"] - d1[i]["open"]
            if move == 0:
                break
            bar_dir = 1 if move > 0 else -1
            if direction is None:
                direction = bar_dir
            elif bar_dir != direction:
                break
            streak += 1
            if streak >= self.min_streak:
                break

        if streak < self.min_streak or direction is None:
            return None

        sl, tp = self._sl_tp(atr)
        self._fired_today = True

        if direction == 1 and allow_short:
            return Signal("SELL", sl_pips=sl, tp_pips=tp,
                          reason=f"DBSF sell streak={streak} bull days")
        if direction == -1 and allow_long:
            return Signal("BUY", sl_pips=sl, tp_pips=tp,
                          reason=f"DBSF buy streak={streak} bear days")
        return None
