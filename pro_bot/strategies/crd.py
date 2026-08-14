"""Candle Rhythm Disruption — anomalously large candle relative to recent rhythm = institutional order → trade its direction (or fade in fade_mode)."""
from typing import Optional

from .research_base import ResearchDailyStrategy
from .base import Signal

_MIN_PREV_RANGE_RATIO = 0.10  # prev bar range must be > this × ATR to avoid inflated ratios


class CRDStrategy(ResearchDailyStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.avg_period     = config.get("avg_period", 10)
        self.disrupt_mult   = config.get("disruption_mult", 2.5)
        self.min_body_ratio = config.get("min_body_ratio", 0.5)
        self.fade_mode      = config.get("fade_mode", False)

    def _check_signal(self, atr, allow_long, allow_short) -> Optional[Signal]:
        bars = list(self._h1)
        P    = self.avg_period
        if len(bars) < P + 2:
            return None

        bar      = bars[-1]
        prev_bar = bars[-2]

        prev_range = prev_bar["high"] - prev_bar["low"]
        if prev_range < _MIN_PREV_RANGE_RATIO * atr:
            return None

        cur_range = bar["high"] - bar["low"]
        if cur_range <= 0:
            return None

        body = abs(bar["close"] - bar["open"])
        if body < self.min_body_ratio * cur_range:
            return None

        # Rolling average of range ratios over prior avg_period bars
        ratios = []
        for i in range(len(bars) - P - 1, len(bars) - 1):
            if i < 1:
                continue
            r_j   = bars[i]["high"]     - bars[i]["low"]
            r_j_1 = bars[i - 1]["high"] - bars[i - 1]["low"]
            if r_j_1 > _MIN_PREV_RANGE_RATIO * atr:
                ratios.append(r_j / r_j_1)

        if len(ratios) < P // 2:
            return None

        avg_ratio = sum(ratios) / len(ratios)
        cur_ratio = cur_range / prev_range

        if cur_ratio <= self.disrupt_mult * avg_ratio:
            return None

        sl, tp = self._sl_tp(atr)
        bar_bull = bar["close"] > bar["open"]
        go_long  = bar_bull if not self.fade_mode else not bar_bull

        if go_long and allow_long:
            return Signal("BUY", sl_pips=sl, tp_pips=tp,
                          reason=f"CRD buy ratio={cur_ratio:.1f}x {'fade' if self.fade_mode else 'trend'}")

        if not go_long and allow_short:
            return Signal("SELL", sl_pips=sl, tp_pips=tp,
                          reason=f"CRD sell ratio={cur_ratio:.1f}x {'fade' if self.fade_mode else 'trend'}")

        return None
