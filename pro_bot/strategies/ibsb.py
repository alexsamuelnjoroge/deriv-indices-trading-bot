"""Inside Bar Sequence Breakout — 2+ consecutive inside bars coil energy; breakout bar → follow breakout direction."""
from typing import Optional

from .research_base import ResearchDailyStrategy
from .base import Signal


class IBSBStrategy(ResearchDailyStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.min_inside = config.get("min_inside_bars", 2)
        self.min_break  = config.get("min_break", 0.0)

        self._inside_streak = 0
        self._cluster_high: Optional[float] = None
        self._cluster_low:  Optional[float] = None

    def _evaluate(self) -> Signal:
        """Override to always track cluster state regardless of cooldown."""
        bars = list(self._h1)
        if len(bars) < 2:
            return Signal("HOLD", reason="warmup")

        bar      = bars[-1]
        prev_bar = bars[-2]
        is_inside = (bar["high"] <= prev_bar["high"] and
                     bar["low"]  >= prev_bar["low"])

        if is_inside:
            if self._inside_streak == 0:
                self._cluster_high = prev_bar["high"]
                self._cluster_low  = prev_bar["low"]
            self._inside_streak += 1
            self._cluster_high = max(self._cluster_high, bar["high"])
            self._cluster_low  = min(self._cluster_low,  bar["low"])
            return Signal("HOLD", reason="inside_bar")

        # Non-inside bar — save cluster, then always reset it
        cluster_valid  = self._inside_streak >= self.min_inside and self._cluster_high is not None
        saved_high     = self._cluster_high
        saved_low      = self._cluster_low
        saved_streak   = self._inside_streak
        self._inside_streak = 0
        self._cluster_high  = None
        self._cluster_low   = None

        if self._bar_idx - self._last_sig_i <= self.cooldown_bars:
            return Signal("HOLD", reason="cooldown")

        if not cluster_valid:
            return Signal("HOLD", reason="no_cluster")

        atr = self._atr14()
        if atr is None or atr <= 0:
            return Signal("HOLD", reason="atr_warmup")

        allow_long, allow_short = self._macro_gate()
        sl, tp = self._sl_tp(atr)

        if (bar["high"] > saved_high
                and bar["high"] - saved_high >= self.min_break * atr
                and bar["close"] > bar["open"]
                and allow_long):
            self._last_sig_i = self._bar_idx
            return Signal("BUY", sl_pips=sl, tp_pips=tp,
                          reason=f"IBSB buy str={saved_streak}")

        if (bar["low"] < saved_low
                and saved_low - bar["low"] >= self.min_break * atr
                and bar["close"] < bar["open"]
                and allow_short):
            self._last_sig_i = self._bar_idx
            return Signal("SELL", sl_pips=sl, tp_pips=tp,
                          reason=f"IBSB sell str={saved_streak}")

        return Signal("HOLD", reason="no_signal")

    def _check_signal(self, atr, allow_long, allow_short) -> Optional[Signal]:
        return None  # logic handled in _evaluate
