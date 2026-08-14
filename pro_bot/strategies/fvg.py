"""Fair Value Gap Fill — impulse bar creates untested zone; entry when price returns to fill it in the original direction."""
from typing import Optional

from .research_base import ResearchDailyStrategy
from .base import Signal


class FVGStrategy(ResearchDailyStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.impulse_atr   = config.get("impulse_atr", 1.0)
        self.max_fill_bars = config.get("max_fill_bars", 24)
        self.min_entry_atr = config.get("min_entry_atr", 0.0)

        self._zones: list = []   # active FVG zones

    def _evaluate(self) -> Signal:
        """Override to always maintain zone state regardless of cooldown."""
        atr = self._atr14()
        if atr is None or atr <= 0:
            return Signal("HOLD", reason="atr_warmup")

        bar = list(self._h1)[-1]

        # Always expire old/filled zones
        self._zones = [z for z in self._zones
                       if (self._bar_idx - z["created_i"]) <= self.max_fill_bars
                       and not z["filled"]]

        # Always create a new zone if this bar is an impulse
        body = abs(bar["close"] - bar["open"])
        if body >= self.impulse_atr * atr:
            bull = bar["close"] > bar["open"]
            self._zones.append({
                "created_i": self._bar_idx,
                "direction": "bull" if bull else "bear",
                "zone_top":  bar["close"] if bull else bar["open"],
                "zone_bot":  bar["open"]  if bull else bar["close"],
                "filled":    False,
            })

        # Fill check only when past cooldown
        if self._bar_idx - self._last_sig_i <= self.cooldown_bars:
            return Signal("HOLD", reason="cooldown")

        allow_long, allow_short = self._macro_gate()
        sl, tp = self._sl_tp(atr)

        for z in self._zones:
            if z["filled"] or z["created_i"] == self._bar_idx:
                continue

            if z["direction"] == "bull" and allow_long:
                if (bar["low"] <= z["zone_top"]
                        and bar["low"] >= z["zone_bot"] - self.min_entry_atr * atr
                        and bar["close"] >= z["zone_bot"]):
                    z["filled"] = True
                    self._last_sig_i = self._bar_idx
                    return Signal("BUY", sl_pips=sl, tp_pips=tp,
                                  reason=f"FVG bull fill zone_top={z['zone_top']:.5f}")

            elif z["direction"] == "bear" and allow_short:
                if (bar["high"] >= z["zone_bot"]
                        and bar["high"] <= z["zone_top"] + self.min_entry_atr * atr
                        and bar["close"] <= z["zone_top"]):
                    z["filled"] = True
                    self._last_sig_i = self._bar_idx
                    return Signal("SELL", sl_pips=sl, tp_pips=tp,
                                  reason=f"FVG bear fill zone_bot={z['zone_bot']:.5f}")

        return Signal("HOLD", reason="no_signal")

    def _check_signal(self, atr, allow_long, allow_short) -> Optional[Signal]:
        return None  # logic handled in _evaluate
