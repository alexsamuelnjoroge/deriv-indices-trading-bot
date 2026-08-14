"""CPF Acceleration — N consecutive extreme CPF bars signal exhaustion → fade."""
from typing import Optional

from .research_base import ResearchDailyStrategy
from .base import Signal


class CPFAStrategy(ResearchDailyStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.min_streak  = config.get("min_streak", 4)
        self.cpf_high    = config.get("cpf_high", 0.7)   # close in top X% of range
        self.cpf_low     = config.get("cpf_low",  0.3)   # close in bottom X% of range
        self.signal_type = config.get("signal_type", "streak")

    @staticmethod
    def _cpf(bar: dict) -> float:
        rng = bar["high"] - bar["low"]
        if rng == 0:
            return 0.5
        return (bar["close"] - bar["low"]) / rng

    def _check_signal(self, atr, allow_long, allow_short) -> Optional[Signal]:
        bars = list(self._h1)
        N    = self.min_streak
        sl, tp = self._sl_tp(atr)

        if self.signal_type == "streak":
            # Fire while still IN the streak: last N bars all extreme CPF
            if len(bars) < N + 1:
                return None
            recent = bars[-(N + 1):-1]
            cpfs   = [self._cpf(b) for b in recent]

            if all(c >= self.cpf_high for c in cpfs) and allow_short:
                return Signal("SELL", sl_pips=sl, tp_pips=tp,
                              reason=f"CPFA sell streak={N} cpf_min={min(cpfs):.2f}")
            if all(c <= self.cpf_low for c in cpfs) and allow_long:
                return Signal("BUY", sl_pips=sl, tp_pips=tp,
                              reason=f"CPFA buy streak={N} cpf_max={max(cpfs):.2f}")

        elif self.signal_type == "reversal":
            # Fire on the FIRST bar that breaks the streak (entry on reversal)
            if len(bars) < N + 2:
                return None
            prev_bars  = bars[-(N + 2):-1]
            prev_cpfs  = [self._cpf(b) for b in prev_bars[-N:]]
            current_cpf = self._cpf(bars[-1])

            # Prior N bars all bullish, current bar breaks back down → SELL
            if (all(c >= self.cpf_high for c in prev_cpfs)
                    and current_cpf < self.cpf_high
                    and allow_short):
                return Signal("SELL", sl_pips=sl, tp_pips=tp,
                              reason=f"CPFA reversal sell after streak={N}")

            # Prior N bars all bearish, current bar breaks back up → BUY
            if (all(c <= self.cpf_low for c in prev_cpfs)
                    and current_cpf > self.cpf_low
                    and allow_long):
                return Signal("BUY", sl_pips=sl, tp_pips=tp,
                              reason=f"CPFA reversal buy after streak={N}")

        return None
