"""Order Flow Balance Shift — rolling CPF average flips from bull zone to bear zone (or vice versa) → institutional reversal."""
from typing import Optional

from .research_base import ResearchDailyStrategy
from .base import Signal


class OFBSStrategy(ResearchDailyStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.avg_period  = config.get("avg_period", 15)
        self.bull_thresh = config.get("bull_thresh", 0.6)
        self.bear_thresh = config.get("bear_thresh", 0.35)

    @staticmethod
    def _cpf(bar: dict) -> float:
        rng = bar["high"] - bar["low"]
        if rng == 0:
            return 0.5
        return (bar["close"] - bar["low"]) / rng

    def _check_signal(self, atr, allow_long, allow_short) -> Optional[Signal]:
        bars = list(self._h1)
        N    = self.avg_period
        if len(bars) < N + 1:
            return None

        prior_bars = bars[-(N + 1):-1]
        avg_prior  = sum(self._cpf(b) for b in prior_bars) / len(prior_bars)
        cur_cpf    = self._cpf(bars[-1])
        sl, tp     = self._sl_tp(atr)

        # Prior bars bullish (top of ranges), current flips to bearish → SELL
        if avg_prior > self.bull_thresh and cur_cpf < self.bear_thresh and allow_short:
            return Signal("SELL", sl_pips=sl, tp_pips=tp,
                          reason=f"OFBS sell avg_cpf={avg_prior:.3f} cur={cur_cpf:.3f}")

        # Prior bars bearish (bottom of ranges), current flips to bullish → BUY
        if avg_prior < (1 - self.bull_thresh) and cur_cpf > (1 - self.bear_thresh) and allow_long:
            return Signal("BUY", sl_pips=sl, tp_pips=tp,
                          reason=f"OFBS buy avg_cpf={avg_prior:.3f} cur={cur_cpf:.3f}")

        return None
