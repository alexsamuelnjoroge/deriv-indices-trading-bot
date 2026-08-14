"""Previous Day Level — first touch or failed break of PDH/PDL intraday."""
from typing import Optional

from .research_base import ResearchDailyStrategy
from .base import Signal


class PDLStrategy(ResearchDailyStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.signal_type = config.get("signal_type", "first_touch")
        self.touch_zone  = config.get("touch_zone", 0.5)   # ATR multiples

        self._day_state: dict = {}   # day_key → {pdh_touched, pdl_touched}

    def _check_signal(self, atr, allow_long, allow_short) -> Optional[Signal]:
        bar = list(self._h1)[-1]
        dk  = bar.get("epoch", 0) // 86400

        pdh_pdl = self._pdh_pdl()
        if pdh_pdl is None:
            return None
        pdh, pdl = pdh_pdl

        if dk not in self._day_state:
            self._day_state[dk] = {"pdh_touched": False, "pdl_touched": False}
        state = self._day_state[dk]

        zone   = self.touch_zone * atr
        sl, tp = self._sl_tp(atr)

        if self.signal_type == "first_touch":
            if not state["pdh_touched"] and abs(bar["close"] - pdh) <= zone:
                state["pdh_touched"] = True
                if allow_short:
                    return Signal("SELL", sl_pips=sl, tp_pips=tp,
                                  reason=f"PDL first touch sell PDH={pdh:.5f}")

            if not state["pdl_touched"] and abs(bar["close"] - pdl) <= zone:
                state["pdl_touched"] = True
                if allow_long:
                    return Signal("BUY", sl_pips=sl, tp_pips=tp,
                                  reason=f"PDL first touch buy PDL={pdl:.5f}")

        elif self.signal_type == "failed_break":
            if bar["high"] > pdh and bar["close"] < pdh and allow_short:
                state["pdh_touched"] = True
                return Signal("SELL", sl_pips=sl, tp_pips=tp,
                              reason=f"PDL failed break sell PDH={pdh:.5f}")

            if bar["low"] < pdl and bar["close"] > pdl and allow_long:
                state["pdl_touched"] = True
                return Signal("BUY", sl_pips=sl, tp_pips=tp,
                              reason=f"PDL failed break buy PDL={pdl:.5f}")

        return None
