"""Bar Pattern Sequences — pin bar at PDH/PDL key levels."""
from typing import Optional

from .research_base import ResearchDailyStrategy
from .base import Signal


class BPSStrategy(ResearchDailyStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.pattern_type = config.get("pattern_type", "pin_bar")
        self.zone_atr     = config.get("zone_atr", 0.3)    # level proximity in ATR multiples
        self.body_mult    = config.get("body_mult", 2.0)    # shadow must be >= N × body

    def _check_signal(self, atr, allow_long, allow_short) -> Optional[Signal]:
        bar     = list(self._h1)[-1]
        pdh_pdl = self._pdh_pdl()
        if pdh_pdl is None:
            return None
        pdh, pdl = pdh_pdl

        zone     = self.zone_atr * atr
        near_pdh = abs(bar["high"] - pdh) <= zone or abs(bar["close"] - pdh) <= zone
        near_pdl = abs(bar["low"]  - pdl) <= zone or abs(bar["close"] - pdl) <= zone
        if not (near_pdh or near_pdl):
            return None

        o, h, l, c = bar["open"], bar["high"], bar["low"], bar["close"]
        body      = abs(c - o)
        up_wick   = h - max(o, c)
        dn_wick   = min(o, c) - l
        bar_range = h - l
        if bar_range == 0:
            return None

        # Minimum meaningful body/wick to avoid near-doji noise
        min_ref = max(body, atr * 0.01)
        sl, tp  = self._sl_tp(atr)

        if self.pattern_type == "pin_bar":
            # Bearish pin at PDH: large upper shadow, close in lower 40% of range
            if (near_pdh
                    and up_wick >= self.body_mult * min_ref
                    and (c - l) / bar_range < 0.4
                    and allow_short):
                return Signal("SELL", sl_pips=sl, tp_pips=tp,
                              reason=f"BPS bearish pin at PDH={pdh:.5f}")

            # Bullish pin at PDL: large lower shadow, close in upper 40% of range
            if (near_pdl
                    and dn_wick >= self.body_mult * min_ref
                    and (h - c) / bar_range < 0.4
                    and allow_long):
                return Signal("BUY", sl_pips=sl, tp_pips=tp,
                              reason=f"BPS bullish pin at PDL={pdl:.5f}")

        return None
