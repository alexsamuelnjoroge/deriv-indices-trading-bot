"""NY Reversal Fade — large London move faded at NY open on M15 XAUUSD.

Logic:
  1. Record the London open price (07:00 UTC bar open) each day.
  2. At NY open (13:00–16:00 UTC), if close has moved ≥ min_london_atr × ATR(14)
     away from the London open, fade that move.
  London ran up  → SELL  (profit-taking / US open fades European highs)
  London ran down → BUY  (same, inverse)

Validated MOSTLY OK 3/4 walk-forward windows on XAUUSD M15.
Best config: min_london_atr=1.5, ny_start_h=13, ny_end_h=16,
             atr_mult_sl=1.5, tp_rr=3.0, macro_filter=True
"""

from collections import deque
from typing import Optional

from .research_base import ResearchDailyStrategy
from .base import Signal


class NRFStrategy(ResearchDailyStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.min_london_atr = config.get("min_london_atr", 1.5)
        self.ny_start_h     = config.get("ny_start_h",    13)
        self.ny_end_h       = config.get("ny_end_h",      16)
        self._london_open: dict[int, float] = {}   # day (epoch//86400) → 07:00 open

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _day(epoch: int) -> int:
        return epoch // 86400

    @staticmethod
    def _utc_hour(epoch: int) -> int:
        return (epoch % 86400) // 3600

    def _record_london_open(self, bar: dict) -> None:
        epoch = bar.get("epoch", 0)
        if self._utc_hour(epoch) == 7 and (epoch % 3600) // 60 == 0:
            self._london_open[self._day(epoch)] = bar["open"]

    # ── Seeding ───────────────────────────────────────────────────────────────

    def _seed_h1(self, bars: list) -> None:
        super()._seed_h1(bars)
        for bar in bars:
            self._record_london_open(bar)

    # ── Per-bar update ────────────────────────────────────────────────────────

    def feed(self, bar: dict) -> Signal:
        self._record_london_open(bar)
        return super().feed(bar)

    # ── Signal logic ──────────────────────────────────────────────────────────

    def _check_signal(self, atr: float,
                      allow_long: bool, allow_short: bool) -> Optional[Signal]:
        bars = list(self._h1)
        if not bars:
            return None

        bar   = bars[-1]
        epoch = bar.get("epoch", 0)
        h     = self._utc_hour(epoch)
        day   = self._day(epoch)

        # NY session only
        if not (self.ny_start_h <= h < self.ny_end_h):
            return None

        lo_price = self._london_open.get(day)
        if lo_price is None:
            return None

        close        = bar["close"]
        displacement = close - lo_price
        threshold    = self.min_london_atr * atr

        if abs(displacement) < threshold:
            return None

        sl, tp = self._sl_tp(atr)

        # London moved UP → fade with SELL
        if displacement > 0 and allow_short:
            return Signal("SELL", sl_pips=sl, tp_pips=tp,
                          reason=f"NRF sell London+{displacement:.1f}>{threshold:.1f}")

        # London moved DOWN → fade with BUY
        if displacement < 0 and allow_long:
            return Signal("BUY",  sl_pips=sl, tp_pips=tp,
                          reason=f"NRF buy London{displacement:.1f}<-{threshold:.1f}")

        return None
