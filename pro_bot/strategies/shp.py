"""
Session Handoff Pullback (SHP).

Stateful — overrides _evaluate() to track intraday session data across bars.

Logic:
  1. Accumulate H1 bars during the London session (7-12 UTC) to measure net displacement.
  2. During the NY open window (13-16 UTC), track the pullback against the London trend.
  3. When pullback is in the valid retracement range AND current bar rejects in the
     London trend direction → enter in the trend direction.
  4. One signal per calendar day.

Deployed config (MOSTLY OK 3/4 on XAUUSD):
  min_displacement_atr=0.5, pullback_min=0.20, pullback_max=0.60, RR=1.5, ATR×1.5
"""

from .research_base import ResearchDailyStrategy
from .base import Signal

LONDON_START = 7
LONDON_END   = 12
NY_START     = 13
NY_END       = 16


class SHPStrategy(ResearchDailyStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.min_disp_atr  = config.get("min_displacement_atr", 0.5)
        self.pullback_min  = config.get("pullback_min_pct", 0.20)
        self.pullback_max  = config.get("pullback_max_pct", 0.60)

        # Per-day state — reset each new calendar day
        self._current_day        = -1
        self._london_open_price  = None
        self._london_disp        = None  # set at hour == LONDON_END
        self._ny_pullback_lo     = None
        self._ny_pullback_hi     = None
        self._signalled_today    = False

    def _reset_day(self, day: int) -> None:
        self._current_day       = day
        self._london_open_price = None
        self._london_disp       = None
        self._ny_pullback_lo    = None
        self._ny_pullback_hi    = None
        self._signalled_today   = False

    def _evaluate(self) -> Signal:
        bars = list(self._h1)
        if not bars:
            return Signal("HOLD", reason="no_bars")

        bar   = bars[-1]
        epoch = bar["epoch"]
        hour  = (epoch // 3600) % 24
        day   = epoch // 86400

        if day != self._current_day:
            self._reset_day(day)

        atr = self._atr14()
        if atr is None or atr <= 0:
            return Signal("HOLD", reason="atr_warmup")

        # ── London accumulation ──────────────────────────────────────────────
        if LONDON_START <= hour <= LONDON_END:
            if self._london_open_price is None:
                self._london_open_price = bar["open"]
            if hour == LONDON_END:
                self._london_disp = bar["close"] - self._london_open_price
            return Signal("HOLD", reason="london_session")

        # ── NY window evaluation ─────────────────────────────────────────────
        if NY_START <= hour <= NY_END:
            if self._signalled_today:
                return Signal("HOLD", reason="already_signalled")
            if self._london_disp is None:
                return Signal("HOLD", reason="no_london_disp")
            if abs(self._london_disp) < self.min_disp_atr * atr:
                return Signal("HOLD", reason="london_too_small")

            # Cooldown check (inherited)
            if self._bar_idx - self._last_sig_i <= self.cooldown_bars:
                return Signal("HOLD", reason="cooldown")

            lon_ref = self._london_open_price + self._london_disp
            london_move = abs(self._london_disp)

            sl, tp = self._sl_tp(atr)

            if self._london_disp > 0:
                # Bullish London → track NY pullback low
                if self._ny_pullback_lo is None:
                    self._ny_pullback_lo = bar["low"]
                else:
                    self._ny_pullback_lo = min(self._ny_pullback_lo, bar["low"])

                retrace = (lon_ref - self._ny_pullback_lo) / london_move
                if self.pullback_min <= retrace <= self.pullback_max:
                    if bar["close"] > bar["open"]:
                        self._last_sig_i     = self._bar_idx
                        self._signalled_today = True
                        return Signal("BUY", sl_pips=sl, tp_pips=tp,
                                      reason="SHP_bull_resume")

            elif self._london_disp < 0:
                # Bearish London → track NY pullback high
                if self._ny_pullback_hi is None:
                    self._ny_pullback_hi = bar["high"]
                else:
                    self._ny_pullback_hi = max(self._ny_pullback_hi, bar["high"])

                retrace = (self._ny_pullback_hi - lon_ref) / london_move
                if self.pullback_min <= retrace <= self.pullback_max:
                    if bar["close"] < bar["open"]:
                        self._last_sig_i     = self._bar_idx
                        self._signalled_today = True
                        return Signal("SELL", sl_pips=sl, tp_pips=tp,
                                      reason="SHP_bear_resume")

        return Signal("HOLD", reason="no_signal")
