"""Progressive Level Tests — successive approaches to structural level with declining reach → distribution/accumulation fade."""
from typing import Optional

from .research_base import ResearchDailyStrategy
from .base import Signal


class PLTStrategy(ResearchDailyStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.level_lookback = config.get("level_lookback", 50)
        self.probe_zone     = config.get("probe_zone", 0.5)
        self.min_tests      = config.get("min_tests", 2)

        self._res_tests: list  = []   # highs of successive resistance approaches
        self._sup_tests: list  = []   # lows  of successive support approaches
        self._in_res_probe     = False
        self._in_sup_probe     = False
        self._res_peak: float  = 0.0  # current approach running peak high
        self._sup_trough: float = 0.0  # current approach running trough low

    def _evaluate(self) -> Signal:
        """Override to always track probe zone approaches regardless of cooldown."""
        bars = list(self._h1)
        N    = self.level_lookback
        if len(bars) < N + 1:
            return Signal("HOLD", reason="warmup")

        atr = self._atr14()
        if atr is None or atr <= 0:
            return Signal("HOLD", reason="atr_warmup")

        bar       = bars[-1]
        lookback  = bars[-(N + 1):-1]
        res_level = max(b["high"] for b in lookback)
        sup_level = min(b["low"]  for b in lookback)
        zone      = self.probe_zone * atr

        # ── Resistance probe tracking ──────────────────────────────────────
        in_res = bar["high"] >= res_level - zone
        sig_res = None

        if in_res:
            if not self._in_res_probe:
                # New approach begins — initialize or extend
                self._in_res_probe = True
                self._res_peak     = bar["high"]
            else:
                self._res_peak = max(self._res_peak, bar["high"])
        else:
            if self._in_res_probe:
                # Approach just ended — finalize test
                self._in_res_probe = False
                peak = self._res_peak

                if self._res_tests and peak >= self._res_tests[-1]:
                    # New peak >= previous: level was challenged/broken → reset
                    self._res_tests = [peak]
                else:
                    self._res_tests.append(peak)

                # Signal if enough declining tests
                if (len(self._res_tests) >= self.min_tests
                        and all(self._res_tests[i] < self._res_tests[i - 1]
                                for i in range(1, len(self._res_tests)))):
                    sig_res = "SELL"
                    self._res_tests = []

        # ── Support probe tracking ─────────────────────────────────────────
        in_sup = bar["low"] <= sup_level + zone
        sig_sup = None

        if in_sup:
            if not self._in_sup_probe:
                self._in_sup_probe = True
                self._sup_trough   = bar["low"]
            else:
                self._sup_trough = min(self._sup_trough, bar["low"])
        else:
            if self._in_sup_probe:
                self._in_sup_probe = False
                trough = self._sup_trough

                if self._sup_tests and trough <= self._sup_tests[-1]:
                    self._sup_tests = [trough]
                else:
                    self._sup_tests.append(trough)

                if (len(self._sup_tests) >= self.min_tests
                        and all(self._sup_tests[i] > self._sup_tests[i - 1]
                                for i in range(1, len(self._sup_tests)))):
                    sig_sup = "BUY"
                    self._sup_tests = []

        # ── Signal gate ───────────────────────────────────────────────────
        if sig_res is None and sig_sup is None:
            return Signal("HOLD", reason="no_signal")

        if self._bar_idx - self._last_sig_i <= self.cooldown_bars:
            return Signal("HOLD", reason="cooldown")

        allow_long, allow_short = self._macro_gate()
        sl, tp = self._sl_tp(atr)

        if sig_res == "SELL" and allow_short:
            self._last_sig_i = self._bar_idx
            return Signal("SELL", sl_pips=sl, tp_pips=tp,
                          reason=f"PLT sell declining res tests")

        if sig_sup == "BUY" and allow_long:
            self._last_sig_i = self._bar_idx
            return Signal("BUY", sl_pips=sl, tp_pips=tp,
                          reason=f"PLT buy rising sup tests")

        return Signal("HOLD", reason="macro_blocked")

    def _check_signal(self, atr, allow_long, allow_short) -> Optional[Signal]:
        return None  # logic handled in _evaluate
