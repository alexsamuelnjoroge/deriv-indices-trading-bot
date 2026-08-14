"""
ResearchDailyStrategy — shared base for all walk-forward validated 1H strategies.

Subclasses override _check_signal(atr, allow_long, allow_short) to return Signal or None.
feed_daily(bar) receives completed daily bars.
feed(bar)       receives completed 1H bars and emits signals.
_seed_h1(bars)  pre-populates the 1H buffer without triggering signal evaluation.
"""
from collections import deque
from typing import Optional, Tuple

from .base import BaseProStrategy, Signal


class ResearchDailyStrategy(BaseProStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self._h1: deque = deque(maxlen=300)
        self._d1: deque = deque(maxlen=120)
        self._bar_idx    = 0
        self._last_sig_i = -9999

        self.cooldown_bars    = config.get("cooldown_bars", 12)
        self.tp_rr            = config.get("tp_rr", 2.0)
        self.atr_mult_sl      = config.get("atr_mult_sl", 1.5)
        self.macro_filter     = config.get("macro_filter", False)
        self.macro_ema_period = config.get("macro_ema_period", 20)

    # ── Public feed interface ────────────────────────────────────────────────

    def feed_daily(self, bar: dict) -> None:
        self._d1.append(bar)

    def _seed_h1(self, bars: list) -> None:
        for bar in bars:
            self._h1.append(bar)
            self._bar_idx += 1

    def feed(self, bar: dict) -> Signal:
        self._h1.append(bar)
        self._bar_idx += 1
        return self._evaluate()

    # ── Core evaluation ──────────────────────────────────────────────────────

    def _evaluate(self) -> Signal:
        if self._bar_idx - self._last_sig_i <= self.cooldown_bars:
            return Signal("HOLD", reason="cooldown")
        atr = self._atr14()
        if atr is None or atr <= 0:
            return Signal("HOLD", reason="atr_warmup")
        allow_long, allow_short = self._macro_gate()
        sig = self._check_signal(atr, allow_long, allow_short)
        if sig is not None and sig.action != "HOLD":
            self._last_sig_i = self._bar_idx
            return sig
        return Signal("HOLD", reason="no_signal")

    def _check_signal(self, atr: float,
                      allow_long: bool, allow_short: bool) -> Optional[Signal]:
        raise NotImplementedError

    # ── Common helpers ───────────────────────────────────────────────────────

    def _atr14(self) -> Optional[float]:
        bars = list(self._h1)
        if len(bars) < 15:
            return None
        trs = []
        for i in range(max(1, len(bars) - 14), len(bars)):
            h, l, cp = bars[i]["high"], bars[i]["low"], bars[i - 1]["close"]
            trs.append(max(h - l, abs(h - cp), abs(l - cp)))
        return sum(trs) / len(trs) if trs else None

    def _macro_gate(self) -> Tuple[bool, bool]:
        if not self.macro_filter:
            return True, True
        d1 = list(self._d1)
        p  = self.macro_ema_period
        if len(d1) < p + 1:
            return True, True
        closes   = [b["close"] for b in d1]
        k        = 2 / (p + 1)
        ema      = sum(closes[:p]) / p
        ema_prev = ema
        for c in closes[p:]:
            ema_prev = ema
            ema      = c * k + ema * (1 - k)
        return ema > ema_prev, ema <= ema_prev

    def _sl_tp(self, atr: float) -> Tuple[float, float]:
        sl = atr * self.atr_mult_sl
        return sl, sl * self.tp_rr

    def _pdh_pdl(self) -> Optional[Tuple[float, float]]:
        if not self._d1:
            return None
        d = self._d1[-1]
        return d["high"], d["low"]
