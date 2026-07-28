"""
EMA Crossover Strategy.

Validated on frxGBPUSD 1h bars (walk-forward 3/3 folds):
  EMA(20) crosses above EMA(50) -> MULTUP
  EMA(20) crosses below EMA(50) -> MULTDOWN
  SL=0.75% TP=2.00% | BE WR=28.0%
  Walk-forward mean EV = +0.329
"""

from typing import Optional
from .base import BaseStrategy, Signal


class EMACrossStrategy(BaseStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.ema_fast    = config.get("ema_fast",   20)
        self.ema_slow    = config.get("ema_slow",   50)
        self.sl_pct      = config.get("sl_pct",  0.0075)
        self.tp_pct      = config.get("tp_pct",  0.020)
        self.bar_seconds = config.get("bar_seconds", 3600)

        self._bar_closes:   list[float]    = []
        self._bar_start:    Optional[int]  = None
        self._prev_fast:    Optional[float] = None
        self._prev_slow:    Optional[float] = None
        self._ema_fast_val: Optional[float] = None
        self._ema_slow_val: Optional[float] = None

    def seed_candles(self, closes: list[float]) -> None:
        self._bar_closes = list(closes)
        if len(closes) >= self.ema_slow:
            self._rebuild_ema_state()

    def _rebuild_ema_state(self) -> None:
        closes = self._bar_closes
        ka = 2 / (self.ema_fast + 1)
        kb = 2 / (self.ema_slow + 1)
        vf = sum(closes[:self.ema_fast]) / self.ema_fast
        vs = sum(closes[:self.ema_slow]) / self.ema_slow
        for c in closes[self.ema_fast:]:
            vf = c * ka + vf * (1 - ka)
        for c in closes[self.ema_slow:]:
            vs = c * kb + vs * (1 - kb)
        # prev values: EMA state one bar ago
        pf = sum(closes[:self.ema_fast]) / self.ema_fast
        ps = sum(closes[:self.ema_slow]) / self.ema_slow
        for c in closes[self.ema_fast:-1]:
            pf = c * ka + pf * (1 - ka)
        for c in closes[self.ema_slow:-1]:
            ps = c * kb + ps * (1 - kb)
        self._ema_fast_val = vf
        self._ema_slow_val = vs
        self._prev_fast    = pf
        self._prev_slow    = ps

    # ------------------------------------------------------------------ #

    def evaluate(self, tick_store) -> Signal:
        price = tick_store.latest_price
        epoch = tick_store.latest_epoch
        if price is None or epoch is None:
            return Signal(action="HOLD", reason="No ticks yet")
        if self._bar_start is None:
            self._bar_start = epoch
            return Signal(action="HOLD", reason="Bar started")
        elapsed = epoch - self._bar_start
        if elapsed < self.bar_seconds:
            return Signal(action="HOLD", reason=f"In bar ({elapsed}s/{self.bar_seconds}s)")
        self._bar_closes.append(price)
        self._bar_start = epoch
        if len(self._bar_closes) < self.ema_slow + 2:
            return Signal(action="HOLD",
                          reason=f"Warming ({len(self._bar_closes)}/{self.ema_slow + 2})")
        return self._evaluate_bar()

    def _evaluate_bar(self) -> Signal:
        price = self._bar_closes[-1]
        ka    = 2 / (self.ema_fast + 1)
        kb    = 2 / (self.ema_slow + 1)

        if self._ema_fast_val is None:
            self._rebuild_ema_state()
            return Signal(action="HOLD", reason="EMA state initialised")

        prev_fast, prev_slow = self._ema_fast_val, self._ema_slow_val
        self._ema_fast_val = price * ka + self._ema_fast_val * (1 - ka)
        self._ema_slow_val = price * kb + self._ema_slow_val * (1 - kb)
        self._prev_fast    = prev_fast
        self._prev_slow    = prev_slow

        cross_up   = self._ema_fast_val > self._ema_slow_val and prev_fast <= prev_slow
        cross_down = self._ema_fast_val < self._ema_slow_val and prev_fast >= prev_slow

        if cross_up:
            return Signal(
                action="BUY_RISE",
                reason=f"EMA({self.ema_fast}) crossed above EMA({self.ema_slow})",
                sl_pct=self.sl_pct,
                tp_pct=self.tp_pct,
            )
        if cross_down:
            return Signal(
                action="BUY_FALL",
                reason=f"EMA({self.ema_fast}) crossed below EMA({self.ema_slow})",
                sl_pct=self.sl_pct,
                tp_pct=self.tp_pct,
            )

        gap = self._ema_fast_val - self._ema_slow_val
        return Signal(action="HOLD", reason=f"EMA gap={gap:.5f}")

    def on_result(self, won: bool) -> None:
        pass
