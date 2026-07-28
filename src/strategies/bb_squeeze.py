"""
Bollinger Band Squeeze Breakout Strategy.

Validated on frxXAGUSD 1h bars (walk-forward 3/3 folds):
  BB(50, std=1.5, squeeze=30%) SL=1.00% TP=2.00% | BE WR=34.0%
  Walk-forward mean EV = +0.360

Squeeze: BB width falls into the lowest 30% of recent 100-bar widths.
Release: width expands back out; trade direction of price vs midline.
"""

from typing import Optional
from .base import BaseStrategy, Signal


class BBSqueezeStrategy(BaseStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.bb_period    = config.get("bb_period",    50)
        self.bb_std       = config.get("bb_std",      1.5)
        self.squeeze_pct  = config.get("squeeze_pct",  30)   # % percentile threshold
        self.sl_pct       = config.get("sl_pct",     0.010)
        self.tp_pct       = config.get("tp_pct",     0.020)
        self.bar_seconds  = config.get("bar_seconds", 3600)

        self._bar_closes:    list[float] = []
        self._width_history: list[float] = []
        self._bar_start:     Optional[int] = None

    def seed_candles(self, closes: list[float]) -> None:
        self._bar_closes = list(closes)
        # Pre-build width history so first live bar has context immediately
        self._rebuild_width_history()

    def _rebuild_width_history(self) -> None:
        closes = self._bar_closes
        self._width_history = []
        for i in range(self.bb_period, len(closes)):
            window = closes[i - self.bb_period:i]
            mid    = sum(window) / self.bb_period
            std    = (sum((p - mid) ** 2 for p in window) / self.bb_period) ** 0.5
            w = (2 * self.bb_std * std) / mid if mid > 0 else 0
            self._width_history.append(w)

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
        if len(self._bar_closes) < self.bb_period + 20:
            return Signal(action="HOLD",
                          reason=f"Warming ({len(self._bar_closes)}/{self.bb_period + 20})")
        return self._evaluate_bar()

    def _evaluate_bar(self) -> Signal:
        closes = self._bar_closes
        window = closes[-self.bb_period:]
        mid    = sum(window) / self.bb_period
        std    = (sum((p - mid) ** 2 for p in window) / self.bb_period) ** 0.5
        width  = (2 * self.bb_std * std) / mid if mid > 0 else 0

        self._width_history.append(width)
        if len(self._width_history) < 20:
            return Signal(action="HOLD", reason="Building width history")

        recent = sorted(self._width_history[-100:])
        idx    = max(0, int(len(recent) * self.squeeze_pct / 100) - 1)
        thresh = recent[idx]

        was_squeeze = self._width_history[-2] <= thresh if len(self._width_history) >= 2 else False
        is_squeeze  = width <= thresh

        if was_squeeze and not is_squeeze:
            direction = "BUY_RISE" if closes[-1] > mid else "BUY_FALL"
            label     = "up" if direction == "BUY_RISE" else "dn"
            return Signal(
                action=direction,
                reason=f"BB squeeze release {label} (width={width:.5f} thresh={thresh:.5f})",
                sl_pct=self.sl_pct,
                tp_pct=self.tp_pct,
            )

        state = "squeeze" if is_squeeze else "normal"
        return Signal(action="HOLD", reason=f"BB {state} (w={width:.5f})")

    def on_result(self, won: bool) -> None:
        pass
