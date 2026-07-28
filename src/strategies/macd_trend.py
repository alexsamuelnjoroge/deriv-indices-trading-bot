"""
MACD Momentum Strategy — MACD histogram zero-cross with trend confirmation.

Validated on frxXAUUSD 1h bars (walk-forward 3/3 folds):
  MACD(12,26,9): histogram crosses above 0 while MACD line > 0 -> MULTUP
                 histogram crosses below 0 while MACD line < 0 -> MULTDOWN
  SL=0.75%, TP=2.00% on x100 multiplier | BE WR = 28.0%
  Walk-forward mean EV = +0.497

Signal fires on bar close (time-based, default 1h).
Call seed_candles(closes) at startup with historical hourly closes.
"""

from typing import Optional
from .base import BaseStrategy, Signal


class MACDTrendStrategy(BaseStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.fast_period  = config.get("macd_fast",   12)
        self.slow_period  = config.get("macd_slow",   26)
        self.signal_period = config.get("macd_signal",  9)
        self.sl_pct       = config.get("sl_pct",    0.0075)
        self.tp_pct       = config.get("tp_pct",    0.020)
        self.bar_seconds  = config.get("bar_seconds", 3600)

        self._bar_closes: list[float] = []
        self._bar_start:  Optional[int] = None

        # Rolling EMA state for live incremental updates
        self._ema_fast:   Optional[float] = None
        self._ema_slow:   Optional[float] = None
        self._ema_signal: Optional[float] = None
        self._prev_hist:  Optional[float] = None
        self._prev_macd:  Optional[float] = None

    # ------------------------------------------------------------------ #
    #  Historical seed
    # ------------------------------------------------------------------ #

    def seed_candles(self, closes: list[float]) -> None:
        """Pre-load historical hourly closes (oldest first) before live data."""
        self._bar_closes = list(closes)
        # Pre-compute EMA state so first live bar is ready immediately
        if len(closes) >= self.slow_period + self.signal_period:
            self._rebuild_ema_state()

    def _rebuild_ema_state(self) -> None:
        closes = self._bar_closes
        ka = 2.0 / (self.fast_period  + 1)
        kb = 2.0 / (self.slow_period   + 1)
        kc = 2.0 / (self.signal_period + 1)

        ema_f = sum(closes[:self.fast_period])  / self.fast_period
        ema_s = sum(closes[:self.slow_period])  / self.slow_period
        for c in closes[self.slow_period:]:
            ema_f = c * ka + ema_f * (1 - ka)
            ema_s = c * kb + ema_s * (1 - kb)
        macd_vals = []
        ema_f2 = sum(closes[:self.fast_period]) / self.fast_period
        ema_s2 = sum(closes[:self.slow_period]) / self.slow_period
        for c in closes[self.slow_period:]:
            ema_f2 = c * ka + ema_f2 * (1 - ka)
            ema_s2 = c * kb + ema_s2 * (1 - kb)
            macd_vals.append(ema_f2 - ema_s2)

        sig = sum(macd_vals[:self.signal_period]) / self.signal_period
        for m in macd_vals[self.signal_period:]:
            sig = m * kc + sig * (1 - kc)

        self._ema_fast   = ema_f
        self._ema_slow   = ema_s
        self._ema_signal = sig
        macd_now         = ema_f - ema_s
        self._prev_hist  = macd_now - sig
        self._prev_macd  = macd_now

    # ------------------------------------------------------------------ #
    #  Evaluation (called on every tick)
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

        needed = self.slow_period + self.signal_period + 2
        if len(self._bar_closes) < needed:
            return Signal(
                action="HOLD",
                reason=f"Warming candles ({len(self._bar_closes)}/{needed})",
            )

        return self._evaluate_bar()

    # ------------------------------------------------------------------ #
    #  Signal logic (runs once per bar close)
    # ------------------------------------------------------------------ #

    def _evaluate_bar(self) -> Signal:
        # Incremental EMA update if state exists, else rebuild
        if self._ema_fast is None:
            self._rebuild_ema_state()
            return Signal(action="HOLD", reason="EMA state initialised")

        price = self._bar_closes[-1]
        ka = 2.0 / (self.fast_period  + 1)
        kb = 2.0 / (self.slow_period   + 1)
        kc = 2.0 / (self.signal_period + 1)

        self._ema_fast  = price * ka + self._ema_fast  * (1 - ka)
        self._ema_slow  = price * kb + self._ema_slow  * (1 - kb)
        macd_now        = self._ema_fast - self._ema_slow
        self._ema_signal = macd_now * kc + self._ema_signal * (1 - kc)
        hist_now        = macd_now - self._ema_signal

        prev_hist = self._prev_hist
        prev_macd = self._prev_macd
        self._prev_hist = hist_now
        self._prev_macd = macd_now

        if prev_hist is None or prev_macd is None:
            return Signal(action="HOLD", reason="Waiting for prior bar")

        cross_up   = hist_now > 0 and prev_hist <= 0 and macd_now > 0
        cross_down = hist_now < 0 and prev_hist >= 0 and macd_now < 0

        if cross_up:
            return Signal(
                action="BUY_RISE",
                reason=f"MACD hist cross↑ (hist={hist_now:.5f} MACD={macd_now:.5f})",
                sl_pct=self.sl_pct,
                tp_pct=self.tp_pct,
            )
        if cross_down:
            return Signal(
                action="BUY_FALL",
                reason=f"MACD hist cross↓ (hist={hist_now:.5f} MACD={macd_now:.5f})",
                sl_pct=self.sl_pct,
                tp_pct=self.tp_pct,
            )

        return Signal(
            action="HOLD",
            reason=f"MACD hist={hist_now:.5f} line={macd_now:.5f}",
        )

    def on_result(self, won: bool) -> None:
        pass
