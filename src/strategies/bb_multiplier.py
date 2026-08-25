"""
Bollinger Band Multiplier Strategy — bar-based MULTUP/MULTDOWN signal.

Two modes controlled by bb_mode config key:

  touch (default):
    Fires when close touches or crosses the outer band — mean-reversion into
    an extreme move.

  reenter:
    Fires when close was outside the BB on the previous bar and re-enters.
    Higher conviction; fewer signals.

Validated configurations:

  Phase 1 — frxXAUUSD 5-min bars, London session (10-15 EAT):
    BB(10, 2.0σ) touch mode
    300x mult, TP=2%, SL=natural(1/300=0.33%)  WR=17.3%  Exp=+0.199R  4/4
    100x mult, TP=2%, SL=natural(1/100=1.00%)  WR=35.7%  Exp=+0.059R  4/4
    500x mult, TP=2%, SL=natural(1/500=0.20%)  WR=11.6%  Exp=+0.256R  4/4

  Phase 2 — R_75 5-min bars, 24/7:
    BB(20, 2.0σ) touch mode
    200x mult, TP=3%, SL=natural(1/200=0.50%)  WR=17.2%  Exp=+0.197R  4/4
    500x mult, TP=3%, SL=natural(1/500=0.20%)  WR=7.4%   Exp=+0.175R  4/4

Config keys:
  bb_period    : int   (default 10)     — lookback for BB calculation
  bb_std       : float (default 2.0)    — standard deviation multiplier
  bb_mode      : str   (default "touch")  — "touch" | "reenter"
  bar_seconds  : int   (default 300)    — 5-min bars
  tp_pct       : float (default 0.02)   — take-profit as fraction of price
  sl_pct       : float (default 0.0)    — if 0, uses natural stop (1/multiplier)
  multiplier   : int   (default 100)    — Deriv multiplier level (for natural SL calc)
"""

from typing import Optional
from .base import BaseStrategy, Signal


class BBMultiplierStrategy(BaseStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.bb_period   = config.get("bb_period",   10)
        self.bb_std      = config.get("bb_std",      2.0)
        self.bb_mode     = config.get("bb_mode",     "touch")
        self.bar_seconds = config.get("bar_seconds", 300)
        self.tp_pct      = config.get("tp_pct",      0.02)

        # SL: explicit override, else natural stop-out level = 1/multiplier
        mult = config.get("multiplier", 100)
        self.sl_pct = config.get("sl_pct") or (1.0 / mult if mult > 0 else 0.01)

        self._bar_closes: list[float] = []
        self._bar_start:  Optional[int] = None
        self._prev_close: Optional[float] = None
        self._prev_upper: Optional[float] = None
        self._prev_lower: Optional[float] = None

    def seed_candles(self, closes: list[float]) -> None:
        self._bar_closes = list(closes)
        if len(closes) >= 2:
            self._prev_close = closes[-2]
            bb = _bb(closes[:-1], self.bb_period, self.bb_std)
            if bb is not None:
                self._prev_upper, self._prev_lower = bb

    def evaluate(self, tick_store) -> Signal:
        price = tick_store.latest_price
        epoch = tick_store.latest_epoch
        if price is None or epoch is None:
            return Signal(action="HOLD", reason="No ticks yet")
        if self._bar_start is None:
            self._bar_start = epoch
            return Signal(action="HOLD", reason="Bar started")

        if epoch - self._bar_start < self.bar_seconds:
            return Signal(action="HOLD",
                          reason=f"In bar ({epoch - self._bar_start}s/{self.bar_seconds}s)")

        self._bar_closes.append(price)
        self._bar_start = epoch

        needed = self.bb_period + 1
        if len(self._bar_closes) < needed:
            return Signal(action="HOLD", reason=f"Warming ({len(self._bar_closes)}/{needed})")

        return self._evaluate_bar()

    def _evaluate_bar(self) -> Signal:
        closes = self._bar_closes
        close  = closes[-1]
        bb_now = _bb(closes, self.bb_period, self.bb_std)
        if bb_now is None:
            return Signal(action="HOLD", reason="BB warming")

        upper, lower = bb_now

        if self.bb_mode == "reenter":
            action, reason = _check_reenter(
                close, upper, lower,
                self._prev_close, self._prev_upper, self._prev_lower,
            )
        else:
            action, reason = _check_touch(close, upper, lower)

        self._prev_close = close
        self._prev_upper = upper
        self._prev_lower = lower

        if action == "HOLD":
            return Signal(action="HOLD", reason=reason)

        return Signal(
            action=action,
            reason=reason,
            sl_pct=self.sl_pct,
            tp_pct=self.tp_pct,
        )

    def on_result(self, won: bool) -> None:
        pass


def _bb(closes: list[float], period: int, n_std: float):
    if len(closes) < period:
        return None
    w    = closes[-period:]
    mean = sum(w) / period
    std  = (sum((x - mean) ** 2 for x in w) / period) ** 0.5
    return mean + n_std * std, mean - n_std * std


def _check_touch(close, upper, lower):
    if close <= lower:
        return "BUY_RISE", f"BB touch lower: {close:.5f}<={lower:.5f}"
    if close >= upper:
        return "BUY_FALL", f"BB touch upper: {close:.5f}>={upper:.5f}"
    return "HOLD", f"BB({close:.5f}) upper={upper:.5f} lower={lower:.5f}"


def _check_reenter(close, upper, lower, prev_close, prev_upper, prev_lower):
    if prev_close is None or prev_lower is None or prev_upper is None:
        return "HOLD", "No previous bar"
    if prev_close < prev_lower and close >= prev_lower:
        return "BUY_RISE", f"BB re-enter below: {prev_close:.5f}->{close:.5f}>={prev_lower:.5f}"
    if prev_close > prev_upper and close <= prev_upper:
        return "BUY_FALL", f"BB re-enter above: {prev_close:.5f}->{close:.5f}<={prev_upper:.5f}"
    return "HOLD", f"BB({close:.5f}) upper={upper:.5f} lower={lower:.5f}"
