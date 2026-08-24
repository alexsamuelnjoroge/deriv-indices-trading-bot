"""
Bollinger Band Binary Strategy — bar-based CALL/PUT signal.

Two modes controlled by bb_mode config key:

  reenter (default):
    Fires when close was outside the BB on the previous bar and re-enters
    inside the band on the current bar.  Represents a sharp spike that
    reverses — high-conviction mean-reversion setup.

    Validated on frxXAUUSD 5-min bars (67k candles, 4-fold walk-forward):
      BB(10, 2.5σ) + 10-min hold + ATR 1.25x gate → WR=65.3% BE=55.6% EV=+7.83% 4/4

  touch:
    Fires when close touches or crosses the outer band.
    Lower threshold — more signals, lower WR.

    Validated: BB(10, 2.0σ) touch + 5-min hold + London + ATR 1.25x
               WR=60.4% EV=+3.88% 4/4

ATR gate (optional): only fire when close-ATR >= atr_min_mult * 100-bar mean ATR.
Set atr_min_mult: 0.0 to disable (default).
"""

from typing import Optional
from .base import BaseStrategy, Signal


class BBBinaryStrategy(BaseStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.bb_period    = config.get("bb_period",    10)
        self.bb_std       = config.get("bb_std",       2.5)
        self.bb_mode      = config.get("bb_mode",      "reenter")  # "reenter" | "touch"
        self.bar_seconds  = config.get("bar_seconds",  300)
        self.atr_period   = config.get("atr_period",   14)
        self.atr_min_mult = config.get("atr_min_mult", 0.0)

        self._bar_closes: list[float] = []
        self._bar_start: Optional[int] = None
        self._prev_close: Optional[float] = None
        self._prev_lower: Optional[float] = None
        self._prev_upper: Optional[float] = None

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
            return Signal(action="HOLD", reason=f"In bar ({epoch - self._bar_start}s/{self.bar_seconds}s)")

        self._bar_closes.append(price)
        self._bar_start = epoch

        needed = self.bb_period + 1
        if len(self._bar_closes) < needed:
            return Signal(action="HOLD", reason=f"Warming ({len(self._bar_closes)}/{needed})")

        return self._evaluate_bar()

    def _evaluate_bar(self) -> Signal:
        closes = self._bar_closes
        current_close = closes[-1]
        bb_now = _bb(closes, self.bb_period, self.bb_std)
        if bb_now is None:
            return Signal(action="HOLD", reason="BB warming")

        upper, lower = bb_now

        if self.bb_mode == "reenter":
            action, reason = self._check_reenter(
                current_close, upper, lower,
                self._prev_close, self._prev_upper, self._prev_lower,
            )
        else:
            action, reason = self._check_touch(current_close, upper, lower)

        # Update previous bar state
        self._prev_close = current_close
        self._prev_upper = upper
        self._prev_lower = lower

        if action == "HOLD":
            return Signal(action="HOLD", reason=reason)

        if self.atr_min_mult > 0:
            atr, atr_mean = _close_atr(closes, self.atr_period)
            if atr is None or atr_mean is None or atr < self.atr_min_mult * atr_mean:
                return Signal(
                    action="HOLD",
                    reason=f"ATR gate: {atr:.5f if atr else 'n/a'} < {self.atr_min_mult}x mean",
                )

        return Signal(action=action, reason=reason)

    @staticmethod
    def _check_reenter(close, upper, lower, prev_close, prev_upper, prev_lower):
        if prev_close is None or prev_lower is None or prev_upper is None:
            return "HOLD", "No previous bar"
        if prev_close < prev_lower and close >= prev_lower:
            return "BUY_RISE", f"BB re-enter from below: prev={prev_close:.2f}<{prev_lower:.2f}, now={close:.2f}"
        if prev_close > prev_upper and close <= prev_upper:
            return "BUY_FALL", f"BB re-enter from above: prev={prev_close:.2f}>{prev_upper:.2f}, now={close:.2f}"
        return "HOLD", f"BB({close:.2f}) upper={upper:.2f} lower={lower:.2f}"

    @staticmethod
    def _check_touch(close, upper, lower):
        if close <= lower:
            return "BUY_RISE", f"BB touch lower: {close:.2f}<={lower:.2f}"
        if close >= upper:
            return "BUY_FALL", f"BB touch upper: {close:.2f}>={upper:.2f}"
        return "HOLD", f"BB({close:.2f}) upper={upper:.2f} lower={lower:.2f}"

    def on_result(self, won: bool) -> None:
        pass


# ── Indicator helpers ──────────────────────────────────────────────────────────

_ATR_MEAN_WINDOW = 100


def _bb(closes: list[float], period: int, n_std: float):
    if len(closes) < period:
        return None
    w    = closes[-period:]
    mean = sum(w) / period
    std  = (sum((x - mean) ** 2 for x in w) / period) ** 0.5
    return mean + n_std * std, mean - n_std * std


def _close_atr(closes: list[float], period: int) -> tuple:
    needed = period + _ATR_MEAN_WINDOW + 1
    if len(closes) < needed:
        return None, None
    tail   = closes[-(period + _ATR_MEAN_WINDOW + 1):]
    deltas = [abs(tail[i] - tail[i - 1]) for i in range(1, len(tail))]
    atr    = sum(deltas[-period:]) / period
    atr_vals = [
        sum(deltas[i - period: i]) / period
        for i in range(period, len(deltas))
    ]
    atr_mean = sum(atr_vals) / len(atr_vals) if atr_vals else None
    return atr, atr_mean
