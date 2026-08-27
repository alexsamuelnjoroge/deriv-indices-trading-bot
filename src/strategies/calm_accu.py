"""
Calm-Period ACCU Strategy — inter-spike accumulator entries.

Opposite of spike-recoil: avoids the post-spike turbulence window entirely.
Instead enters ACCU contracts during the long, quiet trending phases between
spikes where barrier survival probability should be highest.

Entry conditions (all must pass):
  1. ticks since last spike > spike_cooldown
  2. short-term ATR < calm_atr_ratio x long-term ATR  (market is calm)
  3. entry_cooldown ticks elapsed since last entry signal

Config keys:
  symbol_type       "crash" or "boom"              (default: crash)
  long_atr_period   Baseline ATR window            (default: 50)
  short_atr_period  Recent ATR window              (default: 10)
  spike_mult        ATR multiple that marks a spike (default: 15.0)
  spike_cooldown    Ticks to avoid after spike     (default: 50)
  calm_atr_ratio    short_atr must be < ratio x long_atr  (default: 1.0)
  entry_cooldown    Min ticks between entry signals (default: 5)
  loss_cooldown     Losses before 15-tick pause    (default: 0)
"""

from .base import BaseStrategy, Signal


class CalmAccuStrategy(BaseStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.long_atr_period  = int(config.get("long_atr_period", 50))
        self.short_atr_period = int(config.get("short_atr_period", 10))
        self.spike_mult       = float(config.get("spike_mult", 15.0))
        self.spike_cooldown   = int(config.get("spike_cooldown", 50))
        self.calm_atr_ratio   = float(config.get("calm_atr_ratio", 1.0))
        self.entry_cooldown   = int(config.get("entry_cooldown", 5))
        self.loss_cooldown    = int(config.get("loss_cooldown", 0))

        self._ticks_since_spike = self.spike_cooldown  # start ready
        self._ticks_since_entry = self.entry_cooldown
        self._consecutive_losses = 0
        self._extra_cooldown = 0

    def _atr(self, prices: list[float], period: int) -> float | None:
        hist = prices[-(period + 1):]
        if len(hist) < period + 1:
            return None
        ranges = [abs(hist[i] - hist[i - 1]) for i in range(1, len(hist))]
        avg = sum(ranges) / len(ranges)
        return avg if avg > 0 else None

    def on_result(self, won: bool) -> None:
        if won:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            if self.loss_cooldown > 0 and self._consecutive_losses >= self.loss_cooldown:
                self._extra_cooldown = 15
                self._consecutive_losses = 0

    def evaluate(self, tick_store) -> Signal:
        prices = tick_store.prices
        n      = len(prices)

        min_len = self.long_atr_period + 2
        if n < min_len:
            return Signal(action="HOLD", reason=f"Warming up ({n}/{min_len})")

        long_atr  = self._atr(prices, self.long_atr_period)
        short_atr = self._atr(prices, self.short_atr_period)

        if long_atr is None or short_atr is None or long_atr <= 0:
            return Signal(action="HOLD", reason="ATR not ready")

        # Always detect spikes so we can maintain the cooldown counter
        last_move = abs(prices[-1] - prices[-2])
        if last_move > self.spike_mult * long_atr:
            self._ticks_since_spike = 0
        else:
            self._ticks_since_spike = min(self._ticks_since_spike + 1, 9999)

        self._ticks_since_entry = min(self._ticks_since_entry + 1, 9999)

        if self._extra_cooldown > 0:
            self._extra_cooldown -= 1
            return Signal(action="HOLD", reason=f"Loss cooldown ({self._extra_cooldown + 1} left)", atr=long_atr)

        if self._ticks_since_spike < self.spike_cooldown:
            return Signal(
                action="HOLD",
                reason=f"Spike cooldown ({self.spike_cooldown - self._ticks_since_spike} left)",
                atr=long_atr,
            )

        if self._ticks_since_entry < self.entry_cooldown:
            return Signal(
                action="HOLD",
                reason=f"Entry cooldown ({self.entry_cooldown - self._ticks_since_entry} left)",
                atr=long_atr,
            )

        if short_atr > self.calm_atr_ratio * long_atr:
            return Signal(
                action="HOLD",
                reason=f"Not calm: s_atr={short_atr:.5f} > {self.calm_atr_ratio}x l_atr={long_atr:.5f}",
                atr=long_atr,
            )

        self._ticks_since_entry = 0
        ratio = short_atr / long_atr
        return Signal(
            action="BUY_ACCU",
            reason=f"Calm entry: {self._ticks_since_spike}t since spike | s/l ATR={ratio:.0%}",
            atr=long_atr,
        )
