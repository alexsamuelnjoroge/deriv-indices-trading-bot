"""
Digit Scalper — exploits positive autocorrelation in the JD100 digit stream.

Edge: after digits {6, 7, 8} the next digit is >4 with ~74-86% probability.
      after digits {1, 2, 3} the next digit is ≤4  with ~70-84% probability.
Payout ~79%, BE=55.9%.

Requires use_pre_proposal: true in config to cut API latency to ~1s so the
contract settles on lag-1 (r=+0.39) rather than lag-3/4 (r≈+0.05).

Config keys:
  barrier        DIGITOVER / DIGITUNDER threshold    (default: 4)
  over_digits    digits that trigger DIGITOVER        (default: [6, 7, 8])
  under_digits   digits that trigger DIGITUNDER       (default: [1, 2, 3])
  pip_size       decimal places in price              (default: 2)
  cooldown_ticks ticks to skip after each trade       (default: 2)
"""
from .base import BaseStrategy, Signal


class DigitScalerStrategy(BaseStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self._barrier      = int(config.get("barrier", 4))
        self._over_digits  = set(config.get("over_digits", [6, 7, 8]))
        self._under_digits = set(config.get("under_digits", [1, 2, 3]))
        self._pip_size     = int(config.get("pip_size", 2))
        self._cooldown_max = int(config.get("cooldown_ticks", 2))
        self._cooldown     = 0

    def _digit(self, price: float) -> int:
        return int(f"{price:.{self._pip_size}f}"[-1])

    def evaluate(self, tick_store) -> Signal:
        if self._cooldown > 0:
            self._cooldown -= 1
            return Signal(action="HOLD", reason=f"digit_scalper: cooldown ({self._cooldown + 1} left)")

        ticks = tick_store._ticks
        if len(ticks) < 2:
            return Signal(action="HOLD", reason="digit_scalper: warming up")

        # Trigger = digit of the previous tick; contract settles on the NEXT tick (lag-1)
        prev_digit = self._digit(ticks[-2].price)

        if prev_digit in self._over_digits:
            return Signal(
                action="BUY_DIGITOVER",
                reason=f"digit_scalper: prev={prev_digit} ∈ over_digits → DIGITOVER({self._barrier})",
            )
        if prev_digit in self._under_digits:
            return Signal(
                action="BUY_DIGITUNDER",
                reason=f"digit_scalper: prev={prev_digit} ∈ under_digits → DIGITUNDER({self._barrier})",
            )
        return Signal(action="HOLD", reason=f"digit_scalper: prev={prev_digit} → skip")

    def on_result(self, won: bool) -> None:
        self._cooldown = self._cooldown_max
