"""
Digit Scalper — exploits positive autocorrelation in the JD100 digit stream.

Edge: after digit 7, the next digit is >4 (DIGITOVER) with ~86% probability.
      DIGITUNDER disabled — actual payout=43% sets BE=69.9%, too high to sustain.

Actual Deriv payouts on JD100 (measured from live statement, not estimates):
  DIGITOVER(4):  25% payout → BE=80.0%  (digit 7 trigger: ~86% WR ✓)
  DIGITUNDER(4): 43% payout → BE=69.9%  (digit 2 trigger: only 61.8% WR ✗)

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
        if len(ticks) < 1:
            return Signal(action="HOLD", reason="digit_scalper: warming up")

        # Warn once if API-reported pip_size differs from config
        api_pip = getattr(ticks[-1], "pip_size", None)
        if api_pip is not None and api_pip != self._pip_size and not getattr(self, "_pip_warned", False):
            self._pip_warned = True
            import logging
            logging.getLogger(__name__).warning(
                f"digit_scalper: config pip_size={self._pip_size} but API reports {api_pip} — "
                "check config or digits will be wrong"
            )

        # Trigger = digit of the CURRENT tick; contract settles on the NEXT tick (lag-1).
        # Using ticks[-1] (not ticks[-2]) ensures the edge matches the measured r=+0.39
        # at lag-1. Using ticks[-2] would trade lag-2 which is near zero.
        curr_digit = self._digit(ticks[-1].price)

        if curr_digit in self._over_digits:
            return Signal(
                action="BUY_DIGITOVER",
                reason=f"digit_scalper: curr={curr_digit} ∈ over_digits → DIGITOVER({self._barrier})",
            )
        if curr_digit in self._under_digits:
            return Signal(
                action="BUY_DIGITUNDER",
                reason=f"digit_scalper: curr={curr_digit} ∈ under_digits → DIGITUNDER({self._barrier})",
            )
        return Signal(action="HOLD", reason=f"digit_scalper: curr={curr_digit} → skip")

    def on_result(self, won: bool) -> None:
        self._cooldown = self._cooldown_max
