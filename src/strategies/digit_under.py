"""
Digit-Under Strategy — exploits the structural digit-0 scarcity on R_ volatility indices.

Digit 0 is almost absent from Deriv's R_50/R_75 price streams. With only digits 1-9
present, DIGITUNDER(6) (win if last digit < 6) wins on digits {1,2,3,4,5} = 5/9 = 55.56%,
identical to DIGITOVER(4) but on the complementary set of digits.

OVER(4) and UNDER(6) never both lose on the same tick:
  OVER(4) loses on {1,2,3,4}, UNDER(6) loses on {6,7,8,9} — no overlap.
  On digit 5, both win simultaneously.

Config keys (all optional):
  barrier        digit threshold for DIGITUNDER contract  (default: 6)
  loss_cooldown  consecutive losses before a short pause  (default: 0 = off)
"""

from .base import BaseStrategy, Signal


class DigitUnderStrategy(BaseStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self._barrier            = int(config.get("barrier", 6))
        self._loss_cooldown      = int(config.get("loss_cooldown", 0))
        self._consecutive_losses = 0
        self._extra_cooldown     = 0

    def evaluate(self, tick_store) -> Signal:
        if self._extra_cooldown > 0:
            self._extra_cooldown -= 1
            return Signal(
                action="HOLD",
                reason=f"Loss cooldown ({self._extra_cooldown + 1} remaining)",
            )
        return Signal(
            action="BUY_DIGITUNDER",
            reason=f"digit_under: structural digit-0 scarcity -> DIGITUNDER({self._barrier})",
        )

    def on_result(self, won: bool) -> None:
        if won:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            if self._loss_cooldown > 0 and self._consecutive_losses >= self._loss_cooldown:
                self._extra_cooldown     = self._loss_cooldown * 2
                self._consecutive_losses = 0
