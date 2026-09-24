"""
Digit-Over-Under Strategy — places OVER(4) and UNDER(6) on the same tick.

By firing both contracts within a single tick handler, both settle on the
identical next price tick. This guarantees the structural hedge:
  - OVER(4) loses on {1,2,3,4}, UNDER(6) loses on {6,7,8,9} — never the same digit.
  - Digit 5 (11.1% of ticks): BOTH WIN → net +$1.90 per pair.
  - All other digits: one wins, one loses → net -$0.05 per pair.

Emits BUY_DIGIT_PAIR; the trader places both contracts atomically.
"""

from .base import BaseStrategy, Signal


class DigitOverUnderStrategy(BaseStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
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
            action="BUY_DIGIT_PAIR",
            reason="digit_over_under: structural digit-0 scarcity, same-tick hedge",
        )

    def on_result(self, won: bool) -> None:
        if won:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            if self._loss_cooldown > 0 and self._consecutive_losses >= self._loss_cooldown:
                self._extra_cooldown     = self._loss_cooldown * 2
                self._consecutive_losses = 0
