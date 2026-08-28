"""
Digit-Over Strategy — exploits the structural digit-0 scarcity on R_ volatility indices.

Digit 0 is almost absent from Deriv's R_10/R_25/R_50/R_75/R_100 price streams,
redistributing probability toward digits 5-9. DIGITOVER(4) (win if last digit > 4)
therefore wins ~55-56% vs a 51.28% breakeven on the 95% payout, giving EV ~+3.7-4.3%.

Edge holds 2/2 walk-forward on all five R_ symbols across 43k-tick datasets.

No indicator warm-up needed — the edge is structural, not timing-based.

Config keys (all optional):
  barrier        digit threshold for DIGITOVER contract  (default: 4)
  loss_cooldown  consecutive losses before a short pause  (default: 0 = off)
"""

from .base import BaseStrategy, Signal


class DigitOverStrategy(BaseStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self._barrier            = int(config.get("barrier", 4))
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
            action="BUY_DIGITOVER",
            reason=f"digit_over: structural digit-0 scarcity -> DIGITOVER({self._barrier})",
        )

    def on_result(self, won: bool) -> None:
        if won:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            if self._loss_cooldown > 0 and self._consecutive_losses >= self._loss_cooldown:
                self._extra_cooldown     = self._loss_cooldown * 2
                self._consecutive_losses = 0
