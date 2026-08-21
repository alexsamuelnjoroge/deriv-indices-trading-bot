"""
Digit-Even Strategy — exploits the structural even/odd digit bias in R_50 and R_75.

Deriv's synthetic price generator for Volatility 50 and 75 indices produces
last digits that are even ~55% of the time vs the expected 50%. DIGITEVEN
contracts pay 95% profit with a BE of 51.28%, leaving a structural edge of
~3.4–3.7% per trade that holds across all 4 walk-forward windows.

No indicator warm-up is required — the edge is in the price generation
algorithm, not in market timing. The strategy fires on every tick.

Config keys (all optional):
  loss_cooldown    consecutive losses before a short pause  (default: 0 = off)
  precision        decimal places for last-digit extraction  (default: 3)
"""

from .base import BaseStrategy, Signal


class DigitEvenStrategy(BaseStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self._loss_cooldown       = int(config.get("loss_cooldown", 0))
        self._consecutive_losses  = 0
        self._extra_cooldown      = 0

    def evaluate(self, tick_store) -> Signal:
        if self._extra_cooldown > 0:
            self._extra_cooldown -= 1
            return Signal(
                action="HOLD",
                reason=f"Loss cooldown ({self._extra_cooldown + 1} remaining)",
            )
        return Signal(action="BUY_EVEN", reason="digit_even: structural even-digit bias")

    def on_result(self, won: bool) -> None:
        if won:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            if self._loss_cooldown > 0 and self._consecutive_losses >= self._loss_cooldown:
                self._extra_cooldown     = self._loss_cooldown * 2
                self._consecutive_losses = 0
