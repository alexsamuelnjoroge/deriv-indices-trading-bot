"""
Consecutive Tick Streak Reversal Strategy.

Trades against sustained directional runs in raw tick data.

  After N consecutive up-ticks  → BUY_FALL  (expect reversion)
  After N consecutive down-ticks → BUY_RISE  (expect reversion)

This directly exploits negative autocorrelation in tick returns without
any smoothing lag. The streak itself is the signal — no additional
confirmation layer is needed.
"""

from .base import BaseStrategy, Signal


class StreakReversalStrategy(BaseStrategy):

    def __init__(self, config: dict):
        super().__init__(config)

        self.streak_length:    int   = config.get("streak_length", 6)
        self.use_atr_filter:   bool  = config.get("use_atr_filter", False)
        self.atr_filter_ratio: float = config.get("atr_filter_ratio", 1.0)
        self.atr_period:       int   = config.get("atr_period", 14)
        self.atr_baseline_period: int = config.get("atr_baseline_period", 50)
        self.loss_cooldown:    int   = config.get("loss_cooldown", 0)

        self._last_fired:         str = "HOLD"
        self._consecutive_losses: int = 0
        self._cooldown_remaining: int = 0

    def on_result(self, won: bool) -> None:
        if won:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            if self.loss_cooldown > 0 and self._consecutive_losses >= self.loss_cooldown:
                self._cooldown_remaining = 1
                self._consecutive_losses = 0
                self._last_fired = "HOLD"

    def evaluate(self, tick_store) -> Signal:
        price = tick_store.latest_price

        if price is None or tick_store.tick_count < self.streak_length + 1:
            return Signal(
                action="HOLD",
                reason=f"Warming up ({tick_store.tick_count}/{self.streak_length + 1} ticks)",
            )

        atr          = tick_store.atr(self.atr_period)
        atr_baseline = tick_store.atr(self.atr_baseline_period)

        # ── ATR gate ──────────────────────────────────────────────
        if (self.use_atr_filter
                and atr is not None
                and atr_baseline is not None
                and atr_baseline > 0):
            if atr > atr_baseline * self.atr_filter_ratio:
                self._last_fired = "HOLD"
                return Signal(
                    action="HOLD",
                    reason=f"ATR gate: volatile ({atr:.6f} > {atr_baseline:.6f}×{self.atr_filter_ratio})",
                    atr=atr, atr_baseline=atr_baseline,
                )

        streak = tick_store.tick_streak()

        # ── Streak detection ──────────────────────────────────────
        if streak >= self.streak_length:
            candidate = "BUY_FALL"
        elif streak <= -self.streak_length:
            candidate = "BUY_RISE"
        else:
            # Streak reset — allow re-entry in whichever direction fires next
            if abs(streak) < 2:
                self._last_fired = "HOLD"
            return Signal(
                action="HOLD",
                reason=f"Streak {streak:+d} (need ±{self.streak_length})",
                atr=atr, atr_baseline=atr_baseline,
            )

        # ── Loss cooldown ─────────────────────────────────────────
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return Signal(
                action="HOLD",
                reason=f"Loss cooldown ({self._cooldown_remaining + 1} remaining)",
                atr=atr, atr_baseline=atr_baseline,
            )

        # ── Suppression — already traded this streak ──────────────
        if candidate == self._last_fired:
            return Signal(
                action="HOLD",
                reason=f"Streak {streak:+d} — already traded this run",
                atr=atr, atr_baseline=atr_baseline,
            )

        self._last_fired = candidate
        direction = "up-run->fall" if candidate == "BUY_FALL" else "down-run->rise"
        return Signal(
            action=candidate,
            reason=f"Streak {streak:+d} ticks {direction}",
            atr=atr, atr_baseline=atr_baseline,
        )
