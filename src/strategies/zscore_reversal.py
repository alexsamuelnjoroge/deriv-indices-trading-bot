"""
Z-Score Mean Reversion Strategy.

Mathematically cleaner version of RSI reversal: instead of normalizing
gains vs losses (RSI), directly measure how many standard deviations the
current price is from its rolling mean.

  Z = (price - MA_n) / std_n

Signal logic:
  Z >  +threshold  → BUY_FALL  (price far above mean, expect reversion)
  Z <  -threshold  → BUY_RISE  (price far below mean, expect reversion)
  |Z| ≤ threshold  → HOLD

Advantages over RSI:
  - Threshold is in units of standard deviations (direct probability meaning)
  - Automatically adapts to current volatility regime
  - No smoothing artifacts from Wilder's average
"""

from .base import BaseStrategy, Signal


class ZScoreReversalStrategy(BaseStrategy):

    def __init__(self, config: dict):
        super().__init__(config)

        self.zscore_period:    int   = config.get("zscore_period", 20)
        self.zscore_threshold: float = config.get("zscore_threshold", 2.5)
        self.confirm_ticks:    int   = config.get("confirm_ticks", 2)

        self.use_atr_filter:   bool  = config.get("use_atr_filter", False)
        self.atr_filter_ratio: float = config.get("atr_filter_ratio", 1.0)
        self.atr_period:       int   = config.get("atr_period", 14)
        self.atr_baseline_period: int = config.get("atr_baseline_period", 50)

        self.loss_cooldown: int = config.get("loss_cooldown", 0)

        self._consecutive_signal: str = "HOLD"
        self._consecutive_count:  int = 0
        self._last_fired:         str = "HOLD"
        self._consecutive_losses: int = 0
        self._cooldown_remaining: int = 0

    # ------------------------------------------------------------------ #
    #  Result callback — called after every settled contract
    # ------------------------------------------------------------------ #

    def on_result(self, won: bool) -> None:
        if won:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            if self.loss_cooldown > 0 and self._consecutive_losses >= self.loss_cooldown:
                self._cooldown_remaining = 1
                self._consecutive_losses = 0
                self._last_fired = "HOLD"

    # ------------------------------------------------------------------ #
    #  Main evaluate loop
    # ------------------------------------------------------------------ #

    def evaluate(self, tick_store) -> Signal:
        price = tick_store.latest_price
        z     = tick_store.zscore(self.zscore_period)

        if price is None or z is None:
            needed = max(self.zscore_period, self.atr_baseline_period + 1)
            return Signal(
                action="HOLD",
                reason=f"Warming up ({tick_store.tick_count}/{needed} ticks)",
            )

        atr          = tick_store.atr(self.atr_period)
        atr_baseline = tick_store.atr(self.atr_baseline_period)

        # ── ATR ranging-market gate ───────────────────────────────
        if (self.use_atr_filter
                and atr is not None
                and atr_baseline is not None
                and atr_baseline > 0):
            if atr > atr_baseline * self.atr_filter_ratio:
                self._reset_confirmation()
                return Signal(
                    action="HOLD",
                    reason=f"ATR gate: volatile ({atr:.6f} > {atr_baseline:.6f}×{self.atr_filter_ratio})",
                    atr=atr, atr_baseline=atr_baseline,
                )

        # ── Z-score zone detection ────────────────────────────────
        if z > self.zscore_threshold:
            candidate = "BUY_FALL"
        elif z < -self.zscore_threshold:
            candidate = "BUY_RISE"
        else:
            self._reset_confirmation()
            return Signal(
                action="HOLD",
                reason=f"Z={z:.2f} neutral (|Z|<{self.zscore_threshold})",
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

        # ── Suppression ───────────────────────────────────────────
        if candidate == self._last_fired:
            return Signal(
                action="HOLD",
                reason=f"Z={z:.2f} — already traded this zone, waiting for reset",
                atr=atr, atr_baseline=atr_baseline,
            )

        # ── Confirmation ticks ────────────────────────────────────
        if candidate == self._consecutive_signal:
            self._consecutive_count += 1
        else:
            self._consecutive_signal = candidate
            self._consecutive_count  = 1

        if self._consecutive_count < self.confirm_ticks:
            return Signal(
                action="HOLD",
                reason=f"Z={z:.2f} confirming ({self._consecutive_count}/{self.confirm_ticks})",
                atr=atr, atr_baseline=atr_baseline,
            )

        # ── Signal confirmed ──────────────────────────────────────
        self._last_fired = candidate
        direction = "below->rise" if candidate == "BUY_RISE" else "above->fall"
        return Signal(
            action=candidate,
            reason=f"Z{self.zscore_period}:{z:.2f} {direction} | confirmed {self._consecutive_count}/{self.confirm_ticks}",
            atr=atr,
            atr_baseline=atr_baseline,
        )

    def _reset_confirmation(self):
        self._consecutive_signal = "HOLD"
        self._consecutive_count  = 0
