"""
Crash/Boom Post-Spike Recoil Strategy for Deriv synthetic indices.

Crash indices: algorithmically generated downward price spikes at a rate of
  approximately 1 per N ticks (N = 500, 1000).
Boom indices:  algorithmically generated upward price spikes at the same rates.

Edge: immediately after a spike, price reverts toward the pre-spike level for
the next several ticks. This strategy detects the spike and enters a short
binary contract in the opposite (recoil) direction.

ATR design: the ATR is computed from prices[:-1] (EXCLUDING the current tick).
This prevents the spike itself from inflating the baseline and masking the
detection. With this approach, a real Crash spike appears as 50-500x ATR
rather than 5-15x (after Wilder inflation), so threshold tuning is intuitive.

Detection rule:
  abs_move = |prices[-1] - prices[-2]|
  pre_spike_atr = ATR computed on prices[-atr_period-2 : -1]
  spike detected if abs_move > spike_mult * pre_spike_atr AND direction matches
    Crash + downward move (prices[-1] < prices[-2]) -> BUY_RISE
    Boom  + upward   move (prices[-1] > prices[-2]) -> BUY_FALL

Config keys (all optional):
  symbol_type      "crash" or "boom"       (default: "crash")
  spike_mult       ATR multiple threshold  (default: 15.0)
  atr_period       Lookback for baseline   (default: 50)
  cooldown_ticks   evaluate() skips after spike fires  (default: 5)
  loss_cooldown    Consecutive losses before extra skip (default: 0)
"""

from .base import BaseStrategy, Signal


class CrashBoomRecoilStrategy(BaseStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.symbol_type    = str(config.get("symbol_type", "crash")).lower()
        self.spike_mult     = float(config.get("spike_mult", 15.0))
        self.atr_period     = int(config.get("atr_period", 50))
        self.cooldown_ticks = int(config.get("cooldown_ticks", 5))
        self.loss_cooldown  = int(config.get("loss_cooldown", 0))

        self._cooldown           = 0
        self._consecutive_losses = 0
        self._extra_cooldown     = 0

    # ------------------------------------------------------------------ #
    #  Pre-spike ATR (computed from prices excluding the current tick)
    # ------------------------------------------------------------------ #

    def _pre_spike_atr(self, prices: list[float]) -> float | None:
        """
        Wilder ATR on prices[:-1] so the current (spike) tick doesn't inflate
        the baseline. Uses only the most recent atr_period+1 prices for speed.
        """
        hist = prices[-(self.atr_period + 2): -1]
        if len(hist) < self.atr_period + 1:
            return None
        ranges = [abs(hist[i] - hist[i - 1]) for i in range(1, len(hist))]
        avg = sum(ranges[: self.atr_period]) / self.atr_period
        for r in ranges[self.atr_period :]:
            avg = (avg * (self.atr_period - 1) + r) / self.atr_period
        return avg if avg > 0 else None

    # ------------------------------------------------------------------ #
    #  Trade result callback
    # ------------------------------------------------------------------ #

    def on_result(self, won: bool) -> None:
        if won:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            if self.loss_cooldown > 0 and self._consecutive_losses >= self.loss_cooldown:
                self._extra_cooldown     = 15
                self._consecutive_losses = 0

    # ------------------------------------------------------------------ #
    #  Main evaluation
    # ------------------------------------------------------------------ #

    def evaluate(self, tick_store) -> Signal:
        prices = tick_store.prices
        n      = len(prices)

        # Need atr_period+2 prices so pre_spike_atr can look at atr_period+1 of them
        if n < self.atr_period + 2:
            return Signal(
                action="HOLD",
                reason=f"Warming up ({n}/{self.atr_period + 2})",
            )

        pre_atr = self._pre_spike_atr(prices)
        if pre_atr is None or pre_atr <= 0:
            return Signal(action="HOLD", reason="Pre-spike ATR not ready")

        if self._extra_cooldown > 0:
            self._extra_cooldown -= 1
            return Signal(
                action="HOLD",
                reason=f"Loss cooldown ({self._extra_cooldown + 1} remaining)",
                atr=pre_atr,
            )

        if self._cooldown > 0:
            self._cooldown -= 1
            return Signal(
                action="HOLD",
                reason=f"Post-spike cooldown ({self._cooldown + 1} remaining)",
                atr=pre_atr,
            )

        last_move = prices[-1] - prices[-2]
        abs_move  = abs(last_move)
        mult      = abs_move / pre_atr
        threshold = self.spike_mult * pre_atr

        if self.symbol_type == "crash" and last_move < 0 and abs_move >= threshold:
            self._cooldown = self.cooldown_ticks
            return Signal(
                action="BUY_ACCU",
                reason=f"CRASH spike: -{abs_move:.4f} ({mult:.0f}xATR) -> ACCU recoil",
                atr=pre_atr,
            )

        if self.symbol_type == "boom" and last_move > 0 and abs_move >= threshold:
            self._cooldown = self.cooldown_ticks
            return Signal(
                action="BUY_ACCU",
                reason=f"BOOM spike: +{abs_move:.4f} ({mult:.0f}xATR) -> ACCU recoil",
                atr=pre_atr,
            )

        return Signal(
            action="HOLD",
            reason=f"No spike (move={last_move:+.4f}, {mult:.1f}xATR, need {self.spike_mult:.0f}x)",
            atr=pre_atr,
        )
