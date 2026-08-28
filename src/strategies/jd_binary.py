"""
JD Binary Strategy — directional post-spike binary on Jump Diffusion indices.

JD indices have bidirectional algorithmic spikes (both up and down). After
a spike, the next few ticks tend to retrace/recoil:
  - Down spike -> BUY_RISE (CALL): price rebounds upward
  - Up spike   -> BUY_FALL (PUT):  price retraces downward

Validated on JD75: settle=1t, dur=1t -> WR=59.6% EV=+6.9% (2/2 walk-forward)
                   settle=1t, dur=5t -> WR=61.5% EV=+8.7% (2/2 walk-forward)
Payout: 92% (BE=52.1%) for JD50/JD75. JD100: 96% (BE=51.0%).

Config keys:
  spike_mult    ATR multiple that marks a spike     (default: 10.0)
  atr_period    ATR window size                     (default: 30)
  settle_ticks  Ticks to wait after spike before entry (default: 1)
  spike_cooldown  Ticks to block after an entry     (default: 5)
  loss_cooldown   Consecutive losses before pause   (default: 2)
"""

from .base import BaseStrategy, Signal


class JDBinaryStrategy(BaseStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.spike_mult    = float(config.get("spike_mult", 10.0))
        self.atr_period    = int(config.get("atr_period", 30))
        self.settle_ticks  = int(config.get("settle_ticks", 1))
        self.spike_cooldown = int(config.get("spike_cooldown", 5))
        self.loss_cooldown = int(config.get("loss_cooldown", 2))

        self._pending_dir    = None   # +1 = up spike (BUY_FALL), -1 = down spike (BUY_RISE)
        self._settle_left    = 0
        self._cooldown       = 0
        self._consec_losses  = 0
        self._extra_cooldown = 0

    def _atr(self, prices: list[float]) -> float | None:
        hist = prices[-(self.atr_period + 1):]
        if len(hist) < self.atr_period + 1:
            return None
        ranges = [abs(hist[i] - hist[i - 1]) for i in range(1, len(hist))]
        avg = sum(ranges) / len(ranges)
        return avg if avg > 0 else None

    def on_result(self, won: bool) -> None:
        if won:
            self._consec_losses = 0
        else:
            self._consec_losses += 1
            if self.loss_cooldown > 0 and self._consec_losses >= self.loss_cooldown:
                self._extra_cooldown = self.loss_cooldown * 3
                self._consec_losses  = 0

    def evaluate(self, tick_store) -> Signal:
        prices = tick_store.prices
        if len(prices) < self.atr_period + 2:
            return Signal(action="HOLD", reason=f"Warming up ({len(prices)}/{self.atr_period + 2})")

        atr = self._atr(prices)
        if atr is None or atr <= 0:
            return Signal(action="HOLD", reason="ATR not ready")

        # Tick-by-tick state updates
        if self._cooldown > 0:
            self._cooldown -= 1

        if self._extra_cooldown > 0:
            self._extra_cooldown -= 1
            return Signal(action="HOLD", reason=f"Loss cooldown ({self._extra_cooldown + 1} left)")

        move = prices[-1] - prices[-2]

        # Spike detection — bidirectional
        if abs(move) > self.spike_mult * atr:
            self._pending_dir = 1 if move > 0 else -1
            self._settle_left = self.settle_ticks
            self._cooldown    = self.spike_cooldown
            direction = "UP" if self._pending_dir == 1 else "DOWN"
            return Signal(action="HOLD", reason=f"Spike {direction} detected — settling {self.settle_ticks}t")

        # Count down settle period after spike
        if self._pending_dir is not None and self._settle_left > 0:
            self._settle_left -= 1
            return Signal(action="HOLD", reason=f"Settling ({self._settle_left} left)")

        # Fire entry signal
        if self._pending_dir is not None and self._settle_left == 0 and self._cooldown == 0:
            direction = self._pending_dir
            self._pending_dir = None
            if direction == -1:
                return Signal(action="BUY_RISE", reason="JD down-spike recoil: CALL", atr=atr)
            else:
                return Signal(action="BUY_FALL", reason="JD up-spike retrace: PUT", atr=atr)

        return Signal(action="HOLD", reason="No spike signal")
