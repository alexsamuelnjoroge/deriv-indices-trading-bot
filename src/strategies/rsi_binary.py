"""
RSI Mean Reversion Strategy for Binary Option contracts (CALL/PUT).

Bar-based RSI reversal: fires when RSI exits an extreme zone.
No sl_pct/tp_pct in Signal — routes to buy_contract() not buy_multiplier().
Contract duration set via config (contract_duration / contract_duration_unit).

Validated on:
  1HZ10V @5-min RSI(14) OS=30: 3/3 folds STRONG, MeanEV +0.1822 (~2.1 trades/day)
  JD10   @5-min RSI(14) OS=25: 3/3 folds STRONG, MeanEV +0.0926 (~1.2 trades/day)
"""

from typing import Optional
from .base import BaseStrategy, Signal


class RSIBinaryStrategy(BaseStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.rsi_period   = config.get("rsi_period",   14)
        self.rsi_os       = config.get("rsi_os",       30.0)
        self.rsi_ob       = config.get("rsi_ob",       70.0)
        self.bar_seconds  = config.get("bar_seconds",   300)
        self.atr_period   = config.get("atr_period",    14)
        self.atr_min_mult = config.get("atr_min_mult",  0.0)

        self._bar_closes: list[float] = []
        self._bar_start:  Optional[int] = None
        self._prev_rsi:   Optional[float] = None

    def seed_candles(self, closes: list[float]) -> None:
        self._bar_closes = list(closes)
        if len(closes) >= self.rsi_period + 1:
            self._prev_rsi = _rsi(closes, self.rsi_period)

    def evaluate(self, tick_store) -> Signal:
        price = tick_store.latest_price
        epoch = tick_store.latest_epoch
        if price is None or epoch is None:
            return Signal(action="HOLD", reason="No ticks yet")
        if self._bar_start is None:
            self._bar_start = epoch
            return Signal(action="HOLD", reason="Bar started")

        elapsed = epoch - self._bar_start
        if elapsed < self.bar_seconds:
            return Signal(action="HOLD", reason=f"In bar ({elapsed}s/{self.bar_seconds}s)")

        self._bar_closes.append(price)
        self._bar_start = epoch

        needed = self.rsi_period + 2
        if len(self._bar_closes) < needed:
            return Signal(action="HOLD",
                          reason=f"Warming ({len(self._bar_closes)}/{needed})")

        return self._evaluate_bar()

    def _evaluate_bar(self) -> Signal:
        rsi_now  = _rsi(self._bar_closes, self.rsi_period)
        prev_rsi = self._prev_rsi
        self._prev_rsi = rsi_now

        if prev_rsi is None:
            return Signal(action="HOLD", reason=f"RSI {rsi_now:.1f} (warming)")

        if prev_rsi < self.rsi_os <= rsi_now:
            action, reason = "BUY_RISE", f"RSI exited oversold {prev_rsi:.1f}->{rsi_now:.1f} CALL"
        elif prev_rsi > self.rsi_ob >= rsi_now:
            action, reason = "BUY_FALL", f"RSI exited overbought {prev_rsi:.1f}->{rsi_now:.1f} PUT"
        else:
            return Signal(action="HOLD", reason=f"RSI {rsi_now:.1f}")

        if self.atr_min_mult > 0:
            atr, atr_mean = _close_atr(self._bar_closes, self.atr_period)
            if atr is None or atr_mean is None or atr < self.atr_min_mult * atr_mean:
                return Signal(
                    action="HOLD",
                    reason=f"ATR gate: {atr:.5f if atr else 'n/a'} < {self.atr_min_mult}x mean",
                    rsi=rsi_now,
                )

        return Signal(action=action, reason=reason, rsi=rsi_now)

    def on_result(self, won: bool) -> None:
        pass


_ATR_MEAN_WINDOW = 100


def _close_atr(closes: list[float], period: int) -> tuple:
    needed = period + _ATR_MEAN_WINDOW + 1
    if len(closes) < needed:
        return None, None
    tail   = closes[-(period + _ATR_MEAN_WINDOW + 1):]
    deltas = [abs(tail[i] - tail[i - 1]) for i in range(1, len(tail))]
    atr    = sum(deltas[-period:]) / period
    atr_vals = [
        sum(deltas[i - period: i]) / period
        for i in range(period, len(deltas))
    ]
    atr_mean = sum(atr_vals) / len(atr_vals) if atr_vals else None
    return atr, atr_mean


def _rsi(closes: list[float], period: int) -> float:
    window  = closes[-(period + 1):]
    changes = [window[i] - window[i - 1] for i in range(1, len(window))]
    avg_gain = sum(c for c in changes if c > 0) / period
    avg_loss = sum(abs(c) for c in changes if c < 0) / period
    if avg_loss == 0:
        return 100.0
    return round(100 - (100 / (1 + avg_gain / avg_loss)), 2)
