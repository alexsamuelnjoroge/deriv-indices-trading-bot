"""
Gold Multiplier Strategy — EMA(200) trend filter + RSI(7) pullback entry.

Validated on frxXAUUSD 1h bars (7 months, walk-forward 3 folds):
  EMA(200) slope_bars=5, RSI(7) < 45 → MULTUP
  EMA(200) slope_bars=5, RSI(7) > 55 → MULTDOWN
  SL=0.30%, TP=0.75% on x100 multiplier | BE WR = 30.5%
  Walk-forward: 34.5% / 36.7% / 32.3% (all above breakeven), p=0.028

Signal fires on bar close (time-based, default 1h).
Call seed_candles(closes) at startup with historical hourly closes.
"""

from typing import Optional
from .base import BaseStrategy, Signal


class GoldTrendStrategy(BaseStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.ema_period   = config.get("ema_period",   200)
        self.slope_bars   = config.get("slope_bars",     5)
        self.rsi_period   = config.get("rsi_period",     7)
        self.rsi_entry    = config.get("rsi_entry",    45.0)
        self.sl_pct       = config.get("sl_pct",      0.003)
        self.tp_pct       = config.get("tp_pct",     0.0075)
        self.bar_seconds  = config.get("bar_seconds",  3600)
        # ATR gate: only fire when close-ATR >= atr_min_mult * 100-bar mean ATR.
        # 0.0 disables the filter (default = backward compatible).
        self.atr_period   = config.get("atr_period",    14)
        self.atr_min_mult = config.get("atr_min_mult",  0.0)

        self._bar_closes: list[float] = []
        self._bar_start: Optional[int] = None
        self._last_bar_signal: Optional[str] = None  # suppress repeat fires on same bar

    # ------------------------------------------------------------------ #
    #  Historical seed
    # ------------------------------------------------------------------ #

    def seed_candles(self, closes: list[float]) -> None:
        """Pre-load historical hourly closes (oldest first) before live data."""
        self._bar_closes = list(closes)

    # ------------------------------------------------------------------ #
    #  Evaluation (called on every tick)
    # ------------------------------------------------------------------ #

    def evaluate(self, tick_store) -> Signal:
        price = tick_store.latest_price
        epoch = tick_store.latest_epoch

        if price is None or epoch is None:
            return Signal(action="HOLD", reason="No ticks yet")

        # Initialise bar tracker on first tick
        if self._bar_start is None:
            self._bar_start = epoch
            return Signal(action="HOLD", reason="Bar started")

        elapsed = epoch - self._bar_start

        if elapsed < self.bar_seconds:
            return Signal(
                action="HOLD",
                reason=f"In bar ({elapsed}s/{self.bar_seconds}s)",
            )

        # ── Bar close ────────────────────────────────────────────────
        self._bar_closes.append(price)
        self._bar_start = epoch

        needed = self.ema_period + self.slope_bars + 1
        if len(self._bar_closes) < needed:
            return Signal(
                action="HOLD",
                reason=f"Warming candles ({len(self._bar_closes)}/{needed})",
            )

        return self._evaluate_bar()

    # ------------------------------------------------------------------ #
    #  Signal logic (runs once per bar close)
    # ------------------------------------------------------------------ #

    def _evaluate_bar(self) -> Signal:
        closes   = self._bar_closes
        ema_now  = _ema(closes,                    self.ema_period)
        ema_prev = _ema(closes[:-self.slope_bars], self.ema_period)
        rsi      = _rsi(closes, self.rsi_period)

        slope_up   = ema_now > ema_prev
        slope_down = ema_now < ema_prev

        if slope_up and rsi < self.rsi_entry:
            action = "BUY_RISE"
            reason = f"EMA slope↑ RSI {rsi:.1f}<{self.rsi_entry} MULTUP"
        elif slope_down and rsi > (100.0 - self.rsi_entry):
            action = "BUY_FALL"
            reason = f"EMA slope↓ RSI {rsi:.1f}>{100.0 - self.rsi_entry:.0f} MULTDOWN"
        else:
            return Signal(
                action="HOLD",
                reason=f"EMA slope={'↑' if slope_up else '↓' if slope_down else '='} RSI {rsi:.1f}",
                rsi=rsi,
            )

        if self.atr_min_mult > 0:
            atr, atr_mean = _close_atr(closes, self.atr_period)
            if atr is None or atr_mean is None or atr < self.atr_min_mult * atr_mean:
                return Signal(
                    action="HOLD",
                    reason=f"ATR gate: {atr:.5f if atr else 'n/a'} < {self.atr_min_mult}x mean",
                    rsi=rsi,
                )

        return Signal(
            action=action,
            reason=reason,
            rsi=rsi,
            sl_pct=self.sl_pct,
            tp_pct=self.tp_pct,
        )

    def on_result(self, won: bool) -> None:
        pass   # no cooldown logic needed — signal rate is ~1-2 per day


# ── Indicator helpers ────────────────────────────────────────────────────

_ATR_MEAN_WINDOW = 100   # bars for reference ATR level


def _close_atr(closes: list[float], period: int) -> tuple:
    """
    Close-to-close ATR proxy.  Returns (current_atr, mean_atr) using the last
    period + ATR_MEAN_WINDOW + 1 closes, or (None, None) if not enough data.
    """
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


def _ema(closes: list[float], period: int) -> float:
    k = 2.0 / (period + 1)
    val = sum(closes[:period]) / period
    for c in closes[period:]:
        val = c * k + val * (1 - k)
    return val


def _rsi(closes: list[float], period: int) -> float:
    window  = closes[-(period + 1):]
    changes = [window[i] - window[i - 1] for i in range(1, len(window))]
    avg_gain = sum(c for c in changes if c > 0) / period
    avg_loss = sum(abs(c) for c in changes if c < 0) / period
    if avg_loss == 0:
        return 100.0
    return round(100 - (100 / (1 + avg_gain / avg_loss)), 2)
