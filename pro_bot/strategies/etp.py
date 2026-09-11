"""EMA Trend Pullback — RSI oversold recovery in the direction of the H1 EMA trend.

Long:  EMA rising, close > EMA, RSI(7) crosses back above rsi_low  → BUY
Short: EMA falling, close < EMA, RSI(7) crosses back below rsi_high → SELL

Validated ROBUST 4/4 on XAUUSD H1 (EMA=15, RSI=7, r_l=38, sl=1.5×ATR, tp=3R).
"""

from collections import deque
from typing import Optional

from .research_base import ResearchDailyStrategy
from .base import Signal


class ETPStrategy(ResearchDailyStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.ema_period = config.get("ema_period", 15)
        self.rsi_period = config.get("rsi_period", 7)
        self.rsi_low    = config.get("rsi_low",    38)
        self.rsi_high   = config.get("rsi_high",   100 - config.get("rsi_low", 38))

        # Internal state for incremental EMA/RSI
        self._ema:      Optional[float] = None
        self._ema_prev: Optional[float] = None
        self._rsi:      Optional[float] = None
        self._rsi_prev: Optional[float] = None
        self._avg_gain: Optional[float] = None
        self._avg_loss: Optional[float] = None
        self._rsi_buf:  deque = deque(maxlen=self.rsi_period + 2)

    # ── Seeding ───────────────────────────────────────────────────────────────

    def _seed_h1(self, bars: list) -> None:
        """Override to seed EMA/RSI state from historical bars."""
        super()._seed_h1(bars)
        closes = [b["close"] for b in bars]
        self._seed_indicators(closes)

    def _seed_indicators(self, closes: list) -> None:
        p = self.ema_period
        rp = self.rsi_period

        # Seed EMA
        if len(closes) >= p:
            k = 2 / (p + 1)
            self._ema_prev = None
            self._ema = sum(closes[:p]) / p
            for c in closes[p:]:
                self._ema_prev = self._ema
                self._ema = c * k + self._ema * (1 - k)

        # Seed RSI
        if len(closes) >= rp + 1:
            diffs = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
            self._avg_gain = sum(max(d, 0) for d in diffs[:rp]) / rp
            self._avg_loss = sum(max(-d, 0) for d in diffs[:rp]) / rp
            self._rsi_prev = None
            for diff in diffs[rp:]:
                g = max(diff, 0)
                l = max(-diff, 0)
                self._avg_gain = (self._avg_gain * (rp - 1) + g) / rp
                self._avg_loss = (self._avg_loss * (rp - 1) + l) / rp
                self._rsi_prev = self._rsi
                if self._avg_loss == 0:
                    self._rsi = 100.0
                else:
                    self._rsi = 100 - 100 / (1 + self._avg_gain / self._avg_loss)

    # ── Per-bar update ────────────────────────────────────────────────────────

    def _update_indicators(self, close: float, prev_close: float) -> None:
        p  = self.ema_period
        rp = self.rsi_period
        k  = 2 / (p + 1)

        # Update EMA
        if self._ema is None:
            return
        self._ema_prev = self._ema
        self._ema = close * k + self._ema * (1 - k)

        # Update RSI
        if self._avg_gain is None:
            return
        diff = close - prev_close
        g = max(diff, 0)
        l = max(-diff, 0)
        self._avg_gain = (self._avg_gain * (rp - 1) + g) / rp
        self._avg_loss = (self._avg_loss * (rp - 1) + l) / rp
        self._rsi_prev = self._rsi
        if self._avg_loss == 0:
            self._rsi = 100.0
        else:
            self._rsi = 100 - 100 / (1 + self._avg_gain / self._avg_loss)

    # ── Signal ────────────────────────────────────────────────────────────────

    def _check_signal(self, atr: float,
                      allow_long: bool, allow_short: bool) -> Optional[Signal]:
        bars = list(self._h1)
        if len(bars) < 2:
            return None

        bar      = bars[-1]
        prev_bar = bars[-2]

        self._update_indicators(bar["close"], prev_bar["close"])

        ema      = self._ema
        ema_prev = self._ema_prev
        rsi      = self._rsi
        rsi_prev = self._rsi_prev

        if any(v is None for v in [ema, ema_prev, rsi, rsi_prev]):
            return None

        close        = bar["close"]
        ema_rising   = ema > ema_prev
        ema_falling  = ema < ema_prev
        sl, tp       = self._sl_tp(atr)

        # Long: uptrend, RSI oversold recovery
        if (ema_rising and close > ema
                and rsi_prev < self.rsi_low <= rsi
                and allow_long):
            return Signal("BUY", sl_pips=sl, tp_pips=tp,
                          reason=f"ETP buy RSI {rsi_prev:.1f}→{rsi:.1f} cross {self.rsi_low}")

        # Short: downtrend, RSI overbought fade
        if (ema_falling and close < ema
                and rsi_prev > self.rsi_high >= rsi
                and allow_short):
            return Signal("SELL", sl_pips=sl, tp_pips=tp,
                          reason=f"ETP sell RSI {rsi_prev:.1f}→{rsi:.1f} cross {self.rsi_high}")

        return None
