"""M15 RSI Recovery — ETP pattern on M15 bars, London session only.

Same logic as ETPStrategy: EMA trend agreement + RSI crossover at threshold.
Session-filtered to London (07:00–12:00 UTC) to avoid thin-session noise.

Long:  EMA(20) rising, close > EMA, RSI(14) crosses above rsi_low  → BUY
Short: EMA(20) falling, close < EMA, RSI(14) crosses below rsi_high → SELL

Validated MOSTLY OK 3/4 walk-forward windows on XAUUSD M15.
Best config: ema_period=20, rsi_period=14, rsi_low=42, atr_mult_sl=2.0,
             tp_rr=3.0, macro_filter=True, session=london
"""

from typing import Optional

from .etp import ETPStrategy
from .base import Signal


class MRRStrategy(ETPStrategy):

    def __init__(self, config: dict):
        # Apply MRR-specific defaults before passing to ETPStrategy
        merged = {
            "ema_period": 20,
            "rsi_period": 14,
            "rsi_low":    42,
        }
        merged.update(config)
        super().__init__(merged)
        self.session = merged.get("session", "london")

    # ── Signal logic ──────────────────────────────────────────────────────────

    def _check_signal(self, atr: float,
                      allow_long: bool, allow_short: bool) -> Optional[Signal]:
        bars = list(self._h1)
        if not bars:
            return None

        epoch = bars[-1].get("epoch", 0)
        h     = (epoch % 86400) // 3600

        if self.session == "london":
            if not (7 <= h < 12):
                return None
        elif self.session == "london_ny":
            if not ((7 <= h < 12) or (13 <= h < 18)):
                return None
        # "all" → no filter

        sig = super()._check_signal(atr, allow_long, allow_short)
        if sig is None:
            return None

        # Re-tag reason with MRR prefix
        if sig.action != "HOLD":
            sig.reason = sig.reason.replace("ETP ", "MRR ", 1)
        return sig
