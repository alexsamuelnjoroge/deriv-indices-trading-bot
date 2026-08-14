"""Volatility Regime Transition — first bar of vol expansion after N quiet bars reveals institutional direction."""
from typing import Optional

from .research_base import ResearchDailyStrategy
from .base import Signal


class VRTStrategy(ResearchDailyStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.atr_sma_period = config.get("atr_sma_period", 20)
        self.min_quiet_bars = config.get("min_quiet_bars", 5)
        self.min_body_ratio = config.get("min_body_ratio", 0.2)

        self._quiet_count = 0

    def _atr_sma(self, bars: list) -> Optional[float]:
        P = self.atr_sma_period
        if len(bars) < P + 14:
            return None
        atrs = []
        for i in range(len(bars) - P - 13, len(bars) - 13):
            h, l, cp = bars[i]["high"], bars[i]["low"], bars[i - 1]["close"]
            atrs.append(max(h - l, abs(h - cp), abs(l - cp)))
        if len(atrs) < P:
            return None
        return sum(atrs[-P:]) / P

    def _evaluate(self) -> Signal:
        """Override to always track quiet_count regardless of cooldown."""
        bars = list(self._h1)
        if len(bars) < self.atr_sma_period + 20:
            return Signal("HOLD", reason="warmup")

        atr_cur  = self._atr14()
        if atr_cur is None or atr_cur <= 0:
            self._quiet_count = 0
            return Signal("HOLD", reason="atr_warmup")

        # Compute ATR for previous bar too
        bars_prev = bars[:-1]
        if len(bars_prev) < 15:
            return Signal("HOLD", reason="warmup")
        trs_prev = []
        for i in range(max(1, len(bars_prev) - 14), len(bars_prev)):
            h, l, cp = bars_prev[i]["high"], bars_prev[i]["low"], bars_prev[i - 1]["close"]
            trs_prev.append(max(h - l, abs(h - cp), abs(l - cp)))
        atr_prev = sum(trs_prev) / len(trs_prev) if trs_prev else None

        atr_sma_cur  = self._atr_sma(bars)
        atr_sma_prev = self._atr_sma(bars_prev) if len(bars_prev) >= self.atr_sma_period + 14 else None

        if atr_prev is None or atr_sma_cur is None or atr_sma_prev is None:
            self._quiet_count = 0
            return Signal("HOLD", reason="sma_warmup")

        # Always update quiet_count
        if atr_prev <= atr_sma_prev:
            self._quiet_count += 1
        else:
            self._quiet_count = 0

        # Transition trigger: prev was quiet, current is NOT quiet
        if atr_cur <= atr_sma_cur:
            return Signal("HOLD", reason="still_quiet")
        if atr_prev > atr_sma_prev:
            return Signal("HOLD", reason="already_elevated")
        if self._quiet_count < self.min_quiet_bars:
            return Signal("HOLD", reason="quiet_too_short")

        saved_quiet = self._quiet_count
        self._quiet_count = 0

        if self._bar_idx - self._last_sig_i <= self.cooldown_bars:
            return Signal("HOLD", reason="cooldown")

        bar  = bars[-1]
        body = abs(bar["close"] - bar["open"])
        if body < self.min_body_ratio * atr_cur:
            return Signal("HOLD", reason="doji_transition")

        allow_long, allow_short = self._macro_gate()
        sl, tp = self._sl_tp(atr_cur)

        bar_bull = bar["close"] > bar["open"]

        if bar_bull and allow_long:
            self._last_sig_i = self._bar_idx
            return Signal("BUY", sl_pips=sl, tp_pips=tp,
                          reason=f"VRT bull q={saved_quiet}")

        if not bar_bull and allow_short:
            self._last_sig_i = self._bar_idx
            return Signal("SELL", sl_pips=sl, tp_pips=tp,
                          reason=f"VRT bear q={saved_quiet}")

        return Signal("HOLD", reason="macro_blocked")

    def _check_signal(self, atr, allow_long, allow_short) -> Optional[Signal]:
        return None  # logic handled in _evaluate
