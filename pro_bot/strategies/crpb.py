"""Compression Range Position Bias — ATR compression phase + average close position bias predicts expansion direction."""
from typing import Optional

from .research_base import ResearchDailyStrategy
from .base import Signal


class CRPBStrategy(ResearchDailyStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.atr_sma_period      = config.get("atr_sma_period", 20)
        self.min_quiet_bars      = config.get("min_quiet_bars", 5)
        self.upper_bias          = config.get("upper_bias", 0.55)
        self.lower_bias          = config.get("lower_bias", 0.40)
        self.bias_overrides_body = config.get("bias_overrides_body", True)

        self._in_compression = False
        self._quiet_count    = 0
        self._comp_high      = None
        self._comp_low       = None
        self._pos_sum        = 0.0

    def _atr_sma(self, bars: list) -> Optional[float]:
        P = self.atr_sma_period
        needed = P + 14
        if len(bars) < needed + 1:
            return None
        # Compute ATR for each bar in the SMA window
        atrs = []
        for i in range(len(bars) - P - 14, len(bars) - 14):
            h, l, cp = bars[i]["high"], bars[i]["low"], bars[i - 1]["close"]
            atrs.append(max(h - l, abs(h - cp), abs(l - cp)))
        if len(atrs) < P:
            return None
        return sum(atrs[-P:]) / P

    def _evaluate(self) -> Signal:
        """Override to always track compression state regardless of cooldown."""
        bars = list(self._h1)
        if len(bars) < self.atr_sma_period + 20:
            return Signal("HOLD", reason="warmup")

        bar = bars[-1]
        atr = self._atr14()
        if atr is None or atr <= 0:
            return Signal("HOLD", reason="atr_warmup")

        atr_sma = self._atr_sma(bars)
        if atr_sma is None:
            return Signal("HOLD", reason="sma_warmup")

        currently_compressed = atr < atr_sma

        if currently_compressed:
            if not self._in_compression:
                self._in_compression = True
                self._quiet_count    = 0
                self._comp_high      = bar["high"]
                self._comp_low       = bar["low"]
                self._pos_sum        = 0.0
            else:
                self._comp_high = max(self._comp_high, bar["high"])
                self._comp_low  = min(self._comp_low,  bar["low"])

            comp_range = self._comp_high - self._comp_low
            if comp_range > 0:
                self._pos_sum     += (bar["close"] - self._comp_low) / comp_range
                self._quiet_count += 1
            return Signal("HOLD", reason="in_compression")

        # Expansion bar
        was_in       = self._in_compression
        quiet_count  = self._quiet_count
        avg_pos      = self._pos_sum / quiet_count if quiet_count > 0 else 0.5
        self._in_compression = False
        self._quiet_count    = 0

        if not was_in or quiet_count < self.min_quiet_bars:
            return Signal("HOLD", reason="no_compression")

        if self._bar_idx - self._last_sig_i <= self.cooldown_bars:
            return Signal("HOLD", reason="cooldown")

        allow_long, allow_short = self._macro_gate()
        sl, tp = self._sl_tp(atr)
        bull_body = bar["close"] > bar["open"]

        if avg_pos > self.upper_bias and allow_long:
            if self.bias_overrides_body or bull_body:
                self._last_sig_i = self._bar_idx
                return Signal("BUY", sl_pips=sl, tp_pips=tp,
                              reason=f"CRPB buy avg_pos={avg_pos:.3f} q={quiet_count}")

        if avg_pos < self.lower_bias and allow_short:
            if self.bias_overrides_body or not bull_body:
                self._last_sig_i = self._bar_idx
                return Signal("SELL", sl_pips=sl, tp_pips=tp,
                              reason=f"CRPB sell avg_pos={avg_pos:.3f} q={quiet_count}")

        return Signal("HOLD", reason="no_bias")

    def _check_signal(self, atr, allow_long, allow_short) -> Optional[Signal]:
        return None  # logic handled in _evaluate
