"""
RSI Reversal Strategy — enhanced with EMA trend filter and Bollinger Band confirmation.

Three-layer signal logic:
  1. RSI extreme      — RSI > overbought or RSI < oversold (momentum exhaustion)
  2. EMA trend filter — only trade reversals that align with the dominant trend:
                         oversold bounce only when price > EMA  (uptrend pullback)
                         overbought drop only when price < EMA  (downtrend bounce)
  3. BB confirmation  — require price to be at or beyond the corresponding band
                         (price at upper BB confirms overbought; lower BB confirms oversold)
  4. Confirmation     — RSI must stay in the zone for `confirm_ticks` consecutive ticks

Once a signal fires, the same direction is suppressed until RSI returns to neutral,
preventing repeated entries in a single sustained extreme.
"""

from .base import BaseStrategy, Signal


class RSIReversalStrategy(BaseStrategy):
    def __init__(self, config: dict):
        super().__init__(config)
        self.overbought: float = config.get("rsi_overbought", 70)
        self.oversold: float = config.get("rsi_oversold", 30)
        self.confirm_ticks: int = config.get("confirm_ticks", 3)

        # EMA trend filter — set to 0 to disable
        self.ema_period: int = config.get("ema_trend_period", 50)

        # Bollinger Band confirmation
        self.use_bb_filter: bool = config.get("use_bb_filter", True)
        self.bb_period: int = config.get("bb_period", 20)
        self.bb_std: float = config.get("bb_std_dev", 2.0)

        # ATR periods for volatility-aware stake sizing
        self.atr_period: int = config.get("atr_period", 14)
        self.atr_baseline_period: int = config.get("atr_baseline_period", 50)

        # Confirmation state
        self._consecutive_signal: str = "HOLD"
        self._consecutive_count: int = 0
        self._last_fired: str = "HOLD"

    def evaluate(self, tick_store) -> Signal:
        rsi = tick_store.rsi()
        price = tick_store.latest_price

        # ── Warmup ───────────────────────────────────────────────
        if rsi is None or price is None:
            needed = tick_store.rsi_period + 1
            have = tick_store.tick_count
            return Signal(
                action="HOLD",
                reason=f"Warming up ({have}/{needed} ticks collected)",
            )

        # ── Collect indicator values ─────────────────────────────
        ema = tick_store.ema(self.ema_period) if self.ema_period > 0 else None
        bb = tick_store.bollinger_bands(self.bb_period, self.bb_std) if self.use_bb_filter else None
        atr = tick_store.atr(self.atr_period)
        atr_baseline = tick_store.atr(self.atr_baseline_period)

        # ── Step 1: Determine RSI zone ───────────────────────────
        if rsi > self.overbought:
            candidate = "BUY_FALL"
        elif rsi < self.oversold:
            candidate = "BUY_RISE"
        else:
            self._reset_state()
            return Signal(
                action="HOLD",
                reason=f"RSI {rsi} neutral ({self.oversold}–{self.overbought})",
                rsi=rsi,
                atr=atr,
                atr_baseline=atr_baseline,
            )

        # ── Step 2: EMA trend filter ─────────────────────────────
        if ema is not None:
            trend_up = price > ema
            if candidate == "BUY_RISE" and not trend_up:
                return Signal(
                    action="HOLD",
                    reason=f"OS but trend DOWN ({price:.5f} < EMA{self.ema_period} {ema:.5f})",
                    rsi=rsi, atr=atr, atr_baseline=atr_baseline,
                )
            if candidate == "BUY_FALL" and trend_up:
                return Signal(
                    action="HOLD",
                    reason=f"OB but trend UP ({price:.5f} > EMA{self.ema_period} {ema:.5f})",
                    rsi=rsi, atr=atr, atr_baseline=atr_baseline,
                )

        # ── Step 3: Bollinger Band confirmation ──────────────────
        # Require price to be in the outer half of the BB channel
        # (between middle and the relevant band), confirming the
        # RSI signal is backed by price being genuinely elevated/depressed.
        if bb is not None:
            upper, middle, lower = bb
            if candidate == "BUY_FALL" and price <= middle:
                return Signal(
                    action="HOLD",
                    reason=f"RSI overbought but price {price:.5f} not in upper BB half (mid {middle:.5f})",
                    rsi=rsi, atr=atr, atr_baseline=atr_baseline,
                )
            if candidate == "BUY_RISE" and price >= middle:
                return Signal(
                    action="HOLD",
                    reason=f"RSI oversold but price {price:.5f} not in lower BB half (mid {middle:.5f})",
                    rsi=rsi, atr=atr, atr_baseline=atr_baseline,
                )

        # ── Step 4: Suppression — already traded this zone ───────
        if candidate == self._last_fired:
            return Signal(
                action="HOLD",
                reason=f"RSI {rsi} — already traded this zone, waiting for RSI reset",
                rsi=rsi, atr=atr, atr_baseline=atr_baseline,
            )

        # ── Step 5: Confirmation ticks ───────────────────────────
        if candidate == self._consecutive_signal:
            self._consecutive_count += 1
        else:
            self._consecutive_signal = candidate
            self._consecutive_count = 1

        if self._consecutive_count < self.confirm_ticks:
            return Signal(
                action="HOLD",
                reason=f"RSI {rsi} — confirming ({self._consecutive_count}/{self.confirm_ticks} ticks)",
                rsi=rsi, atr=atr, atr_baseline=atr_baseline,
            )

        # ── Signal confirmed ──────────────────────────────────────
        self._last_fired = candidate
        direction = "OB->fall" if candidate == "BUY_FALL" else "OS->rise"
        trend_arrow = "^" if (ema and price > ema) else "v"
        return Signal(
            action=candidate,
            reason=(
                f"RSI {rsi} {direction} | EMA{self.ema_period}{trend_arrow} | "
                f"BB {'on' if bb else 'off'} | "
                f"confirmed {self._consecutive_count}/{self.confirm_ticks}"
            ),
            rsi=rsi,
            atr=atr,
            atr_baseline=atr_baseline,
        )

    def _reset_state(self):
        self._consecutive_signal = "HOLD"
        self._consecutive_count = 0
        self._last_fired = "HOLD"
