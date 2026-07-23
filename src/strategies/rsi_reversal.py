"""
RSI Reversal Strategy — multi-layer confirmation for higher win-rate entries.

Signal pipeline (each layer must pass before moving to the next):

  0. ATR gate        Block ALL signals when ATR(fast) > ATR(slow) * ratio.
                     Trending bursts are the primary cause of consecutive RSI
                     reversal failures; this keeps the bot out of those periods.
  1. RSI zone        RSI(fast) must be in oversold/overbought territory.
  2. RSI slope       RSI must be TURNING toward neutral — catching the actual
                     reversal moment, not entering into a continuing move.
  3. Price confirm   The last price tick must be moving in the reversal
                     direction (price already ticking up for BUY_RISE, etc.).
  4. Dual RSI        A slower RSI(slow) must also be approaching the extreme,
                     confirming the move is not just short-term tick noise.
  5. Confirm ticks   All of the above must hold for N consecutive ticks.
  6. Suppression     Same direction cannot fire again until RSI resets to
                     neutral, preventing repeated entries in one sustained run.
  7. Loss cooldown   After M consecutive losses the bot skips the next signal,
                     protecting against trending periods that defeat reversal.
"""

from collections import deque
from typing import Optional
from .base import BaseStrategy, Signal


class RSIReversalStrategy(BaseStrategy):

    def __init__(self, config: dict):
        super().__init__(config)

        # ── Core thresholds ───────────────────────────────────────
        self.overbought: float = config.get("rsi_overbought", 70)
        self.oversold:   float = config.get("rsi_oversold",   30)
        self.confirm_ticks: int = config.get("confirm_ticks", 2)

        # ── EMA / BB (kept for config compatibility, not used in signal logic)
        self.ema_period:    int   = config.get("ema_trend_period", 0)
        self.use_bb_filter: bool  = config.get("use_bb_filter", False)

        # ── Layer 0: ATR ranging-market gate ─────────────────────
        # Block ALL signals when short-term ATR > long-term ATR * ratio.
        # Trending bursts are the main cause of consecutive RSI reversal failures.
        self.use_atr_filter:   bool  = config.get("use_atr_filter", True)
        self.atr_filter_ratio: float = config.get("atr_filter_ratio", 1.0)

        # ── Layer 2: RSI slope confirmation ───────────────────────
        # RSI must be moving toward neutral (turning from the extreme).
        # Filters out entries where price is still diving / still surging.
        self.rsi_slope_confirm: bool = config.get("rsi_slope_confirm", True)

        # ── Layer 3: Price direction confirmation ─────────────────
        # The last tick must move in the expected reversal direction.
        self.price_confirm: bool = config.get("price_confirm", True)

        # ── Layer 4: Dual RSI (multi-timeframe) ───────────────────
        # A second, slower RSI(dual_rsi_period) must also be near the extreme.
        # dual_rsi_threshold: slow RSI must be < this for BUY_RISE,
        #                                       > (100 - this) for BUY_FALL.
        # Set dual_rsi_period = 0 to disable.
        self.dual_rsi_period:    int   = config.get("dual_rsi_period", 14)
        self.dual_rsi_threshold: float = config.get("dual_rsi_threshold", 45.0)

        # ── Layer 4b: RSI Divergence ──────────────────────────────
        # Only trade when price and RSI are diverging: price reaches a new
        # extreme but RSI fails to confirm it — the classic reversal setup.
        # use_divergence=False means divergence is ignored (same as before).
        self.use_divergence:       bool = config.get("use_divergence", False)
        self.divergence_lookback:  int  = config.get("divergence_lookback", 20)

        # ── Adaptive RSI thresholds ───────────────────────────────
        # Dynamically set overbought/oversold as percentiles of recent RSI history.
        # Adapts to slow/fast market regimes. Falls back to fixed thresholds during
        # warmup (first `adaptive_window` ticks).
        # Set adaptive_threshold=False to keep fixed 80/20 (default).
        self.adaptive_threshold: bool  = config.get("adaptive_threshold", False)
        self.adaptive_window:    int   = config.get("adaptive_window", 500)
        self.adaptive_pct:       float = config.get("adaptive_pct", 95.0)

        # ── RSI-depth contract duration ───────────────────────────
        # When RSI is more extreme than `rsi_extreme_threshold`, use a longer
        # contract duration (`rsi_extreme_duration` ticks) to give more time to
        # revert. Set rsi_extreme_threshold=0 to disable.
        self.rsi_extreme_threshold: float = config.get("rsi_extreme_threshold", 0.0)
        self.rsi_extreme_duration:  int   = config.get("rsi_extreme_duration", 15)

        # ── Layer 7: Loss cooldown ────────────────────────────────
        # After `loss_cooldown` consecutive losses skip the next signal entirely.
        # Set to 0 to disable.
        self.loss_cooldown: int = config.get("loss_cooldown", 3)

        # ── Midline crossover mode ────────────────────────────────
        # When True: trade momentum at RSI 50 crossover instead of
        # reversal at RSI extremes. BUY_RISE when RSI crosses above 50,
        # BUY_FALL when RSI crosses below 50.
        self.use_midline_cross: bool = config.get("use_midline_cross", False)

        # ── Momentum mode ─────────────────────────────────────────
        # When True: trade WITH momentum instead of against it.
        # RSI > overbought → BUY_RISE (riding upward momentum)
        # RSI < oversold   → BUY_FALL (riding downward momentum)
        # Pair with asymmetric thresholds e.g. rsi_overbought:50, rsi_oversold:30
        self.momentum_mode: bool = config.get("momentum_mode", False)

        # ── ATR periods (passed through to Signal for stake sizing) ──
        self.atr_period:          int = config.get("atr_period", 14)
        self.atr_baseline_period: int = config.get("atr_baseline_period", 50)

        # ── Internal state ────────────────────────────────────────
        self._prev_rsi:   Optional[float] = None
        self._consecutive_signal: str = "HOLD"
        self._consecutive_count:  int  = 0
        self._last_fired:         str  = "HOLD"
        self._consecutive_losses: int  = 0
        self._cooldown_remaining: int  = 0

    # ------------------------------------------------------------------ #
    #  Public result callback — call after every closed trade
    # ------------------------------------------------------------------ #

    def on_result(self, won: bool) -> None:
        """Trader / backtest engine calls this after each contract settles."""
        if won:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            if self.loss_cooldown > 0 and self._consecutive_losses >= self.loss_cooldown:
                self._cooldown_remaining = 1          # skip next 1 signal
                self._consecutive_losses = 0
                self._last_fired = "HOLD"             # allow fresh entry after cooldown

    # ------------------------------------------------------------------ #
    #  Main evaluate loop
    # ------------------------------------------------------------------ #

    def evaluate(self, tick_store) -> Signal:
        rsi   = tick_store.rsi()
        price = tick_store.latest_price

        # ── Warmup ───────────────────────────────────────────────
        if rsi is None or price is None:
            needed = tick_store.warmup_ticks
            return Signal(
                action="HOLD",
                reason=f"Warming up ({tick_store.tick_count}/{needed} ticks)",
            )

        # ── Collect ATR for ranging filter + stake sizing ────────
        atr          = tick_store.atr(self.atr_period)
        atr_baseline = tick_store.atr(self.atr_baseline_period)

        # ── Adaptive RSI thresholds ───────────────────────────────
        if self.adaptive_threshold:
            dyn_ob = tick_store.rsi_percentile(self.adaptive_window, self.adaptive_pct)
            dyn_os = tick_store.rsi_percentile(self.adaptive_window, 100.0 - self.adaptive_pct)
            effective_ob = dyn_ob if dyn_ob is not None else self.overbought
            effective_os = dyn_os if dyn_os is not None else self.oversold
        else:
            effective_ob = self.overbought
            effective_os = self.oversold

        # ── Layer 0: ATR ranging-market gate ─────────────────────
        # Skip everything when the market is trending (ATR spiking above baseline).
        if (self.use_atr_filter
                and atr is not None
                and atr_baseline is not None
                and atr_baseline > 0):
            if atr > atr_baseline * self.atr_filter_ratio:
                self._reset_confirmation()
                return Signal(
                    action="HOLD",
                    reason=f"ATR gate: volatile ({atr:.6f} > {atr_baseline:.6f}x{self.atr_filter_ratio})",
                    rsi=rsi, atr=atr, atr_baseline=atr_baseline,
                )

        # Track previous RSI for slope check
        prev_rsi      = self._prev_rsi
        self._prev_rsi = rsi

        prices = tick_store.prices

        # ── Step 1: RSI zone / crossover detection ───────────────
        if self.use_midline_cross:
            # Momentum mode: fire when RSI crosses the 50 midline.
            if prev_rsi is None:
                return Signal(action="HOLD", reason="Crossover: waiting for prev RSI",
                              rsi=rsi, atr=atr, atr_baseline=atr_baseline)
            above      = rsi      >= 50
            prev_above = prev_rsi >= 50
            if above and not prev_above:
                candidate = "BUY_RISE"          # fresh cross above 50
            elif not above and prev_above:
                candidate = "BUY_FALL"          # fresh cross below 50
            elif above and self._consecutive_signal == "BUY_RISE":
                candidate = "BUY_RISE"          # confirming above 50
            elif not above and self._consecutive_signal == "BUY_FALL":
                candidate = "BUY_FALL"          # confirming below 50
            else:
                self._reset_confirmation()
                return Signal(action="HOLD",
                              reason=f"RSI {rsi} — no crossover signal",
                              rsi=rsi, atr=atr, atr_baseline=atr_baseline)
        else:
            if rsi > effective_ob:
                candidate = "BUY_RISE" if self.momentum_mode else "BUY_FALL"
            elif rsi < effective_os:
                candidate = "BUY_FALL" if self.momentum_mode else "BUY_RISE"
            else:
                self._reset_confirmation()
                return Signal(
                    action="HOLD",
                    reason=f"RSI {rsi} neutral ({effective_os:.0f}-{effective_ob:.0f})",
                    rsi=rsi, atr=atr, atr_baseline=atr_baseline,
                )

        # ── Loss cooldown ─────────────────────────────────────────
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return Signal(
                action="HOLD",
                reason=f"Loss cooldown active ({self._cooldown_remaining + 1} remaining)",
                rsi=rsi, atr=atr, atr_baseline=atr_baseline,
            )

        # ── Suppression — already traded this zone ────────────────
        if candidate == self._last_fired:
            return Signal(
                action="HOLD",
                reason=f"RSI {rsi} — already traded this zone, waiting for reset",
                rsi=rsi, atr=atr, atr_baseline=atr_baseline,
            )

        # ── Step 2: RSI slope confirmation ───────────────────────
        if self.rsi_slope_confirm and prev_rsi is not None:
            turning = (
                (candidate == "BUY_RISE" and rsi > prev_rsi) or
                (candidate == "BUY_FALL" and rsi < prev_rsi)
            )
            if not turning:
                self._reset_confirmation()
                return Signal(
                    action="HOLD",
                    reason=f"RSI {rsi} in zone but not turning yet (slope {'up' if rsi > prev_rsi else 'down'} needed {'up' if candidate == 'BUY_RISE' else 'down'})",
                    rsi=rsi, atr=atr, atr_baseline=atr_baseline,
                )

        # ── Step 3: Price direction confirmation ─────────────────
        if self.price_confirm and len(prices) >= 2:
            price_ok = (
                (candidate == "BUY_RISE" and prices[-1] > prices[-2]) or
                (candidate == "BUY_FALL" and prices[-1] < prices[-2])
            )
            if not price_ok:
                self._reset_confirmation()
                return Signal(
                    action="HOLD",
                    reason=f"RSI {rsi} turning but price not confirming yet",
                    rsi=rsi, atr=atr, atr_baseline=atr_baseline,
                )

        # ── Step 4: Dual RSI confirmation ─────────────────────────
        if self.dual_rsi_period > 0:
            rsi_slow = tick_store.rsi_simple(self.dual_rsi_period)
            if rsi_slow is not None:
                slow_ok = (
                    (candidate == "BUY_RISE" and rsi_slow < self.dual_rsi_threshold) or
                    (candidate == "BUY_FALL" and rsi_slow > (100 - self.dual_rsi_threshold))
                )
                if not slow_ok:
                    return Signal(
                        action="HOLD",
                        reason=f"RSI{self.dual_rsi_period} {rsi_slow} not confirming (need {'<' if candidate == 'BUY_RISE' else '>'}{self.dual_rsi_threshold if candidate == 'BUY_RISE' else 100 - self.dual_rsi_threshold})",
                        rsi=rsi, atr=atr, atr_baseline=atr_baseline,
                    )

        # ── Step 5: Confirmation ticks ────────────────────────────
        if candidate == self._consecutive_signal:
            self._consecutive_count += 1
        else:
            self._consecutive_signal = candidate
            self._consecutive_count  = 1

        if self._consecutive_count < self.confirm_ticks:
            return Signal(
                action="HOLD",
                reason=f"RSI {rsi} confirming ({self._consecutive_count}/{self.confirm_ticks})",
                rsi=rsi, atr=atr, atr_baseline=atr_baseline,
            )

        # ── Step 5b: RSI divergence (final gate on fire tick) ────
        # Checked after confirm so confirmation state is never disrupted.
        # If divergence not present, keep waiting — next tick re-checks.
        if self.use_divergence:
            if candidate == "BUY_FALL":
                has_div = tick_store.bearish_divergence(self.divergence_lookback)
            else:
                has_div = tick_store.bullish_divergence(self.divergence_lookback)
            if not has_div:
                return Signal(
                    action="HOLD",
                    reason=f"RSI {rsi} confirmed but waiting for divergence (lookback={self.divergence_lookback})",
                    rsi=rsi, atr=atr, atr_baseline=atr_baseline,
                )

        # ── Signal confirmed ──────────────────────────────────────
        self._last_fired = candidate
        if self.use_midline_cross:
            direction = "cross>50->rise" if candidate == "BUY_RISE" else "cross<50->fall"
        elif self.momentum_mode:
            direction = "MOM->rise" if candidate == "BUY_RISE" else "MOM->fall"
        else:
            direction = "OS->rise" if candidate == "BUY_RISE" else "OB->fall"
        slow_tag  = ""
        if self.dual_rsi_period > 0:
            rs = tick_store.rsi_simple(self.dual_rsi_period)
            slow_tag = f" | RSI{self.dual_rsi_period}:{rs}"

        # RSI-depth duration: deeper extreme → longer contract
        duration = None
        if self.rsi_extreme_threshold > 0:
            deep = (candidate == "BUY_FALL" and rsi >= self.rsi_extreme_threshold) or \
                   (candidate == "BUY_RISE" and rsi <= (100.0 - self.rsi_extreme_threshold))
            if deep:
                duration = self.rsi_extreme_duration

        dur_tag = f" | dur:{duration}t" if duration is not None else ""
        return Signal(
            action=candidate,
            reason=f"RSI{tick_store.rsi_period}:{rsi} {direction}{slow_tag}{dur_tag} | confirmed {self._consecutive_count}/{self.confirm_ticks}",
            rsi=rsi,
            atr=atr,
            atr_baseline=atr_baseline,
            contract_duration=duration,
        )

    def _reset_confirmation(self):
        self._consecutive_signal = "HOLD"
        self._consecutive_count  = 0
