"""
Crash/Boom Post-Spike Recoil Strategy — Professional Entry Edition.

Core edge: after an algorithmic spike, the next several ticks revert toward
the pre-spike level (recoil). An Accumulator (ACCU) contract grows each tick
the price stays within Deriv's barrier band, then is sold for compound profit.

Professional entry rules (3 gates, all must pass):
  1. Spike gate   — current tick move > spike_mult x pre-spike ATR
  2. Confirm gate — NEXT tick (first recoil tick) must be SMALL:
                    consume < confirm_threshold x barrier_pct of price.
                    Direction is NOT checked — ACCU is direction-agnostic.
                    A large first tick signals ongoing volatility → skip.
  3. Volatility   — recent 1m tick range < volatility_filter_mult x ATR (optional)

The confirm gate filters out entries where the market is still turbulent
right after the spike, which predicts barrier breach during the hold.
barrier_pct is the real Deriv barrier fraction from check_contracts.py.

Config keys (all optional):
  symbol_type              "crash" or "boom"            (default: "crash")
  spike_mult               ATR multiple threshold       (default: 15.0)
  atr_period               Lookback for baseline ATR    (default: 50)
  cooldown_ticks           Skips after spike fires      (default: 5)
  loss_cooldown            Consecutive losses -> pause  (default: 0)
  barrier_pct              Real Deriv barrier fraction  (default: 0 = skip gate)
                           Get from check_contracts.py
  confirm_threshold        Max barrier fraction a confirm tick may use
                           (default: 1.0 = 100% of barrier)
  volatility_filter_window 1m tick range window; 0=off  (default: 0)
  volatility_filter_mult   Max range in ATR multiples   (default: 3.0)
  trend_filter_window      Pre-spike trend window; 0=off (default: 0)
"""

from .base import BaseStrategy, Signal


class CrashBoomRecoilStrategy(BaseStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.symbol_type              = str(config.get("symbol_type", "crash")).lower()
        self.spike_mult               = float(config.get("spike_mult", 15.0))
        self.atr_period               = int(config.get("atr_period", 50))
        self.cooldown_ticks           = int(config.get("cooldown_ticks", 5))
        self.loss_cooldown            = int(config.get("loss_cooldown", 0))
        self.barrier_pct              = float(config.get("barrier_pct", 0.0))
        self.confirm_threshold        = float(config.get("confirm_threshold", 1.0))
        self.volatility_filter_window = int(config.get("volatility_filter_window", 0))
        self.volatility_filter_mult   = float(config.get("volatility_filter_mult", 3.0))
        self.trend_filter_window      = int(config.get("trend_filter_window", 0))
        # Binary mode: enter CALL/PUT after spike instead of ACCU.
        # CRASH spike → BUY_RISE (recoil up). BOOM spike → BUY_FALL (recoil down).
        self.use_binary               = bool(config.get("use_binary", False))

        self._cooldown             = 0
        self._consecutive_losses   = 0
        self._extra_cooldown       = 0
        self._waiting_confirmation = False
        self._pending_reason       = ""
        self._pending_atr          = None
        self._pending_action       = ""   # stores the buy action through the confirm-tick delay

    # ------------------------------------------------------------------ #
    #  Pre-spike ATR (computed excluding the current tick)
    # ------------------------------------------------------------------ #

    def _pre_spike_atr(self, prices: list[float]) -> float | None:
        hist   = prices[-(self.atr_period + 2): -1]
        if len(hist) < self.atr_period + 1:
            return None
        ranges = [abs(hist[i] - hist[i - 1]) for i in range(1, len(hist))]
        avg    = sum(ranges[: self.atr_period]) / self.atr_period
        for r in ranges[self.atr_period:]:
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

        if n < self.atr_period + 2:
            return Signal(action="HOLD", reason=f"Warming up ({n}/{self.atr_period + 2})")

        pre_atr = self._pre_spike_atr(prices)
        if pre_atr is None or pre_atr <= 0:
            return Signal(action="HOLD", reason="Pre-spike ATR not ready")

        # ── Gate 2: Confirmation tick (fires on tick AFTER spike) ─────────
        # ACCU only — checks SIZE of first post-spike tick; skips if still volatile.
        # Binary mode skips this gate (direction, not size, determines the win).
        if self._waiting_confirmation:
            self._waiting_confirmation = False
            abs_tick = abs(prices[-1] - prices[-2])

            if (not self.use_binary
                    and self.barrier_pct > 0
                    and self.confirm_threshold > 0
                    and prices[-2] > 0):
                tick_pct    = abs_tick / prices[-2]
                max_allowed = self.confirm_threshold * self.barrier_pct
                if tick_pct > max_allowed:
                    return Signal(
                        action="HOLD",
                        reason=(
                            f"Confirm SKIP: tick {tick_pct:.2e} > "
                            f"{max_allowed:.2e} ({self.confirm_threshold:.0%} of barrier)"
                        ),
                        atr=self._pending_atr,
                    )

            return Signal(action=self._pending_action, reason=self._pending_reason, atr=self._pending_atr)

        # ── Cooldown checks ───────────────────────────────────────────────
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

        # ── Gate 1: Spike detection ───────────────────────────────────────
        last_move = prices[-1] - prices[-2]
        abs_move  = abs(last_move)
        mult      = abs_move / pre_atr
        threshold = self.spike_mult * pre_atr

        # Gate 3a: Trend filter — pre-spike trend must align with recoil direction
        if self.trend_filter_window > 0 and abs_move >= threshold:
            w         = min(self.trend_filter_window, n - 2)
            pre_trend = prices[-2] - prices[-(w + 2)]
            if self.symbol_type == "crash" and pre_trend <= 0:
                return Signal(
                    action="HOLD",
                    reason=f"Trend filter: pre-spike trend {pre_trend:+.4f} not upward",
                    atr=pre_atr,
                )
            if self.symbol_type == "boom" and pre_trend >= 0:
                return Signal(
                    action="HOLD",
                    reason=f"Trend filter: pre-spike trend {pre_trend:+.4f} not downward",
                    atr=pre_atr,
                )

        # Gate 3b: Volatility filter — 1m range must be calm relative to ATR
        if self.volatility_filter_window > 0 and abs_move >= threshold:
            w          = min(self.volatility_filter_window, n - 2)
            recent     = prices[-(w + 2):-1]
            tick_range = max(recent) - min(recent)
            if tick_range > self.volatility_filter_mult * pre_atr:
                return Signal(
                    action="HOLD",
                    reason=f"Volatility filter: range {tick_range:.4f} > {self.volatility_filter_mult}x ATR",
                    atr=pre_atr,
                )

        if self.symbol_type == "crash" and last_move < 0 and abs_move >= threshold:
            self._cooldown = self.cooldown_ticks
            buy_action = "BUY_RISE" if self.use_binary else "BUY_ACCU"
            mode       = "binary CALL" if self.use_binary else "ACCU recoil"
            reason     = f"CRASH spike: -{abs_move:.4f} ({mult:.0f}xATR) -> {mode}"
            if self.barrier_pct > 0 or self.use_binary:
                self._waiting_confirmation = True
                self._pending_reason       = reason
                self._pending_atr          = pre_atr
                self._pending_action       = buy_action
                return Signal(action="HOLD", reason="CRASH spike: awaiting confirm tick", atr=pre_atr)
            return Signal(action="BUY_ACCU", reason=reason, atr=pre_atr)

        if self.symbol_type == "boom" and last_move > 0 and abs_move >= threshold:
            self._cooldown = self.cooldown_ticks
            buy_action = "BUY_FALL" if self.use_binary else "BUY_ACCU"
            mode       = "binary PUT" if self.use_binary else "ACCU recoil"
            reason     = f"BOOM spike: +{abs_move:.4f} ({mult:.0f}xATR) -> {mode}"
            if self.barrier_pct > 0 or self.use_binary:
                self._waiting_confirmation = True
                self._pending_reason       = reason
                self._pending_atr          = pre_atr
                self._pending_action       = buy_action
                return Signal(action="HOLD", reason="BOOM spike: awaiting confirm tick", atr=pre_atr)
            return Signal(action="BUY_ACCU", reason=reason, atr=pre_atr)

        # Jump indices: spikes in both directions.
        # Large DOWN spike → BUY_RISE (CALL). Large UP spike → BUY_FALL (PUT).
        if self.symbol_type == "jump" and abs_move >= threshold:
            self._cooldown = self.cooldown_ticks
            if last_move < 0:
                buy_action = "BUY_RISE"
                reason     = f"JUMP down spike: -{abs_move:.4f} ({mult:.0f}xATR) -> binary CALL"
            else:
                buy_action = "BUY_FALL"
                reason     = f"JUMP up spike: +{abs_move:.4f} ({mult:.0f}xATR) -> binary PUT"
            self._waiting_confirmation = True
            self._pending_reason       = reason
            self._pending_atr          = pre_atr
            self._pending_action       = buy_action
            return Signal(action="HOLD", reason="JUMP spike: awaiting confirm tick", atr=pre_atr)

        return Signal(
            action="HOLD",
            reason=f"No spike (move={last_move:+.4f}, {mult:.1f}xATR, need {self.spike_mult:.0f}x)",
            atr=pre_atr,
        )
