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
  min_spike_ratio          Min spike size in ATR multiples; 0=off (default: 0.0)
                           Filters weak spikes near the spike_mult threshold.
  cluster_window           Tick window for cluster detection; 0=off (default: 0)
  max_cluster_spikes       Spikes in cluster_window that trigger block (default: 3)
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
        self.settle_ticks             = int(config.get("settle_ticks", 0))
        self.min_spike_ratio          = float(config.get("min_spike_ratio", 0.0))
        self.cluster_window           = int(config.get("cluster_window", 0))
        self.max_cluster_spikes       = int(config.get("max_cluster_spikes", 3))
        # Adaptive ATR settle: instead of waiting a fixed settle_ticks, wait until
        # the short-term ATR drops below settle_atr_ratio × barrier width.
        # Requires barrier_pct > 0. Overrides settle_ticks when enabled.
        self.adaptive_settle          = bool(config.get("adaptive_settle", False))
        self.settle_atr_ratio         = float(config.get("settle_atr_ratio", 0.5))
        self.settle_short_period      = int(config.get("settle_short_period", 5))
        self.max_settle_ticks         = int(config.get("max_settle_ticks", 30))

        self._cooldown             = 0
        self._consecutive_losses   = 0
        self._extra_cooldown       = 0
        self._waiting_confirmation = False
        self._settle_remaining     = 0
        self._adaptive_settling    = False
        self._adaptive_settle_count = 0
        self._pending_reason       = ""
        self._pending_atr          = None
        self._pending_action       = ""
        self._tick_idx             = 0
        self._spike_ticks: list[int] = []

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

    def _short_atr(self, prices: list[float], period: int) -> float | None:
        hist = prices[-(period + 1):]
        if len(hist) < period + 1:
            return None
        ranges = [abs(hist[i] - hist[i - 1]) for i in range(1, len(hist))]
        avg = sum(ranges) / len(ranges)
        return avg if avg > 0 else None

    def evaluate(self, tick_store) -> Signal:
        self._tick_idx += 1
        prices = tick_store.prices
        n      = len(prices)

        if n < self.atr_period + 2:
            return Signal(action="HOLD", reason=f"Warming up ({n}/{self.atr_period + 2})")

        pre_atr = self._pre_spike_atr(prices)
        if pre_atr is None or pre_atr <= 0:
            return Signal(action="HOLD", reason="Pre-spike ATR not ready")

        # ── Adaptive ATR settle: wait for volatility to fall below barrier fraction ─
        if self._adaptive_settling:
            self._adaptive_settle_count += 1
            if self._cooldown > 0:
                self._cooldown -= 1

            # New spike during settle — abandon current entry, restart for new spike
            last_move = prices[-1] - prices[-2]
            abs_move  = abs(last_move)
            if abs_move > self.spike_mult * pre_atr:
                self._adaptive_settling = False
                self._cooldown = self.cooldown_ticks
                if ((self.symbol_type == "crash" and last_move < 0) or
                        (self.symbol_type == "boom" and last_move > 0)):
                    buy_action = "BUY_RISE" if self.use_binary else "BUY_ACCU"
                    self._waiting_confirmation = True
                    self._pending_action = buy_action
                    self._pending_reason = (
                        f"{self.symbol_type.upper()} spike during adaptive settle "
                        f"({abs_move:.4f}, {abs_move/pre_atr:.0f}xATR) — restart"
                    )
                    self._pending_atr = pre_atr
                return Signal(action="HOLD", reason="Adaptive settle: new spike — restarting",
                              atr=pre_atr, close_open_accus=True)

            if self._adaptive_settle_count > self.max_settle_ticks:
                self._adaptive_settling = False
                return Signal(
                    action="HOLD",
                    reason=f"Adaptive settle: timed out at {self.max_settle_ticks}t — skipping entry",
                    atr=pre_atr,
                )

            satr = self._short_atr(prices, self.settle_short_period)
            barrier_abs = self.barrier_pct * prices[-1] if prices[-1] > 0 else 0.0
            if satr is not None and barrier_abs > 0 and satr < self.settle_atr_ratio * barrier_abs:
                self._adaptive_settling = False
                return Signal(
                    action=self._pending_action,
                    reason=(
                        f"{self._pending_reason} | ATR settle {self._adaptive_settle_count}t: "
                        f"atr={satr:.2e} < {self.settle_atr_ratio}×barrier={barrier_abs:.2e}"
                    ),
                    atr=self._pending_atr,
                )

            satr_str = f"{satr:.2e}" if satr is not None else "?"
            return Signal(
                action="HOLD",
                reason=f"Adaptive settle ({self._adaptive_settle_count}t): atr={satr_str} need <{self.settle_atr_ratio}×{barrier_abs:.2e}",
                atr=pre_atr,
            )

        # ── Settle delay: ticks after confirm gate, before opening ACCU ───
        if self._settle_remaining > 0:
            self._settle_remaining -= 1
            if self._settle_remaining == 0:
                return Signal(action=self._pending_action, reason=self._pending_reason, atr=self._pending_atr)
            return Signal(action="HOLD", reason=f"Settle delay ({self._settle_remaining} ticks left)", atr=self._pending_atr)

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

            # Confirm gate passed — choose settle mode
            if self.adaptive_settle and self.barrier_pct > 0:
                self._adaptive_settling = True
                self._adaptive_settle_count = 0
                return Signal(action="HOLD", reason="Adaptive settle: waiting for ATR calm", atr=self._pending_atr)

            if self.settle_ticks > 0:
                self._settle_remaining = self.settle_ticks
                return Signal(action="HOLD", reason=f"Settle delay: {self.settle_ticks} ticks", atr=self._pending_atr)

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

        # Gate: spike quality + cluster — shared across all symbol types
        if abs_move >= threshold:
            if self.min_spike_ratio > 0 and mult < self.min_spike_ratio:
                return Signal(
                    action="HOLD",
                    reason=f"Spike weak: {mult:.1f}x < min {self.min_spike_ratio:.0f}x ATR — skipping",
                    atr=pre_atr,
                )
            if self.cluster_window > 0:
                self._spike_ticks.append(self._tick_idx)
                cutoff = self._tick_idx - self.cluster_window
                self._spike_ticks = [t for t in self._spike_ticks if t > cutoff]
                if len(self._spike_ticks) >= self.max_cluster_spikes:
                    return Signal(
                        action="HOLD",
                        reason=f"Spike cluster: {len(self._spike_ticks)} in {self.cluster_window}t — regime too noisy",
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
                return Signal(action="HOLD", reason="CRASH spike: awaiting confirm tick",
                              atr=pre_atr, close_open_accus=not self.use_binary)
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
                return Signal(action="HOLD", reason="BOOM spike: awaiting confirm tick",
                              atr=pre_atr, close_open_accus=not self.use_binary)
            return Signal(action="BUY_ACCU", reason=reason, atr=pre_atr)

        # Volatility indices: ACCU is direction-agnostic — fire on any large tick.
        if self.symbol_type == "vol" and abs_move >= threshold:
            self._cooldown = self.cooldown_ticks
            reason = f"VOL spike: {last_move:+.4f} ({mult:.0f}xATR) -> ACCU"
            if self.barrier_pct > 0:
                self._waiting_confirmation = True
                self._pending_reason       = reason
                self._pending_atr          = pre_atr
                self._pending_action       = "BUY_ACCU"
                return Signal(action="HOLD", reason="VOL spike: awaiting confirm tick", atr=pre_atr)
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
