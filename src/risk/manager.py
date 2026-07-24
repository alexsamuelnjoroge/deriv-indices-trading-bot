"""
Risk Manager — protects the account balance.

Rules enforced:
  1. Kelly stake sizing: optimal fraction based on measured edge.
     Falls back to flat stake_percent until kelly_min_trades reached.
  2. ATR-adaptive sizing: further scale stake with current volatility.
  3. Daily loss circuit breaker: stops trading when daily loss > limit.
  4. Rolling performance monitor: halts if recent win rate falls below
     min_rolling_wr over the last rolling_window trades (regime change guard).
  5. Tracks open contract count; blocks new trades if limit reached.
"""

import json
import os
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from loguru import logger


@dataclass
class TradeResult:
    contract_id: str
    contract_type: str   # CALL or PUT
    stake: float
    payout: float
    profit: float        # positive = win, negative = loss
    status: str          # "won" or "lost"


class RiskManager:
    def __init__(self, config: dict, starting_balance: float, symbol: str = "", persist_halt: bool = True):
        self.symbol = symbol
        self.stake_percent: float  = config.get("stake_percent", 2.0)
        self.max_stake: float      = config.get("max_stake", 100.0)
        self.daily_loss_limit: float = config.get("daily_loss_limit", 10.0)
        self.max_open_contracts: int = config.get("max_open_contracts", 1)
        self.use_atr_stake: bool   = config.get("use_atr_stake", True)

        # ── Kelly sizing ──────────────────────────────────────────────
        self.use_kelly: bool        = config.get("use_kelly", True)
        self.kelly_fraction: float  = config.get("kelly_fraction", 0.5)
        self.kelly_min_trades: int  = config.get("kelly_min_trades", 20)
        self.payout_pct: float      = config.get("payout_pct", 0.87)

        # ── Rolling performance monitor ───────────────────────────────
        self.rolling_window: int    = config.get("rolling_window", 30)
        self.min_rolling_wr: float  = config.get("min_rolling_wr", 48.0)
        self.perf_halt_reset_hours: float = config.get("perf_halt_reset_hours", 4.0)

        # ── Time filter ───────────────────────────────────────────────
        self.blocked_hours_eat: list = config.get("blocked_hours_eat", [])

        # ── Paper trade recovery ──────────────────────────────────────
        self.paper_resume_min_trades: int = config.get("paper_resume_min_trades", 5)
        self.paper_resume_wr: float       = config.get("paper_resume_wr", 60.0)
        self._paper_pending: Optional[dict] = None   # {ct, entry_price, ticks_left}
        self._paper_wins: int               = 0
        self._paper_losses: int             = 0

        self.starting_balance: float    = starting_balance
        self.current_balance: float     = starting_balance
        self.daily_start_balance: float = starting_balance

        self._persist_halt: bool       = persist_halt
        self.open_contracts: int       = 0
        self.trade_history: list[TradeResult] = []
        self._rolling: deque           = deque(maxlen=self.rolling_window)
        self.total_trades: int         = 0
        self.wins: int                 = 0
        self.losses: int               = 0
        self.is_halted: bool           = False
        self._halt_reason: str         = ""
        self._halt_time: Optional[float] = None

        if self._persist_halt:
            self._load_halt_state()

    # ------------------------------------------------------------------ #
    #  Pre-trade checks
    # ------------------------------------------------------------------ #

    def check_auto_reset(self) -> None:
        """Call on every tick — fires the 4h performance halt auto-reset on schedule."""
        if (
            self.is_halted
            and "rolling WR" in self._halt_reason
            and self._halt_time is not None
            and time.time() - self._halt_time >= self.perf_halt_reset_hours * 3600
        ):
            tag = f"[{self.symbol}] " if self.symbol else ""
            logger.info(
                f"{tag}Performance halt auto-reset after {self.perf_halt_reset_hours:.0f}h — resuming trading."
            )
            self.is_halted    = False
            self._halt_reason = ""
            self._halt_time   = None
            self._rolling.clear()
            self._clear_halt_state()

    def can_trade(self) -> tuple[bool, str]:
        if self.is_halted:
            return False, f"Bot halted: {self._halt_reason}"

        if self.blocked_hours_eat:
            current_hour = datetime.now().hour
            if current_hour in self.blocked_hours_eat:
                return False, f"time filter: {current_hour:02d}:00 EAT blocked"

        if self.open_contracts >= self.max_open_contracts:
            return False, f"Max open contracts reached ({self.open_contracts}/{self.max_open_contracts})"

        daily_loss_pct = self._daily_loss_pct()
        if daily_loss_pct >= self.daily_loss_limit:
            self._halt("daily loss limit reached")
            logger.warning(
                f"[{self.symbol}] CIRCUIT BREAKER: Daily loss {daily_loss_pct:.1f}% >= limit {self.daily_loss_limit}%."
            )
            return False, "Circuit breaker triggered"

        if len(self._rolling) >= self.rolling_window:
            rwr = self.rolling_win_rate
            if rwr < self.min_rolling_wr:
                self._halt(f"rolling WR {rwr:.1f}% < {self.min_rolling_wr}% over last {self.rolling_window} trades")
                logger.warning(
                    f"[{self.symbol}] PERFORMANCE HALT: Rolling WR {rwr:.1f}% < {self.min_rolling_wr}%. "
                    f"Market regime may have changed."
                )
                return False, "Rolling performance halt"

        return True, "ok"

    # ------------------------------------------------------------------ #
    #  Stake calculation
    # ------------------------------------------------------------------ #

    def calculate_stake(
        self,
        atr: Optional[float] = None,
        atr_baseline: Optional[float] = None,
    ) -> float:
        """
        Kelly stake when enough trades observed; falls back to flat stake_percent.

        Kelly formula for binary outcomes:
            f* = (p * b - q) / b
        where b = payout, p = win_rate, q = 1 - p.
        We apply kelly_fraction (default 0.5 = half-Kelly) for safety.

        ATR-adaptive modifier still applied on top of Kelly stake.
        """
        if self.use_kelly and self.total_trades >= self.kelly_min_trades:
            p = self.wins / self.total_trades
            q = 1.0 - p
            b = self.payout_pct
            kelly_full = (p * b - q) / b
            fraction   = max(0.0, kelly_full) * self.kelly_fraction
            stake = self.current_balance * fraction
            logger.debug(
                f"Kelly stake: WR={p*100:.1f}% f*={kelly_full*100:.1f}% "
                f"half-K={fraction*100:.1f}% stake={stake:.2f}"
            )
        else:
            stake = self.current_balance * (self.stake_percent / 100)

        if self.use_atr_stake and atr and atr_baseline and atr_baseline > 0:
            vol_ratio  = atr / atr_baseline
            vol_factor = max(0.5, min(1.5, 1.0 / vol_ratio))
            stake     *= vol_factor
            logger.debug(
                f"ATR stake adjustment: ratio={vol_ratio:.2f} factor={vol_factor:.2f}"
            )

        stake = min(stake, self.max_stake)
        stake = max(stake, 0.35)
        return round(stake, 2)

    # ------------------------------------------------------------------ #
    #  Contract lifecycle
    # ------------------------------------------------------------------ #

    def on_contract_opened(self):
        self.open_contracts += 1

    def on_contract_closed(self, result: TradeResult):
        self.open_contracts = max(0, self.open_contracts - 1)
        self.current_balance += result.profit
        self.total_trades    += 1
        self.trade_history.append(result)
        won = result.status == "won"
        self._rolling.append(1 if won else 0)

        tag = f"[{self.symbol}] " if self.symbol else ""
        if won:
            self.wins += 1
            logger.info(f"{tag}WIN  | Profit: +{result.profit:.2f} | Balance: {self.current_balance:.2f}")
        else:
            self.losses += 1
            logger.info(f"{tag}LOSS | Loss:   -{result.stake:.2f} | Balance: {self.current_balance:.2f}")

    def update_balance(self, balance: float):
        self.current_balance = balance

    # ------------------------------------------------------------------ #
    #  Daily reset (call at midnight)
    # ------------------------------------------------------------------ #

    def reset_daily(self):
        self.daily_start_balance = self.current_balance
        if self.is_halted and "daily loss" in self._halt_reason:
            self.is_halted    = False
            self._halt_reason = ""
            self._halt_time   = None
            self._clear_halt_state()
            logger.info("Daily reset: circuit breaker cleared for new trading day.")
        logger.info(f"Daily reset | New baseline: {self.daily_start_balance:.2f}")

    # ------------------------------------------------------------------ #
    #  Stats
    # ------------------------------------------------------------------ #

    def on_tick_halted(
        self,
        current_price: float,
        signal_action: str,
        contract_type: str,
        duration: int,
    ) -> None:
        """
        Call on every tick while halted (rolling WR halts only).

        Tracks simulated signal outcomes without risking real money.
        If paper WR recovers to paper_resume_wr over paper_resume_min_trades
        signals, the halt is cleared and trading resumes immediately.
        Applies to both rolling-WR halts and daily-loss halts.
        """
        if not self.is_halted:
            return

        tag = f"[{self.symbol}] " if self.symbol else ""

        # ── Settle pending paper trade ────────────────────────────────
        if self._paper_pending is not None:
            self._paper_pending["ticks_left"] -= 1
            if self._paper_pending["ticks_left"] <= 0:
                ct    = self._paper_pending["ct"]
                entry = self._paper_pending["entry_price"]
                won   = (
                    (ct == "CALL" and current_price > entry) or
                    (ct == "PUT"  and current_price < entry)
                )
                if won:
                    self._paper_wins += 1
                else:
                    self._paper_losses += 1
                total  = self._paper_wins + self._paper_losses
                pwr    = round(self._paper_wins / total * 100, 1)
                label  = "WIN " if won else "LOSS"
                logger.info(
                    f"{tag}[paper] {label} | WR {pwr:.0f}% "
                    f"({self._paper_wins}W/{self._paper_losses}L of {total})"
                )
                self._paper_pending = None

                # ── Check resume threshold ────────────────────────────
                if total >= self.paper_resume_min_trades and pwr >= self.paper_resume_wr:
                    logger.info(
                        f"{tag}Paper WR {pwr:.1f}% >= {self.paper_resume_wr:.0f}% "
                        f"over {total} trades — market recovered, resuming trading."
                    )
                    self.is_halted    = False
                    self._halt_reason = ""
                    self._halt_time   = None
                    self._rolling.clear()  # fresh rolling window on resume
                    self._paper_pending = None
                    self._paper_wins    = 0
                    self._paper_losses  = 0
                    self._clear_halt_state()
            return  # don't open a new paper trade on the same tick as settlement

        # ── Open new paper trade if a signal fires ────────────────────
        if signal_action != "HOLD" and contract_type:
            total = self._paper_wins + self._paper_losses
            self._paper_pending = {
                "ct":          contract_type,
                "entry_price": current_price,
                "ticks_left":  duration,
            }
            logger.info(
                f"{tag}[paper] {contract_type} @ {current_price:.5f} | paper #{total + 1}"
            )

    def _halt(self, reason: str):
        self.is_halted    = True
        self._halt_reason = reason
        self._halt_time   = time.time()
        self._paper_pending = None  # reset paper state on each new halt
        self._paper_wins    = 0
        self._paper_losses  = 0
        self._save_halt_state()

    def _halt_state_path(self) -> str:
        return f"halt_state_{self.symbol or 'default'}.json"

    def _save_halt_state(self):
        if not self._persist_halt:
            return
        try:
            with open(self._halt_state_path(), "w") as f:
                json.dump({
                    "symbol":      self.symbol,
                    "is_halted":   self.is_halted,
                    "halt_reason": self._halt_reason,
                    "halt_time":   self._halt_time,
                }, f)
        except Exception as e:
            logger.warning(f"[{self.symbol}] Could not save halt state: {e}")

    def _clear_halt_state(self):
        if not self._persist_halt:
            return
        path = self._halt_state_path()
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass

    def _load_halt_state(self):
        path = self._halt_state_path()
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                state = json.load(f)
            if state.get("symbol") != self.symbol:
                return
            reason    = state.get("halt_reason", "")
            halt_time = state.get("halt_time")
            if not (state.get("is_halted") and halt_time):
                return
            elapsed = time.time() - halt_time
            # Rolling WR halt: clear if 4h already passed
            if "rolling WR" in reason:
                if elapsed >= self.perf_halt_reset_hours * 3600:
                    logger.info(f"[{self.symbol}] Stale performance halt cleared on startup.")
                    self._clear_halt_state()
                    return
            # Daily loss halt: clear if it fired on a previous calendar day
            if "daily loss" in reason:
                halt_dt = datetime.fromtimestamp(halt_time)
                if halt_dt.date() < datetime.now().date():
                    logger.info(f"[{self.symbol}] Previous-day loss halt cleared on startup.")
                    self._clear_halt_state()
                    return
            self.is_halted    = True
            self._halt_reason = reason
            self._halt_time   = halt_time
            tag = f"[{self.symbol}] " if self.symbol else ""
            logger.warning(f"{tag}Restored halt from disk: {reason} (halted {elapsed/3600:.1f}h ago)")
        except Exception as e:
            logger.warning(f"[{self.symbol}] Could not load halt state: {e}")

    def _daily_loss_pct(self) -> float:
        loss = self.daily_start_balance - self.current_balance
        if self.daily_start_balance == 0:
            return 0.0
        return max(0.0, (loss / self.daily_start_balance) * 100)

    @property
    def rolling_win_rate(self) -> float:
        if not self._rolling:
            return 100.0
        return round(sum(self._rolling) / len(self._rolling) * 100, 1)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return round((self.wins / self.total_trades) * 100, 1)

    @property
    def net_profit(self) -> float:
        return round(self.current_balance - self.starting_balance, 2)

    @property
    def daily_pnl(self) -> float:
        return round(self.current_balance - self.daily_start_balance, 2)
