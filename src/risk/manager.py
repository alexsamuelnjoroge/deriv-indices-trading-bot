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

from collections import deque
from dataclasses import dataclass
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
    def __init__(self, config: dict, starting_balance: float):
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

        self.starting_balance: float    = starting_balance
        self.current_balance: float     = starting_balance
        self.daily_start_balance: float = starting_balance

        self.open_contracts: int       = 0
        self.trade_history: list[TradeResult] = []
        self._rolling: deque           = deque(maxlen=self.rolling_window)
        self.total_trades: int         = 0
        self.wins: int                 = 0
        self.losses: int               = 0
        self.is_halted: bool           = False
        self._halt_reason: str         = ""

    # ------------------------------------------------------------------ #
    #  Pre-trade checks
    # ------------------------------------------------------------------ #

    def can_trade(self) -> tuple[bool, str]:
        if self.is_halted:
            return False, f"Bot halted: {self._halt_reason}"

        if self.open_contracts >= self.max_open_contracts:
            return False, f"Max open contracts reached ({self.open_contracts}/{self.max_open_contracts})"

        daily_loss_pct = self._daily_loss_pct()
        if daily_loss_pct >= self.daily_loss_limit:
            self._halt("daily loss limit reached")
            logger.warning(
                f"CIRCUIT BREAKER: Daily loss {daily_loss_pct:.1f}% >= limit {self.daily_loss_limit}%."
            )
            return False, "Circuit breaker triggered"

        if len(self._rolling) >= self.rolling_window:
            rwr = self.rolling_win_rate
            if rwr < self.min_rolling_wr:
                self._halt(f"rolling WR {rwr:.1f}% < {self.min_rolling_wr}% over last {self.rolling_window} trades")
                logger.warning(
                    f"PERFORMANCE HALT: Rolling WR {rwr:.1f}% < {self.min_rolling_wr}%. "
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

        if won:
            self.wins += 1
            logger.info(f"WIN  | Profit: +{result.profit:.2f} | Balance: {self.current_balance:.2f}")
        else:
            self.losses += 1
            logger.info(f"LOSS | Loss:   -{result.stake:.2f} | Balance: {self.current_balance:.2f}")

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
            logger.info("Daily reset: circuit breaker cleared for new trading day.")
        logger.info(f"Daily reset | New baseline: {self.daily_start_balance:.2f}")

    # ------------------------------------------------------------------ #
    #  Stats
    # ------------------------------------------------------------------ #

    def _halt(self, reason: str):
        self.is_halted    = True
        self._halt_reason = reason

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
