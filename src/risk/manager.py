"""
Risk Manager — protects the account balance.

Rules enforced:
  1. Stake = stake_percent % of current balance (never more than max_stake)
  2. ATR-adaptive sizing: when current volatility > baseline, reduce stake
     proportionally (high vol → smaller bet; low vol → full bet)
  3. Daily loss circuit breaker: stops the bot when daily loss > daily_loss_limit %
  4. Tracks open contract count; blocks new trades if limit reached
"""

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
        self.stake_percent: float = config.get("stake_percent", 1.0)
        self.max_stake: float = config.get("max_stake", 5.0)
        self.daily_loss_limit: float = config.get("daily_loss_limit", 10.0)
        self.max_open_contracts: int = config.get("max_open_contracts", 1)
        self.use_atr_stake: bool = config.get("use_atr_stake", True)

        self.starting_balance: float = starting_balance
        self.current_balance: float = starting_balance
        self.daily_start_balance: float = starting_balance

        self.open_contracts: int = 0
        self.trade_history: list[TradeResult] = []
        self.total_trades: int = 0
        self.wins: int = 0
        self.losses: int = 0
        self.is_halted: bool = False

    # ------------------------------------------------------------------ #
    #  Pre-trade checks
    # ------------------------------------------------------------------ #

    def can_trade(self) -> tuple[bool, str]:
        if self.is_halted:
            return False, "Bot halted: daily loss limit reached"

        if self.open_contracts >= self.max_open_contracts:
            return False, f"Max open contracts reached ({self.open_contracts}/{self.max_open_contracts})"

        daily_loss_pct = self._daily_loss_pct()
        if daily_loss_pct >= self.daily_loss_limit:
            self.is_halted = True
            logger.warning(
                f"CIRCUIT BREAKER: Daily loss {daily_loss_pct:.1f}% >= limit {self.daily_loss_limit}%. Bot stopped."
            )
            return False, "Circuit breaker triggered"

        return True, "ok"

    def calculate_stake(
        self,
        atr: Optional[float] = None,
        atr_baseline: Optional[float] = None,
    ) -> float:
        """
        Base stake = stake_percent% of balance, capped at max_stake.
        When use_atr_stake is enabled, scales inversely with current volatility:
          - current ATR == baseline  → full stake (×1.0)
          - current ATR == 2× baseline → half stake (×0.5)
          - current ATR == 0.5× baseline → +50% stake (×1.5, capped)
        Multiplier is clamped to [0.5, 1.5] so a volatility spike never wipes stake.
        """
        stake = self.current_balance * (self.stake_percent / 100)

        if self.use_atr_stake and atr and atr_baseline and atr_baseline > 0:
            vol_ratio = atr / atr_baseline          # >1 = more volatile than normal
            vol_factor = 1.0 / vol_ratio            # inverse: high vol → smaller stake
            vol_factor = max(0.5, min(1.5, vol_factor))
            stake *= vol_factor
            logger.debug(
                f"ATR stake adjustment: ATR={atr:.6f} baseline={atr_baseline:.6f} "
                f"ratio={vol_ratio:.2f} factor={vol_factor:.2f}"
            )

        stake = min(stake, self.max_stake)
        stake = max(stake, 0.35)  # Deriv minimum stake
        return round(stake, 2)

    # ------------------------------------------------------------------ #
    #  Contract lifecycle
    # ------------------------------------------------------------------ #

    def on_contract_opened(self):
        self.open_contracts += 1

    def on_contract_closed(self, result: TradeResult):
        self.open_contracts = max(0, self.open_contracts - 1)
        self.current_balance += result.profit
        self.total_trades += 1
        self.trade_history.append(result)

        if result.status == "won":
            self.wins += 1
            logger.info(f"WIN  | Profit: +{result.profit:.2f} | Balance: {self.current_balance:.2f}")
        else:
            self.losses += 1
            logger.info(f"LOSS | Loss:   -{result.stake:.2f} | Balance: {self.current_balance:.2f}")

    def update_balance(self, balance: float):
        self.current_balance = balance

    # ------------------------------------------------------------------ #
    #  Stats
    # ------------------------------------------------------------------ #

    def _daily_loss_pct(self) -> float:
        loss = self.daily_start_balance - self.current_balance
        if self.daily_start_balance == 0:
            return 0.0
        return max(0.0, (loss / self.daily_start_balance) * 100)

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
