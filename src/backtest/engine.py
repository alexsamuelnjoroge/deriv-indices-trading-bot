"""
Backtest engine — replays historical ticks through the live strategy stack.

Uses the real TickStore, RSIReversalStrategy, and RiskManager so the backtest
behaviour is identical to the live bot. The only simulation is contract settlement:
entry is at the tick price when the signal fires, exit is at the price
`contract_duration` ticks later.

Payout model (binary tick contracts):
  Win  →  +stake × payout_pct   (e.g. 87% of stake as profit)
  Loss →  -stake
  Breakeven win rate = 1 / (1 + payout_pct)  ≈ 53.5% at 87%
"""

from dataclasses import dataclass, field
from typing import Optional

from src.data.tick_store import TickStore
from src.strategies.rsi_reversal import RSIReversalStrategy
from src.risk.manager import RiskManager, TradeResult


CONTRACT_TYPE = {"BUY_RISE": "CALL", "BUY_FALL": "PUT"}


@dataclass
class BacktestTrade:
    index: int            # tick index when signal fired
    contract_type: str    # CALL or PUT
    entry_price: float
    exit_price: float
    stake: float
    profit: float
    status: str           # "won" or "lost"
    reason: str           # signal reason for inspection


@dataclass
class BacktestResult:
    symbol: str
    ticks_total: int
    starting_balance: float
    payout_pct: float
    trades: list[BacktestTrade] = field(default_factory=list)
    final_balance: float = 0.0

    # ── Derived stats ──────────────────────────────────────────────
    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.status == "won")

    @property
    def losses(self) -> int:
        return sum(1 for t in self.trades if t.status == "lost")

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return round(self.wins / self.total_trades * 100, 1)

    @property
    def net_profit(self) -> float:
        return round(self.final_balance - self.starting_balance, 2)

    @property
    def breakeven_win_rate(self) -> float:
        """Minimum win rate needed to break even given the payout percentage."""
        return round(100 / (1 + self.payout_pct), 1)

    @property
    def best_win_streak(self) -> int:
        return _max_streak(self.trades, "won")

    @property
    def worst_loss_streak(self) -> int:
        return _max_streak(self.trades, "lost")

    @property
    def avg_stake(self) -> float:
        if not self.trades:
            return 0.0
        return round(sum(t.stake for t in self.trades) / self.total_trades, 2)


def _max_streak(trades: list[BacktestTrade], status: str) -> int:
    best = cur = 0
    for t in trades:
        if t.status == status:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


class BacktestEngine:
    def __init__(
        self,
        strategy_cfg: dict,
        risk_cfg: dict,
        payout_pct: float = 0.87,
    ):
        self.strategy_cfg = strategy_cfg
        self.risk_cfg = risk_cfg
        self.payout_pct = payout_pct
        self.duration: int = strategy_cfg.get("contract_duration", 5)

    def run(
        self,
        ticks: list[dict],
        starting_balance: float = 1000.0,
    ) -> BacktestResult:
        symbol = ticks[0]["symbol"] if ticks else "unknown"
        result = BacktestResult(
            symbol=symbol,
            ticks_total=len(ticks),
            starting_balance=starting_balance,
            payout_pct=self.payout_pct,
        )

        tick_store = TickStore(rsi_period=self.strategy_cfg.get("rsi_period", 14))
        strategy = RSIReversalStrategy(self.strategy_cfg)
        risk = RiskManager(self.risk_cfg, starting_balance=starting_balance)

        # pending contract: set when a signal fires, settled `duration` ticks later
        pending: Optional[dict] = None

        for i, raw in enumerate(ticks):
            tick_store.add(raw)
            current_price = float(raw["quote"])

            # ── Settle pending contract ───────────────────────────
            if pending is not None and i >= pending["exit_idx"]:
                ct = pending["contract_type"]
                entry_price = pending["entry_price"]
                stake = pending["stake"]

                won = (
                    (ct == "CALL" and current_price > entry_price) or
                    (ct == "PUT"  and current_price < entry_price)
                )
                profit = round(stake * self.payout_pct if won else -stake, 2)
                status = "won" if won else "lost"

                bt_trade = BacktestTrade(
                    index=pending["entry_idx"],
                    contract_type=ct,
                    entry_price=entry_price,
                    exit_price=current_price,
                    stake=stake,
                    profit=profit,
                    status=status,
                    reason=pending["reason"],
                )
                result.trades.append(bt_trade)

                risk.on_contract_closed(TradeResult(
                    contract_id=str(pending["entry_idx"]),
                    contract_type=ct,
                    stake=stake,
                    payout=stake + profit if won else 0.0,
                    profit=profit,
                    status=status,
                ))
                pending = None

            # ── Evaluate strategy for new signal ─────────────────
            if pending is None and not risk.is_halted:
                signal = strategy.evaluate(tick_store)

                if signal.action != "HOLD":
                    ok, _ = risk.can_trade()
                    if ok:
                        stake = risk.calculate_stake(
                            atr=signal.atr,
                            atr_baseline=signal.atr_baseline,
                        )
                        pending = {
                            "entry_idx":    i,
                            "exit_idx":     i + self.duration,
                            "entry_price":  current_price,
                            "contract_type": CONTRACT_TYPE[signal.action],
                            "stake":        stake,
                            "reason":       signal.reason,
                        }
                        risk.on_contract_opened()

        result.final_balance = risk.current_balance
        return result
