"""
Trader — bridges strategy signals → API calls → risk manager updates.

Flow:
  signal arrives → check risk → calculate stake → buy contract via API
  contract closes (API callback) → update risk manager → log result → alert

Supports both binary options (CALL/PUT) and multiplier contracts (MULTUP/MULTDOWN).
Multiplier signals carry sl_pct and tp_pct; binary signals do not.
"""

import asyncio
from loguru import logger

from src.api.client import DerivClient
from src.risk.manager import RiskManager, TradeResult
from src.strategies.base import Signal


_BINARY_TYPE = {
    "BUY_RISE": "CALL",
    "BUY_FALL": "PUT",
}
_MULTI_TYPE = {
    "BUY_RISE": "MULTUP",
    "BUY_FALL": "MULTDOWN",
}

# Keep for backward-compat with any code that imports CONTRACT_TYPE
CONTRACT_TYPE = _BINARY_TYPE


class Trader:
    def __init__(
        self,
        client: DerivClient,
        risk: RiskManager,
        symbol: str,
        duration: int,
        duration_unit: str,
        multiplier: int = 0,
        growth_rate: float = 0.03,
        hold_ticks: int = 5,
        early_sell_pct: float = 0.0,
        digit_barrier: int = 4,
        strategy=None,
        alerter=None,
    ):
        self.client          = client
        self.risk            = risk
        self.symbol          = symbol
        self.duration        = duration
        self.duration_unit   = duration_unit
        self.multiplier      = multiplier    # 0 = binary options mode
        self.growth_rate     = growth_rate   # ACCU: fraction growth per tick (e.g. 0.03)
        self.hold_ticks      = hold_ticks    # ACCU: ticks to hold before auto-sell
        self.early_sell_pct  = early_sell_pct  # ACCU: sell early when profit fraction >= this (0 = disabled)
        self.digit_barrier   = digit_barrier   # DIGITOVER: win if last digit > this value
        self._open: dict[str, dict] = {}
        self._open_accu: dict[str, int] = {}  # contract_id -> ticks remaining
        self._strategy = strategy
        self._alerter  = alerter

        self.client.on_contract_update(self._on_contract_update)

    async def _tick_open_accus(self) -> None:
        """Decrement tick counters for open Accumulator contracts; sell when expired or profit target hit."""
        if not self._open_accu:
            return
        for cid in list(self._open_accu):
            self._open_accu[cid] -= 1
            ticks_left = self._open_accu[cid]

            sell = ticks_left <= 0

            if not sell and self.early_sell_pct > 0 and cid in self._open:
                elapsed = self.hold_ticks - ticks_left
                profit_pct = (1 + self.growth_rate) ** elapsed - 1
                if profit_pct >= self.early_sell_pct:
                    sell = True
                    logger.warning(
                        f"[{self.symbol}] ACCU {cid} early-sell at tick "
                        f"{elapsed}/{self.hold_ticks}: "
                        f"profit={profit_pct:.1%} >= threshold={self.early_sell_pct:.1%}"
                    )

            if sell:
                del self._open_accu[cid]
                if cid in self._open:   # still open (not knocked out by barrier)
                    try:
                        await self.client.sell_contract(cid)
                    except Exception as e:
                        if "BetExpired" in str(e):
                            logger.info(f"[{self.symbol}] ACCU {cid} already settled by Deriv before sell — normal race condition")
                        else:
                            logger.error(f"[{self.symbol}] Failed to close ACCU {cid}: {e}")

    async def _close_all_accus(self, reason: str) -> None:
        """Sell all open ACCU contracts immediately (spike guard or forced exit)."""
        for cid in list(self._open_accu):
            del self._open_accu[cid]
            if cid in self._open:
                try:
                    await self.client.sell_contract(cid)
                    logger.warning(f"[{self.symbol}] ACCU {cid} sold ({reason})")
                except Exception as e:
                    if "BetExpired" in str(e):
                        logger.info(f"[{self.symbol}] ACCU {cid} already settled by Deriv ({reason}) — normal race condition")
                    else:
                        logger.error(f"[{self.symbol}] Failed to sell ACCU {cid} ({reason}): {e}")

    async def execute(self, signal: Signal):
        # Always tick ACCU counters so hold-period closes happen on time
        await self._tick_open_accus()

        # Spike detected on this tick: attempt to sell open ACCUs before barrier KO.
        # Won't always beat the server-side KO, but protects against partial spikes
        # and secondary spikes that approach but don't immediately breach the barrier.
        if signal.close_open_accus and self._open_accu:
            await self._close_all_accus("spike guard")

        if signal.action == "HOLD":
            return

        ok, reason = self.risk.can_trade()
        if not ok:
            logger.info(f"[{self.symbol}] Trade blocked: {reason}")
            if "halted" in reason.lower() or "circuit" in reason.lower() or "performance" in reason.lower():
                if self._alerter:
                    asyncio.create_task(
                        self._alerter.send_halt(self.symbol, reason, self.risk.current_balance)
                    )
            return

        logger.info(f"[{self.symbol}] Signal: {signal.action} | {signal.reason}")
        stake    = self.risk.calculate_stake(atr=signal.atr, atr_baseline=signal.atr_baseline)
        is_multi = signal.sl_pct is not None and self.multiplier > 0

        try:
            if signal.action == "BUY_ACCU":
                result = await self.client.buy_accumulator(
                    symbol=self.symbol,
                    growth_rate=self.growth_rate,
                    stake=stake,
                )
                contract_id = str(result["contract_id"])
                self._open[contract_id] = {
                    "signal_action": "BUY_ACCU",
                    "contract_type": "ACCU",
                    "stake":         stake,
                    "buy_price":     float(result.get("buy_price", stake)),
                    "is_multiplier": False,
                    "is_accumulator": True,
                }
                self._open_accu[contract_id] = self.hold_ticks
                self.risk.on_contract_opened()
                logger.warning(
                    f"[{self.symbol}] ACCU opened | ID: {contract_id} | "
                    f"stake={stake:.2f} | hold={self.hold_ticks}t | {signal.reason}"
                )

            elif is_multi:
                contract_type = _MULTI_TYPE[signal.action]
                sl_amount     = max(round(stake * self.multiplier * signal.sl_pct, 2), 0.10)
                tp_amount     = max(round(stake * self.multiplier * signal.tp_pct, 2), 0.10)
                result = await self.client.buy_multiplier(
                    symbol=self.symbol,
                    contract_type=contract_type,
                    stake=stake,
                    multiplier=self.multiplier,
                    sl_amount=sl_amount,
                    tp_amount=tp_amount,
                )
                contract_id = str(result["contract_id"])
                self._open[contract_id] = {
                    "signal_action": signal.action,
                    "contract_type": contract_type,
                    "stake":         stake,
                    "buy_price":     float(result.get("buy_price", stake)),
                    "is_multiplier": True,
                }
                self.risk.on_contract_opened()
                logger.info(
                    f"Opened {contract_type} | ID: {contract_id} | Stake: {stake} | Reason: {signal.reason}"
                )

            elif signal.action == "BUY_EVEN":
                result = await self.client.buy_contract(
                    symbol=self.symbol,
                    contract_type="DIGITEVEN",
                    duration=1,
                    duration_unit="t",
                    stake=stake,
                )
                contract_id = str(result["contract_id"])
                self._open[contract_id] = {
                    "signal_action": "BUY_EVEN",
                    "contract_type": "DIGITEVEN",
                    "stake":         stake,
                    "buy_price":     float(result.get("buy_price", stake)),
                    "is_multiplier": False,
                }
                self.risk.on_contract_opened()
                logger.info(
                    f"[{self.symbol}] DIGITEVEN | ID: {contract_id} | stake={stake:.2f}"
                )

            elif signal.action == "BUY_DIGITOVER":
                result = await self.client.buy_contract(
                    symbol=self.symbol,
                    contract_type="DIGITOVER",
                    duration=1,
                    duration_unit="t",
                    stake=stake,
                    barrier=str(self.digit_barrier),
                )
                contract_id = str(result["contract_id"])
                self._open[contract_id] = {
                    "signal_action": "BUY_DIGITOVER",
                    "contract_type": "DIGITOVER",
                    "stake":         stake,
                    "buy_price":     float(result.get("buy_price", stake)),
                    "is_multiplier": False,
                }
                self.risk.on_contract_opened()
                logger.info(
                    f"[{self.symbol}] DIGITOVER({self.digit_barrier}) | ID: {contract_id} | stake={stake:.2f}"
                )

            else:
                contract_type = _BINARY_TYPE[signal.action]
                duration      = signal.contract_duration if signal.contract_duration is not None else self.duration
                result = await self.client.buy_contract(
                    symbol=self.symbol,
                    contract_type=contract_type,
                    duration=duration,
                    duration_unit=self.duration_unit,
                    stake=stake,
                )
                contract_id = str(result["contract_id"])
                self._open[contract_id] = {
                    "signal_action": signal.action,
                    "contract_type": contract_type,
                    "stake":         stake,
                    "buy_price":     float(result.get("buy_price", stake)),
                    "is_multiplier": False,
                }
                self.risk.on_contract_opened()
                logger.info(
                    f"Opened {contract_type} | ID: {contract_id} | Stake: {stake} | Reason: {signal.reason}"
                )

        except Exception as e:
            logger.error(f"Failed to open contract [{type(e).__name__}]: {e}")

    async def _on_contract_update(self, contract: dict):
        contract_id = str(contract.get("contract_id", ""))
        if contract_id not in self._open:
            return

        meta      = self._open.pop(contract_id)
        self._open_accu.pop(contract_id, None)   # clean up if barrier knocked it out
        buy_price = meta["stake"]

        # Multiplier contracts carry a direct 'profit' field; binary use sell_price - buy_price
        if meta.get("is_multiplier") and "profit" in contract:
            profit    = float(contract["profit"])
            sell_price = buy_price + profit
        else:
            sell_price = float(contract.get("sell_price", 0))
            profit     = sell_price - buy_price

        trade = TradeResult(
            contract_id=contract_id,
            contract_type=meta["contract_type"],
            stake=buy_price,
            payout=sell_price,
            profit=profit,
            status="won" if profit > 0 else "lost",
        )

        self.risk.on_contract_closed(trade)

        if meta.get("is_accumulator"):
            outcome = "WIN" if profit > 0 else "LOSS (knockout)"
            logger.warning(
                f"[{self.symbol}] ACCU {contract_id} {outcome} | "
                f"stake={buy_price:.2f} profit={profit:+.2f} | "
                f"WR={self.risk.win_rate:.1f}% ({self.risk.total_trades}t) | "
                f"bal={self.risk.current_balance:.2f}"
            )
        else:
            outcome = "WIN" if profit > 0 else "LOSS"
            logger.warning(
                f"[{self.symbol}] {meta['contract_type']} {outcome} | "
                f"stake={buy_price:.2f} profit={profit:+.2f} | "
                f"WR={self.risk.win_rate:.1f}% ({self.risk.total_trades}t) | "
                f"bal={self.risk.current_balance:.2f}"
            )

        if self._strategy and hasattr(self._strategy, "on_result"):
            self._strategy.on_result(profit > 0)

        if self._alerter:
            asyncio.create_task(
                self._alerter.send_trade(
                    symbol=self.symbol,
                    action=meta["signal_action"],
                    stake=buy_price,
                    profit=profit,
                    balance=self.risk.current_balance,
                    win_rate=self.risk.win_rate,
                    total_trades=self.risk.total_trades,
                )
            )
