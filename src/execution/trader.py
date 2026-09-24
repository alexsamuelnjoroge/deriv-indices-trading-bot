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
        over_barrier: int = 4,
        under_barrier: int = 6,
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
        self.digit_barrier   = digit_barrier   # DIGITOVER/DIGITUNDER: digit threshold
        self.over_barrier    = over_barrier    # DIGIT_PAIR: OVER barrier (default 4)
        self.under_barrier   = under_barrier   # DIGIT_PAIR: UNDER barrier (default 6)
        self._open: dict[str, dict] = {}
        self._open_accu: dict[str, int] = {}  # contract_id -> ticks remaining
        self._strategy = strategy
        self._alerter  = alerter

        self.client.on_contract_update(self._on_contract_update)
        self.client.on_reconnect(self._on_reconnect)

    async def _on_reconnect(self) -> None:
        """Re-subscribe to proposal_open_contract for any contracts still open after a WS reconnect."""
        if not self._open:
            return
        for contract_id in list(self._open):
            try:
                await self.client._send(
                    {"proposal_open_contract": 1, "contract_id": int(contract_id), "subscribe": 1}
                )
                logger.info(f"[{self.symbol}] Re-subscribed to open contract {contract_id} after reconnect")
            except Exception as e:
                logger.warning(f"[{self.symbol}] Could not re-subscribe to contract {contract_id}: {e}")

    async def _tick_open_accus(self) -> None:
        """Decrement tick counters for open Accumulator contracts; sell when expired or profit target hit."""
        if not self._open_accu:
            return
        for cid in list(self._open_accu):
            self._open_accu[cid] -= 1
            ticks_left = self._open_accu[cid]

            sell = ticks_left < 0  # < 0 (not <= 0): Deriv applies growth on the sell tick too, so wait one extra tick

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
                        if "BetExpired" in str(e) or "ContractNotFound" in str(e):
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
                    if "BetExpired" in str(e) or "ContractNotFound" in str(e):
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

            elif signal.action == "BUY_DIGITUNDER":
                result = await self.client.buy_contract(
                    symbol=self.symbol,
                    contract_type="DIGITUNDER",
                    duration=1,
                    duration_unit="t",
                    stake=stake,
                    barrier=str(self.digit_barrier),
                )
                contract_id = str(result["contract_id"])
                self._open[contract_id] = {
                    "signal_action": "BUY_DIGITUNDER",
                    "contract_type": "DIGITUNDER",
                    "stake":         stake,
                    "buy_price":     float(result.get("buy_price", stake)),
                    "is_multiplier": False,
                }
                self.risk.on_contract_opened()
                logger.info(
                    f"[{self.symbol}] DIGITUNDER({self.digit_barrier}) | ID: {contract_id} | stake={stake:.2f}"
                )

            elif signal.action == "BUY_DIGIT_PAIR":
                # Both previous contracts must be fully settled before opening a new pair.
                if self.risk.open_contracts > 0:
                    return
                # Place OVER and UNDER atomically — both settle on the same next price tick.
                result_over = await self.client.buy_contract(
                    symbol=self.symbol,
                    contract_type="DIGITOVER",
                    duration=1,
                    duration_unit="t",
                    stake=stake,
                    barrier=str(self.over_barrier),
                )
                cid_over = str(result_over["contract_id"])
                self._open[cid_over] = {
                    "signal_action": "BUY_DIGIT_PAIR",
                    "contract_type": "DIGITOVER",
                    "stake":         stake,
                    "buy_price":     float(result_over.get("buy_price", stake)),
                    "is_multiplier": False,
                }
                self.risk.on_contract_opened()

                result_under = await self.client.buy_contract(
                    symbol=self.symbol,
                    contract_type="DIGITUNDER",
                    duration=1,
                    duration_unit="t",
                    stake=stake,
                    barrier=str(self.under_barrier),
                )
                cid_under = str(result_under["contract_id"])
                self._open[cid_under] = {
                    "signal_action": "BUY_DIGIT_PAIR",
                    "contract_type": "DIGITUNDER",
                    "stake":         stake,
                    "buy_price":     float(result_under.get("buy_price", stake)),
                    "is_multiplier": False,
                }
                self.risk.on_contract_opened()
                logger.info(
                    f"[{self.symbol}] DIGIT_PAIR | OVER({self.over_barrier})={cid_over} "
                    f"UNDER({self.under_barrier})={cid_under} | stake={stake:.2f}×2"
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

        meta          = self._open.pop(contract_id)
        ticks_remaining = self._open_accu.pop(contract_id, None)  # None = normal sell (already removed)
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
            if profit > 0:
                outcome = "WIN"
                tick_info = ""
            else:
                ticks_held = (self.hold_ticks - ticks_remaining) if ticks_remaining is not None else "?"
                outcome = f"LOSS (knockout @t{ticks_held}/{self.hold_ticks})"
                tick_info = f" ticks_survived={ticks_held}"
            logger.warning(
                f"[{self.symbol}] ACCU {contract_id} {outcome} | "
                f"stake={buy_price:.2f} profit={profit:+.2f}{tick_info} | "
                f"WR={self.risk.win_rate:.1f}% ({self.risk.total_trades}t) | "
                f"bal={self.risk.current_balance:.2f}"
            )
        else:
            outcome = "WIN" if profit > 0 else "LOSS"
            extra = ""
            if meta["contract_type"] in ("DIGITOVER", "DIGITUNDER"):
                entry_disp = contract.get("entry_tick_display_value", "?")
                exit_disp  = contract.get("exit_tick_display_value", "?")
                if exit_disp and exit_disp != "?":
                    try:
                        exit_digit = int(str(round(float(exit_disp), 4)).replace(".", "")[-1])
                        extra = f" | entry={entry_disp} exit={exit_disp} digit={exit_digit}"
                    except Exception:
                        extra = f" | entry={entry_disp} exit={exit_disp}"
                else:
                    extra = f" | contract_keys={list(contract.keys())[:12]}"
            logger.warning(
                f"[{self.symbol}] {meta['contract_type']} {outcome} | "
                f"stake={buy_price:.2f} profit={profit:+.2f}{extra} | "
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
