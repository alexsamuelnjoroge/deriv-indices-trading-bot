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
        strategy=None,
        alerter=None,
    ):
        self.client        = client
        self.risk          = risk
        self.symbol        = symbol
        self.duration      = duration
        self.duration_unit = duration_unit
        self.multiplier    = multiplier   # 0 = binary options mode
        self._open: dict[str, dict] = {}
        self._strategy = strategy
        self._alerter  = alerter

        self.client.on_contract_update(self._on_contract_update)

    async def execute(self, signal: Signal):
        if signal.action == "HOLD":
            return

        ok, reason = self.risk.can_trade()
        if not ok:
            logger.debug(f"Trade blocked: {reason}")
            if "halted" in reason.lower() or "circuit" in reason.lower() or "performance" in reason.lower():
                if self._alerter:
                    asyncio.create_task(
                        self._alerter.send_halt(self.symbol, reason, self.risk.current_balance)
                    )
            return

        stake        = self.risk.calculate_stake(atr=signal.atr, atr_baseline=signal.atr_baseline)
        is_multi     = signal.sl_pct is not None and self.multiplier > 0

        try:
            if is_multi:
                contract_type = _MULTI_TYPE[signal.action]
                sl_amount     = round(stake * self.multiplier * signal.sl_pct, 2)
                tp_amount     = round(stake * self.multiplier * signal.tp_pct, 2)

                result = await self.client.buy_multiplier(
                    symbol=self.symbol,
                    contract_type=contract_type,
                    stake=stake,
                    multiplier=self.multiplier,
                    sl_amount=sl_amount,
                    tp_amount=tp_amount,
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
                "is_multiplier": is_multi,
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
