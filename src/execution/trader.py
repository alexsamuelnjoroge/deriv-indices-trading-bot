"""
Trader — bridges strategy signals → API calls → risk manager updates.

Flow:
  signal arrives → check risk → calculate stake → buy contract via API
  contract closes (API callback) → update risk manager → log result
"""

import asyncio
from loguru import logger

from src.api.client import DerivClient
from src.risk.manager import RiskManager, TradeResult
from src.strategies.base import Signal


CONTRACT_TYPE = {
    "BUY_RISE": "CALL",
    "BUY_FALL": "PUT",
}


class Trader:
    def __init__(
        self,
        client: DerivClient,
        risk: RiskManager,
        symbol: str,
        duration: int,
        duration_unit: str,
    ):
        self.client = client
        self.risk = risk
        self.symbol = symbol
        self.duration = duration
        self.duration_unit = duration_unit
        self._open: dict[str, dict] = {}  # contract_id → metadata

        self.client.on_contract_update(self._on_contract_update)

    async def execute(self, signal: Signal):
        if signal.action == "HOLD":
            return

        ok, reason = self.risk.can_trade()
        if not ok:
            logger.debug(f"Trade blocked: {reason}")
            return

        contract_type = CONTRACT_TYPE[signal.action]
        stake = self.risk.calculate_stake(atr=signal.atr, atr_baseline=signal.atr_baseline)

        try:
            result = await self.client.buy_contract(
                symbol=self.symbol,
                contract_type=contract_type,
                duration=self.duration,
                duration_unit=self.duration_unit,
                stake=stake,
            )
            contract_id = str(result["contract_id"])
            self._open[contract_id] = {
                "contract_type": contract_type,
                "stake": stake,
                "buy_price": float(result.get("buy_price", stake)),
            }
            self.risk.on_contract_opened()
            logger.info(f"Opened {contract_type} | ID: {contract_id} | Stake: {stake} | Reason: {signal.reason}")

        except Exception as e:
            logger.error(f"Failed to open contract: {e}")

    async def _on_contract_update(self, contract: dict):
        contract_id = str(contract.get("contract_id", ""))
        status = contract.get("status", "")

        if contract_id not in self._open:
            return

        meta = self._open.pop(contract_id)
        buy_price = meta["stake"]
        sell_price = float(contract.get("sell_price", 0))
        profit = sell_price - buy_price

        trade = TradeResult(
            contract_id=contract_id,
            contract_type=meta["contract_type"],
            stake=buy_price,
            payout=sell_price,
            profit=profit,
            status="won" if profit > 0 else "lost",
        )

        # Sync balance from API for accuracy
        try:
            balance = await self.client.get_balance()
            self.risk.update_balance(balance)
        except Exception:
            pass

        self.risk.on_contract_closed(trade)
