"""
Deriv WebSocket API client.
Handles connection, authentication, tick subscriptions, and contract operations.
"""

import asyncio
import json
import os
from typing import Callable, Optional

import websockets
from loguru import logger


class DerivClient:
    WS_URL = "wss://ws.derivws.com/websockets/v3?app_id={app_id}"

    def __init__(self, api_token: str, app_id: str = "1089"):
        self.api_token = api_token
        self.app_id = app_id
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._req_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._tick_callbacks: list[Callable] = []
        self._contract_callbacks: list[Callable] = []
        self._running = False
        self.account_info: dict = {}
        self._subscribed_symbols: list[str] = []

    # ------------------------------------------------------------------ #
    #  Connection
    # ------------------------------------------------------------------ #

    async def connect(self):
        url = self.WS_URL.format(app_id=self.app_id)
        self.ws = await websockets.connect(url)
        self._running = True
        logger.info("Connected to Deriv WebSocket")

        asyncio.create_task(self._listen())
        await self._authorize()

    async def disconnect(self):
        self._running = False
        if self.ws:
            await self.ws.close()
            logger.info("Disconnected from Deriv WebSocket")

    # ------------------------------------------------------------------ #
    #  Internal message loop
    # ------------------------------------------------------------------ #

    async def _listen(self):
        try:
            async for raw in self.ws:
                msg = json.loads(raw)
                msg_type = msg.get("msg_type")
                req_id = msg.get("req_id")

                if msg.get("error"):
                    code = msg["error"].get("code", "")
                    message = msg["error"].get("message", "")
                    logger.error(f"API error [{code}]: {message}")
                    if req_id and req_id in self._pending:
                        self._pending.pop(req_id).set_exception(
                            RuntimeError(f"{code}: {message}")
                        )
                    continue

                if req_id and req_id in self._pending:
                    self._pending.pop(req_id).set_result(msg)

                if msg_type == "tick":
                    for cb in self._tick_callbacks:
                        asyncio.create_task(cb(msg["tick"]))

                elif msg_type == "proposal_open_contract":
                    contract = msg.get("proposal_open_contract", {})
                    if contract.get("status") in ("sold", "won", "lost"):
                        for cb in self._contract_callbacks:
                            asyncio.create_task(cb(contract))

        except websockets.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            if self._running:
                asyncio.create_task(self._reconnect())

    # ------------------------------------------------------------------ #
    #  Reconnect
    # ------------------------------------------------------------------ #

    async def _reconnect(self):
        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        self._pending.clear()

        delay = 2
        for attempt in range(1, 6):
            logger.warning(f"Reconnecting... attempt {attempt}/5 (in {delay}s)")
            await asyncio.sleep(delay)
            try:
                url = self.WS_URL.format(app_id=self.app_id)
                self.ws = await websockets.connect(url)
                asyncio.create_task(self._listen())
                await self._authorize()
                for symbol in self._subscribed_symbols:
                    await self._send({"ticks": symbol, "subscribe": 1})
                    logger.info(f"Re-subscribed to ticks: {symbol}")
                logger.info("Reconnected successfully")
                return
            except Exception as e:
                logger.warning(f"Reconnect attempt {attempt} failed: {e}")
                delay = min(delay * 2, 60)

        logger.error("All reconnect attempts failed. Bot stopping.")
        self._running = False

    # ------------------------------------------------------------------ #
    #  Request helper
    # ------------------------------------------------------------------ #

    async def _send(self, payload: dict) -> dict:
        self._req_id += 1
        payload["req_id"] = self._req_id
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[self._req_id] = future
        await self.ws.send(json.dumps(payload))
        return await asyncio.wait_for(future, timeout=15)

    # ------------------------------------------------------------------ #
    #  Auth
    # ------------------------------------------------------------------ #

    async def _authorize(self):
        resp = await self._send({"authorize": self.api_token})
        self.account_info = resp.get("authorize", {})
        balance = self.account_info.get("balance", "?")
        currency = self.account_info.get("currency", "")
        loginid = self.account_info.get("loginid", "?")
        logger.info(f"Authorized as {loginid} | Balance: {balance} {currency}")

    # ------------------------------------------------------------------ #
    #  Balance
    # ------------------------------------------------------------------ #

    async def get_balance(self) -> float:
        resp = await self._send({"balance": 1, "account": "current"})
        return float(resp["balance"]["balance"])

    # ------------------------------------------------------------------ #
    #  Tick subscription
    # ------------------------------------------------------------------ #

    def on_tick(self, callback: Callable):
        self._tick_callbacks.append(callback)

    def on_contract_update(self, callback: Callable):
        self._contract_callbacks.append(callback)

    async def subscribe_ticks(self, symbol: str):
        if symbol not in self._subscribed_symbols:
            self._subscribed_symbols.append(symbol)
        await self._send({"ticks": symbol, "subscribe": 1})
        logger.info(f"Subscribed to ticks: {symbol}")

    # ------------------------------------------------------------------ #
    #  Contracts
    # ------------------------------------------------------------------ #

    async def buy_contract(
        self,
        symbol: str,
        contract_type: str,   # "CALL" = Rise, "PUT" = Fall
        duration: int,
        duration_unit: str,   # "t" = ticks, "m" = minutes
        stake: float,
        currency: str = "USD",
    ) -> dict:
        proposal_resp = await self._send({
            "proposal": 1,
            "amount": round(stake, 2),
            "basis": "stake",
            "contract_type": contract_type,
            "currency": currency,
            "duration": duration,
            "duration_unit": duration_unit,
            "symbol": symbol,
        })

        proposal_id = proposal_resp["proposal"]["id"]
        payout = proposal_resp["proposal"]["payout"]
        logger.debug(f"Proposal {proposal_id} | Stake: {stake} | Payout: {payout}")

        buy_resp = await self._send({"buy": proposal_id, "price": stake})
        contract_id = buy_resp["buy"]["contract_id"]
        logger.info(f"Contract bought: {contract_id} | {contract_type} | Stake: {stake}")

        await self._send({"proposal_open_contract": 1, "contract_id": contract_id, "subscribe": 1})
        return buy_resp["buy"]
