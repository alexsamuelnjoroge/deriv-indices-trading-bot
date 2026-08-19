"""
Deriv WebSocket API client.
Handles connection, authentication, tick subscriptions, and contract operations.

Supports two auth modes detected automatically from the token format:
  - Legacy (old app.deriv.com tokens): authorize via WebSocket message
  - PAT  (pat_... tokens from developers.deriv.com): OTP REST flow
"""

import asyncio
import json
import os
from typing import Callable, Optional

import aiohttp
import websockets
from loguru import logger


class DerivClient:
    LEGACY_WS_URL = "wss://ws.derivws.com/websockets/v3?app_id={app_id}"
    NEW_REST_BASE  = "https://api.derivws.com"

    def __init__(self, api_token: str, app_id: str = "1089"):
        self.api_token = api_token
        self.app_id = app_id
        self._use_new_api = api_token.startswith("pat_")
        self._ws_url: str = ""           # resolved at connect time
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._req_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._tick_callbacks: dict[str, list[Callable]] = {}  # symbol -> callbacks
        self._contract_callbacks: list[Callable] = []
        self._running = False
        self.account_info: dict = {}
        self._subscribed_symbols: list[str] = []

    # ------------------------------------------------------------------ #
    #  Connection
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    #  New-API helpers (PAT / OTP flow)
    # ------------------------------------------------------------------ #

    async def _resolve_ws_url(self) -> str:
        """For PAT tokens: POST to OTP endpoint → returns authenticated WebSocket URL."""
        app_id_header = os.getenv("DERIV_APP_KEY", "")
        if not app_id_header:
            raise RuntimeError(
                "DERIV_APP_KEY not set in .env — add your App ID from developers.deriv.com/dashboard"
            )
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "Deriv-App-ID": app_id_header,
        }

        _timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=_timeout) as session:
            # Step 1 — discover account_id
            async with session.get(
                f"{self.NEW_REST_BASE}/trading/v1/options/accounts",
                headers=headers,
            ) as resp:
                body = await resp.text()
                logger.info(f"Accounts API [{resp.status}]: {body[:500]}")
                if resp.status != 200:
                    raise RuntimeError(f"Accounts API returned {resp.status}: {body[:300]}")
                import json as _json
                accounts_raw = _json.loads(body)

                account_id = _extract_account_id(accounts_raw)
                if not account_id:
                    raise RuntimeError(
                        f"Could not find account_id in accounts response: {accounts_raw}"
                    )
                logger.info(f"Using account: {account_id}")

            # Step 2 — get OTP WebSocket URL
            async with session.post(
                f"{self.NEW_REST_BASE}/trading/v1/options/accounts/{account_id}/otp",
                headers=headers,
            ) as resp:
                body = await resp.text()
                logger.info(f"OTP API [{resp.status}]: {body[:500]}")
                if resp.status != 200:
                    raise RuntimeError(f"OTP API returned {resp.status}: {body[:300]}")
                import json as _json
                otp_raw = _json.loads(body)

                ws_url = _extract_ws_url(otp_raw)
                if not ws_url:
                    raise RuntimeError(
                        f"Could not find WebSocket URL in OTP response: {otp_raw}"
                    )
                return ws_url

    # ------------------------------------------------------------------ #
    #  Connection
    # ------------------------------------------------------------------ #

    async def connect(self):
        if self._use_new_api:
            logger.info("PAT token detected — using new OTP auth flow")
            self._ws_url = await self._resolve_ws_url()
            logger.info(f"Connecting to: {self._ws_url[:60]}...")
        else:
            self._ws_url = self.LEGACY_WS_URL.format(app_id=self.app_id)

        self.ws = await websockets.connect(
            self._ws_url,
            ping_interval=30,  # WS-level ping every 30s
            ping_timeout=10,   # close + reconnect if pong missing after 10s
        )
        self._running = True
        self._reconnecting = False
        logger.info("Connected to Deriv WebSocket")

        asyncio.create_task(self._listen())
        asyncio.create_task(self._heartbeat())

        if not self._use_new_api:
            await self._authorize()
        else:
            # New API: auth is embedded in the OTP URL — no authorize message needed.
            # Fetch account info via balance call so the rest of the bot has it.
            try:
                resp = await self._send({"balance": 1})
                logger.debug(f"Balance response: {resp}")
                bal = resp.get("balance", {})
                self.account_info = {
                    "balance":  bal.get("balance", 0),
                    "currency": bal.get("currency", "USD"),
                    "loginid":  bal.get("loginid", bal.get("login_id", "?")),
                }
                logger.info(
                    f"Authenticated | Account: {self.account_info['loginid']} | "
                    f"Balance: {self.account_info['balance']} {self.account_info['currency']}"
                )
            except Exception as e:
                logger.warning(f"Could not fetch initial balance: {e}")

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
                    details = msg["error"].get("details", msg["error"])
                    logger.error(f"API error [{code}]: {message} | details: {details}")
                    if req_id and req_id in self._pending:
                        self._pending.pop(req_id).set_exception(
                            RuntimeError(f"{code}: {message}")
                        )
                    continue

                if req_id and req_id in self._pending:
                    self._pending.pop(req_id).set_result(msg)

                # DEBUG: log first 5 unknown message types so we can map new API format
                if msg_type not in ("tick", "proposal_open_contract", "balance",
                                    "proposal", "buy", "authorize", "ping", None):
                    logger.info(f"[DEBUG] msg_type={msg_type!r} keys={list(msg.keys())[:8]}")

                if msg_type == "tick":
                    tick   = msg["tick"]
                    symbol = tick.get("symbol", "")
                    if not hasattr(self, "_tick_logged"):
                        self._tick_logged = True
                        logger.info(f"[DEBUG] First tick keys: {list(tick.keys())} | symbol={symbol!r} | data={tick}")
                    for cb in self._tick_callbacks.get(symbol, []):
                        asyncio.create_task(cb(tick))

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
    #  Heartbeat — keeps the Deriv session alive
    # ------------------------------------------------------------------ #

    async def _heartbeat(self):
        """Send a Deriv-level ping every 60 s to keep the session alive."""
        while self._running:
            await asyncio.sleep(60)
            if self.ws is None or self._reconnecting:
                continue
            try:
                await self._send({"ping": 1})
            except Exception:
                pass  # ConnectionClosed in _listen() will handle reconnect

    # ------------------------------------------------------------------ #
    #  Reconnect
    # ------------------------------------------------------------------ #

    async def _reconnect(self):
        if self._reconnecting:
            return
        self._reconnecting = True

        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        self._pending.clear()

        delay = 2
        attempt = 0
        while self._running:
            attempt += 1
            logger.warning(f"Reconnecting... attempt {attempt} (retry in {delay}s)")
            await asyncio.sleep(delay)
            try:
                if self._use_new_api:
                    # OTP tokens are single-use; need a fresh URL each reconnect
                    self._ws_url = await self._resolve_ws_url()
                self.ws = await websockets.connect(
                    self._ws_url,
                    open_timeout=15,
                    ping_interval=30,
                    ping_timeout=10,
                )
                asyncio.create_task(self._listen())
                if not self._use_new_api:
                    await self._authorize()
                for symbol in self._subscribed_symbols:
                    await self._send({"ticks": symbol, "subscribe": 1})
                    logger.info(f"Re-subscribed to ticks: {symbol}")
                logger.info(f"Reconnected successfully after {attempt} attempt(s)")
                self._reconnecting = False
                return
            except Exception as e:
                logger.warning(f"Reconnect attempt {attempt} failed: {e}")
                delay = min(delay * 2, 60)  # cap at 60s between retries

    # ------------------------------------------------------------------ #
    #  Request helper
    # ------------------------------------------------------------------ #

    async def _send(self, payload: dict) -> dict:
        self._req_id += 1
        payload["req_id"] = self._req_id
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[self._req_id] = future
        await self.ws.send(json.dumps(payload))
        return await asyncio.wait_for(future, timeout=45)

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
        resp = await self._send({"balance": 1})
        logger.debug(f"get_balance response: {resp}")
        return float(resp["balance"]["balance"])

    # ------------------------------------------------------------------ #
    #  Tick subscription
    # ------------------------------------------------------------------ #

    def on_tick(self, symbol: str, callback: Callable):
        self._tick_callbacks.setdefault(symbol, []).append(callback)

    def on_contract_update(self, callback: Callable):
        self._contract_callbacks.append(callback)

    async def subscribe_ticks(self, symbol: str):
        if symbol not in self._subscribed_symbols:
            self._subscribed_symbols.append(symbol)
        retry_delay = 60
        while True:
            try:
                resp = await self._send({"ticks": symbol, "subscribe": 1})
                logger.info(f"Subscribed to ticks: {symbol} | response keys: {list(resp.keys())}")
                return
            except RuntimeError as e:
                if "MarketIsClosed" in str(e):
                    logger.info(f"[{symbol}] Market closed — waiting {retry_delay}s before retry...")
                    await asyncio.sleep(retry_delay)
                else:
                    raise

    async def subscribe_ticks_multi(self, symbols: list[str]):
        for symbol in symbols:
            await self.subscribe_ticks(symbol)

    # ------------------------------------------------------------------ #
    #  Contracts
    # ------------------------------------------------------------------ #

    async def buy_multiplier(
        self,
        symbol: str,
        contract_type: str,   # "MULTUP" or "MULTDOWN"
        stake: float,
        multiplier: int,
        sl_amount: float,
        tp_amount: float,
        currency: str = "USD",
    ) -> dict:
        """Buy a Multiplier contract with fixed-dollar SL and TP."""
        proposal_resp = await self._send({
            "proposal": 1,
            "amount": round(stake, 2),
            "basis": "stake",
            "contract_type": contract_type,
            "currency": currency,
            "multiplier": multiplier,
            "underlying_symbol": symbol,
            "limit_order": {
                "stop_loss":   round(sl_amount, 2),
                "take_profit": round(tp_amount, 2),
            },
        })

        proposal_id = proposal_resp["proposal"]["id"]
        logger.debug(
            f"Multiplier proposal {proposal_id} | {contract_type} | "
            f"Stake: {stake} | x{multiplier} | SL: {sl_amount:.2f} TP: {tp_amount:.2f}"
        )

        buy_resp    = await self._send({"buy": proposal_id, "price": stake})
        contract_id = buy_resp["buy"]["contract_id"]
        logger.info(
            f"Multiplier bought: {contract_id} | {contract_type} | "
            f"Stake: {stake} x{multiplier} | SL ${sl_amount:.2f} / TP ${tp_amount:.2f}"
        )

        await self._send({"proposal_open_contract": 1, "contract_id": contract_id, "subscribe": 1})
        return buy_resp["buy"]

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
            "underlying_symbol": symbol,
        })

        proposal_id = proposal_resp["proposal"]["id"]
        payout = proposal_resp["proposal"]["payout"]
        logger.debug(f"Proposal {proposal_id} | Stake: {stake} | Payout: {payout}")

        buy_resp = await self._send({"buy": proposal_id, "price": stake})
        contract_id = buy_resp["buy"]["contract_id"]
        logger.info(f"Contract bought: {contract_id} | {contract_type} | Stake: {stake}")

        await self._send({"proposal_open_contract": 1, "contract_id": contract_id, "subscribe": 1})
        return buy_resp["buy"]

    async def buy_accumulator(
        self,
        symbol: str,
        growth_rate: float,  # e.g. 0.03 = 3% per tick
        stake: float,
        currency: str = "USD",
    ) -> dict:
        """Buy an Accumulator contract. Grows by growth_rate each tick price stays within barrier.

        ACCU proposals expire within 1-2 ticks (price is always moving), so the buy
        is retried once with a fresh proposal if the first attempt gets InvalidContractProposal.
        """
        params = {
            "proposal": 1,
            "amount": round(stake, 2),
            "basis": "stake",
            "contract_type": "ACCU",
            "currency": currency,
            "growth_rate": growth_rate,
            "underlying_symbol": symbol,
        }
        for attempt in range(2):
            proposal_resp = await self._send(params)
            proposal_id   = proposal_resp["proposal"]["id"]
            logger.debug(
                f"ACCU proposal {proposal_id} | {symbol} | "
                f"growth={growth_rate * 100:.0f}%/tick | stake={stake}"
                + (f" (retry {attempt})" if attempt else "")
            )
            try:
                buy_resp = await self._send({"buy": proposal_id, "price": stake})
                break
            except RuntimeError as exc:
                if attempt == 0 and "InvalidContractProposal" in str(exc):
                    logger.warning(
                        f"[{symbol}] ACCU proposal expired, retrying with fresh quote"
                    )
                    continue
                raise
        contract_id = buy_resp["buy"]["contract_id"]
        logger.info(
            f"ACCU bought: {contract_id} | {symbol} | "
            f"growth={growth_rate * 100:.0f}%/tick | stake={stake}"
        )
        await self._send({"proposal_open_contract": 1, "contract_id": contract_id, "subscribe": 1})
        return buy_resp["buy"]

    async def sell_contract(self, contract_id: str) -> dict:
        """Manually close an open contract (Accumulator sell-at-market)."""
        resp = await self._send({"sell": int(contract_id), "price": 0})
        logger.info(f"Contract sold: {contract_id}")
        return resp


# ── Module-level helpers ────────────────────────────────────────────────


def _extract_account_id(data) -> str:
    """Best-effort extraction of account_id from the accounts REST response."""
    if isinstance(data, list) and data:
        item = data[0]
        return item.get("account_id") or item.get("id") or item.get("loginid") or ""
    if isinstance(data, dict):
        # Might be wrapped: {"accounts": [...]} or {"data": [...]}
        for key in ("accounts", "data", "result"):
            if key in data and isinstance(data[key], list) and data[key]:
                item = data[key][0]
                return item.get("account_id") or item.get("id") or item.get("loginid") or ""
        return data.get("account_id") or data.get("id") or data.get("loginid") or ""
    return ""


def _extract_ws_url(data) -> str:
    """Best-effort extraction of the WebSocket URL from the OTP REST response."""
    if isinstance(data, dict):
        for key in ("websocket_url", "ws_url", "url", "wss_url", "socket_url"):
            if key in data:
                return data[key]
        # Maybe nested under "data" or "result"
        for wrapper in ("data", "result"):
            if wrapper in data and isinstance(data[wrapper], dict):
                return _extract_ws_url(data[wrapper])
    return ""
