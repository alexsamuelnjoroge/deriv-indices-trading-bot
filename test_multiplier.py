"""
Quick test: can this account buy a MULTUP proposal on frxXAUUSD?
Connects, fetches a proposal, prints result. Does NOT place a trade.
"""
import asyncio
import json
import os
import sys
import aiohttp
import websockets
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("DERIV_API_TOKEN")
APP_KEY = os.getenv("DERIV_APP_KEY", "")
REST_BASE = "https://api.derivws.com"

SYMBOLS = ["frxXAUUSD", "frxXAGUSD", "frxGBPUSD"]

async def get_ws_url():
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json",
        "Deriv-App-ID": APP_KEY,
    }
    async with aiohttp.ClientSession() as s:
        async with s.get(f"{REST_BASE}/trading/v1/options/accounts", headers=headers) as r:
            data = await r.json()
            account_id = data["data"][0]["account_id"]
            print(f"Account: {account_id}")

        async with s.post(
            f"{REST_BASE}/trading/v1/options/accounts/{account_id}/otp",
            headers=headers,
        ) as r:
            data = await r.json()
            return data["data"]["url"]

async def test_proposal(ws, symbol, req_id):
    payload = {
        "proposal": 1,
        "req_id": req_id,
        "amount": 1.0,
        "basis": "stake",
        "contract_type": "MULTUP",
        "currency": "USD",
        "multiplier": 100,
        "underlying_symbol": symbol,
        "limit_order": {"stop_loss": 0.75, "take_profit": 2.00},
    }
    await ws.send(json.dumps(payload))
    raw = await asyncio.wait_for(ws.recv(), timeout=45)
    msg = json.loads(raw)
    if msg.get("error"):
        print(f"  FAIL [{symbol}]: {msg['error']['code']} — {msg['error']['message']}")
    else:
        p = msg.get("proposal", {})
        print(f"  OK   [{symbol}]: proposal id={p.get('id')} | ask={p.get('ask_price')} | display={p.get('display_value')}")

async def main():
    print("Fetching OTP WebSocket URL...")
    ws_url = await get_ws_url()
    print(f"Connecting to API...")

    async with websockets.connect(ws_url) as ws:
        print(f"Connected. Testing multiplier proposals...\n")
        for i, sym in enumerate(SYMBOLS, start=1):
            print(f"Testing {sym}...")
            await test_proposal(ws, sym, req_id=i)

asyncio.run(main())
