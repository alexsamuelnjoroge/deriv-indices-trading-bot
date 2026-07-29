"""
Test: what binary option durations does Deriv accept for real forex + metals?
Tries CALL proposals at 1, 2, 3, 5, 10, 15-minute durations for each symbol.
Also tests tick-duration contracts (5t, 10t) to see if they're supported.
Reports OK / FAIL + payout for each accepted duration.
"""

import asyncio
import json
import os

import aiohttp
import websockets
from dotenv import load_dotenv

load_dotenv()
TOKEN    = os.getenv("DERIV_API_TOKEN")
APP_KEY  = os.getenv("DERIV_APP_KEY", "")
REST_BASE = "https://api.derivws.com"

SYMBOLS = ["frxXAUUSD", "frxXAGUSD", "frxUSDJPY", "frxGBPUSD", "frxEURUSD"]

DURATIONS = [
    (1,  "m", " 1min"),
    (2,  "m", " 2min"),
    (3,  "m", " 3min"),
    (5,  "m", " 5min"),
    (10, "m", "10min"),
    (15, "m", "15min"),
    (5,  "t", " 5tick"),
    (10, "t", "10tick"),
]


async def get_ws_url():
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type":  "application/json",
        "Deriv-App-ID":  APP_KEY,
    }
    async with aiohttp.ClientSession() as s:
        async with s.get(
            f"{REST_BASE}/trading/v1/options/accounts", headers=headers
        ) as r:
            data = await r.json()
            account_id = data["data"][0]["account_id"]

        async with s.post(
            f"{REST_BASE}/trading/v1/options/accounts/{account_id}/otp",
            headers=headers,
        ) as r:
            data = await r.json()
            return data["data"]["url"]


async def test_proposal(ws, symbol, duration, unit, req_id):
    await ws.send(json.dumps({
        "proposal":         1,
        "req_id":           req_id,
        "amount":           1.0,
        "basis":            "stake",
        "contract_type":    "CALL",
        "currency":         "USD",
        "duration":         duration,
        "duration_unit":    unit,
        "underlying_symbol": symbol,
    }))
    raw = await asyncio.wait_for(ws.recv(), timeout=30)
    msg = json.loads(raw)

    if msg.get("error"):
        return False, msg["error"]["code"], None

    p      = msg.get("proposal", {})
    payout = p.get("payout", "?")
    ask    = p.get("ask_price", "?")
    # Implied payout % = (payout - ask) / ask  → i.e. profit/stake
    try:
        payout_pct = round((float(payout) - float(ask)) / float(ask) * 100, 1)
    except Exception:
        payout_pct = "?"
    return True, None, f"ask=${ask} payout=${payout} ({payout_pct}% profit)"


async def main():
    print("Fetching OTP WebSocket URL...")
    ws_url = await get_ws_url()
    print("Connected.\n")

    async with websockets.connect(ws_url) as ws:
        req_id = 1
        for sym in SYMBOLS:
            print(f"── {sym} ──")
            for duration, unit, label in DURATIONS:
                ok, err_code, detail = await test_proposal(ws, sym, duration, unit, req_id)
                req_id += 1
                if ok:
                    print(f"  OK   {label} | {detail}")
                else:
                    print(f"  FAIL {label} | {err_code}")
            print()


asyncio.run(main())
