"""
Subscribe to LIVE ticks (not history) and compute digit distribution
from the actual tick_display_value strings, if the API provides them.

This is the ground truth: does digit-0 appear in the live feed?

Usage:  python analyze_live_ticks.py           # R_50 and R_75, 3 min
        python analyze_live_ticks.py R_25 120  # R_25, 2 min
"""
import asyncio
import json
import os
import sys
import time
from collections import Counter
from dotenv import load_dotenv
from src.api.client import DerivClient

SYMBOLS  = [a for a in sys.argv[1:] if not a.isdigit()] or ["R_50", "R_75"]
DURATION = int(next((a for a in sys.argv[1:] if a.isdigit()), 180))

PIP_SIZE = {"R_10": 3, "R_25": 3, "R_50": 4, "R_75": 4, "R_100": 2}


async def collect(client, symbol: str, duration: int):
    pip = PIP_SIZE.get(symbol, 4)
    ticks_float   = []  # digit from float via f-format (same as our analysis)
    ticks_display = []  # digit from tick_display_value string if API sends it
    has_display   = False

    # Send tick subscription
    await client._send({"ticks": symbol, "subscribe": 1})

    deadline = time.time() + duration
    print(f"  [{symbol}] Collecting live ticks for {duration}s …", flush=True)

    # Tap into the raw websocket stream
    async for raw in client.ws:
        if time.time() > deadline:
            break
        try:
            msg = json.loads(raw)
        except Exception:
            continue
        if msg.get("msg_type") != "tick":
            continue
        tick = msg.get("tick", {})

        # Float digit (same method as check_digit_edge.py)
        quote = tick.get("quote")
        if quote is not None:
            d_float = int(f"{float(quote):.{pip}f}"[-1])
            ticks_float.append(d_float)

        # Display-value digit — if the new API provides it
        disp = tick.get("tick_display_value") or tick.get("display_value")
        if disp:
            has_display = True
            ticks_display.append(int(str(disp)[-1]))

    # Unsubscribe
    try:
        await client._send({"forget_all": "ticks"})
    except Exception:
        pass

    return pip, ticks_float, ticks_display, has_display


def report(symbol, pip, digits, label):
    if not digits:
        print(f"  [{symbol}] No {label} data collected")
        return
    total = len(digits)
    c = Counter(digits)
    d0 = c.get(0, 0)
    over4 = sum(c.get(d, 0) for d in range(5, 10))
    dist = "  ".join(f"{d}:{c.get(d,0)/total*100:.1f}%" for d in range(10))
    ev = over4 / total * 0.95 - (total - over4) / total
    print(
        f"\n  [{symbol}] {label} — {total} ticks, {pip}dp\n"
        f"    digit dist : {dist}\n"
        f"    digit-0    : {d0}/{total} ({d0/total*100:.2f}%)  "
        f"{'TRULY ABSENT' if d0 == 0 else f'PRESENT at {d0/total*100:.1f}%'}\n"
        f"    OVER(4) WR : {over4/total*100:.2f}%\n"
        f"    EV/trade   : {ev*100:+.2f}%"
    )


async def main():
    load_dotenv()
    token = os.getenv("DERIV_API_TOKEN") or os.getenv("DERIV_TOKEN")
    if not token:
        print("ERROR: DERIV_API_TOKEN not set")
        return

    client = DerivClient(token)
    await client.connect()

    for symbol in SYMBOLS:
        pip, floats, displays, has_display = await collect(client, symbol, DURATION)
        report(symbol, pip, floats, "float f-format")
        if has_display:
            report(symbol, pip, displays, "tick_display_value")
        else:
            print(f"\n  [{symbol}] NOTE: tick_display_value not in live feed — "
                  "API only sends numeric quote; float f-format is our best proxy")

    await client.disconnect()


asyncio.run(main())
