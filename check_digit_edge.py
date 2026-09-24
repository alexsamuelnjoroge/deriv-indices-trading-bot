"""
Run this daily to confirm the digit-0 scarcity edge is still present.

Usage:  python check_digit_edge.py
        python check_digit_edge.py R_50 R_75   # specific symbols
"""
import asyncio
import os
import sys
from collections import Counter
from dotenv import load_dotenv
from src.api.client import DerivClient

SYMBOLS = sys.argv[1:] or ["R_50", "R_75"]
TICK_COUNT = 3000


async def main():
    load_dotenv()
    token = os.getenv("DERIV_API_TOKEN") or os.getenv("DERIV_TOKEN")
    if not token:
        print("ERROR: DERIV_API_TOKEN not set")
        return

    client = DerivClient(token)
    await client.connect()

    for symbol in SYMBOLS:
        resp = await client._send({
            "ticks_history": symbol,
            "count": TICK_COUNT,
            "end": "latest",
            "style": "ticks",
        })
        pip_size = int(resp.get("pip_size", 4))
        prices = resp.get("history", {}).get("prices", [])

        if not prices:
            print(f"{symbol}: no data")
            continue

        digits = []
        for p in prices:
            try:
                digits.append(int(str(round(float(p), pip_size)).replace(".", "")[-1]))
            except Exception:
                pass

        total = len(digits)
        c = Counter(digits)
        over4 = sum(c.get(d, 0) for d in range(5, 10))
        d0 = c.get(0, 0)
        ev_pct = over4 / total * 0.95 - (total - over4) / total  # payout=95%

        dist = "  ".join(f"{d}:{c.get(d,0)/total*100:.1f}%" for d in range(10))
        print(
            f"\n{symbol} ({total} ticks, {pip_size}dp)\n"
            f"  digit dist: {dist}\n"
            f"  digit-0:    {d0}/{total} ({d0/total*100:.2f}%)  "
            f"{'EDGE OK: 0-scarcity present' if d0 == 0 else f'WARNING: digit-0 at {d0/total*100:.1f}%'}\n"
            f"  OVER(4) WR: {over4/total*100:.2f}%  "
            f"{'PROFITABLE' if over4/total > 0.52 else 'MARGINAL' if over4/total > 0.50 else 'UNPROFITABLE'}\n"
            f"  EV/trade:   {ev_pct*100:+.2f}%"
        )

    await client.disconnect()


asyncio.run(main())
