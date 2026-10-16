"""
1-tick Accumulator on Crash/Boom: buy, survive one tick, sell.

Timing: decision at tick t, entry spot = p[t+1]; win if p[t+2] stays within
+/- tick_size_barrier of p[t+1]. Win pays stake*g, loss costs the stake.
Break-even survival = 1/(1+g). Sell latency is ignored (optimistic).

Barriers come from live ACCU proposals. Filters are ranked on the first half
of the data and the winner is re-checked on the untouched second half.

Usage:
    python analyze_1tick_accu_cb.py
"""

import asyncio
import math
import os
import sys

import numpy as np
from dotenv import load_dotenv
from loguru import logger

from src.api.client import DerivClient
from src.data.history import fetch_ticks

logger.remove()
logger.add(sys.stderr, level="CRITICAL")

SYMBOLS = ["CRASH50", "CRASH150N", "CRASH300N", "CRASH500", "CRASH600", "CRASH900", "CRASH1000",
           "BOOM50", "BOOM150N", "BOOM300N", "BOOM500", "BOOM600", "BOOM900", "BOOM1000"]
GROWTH_RATES = [0.01, 0.02, 0.03, 0.04, 0.05]
TICK_COUNT = 60_000
VOL_WINDOW = 20


async def fetch_barriers() -> dict:
    load_dotenv(".env")
    token = os.getenv("DERIV_API_TOKEN") or os.getenv("DERIV_TOKEN")
    client = DerivClient(token)
    await client.connect()
    out = {}
    for sym in SYMBOLS:
        for g in GROWTH_RATES:
            try:
                resp = await client._send({
                    "proposal": 1, "amount": 1.0, "basis": "stake",
                    "contract_type": "ACCU", "currency": "USD",
                    "growth_rate": g, "underlying_symbol": sym,
                })
                out[(sym, g)] = float(resp["proposal"]["contract_details"]["tick_size_barrier"])
            except Exception as e:
                out[(sym, g)] = None
                print(f"  {sym} g={g}: {str(e)[:80]}")
    await client.disconnect()
    return out


def build(p: np.ndarray, tsb: float):
    r = np.diff(p) / p[:-1]                 # r[i] = move p[i] -> p[i+1]
    n = len(p)
    t = np.arange(VOL_WINDOW + 1, n - 2)
    win = np.abs(r[t + 1]) < tsb             # entry p[t+1] -> p[t+2]

    absr = np.abs(r)
    csum = np.concatenate([[0], np.cumsum(absr)])
    vol = (csum[t] - csum[t - VOL_WINDOW]) / VOL_WINDOW
    last = absr[t - 1]

    breach = absr >= tsb
    idx = np.where(breach, np.arange(len(r)), -1)
    last_breach = np.maximum.accumulate(idx)
    since = t - 1 - last_breach[t - 1]       # ticks since last breaching move

    masks = {
        "all":          np.ones_like(win),
        "vol<q20":      vol < np.quantile(vol, 0.20),
        "vol<q10":      vol < np.quantile(vol, 0.10),
        "last<q20":     last < np.quantile(last, 0.20),
        "calm>=5":      since >= 5,
        "calm>=20":     since >= 20,
        "calm<2":       since < 2,
        "vol<q20&calm>=5": (vol < np.quantile(vol, 0.20)) & (since >= 5),
    }
    return win, masks


def zscore(wins, be):
    n = len(wins)
    if n == 0:
        return 0.0, 0, 0.0
    wr = wins.mean()
    return wr, n, (wr - be) / math.sqrt(be * (1 - be) / n)


def main():
    print("\nFetching live ACCU barriers...")
    barriers = asyncio.run(fetch_barriers())

    print("\n" + "=" * 100)
    print(f"  {'symbol':<10} {'g':>4} {'barrier%':>9} {'BE%':>6} {'all WR%':>8}   "
          f"{'best filter (H1)':<17} {'H1 WR%':>7} {'H2 WR%':>7} {'H2 z':>6} {'H2 EV/$':>8} {'H2 n':>6}")
    print("=" * 100)

    results = []
    for sym in SYMBOLS:
        try:
            raw = fetch_ticks(sym, count=TICK_COUNT)
        except Exception as e:
            print(f"  {sym:<10} tick fetch failed: {e}")
            continue
        p = np.array([float(x["quote"]) if isinstance(x, dict) else float(x) for x in raw])
        half = None
        for g in GROWTH_RATES:
            tsb = barriers.get((sym, g))
            if not tsb:
                continue
            be = 1 / (1 + g)
            win, masks = build(p, tsb)
            half = len(win) // 2
            h1, h2 = slice(0, half), slice(half, None)

            best = None
            for name, m in masks.items():
                wr1, n1, _ = zscore(win[h1][m[h1]], be)
                if n1 < 500:
                    continue
                if best is None or wr1 > best[1]:
                    best = (name, wr1)
            name, wr1 = best
            wr2, n2, z2 = zscore(win[h2][masks[name][h2]], be)
            ev2 = wr2 * (1 + g) - 1
            print(f"  {sym:<10} {g*100:>3.0f}% {tsb*100:>8.5f}% {be*100:>6.2f} {win.mean()*100:>8.2f}   "
                  f"{name:<17} {wr1*100:>7.2f} {wr2*100:>7.2f} {z2:>6.2f} {ev2:>+8.4f} {n2:>6}")
            results.append((sym, g, name, ev2, z2, n2))
        print("-" * 100)

    passing = [x for x in results if x[3] > 0 and x[4] > 2.0]
    print("\n  Out-of-sample (H2) EV > 0 with z > 2:")
    if passing:
        for sym, g, name, ev, z, n in sorted(passing, key=lambda x: -x[3]):
            print(f"    {sym} g={g*100:.0f}% {name}: EV={ev:+.4f}/$  z={z:.2f}  n={n}")
    else:
        print("    NONE - no symbol/growth/filter beats break-even out of sample.")
    print()


if __name__ == "__main__":
    main()
