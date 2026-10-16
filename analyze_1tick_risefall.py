"""
Is there an edge in 1-tick Rise/Fall contracts, unconditionally or when the
symbol is in a low-volatility state?

Step 1: live 1-tick CALL/PUT proposals -> real payout -> break-even win rate.
Step 2: historical ticks -> win rate per symbol, unfiltered and with filters.

Timing: decision at tick t, contract entry spot = tick t+1, exit = tick t+2
(Deriv uses the next tick after purchase as entry). Exit == entry is a loss
for both CALL and PUT.

Selection bias guard: filters are ranked on the first half of the data and
the winner is re-checked on the untouched second half.

Usage:
    python analyze_1tick_risefall.py
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
logger.add(sys.stderr, level="ERROR")

SYMBOLS = ["R_10", "R_25", "R_50", "R_75", "R_100",
           "1HZ10V", "1HZ15V", "1HZ25V", "1HZ30V", "1HZ50V", "1HZ75V", "1HZ90V", "1HZ100V",
           "CRASH50", "CRASH150N", "CRASH300N", "CRASH500", "CRASH600", "CRASH900", "CRASH1000",
           "BOOM50", "BOOM150N", "BOOM300N", "BOOM500", "BOOM600", "BOOM900", "BOOM1000"]
TICK_COUNT = 50_000
TICK_COUNT_CB = 60_000  # matches existing Crash/Boom caches
VOL_WINDOW = 20


async def fetch_payouts() -> dict:
    load_dotenv()
    token = os.getenv("DERIV_API_TOKEN") or os.getenv("DERIV_TOKEN")
    client = DerivClient(token)
    await client.connect()
    out = {}
    for sym in SYMBOLS:
        out[sym] = {}
        for ct in ("CALL", "PUT"):
            try:
                resp = await client._send({
                    "proposal": 1, "amount": 1.0, "basis": "stake",
                    "contract_type": ct, "currency": "USD",
                    "duration": 1, "duration_unit": "t",
                    "underlying_symbol": sym,
                })
                if "proposal" in resp:
                    out[sym][ct] = float(resp["proposal"]["payout"])
                else:
                    out[sym][ct] = None
                    out[sym]["err"] = resp.get("error", {}).get("message", "?")
            except Exception as e:
                out[sym][ct] = None
                out[sym]["err"] = str(e)[:90]
    await client.disconnect()
    return out


def load_prices(sym) -> np.ndarray:
    count = TICK_COUNT_CB if sym.startswith(("CRASH", "BOOM")) else TICK_COUNT
    raw = fetch_ticks(sym, count=count)
    return np.array([float(t["quote"]) if isinstance(t, dict) else float(t) for t in raw])


def build_masks(p: np.ndarray):
    """Returns (call_win, put_win, masks) aligned to decision index t."""
    r = np.diff(np.log(p))                       # r[i] = move from p[i] to p[i+1]
    n = len(p)
    t = np.arange(VOL_WINDOW, n - 2)
    entry, exit_ = p[t + 1], p[t + 2]
    call_win = exit_ > entry
    put_win = exit_ < entry

    # trailing realised vol known at time t (returns up to p[t])
    csum = np.concatenate([[0], np.cumsum(r ** 2)])
    vol = np.sqrt((csum[t] - csum[t - VOL_WINDOW]) / VOL_WINDOW)
    last_abs = np.abs(r[t - 1])
    last_up = r[t - 1] > 0
    q20 = np.quantile(vol, 0.20)
    q10 = np.quantile(vol, 0.10)
    l20 = np.quantile(last_abs, 0.20)

    masks = {
        "all":                np.ones_like(call_win, dtype=bool),
        "vol<q20":            vol < q20,
        "vol<q10":            vol < q10,
        "last|r|<q20":        last_abs < l20,
        "lastUp":             last_up,
        "lastDown":           ~last_up,
        "vol<q20&lastUp":     (vol < q20) & last_up,
        "vol<q20&lastDown":   (vol < q20) & ~last_up,
    }
    return call_win, put_win, masks


def wr_stats(wins: np.ndarray, be: float):
    n = len(wins)
    if n == 0:
        return 0.0, 0, 0.0
    wr = wins.mean()
    z = (wr - be) / math.sqrt(be * (1 - be) / n)
    return wr, n, z


def main():
    print("\nFetching live 1-tick payouts...")
    payouts = asyncio.run(fetch_payouts())

    print("\n" + "=" * 92)
    print(f"  {'symbol':<8} {'payout C/P':>12} {'BE%':>6}   {'best filter (in-sample H1)':<26} "
          f"{'H1 WR%':>7} {'H2 WR%':>7} {'H2 z':>6} {'H2 EV/$':>8}")
    print("=" * 92)

    summary = []
    for sym in SYMBOLS:
        pc, pp = payouts[sym].get("CALL"), payouts[sym].get("PUT")
        if not pc or not pp:
            print(f"  {sym:<8} no 1-tick Rise/Fall: {payouts[sym].get('err', '?')}")
            continue
        try:
            prices = load_prices(sym)
        except Exception as e:
            print(f"  {sym:<8} tick fetch failed: {e}")
            continue

        call_win, put_win, masks = build_masks(prices)
        half = len(call_win) // 2
        h1, h2 = slice(0, half), slice(half, None)

        best = None
        for name, m in masks.items():
            for side, wins, pay in (("CALL", call_win, pc), ("PUT", put_win, pp)):
                be = 1.0 / pay
                wr1, n1, z1 = wr_stats(wins[h1][m[h1]], be)
                if n1 < 500:
                    continue
                if best is None or wr1 - be > best[0]:
                    best = (wr1 - be, name, side, pay, be, wr1)

        _, name, side, pay, be, wr1 = best
        wins = call_win if side == "CALL" else put_win
        wr2, n2, z2 = wr_stats(wins[h2][masks[name][h2]], be)
        ev2 = wr2 * pay - 1.0
        tie_rate = 1 - call_win.mean() - put_win.mean()

        print(f"  {sym:<8} {pc:>5.3f}/{pp:<5.3f} {be*100:>6.2f}   {side+' '+name:<26} "
              f"{wr1*100:>7.2f} {wr2*100:>7.2f} {z2:>6.2f} {ev2:>+8.4f}"
              + (f"   ties={tie_rate*100:.1f}%" if tie_rate > 0.005 else ""))
        summary.append((sym, side, name, ev2, z2, n2))

    print("=" * 92)
    passing = [s for s in summary if s[3] > 0 and s[4] > 2.0]
    print("\n  Out-of-sample (H2) EV > 0 with z > 2:")
    if passing:
        for sym, side, name, ev, z, n in passing:
            print(f"    {sym} {side} {name}: EV={ev:+.4f}/$  z={z:.2f}  n={n}")
    else:
        print("    NONE - no symbol/filter beats the payout out of sample.")
    print()


if __name__ == "__main__":
    main()
