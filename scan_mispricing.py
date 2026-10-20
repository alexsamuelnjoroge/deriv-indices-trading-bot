"""
Mispricing scanner for Deriv synthetic indices (phase 1: tick-expiry contracts).

Does NOT predict prices. For every tick-expiry contract variant it compares
Deriv's live payout (proposal only — nothing is bought) with the win
probability measured on 50k real ticks of that symbol:

    EV per $1 staked = P(win) * payout_ratio - 1

A contract is flagged only if EV > 0 with z >= 3.5 (about 2,000 variants are
tested, so weaker evidence would just be noise). Standard errors use
non-overlapping windows so overlapping tick windows don't inflate confidence.

Settlement rules modelled (Deriv tick contracts): entry spot = first tick after
purchase (index e), exit spot = N-th tick after it (index e+N).

Usage:
    python scan_mispricing.py [--symbols R_75 JD75 ...] [--refresh]
"""

import argparse
import asyncio
import json
import math
import os
from pathlib import Path

import numpy as np
import websockets
from dotenv import load_dotenv
from loguru import logger

from src.api.client import DerivClient

logger.remove()

SYMBOLS = ["R_10", "R_25", "R_50", "R_75", "R_100",
           "1HZ10V", "1HZ15V", "1HZ25V", "1HZ30V", "1HZ50V", "1HZ75V", "1HZ90V", "1HZ100V",
           "JD10", "JD25", "JD50", "JD75", "JD100"]
N_TICKS = 50_000
CACHE = Path("data/ticks")
STAKE = 10.0
Z_FLAG = 3.5


# ── data ─────────────────────────────────────────────────────────────────

async def ticks(c: DerivClient, sym: str, refresh: bool) -> tuple[np.ndarray, int]:
    f = CACHE / f"{sym}_{N_TICKS}.json"
    if f.exists() and not refresh:
        d = json.loads(f.read_text())
        return np.array(d["prices"]), d["pip"]
    prices, times, end, pip = [], [], "latest", 2
    while len(prices) < N_TICKS:
        r = await c._send({"ticks_history": sym, "style": "ticks", "count": 5000, "end": end})
        h = r["history"]
        pip = int(r.get("pip_size", pip))
        new = [(t, p) for t, p in zip(h["times"], h["prices"]) if not times or t < times[0]]
        if not new:
            break
        times = [t for t, _ in new] + times
        prices = [float(p) for _, p in new] + prices
        end = times[0] - 1
        await asyncio.sleep(0.3)
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"prices": prices[-N_TICKS:], "pip": pip}))
    return np.array(prices[-N_TICKS:]), pip


# ── outcomes from history ─────────────────────────────────────────────────

def windows(p: np.ndarray, n: int) -> np.ndarray:
    """Rows = entry index e, columns = ticks e..e+n."""
    m = len(p) - n
    return np.lib.stride_tricks.sliding_window_view(p, n + 1)[:m]


def outcome(p, pip, ct, n, barrier=None, k=None):
    w = windows(p, n)
    e, x, path = w[:, 0], w[:, -1], w[:, 1:]
    if ct == "CALL":  return x > e
    if ct == "PUT":   return x < e
    if ct == "CALLE": return x >= e
    if ct == "PUTE":  return x <= e
    if ct == "HIGHER": return x > e + barrier
    if ct == "LOWER":  return x < e + barrier
    if ct in ("ONETOUCH", "NOTOUCH"):
        hit = (path.max(axis=1) >= e + barrier) if barrier > 0 else (path.min(axis=1) <= e + barrier)
        return hit if ct == "ONETOUCH" else ~hit
    if ct.startswith("DIGIT"):
        d = np.rint(x * 10 ** pip).astype(np.int64) % 10
        if ct == "DIGITEVEN": return d % 2 == 0
        if ct == "DIGITODD":  return d % 2 == 1
        return {"DIGITMATCH": lambda: d == k, "DIGITDIFF": lambda: d != k,
                "DIGITOVER": lambda: d > k, "DIGITUNDER": lambda: d < k}[ct]()
    if ct == "ASIANU": return x > w.mean(axis=1)
    if ct == "ASIAND": return x < w.mean(axis=1)
    if ct in ("TICKHIGH", "TICKLOW"):
        sel = path[:, k - 1]
        return sel >= path.max(axis=1) if ct == "TICKHIGH" else sel <= path.min(axis=1)
    if ct in ("RUNHIGH", "RUNLOW"):
        d = np.diff(w, axis=1)
        return (d > 0).all(axis=1) if ct == "RUNHIGH" else (d < 0).all(axis=1)
    raise ValueError(ct)


# ── live payouts ─────────────────────────────────────────────────────────

async def payout(c, sym, ct, n, barrier_str=None, k=None):
    req = {"proposal": 1, "amount": STAKE, "basis": "stake", "contract_type": ct, "currency": "USD",
           "duration": n, "duration_unit": "t", "underlying_symbol": sym}
    if barrier_str is not None:
        req["barrier"] = barrier_str
    if ct in ("TICKHIGH", "TICKLOW"):
        req["selected_tick"] = k
    elif ct.startswith("DIGIT") and k is not None:
        req["barrier"] = str(k)
    for attempt in range(4):
        try:
            r = await c._send(req)
            return float(r["proposal"]["payout"]) / STAKE
        except RuntimeError as e:
            if "RateLimit" in str(e):
                await asyncio.sleep(5 * (attempt + 1))
                continue
            return None
        except (asyncio.CancelledError, asyncio.TimeoutError, ConnectionError,
                websockets.exceptions.ConnectionClosed):          # client reconnecting
            await asyncio.sleep(15)
    return None


def variants(sigma: float, pip: int):
    """(contract_type, duration, barrier_offset_or_None, digit_or_selected_tick, label)."""
    fmt = lambda v: f"{'+' if v >= 0 else '-'}{abs(v):.{pip}f}"
    out = []
    for n in (1, 2, 3, 5, 10):
        for ct in ("CALL", "PUT", "CALLE", "PUTE"):
            out.append((ct, n, None, None, f"{ct} {n}t"))
    for n in (5, 10):
        for mult in (-2, -1, -0.5, 0.5, 1, 2):
            off = round(mult * sigma * math.sqrt(n), pip)
            for ct in ("HIGHER", "LOWER"):
                out.append((ct, n, (off, fmt(off)), None, f"{ct} {n}t barrier {mult:+g}σ"))
            if abs(mult) >= 1:
                for ct in ("ONETOUCH", "NOTOUCH"):
                    out.append((ct, n, (off, fmt(off)), None, f"{ct} {n}t barrier {mult:+g}σ"))
        for ct in ("ASIANU", "ASIAND"):
            out.append((ct, n, None, None, f"{ct} {n}t"))
    for n in (1, 5):
        for d in range(10):
            out.append(("DIGITMATCH", n, None, d, f"DIGITMATCH {d} {n}t"))
            out.append(("DIGITDIFF", n, None, d, f"DIGITDIFF {d} {n}t"))
        for d in range(9):
            out.append(("DIGITOVER", n, None, d, f"DIGITOVER {d} {n}t"))
        for d in range(1, 10):
            out.append(("DIGITUNDER", n, None, d, f"DIGITUNDER {d} {n}t"))
        out += [("DIGITEVEN", n, None, None, f"DIGITEVEN {n}t"), ("DIGITODD", n, None, None, f"DIGITODD {n}t")]
    for k in range(1, 6):
        out += [("TICKHIGH", 5, None, k, f"TICKHIGH tick {k}"), ("TICKLOW", 5, None, k, f"TICKLOW tick {k}")]
    for n in range(2, 6):
        out += [("RUNHIGH", n, None, None, f"RUNHIGH {n}t"), ("RUNLOW", n, None, None, f"RUNLOW {n}t")]
    return out


async def accu(c, sym, p):
    rows = []
    r = np.abs(np.diff(p) / p[:-1])
    for g in (0.01, 0.02, 0.03, 0.04, 0.05):
        try:
            resp = await c._send({"proposal": 1, "amount": STAKE, "basis": "stake", "contract_type": "ACCU",
                                  "currency": "USD", "growth_rate": g, "underlying_symbol": sym})
            tsb = float(resp["proposal"]["contract_details"]["tick_size_barrier"])
        except Exception:
            continue
        win = r[1:] < tsb
        rows.append((f"ACCU {int(g*100)}% (1 growth tick)", win, 1 + g))
    return rows


def evaluate(win: np.ndarray, ratio: float, stride: int):
    pw = win.mean()
    ev = pw * ratio - 1
    nonover = win[::max(stride, 1)]
    se = math.sqrt(max(nonover.mean() * (1 - nonover.mean()), 1e-12) / len(nonover)) * ratio
    return pw, ev, ev / se if se else 0.0


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=SYMBOLS)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--resume", action="store_true", help="keep saved results, skip symbols already scanned")
    a = ap.parse_args()

    load_dotenv(".env")
    c = DerivClient(os.getenv("DERIV_API_TOKEN"))
    await c.connect()
    out = Path("data/mispricing_scan.json")
    results = [tuple(r) for r in json.loads(out.read_text())] if a.resume and out.exists() else []
    done = {r[0] for r in results}
    for sym in a.symbols:
        if sym in done:
            continue
        try:
            p, pip = await ticks(c, sym, a.refresh)
        except Exception as e:
            print(f"{sym}: tick download failed ({e})")
            continue
        sigma = float(np.std(np.diff(p)))
        tested = 0
        for ct, n, bar, k, lab in variants(sigma, pip):
            ratio = await payout(c, sym, ct, n, bar[1] if bar else None, k)
            await asyncio.sleep(0.15)
            if ratio is None:
                continue
            win = outcome(p, pip, ct, n, bar[0] if bar else None, k)
            pw, ev, z = evaluate(win, ratio, n)
            results.append((sym, lab, pw, ratio, ev, z))
            tested += 1
        for lab, win, ratio in await accu(c, sym, p):
            pw, ev, z = evaluate(win, ratio, 1)
            results.append((sym, lab, pw, ratio, ev, z))
            tested += 1
        out.write_text(json.dumps([list(r[:2]) + [float(x) for x in r[2:]] for r in results]))
        best = max((r for r in results if r[0] == sym), key=lambda r: r[4], default=None)
        print(f"{sym:<8} {len(p)} ticks, {tested} contract variants priced"
              + (f" | best: {best[1]} EV {best[4]*100:+.2f}c (z={best[5]:+.1f})" if best else ""), flush=True)
    await c.disconnect()

    evs = np.array([r[4] for r in results])
    print(f"\n{len(results)} variants. Deriv margin (−EV per $1): median {-np.median(evs)*100:.2f}c, "
          f"best {-(evs.max())*100:+.2f}c, worst {-(evs.min())*100:.2f}c")
    print("\nTop 15 by EV:")
    print(f"  {'symbol':<8} {'contract':<32} {'P(win)':>8} {'payout':>7} {'EV per $1':>10} {'z':>6}")
    for sym, lab, pw, ratio, ev, z in sorted(results, key=lambda r: -r[4])[:15]:
        print(f"  {sym:<8} {lab:<32} {pw*100:7.2f}% {ratio:7.3f} {ev*100:+9.2f}c {z:+6.1f}")
    flagged = [r for r in results if r[4] > 0 and r[5] >= Z_FLAG]
    print(f"\nFLAGGED (EV > 0 and z >= {Z_FLAG}): {len(flagged)}")
    for sym, lab, pw, ratio, ev, z in flagged:
        print(f"  {sym:<8} {lab:<32} EV {ev*100:+.2f}c z={z:.1f}")
    print("\nMargin by contract family (median −EV per $1):")
    fam = {}
    for sym, lab, *_rest in results:
        fam.setdefault(lab.split()[0], []).append(_rest[2])
    for f_, v in sorted(fam.items(), key=lambda kv: np.median(kv[1])):
        print(f"  {f_:<12} {-np.median(v)*100:6.2f}c   (n={len(v)})")


if __name__ == "__main__":
    asyncio.run(main())
