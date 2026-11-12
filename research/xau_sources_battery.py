"""
Gold strategies taken from published sources (pre-registered 2026-11-11). Rules follow the
sources as closely as published; where a source withholds a detail, the choice is noted.

Data: Dukascopy M5 2008-2026 converted to IC server time (New York + 7h), via
xau_session_breakout_ea. ET = server - 7h.
Costs 0.36bp RT (measured on IC MT5), stress 0.72bp. Ambiguity inside a bar is booked
against us (stop before target; a fill and a stop in the same bar = stop).
Split TRAIN 2008-2016 | TEST 2017-2026. Pass: TRAIN mean > 0, t >= 2.5; TEST mean > 0,
t >= 1.5, and > 0 at 2x cost.

C1 Initial Balance retracement (tradethatswing.com, "one trade a day gold strategy")
   IB = 09:30-10:30 ET. After price first breaks the IB, limit entry 25% of the IB back
   inside it, in the breakout direction. Flat at 16:00 ET. One trade a day.
   C1a original : IB 0.3%-1.5% of price; SL 60% of IB from entry; TP 50% of IB beyond the IB edge
   C1b April 26 : IB 0.4%-2%; longs only; SL 100% of IB from entry; TP 20% of IB beyond the edge
C2 Prior-day high/low false break (ungeracademy.com, mean-reverting gold strategies)
   A bar trades above yesterday's high and closes back below it -> short at the close
   (mirror for lows). Stop: the sweep extreme (source withholds its stop). Flat at 16:00 ET.
   First signal of the day only. C2a on M5, C2b on M15.
C4 Asian-hours long (CBS thesis "Gold price dynamics around the clock": gold rises in
   eastern hours). Long at the open of server 01:00 (after the daily rollover, so no swap),
   flat at C4a 09:00 / C4b 07:00 server.
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xau_session_breakout_ea as base  # noqa: E402

COST, STRESS, TEST_YEAR = 0.36, 0.72, 2017
m5 = base.m5.copy()
m5["mins"] = m5.index.hour * 60 + m5.index.minute
m15 = base.m15
H = lambda h, m=0: (h + 7) * 60 + m            # ET clock -> server minutes


def days(frame):
    for d, g in frame.groupby(frame.index.date):
        if g.index[0].weekday() < 5:
            yield d, g


def walk(bars, start, side, stop, target, end_min):
    """Exit price from bar index start on: stop before target; flat at end_min."""
    for j in range(start, len(bars)):
        b = bars.iloc[j]
        if b["mins"] >= end_min:
            return b["open"]
        if (side > 0 and b["low"] <= stop) or (side < 0 and b["high"] >= stop):
            return stop
        if target is not None and ((side > 0 and b["high"] >= target) or (side < 0 and b["low"] <= target)):
            return target
    return bars.iloc[-1]["close"]


def c1(variant):
    lo_pct, hi_pct, sl_f, tp_f, longs_only = ((0.003, 0.015, 0.6, 0.5, False) if variant == "a"
                                              else (0.004, 0.02, 1.0, 0.2, True))
    out = []
    for d, g in days(m5):
        ib = g[(g["mins"] >= H(9, 30)) & (g["mins"] < H(10, 30))]
        if len(ib) < 10:
            continue
        hi, lo = ib["high"].max(), ib["low"].min()
        w = hi - lo
        if not (lo_pct <= w / lo <= hi_pct):
            continue
        rest = g[(g["mins"] >= H(10, 30))].reset_index(drop=True)
        side, k0 = 0, None
        for k in range(len(rest)):
            b = rest.iloc[k]
            if b["mins"] >= H(16):
                break
            if b["high"] > hi and b["low"] < lo:
                break                                   # double break: not a single break
            if b["high"] > hi or b["low"] < lo:
                side, k0 = (1 if b["high"] > hi else -1), k
                break
        if side == 0 or (longs_only and side < 0):
            continue
        entry = hi - 0.25 * w if side > 0 else lo + 0.25 * w
        stop = entry - side * sl_f * w
        target = hi + tp_f * w if side > 0 else lo - tp_f * w
        for k in range(k0 + 1, len(rest)):
            b = rest.iloc[k]
            if b["mins"] >= H(16):
                break
            if (side > 0 and b["low"] <= entry) or (side < 0 and b["high"] >= entry):
                if (side > 0 and b["low"] <= stop) or (side < 0 and b["high"] >= stop):
                    px = stop
                else:
                    px = walk(rest, k + 1, side, stop, target, H(16))
                out.append((d, side, side * (px / entry - 1) * 1e4))
                break
    return out


def c2(frame):
    out = []
    prev = None
    for d, g in days(frame):
        if prev is not None:
            pdh, pdl = prev
            sweep_hi, sweep_lo = -np.inf, np.inf
            gg = g.reset_index(drop=True)
            for k in range(len(gg)):
                b = gg.iloc[k]
                if b["mins"] < H(-6) or b["mins"] >= H(15, 45):     # from server 01:00 to 15:45 ET
                    continue
                sweep_hi, sweep_lo = max(sweep_hi, b["high"]), min(sweep_lo, b["low"])
                side = -1 if (b["high"] > pdh and b["close"] < pdh) else 1 if (b["low"] < pdl and b["close"] > pdl) else 0
                if side:
                    entry = b["close"]
                    stop = sweep_hi if side < 0 else sweep_lo
                    px = walk(gg, k + 1, side, stop, None, H(16))
                    out.append((d, side, side * (px / entry - 1) * 1e4))
                    break
        prev = (g["high"].max(), g["low"].min())
    return out


def c4(exit_hour):
    out = []
    for d, g in days(m15):
        a = g[g["mins"] >= 60]
        b = g[g["mins"] >= exit_hour * 60]
        if len(a) == 0 or len(b) == 0 or a.index[0].hour != 1 or b.index[0].hour != exit_hour:
            continue
        out.append((d, 1, (b.iloc[0]["open"] / a.iloc[0]["open"] - 1) * 1e4))
    return out


def stats(tr, cost):
    if len(tr) < 20:
        return None
    net = np.array([x[2] for x in tr]) - cost
    sd = net.std(ddof=1)
    return dict(n=len(net), tpy=len(net) / len({x[0].year for x in tr}), mean=net.mean(),
                t=net.mean() / sd * math.sqrt(len(net)) if sd > 0 else 0, wr=(net > 0).mean() * 100)


def fmt(s):
    return "n<20" if not s else f"n={s['n']:4d} /yr={s['tpy']:4.0f} mean={s['mean']:+6.2f}bp t={s['t']:+5.2f} WR={s['wr']:4.1f}%"


VARIANTS = {
    "C1a IB retracement (original)": lambda: c1("a"),
    "C1b IB retracement (April, longs only)": lambda: c1("b"),
    "C2a prior-day false break M5": lambda: c2(m5),
    "C2b prior-day false break M15": lambda: c2(m15),
    "C4a Asian-hours long 01->09 server": lambda: c4(9),
    "C4b Asian-hours long 01->07 server": lambda: c4(7),
}

if __name__ == "__main__":
    for name, fn in VARIANTS.items():
        tr = fn()
        a = [x for x in tr if x[0].year < TEST_YEAR]
        b = [x for x in tr if x[0].year >= TEST_YEAR]
        sa, sb, sb2 = stats(a, COST), stats(b, COST), stats(b, STRESS)
        ok = sa and sb and sb2 and sa["mean"] > 0 and sa["t"] >= 2.5 and sb["mean"] > 0 and sb["t"] >= 1.5 and sb2["mean"] > 0
        print(f"\n[{'PASS' if ok else 'fail'}] {name}")
        print(f"   TRAIN {fmt(sa)}")
        print(f"   TEST  {fmt(sb)}")
        for side, lab in ((1, "long "), (-1, "short")):
            print(f"     {lab} train {fmt(stats([x for x in a if x[1] == side], COST))} | "
                  f"test {fmt(stats([x for x in b if x[1] == side], COST))}")
        yr = {}
        for x in tr:
            yr.setdefault(x[0].year, []).append(x[2] - COST)
        print("   years: " + " ".join(f"{y % 100:02d}:{np.mean(v):+.1f}" for y, v in sorted(yr.items())), flush=True)
