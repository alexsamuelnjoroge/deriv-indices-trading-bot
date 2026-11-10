"""
XAUUSD professional playbook — INTRADAY session-structure setups (pre-registered 2026-11-10).

"Asia ranges, London breaks it, then it either runs or traps." All times London local
(DST-aware). Asian range = 00:00-07:00 London. Max one trade per setup per day; flat by 16:00.

Data : Dukascopy M5 bid 2008 -> 2026 resampled to M15 (data/duka/frxXAUUSD_M5_*.csv).
Exits: stop checked before target inside each M15 bar (conservative); time exit at 16:00
       (I1: 07:00). Costs 0.6bp RT (today's raw-spread level; generous for 2008-12),
       stress 1.2bp. No swap (intraday).
Split: TRAIN 2008-2016 | TEST 2017-2026.
Pass : TRAIN mean > 0, t >= 2.5; TEST mean > 0, t >= 1.5; TEST mean > 0 at 2x cost.

I1 Asian fade      02:00-06:45: an M15 close beyond the running Asian high/low (from 00:00)
                   -> fade; stop = extreme + 0.5 x range so far; target = range midpoint.
                   Variant: only when the Asian range so far < its 20-day median.
I2 London Judas    07:00-10:00: M15 trades beyond the Asian high/low and closes back inside
                   -> fade; stop = sweep extreme + 10% of Asian range;
                   target = opposite Asian extreme | 2R.
I3 London breakout 07:00-10:00: first M15 close beyond the Asian range -> follow;
                   stop = Asian midpoint; target = 1 Asian range from entry.
                   Variant: only when the Asian range < its 20-day median.
"""

import glob
import math

import numpy as np
import pandas as pd

COST, STRESS = 0.6, 1.2
TEST_YEAR = 2017

parts = [p for p in (pd.read_csv(f, index_col=0, parse_dates=True)
                     for f in sorted(glob.glob("data/duka/frxXAUUSD_M5_*.csv"))) if len(p)]
m5 = pd.concat(parts)
m5 = m5[~m5.index.duplicated()].sort_index()
m5 = m5[m5["volume"] > 0]
m5.index = m5.index.tz_convert("Europe/London")
m15 = m5.resample("15min", label="left", closed="left").agg(
    {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
print(f"M15 bars {len(m15)}: {m15.index[0]} -> {m15.index[-1]}")

m15["date"] = m15.index.date
m15["mins"] = m15.index.hour * 60 + m15.index.minute
days = {d: g for d, g in m15.groupby("date") if g.index[0].weekday() < 5}

asia = {}
for d, g in days.items():
    a = g[g["mins"] < 7 * 60]
    if len(a) >= 20:
        asia[d] = (a["high"].max(), a["low"].min())
ad = sorted(asia)
med20 = {}
for i, d in enumerate(ad):
    if i >= 20:
        med20[d] = np.median([asia[x][0] - asia[x][1] for x in ad[i - 20:i]])


def simulate(g, k, side, entry, stop, target, end_min):
    """Walk M15 bars after bar k; stop before target within a bar; time exit at end_min."""
    rows = g.iloc[k + 1:]
    for _, b in rows.iterrows():
        if b["mins"] >= end_min:
            return side * (b["open"] / entry - 1) * 1e4
        if (side > 0 and b["low"] <= stop) or (side < 0 and b["high"] >= stop):
            return side * (stop / entry - 1) * 1e4
        if (side > 0 and b["high"] >= target) or (side < 0 and b["low"] <= target):
            return side * (target / entry - 1) * 1e4
    last = rows.iloc[-1]["close"] if len(rows) else entry
    return side * (last / entry - 1) * 1e4


def i1(quiet_only):
    out = []
    for d, g in days.items():
        if quiet_only and d not in med20:
            continue
        hi = lo = None
        for k in range(len(g)):
            b = g.iloc[k]
            if b["mins"] >= 7 * 60:
                break
            if hi is not None and 2 * 60 <= b["mins"] < 6 * 60 + 45:
                width = hi - lo
                if quiet_only and width >= med20[d]:
                    pass
                elif width > 0 and (b["close"] > hi or b["close"] < lo):
                    side = -1 if b["close"] > hi else 1
                    ext = b["high"] if side < 0 else b["low"]
                    stop = ext - side * 0.5 * width
                    target = (hi + lo) / 2
                    out.append((d, side, simulate(g, k, side, b["close"], stop, target, 7 * 60)))
                    break
            hi = b["high"] if hi is None else max(hi, b["high"])
            lo = b["low"] if lo is None else min(lo, b["low"])
    return out


def i2(target_mode):
    out = []
    for d, g in days.items():
        if d not in asia:
            continue
        ah, al = asia[d]
        w = ah - al
        if w <= 0:
            continue
        swept_hi, swept_lo = -np.inf, np.inf
        for k in range(len(g)):
            b = g.iloc[k]
            if b["mins"] < 7 * 60:
                continue
            if b["mins"] >= 10 * 60:
                break
            swept_hi, swept_lo = max(swept_hi, b["high"]), min(swept_lo, b["low"])
            side = 0
            if b["high"] > ah and al < b["close"] < ah:
                side = -1
            elif b["low"] < al and al < b["close"] < ah:
                side = 1
            if side:
                entry = b["close"]
                stop = swept_hi + 0.1 * w if side < 0 else swept_lo - 0.1 * w
                risk = abs(entry - stop)
                target = (al if side < 0 else ah) if target_mode == "opp" else entry + side * 2 * risk
                out.append((d, side, simulate(g, k, side, entry, stop, target, 16 * 60)))
                break
    return out


def i3(quiet_only):
    out = []
    for d, g in days.items():
        if d not in asia or (quiet_only and d not in med20):
            continue
        ah, al = asia[d]
        w = ah - al
        if w <= 0 or (quiet_only and w >= med20[d]):
            continue
        for k in range(len(g)):
            b = g.iloc[k]
            if b["mins"] < 7 * 60:
                continue
            if b["mins"] >= 10 * 60:
                break
            if b["close"] > ah or b["close"] < al:
                side = 1 if b["close"] > ah else -1
                entry = b["close"]
                out.append((d, side, simulate(g, k, side, entry, (ah + al) / 2, entry + side * w, 16 * 60)))
                break
    return out


def stats(tr, cost):
    if len(tr) < 20:
        return None
    net = np.array([x[2] for x in tr]) - cost
    yrs = len({x[0].year for x in tr})
    sd = net.std(ddof=1)
    return dict(n=len(net), tpy=len(net) / max(yrs, 1), mean=net.mean(),
                t=net.mean() / sd * math.sqrt(len(net)) if sd > 0 else 0, wr=(net > 0).mean() * 100)


def fmt(s):
    return "n<20" if not s else (f"n={s['n']:4d} /yr={s['tpy']:4.0f} mean={s['mean']:+6.2f}bp "
                                 f"t={s['t']:+5.2f} WR={s['wr']:4.1f}%")


VARIANTS = {
    "I1 Asian fade (all)": lambda: i1(False),
    "I1 Asian fade (quiet Asia)": lambda: i1(True),
    "I2 London Judas -> opposite side": lambda: i2("opp"),
    "I2 London Judas -> 2R": lambda: i2("2r"),
    "I3 London breakout (all)": lambda: i3(False),
    "I3 London breakout (compressed Asia)": lambda: i3(True),
}

print("\nPass: TRAIN t>=2.5 & mean>0, TEST t>=1.5 & mean>0, TEST mean>0 at 2x cost\n")
for name, fn in VARIANTS.items():
    tr = fn()
    a = [x for x in tr if x[0].year < TEST_YEAR]
    b = [x for x in tr if x[0].year >= TEST_YEAR]
    sa, sb, sb2 = stats(a, COST), stats(b, COST), stats(b, STRESS)
    ok = sa and sb and sb2 and sa["mean"] > 0 and sa["t"] >= 2.5 and sb["mean"] > 0 and sb["t"] >= 1.5 and sb2["mean"] > 0
    print(f"[{'PASS' if ok else 'fail'}] {name}")
    print(f"   TRAIN {fmt(sa)}")
    print(f"   TEST  {fmt(sb)}")
    for side, lab in ((1, "long "), (-1, "short")):
        print(f"     {lab} train {fmt(stats([x for x in a if x[1] == side], COST))} | "
              f"test {fmt(stats([x for x in b if x[1] == side], COST))}")
    yr = {}
    for x in tr:
        yr.setdefault(x[0].year, []).append(x[2] - COST)
    print("   years: " + " ".join(f"{y % 100:02d}:{np.mean(v):+.1f}" for y, v in sorted(yr.items())), flush=True)
