"""
XAUUSD professional playbook — DAILY setups (pre-registered 2026-11-10).

Each setup is a mechanical version of a well-known discretionary gold swing setup.
At most two variants per setup; long and short legs reported separately.

Data : MT5 IC Markets XAUUSD D1, 1998 -> 2026 (data/pro/mt5_XAUUSD_D1.csv, server-day candles).
Costs: 0.6bp round trip + long swap 1.3bp per calendar night (today's IC Markets rate,
       conservative for the low-rate 2009-21 years); short swap credit ignored.
Split: TRAIN 1998-2011 | TEST 2012-2026 (includes the 2013-15 bear market).
Pass : TRAIN mean > 0 with t >= 2.5 (~10 variants tested), TEST mean > 0 with t >= 1.5,
       TEST mean still > 0 at 2x costs.

S1 RSI2 pullback  : close > SMA200 and RSI(2) < 10 (5) -> buy next open; exit at the first
                    close > SMA5 or after 10 days. Mirror for shorts (< SMA200, RSI2 > 90 (95)).
S2 EMA20 retest   : EMA20 > EMA50 and close > EMA50; low <= EMA20 < close -> buy next open;
                    stop 1.5 ATR, target 3 ATR (2R), max 15 days. Mirror for shorts.
S3 NR7 breakout   : today's range is the narrowest of 7 -> next day buy-stop at its high /
                    sell-stop at its low; stop at the other side; exit close of day 3.
                    Variant: only in the direction of the EMA50 slope.
S4 Expansion day  : range > 2 ATR14 and close in the outer 20% of the range -> next day
                    FOLLOW or FADE, open -> close of day 1 / day 3.
"""

import csv
import math
from datetime import datetime, timezone

import numpy as np

COST, STRESS = 0.6, 1.2
LONG_SWAP_BP = 1.3
TRAIN_END = datetime(2012, 1, 1, tzinfo=timezone.utc)

rows = list(csv.DictReader(open("data/pro/mt5_XAUUSD_D1.csv")))
ep = np.array([int(r["epoch"]) for r in rows])
o = np.array([float(r["open"]) for r in rows])
h = np.array([float(r["high"]) for r in rows])
l = np.array([float(r["low"]) for r in rows])
c = np.array([float(r["close"]) for r in rows])
dates = [datetime.fromtimestamp(e, timezone.utc) for e in ep]
N = len(c)
print(f"D1 bars {N}: {dates[0]:%Y-%m-%d} -> {dates[-1]:%Y-%m-%d}")


def sma(x, n):
    out = np.full(len(x), np.nan)
    cs = np.cumsum(np.r_[0, x])
    out[n - 1:] = (cs[n:] - cs[:-n]) / n
    return out


def ema(x, n):
    out = np.empty(len(x))
    a = 2 / (n + 1)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def rsi(x, n):
    d = np.diff(x)
    up, dn = np.maximum(d, 0), np.maximum(-d, 0)
    out = np.full(len(x), np.nan)
    au, ad = up[:n].mean(), dn[:n].mean()
    for i in range(n, len(d)):
        au = (au * (n - 1) + up[i]) / n
        ad = (ad * (n - 1) + dn[i]) / n
        out[i + 1] = 100 if ad == 0 else 100 - 100 / (1 + au / ad)
    return out


tr = np.r_[h[0] - l[0], np.maximum(h[1:], c[:-1]) - np.minimum(l[1:], c[:-1])]
atr = ema(tr, 27)                     # Wilder 14 == EMA(27)
sma200, sma5 = sma(c, 200), sma(c, 5)
e20, e50 = ema(c, 20), ema(c, 50)
r2 = rsi(c, 2)
rng = h - l


def trade(side, i_entry, px_entry, j_exit, px_exit):
    gross = side * (px_exit / px_entry - 1) * 1e4
    nights = (dates[j_exit] - dates[i_entry]).days
    swap = LONG_SWAP_BP * nights if side > 0 else 0.0
    return dict(i=i_entry, side=side, gross=gross, swap=swap, days=nights)


def run_s1(lo, hi):
    out, i = [], 201
    while i < N - 1:
        side = 1 if (c[i] > sma200[i] and r2[i] < lo) else -1 if (c[i] < sma200[i] and r2[i] > hi) else 0
        if side == 0:
            i += 1
            continue
        k = i + 1
        j = k
        while j < min(k + 10, N - 1):
            if (side > 0 and c[j] > sma5[j]) or (side < 0 and c[j] < sma5[j]):
                break
            j += 1
        out.append(trade(side, k, o[k], j, c[j]))
        i = j + 1
    return out


def stop_target_exit(side, k, entry, stop, target, max_days):
    for j in range(k, min(k + max_days, N)):
        if (side > 0 and l[j] <= stop) or (side < 0 and h[j] >= stop):
            return j, stop
        if (side > 0 and h[j] >= target) or (side < 0 and l[j] <= target):
            return j, target
    j = min(k + max_days, N) - 1
    return j, c[j]


def run_s2():
    out, i = [], 60
    while i < N - 1:
        side = 0
        if e20[i] > e50[i] and c[i] > e50[i] and l[i] <= e20[i] < c[i]:
            side = 1
        elif e20[i] < e50[i] and c[i] < e50[i] and h[i] >= e20[i] > c[i]:
            side = -1
        if side == 0:
            i += 1
            continue
        k = i + 1
        a = atr[i]
        j, px = stop_target_exit(side, k, o[k], o[k] - side * 1.5 * a, o[k] + side * 3.0 * a, 15)
        out.append(trade(side, k, o[k], j, px))
        i = j + 1
    return out


def run_s3(with_trend):
    out, i = [], 60
    while i < N - 4:
        if rng[i] > rng[i - 6:i].min():
            i += 1
            continue
        k = i + 1
        slope = e50[i] - e50[i - 5]
        long_ok = (not with_trend) or slope > 0
        short_ok = (not with_trend) or slope < 0
        hit_up, hit_dn = h[k] > h[i] and long_ok, l[k] < l[i] and short_ok
        if hit_up and hit_dn:
            # Both stops hit the same day: one filled first and the other side is its stop,
            # so book a full stop-out on the side nearer the open (filled first).
            side = 1 if (h[i] - o[k]) <= (o[k] - l[i]) else -1
            entry = max(o[k], h[i]) if side > 0 else min(o[k], l[i])
            out.append(trade(side, k, entry, k, l[i] if side > 0 else h[i]))
            i = k + 1
            continue
        if not (hit_up or hit_dn):
            i = k
            continue
        side = 1 if hit_up else -1
        entry = max(o[k], h[i]) if side > 0 else min(o[k], l[i])
        stop = l[i] if side > 0 else h[i]
        j, px = k, None
        for j in range(k, min(k + 3, N)):
            if (side > 0 and l[j] <= stop) or (side < 0 and h[j] >= stop):
                px = stop
                break
        if px is None:
            px = c[j]
        out.append(trade(side, k, entry, j, px))
        i = j + 1
    return out


def run_s4(mode, hold):
    out, i = [], 30
    while i < N - hold - 1:
        if rng[i] <= 2 * atr[i - 1]:
            i += 1
            continue
        pos = (c[i] - l[i]) / rng[i]
        d = 1 if pos >= 0.8 else -1 if pos <= 0.2 else 0
        if d == 0:
            i += 1
            continue
        side = d if mode == "follow" else -d
        k = i + 1
        j = k + hold - 1
        out.append(trade(side, k, o[k], j, c[j]))
        i = j + 1
    return out


VARIANTS = {
    "S1 RSI2<10/>90 pullback": lambda: run_s1(10, 90),
    "S1 RSI2<5/>95 pullback": lambda: run_s1(5, 95),
    "S2 EMA20 retest 2R": run_s2,
    "S3 NR7 breakout": lambda: run_s3(False),
    "S3 NR7 breakout w/ trend": lambda: run_s3(True),
    "S4 expansion FOLLOW 1d": lambda: run_s4("follow", 1),
    "S4 expansion FOLLOW 3d": lambda: run_s4("follow", 3),
    "S4 expansion FADE 1d": lambda: run_s4("fade", 1),
    "S4 expansion FADE 3d": lambda: run_s4("fade", 3),
}


def stats(ts, cost):
    if len(ts) < 10:
        return None
    net = np.array([t["gross"] - t["swap"] - cost for t in ts])
    sd = net.std(ddof=1)
    yrs = (dates[ts[-1]["i"]] - dates[ts[0]["i"]]).days / 365.25
    return dict(n=len(ts), tpy=len(ts) / max(yrs, 0.5), mean=net.mean(),
                t=net.mean() / sd * math.sqrt(len(ts)) if sd > 0 else 0,
                wr=(net > 0).mean() * 100, days=np.mean([t["days"] for t in ts]))


def fmt(s):
    if not s:
        return "n<10"
    return (f"n={s['n']:4d} /yr={s['tpy']:4.1f} mean={s['mean']:+6.1f}bp t={s['t']:+5.2f} "
            f"WR={s['wr']:4.1f}% hold={s['days']:.1f}d")


print("\nPass: TRAIN t>=2.5 & mean>0, TEST t>=1.5 & mean>0, TEST mean>0 at 2x cost\n")
for name, fn in VARIANTS.items():
    ts = fn()
    tr_, te_ = [t for t in ts if dates[t["i"]] < TRAIN_END], [t for t in ts if dates[t["i"]] >= TRAIN_END]
    a, b, b2 = stats(tr_, COST), stats(te_, COST), stats(te_, STRESS)
    ok = (a and b and b2 and a["mean"] > 0 and a["t"] >= 2.5 and b["mean"] > 0 and b["t"] >= 1.5 and b2["mean"] > 0)
    print(f"[{'PASS' if ok else 'fail'}] {name}")
    print(f"   TRAIN {fmt(a)}")
    print(f"   TEST  {fmt(b)}")
    for side, lab in ((1, "long "), (-1, "short")):
        sa = stats([t for t in tr_ if t["side"] == side], COST)
        sb = stats([t for t in te_ if t["side"] == side], COST)
        print(f"     {lab} train {fmt(sa)}")
        print(f"     {lab} test  {fmt(sb)}")
    yr = {}
    for t in te_:
        yr.setdefault(dates[t["i"]].year, []).append(t["gross"] - t["swap"] - COST)
    print("   TEST years: " + " ".join(f"{y % 100:02d}:{np.mean(v):+.0f}" for y, v in sorted(yr.items())))
