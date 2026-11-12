"""
Test of the session range-breakout EA described at mql5.com/en/blogs/post/767519 (2026-11-11).

Published rules: XAUUSD M15; measure the high/low of the 03:00-08:00 server-time window
("conservative") or 04:00-09:00 ("aggressive") and place pending stop orders above and below
the range; optional close before the session ends. SL/TP are not published, so two common
choices are tested (fixed before running):
  (a) SL = opposite side of the range, TP = 1R
  (b) SL = range midpoint,             TP = 2R
OCO: the first order to fill cancels the other. Orders live until 22:00 server; open trades
close at 23:00 server. Server time = New York + 7h (IC Markets convention).
Same-bar ambiguity is booked against us: if both orders trigger in one M15 bar, a stop-out;
if stop and target are both inside one bar, the stop. Costs 0.36bp RT measured live
(stress 0.72bp). Split: TRAIN 2008-2016 | TEST 2017-2026.
Pass: TRAIN mean > 0 with t >= 2.5; TEST mean > 0 with t >= 1.5 and > 0 at 2x cost.
"""

import glob
import math

import numpy as np
import pandas as pd

COST, STRESS = 0.36, 0.72
TEST_YEAR = 2017

parts = [p for p in (pd.read_csv(f, index_col=0, parse_dates=True)
                     for f in sorted(glob.glob("data/duka/frxXAUUSD_M5_*.csv"))) if len(p)]
m5 = pd.concat(parts)
m5 = m5[~m5.index.duplicated()].sort_index()
m5 = m5[m5["volume"] > 0]
ny = m5.index.tz_convert("America/New_York")
m5.index = (ny + pd.Timedelta(hours=7)).tz_localize(None)        # server wall-clock time
m15 = m5.resample("15min", label="left", closed="left").agg(
    {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
m15["mins"] = m15.index.hour * 60 + m15.index.minute
print(f"M15 bars {len(m15)}: {m15.index[0]} -> {m15.index[-1]} (server time)")


def walk(after, k, side, entry, stop, target):
    """Exit price walking M15 bars after bar k: stop before target; flat at 23:00 server."""
    for j in range(k + 1, len(after)):
        c = after.iloc[j]
        if c["mins"] >= 23 * 60:
            return c["open"]
        if (side > 0 and c["low"] <= stop) or (side < 0 and c["high"] >= stop):
            return stop
        if (side > 0 and c["high"] >= target) or (side < 0 and c["low"] <= target):
            return target
    return after.iloc[-1]["close"]


def placebo(r0, r1, seed):
    """Same day, same bracket size (range width), but a coin-flip direction entered at the
    open of the first bar after the range window."""
    rs = np.random.default_rng(seed)
    out = []
    for d, g in m15.groupby(m15.index.date):
        if g.index[0].weekday() >= 5:
            continue
        rng = g[(g["mins"] >= r0 * 60) & (g["mins"] < r1 * 60)]
        if len(rng) < (r1 - r0) * 4 - 2:
            continue
        w = rng["high"].max() - rng["low"].min()
        after = g[g["mins"] >= r1 * 60]
        if w <= 0 or len(after) < 2:
            continue
        side = 1 if rs.random() < 0.5 else -1
        entry = after.iloc[0]["open"]
        res = walk(after, -1, side, entry, entry - side * w, entry + side * w)
        out.append((d, side, side * (res / entry - 1) * 1e4, w / entry * 1e4))
    return out


def run(r0, r1, mode, tp_r=None):
    """Returns (date, side, gross bp, risk bp). tp_r overrides the TP multiple."""
    out = []
    for d, g in m15.groupby(m15.index.date):
        if g.index[0].weekday() >= 5:
            continue
        rng = g[(g["mins"] >= r0 * 60) & (g["mins"] < r1 * 60)]
        if len(rng) < (r1 - r0) * 4 - 2:
            continue
        hi, lo = rng["high"].max(), rng["low"].min()
        w = hi - lo
        if w <= 0:
            continue
        after = g[g["mins"] >= r1 * 60]
        side = 0
        for k in range(len(after)):
            b = after.iloc[k]
            if b["mins"] >= 22 * 60:
                break
            up, dn = b["high"] > hi, b["low"] < lo
            if up and dn:
                entry = hi if (hi - b["open"]) <= (b["open"] - lo) else lo
                sd = 1 if entry == hi else -1
                stop = (lo if sd > 0 else hi) if mode == "a" else (hi + lo) / 2
                out.append((d, sd, sd * (stop / entry - 1) * 1e4, abs(entry - stop) / entry * 1e4))
                break
            if up or dn:
                side = 1 if up else -1
                entry = max(b["open"], hi) if up else min(b["open"], lo)
                stop = (lo if up else hi) if mode == "a" else (hi + lo) / 2
                mult = tp_r if tp_r is not None else (1.0 if mode == "a" else 2.0)
                target = entry + side * mult * abs(entry - stop)
                if (side > 0 and b["low"] <= stop) or (side < 0 and b["high"] >= stop):
                    res = stop
                else:
                    res = walk(after, k, side, entry, stop, target)
                out.append((d, side, side * (res / entry - 1) * 1e4, abs(entry - stop) / entry * 1e4))
                break
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


def main():
    for (r0, r1, lab) in ((3, 8, "conservative 03-08"), (4, 9, "aggressive 04-09")):
        for mode, desc in (("a", "SL opposite side, TP 1R"), ("b", "SL midpoint, TP 2R")):
            tr = run(r0, r1, mode)
            a = [x for x in tr if x[0].year < TEST_YEAR]
            b = [x for x in tr if x[0].year >= TEST_YEAR]
            sa, sb, sb2 = stats(a, COST), stats(b, COST), stats(b, STRESS)
            ok = sa and sb and sb2 and sa["mean"] > 0 and sa["t"] >= 2.5 and sb["mean"] > 0 and sb["t"] >= 1.5 and sb2["mean"] > 0
            print(f"\n[{'PASS' if ok else 'fail'}] {lab} | {desc}")
            print(f"   TRAIN {fmt(sa)}")
            print(f"   TEST  {fmt(sb)}")
            yr = {}
            for x in tr:
                yr.setdefault(x[0].year, []).append(x[2] - COST)
            print("   years: " + " ".join(f"{y % 100:02d}:{np.mean(v):+.1f}" for y, v in sorted(yr.items())))
            y25 = [x for x in tr if x[0].year == 2025]
            if y25:
                print(f"   2025 only (the EA's backtest year): {fmt(stats(y25, COST))}")


if __name__ == "__main__":
    main()
