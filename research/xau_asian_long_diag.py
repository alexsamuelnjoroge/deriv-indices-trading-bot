"""
Robustness checks for C4 "long gold in Asian hours" (post-hoc inspection, nothing re-selected).
  1. delayed entries (01:00 server is the daily reopen, when spreads are widest)
  2. exit-time surface
  3. control: the rest of the day (exit -> 23:00 server) should be weak/negative
  4. IC Markets MT5 M15 data 2022-2026 (independent source, real broker feed)
  5. tail risk: worst days, longest losing run, max drawdown in bp, and a 1.5% disaster stop
Also C2 (prior-day false break, M15) on the MT5 data.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xau_sources_battery as sb  # noqa: E402

COST = 0.36
m15 = sb.m15


def window(frame, t0, t1, stop_pct=None):
    """Long from the open of server minute t0 to the open of minute t1, same server day."""
    out = []
    for d, g in sb.days(frame):
        a = g[g["mins"] >= t0]
        if len(a) == 0 or a.iloc[0]["mins"] != t0:
            continue
        b = a[a["mins"] >= t1]
        if len(b) == 0 or b.iloc[0]["mins"] != t1:
            continue
        entry = a.iloc[0]["open"]
        px = b.iloc[0]["open"]
        if stop_pct:
            held = a[a["mins"] < t1]
            if held["low"].min() <= entry * (1 - stop_pct):
                px = entry * (1 - stop_pct)
        out.append((d, (px / entry - 1) * 1e4))
    return out


def summ(tr):
    net = np.array([x[1] for x in tr]) - COST
    if len(net) < 20:
        return "n<20"
    t = net.mean() / net.std(ddof=1) * np.sqrt(len(net))
    return f"n={len(net):4d} {net.mean():+6.2f}bp t={t:+5.2f} WR={(net > 0).mean() * 100:4.1f}%"


def split(tr):
    return (summ([x for x in tr if x[0].year < 2017]), summ([x for x in tr if x[0].year >= 2017]))


print("1) Delayed entries, exit 09:00 server   [TRAIN 2008-16 | TEST 2017-26]")
for t0 in (60, 75, 90, 120, 180):
    a, b = split(window(m15, t0, 9 * 60))
    print(f"   entry {t0 // 60:02d}:{t0 % 60:02d}  {a}  |  {b}")

print("\n2) Exit-time surface, entry 01:15 server")
for t1 in (3, 5, 7, 9, 11, 13):
    a, b = split(window(m15, 75, t1 * 60))
    print(f"   exit {t1:02d}:00  {a}  |  {b}")

print("\n3) Control: rest of the day, long 09:00 -> 23:00 server")
a, b = split(window(m15, 9 * 60, 23 * 60))
print(f"   {a}  |  {b}")

print("\n4) IC Markets MT5 M15, 2022-07 -> 2026-10")
d = pd.read_csv("data/pro/mt5_XAUUSD_M15.csv")
mt = pd.DataFrame({"open": d.open.values, "high": d.high.values, "low": d.low.values, "close": d.close.values},
                  index=pd.to_datetime(d.epoch, unit="s"))
mt["mins"] = mt.index.hour * 60 + mt.index.minute
for t0, t1 in ((75, 9 * 60), (75, 7 * 60), (120, 9 * 60)):
    tr = window(mt, t0, t1)
    print(f"   {t0 // 60:02d}:{t0 % 60:02d}->{t1 // 60:02d}:00  all {summ(tr)}")
    print("      years: " + " ".join(f"{y}:{np.mean([x[1] - COST for x in tr if x[0].year == y]):+.1f}"
                                     for y in sorted({x[0].year for x in tr})))
duka_same = [x for x in window(m15, 75, 9 * 60) if x[0] >= mt.index[0].date()]
print(f"   Dukascopy, same span, 01:15->09:00: {summ(duka_same)}")
c2 = sb.c2(mt)
net = np.array([x[2] for x in c2]) - COST
print(f"   C2b prior-day false break on MT5: n={len(net)} {net.mean():+.2f}bp "
      f"t={net.mean() / net.std(ddof=1) * np.sqrt(len(net)):+.2f}")

print("\n5) Tail risk, 01:15 -> 09:00 server, 2008-2026")
tr = window(m15, 75, 9 * 60)
net = np.array([x[1] for x in tr]) - COST
worst = sorted(tr, key=lambda x: x[1])[:5]
print("   worst days: " + ", ".join(f"{dd} {r:+.0f}bp" for dd, r in worst))
eq = np.cumsum(net)
dd = (eq - np.maximum.accumulate(eq)).min()
runs, cur = 0, 0
for v in net:
    cur = cur + 1 if v < 0 else 0
    runs = max(runs, cur)
print(f"   total {eq[-1]:+.0f}bp over {len(net)} days | max drawdown {dd:.0f}bp | longest losing run {runs} days")
for sp in (0.01, 0.015, 0.02):
    a, b = split(window(m15, 75, 9 * 60, stop_pct=sp))
    print(f"   with {sp * 100:.1f}% disaster stop: {a}  |  {b}")
