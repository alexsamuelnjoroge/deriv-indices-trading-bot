"""
USDJPY Tokyo fix / Gotobi effect (pre-registered 2026-11-16).

Background (Ito & Yamada, "Puzzles in the Tokyo fixing in the forex market"): Japanese
importers buy USD at the 09:55 JST Tokyo fix, especially on Gotobi days (5th, 10th, 15th,
20th, 25th, month-end; shifted to the previous business day when on a weekend/holiday),
pushing USDJPY up into the fix and letting it ease afterwards.

Variants (8), each on Gotobi days and on other business days separately:
  G1 long into the fix : buy 08:00 or 09:00 JST, sell 09:55 JST
  G2 fade after the fix: sell 09:55 JST, buy back 11:00 or 12:00 JST
Data: Dukascopy M5 bid 2010-2026 (UTC -> Asia/Tokyo; JST has no DST).
Japanese holidays: fixed-date national holidays + Happy-Monday holidays + equinoxes
approximated as Mar 20 / Sep 23 (a day off in some years; affects a handful of Gotobi shifts).
Cost 0.86bp RT (IC MT5 measured: spread 0.06 + commission 0.8), stress 1.72bp.
Split TRAIN 2010-2016 | TEST 2017-2026.
Pass: TRAIN mean > 0, t >= 2.5; TEST mean > 0, t >= 1.5, and > 0 at stress cost.
"""

import glob
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

COST, STRESS, TEST_YEAR = 0.86, 1.72, 2017


def nth_monday(y, m, n):
    d = date(y, m, 1)
    d += timedelta(days=(7 - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def jp_holidays(y):
    fixed = [(1, 1), (1, 2), (1, 3), (2, 11), (3, 20), (4, 29), (5, 3), (5, 4), (5, 5),
             (9, 23), (11, 3), (11, 23), (12, 31)]
    if y >= 2020:
        fixed.append((2, 23))
    elif y <= 2018:
        fixed.append((12, 23))
    h = {date(y, m, d) for m, d in fixed}
    h |= {nth_monday(y, 1, 2), nth_monday(y, 7, 3), nth_monday(y, 9, 3), nth_monday(y, 10, 2)}
    for d in list(h):                       # substitute holiday: Sunday holiday -> Monday off
        if d.weekday() == 6:
            h.add(d + timedelta(days=1))
    return h


def business_days(y0, y1):
    hol = set().union(*(jp_holidays(y) for y in range(y0, y1 + 1)))
    d, out = date(y0, 1, 1), []
    while d <= date(y1, 12, 31):
        if d.weekday() < 5 and d not in hol:
            out.append(d)
        d += timedelta(days=1)
    return out


def gotobi_days(bdays):
    bset = sorted(bdays)
    out = set()
    months = sorted({(d.year, d.month) for d in bset})
    for y, m in months:
        last = (date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1)).day
        for dom in (5, 10, 15, 20, 25, last):
            t = date(y, m, min(dom, last))
            prior = [b for b in bset if b <= t and (t - b).days < 7]
            if prior:
                out.add(prior[-1])
    return out


def load():
    parts = [p for p in (pd.read_csv(f, index_col=0, parse_dates=True)
                         for f in sorted(glob.glob("data/duka/frxUSDJPY_M5_*.csv"))) if len(p)]
    df = pd.concat(parts)
    df = df[~df.index.duplicated()].sort_index()
    df.index = df.index.tz_convert("Asia/Tokyo").tz_localize(None)
    return df


def price_at(df, t):
    i = df.index.searchsorted(t)
    if i >= len(df) or df.index[i] - t > pd.Timedelta(minutes=10):
        return None
    return df.iloc[i]["open"]


def window_returns(df, days, h0, m0, h1, m1, side):
    out = []
    for d in days:
        base = pd.Timestamp(d)
        a = price_at(df, base + pd.Timedelta(hours=h0, minutes=m0))
        b = price_at(df, base + pd.Timedelta(hours=h1, minutes=m1))
        if a and b:
            out.append((d, side * (b / a - 1) * 1e4))
    return out


def stats(x, cost):
    x = np.array([v for _, v in x]) - cost
    if len(x) < 30:
        return None
    sd = x.std(ddof=1)
    return dict(n=len(x), mean=x.mean(), t=x.mean() / sd * math.sqrt(len(x)) if sd > 0 else 0,
                wr=(x > 0).mean() * 100)


def fmt(s):
    return "n<30" if not s else f"n={s['n']:4d} {s['mean']:+6.2f}bp t={s['t']:+5.2f} WR={s['wr']:4.1f}%"


if __name__ == "__main__":
    df = load()
    bdays = [d for d in business_days(2010, 2026) if pd.Timestamp(d) <= df.index[-1]]
    goto = gotobi_days(bdays)
    groups = {"Gotobi": [d for d in bdays if d in goto], "other": [d for d in bdays if d not in goto]}
    print(f"USDJPY M5 {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d} JST | business days {len(bdays)}, "
          f"Gotobi {len(groups['Gotobi'])}")

    variants = [
        ("G1 long 08:00 -> 09:55", 8, 0, 9, 55, 1),
        ("G1 long 09:00 -> 09:55", 9, 0, 9, 55, 1),
        ("G2 fade 09:55 -> 11:00", 9, 55, 11, 0, -1),
        ("G2 fade 09:55 -> 12:00", 9, 55, 12, 0, -1),
    ]
    for name, h0, m0, h1, m1, side in variants:
        for gname, days in groups.items():
            tr = window_returns(df, days, h0, m0, h1, m1, side)
            a = [x for x in tr if x[0].year < TEST_YEAR]
            b = [x for x in tr if x[0].year >= TEST_YEAR]
            sa, sb, sb2 = stats(a, COST), stats(b, COST), stats(b, STRESS)
            ok = bool(sa and sb and sb2 and sa["mean"] > 0 and sa["t"] >= 2.5 and sb["mean"] > 0
                      and sb["t"] >= 1.5 and sb2["mean"] > 0)
            print(f"\n[{'PASS' if ok else 'fail'}] {name} | {gname} days")
            print(f"   TRAIN {fmt(sa)}\n   TEST  {fmt(sb)}  | stress {fmt(sb2)}")
            gross = [(d, v) for d, v in tr]
            print(f"   gross (no cost) all years: {fmt(stats(gross, 0))}")
            yr = pd.Series([v - COST for _, v in tr], index=[d.year for d, _ in tr]).groupby(level=0).mean()
            print("   years: " + " ".join(f"{y % 100:02d}:{v:+.1f}" for y, v in yr.items()), flush=True)
