"""
Crude oil around the weekly EIA inventory report (pre-registered 2026-11-13).

Release: Wednesday 10:30 New York time; in weeks with a US federal holiday on Mon-Wed,
Thursday 11:00 (EIA practice). The script first checks the chosen minutes really carry the
release volatility spike (vs the same clock on other days).

E1 pre-report drift : 09:00 ET -> release, direction = sign of the TRAIN mean (1 variant)
E2 reaction follow  : side = sign of the first 5 min after release; enter at release+5 min,
                      exit at +30 / +60 / +120 min / 14:25 ET (4 variants)
E3 big reaction     : as E2, only when |first 5 min| > 2x the median of the previous 20
                      releases (4 variants)
Placebo             : E2 rules at 10:30 on Tuesdays and Thursdays without a release.
Data: Dukascopy M5 bid, WTI (XTIUSD) and Brent (XBRUSD), 2012-2026.
Costs RT: WTI 3bp (stress 6), Brent 1.5bp (stress 3) — above today's IC medians
(2.26 / 1.02bp) because spreads widen at the release.
Split: TRAIN 2012-2018 | TEST 2019-2026.
Pass: TRAIN mean > 0, t >= 2.5; TEST mean > 0, t >= 1.5, and > 0 at stress cost.
"""

import glob
import math

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

COST = {"XTIUSD": (3.0, 6.0), "XBRUSD": (1.5, 3.0)}
TEST_YEAR = 2019


def load(prefix):
    parts = [p for p in (pd.read_csv(f, index_col=0, parse_dates=True)
                         for f in sorted(glob.glob(f"data/duka/{prefix}_M5_*.csv"))) if len(p)]
    df = pd.concat(parts)
    df = df[~df.index.duplicated()].sort_index()
    df.index = df.index.tz_convert("America/New_York").tz_localize(None)
    return df


def release_times(index):
    hol = set(USFederalHolidayCalendar().holidays(index.min(), index.max()).date)
    days = sorted(set(index.date))
    out = []
    for d in days:
        ts = pd.Timestamp(d)
        if ts.weekday() != 2:
            continue
        week = [(ts - pd.Timedelta(days=k)).date() for k in (2, 1, 0)]       # Mon, Tue, Wed
        if any(w in hol for w in week):
            out.append(ts + pd.Timedelta(days=1, hours=11))
        else:
            out.append(ts + pd.Timedelta(hours=10, minutes=30))
    return out


def price_at(df, t, col="open"):
    """Price of the first bar starting at or after t, if it starts within 10 minutes."""
    i = df.index.searchsorted(t)
    if i >= len(df) or df.index[i] - t > pd.Timedelta(minutes=10):
        return None
    return df.iloc[i][col]


def events(df, times):
    """Per release: pre-drift, first-5-min move, and follow-through returns (all bp, side-free)."""
    rows = []
    for t in times:
        p0 = price_at(df, t)
        p5 = price_at(df, t + pd.Timedelta(minutes=5))
        pre = price_at(df, t.normalize() + pd.Timedelta(hours=9))
        if p0 is None or p5 is None:
            continue
        row = {"t": t, "pre": (p0 / pre - 1) * 1e4 if pre else np.nan, "first5": (p5 / p0 - 1) * 1e4}
        for lab, delta in (("h30", pd.Timedelta(minutes=30)), ("h60", pd.Timedelta(minutes=60)),
                           ("h120", pd.Timedelta(minutes=120))):
            px = price_at(df, t + delta)
            row[lab] = (px / p5 - 1) * 1e4 if px else np.nan
        px = price_at(df, t.normalize() + pd.Timedelta(hours=14, minutes=25))
        row["settle"] = (px / p5 - 1) * 1e4 if px and t.hour < 14 else np.nan
        rows.append(row)
    ev = pd.DataFrame(rows).set_index("t")
    ev["big"] = ev["first5"].abs() > 2 * ev["first5"].abs().rolling(20, min_periods=10).median().shift(1)
    return ev


def stats(x):
    x = pd.Series(x).dropna()
    if len(x) < 30:
        return None
    sd = x.std(ddof=1)
    return dict(n=len(x), mean=x.mean(), t=x.mean() / sd * math.sqrt(len(x)) if sd > 0 else 0,
                wr=(x > 0).mean() * 100)


def fmt(s):
    return "n<30" if not s else f"n={s['n']:4d} {s['mean']:+6.2f}bp t={s['t']:+5.2f} WR={s['wr']:4.1f}%"


def verdict(name, gross, cost, stress, years):
    tr, te = gross[years < TEST_YEAR], gross[years >= TEST_YEAR]
    a, b, b2 = stats(tr - cost), stats(te - cost), stats(te - stress)
    ok = bool(a and b and b2 and a["mean"] > 0 and a["t"] >= 2.5 and b["mean"] > 0 and b["t"] >= 1.5 and b2["mean"] > 0)
    print(f"[{'PASS' if ok else 'fail'}] {name}\n   TRAIN {fmt(a)}\n   TEST  {fmt(b)}  | stress {fmt(b2)}")
    yr = pd.Series(gross.values - cost, index=years).groupby(level=0).mean()
    print("   years: " + " ".join(f"{y % 100:02d}:{v:+.1f}" for y, v in yr.items()))


for sym in ("XTIUSD", "XBRUSD"):
    df = load(sym)
    cost, stress = COST[sym]
    rel = release_times(df.index)
    ev = events(df, rel)
    print(f"\n================ {sym}: {len(ev)} releases {ev.index[0]:%Y-%m-%d} -> {ev.index[-1]:%Y-%m-%d} ================")

    # sanity: release minutes must show the volatility spike
    plc_times = [t - pd.Timedelta(days=1) for t in rel] + [t + pd.Timedelta(days=1) for t in rel if t.weekday() == 2]
    plc_times = [t.normalize() + pd.Timedelta(hours=10, minutes=30) for t in plc_times]
    plc = events(df, plc_times)
    print(f"median |first 5 min|: release {ev['first5'].abs().median():.1f}bp vs placebo days "
          f"{plc['first5'].abs().median():.1f}bp  (release should be clearly larger)")

    years = ev.index.year.values
    # E1 pre-drift, direction from TRAIN
    pre = ev["pre"]
    sign = 1 if pre[years < TEST_YEAR].mean() > 0 else -1
    verdict(f"E1 pre-report drift ({'long' if sign > 0 else 'short'} 09:00 -> release)", sign * pre, cost, stress, years)
    side = np.sign(ev["first5"])
    for lab in ("h30", "h60", "h120", "settle"):
        verdict(f"E2 follow first 5 min, exit {lab}", side * ev[lab], cost, stress, years)
    for lab in ("h30", "h60", "h120", "settle"):
        m = ev["big"].fillna(False).values
        verdict(f"E3 big reaction only, exit {lab}", (side * ev[lab])[m], cost, stress, years[m])
    print("--- placebo (non-report days, same rules) ---")
    py = plc.index.year.values
    for lab in ("h30", "h60", "h120"):
        x = np.sign(plc["first5"]) * plc[lab] - cost
        print(f"   placebo follow, exit {lab}: {fmt(stats(x))}")
