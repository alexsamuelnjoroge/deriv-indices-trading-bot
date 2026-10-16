"""
Step 2 of XAUUSD research: more structural effects + conditioning of the
daily-break drift found in step 1.

Hourly split (MT5 H1 is complete only from mid-2016):
  DEV 2016-07 -> 2020 | VAL 2021 -> 2024-07 | RECENT 2024-08 -> now
Daily effects use D1 from 2010 with DEV 2010-2017 | VAL 2018 -> 2024-07.
Candidate ('<<<') = same sign and |t| >= 2 in both DEV and VAL.
Values in bp. Times are MT5 broker time (daily break = 00:00-01:00).

Usage:
    python pro_bot/research_xau_edges2.py
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "pro"
H_PER = {"DEV": ("2016-07-01", "2021-01-01"), "VAL": ("2021-01-01", "2024-08-01"),
         "RECENT": ("2024-08-01", "2100-01-01")}
D_PER = {"DEV": ("2010-01-01", "2018-01-01"), "VAL": ("2018-01-01", "2024-08-01"),
         "RECENT": ("2024-08-01", "2100-01-01")}


def load(tf):
    df = pd.read_csv(DATA / f"mt5_XAUUSD_{tf}.csv")
    df["t"] = pd.to_datetime(df["epoch"], unit="s")
    return df.set_index("t")


def t_of(x):
    x = x.dropna()
    if len(x) < 15 or x.std() == 0:
        return np.nan, len(x), np.nan
    return x.mean(), len(x), x.mean() / (x.std() / math.sqrt(len(x)))


def report(title, s, per, groups=None):
    groups = groups if groups is not None else pd.Series("all", index=s.index)
    print(f"\n== {title}")
    for g in sorted(groups.dropna().unique(), key=str):
        cells, ts_ = [], {}
        for p, (a, b) in per.items():
            x = s[(groups == g) & (s.index >= a) & (s.index < b)]
            m, n, t = t_of(x)
            ts_[p] = (m, t)
            cells.append(f"{p} {m:+6.2f} t={t:+4.1f} n={n:<5}" if n >= 15 else f"{p} {'--':<22}")
        cand = ("<<<" if all(not np.isnan(ts_[k][1]) and abs(ts_[k][1]) >= 2 for k in ("DEV", "VAL"))
                and np.sign(ts_["DEV"][0]) == np.sign(ts_["VAL"][0]) else "")
        print(f"  {str(g):<22} " + "  ".join(cells) + f"  {cand}")


def at_hour(h1, hour):
    x = h1[h1.index.hour == hour]["close"]
    x.index = x.index.normalize()
    return x[~x.index.duplicated()]


def main():
    h1, d1 = load("H1"), load("D1")
    h1 = h1[h1.index >= "2016-07-01"]
    d1 = d1[d1.index >= "2009-06-01"].copy()

    # ── daily-break trade (from step 1): prev day 23:00-bar close -> 02:00-bar close
    c23 = at_hour(h1, 23); c23.index = c23.index + pd.Timedelta(days=1)
    c2 = at_hour(h1, 2)
    brk = ((c2 / c23 - 1) * 1e4).dropna()

    # conditioning variables known at entry (prev day 23:59)
    day_ret = d1["close"].pct_change() * 1e4
    rng = (d1["high"] - d1["low"])
    clv = ((d1["close"] - d1["low"]) / rng).where(rng > 0)
    atr_bp = rng.rolling(20).mean() / d1["close"] * 1e4
    prev = lambda s: s.shift(1).reindex(brk.index)       # value of the day before the break
    ny = (at_hour(h1, 23) / at_hour(h1, 15) - 1) * 1e4   # NY session move 15:00 -> 23:00 same day
    ny.index = ny.index + pd.Timedelta(days=1)

    report("break trade (all)", brk, H_PER)
    report("break trade by prior day direction", brk, H_PER,
           pd.Series(np.where(prev(day_ret) > 0, "after up day", "after down day"), index=brk.index))
    report("break trade by prior day close location", brk, H_PER,
           pd.cut(prev(clv), [0, 0.3, 0.7, 1.0001], labels=["close near low", "mid", "close near high"]))
    report("break trade by NY-session move (15->23)", brk, H_PER,
           pd.Series(np.where(ny.reindex(brk.index) > 0, "NY up", "NY down"), index=brk.index))
    vol_q = prev(atr_bp)
    report("break trade by volatility regime (20d ATR)", brk, H_PER,
           pd.Series(np.where(vol_q > vol_q.rolling(250, min_periods=60).median(), "high vol", "low vol"),
                     index=brk.index))

    # ── weekend gap: Friday 23:00-bar close -> Monday 02:00-bar close
    fri = at_hour(h1, 23); fri = fri[fri.index.dayofweek == 4]
    mon = c2[c2.index.dayofweek == 0]
    wk = []
    for d, px in fri.items():
        m = d + pd.Timedelta(days=3)
        if m in mon.index:
            wk.append((m, (mon[m] / px - 1) * 1e4))
    wk = pd.Series(dict(wk))
    report("weekend: Fri 23:59 -> Mon 02:59", wk, H_PER)

    # ── intraday momentum: does the Asia session direction carry into London+NY?
    a_ret = (at_hour(h1, 8) / at_hour(h1, 1).reindex(at_hour(h1, 8).index) - 1) * 1e4
    rest = (at_hour(h1, 23) / at_hour(h1, 8) - 1) * 1e4
    report("rest of day (08->23) x sign(Asia 01->08)", rest * np.sign(a_ret), H_PER)
    lon = (at_hour(h1, 14) / at_hour(h1, 8) - 1) * 1e4
    nys = (at_hour(h1, 23) / at_hour(h1, 14) - 1) * 1e4
    report("NY (14->23) x sign(London 08->14)", nys * np.sign(lon), H_PER)

    # ── daily effects (D1 from 2010)
    nxt = day_ret.shift(-1)
    report("next day by today's close location", nxt, D_PER,
           pd.cut(clv, [0, 0.2, 0.8, 1.0001], labels=["close in bottom 20%", "mid", "close in top 20%"]))
    dom = d1.index.day
    last_bd = d1.index.to_series().groupby(d1.index.to_period("M")).transform("max") == d1.index.to_series()
    tom = pd.Series(np.where(last_bd.values | (dom <= 3), "turn of month", "rest"), index=d1.index)
    report("daily return: turn-of-month (last day + first 3)", day_ret, D_PER, tom)
    streak = np.sign(day_ret).groupby((np.sign(day_ret) != np.sign(day_ret).shift()).cumsum()).cumcount() + 1
    lab = pd.Series(np.where((streak >= 3) & (day_ret < 0), "3+ down days",
                    np.where((streak >= 3) & (day_ret > 0), "3+ up days", None)), index=d1.index)
    report("next day after streaks", nxt, D_PER, lab)


if __name__ == "__main__":
    main()
