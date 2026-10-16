"""
Step 3 of research: cross-market checks.

1. Break drift (prev-day 23:00-bar close -> 02:00-bar close, broker time) on
   other MT5 markets: does the gold effect generalise?
2. Lead-lag: gold's next-hour return signed by the previous hour's move in
   USDJPY / EURUSD / US500 / USTEC / XAGUSD / XTIUSD.
3. Gold/silver ratio: forward 10-day change of log(XAU/XAG) after the 60-day
   z-score exceeds +/-2 (positive number = ratio reverted toward its mean).

Split as in step 2 (hourly complete from mid-2016):
  DEV 2016-07 -> 2020 | VAL 2021 -> 2024-07 | RECENT 2024-08 -> now
'<<<' = same sign and |t| >= 2 in both DEV and VAL. Values in bp.

Usage:
    python pro_bot/research_cross_market.py [--refresh]
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "pro"
PER = {"DEV": ("2016-07-01", "2021-01-01"), "VAL": ("2021-01-01", "2024-08-01"),
       "RECENT": ("2024-08-01", "2100-01-01")}
SYMS = ["XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD", "XAUEUR", "EURUSD", "USDJPY", "GBPUSD", "AUDUSD",
        "US500", "USTEC", "US30", "DE40", "JP225", "XTIUSD", "XBRUSD", "XNGUSD", "BTCUSD"]


def load(sym, tf, refresh):
    p = DATA / f"mt5_{sym}_{tf}.csv"
    if refresh or not p.exists():
        import MetaTrader5 as mt5
        assert mt5.initialize(), mt5.last_error()
        mt5.symbol_select(sym, True)
        t = {"H1": mt5.TIMEFRAME_H1, "D1": mt5.TIMEFRAME_D1}[tf]
        r = mt5.copy_rates_from_pos(sym, t, 0, mt5.terminal_info().maxbars - 1)
        mt5.shutdown()
        if r is None or len(r) == 0:
            return None
        df = pd.DataFrame(r)[["time", "open", "high", "low", "close", "spread"]].iloc[:-1]
        df.rename(columns={"time": "epoch"}).to_csv(p, index=False)
    df = pd.read_csv(p)
    df["t"] = pd.to_datetime(df["epoch"], unit="s")
    return df.set_index("t")


def tt(x):
    x = x.dropna()
    if len(x) < 15 or x.std() == 0:
        return np.nan, len(x), np.nan
    return x.mean(), len(x), x.mean() / x.std() * math.sqrt(len(x))


def line(label, s, extra=""):
    cells, st = [], {}
    for p, (a, b) in PER.items():
        m, n, t = tt(s[(s.index >= a) & (s.index < b)])
        st[p] = (m, t)
        cells.append(f"{p} {m:+7.2f} t={t:+4.1f} n={n:<5}" if n >= 15 else f"{p} {'--':<24}")
    ok = all(not np.isnan(st[k][1]) and abs(st[k][1]) >= 2 for k in ("DEV", "VAL")) \
        and np.sign(st["DEV"][0]) == np.sign(st["VAL"][0])
    print(f"  {label:<22} " + "  ".join(cells) + f"  {extra}{'<<<' if ok else ''}")


def at_hour(h1, hour):
    x = h1[h1.index.hour == hour]["close"]
    x.index = x.index.normalize()
    return x[~x.index.duplicated()]


def main():
    refresh = "--refresh" in sys.argv
    h = {}
    for s in SYMS:
        df = load(s, "H1", refresh)
        if df is not None:
            h[s] = df[df.index >= "2016-07-01"]

    print("\n== 1. Break drift: prev 23:00-bar close -> 02:00-bar close (bp, gross)")
    for s, df in h.items():
        c23 = at_hour(df, 23); c23.index = c23.index + pd.Timedelta(days=1)
        c2 = at_hour(df, 2)
        brk = ((c2 / c23 - 1) * 1e4).dropna()
        brk = brk[brk.index.dayofweek.isin([1, 2, 3, 4])]
        line(s, brk)

    print("\n== 2. Lead-lag: gold next-hour return x sign(other market's previous hour)")
    g = h["XAUUSD"]["close"]
    g_next = (g.shift(-1) / g - 1) * 1e4
    for s in ["XAGUSD", "USDJPY", "EURUSD", "US500", "USTEC", "XTIUSD", "BTCUSD"]:
        o = h[s]["close"]
        o_ret = (o / o.shift(1) - 1).reindex(g.index)
        line(f"{s} -> gold", g_next * np.sign(o_ret))
    line("gold own (momentum)", g_next * np.sign((g / g.shift(1) - 1)))

    print("\n== 3. Gold/silver ratio: fwd 10-day log-ratio change toward mean after |z60|>2 (bp)")
    xau, xag = load("XAUUSD", "D1", refresh), load("XAGUSD", "D1", refresh)
    lr = np.log(xau["close"] / xag["close"].reindex(xau.index)).dropna()
    z = (lr - lr.rolling(60).mean()) / lr.rolling(60).std()
    fwd = (lr.shift(-10) - lr) * 1e4
    rev = (-fwd * np.sign(z)).where(z.abs() > 2)
    per_d = {"DEV": ("2010-01-01", "2018-01-01"), "VAL": ("2018-01-01", "2024-08-01"),
             "RECENT": ("2024-08-01", "2100-01-01")}
    global PER
    PER_saved, PER = PER, per_d
    line("ratio |z|>2 reversion", rev)
    line("  z>+2 (gold rich)", rev.where(z > 2))
    line("  z<-2 (silver rich)", rev.where(z < -2))
    PER = PER_saved
    print("\n  (overlapping 10-day windows -> t-stats overstated ~3x for item 3)")


if __name__ == "__main__":
    main()
