"""
Step 1 of new XAUUSD strategy research: scan for raw, structural return effects
before building any strategy. No parameters are tuned here.

Split:  DEV 2010-2017 | VAL 2018 -> 2024-07 | RECENT 2024-08 -> now
An effect is a candidate only if it has the same sign and |t| >= 2 in BOTH
DEV and VAL. RECENT is reported but not used for selection.

Returns are in basis points (bp, 0.01%). Round-trip trading cost on Deriv/
IC Markets gold is ~0.75 bp (0.30 at $4000), so an effect must be several bp
per trade to be tradeable.

Times are MT5 broker server time (UTC+2 winter / UTC+3 summer), the same
clock the live bot uses.

Usage:
    python pro_bot/research_xau_edges.py
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "pro"
COST_BP = 0.75
PERIODS = {"DEV": ("2010-01-01", "2018-01-01"),
           "VAL": ("2018-01-01", "2024-08-01"),
           "RECENT": ("2024-08-01", "2100-01-01")}


def load(tf):
    df = pd.read_csv(DATA / f"mt5_XAUUSD_{tf}.csv")
    df["t"] = pd.to_datetime(df["epoch"], unit="s")
    return df.set_index("t")


def tstat(x):
    x = x.dropna()
    if len(x) < 10 or x.std() == 0:
        return np.nan, len(x), np.nan
    return x.mean(), len(x), x.mean() / (x.std() / math.sqrt(len(x)))


def scan(name, series, groups):
    """series: return in bp indexed by time; groups: Series of labels (same index)."""
    rows = []
    for g in sorted(groups.dropna().unique()):
        r = {"effect": name, "group": g}
        for p, (a, b) in PERIODS.items():
            x = series[(groups == g) & (series.index >= a) & (series.index < b)]
            m, n, t = tstat(x)
            r[f"{p}_bp"], r[f"{p}_n"], r[f"{p}_t"] = m, n, t
        rows.append(r)
    return rows


def main():
    h1, d1 = load("H1"), load("D1")
    h1 = h1[h1.index >= "2009-06-01"]
    d1 = d1[d1.index >= "2009-06-01"]

    # hourly open->close return, bp
    h1["ret"] = (h1["close"] / h1["open"] - 1) * 1e4
    # daily close->close return, bp
    d1["ret"] = d1["close"].pct_change() * 1e4
    d1["next"] = d1["ret"].shift(-1)
    atr = (d1["high"] - d1["low"]).rolling(20).mean() / d1["close"] * 1e4

    rows = []
    rows += scan("hour-of-day (1h return)", h1["ret"], pd.Series(h1.index.hour, index=h1.index))
    rows += scan("day-of-week (daily ret)", d1["ret"], pd.Series(d1.index.dayofweek, index=d1.index))

    # session blocks: sum of hourly returns within a session per day
    sess = {"asia 01-08": range(1, 9), "london 09-14": range(9, 15),
            "ny 15-19": range(15, 20), "late 20-00": list(range(20, 24)) + [0]}
    for sname, hrs in sess.items():
        m = h1[h1.index.hour.isin(hrs)]
        s = m["ret"].groupby(m.index.date).sum()
        s.index = pd.to_datetime(s.index)
        rows += scan("session (summed bp/day)", s, pd.Series(sname, index=s.index))

    # time-series momentum / reversal: sign of past N-day return vs next-day return
    for n in (1, 5, 20, 60, 120):
        past = d1["close"].pct_change(n)
        signed = d1["next"] * np.sign(past)
        rows += scan("momentum: next-day ret x sign(past N)", signed, pd.Series(f"N={n}", index=d1.index))

    # after big days: next-day return signed by today's direction (+ = continuation)
    z = d1["ret"] / atr
    big = pd.Series(np.where(z.abs() > 1.5, "|move|>1.5 ATR", np.where(z.abs() > 1.0, "1-1.5 ATR", None)),
                    index=d1.index)
    rows += scan("after big day: next x sign(today)", d1["next"] * np.sign(d1["ret"]), big)

    # gap: Monday open vs Friday close, then rest of Monday signed by gap
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    fmt = res.copy()
    for p in PERIODS:
        fmt[p] = fmt.apply(lambda r: f"{r[p+'_bp']:+6.2f}bp t={r[p+'_t']:+5.1f} n={int(r[p+'_n'])}"
                           if not pd.isna(r[p + "_bp"]) else "--", axis=1)
    fmt["CANDIDATE"] = res.apply(
        lambda r: "<<<" if (not pd.isna(r["DEV_t"]) and not pd.isna(r["VAL_t"])
                            and abs(r["DEV_t"]) >= 2 and abs(r["VAL_t"]) >= 2
                            and np.sign(r["DEV_bp"]) == np.sign(r["VAL_bp"])) else "", axis=1)
    for eff, g in fmt.groupby("effect", sort=False):
        print(f"\n== {eff}")
        for _, r in g.iterrows():
            print(f"  {str(r['group']):<16} DEV {r['DEV']:<28} VAL {r['VAL']:<28} RECENT {r['RECENT']:<28} {r['CANDIDATE']}")
    print(f"\n(round-trip cost ~{COST_BP} bp; '<<<' = same sign and |t|>=2 in both DEV and VAL)")


if __name__ == "__main__":
    main()
