"""
XAUUSD macro lead-lag research (hypothesis 6), daily data 2010 -> now.

Does a past move in the dollar index (DXY), US 10-year yield, S&P 500, silver,
or copper predict gold's NEXT 1 / 5 / 20 days? Same-day correlation is not
tradeable, so every signal uses information available at the prior close.

Signal = sign(past N-day change of X) (or of the gold-vs-X divergence);
outcome = gold forward return in bp, multiplied by the expected sign.
Split: DEV 2010-2017 | VAL 2018 -> 2024-07 | RECENT 2024-08 -> now.
Candidate = same sign and |t| >= 2.5 in DEV and VAL. Forward windows overlap,
so t-stats use non-overlapping samples (every h-th day).
"""

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

CACHE = Path("data/macro")
TICKERS = {"gold": "GC=F", "dxy": "DX-Y.NYB", "us10y": "^TNX", "spx": "^GSPC", "silver": "SI=F",
           "copper": "HG=F", "tips": "TIP"}
PER = {"DEV": ("2010-01-01", "2018-01-01"), "VAL": ("2018-01-01", "2024-08-01"),
       "RECENT": ("2024-08-01", "2100-01-01")}


def load() -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    cols = {}
    for k, t in TICKERS.items():
        f = CACHE / f"{k}.csv"
        if not f.exists() or time.time() - f.stat().st_mtime > 86400:
            d = yf.download(t, start="2009-01-01", progress=False, auto_adjust=True)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            d[["Close"]].to_csv(f)
        cols[k] = pd.read_csv(f, index_col=0, parse_dates=True)["Close"]
    return pd.DataFrame(cols).dropna()


def stat(x: pd.Series, h: int):
    x = x.dropna().iloc[::h]
    if len(x) < 30:
        return None
    return x.mean(), len(x), x.mean() / x.std() * math.sqrt(len(x))


def report(label, s, h):
    cells, st = [], {}
    for p, (a, b) in PER.items():
        r = stat(s[(s.index >= a) & (s.index < b)], h)
        st[p] = r
        cells.append(f"{p} {r[0]:+7.1f}bp t={r[2]:+4.1f} n={r[1]:<4}" if r else f"{p} {'--':<25}")
    ok = (st["DEV"] and st["VAL"] and abs(st["DEV"][2]) >= 2.5 and abs(st["VAL"][2]) >= 2.5
          and np.sign(st["DEV"][0]) == np.sign(st["VAL"][0]))
    print(f"  {label:<44} " + "  ".join(cells) + ("  <<<" if ok else ""))


def main():
    df = load()
    print(f"daily data {df.index[0].date()} -> {df.index[-1].date()} ({len(df)} days)\n")
    g = df["gold"]
    for h in (1, 5, 20):
        fwd = (g.shift(-h) / g - 1) * 1e4
        print(f"== gold forward {h}-day return, signed by the expected effect")
        for n in (1, 5, 20):
            for x, sign, why in (("dxy", -1, "dollar up -> gold down"), ("us10y", -1, "yields up -> gold down"),
                                 ("spx", -1, "stocks up -> gold down"), ("silver", +1, "silver leads"),
                                 ("copper", +1, "copper leads"), ("tips", +1, "real-yield proxy")):
                chg = df[x].pct_change(n) if x != "us10y" else df[x].diff(n)
                report(f"{x} {n}d change ({why})", fwd * np.sign(chg) * sign, h)
            # divergence: gold failed to react to the dollar's move -> expect catch-up
            div = g.pct_change(n) + 0.5 * df["dxy"].pct_change(n)
            report(f"gold vs dxy divergence {n}d (fade)", fwd * -np.sign(div), h)
        print()


if __name__ == "__main__":
    main()
