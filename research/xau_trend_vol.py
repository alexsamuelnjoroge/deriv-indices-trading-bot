"""
XAUUSD trend and volatility-regime research (hypotheses 2 and 8), daily 2010 -> now.

Trend: time-series momentum. Each day hold sign(past N-day return) (long/short)
or long-only when positive, sized so the position targets constant volatility
(10% annualised, using 20-day realised vol). Daily P&L in % of capital, costs
included: 1bp per unit of position change (Deriv-like) + optional MT5 swap.

Vol regime: does the trend signal work better when realised vol is low vs high
relative to its 1-year median?

Split: DEV 2010-2017 | VAL 2018 -> 2024-07 | RECENT 2024-08 -> now.
Reports annualised return, vol, Sharpe and max drawdown per period. A config
counts only if Sharpe > 0.3 in BOTH DEV and VAL.
"""

import math

import numpy as np
import pandas as pd

PER = {"DEV": ("2010-01-01", "2018-01-01"), "VAL": ("2018-01-01", "2024-08-01"),
       "RECENT": ("2024-08-01", "2100-01-01")}
TARGET_VOL = 0.10
COST_BP = 1.0


def load() -> pd.Series:
    return pd.read_csv("data/macro/gold.csv", index_col=0, parse_dates=True)["Close"].dropna()


def perf(pnl: pd.Series):
    pnl = pnl.dropna()
    if len(pnl) < 100:
        return None
    ann = pnl.mean() * 252
    vol = pnl.std() * math.sqrt(252)
    eq = (1 + pnl).cumprod()
    dd = (1 - eq / eq.cummax()).max()
    return ann, vol, ann / vol if vol else 0, dd


def run(g: pd.Series, n: int, long_only: bool, vol_filter: str | None = None) -> pd.Series:
    r = g.pct_change()
    rv = r.rolling(20).std() * math.sqrt(252)
    mom = g.pct_change(n)
    sig = np.sign(mom)
    if long_only:
        sig = sig.clip(lower=0)
    if vol_filter:
        med = rv.rolling(252, min_periods=120).median()
        keep = rv < med if vol_filter == "low" else rv >= med
        sig = sig.where(keep, 0)
    pos = (sig * TARGET_VOL / rv).clip(-3, 3)          # cap leverage at 3x
    pos = pos.shift(1)                                  # trade on next day's return
    cost = pos.diff().abs() * COST_BP / 1e4
    return pos * r - cost


def show(label, pnl):
    cells, sh = [], {}
    for p, (a, b) in PER.items():
        m = perf(pnl[(pnl.index >= a) & (pnl.index < b)])
        sh[p] = m[2] if m else None
        cells.append(f"{p} {m[0]*100:+6.1f}%/yr SR {m[2]:+.2f} DD {m[3]*100:4.0f}%" if m else f"{p} --")
    ok = sh["DEV"] is not None and sh["VAL"] is not None and sh["DEV"] > 0.3 and sh["VAL"] > 0.3
    print(f"  {label:<34} " + "   ".join(cells) + ("  <<<" if ok else ""))


def main():
    g = load()
    print(f"gold daily {g.index[0].date()} -> {g.index[-1].date()}  | vol target {TARGET_VOL*100:.0f}%, cost {COST_BP}bp/turnover\n")
    show("buy & hold (vol-scaled)", (TARGET_VOL / (g.pct_change().rolling(20).std() * math.sqrt(252))).clip(0, 3).shift(1) * g.pct_change())
    print("\n== time-series momentum: hold sign(past N-day return)")
    for n in (10, 20, 40, 60, 120, 250):
        show(f"long/short N={n}", run(g, n, False))
        show(f"long-only  N={n}", run(g, n, True))
    print("\n== same signal split by volatility regime (vs 1-year median of 20d vol)")
    for n in (20, 60, 120):
        show(f"L/S N={n} low-vol days only", run(g, n, False, "low"))
        show(f"L/S N={n} high-vol days only", run(g, n, False, "high"))
    print("\n== blend: average of N = 20, 60, 120 long/short (diversified lookbacks)")
    show("blend 20/60/120 L/S", sum(run(g, n, False) for n in (20, 60, 120)) / 3)
    show("blend 20/60/120 long-only", sum(run(g, n, True) for n in (20, 60, 120)) / 3)


if __name__ == "__main__":
    main()
