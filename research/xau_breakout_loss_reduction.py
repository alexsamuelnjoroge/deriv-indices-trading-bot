"""
Loss-reduction variants for the Asian-range breakout (incubator rule D), pre-registered 2026-11-11.

Baseline V0: range 03:00-08:00 server, OCO stops at the range, SL opposite side, TP 1R,
cancel 22:00, flat 23:00 (same simulator conventions as xau_session_breakout_ea.py).
  V1 skip wide ranges: range width > 80th percentile of the previous 60 days' widths
  V2 with the daily trend only: side must match the sign of the daily EMA20 slope
     (completed server days only)
  V3 skip extreme volatility: daily ATR14 > 90th percentile of the previous 250 days
  V4 break-even: once price reaches +0.5R, the stop moves to the entry
  V5 no late fills: orders still unfilled at 16:00 server are cancelled
Judged in R (risk = range width) on TRAIN 2008-16, TEST 2017-26 and IC MT5 2022-26:
mean R, total R per year, max drawdown in R, longest losing streak.
A variant "reduces losses" only if drawdown falls without cutting mean R in BOTH Duka periods.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xau_session_breakout_ea as base  # noqa: E402

COST = 0.36


def daily_table(m15):
    d = m15.groupby(m15.index.date).agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
    d["ema20"] = d["close"].ewm(span=20, adjust=False).mean()
    prev_close = d["close"].shift(1)
    tr = np.maximum(d["high"], prev_close) - np.minimum(d["low"], prev_close)
    d["atr14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    # all features shifted one day: only completed days are known at 08:00
    d["trend"] = np.sign(d["ema20"] - d["ema20"].shift(1)).shift(1)
    d["atr_pct"] = d["atr14"].rolling(250, min_periods=120).apply(
        lambda x: (x[:-1] < x[-1]).mean(), raw=True).shift(1)
    return d


def run(m15, variant):
    dt = daily_table(m15)
    widths = []
    out = []
    for d, g in m15.groupby(m15.index.date):
        if g.index[0].weekday() >= 5:
            continue
        rng = g[(g["mins"] >= 180) & (g["mins"] < 480)]
        if len(rng) < 18:
            continue
        hi, lo = rng["high"].max(), rng["low"].min()
        w = hi - lo
        if w <= 0:
            continue
        hist = widths[-60:]
        widths.append(w / lo)
        if variant == "V1" and len(hist) >= 40 and w / lo > np.quantile(hist, 0.8):
            continue
        if variant == "V3" and d in dt.index and dt.loc[d, "atr_pct"] > 0.9:
            continue
        trend = dt.loc[d, "trend"] if d in dt.index else 0
        after = g[g["mins"] >= 480].reset_index(drop=True)
        last_fill = 16 * 60 if variant == "V5" else 22 * 60
        for k in range(len(after)):
            b = after.iloc[k]
            if b["mins"] >= last_fill:
                break
            up, dn = b["high"] > hi, b["low"] < lo
            if not (up or dn):
                continue
            if up and dn:
                side = 1 if (hi - b["open"]) <= (b["open"] - lo) else -1
                if variant == "V2" and side != trend:
                    break
                out.append((d, side, -1.0 - COST / (w / lo * 1e4)))
                break
            side = 1 if up else -1
            if variant == "V2" and side != trend:
                break
            entry = max(b["open"], hi) if up else min(b["open"], lo)
            stop = lo if up else hi
            risk = abs(entry - stop)
            target = entry + side * risk
            be_trig = entry + side * 0.5 * risk
            moved = False
            px = None
            if (side > 0 and b["low"] <= stop) or (side < 0 and b["high"] >= stop):
                px = stop
            for j in range(k + 1, len(after)):
                if px is not None:
                    break
                c = after.iloc[j]
                if c["mins"] >= 23 * 60:
                    px = c["open"]
                    break
                if (side > 0 and c["low"] <= stop) or (side < 0 and c["high"] >= stop):
                    px = stop
                elif (side > 0 and c["high"] >= target) or (side < 0 and c["low"] <= target):
                    px = target
                elif variant == "V4" and not moved and ((side > 0 and c["high"] >= be_trig) or
                                                        (side < 0 and c["low"] <= be_trig)):
                    stop, moved = entry, True       # takes effect from the next bar
            if px is None:
                px = after.iloc[-1]["close"]
            gross_bp = side * (px / entry - 1) * 1e4
            out.append((d, side, (gross_bp - COST) / (risk / entry * 1e4)))
            break
    return out


def summarise(tr):
    r = np.array([x[2] for x in tr])
    if len(r) < 30:
        return "n<30"
    eq = np.cumsum(r)
    dd = (eq - np.maximum.accumulate(eq)).min()
    streak = cur = 0
    for v in r:
        cur = cur + 1 if v < 0 else 0
        streak = max(streak, cur)
    years = len({x[0].year for x in tr})
    t = r.mean() / r.std(ddof=1) * np.sqrt(len(r))
    return (f"n={len(r):4d} mean={r.mean():+.3f}R t={t:+5.2f} R/yr={r.sum() / years:+5.1f} "
            f"maxDD={dd:6.1f}R streak={streak:2d}")


def load_mt5_m15():
    d = pd.read_csv("data/pro/mt5_XAUUSD_M15.csv")
    mt = pd.DataFrame({"open": d.open.values, "high": d.high.values, "low": d.low.values,
                       "close": d.close.values}, index=pd.to_datetime(d.epoch, unit="s"))
    mt["mins"] = mt.index.hour * 60 + mt.index.minute
    return mt


if __name__ == "__main__":
    mt = load_mt5_m15()
    LABEL = {"V0": "baseline", "V1": "skip wide ranges", "V2": "with daily trend",
             "V3": "skip extreme vol", "V4": "break-even at +0.5R", "V5": "no fills after 16:00"}
    for v, lab in LABEL.items():
        tr = run(base.m15, v)
        print(f"{v} {lab}")
        print(f"   TRAIN 08-16 {summarise([x for x in tr if x[0].year < 2017])}")
        print(f"   TEST  17-26 {summarise([x for x in tr if x[0].year >= 2017])}")
        print(f"   MT5   22-26 {summarise(run(mt, v))}", flush=True)
