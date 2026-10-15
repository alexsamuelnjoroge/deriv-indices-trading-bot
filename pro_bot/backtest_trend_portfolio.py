"""
Multi-symbol trend-following portfolio (daily bars, 2012 -> today).

Same fixed, textbook rules on every symbol -- no per-symbol tuning -- and all
variants are reported, not just the best.

Deriv's API only serves ~1 year of candles, so history comes from Yahoo
Finance (same underlying markets). Deriv execution rules are applied:
  * commission (fraction of notional, from live MULTUP proposals)
  * min multiplier x100 => stake lost at a 1% adverse move, so the stop
    distance is capped at STOP_CAP and losses can't exceed the stake.
Each variant is also run WITHOUT the cap to show what the x100 floor costs.

Every trade risks 1R. Equity compounds RISK_PCT per trade at exit.

Usage:
    python pro_bot/backtest_trend_portfolio.py
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

CACHE = Path("data/pro/yf")
START = "2012-01-01"
STOP_CAP = 0.0095
RISK_PCT = 0.005

# deriv symbol: (yahoo ticker, commission as fraction of notional)
UNIVERSE = {
    "frxEURUSD": ("EURUSD=X", 0.00024), "frxGBPUSD": ("GBPUSD=X", 0.00020),
    "frxUSDJPY": ("JPY=X",    0.00028), "frxAUDUSD": ("AUDUSD=X", 0.00040),
    "frxUSDCAD": ("CAD=X",    0.00021), "frxUSDCHF": ("CHF=X",    0.00036),
    "frxEURGBP": ("EURGBP=X", 0.00024), "frxEURJPY": ("EURJPY=X", 0.00025),
    "frxEURAUD": ("EURAUD=X", 0.00026), "frxEURCAD": ("EURCAD=X", 0.00024),
    "frxEURCHF": ("EURCHF=X", 0.00035), "frxGBPJPY": ("GBPJPY=X", 0.00024),
    "frxGBPAUD": ("GBPAUD=X", 0.00021), "frxAUDJPY": ("AUDJPY=X", 0.00038),
    "frxXAUUSD": ("GC=F",     0.00012), "frxXAGUSD": ("SI=F",     0.00060),
    "cryBTCUSD": ("BTC-USD",  0.00031), "cryETHUSD": ("ETH-USD",  0.00082),
}

# Deriv crypto multipliers expire after ~7.6 days (metals/FX don't)
ROLL_DAYS = {"cryBTCUSD": 7, "cryETHUSD": 7}

VARIANTS = {
    "Donchian 20/10":    ("donchian", dict(entry=20,  exit=10, atr_mult=2.0)),
    "Donchian 55/20":    ("donchian", dict(entry=55,  exit=20, atr_mult=2.0)),
    "Donchian 100/50":   ("donchian", dict(entry=100, exit=50, atr_mult=2.0)),
    "EMA 50/200 + 3ATR": ("ema",      dict(fast=50, slow=200, atr_mult=3.0)),
}


def load(sym, ticker):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{sym}_1d.csv"
    if f.exists() and time.time() - f.stat().st_mtime < 86400:
        df = pd.read_csv(f, index_col=0, parse_dates=True)
    else:
        df = yf.download(ticker, start=START, interval="1d", progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close"]].dropna()
        df.to_csv(f)
    # drop bad prints (zero-range bars or high<low) that Yahoo FX sometimes has
    df = df[(df.High >= df.Low) & (df.High >= df.Close) & (df.Low <= df.Close) & (df.Close > 0)]
    return df


def atr(h, l, c, n=20):
    tr = np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
    tr = np.concatenate([[h[0] - l[0]], tr])
    out = np.full_like(c, np.nan)
    out[n - 1] = tr[:n].mean()
    for i in range(n, len(c)):
        out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return out


def ema(x, n):
    out = np.empty_like(x)
    k = 2 / (n + 1)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = x[i] * k + out[i - 1] * (1 - k)
    return out


def roll(x, n, fn):
    s = pd.Series(x).shift(1).rolling(n)
    return (s.max() if fn == "max" else s.min()).to_numpy()


def simulate(df, kind, prm, comm, cap_on, cap_frac=STOP_CAP, roll_days=None):
    """Returns list of (entry_ts, exit_ts, R, capped)."""
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("Open", "High", "Low", "Close"))
    ts = np.array([int(x.timestamp()) for x in df.index])
    a = atr(h, l, c)

    if kind == "donchian":
        up_ch, dn_ch = roll(h, prm["entry"], "max"), roll(l, prm["entry"], "min")
        ex_lo, ex_hi = roll(l, prm["exit"], "min"), roll(h, prm["exit"], "max")
        warm = prm["entry"] + 1
    else:
        ef, es = ema(c, prm["fast"]), ema(c, prm["slow"])
        warm = prm["slow"]

    trades, pos = [], 0
    for i in range(max(warm, 21), len(c) - 1):
        if pos == 0:
            if kind == "donchian":
                sig = 1 if c[i] > up_ch[i] else -1 if c[i] < dn_ch[i] else 0
            else:
                sig = 1 if ef[i] > es[i] and ef[i - 1] <= es[i - 1] else \
                     -1 if ef[i] < es[i] and ef[i - 1] >= es[i - 1] else 0
            if sig and not math.isnan(a[i]):
                entry = o[i + 1]
                want = prm["atr_mult"] * a[i]
                cap = cap_frac * entry if cap_on else float("inf")
                dist = min(want, cap)
                capped = want > cap
                stop = entry - sig * dist
                stake_out = entry - sig * cap
                pos, e_i, e_px, d0, best = sig, i + 1, entry, dist, entry
                last_roll, roll_cost = i + 1, 0.0
            continue

        # Deriv crypto multipliers expire: close and reopen at the close, which
        # re-anchors the stake-out level and pays commission again
        if roll_days and i - last_roll >= roll_days:
            roll_cost += comm * c[i] / d0
            last_roll = i
            if cap_on:
                new_so = c[i] - pos * cap_frac * c[i]
                stake_out = max(stake_out, new_so) if pos == 1 else min(stake_out, new_so)
                stop = max(stop, stake_out) if pos == 1 else min(stop, stake_out)

        done = False
        if pos == 1:
            if l[i] <= stop:
                exit_px, done = max(min(stop, o[i]), stake_out), True
            elif kind == "donchian" and c[i] < ex_lo[i]:
                exit_px, done = o[i + 1], True
            else:
                best = max(best, c[i])
                if kind == "ema":
                    stop = max(stop, best - min(prm["atr_mult"] * a[i], cap))
        else:
            if h[i] >= stop:
                exit_px, done = min(max(stop, o[i]), stake_out), True
            elif kind == "donchian" and c[i] > ex_hi[i]:
                exit_px, done = o[i + 1], True
            else:
                best = min(best, c[i])
                if kind == "ema":
                    stop = min(stop, best + min(prm["atr_mult"] * a[i], cap))

        if done:
            r = pos * (exit_px - e_px) / d0
            if cap_on:
                r = max(r, -cap / d0)
            r -= comm * e_px / d0 + roll_cost
            trades.append((ts[e_i], ts[min(i + 1, len(ts) - 1)], r, capped))
            pos = 0
    return trades


def stats(tr, t0, t1):
    tr = sorted([t for t in tr if t0 <= t[1] < t1], key=lambda x: x[1])
    if not tr:
        return None
    rs = np.array([t[2] for t in tr])
    eq, peak, mdd = 1.0, 1.0, 0.0
    for r in rs:
        eq *= 1 + RISK_PCT * r
        peak = max(peak, eq)
        mdd = max(mdd, 1 - eq / peak)
    yrs = (t1 - t0) / (365.25 * 86400)
    neg = -rs[rs < 0].sum()
    return dict(n=len(rs), wr=(rs > 0).mean(), ev=rs.mean(), net=rs.sum(),
                pf=rs[rs > 0].sum() / neg if neg else float("inf"),
                cagr=eq ** (1 / yrs) - 1, mdd=mdd)


def short(sym):
    return sym.replace("frx", "").replace("cry", "")


def main():
    print("Loading daily history (Yahoo Finance)...")
    data = {}
    for sym, (tk, _) in UNIVERSE.items():
        try:
            data[sym] = load(sym, tk)
        except Exception as e:
            print(f"  {sym}: failed ({e})")
    for sym, df in data.items():
        print(f"  {short(sym):<7} {len(df):>5} bars  {df.index[0].date()} -> {df.index[-1].date()}")

    t0 = int(pd.Timestamp("2013-01-01").timestamp())
    t1 = int(time.time())
    years = list(range(2013, pd.Timestamp.now().year + 1))
    fmt = lambda t: time.strftime("%Y-%m", time.gmtime(t))
    periods = np.linspace(t0, t1, 5)

    summary = []
    for vname, (kind, prm) in VARIANTS.items():
        for cap_on in (True, False):
            all_tr, per_sym = [], {}
            for sym, df in data.items():
                rd = ROLL_DAYS.get(sym)
                tr = [t for t in simulate(df, kind, prm, UNIVERSE[sym][1], cap_on, roll_days=rd) if t[0] >= t0]
                per_sym[sym] = tr
                all_tr += tr
            s = stats(all_tr, t0, t1)
            tag = "DERIV x100 (1% stop cap)" if cap_on else "uncapped (normal broker)"
            capped = np.mean([t[3] for t in all_tr]) if cap_on else 0

            print("\n" + "=" * 100)
            print(f"  {vname}  |  {tag}" + (f"  |  stop capped on {capped*100:.0f}% of trades" if cap_on else ""))
            print("=" * 100)
            print(f"  PORTFOLIO {fmt(t0)}->{fmt(t1)}  n={s['n']}  WR {s['wr']*100:.1f}%  EV {s['ev']:+.3f}R  "
                  f"net {s['net']:+.0f}R  PF {s['pf']:.2f}  CAGR {s['cagr']*100:+.1f}%  maxDD {s['mdd']*100:.1f}%")

            ppos = 0
            for k in range(4):
                ps = stats(all_tr, periods[k], periods[k + 1])
                ppos += bool(ps and ps["ev"] > 0)
                if ps:
                    print(f"    {fmt(periods[k])} -> {fmt(periods[k+1])}  n={ps['n']:>4}  EV {ps['ev']:+.3f}R  "
                          f"net {ps['net']:+6.0f}R  maxDD {ps['mdd']*100:5.1f}%")
            ypos = []
            for y in years:
                ys = stats(all_tr, int(pd.Timestamp(f"{y}-01-01").timestamp()),
                           int(pd.Timestamp(f"{y+1}-01-01").timestamp()))
                ypos.append(f"{y % 100:02d}:{ys['net']:+.0f}" if ys else f"{y % 100:02d}:-")
            print("    by year (net R): " + "  ".join(ypos))

            groups = {"FX": lambda s_: s_.startswith("frx") and s_ not in ("frxXAUUSD", "frxXAGUSD"),
                      "metals": lambda s_: s_ in ("frxXAUUSD", "frxXAGUSD"),
                      "crypto": lambda s_: s_.startswith("cry")}
            gl = []
            for g, f in groups.items():
                gtr = [t for s_, tr in per_sym.items() if f(s_) for t in tr]
                gs = stats(gtr, t0, t1)
                gl.append(f"{g}: EV {gs['ev']:+.3f}R CAGR {gs['cagr']*100:+.1f}% DD {gs['mdd']*100:.0f}%")
            print("    by class: " + "  |  ".join(gl))

            pos_syms = sum(1 for tr in per_sym.values() if sum(t[2] for t in tr) > 0)
            cells = [f"{short(s_):>6} {sum(t[2] for t in tr):+5.0f}R/{len(tr):<3}" for s_, tr in per_sym.items()]
            print(f"  per symbol (net R / trades): {pos_syms}/{len(per_sym)} positive")
            for k in range(0, len(cells), 6):
                print("    " + "  ".join(cells[k:k + 6]))
            summary.append((vname, tag, s, ppos, pos_syms, len(per_sym)))

    print("\n" + "=" * 100)
    print("  SUMMARY")
    print("=" * 100)
    for vname, tag, s, ppos, ps_, ns in summary:
        print(f"  {vname:<18} {tag:<26} EV {s['ev']:+.3f}R  CAGR {s['cagr']*100:+5.1f}%  "
              f"maxDD {s['mdd']*100:4.1f}%  periods+ {ppos}/4  symbols+ {ps_}/{ns}")
    print()


if __name__ == "__main__":
    main()
