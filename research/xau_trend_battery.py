"""
Adversarial test battery for the XAUUSD medium-term time-series-momentum candidate.

Rule: each day hold sign(past N-day return) (long/short, or long-only), sized to a
target annualised vol using 20-day realised vol, capped at 3x, traded with a lag.
Default: N=120, 10% vol target, lag 1, daily rebalance, 1bp per unit turnover.

Tests (see task): 1 cross-source, 2 lookback surface, 3 trend vs buy&hold + short side,
4 year-by-year / regimes, 5 placebo + block bootstrap, 6 costs/swap/rebalance/lag,
7 walk-forward N selection, 8 cross-asset, 9 Monte Carlo drawdowns, 10 multiple testing.

Data: MT5 XAUUSD D1 spot (1998->), Dukascopy M5 spot -> daily (17:00 NY close, 2018->),
Yahoo GC=F (unadjusted front month, 2000->) and a yfinance cross-asset panel cached to
research/cache_xasset.csv.  Research only: reads files, no orders, no live-code changes.

Run: .venv/Scripts/python.exe research/xau_trend_battery.py
"""

import glob
import math
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

RNG = np.random.default_rng(12345)
CACHE = "research/cache_xasset.csv"
TICKERS = ["GC=F", "SI=F", "HG=F", "PL=F", "CL=F", "NG=F", "ZC=F", "ZW=F", "EURUSD=X", "JPY=X",
           "AUDUSD=X", "GBPUSD=X", "^GSPC", "^GDAXI", "^N225", "ZN=F", "ZB=F", "DX-Y.NYB", "^IRX", "GLD"]
NS = [20, 40, 60, 80, 100, 120, 150, 200, 250]
PERIODS = [("1999-04", "1999-04-01", "2005-01-01"), ("2005-09", "2005-01-01", "2010-01-01"),
           ("2010-14", "2010-01-01", "2015-01-01"), ("2015-19", "2015-01-01", "2020-01-01"),
           ("2020-24", "2020-01-01", "2025-01-01"), ("2025-26", "2025-01-01", "2100-01-01")]
# MT5 swap read from terminal 2026-10-21: long -54.724, short +37.742 points/lot/night,
# point 0.01, 100 oz, price ~4122 -> per calendar night as fraction of notional
PX_NOW = 4122.0
SWAP_LONG_NOW = -54.724 / (PX_NOW * 100) * 365     # ~ -4.85%/yr
SWAP_SHORT_NOW = 37.742 / (PX_NOW * 100) * 365     # ~ +3.34%/yr
N_CONFIGS = 0                                       # multiple-testing counter


# ----------------------------------------------------------------------------- data
def load_mt5() -> pd.Series:
    d = pd.read_csv("data/pro/mt5_XAUUSD_D1.csv")
    s = pd.Series(d["close"].values, index=pd.to_datetime(d["epoch"], unit="s").dt.normalize())
    s = s[~s.index.duplicated(keep="last")]
    return s[s.index.dayofweek < 5].sort_index()


def load_duka() -> pd.Series:
    parts = []
    for f in sorted(glob.glob("data/duka/frxXAUUSD_M5_*.csv")):
        d = pd.read_csv(f, usecols=["timestamp", "close"])
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
        parts.append(d)
    d = pd.concat(parts).drop_duplicates("timestamp").set_index("timestamp")["close"].sort_index()
    ny = d.index.tz_convert("America/New_York") + pd.Timedelta(hours=7)   # day ends 17:00 NY
    s = d.groupby(ny.date).last()
    s.index = pd.to_datetime(s.index)
    return s[s.index.dayofweek < 5]


def load_xasset() -> pd.DataFrame:
    if not os.path.exists(CACHE):
        import yfinance as yf
        d = yf.download(TICKERS, start="1990-01-01", auto_adjust=True, progress=False)["Close"]
        d.to_csv(CACHE)
    d = pd.read_csv(CACHE, index_col=0, parse_dates=True)
    return d


# ----------------------------------------------------------------------------- engine
def strat(px: pd.Series, n=120, long_only=False, short_only=False, tv=0.10, lag=1, rebal="daily",
          cost_bp=1.0, swap=None, sig_override=None, cap=3.0) -> pd.Series:
    """Daily P&L (fraction of capital). swap: None | (long_rate_series_or_float, short_rate_...)
    expressed as annual fraction of notional per calendar day (long negative = paid)."""
    px = px.dropna()
    r = px.pct_change()
    rv = r.rolling(20).std() * math.sqrt(252)
    if sig_override is not None:
        sig = sig_override.reindex(px.index)
    else:
        sig = np.sign(px.pct_change(n))
    if long_only:
        sig = sig.clip(lower=0)
    if short_only:
        sig = sig.clip(upper=0)
    size = (tv / rv).clip(upper=cap)
    pos = sig * size
    if rebal == "weekly":
        keep = pd.Series(px.index.dayofweek == 4, index=px.index)
        pos = pos.where(keep).ffill()
    elif rebal == "signal":
        run = (sig != sig.shift()).cumsum()
        pos = (sig * size.groupby(run).transform("first"))
    pos = pos.shift(lag)
    pnl = pos * r - pos.diff().abs() * cost_bp / 1e4
    if swap is not None:
        days = pd.Series(px.index, index=px.index).diff().dt.days.fillna(1)
        lr, sr = swap
        lr = lr.reindex(px.index).ffill() if isinstance(lr, pd.Series) else lr
        sr = sr.reindex(px.index).ffill() if isinstance(sr, pd.Series) else sr
        sw = np.where(pos > 0, pos * lr, -pos * sr) * days / 365
        pnl = pnl + pd.Series(sw, index=px.index).fillna(0)
    return pnl.iloc[max(n, 21) + lag:]


def bh(px, tv=0.10, **kw):
    return strat(px, n=1, sig_override=pd.Series(1.0, index=px.dropna().index), tv=tv, **kw)


def sharpe(p):
    p = p.dropna()
    if len(p) < 60 or p.std() == 0:
        return np.nan
    return p.mean() / p.std() * math.sqrt(252)


def stats(p):
    p = p.dropna()
    eq = (1 + p).cumprod()
    return dict(ann=p.mean() * 252, vol=p.std() * math.sqrt(252), sr=sharpe(p),
                dd=(1 - eq / eq.cummax()).max())


def sl(p, a, b):
    return p[(p.index >= a) & (p.index < b)]


def per_row(p, periods=PERIODS):
    return "  ".join(f"{sharpe(sl(p, a, b)):+5.2f}" if len(sl(p, a, b)) > 60 else "   --" for _, a, b in periods)


def alpha_vs(p, b, lags=20):
    """OLS p = a + beta*b, Newey-West t for a. Returns ann alpha, beta, t."""
    df = pd.concat([p, b], axis=1).dropna()
    y, x = df.iloc[:, 0].values, df.iloc[:, 1].values
    X = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    T = len(y)
    S = (X * e[:, None]).T @ (X * e[:, None])
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        G = (X[l:] * e[l:, None]).T @ (X[:-l] * e[:-l, None])
        S += w * (G + G.T)
    XtXi = np.linalg.inv(X.T @ X)
    V = XtXi @ S @ XtXi
    return beta[0] * 252, beta[1], beta[0] / math.sqrt(V[0, 0])


def block_boot_idx(T, block, rng, src=None):
    src = T if src is None else src
    nb = int(math.ceil(T / block))
    starts = rng.integers(0, src - block, nb)
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:T]


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


# ----------------------------------------------------------------------------- tests
def main():
    global N_CONFIGS
    out_lines = []
    mt5s = load_mt5()
    duka = load_duka()
    xa = load_xasset()
    gc = xa["GC=F"].dropna()
    irx = xa["^IRX"].dropna() / 100

    # MT5 data sanity
    rm = mt5s.pct_change()
    print(f"MT5 D1 spot {mt5s.index[0].date()} -> {mt5s.index[-1].date()} n={len(mt5s)}; "
          f"|ret|>8% days: {[(d.date(), round(v*100,1)) for d, v in rm[rm.abs() > 0.08].items()]}")
    print(f"Duka spot daily {duka.index[0].date()} -> {duka.index[-1].date()} n={len(duka)}")
    print(f"GC=F {gc.index[0].date()} -> {gc.index[-1].date()} n={len(gc)}")
    gold = mt5s[mt5s.index >= "1999-01-01"]   # primary spot series

    # swap models
    sw_const = (SWAP_LONG_NOW, SWAP_SHORT_NOW)
    irx_now = irx.iloc[-1]
    a_l, a_s = -SWAP_LONG_NOW - irx_now, irx_now - SWAP_SHORT_NOW
    sw_rate = (-(irx + a_l), irx - a_s)
    print(f"swap now: long {SWAP_LONG_NOW*100:+.2f}%/yr, short {SWAP_SHORT_NOW*100:+.2f}%/yr; "
          f"T-bill {irx_now*100:.2f}% -> rate-linked model long=-(tbill+{a_l*100:.2f}%), short=tbill-{a_s*100:.2f}%")

    # ---------------------------------------------------------------- 1 cross-source
    hdr("TEST 1  cross-source: Sharpe by period (N=120, 10% vol, 1bp, lag1) | periods: "
        + " ".join(p[0] for p in PERIODS))
    # roll-gap diagnostics GC vs spot
    j = pd.concat([gc.rename("gc"), gold.rename("sp")], axis=1).dropna()
    basis = np.log(j.gc / j.sp)
    db = basis.diff()
    rolls = db[db.abs() > 0.004]
    print(f"GC=F/spot log-basis: mean {basis.mean()*1e4:.0f}bp; days with basis jump >40bp: {len(rolls)} "
          f"(sum {rolls.sum()*100:+.1f}%); total basis drift {basis.iloc[-1]*100-basis.iloc[0]*100:+.1f}% over "
          f"{(j.index[-1]-j.index[0]).days/365:.0f}y; daily ret corr {j.gc.pct_change().corr(j.sp.pct_change()):.3f}")
    # back-adjusted GC: remove basis jumps > 40bp on days where spot didn't move with it
    gr = gc.pct_change()
    spr = gold.pct_change().reindex(gc.index)
    gap = (gr - spr).where((gr - spr).abs() > 0.004, 0).fillna(0)
    gc_adj = (1 + (gr - gap).fillna(0)).cumprod() * gc.iloc[0]
    srcs = {"MT5 spot": gold, "GC=F raw": gc, "GC=F roll-fixed": gc_adj, "Duka spot": duka}
    for name, s in srcs.items():
        for lab, kw in [("L/S", {}), ("LO ", {"long_only": True}), ("BH ", None)]:
            p = bh(s) if kw is None else strat(s, **kw)
            N_CONFIGS += 1
            print(f"  {name:<16} {lab}  {per_row(p)}   full {sharpe(p):+.2f}  (2018+ {sharpe(sl(p,'2018-01-01','2100')):+.2f})")

    # ---------------------------------------------------------------- 2 lookback surface
    hdr("TEST 2  lookback surface on MT5 spot 1999->now: full-sample Sharpe (and min over 6 periods)")
    print("    N  " + "  ".join(f"L/S tv{int(t*100):>2}" for t in (0.05, 0.10, 0.15, 0.20)) + "   |  LO tv10  BH tv10 |  L/S per-period (tv10)")
    for n in NS:
        cells = []
        for t in (0.05, 0.10, 0.15, 0.20):
            p = strat(gold, n=n, tv=t); N_CONFIGS += 1
            cells.append(f"{sharpe(p):+.2f}/{min(sharpe(sl(p,a,b)) for _,a,b in PERIODS):+.2f}")
        lo = strat(gold, n=n, long_only=True); N_CONFIGS += 2
        print(f"  {n:>3}  " + "  ".join(f"{c:>11}" for c in cells) +
              f"   |  {sharpe(lo):+.2f}  {sharpe(bh(gold)):+.2f} |  {per_row(strat(gold, n=n))}")
    print("  LO per-period by N:")
    for n in NS:
        print(f"  {n:>3}  LO {per_row(strat(gold, n=n, long_only=True))}")

    # ---------------------------------------------------------------- 3 trend vs B&H
    hdr("TEST 3  does trend add value beyond vol-scaled buy&hold?  (MT5 spot, N=120, tv10)")
    B = bh(gold)
    LS, LO, SH = strat(gold), strat(gold, long_only=True), strat(gold, short_only=True)
    print("  per-period Sharpe:      " + "  ".join(f"{p[0]:>5}" for p in PERIODS))
    for lab, p in [("BH", B), ("L/S", LS), ("LO", LO), ("short-only", SH)]:
        print(f"  {lab:<22} {per_row(p)}")
    print("  per-period ann return % (L/S - BH, LO - BH):")
    for lab, a, b in PERIODS:
        print(f"    {lab}: BH {sl(B,a,b).mean()*25200:+5.1f}  L/S {sl(LS,a,b).mean()*25200:+5.1f}  "
              f"LO {sl(LO,a,b).mean()*25200:+5.1f}  short {sl(SH,a,b).mean()*25200:+5.1f}  "
              f"| L/S-BH {(sl(LS,a,b)-sl(B,a,b)).mean()*25200:+5.1f}  LO-BH {(sl(LO,a,b)-sl(B,a,b)).mean()*25200:+5.1f}")
    for lab, p in [("L/S", LS), ("LO", LO), ("short-only", SH)]:
        a, be, t = alpha_vs(p, B)
        print(f"  regression {lab:<10} on BH: alpha {a*100:+.2f}%/yr  beta {be:+.2f}  NW t(alpha) {t:+.2f}")
        for nm, x0, x1 in [("1999-2010", "1999", "2011"), ("2011-2026", "2011", "2100")]:
            a, be, t = alpha_vs(sl(p, x0, x1), sl(B, x0, x1))
            print(f"      {nm}: alpha {a*100:+.2f}%/yr beta {be:+.2f} t {t:+.2f}")
    print("  gold bear / drawdown episodes (cum return %):")
    for nm, a, b in [("2008-03..2008-10 GFC dip", "2008-03-17", "2008-10-25"),
                     ("2011-09..2015-12 bear", "2011-09-06", "2015-12-18"),
                     ("2013 crash yr", "2013-01-01", "2014-01-01"),
                     ("2016-07..2018-08", "2016-07-06", "2018-08-16"),
                     ("2020-08..2022-09", "2020-08-07", "2022-09-28"),
                     ("2022-03..2022-11", "2022-03-08", "2022-11-03")]:
        g = sl(gold, a, b)
        f = lambda p: ((1 + sl(p, a, b)).prod() - 1) * 100
        print(f"    {nm:<26} gold {(g.iloc[-1]/g.iloc[0]-1)*100:+6.1f} | BH {f(B):+6.1f}  L/S {f(LS):+6.1f}  "
              f"LO {f(LO):+6.1f}  short {f(SH):+6.1f}")

    # ---------------------------------------------------------------- 4 year by year + regimes
    hdr("TEST 4  year-by-year (ann. return %, MT5 spot, tv10) and regimes")
    dxy = xa["DX-Y.NYB"].dropna()
    yrs = []
    for y in range(2000, 2027):
        a, b = f"{y}-01-01", f"{y+1}-01-01"
        g = sl(gold, a, b); dx = sl(dxy, a, b)
        if len(g) < 50:
            continue
        cum = lambda p: ((1 + sl(p, a, b)).prod() - 1) * 100
        yrs.append(dict(y=y, gold=(g.iloc[-1] / g.iloc[0] - 1) * 100, dxy=(dx.iloc[-1] / dx.iloc[0] - 1) * 100,
                        BH=cum(B), LS=cum(LS), LO=cum(LO), SH=cum(SH)))
    Y = pd.DataFrame(yrs).set_index("y")
    Y["LS-BH"] = Y.LS - Y.BH
    print(Y.round(1).to_string())
    Y["greg"] = np.where(Y.gold > 10, "bull", np.where(Y.gold < -5, "bear", "side"))
    Y["dreg"] = np.where(Y.dxy > 0, "USD up", "USD dn")
    print("\n  mean yearly return by gold regime (bull>+10%, bear<-5%):")
    print(Y.groupby("greg")[["BH", "LS", "LO", "SH", "LS-BH"]].agg(["mean", "count"]).round(1).to_string())
    print("\n  by dollar regime:")
    print(Y.groupby("dreg")[["BH", "LS", "LO", "SH", "LS-BH"]].mean().round(1).to_string())
    print(f"\n  years L/S > 0: {(Y.LS>0).sum()}/{len(Y)}; LO>0: {(Y.LO>0).sum()}/{len(Y)}; "
          f"L/S beats BH: {(Y['LS-BH']>0).sum()}/{len(Y)}; LO beats BH: {(Y.LO>Y.BH).sum()}/{len(Y)}")

    # ---------------------------------------------------------------- 5 placebo + bootstrap
    hdr("TEST 5  placebo (random signs, matched turnover) and block bootstrap  (MT5 spot, N=120, tv10)")
    sig = np.sign(gold.pct_change(120)).iloc[121:]
    flips = (sig != sig.shift()).iloc[1:].mean()
    p_up = (sig > 0).mean()
    # Markov chain matching flip rate (symmetric) and matching flip rate + long fraction (asymmetric)
    # steady state long frac = q_ls/(q_ls+q_sl) = p_up; mean flip rate = 2 p_up q_sl = flips
    q_sl, q_ls = flips / (2 * p_up), flips / (2 * (1 - p_up))
    act = sharpe(LS); act_lo = sharpe(LO)
    print(f"  actual: L/S SR {act:.2f}, LO SR {act_lo:.2f}; sign flip rate {flips*100:.2f}%/day "
          f"(~{flips*252:.1f} flips/yr), long fraction {p_up:.2f}")
    res = {"sym": [], "asym": []}
    idx = gold.index
    nsim = 1000
    for k in range(nsim):
        for mode in ("sym", "asym"):
            a, b = (flips / 2, flips / 2) if mode == "sym" else (q_sl, q_ls)
            u = RNG.random(len(idx))
            s = np.empty(len(idx)); s[0] = 1 if RNG.random() < (0.5 if mode == "sym" else p_up) else -1
            for i in range(1, len(idx)):
                s[i] = -s[i-1] if u[i] < (a if s[i-1] > 0 else b) else s[i-1]
            res[mode].append(sharpe(strat(gold, n=120, sig_override=pd.Series(s, index=idx))))
    for mode, lab in [("sym", "random sign, matched flips, 50% long"),
                      ("asym", f"random sign, matched flips AND {p_up:.0%} long")]:
        arr = np.array(res[mode])
        print(f"  placebo {lab:<42}: median SR {np.median(arr):+.2f}, 95th {np.percentile(arr,95):+.2f}; "
              f"p(placebo>=actual) = {(arr >= act).mean():.3f}")
    df = pd.concat([LS, LO, B], axis=1, keys=["LS", "LO", "B"]).dropna()
    for block in (20, 63, 126):
        bs = {"LS": [], "LO": [], "LS-B": [], "LO-B": []}
        V = df.values
        for _ in range(2000):
            ii = block_boot_idx(len(V), block, RNG)
            x = V[ii]
            bs["LS"].append(x[:, 0].mean() / x[:, 0].std() * math.sqrt(252))
            bs["LO"].append(x[:, 1].mean() / x[:, 1].std() * math.sqrt(252))
            bs["LS-B"].append((x[:, 0] - x[:, 2]).mean() * 252)
            bs["LO-B"].append((x[:, 1] - x[:, 2]).mean() * 252)
        print(f"  block bootstrap (block {block:>3}d): p(SR_LS<=0)={np.mean(np.array(bs['LS'])<=0):.3f}  "
              f"p(SR_LO<=0)={np.mean(np.array(bs['LO'])<=0):.3f}  "
              f"p(LS-BH<=0)={np.mean(np.array(bs['LS-B'])<=0):.3f} [obs {(df.LS-df.B).mean()*25200:+.2f}%/yr]  "
              f"p(LO-BH<=0)={np.mean(np.array(bs['LO-B'])<=0):.3f} [obs {(df.LO-df.B).mean()*25200:+.2f}%/yr]")
    # Sharpe-difference bootstrap (vol-matched comparison)
    dsr = []
    V = df.values
    for _ in range(2000):
        x = V[block_boot_idx(len(V), 63, RNG)]
        s_ = lambda c: x[:, c].mean() / x[:, c].std()
        dsr.append((s_(0) - s_(2)) * math.sqrt(252))
    print(f"  Sharpe(L/S) - Sharpe(BH): obs {sharpe(df.LS)-sharpe(df.B):+.2f}, block-63 bootstrap p(<=0) {np.mean(np.array(dsr)<=0):.3f}")

    # ---------------------------------------------------------------- 6 costs etc
    hdr("TEST 6  costs, swap, rebalance, lag  (MT5 spot, N=120, tv10) â€” Sharpe per period / full / ann%")
    rows = []
    for lab, kw in [("base 1bp lag1 daily", {}), ("3bp", {"cost_bp": 3}), ("10bp", {"cost_bp": 10}),
                    ("lag 2", {"lag": 2}), ("weekly rebal", {"rebal": "weekly"}),
                    ("signal-change only", {"rebal": "signal"}),
                    ("swap const (now)", {"swap": sw_const}), ("swap rate-linked", {"swap": sw_rate}),
                    ("swap rl + 3bp + lag2", {"swap": sw_rate, "cost_bp": 3, "lag": 2})]:
        for side, extra in [("L/S", {}), ("LO", {"long_only": True})]:
            p = strat(gold, **kw, **extra); N_CONFIGS += 1
            to = None
            print(f"  {lab:<22} {side:<3} {per_row(p)}  full {sharpe(p):+.2f}  {p.mean()*25200:+5.1f}%/yr")
    for lab, kw in [("BH swap const", {"swap": sw_const}), ("BH swap rate-linked", {"swap": sw_rate})]:
        p = bh(gold, **kw)
        print(f"  {lab:<26} {per_row(p)}  full {sharpe(p):+.2f}  {p.mean()*25200:+5.1f}%/yr")
    LSr, LOr, Br = strat(gold, swap=sw_rate), strat(gold, long_only=True, swap=sw_rate), bh(gold, swap=sw_rate)
    for lab, p in [("L/S", LSr), ("LO", LOr)]:
        a, be, t = alpha_vs(p, Br)
        print(f"  net of rate-linked swap: {lab} - BH = {(p - Br).mean()*25200:+.2f}%/yr; alpha {a*100:+.2f}% beta {be:.2f} t {t:+.2f}")
    pos = (np.sign(gold.pct_change(120)) * (0.10 / (gold.pct_change().rolling(20).std() * math.sqrt(252))).clip(upper=3))
    print(f"  avg |position| {pos.abs().mean():.2f}x, annual turnover {pos.diff().abs().mean()*252:.1f}x capital "
          f"(cost of 1bp = {pos.diff().abs().mean()*252*1e-4*100:.2f}%/yr)")

    # ---------------------------------------------------------------- 7 walk-forward
    hdr("TEST 7  walk-forward: each year choose N in {40..250} maximising prior-data Sharpe (L/S and LO)")
    grid = [40, 60, 80, 100, 120, 150, 200, 250]
    pre = {(n, lo): strat(gold, n=n, long_only=lo) for n in grid for lo in (False, True)}
    for lo in (False, True):
        for win in ("expanding", "5y"):
            oos, picks = [], []
            for y in range(2004, 2027):
                a = "1999-01-01" if win == "expanding" else f"{y-5}-01-01"
                best = max(grid, key=lambda n: sharpe(sl(pre[(n, lo)], a, f"{y}-01-01")))
                picks.append(best)
                oos.append(sl(pre[(best, lo)], f"{y}-01-01", f"{y+1}-01-01"))
            o = pd.concat(oos)
            fx = sl(pre[(120, lo)], "2004-01-01", "2100")
            b_ = sl(B, "2004-01-01", "2100")
            N_CONFIGS += 1
            print(f"  {'LO ' if lo else 'L/S'} {win:<9} OOS 2004+: SR {sharpe(o):+.2f} ({o.mean()*25200:+.1f}%/yr) | fixed N120 "
                  f"SR {sharpe(fx):+.2f} | BH {sharpe(b_):+.2f}  picks {picks}")
            yy = pd.DataFrame({"wf": o.groupby(o.index.year).apply(lambda p: ((1+p).prod()-1)*100),
                               "N120": fx.groupby(fx.index.year).apply(lambda p: ((1+p).prod()-1)*100),
                               "BH": b_.groupby(b_.index.year).apply(lambda p: ((1+p).prod()-1)*100)})
            if win == "expanding":
                print("    yearly %: " + " ".join(f"{y}:{r.wf:+.0f}/{r.N120:+.0f}/{r.BH:+.0f}" for y, r in yy.iterrows()))
            print(f"    years WF>0 {(yy.wf>0).sum()}/{len(yy)}, WF>BH {(yy.wf>yy.BH).sum()}/{len(yy)}")

    # ---------------------------------------------------------------- 8 cross-asset
    hdr("TEST 8  cross-asset: same rule (N=120, tv10, 1bp) 2001->now. Futures are raw Yahoo continuous (roll gaps!)")
    print(f"  {'asset':<10} {'BH':>6} {'L/S':>6} {'LO':>6} {'SH':>6} | L/S alpha%/t vs BH | L/S 2001-12 / 2013-26 | surface L/S N=40..250")
    xr = {}
    for t in ["GOLD(MT5)", "GC=F", "SI=F", "PL=F", "HG=F", "CL=F", "NG=F", "ZC=F", "ZW=F", "EURUSD=X", "JPY=X",
              "AUDUSD=X", "GBPUSD=X", "^GSPC", "^GDAXI", "^N225", "ZN=F", "ZB=F", "DX-Y.NYB"]:
        s = gold if t == "GOLD(MT5)" else xa[t].dropna()
        s = s[s.index >= "2000-01-01"]
        s = s[s > 0]
        ls, lo, sh, b = strat(s), strat(s, long_only=True), strat(s, short_only=True), bh(s)
        ls, lo, sh, b = [sl(x, "2001-01-01", "2100") for x in (ls, lo, sh, b)]
        a, be, tt = alpha_vs(ls, b)
        surf = [sharpe(sl(strat(s, n=n), "2001-01-01", "2100")) for n in (40, 80, 120, 200, 250)]
        xr[t] = sharpe(ls)
        print(f"  {t:<10} {sharpe(b):+6.2f} {sharpe(ls):+6.2f} {sharpe(lo):+6.2f} {sharpe(sh):+6.2f} | "
              f"{a*100:+5.1f} / {tt:+5.2f}       | {sharpe(sl(ls,'2001','2013')):+.2f} / {sharpe(sl(ls,'2013','2100')):+.2f}"
              f"          | " + " ".join(f"{v:+.2f}" for v in surf))
    peers = {k: v for k, v in xr.items() if k not in ("GOLD(MT5)", "GC=F")}
    pv = np.array(list(peers.values()))
    print(f"  peer L/S Sharpe: mean {pv.mean():+.2f}, median {np.median(pv):+.2f}; gold {xr['GOLD(MT5)']:+.2f} ranks "
          f"{(pv > xr['GOLD(MT5)']).sum()+1}/{len(pv)+1}")
    # equal-risk diversified portfolio of trend sleeves (ex gold) and with gold
    sleeves = {t: sl(strat(xa[t].dropna()[lambda s: (s.index >= "2000-01-01") & (s > 0)]), "2001-01-01", "2100")
               for t in peers}
    P = pd.DataFrame(sleeves).dropna(how="all").fillna(0)
    port = P.mean(axis=1)
    print(f"  equal-weight trend portfolio ex-gold ({len(peers)} assets): SR {sharpe(port):+.2f}; "
          f"corr(gold L/S, portfolio) {pd.concat([sl(LS,'2001','2100'), port], axis=1).dropna().corr().iloc[0,1]:+.2f}")

    # ---------------------------------------------------------------- 9 Monte Carlo
    hdr("TEST 9  Monte Carlo: 63-day block bootstrap of daily P&L (MT5 spot, N=120, 3bp, rate-linked swap, lag1), 5000 x 10y paths")
    for side, lo in (("L/S", False), ("LO", True)):
        for t in (0.10, 0.15, 0.20):
            p = strat(gold, tv=t, long_only=lo, cost_bp=3, swap=sw_rate).dropna().values
            T = 2520
            ann, mdd, tuw, l1y = [], [], [], []
            for _ in range(5000):
                x = p[block_boot_idx(T, 63, RNG, src=len(p))]
                eq = np.cumprod(1 + x)
                ann.append(eq[-1] ** (252 / T) - 1)
                pk = np.maximum.accumulate(eq)
                mdd.append((1 - eq / pk).max())
                # longest underwater spell: gap between successive new highs
                hi = np.flatnonzero(np.r_[True, eq[1:] >= pk[:-1]])
                tuw.append(max(np.diff(np.r_[hi, T]).max() - 1, 0) / 252)
                l1y.append(np.min(eq[252:] / eq[:-252]) - 1)
            ann, mdd, tuw = map(np.array, (ann, mdd, tuw))
            hist = stats(pd.Series(p))
            print(f"  {side:<3} tv{int(t*100):>2}: hist CAGR~{hist['ann']*100:+.1f}% vol {hist['vol']*100:.1f}% DD {hist['dd']*100:.0f}% | "
                  f"MC CAGR p5/p50/p95 {np.percentile(ann,5)*100:+.1f}/{np.percentile(ann,50)*100:+.1f}/{np.percentile(ann,95)*100:+.1f}% "
                  f"| maxDD p50/p95 {np.percentile(mdd,50)*100:.0f}/{np.percentile(mdd,95)*100:.0f}% "
                  f"| longest underwater p50/p95 {np.percentile(tuw,50):.1f}/{np.percentile(tuw,95):.1f}y "
                  f"| P(10y CAGR<0) {np.mean(ann<0):.2f}")

    # ---------------------------------------------------------------- 10 multiple testing
    hdr("TEST 10  multiple testing")
    prior = 2 * 6 + 3 * 1 + 1      # xau_trend_vol.py: 6 N x L/S,LO + 3 vol-filter variants + BH
    total = N_CONFIGS + prior
    from math import erf
    t_obs = sharpe(LS) * math.sqrt(len(LS) / 252)
    t_lo = sharpe(LO) * math.sqrt(len(LO) / 252)
    pnorm = lambda z: 0.5 * (1 + erf(z / math.sqrt(2)))
    for lab, tval in (("L/S", t_obs), ("LO", t_lo)):
        p1 = 1 - pnorm(tval)
        print(f"  {lab} N=120 1999+: SR {tval/math.sqrt(len(LS)/252):.2f}, t {tval:.2f}, one-sided p {p1:.4f}; "
              f"Bonferroni x{total} -> {min(1, p1*total):.3f}; Sidak -> {1-(1-p1)**total:.3f}")
    print(f"  configurations evaluated: {N_CONFIGS} here + ~{prior} in xau_trend_vol.py = {total} "
          f"(highly correlated; effective count much smaller, ~10-15 distinct N/side combos)")
    print(f"  Bonferroni with ~15 effective tests: L/S p {min(1,(1-pnorm(t_obs))*15):.3f}, LO p {min(1,(1-pnorm(t_lo))*15):.3f}")
    # alpha-over-BH t-stat is the economically relevant test
    a, be, t = alpha_vs(LS, B)
    print(f"  L/S alpha over BH t={t:.2f} -> one-sided p {1-pnorm(t):.3f}, x15 -> {min(1,(1-pnorm(t))*15):.3f}")
    a, be, t = alpha_vs(LO, B)
    print(f"  LO  alpha over BH t={t:.2f} -> one-sided p {1-pnorm(t):.3f}, x15 -> {min(1,(1-pnorm(t))*15):.3f}")


if __name__ == "__main__":
    main()
