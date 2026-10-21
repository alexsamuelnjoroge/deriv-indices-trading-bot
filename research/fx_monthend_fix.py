"""
Adversarial battery for MONTH-END WM/Reuters 4pm London FIX FLOWS in USD majors.

Hypothesis (Melvin & Prins 2015; Evans 2018): foreign-equity holders who hedge FX rebalance at the
last-business-day 4pm London fix.  If foreign equities outperformed US equities over the month, their
hedges must be enlarged -> sell foreign currency / BUY USD into the fix (and vice versa).

Trades (all intraday, flat same day, London clock via tz Europe/London so DST is exact):
  a) SIGNAL: sig = MTD(SPY) - MTD(EFA), both USD total-return closes through the PREVIOUS US session
     (known by 10:00 London on the event day).  sig < 0 (foreign outperformed) -> long USD, else short USD.
     Traded on EURUSD GBPUSD USDJPY AUDUSD USDCAD USDCHF + equal-weight USD basket ("DXY-like").
     Primary pre-specified window 10:00 -> 15:55 London.  Surface: entry 10/11/12/13/14/15:00, exit 15:55/16:00.
     Alternative signal constructions: (i) ^GSPC vs local indices converted to USD (STOXX50E, FTSE, N225),
     (ii) pair-specific ETFs (VGK/EWU/EWJ/EWA/EWC/EWL vs SPY).
  b) UNCONDITIONAL USD drift on month-end days: 14:00->16:00 and 16:00->17:00.
  c) POST-FIX REVERSAL: fade the 15:00->16:00 move from 16:05 -> 17:00 / 18:00.
  d) Quarter-end and year-end subsets.
Event day = last weekday of each month that has >= 60 Dukascopy bars between 08:00 and 17:00 London.

Data: Dukascopy M5 bid candles 2010->2026-10 (data/duka/frx<PAIR>_M5_<YEAR>.csv via research/_fetch_duka_fx.py);
second source: MT5 demo H1 (data/pro/mt5_<PAIR>_H1.csv, server time = New York + 7h) for EURUSD GBPUSD USDJPY AUDUSD;
equities: yfinance daily (research/cache_fxfix_equities.csv).
Costs (raw-spread MT5 account, round trip, in bp of notional): commission $7/100k = 0.70bp +
full spread: EURUSD 0.1pip, GBPUSD 0.3, USDJPY 0.2, AUDUSD 0.2, USDCAD 0.4, USDCHF 0.4  (x1); fix-time
widening tested at x2/x3 of the whole cost.
"""
import os
import sys
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
DUKA = ROOT / "data" / "duka"
RNG = np.random.default_rng(7)
PAIRS = os.environ.get("FXFIX_PAIRS", "EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD,USDCHF").split(",")
USD_BASE = {"USDJPY", "USDCAD", "USDCHF"}
PIP = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "USDJPY": 1e-2, "AUDUSD": 1e-4, "USDCAD": 1e-4, "USDCHF": 1e-4}
SPREAD_PIPS = {"EURUSD": 0.1, "GBPUSD": 0.3, "USDJPY": 0.2, "AUDUSD": 0.2, "USDCAD": 0.4, "USDCHF": 0.4}
COMM_BP = 0.70
PAIR_ETF = {"EURUSD": "VGK", "GBPUSD": "EWU", "USDJPY": "EWJ", "AUDUSD": "EWA", "USDCAD": "EWC", "USDCHF": "EWL"}
LON = "Europe/London"
START_YEAR, END_DATE = 2010, pd.Timestamp("2026-10-20")


def P(*a):
    print(*a, flush=True)


def tstat(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return np.nan
    return x.mean() / x.std(ddof=1) * math.sqrt(len(x))


# ----------------------------------------------------------------------------- data
def load_duka(pair):
    parts = []
    for y in range(START_YEAR, 2027):
        f = DUKA / f"frx{pair}_M5_{y}.csv"
        if f.exists():
            d = pd.read_csv(f, index_col=0, parse_dates=True)
            if len(d):
                parts.append(d)
    d = pd.concat(parts)
    d = d[~d.index.duplicated()].sort_index()
    d.index = d.index.tz_convert(LON)
    d = d[d.index.dayofweek < 5]
    return d


def price_grid(d):
    """close-price at the END of each 5-min bar, indexed by (London date, minute-of-day of bar end)."""
    end = d.index + pd.Timedelta(minutes=5)
    s = pd.Series(d["close"].values, index=end)
    s = s[(s.index.hour >= 7) & (s.index.hour <= 19)]
    df = pd.DataFrame({"px": s.values, "date": s.index.tz_localize(None).normalize(),
                       "mod": s.index.hour * 60 + s.index.minute})
    g = df.pivot_table(index="date", columns="mod", values="px", aggfunc="last")
    g = g.reindex(columns=range(7 * 60, 20 * 60 + 1, 5))
    return g.ffill(axis=1, limit=2)       # tolerate <=10 min gaps


def load_mt5_h1(pair):
    f = ROOT / "data" / "pro" / f"mt5_{pair}_H1.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    srv = pd.to_datetime(d.epoch, unit="s")
    ny = (srv - pd.Timedelta(hours=7)).dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    d["loc"] = ny.dt.tz_convert(LON)
    d = d.dropna(subset=["loc"])
    d["date"] = d["loc"].dt.tz_localize(None).dt.normalize()
    d["h"] = d["loc"].dt.hour
    # price at the START of hour h = open of bar h ; price at end of hour h = close of bar h
    o = d.pivot_table(index="date", columns="h", values="open", aggfunc="last")
    c = d.pivot_table(index="date", columns="h", values="close", aggfunc="last")
    return o, c


# ----------------------------------------------------------------------------- build
P("=" * 100)
P("FX MONTH-END 4PM LONDON FIX battery   (run", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), ")")
P("=" * 100)
G = {}
for p in PAIRS:
    d = load_duka(p)
    G[p] = price_grid(d)
    yrs = sorted(set(G[p].index.year))
    P(f"  {p}: Dukascopy M5 {d.index.min():%Y-%m-%d} -> {d.index.max():%Y-%m-%d}  bars={len(d):,}  days={len(G[p])}"
      f"  years={yrs[0]}-{yrs[-1]} (missing: {[y for y in range(yrs[0], yrs[-1] + 1) if y not in yrs]})")

# common trading-day calendar: dates where every pair has a 10:00 and 16:00 price
days = None
for p in PAIRS:
    g = G[p]
    ok = g.index[(g[[600, 960]].notna().all(1)) & (g.notna().sum(1) >= 60)]
    days = ok if days is None else days.intersection(ok)
days = days[(days.dayofweek < 5) & (days <= END_DATE)]
cal = pd.Series(days, index=days)
last_bd = cal.groupby([days.year, days.month]).max().values
last_bd = pd.DatetimeIndex(last_bd)
last_bd = last_bd[last_bd < pd.Timestamp("2026-10-01")]      # Oct-2026 month-end not reached yet
P(f"  common calendar {days.min():%Y-%m-%d} -> {days.max():%Y-%m-%d}: {len(days)} days, month-end events {len(last_bd)}")
# sanity: event should be the true last weekday (flag holidays like Good Friday / Dec-31 thinness)
true_last = pd.DatetimeIndex([pd.Timestamp(y, m, 1) + pd.offsets.BMonthEnd(0) for y, m in zip(last_bd.year, last_bd.month)])
diff = last_bd[last_bd != true_last]
P(f"  events that are NOT the calendar last weekday (holiday/data gap): {len(diff)} -> {[x.strftime('%Y-%m-%d') for x in diff][:12]}")

# equities
EQ = pd.read_csv(ROOT / "research" / "cache_fxfix_equities.csv", index_col=0, parse_dates=True).sort_index()
# local indices in USD
def fxusd(tk):
    import yfinance as yf  # noqa
eqx = EQ.copy()
try:
    fxc = pd.read_csv(ROOT / "research" / "cache_xasset.csv", index_col=0, parse_dates=True)
    eqx["STOXX_USD"] = EQ["^STOXX50E"] * fxc["EURUSD=X"].reindex(EQ.index).ffill()
    eqx["FTSE_USD"] = EQ["^FTSE"] * fxc["GBPUSD=X"].reindex(EQ.index).ffill()
    eqx["N225_USD"] = EQ["^N225"] / fxc["JPY=X"].reindex(EQ.index).ffill()
except Exception as e:  # noqa
    P("  local-index USD conversion unavailable:", e)


def mtd(series, day):
    """month-to-date return using closes strictly BEFORE `day` (known at 10:00 London)."""
    s = series.dropna()
    prev = s[s.index < day]
    if prev.empty:
        return np.nan
    base = s[s.index < pd.Timestamp(day.year, day.month, 1)]
    if base.empty or prev.index[-1] < pd.Timestamp(day.year, day.month, 1):
        return np.nan
    return prev.iloc[-1] / base.iloc[-1] - 1


def signals_for(day_list):
    rows = []
    for dday in day_list:
        r = dict(date=dday, spy=mtd(eqx["SPY"], dday), efa=mtd(eqx["EFA"], dday), gspc=mtd(eqx["^GSPC"], dday))
        loc = [mtd(eqx[c], dday) for c in ["STOXX_USD", "FTSE_USD", "N225_USD"] if c in eqx]
        r["loc"] = np.nanmean(loc) if loc else np.nan
        for p, etf in PAIR_ETF.items():
            r["etf_" + p] = mtd(eqx[etf], dday)
        rows.append(r)
    s = pd.DataFrame(rows).set_index("date")
    s["sig"] = s["spy"] - s["efa"]                    # >0 US outperformed -> short USD
    s["sig_loc"] = s["gspc"] - s["loc"]
    for p in PAIRS:
        s["sig_" + p] = s["spy"] - s["etf_" + p]
    return s


SIG = signals_for(last_bd)
P(f"  signal SPY-EFA MTD (to t-1): mean {SIG.sig.mean()*1e2:+.2f}%  |sig| median {SIG.sig.abs().median()*1e2:.2f}%  "
  f"frac US-outperf {np.mean(SIG.sig > 0):.2f}; corr with sig_loc {SIG[['sig','sig_loc']].corr().iloc[0,1]:.2f}")


def cost_bp(p, mult=1.0):
    px_typ = {"EURUSD": 1.2, "GBPUSD": 1.4, "USDJPY": 115, "AUDUSD": 0.8, "USDCAD": 1.3, "USDCHF": 0.95}[p]
    return mult * (SPREAD_PIPS[p] * PIP[p] / px_typ * 1e4 + COMM_BP)


P("  round-trip cost x1 (bp): " + "  ".join(f"{p} {cost_bp(p):.2f}" for p in PAIRS))


def usd_ret(p, dates, t0, t1, src=None):
    """USD-long log return in bp between minute-of-day t0 and t1 (bar-end prices) for each date."""
    g = (src or G)[p]
    g = g.reindex(dates)
    r = np.log(g[t1] / g[t0]) * 1e4
    return r if p in USD_BASE else -r


def hm(s):
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def side(sig):
    return np.where(sig < 0, 1.0, -1.0)        # foreign outperformed -> long USD


def trade_table(dates, sigser, t0, t1, cmult=1.0, pairs=PAIRS, src=None):
    """net bp per pair per event for the signal trade."""
    out = {}
    pos = pd.Series(side(sigser.reindex(dates).values), index=dates)
    pos[sigser.reindex(dates).isna().values] = np.nan
    for p in pairs:
        out[p] = pos * usd_ret(p, dates, t0, t1, src) - cost_bp(p, cmult)
    df = pd.DataFrame(out)
    df["BASKET"] = df[list(pairs)].mean(1)
    return df


def summ(df, col="BASKET"):
    x = df[col].dropna()
    return f"n={len(x):3d} mean={x.mean():+6.2f}bp t={tstat(x):+5.2f} hit={np.mean(x > 0):.2f}"


VERDICT = {}
# ============================================================================= A) primary signal trade
P("\n" + "=" * 100)
P("A) SIGNAL TRADE  (sig = MTD SPY - MTD EFA through t-1; long USD if foreign outperformed)")
P("=" * 100)
E0, X0 = hm("10:00"), hm("15:55")
A = trade_table(last_bd, SIG.sig, E0, X0)
Ag = trade_table(last_bd, SIG.sig, E0, X0, cmult=0.0)
P(" primary 10:00->15:55, per pair (gross | net x1):")
for c in PAIRS + ["BASKET"]:
    P(f"   {c:7s} gross {summ(Ag, c)}  |  net {summ(A, c)}")
for nm, sg in [("sig_loc (^GSPC vs STOXX/FTSE/N225 in USD)", SIG.sig_loc)]:
    B = trade_table(last_bd, sg, E0, X0)
    P(f"  alt signal {nm}: basket net {summ(B)}")
# pair-specific
ps = {}
for p in PAIRS:
    ps[p] = trade_table(last_bd, SIG["sig_" + p], E0, X0, pairs=[p])[p]
PS = pd.DataFrame(ps)
PS["BASKET"] = PS.mean(1)
P(f"  alt signal pair-specific ETF (VGK/EWU/EWJ/EWA/EWC/EWL): basket net {summ(PS)}  per pair: " +
  " ".join(f"{p}:{PS[p].mean():+.1f}" for p in PAIRS))
# magnitude-weighted (continuous) version
w = (-SIG.sig / SIG.sig.abs().rolling(24, min_periods=6).mean().shift(1)).clip(-3, 3)
MW = pd.DataFrame({p: w * usd_ret(p, last_bd, E0, X0) - w.abs() * cost_bp(p) for p in PAIRS})
MW["BASKET"] = MW.mean(1)
P(f"  magnitude-weighted (|w| scaled by trailing mean |sig|, cap 3): basket net {summ(MW)}")

# ============================================================================= B) unconditional
P("\n" + "=" * 100)
P("B) UNCONDITIONAL USD-LONG on month-end days (gross bp; t vs zero) and vs all other days")
P("=" * 100)
others = days[~days.isin(last_bd)]
for a_, b_ in [("10:00", "15:55"), ("14:00", "16:00"), ("15:00", "16:00"), ("15:30", "16:00"), ("16:00", "17:00"), ("16:05", "18:00")]:
    me = pd.DataFrame({p: usd_ret(p, last_bd, hm(a_), hm(b_)) for p in PAIRS})
    ot = pd.DataFrame({p: usd_ret(p, others, hm(a_), hm(b_)) for p in PAIRS})
    me["B"], ot["B"] = me.mean(1), ot.mean(1)
    P(f"  {a_}->{b_}: month-end basket {me.B.mean():+6.2f}bp t={tstat(me.B):+5.2f} (n={me.B.notna().sum()})  "
      f"| other days {ot.B.mean():+5.2f}bp t={tstat(ot.B):+5.2f} | per pair ME: " +
      " ".join(f"{p[:3] if p not in USD_BASE else p[3:]}:{me[p].mean():+.1f}" for p in PAIRS))
    if a_ == "14:00":
        for nm, mask in [("quarter-end", last_bd.month.isin([3, 6, 9, 12])), ("year-end", last_bd.month == 12)]:
            P(f"      {nm:11s}: basket {me.B[mask].mean():+6.2f}bp t={tstat(me.B[mask]):+5.2f} n={mask.sum()}")
    if a_ == "16:00":
        VERDICT["b_post_fix_unc"] = (me.B.mean(), tstat(me.B))

# signal-conditioned behaviour around the fix: pre-fix vs post-fix
P("  signal-conditioned USD move (side*ret, gross): ")
for a_, b_ in [("10:00", "14:00"), ("14:00", "15:55"), ("15:55", "16:05"), ("16:05", "17:00"), ("16:05", "18:00")]:
    t = trade_table(last_bd, SIG.sig, hm(a_), hm(b_), cmult=0)
    P(f"     {a_}->{b_}: {summ(t)}")

# fix-window detail (descriptive; spreads widen in the window so cost x3 shown)
P("  FIX WINDOW 15:55->16:05, signal-conditioned (predicted-direction USD move), by pair / era / placebo days:")
FW = trade_table(last_bd, SIG.sig, hm("15:55"), hm("16:05"), cmult=0)
P("     gross per pair: " + " ".join(f"{p}:{FW[p].mean():+.2f}(t{tstat(FW[p]):+.1f})" for p in PAIRS))
for k, m in {"2010-16": FW.index.year <= 2016, "2017-21": (FW.index.year >= 2017) & (FW.index.year <= 2021),
             "2022+": FW.index.year >= 2022}.items():
    P(f"     {k}: {summ(FW[m])}")
FW3 = trade_table(last_bd, SIG.sig, hm("15:55"), hm("16:05"), cmult=3)
P(f"     net x3 cost: {summ(FW3)}  (sign-flipped strategy net x3: {(-FW.BASKET - np.mean([cost_bp(p,3) for p in PAIRS])).mean():+.2f}bp)")
bd_ = pd.DataFrame({"d": days}); bd_["ym"] = bd_.d.dt.to_period("M")
bd_["k"] = bd_.groupby("ym").cumcount(ascending=False)
for kk in [1, 2]:
    dl = pd.DatetimeIndex(bd_.d[bd_.k == kk]); dl = dl[dl < pd.Timestamp("2026-10-01")]
    P(f"     placebo {kk+1}-last BD: {summ(trade_table(dl, signals_for(dl).sig, hm('15:55'), hm('16:05'), cmult=0))}")
fixabs = pd.DataFrame({p: usd_ret(p, last_bd, hm("15:55"), hm("16:05")).abs() for p in PAIRS}).mean(1)
othabs = pd.DataFrame({p: usd_ret(p, others, hm("15:55"), hm("16:05")).abs() for p in PAIRS}).mean(1)
P(f"     mean |move| 15:55->16:05 month-end {fixabs.mean():.2f}bp vs other days {othabs.mean():.2f}bp")

# ============================================================================= C) post-fix reversal
P("\n" + "=" * 100)
P("C) POST-FIX REVERSAL: fade sign of 15:00->16:00 move, hold 16:05 -> 17:00 / 18:00 (net x1)")
P("=" * 100)
REV = {}
for x_ in ["17:00", "18:00"]:
    rows = {}
    for p in PAIRS:
        pre = usd_ret(p, last_bd, hm("15:00"), hm("16:00"))
        post = usd_ret(p, last_bd, hm("16:05"), hm(x_))
        rows[p] = -np.sign(pre) * post - cost_bp(p)
    R = pd.DataFrame(rows)
    R["BASKET"] = R[PAIRS].mean(1)
    REV[x_] = R
    # placebo: same on other days
    rows = {}
    for p in PAIRS:
        pre = usd_ret(p, others, hm("15:00"), hm("16:00"))
        post = usd_ret(p, others, hm("16:05"), hm(x_))
        rows[p] = -np.sign(pre) * post - cost_bp(p)
    Ro = pd.DataFrame(rows)
    Ro["BASKET"] = Ro.mean(1)
    P(f"  exit {x_}: month-end {summ(R)} | other days {summ(Ro)}")
    P("     per pair ME: " + " ".join(f"{p}:{R[p].mean():+.1f}(t{tstat(R[p]):+.1f})" for p in PAIRS))
    qm = last_bd.month.isin([3, 6, 9, 12])
    P(f"     quarter-ends: {summ(R[qm])}  non-QE: {summ(R[~qm])}")

# ============================================================================= battery on primary
P("\n" + "=" * 100)
P("BATTERY on (a) primary: SPY-EFA signal, 10:00->15:55, basket of 6 USD pairs, net x1")
P("=" * 100)
res = {}
yr = A.index.year

# 1 two sources
P("\n[1] TWO SOURCES: Dukascopy M5 vs MT5 H1 (hourly: entry H:00 -> exit 16:00, prices = H1 open/close)")
src_ok = []
for p in ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]:
    m = load_mt5_h1(p) if p in G else None
    if m is None:
        continue
    o, c = m
    # MT5 timezone sanity: correlate hourly returns with Dukascopy at offsets -1,0,+1
    gd = G[p]
    common = gd.index.intersection(o.index)
    for e in [10, 14]:
        dd = last_bd.intersection(o.index)
        mt = np.log(c.reindex(dd)[15] / o.reindex(dd)[e]) * 1e4
        mt = mt if p in USD_BASE else -mt
        du = usd_ret(p, dd, e * 60, 16 * 60)
        pos = pd.Series(side(SIG.sig.reindex(dd).values), index=dd)
        a1, a2 = (pos * du - cost_bp(p)), (pos * mt - cost_bp(p))
        ok = a1.notna() & a2.notna()
        cor = np.corrcoef(du[ok], mt[ok])[0, 1]
        src_ok.append((np.sign(a1[ok].mean()) == np.sign(a2[ok].mean())) and cor > 0.9)
        P(f"   {p} {e:02d}:00->16:00 n={ok.sum()} corr(duka,mt5)={cor:.3f}  signal-trade net duka {a1[ok].mean():+.2f}bp vs mt5 {a2[ok].mean():+.2f}bp")
    # offset check over all days 2016+
    hrs = []
    for off in [-1, 0, 1]:
        dd = common[common.year >= 2012]
        h = 13
        mt = np.log(c.reindex(dd)[h] / o.reindex(dd)[h])
        du = np.log(gd.reindex(dd)[(h + 1 + off) * 60] / gd.reindex(dd)[(h + off) * 60])
        ok = mt.notna() & du.notna()
        hrs.append(f"off{off:+d}:{np.corrcoef(mt[ok], du[ok])[0,1]:.2f}")
    P(f"      timezone check {p} 13:00 bar corr vs duka shifted: {' '.join(hrs)}")
res[1] = all(src_ok)
P(f"   -> {'PASS' if res[1] else 'FAIL'} (all pairs/windows: corr>0.9 and same sign of net mean)")

# 2 surface
P("\n[2] PARAMETER SURFACE: basket net bp (t) by entry x exit")
surf = {}
for e in ["10:00", "11:00", "12:00", "13:00", "14:00", "15:00"]:
    row = []
    for x in ["15:55", "16:00"]:
        t = trade_table(last_bd, SIG.sig, hm(e), hm(x))
        surf[(e, x)] = (t.BASKET.mean(), tstat(t.BASKET))
        row.append(f"{x}: {t.BASKET.mean():+6.2f} (t{tstat(t.BASKET):+5.2f})")
    P(f"   entry {e}  " + "   ".join(row))
vals = np.array([v[0] for v in surf.values()])
ts = np.array([v[1] for v in surf.values()])
res[2] = (np.mean(vals > 0) >= 0.8) and (np.mean(ts > 1) >= 0.5)
P(f"   positive cells {np.mean(vals>0):.0%}, cells t>1 {np.mean(ts>1):.0%} -> {'PASS' if res[2] else 'FAIL'}")

# 3 OOS
P("\n[3] OUT-OF-SAMPLE eras and year by year (basket net)")
eras = {"DEV 2010-16": (yr <= 2016), "VAL 2017-21": (yr >= 2017) & (yr <= 2021), "RECENT 2022+": (yr >= 2022)}
em = []
for k, m in eras.items():
    P(f"   {k:13s} {summ(A[m])}")
    em.append(A.BASKET[m].mean())
ys = A.BASKET.groupby(yr).sum()
P("   by year (sum bp): " + " ".join(f"{y}:{v:+.0f}" for y, v in ys.items()))
res[3] = all(v > 0 for v in em) and np.mean(ys > 0) >= 0.6
P(f"   eras positive {sum(v>0 for v in em)}/3, years positive {np.mean(ys>0):.0%} -> {'PASS' if res[3] else 'FAIL'}")

# 4 regimes
P("\n[4] REGIMES")
qe = A.index.month.isin([3, 6, 9, 12])
big = SIG.sig.abs().reindex(A.index) > SIG.sig.abs().median()
spyabs = SIG.spy.abs().reindex(A.index)
crisis = spyabs > spyabs.quantile(0.8)
reg = {"quarter-end": qe, "non-QE": ~qe, "year-end": A.index.month == 12, "big |sig|": big.values,
       "small |sig|": ~big.values, "crisis (|SPY MTD| top20%)": crisis.values, "calm": ~crisis.values}
rm = {}
for k, m in reg.items():
    P(f"   {k:28s} {summ(A[m])}")
    rm[k] = A.BASKET[m].mean()
res[4] = rm["quarter-end"] > 0 and rm["big |sig|"] > 0 and rm["non-QE"] > 0 and rm["calm"] > 0
P(f"   -> {'PASS' if res[4] else 'FAIL'} (need QE, non-QE, big-|sig| and calm all > 0)")

# 5 placebo
P("\n[5] PLACEBO")
# (i) same construction on other business-day-of-month positions
bd = pd.DataFrame({"d": days})
bd["ym"] = bd.d.dt.to_period("M")
bd["k_from_end"] = bd.groupby("ym").cumcount(ascending=False)
bd["k_from_start"] = bd.groupby("ym").cumcount()
plc = {}
for nm, sel in [("2nd-last BD", bd.k_from_end == 1), ("3rd-last BD", bd.k_from_end == 2),
                ("5th-last BD", bd.k_from_end == 4),("mid-month (10th BD)", bd.k_from_start == 9)]:
    dl = pd.DatetimeIndex(bd.d[sel])
    dl = dl[dl < pd.Timestamp("2026-10-01")]
    s = signals_for(dl)
    t = trade_table(dl, s.sig, E0, X0)
    plc[nm] = t.BASKET.mean()
    P(f"   {nm:20s} {summ(t)}")
# (ii) shuffled signals
true_m = A.BASKET.mean()
ret_g = pd.DataFrame({p: usd_ret(p, last_bd, E0, X0) for p in PAIRS})
cmean = np.mean([cost_bp(p) for p in PAIRS])
sv = SIG.sig.reindex(last_bd).values
sh = []
for _ in range(5000):
    ps_ = RNG.permutation(sv)
    sh.append(np.nanmean(side(ps_)[:, None] * ret_g.values, axis=1).mean() - cmean)
sh = np.array(sh)
pval = np.mean(sh >= true_m)
P(f"   shuffled-signal (5000 perms): true {true_m:+.2f}bp, perm mean {sh.mean():+.2f}, 95th pct {np.percentile(sh,95):+.2f}, p={pval:.3f}")
res[5] = pval < 0.05 and all(true_m > v for v in plc.values())
P(f"   -> {'PASS' if res[5] else 'FAIL'}")

# 6 costs & delays
P("\n[6] COSTS x1/x2/x3 and entry delays")
for cm in [0, 1, 2, 3]:
    P(f"   cost x{cm}: {summ(trade_table(last_bd, SIG.sig, E0, X0, cmult=cm))}")
for dl in [5, 15, 30, 60]:
    P(f"   entry delay +{dl:2d}min: {summ(trade_table(last_bd, SIG.sig, E0 + dl, X0))}")
P(f"   exit 15:50 / 15:45 : {summ(trade_table(last_bd, SIG.sig, E0, hm('15:50')))} / {summ(trade_table(last_bd, SIG.sig, E0, hm('15:45')))}")
c2 = trade_table(last_bd, SIG.sig, E0, X0, cmult=2).BASKET
res[6] = c2.mean() > 0 and tstat(c2) > 1.0
P(f"   -> {'PASS' if res[6] else 'FAIL'} (x2 cost basket mean>0 and t>1)")

# 7 walk-forward
P("\n[7] WALK-FORWARD (expanding; each year choose the entry/exit cell and the signal SIGN from prior years; trade only if prior t>1)")
cells = list(surf.keys())
TT = {c: trade_table(last_bd, SIG.sig, hm(c[0]), hm(c[1])).BASKET for c in cells}
TTg = {c: trade_table(last_bd, SIG.sig, hm(c[0]), hm(c[1]), cmult=0).BASKET for c in cells}
wf = []
for y in range(2013, 2027):
    tr = yr < y
    best, bt = None, 0
    for c in cells:
        x = TTg[c][tr]
        t_ = tstat(x)
        if abs(t_) > abs(bt):
            best, bt = c, t_
    te = yr == y
    if best is None or abs(bt) < 1.0:
        wf.append((y, None, 0.0, 0))
        continue
    sgn = np.sign(bt)
    gross = sgn * TTg[best][te]
    cst = TTg[best][te] - TT[best][te]
    wf.append((y, best, (gross - cst).sum(), te.sum()))
for y, b, v, n in wf:
    P(f"   {y}: {('flat' if b is None else b[0]+'->'+b[1])} n={n} sum net {v:+.1f}bp")
tot = sum(v for _, _, v, _ in wf)
res[7] = tot > 0 and np.mean([v > 0 for _, b, v, _ in wf if b is not None] or [0]) >= 0.5
P(f"   WF total {tot:+.1f}bp -> {'PASS' if res[7] else 'FAIL'}")

# 8 cross-pair
P("\n[8] CROSS-PAIR consistency (primary, net x1)")
pm = {p: A[p].mean() for p in PAIRS}
P("   " + "  ".join(f"{p}:{v:+.2f}(t{tstat(A[p]):+.1f})" for p, v in pm.items()))
gross_sgn = np.sign(Ag[PAIRS])
agree = (gross_sgn.sum(1).abs() >= 4).mean()
P(f"   pairs with positive net {sum(v>0 for v in pm.values())}/6; events where >=5/6 pairs share the same sign {agree:.0%}")
res[8] = sum(v > 0 for v in pm.values()) >= 5
P(f"   -> {'PASS' if res[8] else 'FAIL'}")

# 9 Monte Carlo
P("\n[9] MONTE CARLO (block bootstrap by calendar year of the 12 events; 10000 sims of 10-year paths)")
def mc(series_bp, lev_label, notional):
    x = (series_bp.dropna() * notional / 1e4).groupby(series_bp.dropna().index.year)
    years = [v.values for _, v in x]
    ann, dd, lose = [], [], 0
    for _ in range(10000):
        path = np.concatenate([years[i] for i in RNG.integers(0, len(years), 10)])
        eq = np.cumsum(path)
        ann.append(eq[-1] / 10)
        dd.append((np.maximum.accumulate(np.r_[0, eq]) - np.r_[0, eq]).max())
    ay = [np.sum(years[i]) for i in range(len(years))]
    P(f"   {lev_label:38s} mean annual {np.mean(ann)*100:+.2f}%  (5-95%: {np.percentile(ann,5)*100:+.2f}..{np.percentile(ann,95)*100:+.2f})"
      f"  median maxDD(10y) {np.median(dd)*100:.2f}%  95% maxDD {np.percentile(dd,95)*100:.2f}%  hist P(losing yr) {np.mean(np.array(ay)<0):.0%}")
for p in PAIRS:
    mc(A[p], f"{p} 1x notional", 1.0)
mc(A.BASKET, "BASKET 6 pairs each 1/6 (1x total)", 1.0)
mc(A.BASKET, "BASKET 2x total", 2.0)
mc(A.BASKET * 6, "BASKET each pair 1x (6x gross)", 1.0)
res[9] = A.BASKET.groupby(yr).sum().gt(0).mean() >= 0.7
P(f"   -> {'PASS' if res[9] else 'FAIL'} (historical share of positive years >=70%)")

# 10 multiple testing
P("\n[10] MULTIPLE TESTING")
ntests = 12 * 7 + 6 + 3 + 6 + 2 * 2 + 4   # surface cells x pairs+basket, B windows, alt signals, reversal...
tp = tstat(A.BASKET)
from math import erf
pv = 1 - 0.5 * (1 + erf(abs(tp) / math.sqrt(2)))
best_t = max(ts)
pv_best = 1 - 0.5 * (1 + erf(abs(best_t) / math.sqrt(2)))
P(f"   ~{ntests} variants looked at. Primary basket t={tp:+.2f} (one-sided p={pv:.4f}); best surface cell t={best_t:+.2f} "
  f"Bonferroni p={min(1, pv_best*ntests):.3f}; Bonferroni-required |t| ~ {abs(np.percentile(RNG.standard_normal(10**6), 100*(1-0.05/ntests))):.2f}")
res[10] = pv * 2 < 0.05 / 1 and abs(tp) > 3.0
P(f"   -> {'PASS' if res[10] else 'FAIL'} (pre-specified primary needs |t|>3 given the search)")

P("\n" + "=" * 100)
P("SUMMARY  (primary: SPY-EFA signal, 10:00->15:55, 6-pair basket, net x1)")
for k in range(1, 11):
    P(f"   test {k:2d}: {'PASS' if res[k] else 'FAIL'}")
P(f"   passes {sum(res.values())}/10   basket net {A.BASKET.mean():+.2f}bp/event (t={tstat(A.BASKET):+.2f}), gross {Ag.BASKET.mean():+.2f}bp")
P("=" * 100)


# =============================================================================================
# POST-HOC: section B showed an UNCONDITIONAL month-end pattern (USD sold into the fix, bought back
# after it).  It was found after ~107 looks, so it gets the same battery with a post-hoc label.
#   U1: LONG USD basket 16:05 -> 17:00 London on the last business day (post-fix rebound)
#   U2: SHORT USD basket 15:00 -> 15:55 London (pre-fix selling)
# =============================================================================================
def unc(dates, t0, t1, d, cmult=1.0, pairs=PAIRS):
    df = pd.DataFrame({p: d * usd_ret(p, dates, t0, t1) - cost_bp(p, cmult) for p in pairs})
    df["BASKET"] = df[list(pairs)].mean(1)
    return df


def unc_battery(name, t0s, t1s, d, surf_e, surf_x, h1_bar):
    P("\n" + "=" * 100)
    P(f"POST-HOC BATTERY {name}: {'LONG' if d > 0 else 'SHORT'} USD basket {t0s}->{t1s} London on month-end days (net x1)")
    P("=" * 100)
    t0, t1 = hm(t0s), hm(t1s)
    U = unc(last_bd, t0, t1, d)
    Ug = unc(last_bd, t0, t1, d, 0)
    r = {}
    for c in PAIRS + ["BASKET"]:
        P(f"   {c:7s} gross {summ(Ug, c)} | net {summ(U, c)}")
    uy = U.index.year
    # 1
    P("[1] two sources (MT5 H1 bar %02d:00 open->close vs Dukascopy same hour, USD-direction, gross)" % h1_bar)
    ok1 = []
    for p in ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]:
        o, c = load_mt5_h1(p)
        dd = last_bd.intersection(o.index)
        mt = np.log(c.reindex(dd)[h1_bar] / o.reindex(dd)[h1_bar]) * 1e4
        mt = d * (mt if p in USD_BASE else -mt)
        du = d * usd_ret(p, dd, h1_bar * 60, (h1_bar + 1) * 60)
        k = mt.notna() & du.notna()
        cor = np.corrcoef(mt[k], du[k])[0, 1]
        ok1.append(cor > 0.9 and mt[k].mean() > 0 and du[k].mean() > 0)
        P(f"     {p}: n={k.sum()} corr={cor:.3f} duka {du[k].mean():+.2f}bp (t{tstat(du[k]):+.1f})  mt5 {mt[k].mean():+.2f}bp (t{tstat(mt[k]):+.1f})")
    r[1] = all(ok1)
    P(f"   -> {'PASS' if r[1] else 'FAIL'}")
    # 2
    P("[2] surface (basket net bp, t)")
    cells = []
    for e in surf_e:
        row = []
        for x in surf_x:
            if hm(x) <= hm(e):
                row.append(" " * 22)
                continue
            t = unc(last_bd, hm(e), hm(x), d).BASKET
            cells.append((t.mean(), tstat(t)))
            row.append(f"{x}:{t.mean():+5.2f}(t{tstat(t):+4.1f})")
        P(f"     {e}  " + "  ".join(row))
    cv = np.array(cells)
    r[2] = np.mean(cv[:, 0] > 0) >= 0.8 and np.mean(cv[:, 1] > 1.5) >= 0.5
    P(f"   positive {np.mean(cv[:,0]>0):.0%}, t>1.5 {np.mean(cv[:,1]>1.5):.0%} -> {'PASS' if r[2] else 'FAIL'}")
    # 3
    P("[3] eras / years")
    em = []
    for k, m in {"DEV 2012-16": uy <= 2016, "VAL 2017-21": (uy >= 2017) & (uy <= 2021), "RECENT 2022+": uy >= 2022}.items():
        P(f"     {k:13s} {summ(U[m])}")
        em.append(U.BASKET[m].mean())
    ys = U.BASKET.groupby(uy).sum()
    P("     by year (sum bp): " + " ".join(f"{y}:{v:+.0f}" for y, v in ys.items()))
    r[3] = all(v > 0 for v in em) and np.mean(ys > 0) >= 0.6
    P(f"   years positive {np.mean(ys>0):.0%} -> {'PASS' if r[3] else 'FAIL'}")
    # 4
    P("[4] regimes")
    qe = U.index.month.isin([3, 6, 9, 12])
    pre = pd.DataFrame({p: usd_ret(p, last_bd, hm("15:00"), hm("16:00")) for p in PAIRS}).mean(1)
    spyabs = SIG.spy.abs().reindex(U.index)
    cr = (spyabs > spyabs.quantile(0.8)).values
    rg = {"quarter-end": qe, "non-QE": ~qe, "year-end": U.index.month == 12,
          "USD fell 15-16h": (pre < 0).values, "USD rose 15-16h": (pre >= 0).values,
          "foreign outperf (sig<0)": (SIG.sig.reindex(U.index) < 0).values,
          "US outperf (sig>0)": (SIG.sig.reindex(U.index) > 0).values, "crisis": cr, "calm": ~cr}
    if t0 < hm("16:00"):     # conditioning on the 15-16h move would be look-ahead for a pre-fix trade
        rg.pop("USD fell 15-16h"); rg.pop("USD rose 15-16h")
    rv = {}
    for k, m in rg.items():
        P(f"     {k:24s} {summ(U[m])}")
        rv[k] = U.BASKET[m].mean()
    r[4] = rv["quarter-end"] > 0 and rv["non-QE"] > 0 and rv["calm"] > 0 and rv["crisis"] > 0
    P(f"   -> {'PASS' if r[4] else 'FAIL'}")
    # 5
    P("[5] placebo")
    pl = {}
    for nm, sel in [("2nd-last BD", bd.k_from_end == 1), ("3rd-last BD", bd.k_from_end == 2),
                    ("1st BD", bd.k_from_start == 0), ("10th BD", bd.k_from_start == 9)]:
        dl = pd.DatetimeIndex(bd.d[sel]); dl = dl[dl < pd.Timestamp("2026-10-01")]
        t = unc(dl, t0, t1, d)
        pl[nm] = t.BASKET.mean()
        P(f"     {nm:12s} {summ(t)}")
    allo = unc(others, t0, t1, d, 0).BASKET.dropna().values
    boots = np.array([RNG.choice(allo, len(U)).mean() for _ in range(5000)])
    pv = np.mean(boots >= Ug.BASKET.mean())
    P(f"     random non-month-end day sets (5000 x {len(U)}): gross mean {boots.mean():+.2f}bp, 99th pct {np.percentile(boots,99):+.2f}, "
      f"month-end gross {Ug.BASKET.mean():+.2f}bp, p={pv:.4f}")
    r[5] = pv < 0.01 and all(U.BASKET.mean() > v for v in pl.values())
    P(f"   -> {'PASS' if r[5] else 'FAIL'}")
    # 6
    P("[6] costs / delays")
    for cm in [1, 2, 3]:
        P(f"     cost x{cm}: {summ(unc(last_bd, t0, t1, d, cm))}")
    for dl in [5, 10, 15, 30]:
        P(f"     entry +{dl:2d}min: {summ(unc(last_bd, t0 + dl, t1, d))}")
    c2 = unc(last_bd, t0, t1, d, 2).BASKET
    r[6] = c2.mean() > 0 and tstat(c2) > 1.5
    P(f"   -> {'PASS' if r[6] else 'FAIL'} (x2 cost mean>0, t>1.5)")
    # 7
    P("[7] walk-forward: trade year Y only if expanding gross t (years<Y) > 1.0 for the fixed window")
    wf = []
    for y in range(2014, 2027):
        tr = Ug.BASKET[uy < y]
        on = tstat(tr) > 1.0
        v = U.BASKET[uy == y].sum() if on else 0.0
        wf.append((y, on, v))
    P("     " + " ".join(f"{y}:{'on' if o else 'off'}{v:+.0f}" for y, o, v in wf))
    tot = sum(v for _, _, v in wf)
    on_years = [v for _, o, v in wf if o]
    r[7] = tot > 0 and (np.mean(np.array(on_years) > 0) >= 0.6 if on_years else False)
    P(f"   WF total {tot:+.0f}bp, traded years positive {np.mean(np.array(on_years)>0) if on_years else 0:.0%} -> {'PASS' if r[7] else 'FAIL'}")
    # 8
    P("[8] cross-pair")
    pm = {p: U[p].mean() for p in PAIRS}
    P("     " + "  ".join(f"{p}:{v:+.2f}(t{tstat(U[p]):+.1f})" for p, v in pm.items()))
    r[8] = sum(v > 0 for v in pm.values()) >= 5
    P(f"   -> {'PASS' if r[8] else 'FAIL'}")
    # 9
    P("[9] Monte Carlo (year-block bootstrap, 10y paths)")
    mc(U.BASKET, "BASKET 1x total notional", 1.0)
    mc(U.BASKET, "BASKET 2x total notional", 2.0)
    mc(U.BASKET * 6, "BASKET each pair 1x (6x gross)", 1.0)
    mc(U.BASKET * 12, "BASKET each pair 2x (12x gross)", 1.0)
    r[9] = U.BASKET.groupby(uy).sum().gt(0).mean() >= 0.7
    P(f"   -> {'PASS' if r[9] else 'FAIL'} (share of positive years >= 70%)")
    # 10
    tp = tstat(U.BASKET)
    P(f"[10] multiple testing: net t={tp:+.2f}; this window was chosen after ~107 looks + 2 post-hoc batteries; "
      f"Bonferroni needs t>~3.3")
    r[10] = tp > 3.3
    P(f"   -> {'PASS' if r[10] else 'FAIL'}")
    P(f"SUMMARY {name}: " + " ".join(f"{k}:{'P' if v else 'F'}" for k, v in r.items()) +
      f"  passes {sum(r.values())}/10  net {U.BASKET.mean():+.2f}bp/event t={tp:+.2f}")
    return U


U1 = unc_battery("U1 post-fix USD rebound", "16:05", "17:00", +1,
                 ["16:00", "16:05", "16:10", "16:15", "16:30"], ["16:30", "17:00", "17:30", "18:00", "19:00"], 16)
U2 = unc_battery("U2 pre-fix USD selling", "15:00", "15:55", -1,
                 ["13:00", "14:00", "14:30", "15:00", "15:30"], ["15:45", "15:55", "16:00"], 15)
both = (U1.BASKET + U2.BASKET)
P(f"\nCOMBINED U1+U2 (two round trips per event): net {both.mean():+.2f}bp/event t={tstat(both):+.2f}; "
  f"corr(U1,U2) {np.corrcoef(U1.BASKET.fillna(0), U2.BASKET.fillna(0))[0,1]:+.2f}")
mc(both, "U1+U2 BASKET each pair 1x (6x gross)", 6.0)
