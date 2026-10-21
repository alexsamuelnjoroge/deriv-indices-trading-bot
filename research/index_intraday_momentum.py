"""Adversarial validation battery: MARKET INTRADAY MOMENTUM (Gao, Han, Li & Zhou 2018, JFE).

Hypothesis: the first half-hour return of the cash session (prior close -> open+30m, or open -> open+30m)
predicts the last half-hour return (close-30m -> close). Trade: at close-30m go long/short in the
direction of the signal, exit at the cash close. Flat overnight => no swap.

Variants (fixed upfront, no tuning):
  signal  S1 = prior close -> open+30m (paper), S2 = open -> open+30m, S3 = 12th half-hour (close-60 -> close-30),
          S13 = S1 and S3 agree in sign (trade only then)
  trade   T1 = close-30 -> close (primary), T2 = close-60 -> close, T3 = close-15 -> close
          (S3/S13 are not combined with T2, which overlaps the signal window)
  filter  all days / strong only (|signal| > trailing 252-day median of |signal|, strictly past data)
  PRE-REGISTERED CORE: S1 / T1 / all.

Data: Dukascopy 5-min BID candles for index CFDs USA500.IDX, USATECH.IDX, USA30.IDX, DEU.IDX (2018 -> 2026-10),
cached in data/duka/<name>_M5_<YEAR>.csv; MT5 IC Markets H1 (data/pro/mt5_*_H1.csv, server = New York + 7h) and
yfinance SPY/QQQ/DIA 5-min (last ~60 days) as second sources; Yahoo daily ^VIX for regimes.
Costs: one full spread per round trip from live MT5 symbol_info (read 2026-10-21): US500 0.65bp, USTEC 0.34bp,
US30 0.23bp, DE40 0.20bp; commission assumed 0 on IC Markets index CFDs; no swap (intraday).
NO ORDERS. Run:
    .venv/Scripts/python.exe research/index_intraday_momentum.py > research/index_intraday_momentum.out
"""
import math
import warnings
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
rng = np.random.default_rng(20261021)
ROOT = Path(__file__).resolve().parents[1]
DUKA = ROOT / "data" / "duka"
RES = ROOT / "research"
ND = NormalDist()

MK = {  # name: duka file, tz, session open/close (minutes from midnight local), cost bp, MT5 H1, yahoo daily, etf
    "US500": dict(duka="USA500", tz="America/New_York", o=570, c=960, cost=0.65, h1="mt5_US500_H1.csv", yd="^GSPC", etf="SPY"),
    "USTEC": dict(duka="USATECH", tz="America/New_York", o=570, c=960, cost=0.34, h1="mt5_USTEC_H1.csv", yd="^NDX", etf="QQQ"),
    "US30": dict(duka="USA30", tz="America/New_York", o=570, c=960, cost=0.23, h1="mt5_US30_H1.csv", yd="^DJI", etf="DIA"),
    "DE40": dict(duka="DEU40", tz="Europe/Berlin", o=540, c=1050, cost=0.20, h1="mt5_DE40_H1.csv", yd="^GDAXI", etf=None),
}
US = ["US500", "USTEC", "US30"]
PERIODS = [("DEV 2018-2020", 2018, 2020), ("VAL 2021-2023", 2021, 2023), ("RECENT 2024+", 2024, 2030)]
TRIALS = []  # (label, t-stat) of every variant evaluated (multiple-testing accounting)


def hdr(s):
    print("\n" + "=" * 110 + "\n" + s + "\n" + "=" * 110)


def tstat(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return 0.0
    return x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))


# ------------------------------------------------------------------------------------------------ data
def load_duka(name):
    fs = sorted(DUKA.glob(f"{name}_M5_*.csv"))
    df = pd.concat([pd.read_csv(f, index_col=0, parse_dates=True) for f in fs])
    df = df[~df.index.duplicated()].sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def yahoo_calendar(tk):
    y = pd.read_csv(RES / "cache_yahoo_indices.csv")
    d = pd.to_datetime(y.loc[y.tk == tk, "Date"])
    return set(d.dt.date)


def us_half_days(years):
    out = set()
    for y in years:
        nov = pd.date_range(f"{y}-11-01", f"{y}-11-30")
        thu = [d for d in nov if d.weekday() == 3][3]
        out.add((thu + pd.Timedelta(days=1)).date())
        for md in (f"{y}-12-24", f"{y}-07-03"):
            d = pd.Timestamp(md)
            if d.weekday() < 5:
                out.add(d.date())
    return out


def price_grid(df, tz):
    """rows = local date, cols = minute-of-day m, value = price AT time m (= close of the 5m bar starting m-5)."""
    loc = df.index.tz_convert(tz)
    end_min = loc.hour * 60 + loc.minute + 5
    g = pd.DataFrame({"d": loc.date, "m": end_min, "c": df["close"].values, "v": df["volume"].values})
    px = g.pivot_table(index="d", columns="m", values="c", aggfunc="last")
    vol = g.pivot_table(index="d", columns="m", values="v", aggfunc="sum")
    px = px.reindex(columns=range(5, 1445, 5)).ffill(axis=1, limit=2)  # tolerate <=2 empty 5m bars
    return px, vol


def build(mk, src="duka", grid=None):
    p = MK[mk]
    o, c = p["o"], p["c"]
    px = grid
    cal = yahoo_calendar(p["yd"])
    # the cached Yahoo file ends at its download date; keep later CFD days that look like full sessions
    last_cal = max(cal)
    days = [d for d in px.index if (d in cal or d > last_cal) and pd.Timestamp(d).weekday() < 5]
    if mk in US:
        hd = us_half_days(range(2017, 2027))
        days = [d for d in days if d not in hd]
    px = px.loc[days]
    P = lambda m: px[m] if m in px.columns else pd.Series(np.nan, index=px.index)
    prev_close = P(c).shift(1)
    r = lambda a, b: np.log(b / a) * 1e4  # bp
    out = pd.DataFrame(index=pd.to_datetime(px.index))
    v = lambda s: s.values
    out["S1"] = v(r(prev_close, P(o + 30)))
    out["S2"] = v(r(P(o), P(o + 30)))
    out["S3"] = v(r(P(c - 60), P(c - 30)))
    out["T1"] = v(r(P(c - 30), P(c)))
    out["T2"] = v(r(P(c - 60), P(c)))
    out["T3"] = v(r(P(c - 15), P(c)))
    out["T1d"] = v(r(P(c - 25), P(c)))   # 5-minute execution delay (stricter than 1-2 min)
    out["T2d"] = v(r(P(c - 55), P(c)))
    out["T3d"] = v(r(P(c - 10), P(c)))
    out["pre"] = v(r(P(o), P(c - 30)))   # open -> entry (up/down day known at entry)
    # other half-hours as placebo predictors: k-th half-hour, k = 2 .. (n-2)
    nhh = (c - o) // 30
    for k in range(2, nhh - 1):
        out[f"H{k}"] = v(r(P(o + 30 * (k - 1)), P(o + 30 * k)))
    # data gap guard: a day with a stale prev close more than 4 calendar days back is still fine (weekends);
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


VARIANTS = []
for s in ["S1", "S2", "S3", "S13"]:
    for t in ["T1", "T2", "T3"]:
        if s in ("S3", "S13") and t == "T2":
            continue
        for f in ["all", "strong"]:
            VARIANTS.append((s, t, f))
CORE = ("S1", "T1", "all")


def signal(d, s):
    if s == "S13":
        a, b = np.sign(d["S1"]), np.sign(d["S3"])
        return pd.Series(np.where(a == b, d["S1"], 0.0), index=d.index)
    return d[s]


def strat(d, var, cost, cmult=1.0, delay=False):
    """net bp per day (NaN = no trade)."""
    s, t, f = var
    sig = signal(d, s)
    tr = d[t + ("d" if delay else "")]
    sg = np.sign(sig)
    if f == "strong":
        base = d["S1"] if s == "S13" else d[s]
        med = base.abs().rolling(252, min_periods=120).median().shift(1)
        sg = sg.where(base.abs() > med, 0.0)
    pnl = sg * tr - cost * cmult * (sg != 0)
    pnl = pnl.where((sg != 0) & tr.notna() & sig.notna())
    return pnl


def summ(x):
    x = x.dropna()
    if len(x) < 3:
        return "n=%d" % len(x)
    return f"n={len(x):4d} mean {x.mean():+6.2f}bp t={tstat(x):+5.2f} hit {np.mean(x > 0):.0%}"


# ------------------------------------------------------------------------------------------------ main
def main():
    data, grids = {}, {}
    hdr("DATA / ALIGNMENT CHECKS (Dukascopy 5m index CFDs)")
    for mk, p in MK.items():
        df = load_duka(p["duka"])
        px, vol = price_grid(df, p["tz"])
        grids[mk] = (px, vol)
        d = build(mk, grid=px)
        data[mk] = d
        o, c = p["o"], p["c"]
        cover = d[["S1", "T1"]].notna().all(axis=1).mean()
        # DST/session alignment: volume of 5m bar ending o+5 vs bar ending o-5, per year (should jump at the open)
        vr = (vol[o + 5] / vol[o - 5]).groupby(pd.to_datetime(vol.index).year).median()
        cr = (vol[c + 5] / vol[c - 5]).groupby(pd.to_datetime(vol.index).year).median() if (c + 5) in vol else None
        print(f"{mk}: {df.index[0]} -> {df.index[-1]}  bars={len(df)}  trading days={len(d)}  S1&T1 complete {cover:.1%}")
        print("   open-jump volume ratio (bar after/before cash open) by year: " +
              " ".join(f"{y % 100}:{v:.1f}" for y, v in vr.items()))
        if cr is not None:
            print("   close volume ratio (bar after/before cash close) by year: " +
                  " ".join(f"{y % 100}:{v:.2f}" for y, v in cr.items()))
        # per-year DST months check: ratio in DST-transition gap weeks (mid-Mar, late-Oct/early-Nov)
        idx = pd.to_datetime(vol.index)
        gap = ((idx.month == 3) & (idx.day >= 8) & (idx.day <= 31)) | ((idx.month == 11) & (idx.day <= 7)) | \
              ((idx.month == 10) & (idx.day >= 25))
        print(f"   open-jump ratio in DST-gap weeks: {np.nanmedian((vol[o + 5] / vol[o - 5])[gap]):.1f} "
              f"vs other weeks {np.nanmedian((vol[o + 5] / vol[o - 5])[~gap]):.1f}")

    # -------------------------------------------------------------------------------- 0. core description
    hdr("0. CORE (S1 prior-close->open+30 signal, T1 last half-hour, all days) -- predictive regression & net P&L")
    for mk in MK:
        d = data[mk].dropna(subset=["S1", "T1"])
        b = np.polyfit(d["S1"], d["T1"], 1)[0]
        corr = d[["S1", "T1"]].corr().iloc[0, 1]
        g = (np.sign(d["S1"]) * d["T1"])
        n = strat(data[mk], CORE, MK[mk]["cost"])
        print(f"{mk:6s} corr(S1,T1)={corr:+.3f} beta={b:+.4f} | gross {summ(g)} | net {summ(n)} | sd(T1)={d['T1'].std():.1f}bp")

    # -------------------------------------------------------------------------------- 1. two sources
    hdr("1. TWO SOURCES")
    print("(a) Dukascopy vs MT5 IC Markets H1, comparable variant: signal prev close->10:00, trade 15:00->16:00 (T2)")
    two_ok = []
    for mk, p in MK.items():
        h = pd.read_csv(ROOT / "data" / "pro" / p["h1"])
        # server time = New York + 7h for IC Markets; DE40 handled in Berlin tz via NY conversion
        t_ny = pd.to_datetime(h.epoch, unit="s") - pd.Timedelta(hours=7)
        t_ny = t_ny.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
        t_loc = t_ny.dt.tz_convert(p["tz"])
        hh = pd.DataFrame({"d": t_loc.dt.date, "m": t_loc.dt.hour * 60 + t_loc.dt.minute + 60, "c": h.close})
        hp = hh.dropna().pivot_table(index="d", columns="m", values="c", aggfunc="last")
        o, c = p["o"], p["c"]
        o_h = o + 30 if (o + 30) % 60 == 0 else o + 60   # first hour close available on H1 grid (10:00 NY / 10:00 Berlin)
        c_h = c if c % 60 == 0 else c - 30               # DE40: 17:30 close not on H1 grid -> use 17:00
        px = grids[mk][0]
        common = sorted(set(hp.index) & set(data[mk].index.date))
        def mk_df(G):
            G = G.reindex(common)
            s = np.log(G[o_h] / G[c_h].shift(1)) * 1e4
            t = np.log(G[c_h] / G[c_h - 60]) * 1e4
            return s, t
        s_d, t_d = mk_df(px)
        s_m, t_m = mk_df(hp)
        ok = s_d.notna() & s_m.notna() & t_d.notna() & t_m.notna()
        cs, ct = np.corrcoef(s_d[ok], s_m[ok])[0, 1], np.corrcoef(t_d[ok], t_m[ok])[0, 1]
        agree = np.mean(np.sign(s_d[ok]) == np.sign(s_m[ok]))
        pd_ = np.sign(s_d[ok]) * t_d[ok] - p["cost"]
        pm_ = np.sign(s_m[ok]) * t_m[ok] - p["cost"]
        two_ok.append(np.sign(pd_.mean()) == np.sign(pm_.mean()))
        lab = f"sig {o_h // 60}:{o_h % 60:02d} trade {(c_h - 60) // 60}:00-{c_h // 60}:{c_h % 60:02d}"
        print(f"  {mk:6s} n={ok.sum()} ({lab}) corr signal {cs:.3f} corr trade {ct:.3f} sign agree {agree:.1%} | "
              f"net Duka {pd_.mean():+.2f}bp (t={tstat(pd_):+.2f})  MT5 {pm_.mean():+.2f}bp (t={tstat(pm_):+.2f})")
    print("(b) Dukascopy vs yfinance ETF 5m (last ~60 days): 15:30->16:00 return and S1 signal")
    etf_rows = []
    for mk in US:
        etf = MK[mk]["etf"]
        f = RES / f"cache_imom_{etf}_5m.csv"
        try:
            if not f.exists():
                import yfinance as yf
                e = yf.download(etf, period="60d", interval="5m", progress=False, auto_adjust=False)
                e.columns = [c[0] if isinstance(c, tuple) else c for c in e.columns]
                e.to_csv(f)
            e = pd.read_csv(f, index_col=0, parse_dates=True)
            e.index = pd.to_datetime(e.index, utc=True)
            e = e.rename(columns=str.lower)
            e["volume"] = e.get("volume", 0)
            G, _ = price_grid(e[["close", "volume"]], "America/New_York")
            G = G.reindex(columns=range(5, 1445, 5))
            common = sorted(set(G.index) & set(data[mk].index.date))
            G = G.reindex(common)
            t_e = np.log(G[960] / G[930]) * 1e4
            s_e = np.log(G[600] / G[960].shift(1)) * 1e4
            dd = data[mk].reindex(pd.to_datetime(common))
            ok = t_e.notna().values & dd["T1"].notna().values & s_e.notna().values & dd["S1"].notna().values
            ct = np.corrcoef(t_e.values[ok], dd["T1"].values[ok])[0, 1]
            cs = np.corrcoef(s_e.values[ok], dd["S1"].values[ok])[0, 1]
            ag = np.mean(np.sign(s_e.values[ok]) == np.sign(dd["S1"].values[ok]))
            ge = np.sign(s_e.values[ok]) * t_e.values[ok]
            gd = np.sign(dd["S1"].values[ok]) * dd["T1"].values[ok]
            print(f"  {mk:6s} vs {etf}: n={ok.sum()} corr T1 {ct:.3f} corr S1 {cs:.3f} sign agree {ag:.0%} | "
                  f"gross core: ETF {ge.mean():+.2f}bp Duka {gd.mean():+.2f}bp")
            etf_rows.append(ct)
        except Exception as ex:
            print(f"  {mk}: ETF check failed: {ex!r}")
    print(f"  VERDICT 1: {'PASS' if all(two_ok) and (not etf_rows or min(etf_rows) > 0.8) else 'FAIL/CHECK'} "
          f"(sources agree on data; sign of comparable-variant net mean agrees in {sum(two_ok)}/{len(two_ok)} markets)")

    # -------------------------------------------------------------------------------- 2. parameter surface
    hdr("2. PARAMETER SURFACE (net bp per trade, x1 cost, full sample 2018+)")
    surf = {}
    for mk in MK:
        print(f"-- {mk}")
        for var in VARIANTS:
            x = strat(data[mk], var, MK[mk]["cost"]).dropna()
            surf[(mk, var)] = x
            TRIALS.append((f"{mk} {var}", tstat(x)))
            flag = " <-- CORE" if var == CORE else ""
            print(f"   {'/'.join(var):16s} {summ(x)}{flag}")
    for mk in MK:
        means = [surf[(mk, v)].mean() for v in VARIANTS]
        pos = np.mean(np.array(means) > 0)
        sig = np.mean([tstat(surf[(mk, v)]) > 2 for v in VARIANTS])
        print(f"  {mk}: {pos:.0%} of {len(VARIANTS)} variants net-positive, {sig:.0%} with t>2")
    us_pos = np.mean([surf[(mk, v)].mean() > 0 for mk in US for v in VARIANTS])
    print(f"  VERDICT 2: {'PASS' if us_pos >= 0.75 else 'FAIL'} (US: {us_pos:.0%} of variants net-positive; need >=75% broad)")

    # -------------------------------------------------------------------------------- 3. OOS by time
    hdr("3. OUT-OF-SAMPLE BY TIME (core S1/T1/all, net)")
    oos_ok = []
    for mk in MK:
        x = surf[(mk, CORE)]
        yr = x.groupby(x.index.year)
        parts = []
        for lab, a, b in PERIODS:
            xs = x[(x.index.year >= a) & (x.index.year <= b)]
            parts.append(f"{lab}: {xs.mean():+.2f}bp t={tstat(xs):+.2f}")
            if mk in US:
                oos_ok.append(xs.mean() > 0)
        print(f"  {mk:6s} " + " | ".join(parts))
        print("         year-by-year bp: " + " ".join(f"{y % 100}:{v.mean():+.1f}" for y, v in yr))
        print("         year-by-year t : " + " ".join(f"{y % 100}:{tstat(v):+.1f}" for y, v in yr))
    print(f"  VERDICT 3: {'PASS' if all(oos_ok) else 'FAIL'} (US markets positive in {sum(oos_ok)}/{len(oos_ok)} period cells)")

    # -------------------------------------------------------------------------------- 4. regimes
    hdr("4. REGIMES (core, net bp)")
    vf = RES / "cache_imom_vix.csv"
    if not vf.exists():
        import yfinance as yf
        v = yf.download("^VIX", start="2016-01-01", progress=False, auto_adjust=False)
        v.columns = [c[0] if isinstance(c, tuple) else c for c in v.columns]
        v[["Close"]].to_csv(vf)
    vix = pd.read_csv(vf, index_col=0, parse_dates=True)["Close"]
    vprev = vix.shift(1)
    vhi = vprev > vprev.rolling(252, min_periods=120).median()
    reg_ok = []
    for mk in MK:
        x = surf[(mk, CORE)]
        d = data[mk].reindex(x.index)
        hi = vhi.reindex(x.index).fillna(False).astype(bool)
        sg = np.sign(d["S1"])
        cells = {"VIX hi": x[hi], "VIX lo": x[~hi], "up-day(o->15:30>0)": x[d["pre"] > 0],
                 "down-day": x[d["pre"] <= 0], "long signal": x[sg > 0], "short signal": x[sg < 0],
                 "|S1| top tercile": x[d["S1"].abs() > d["S1"].abs().quantile(2 / 3)]}
        print(f"  {mk:6s} " + " | ".join(f"{k} {v.mean():+.2f} (t={tstat(v):+.1f}, n={len(v)})" for k, v in cells.items()))
        if mk in US:
            reg_ok += [cells["VIX hi"].mean() > 0, cells["VIX lo"].mean() > 0]
    print(f"  VERDICT 4: {'PASS' if all(reg_ok) else 'FAIL'} (US VIX-hi and VIX-lo both positive in {sum(reg_ok)}/{len(reg_ok)})")

    # -------------------------------------------------------------------------------- 5. placebo
    hdr("5. PLACEBO")
    plc_ok = []
    for mk in MK:
        d = data[mk]
        real = (np.sign(d["S1"]) * d["T1"]).dropna()
        tr = d["T1"].dropna()
        sims = np.array([(rng.choice([-1, 1], len(tr)) * tr.values).mean() for _ in range(2000)])
        p_rand = np.mean(sims >= real.mean())
        others = {}
        for k in [c for c in d.columns if c.startswith("H")]:
            others[k] = (np.sign(d[k]) * d["T1"]).dropna().mean()
        lag = (np.sign(d["S1"].shift(1)) * d["T1"]).dropna().mean()
        rank = np.mean(np.array(list(others.values())) >= real.mean())
        print(f"  {mk:6s} real gross {real.mean():+.2f}bp | random-direction p={p_rand:.3f} (sims mean {sims.mean():+.2f}, "
              f"95th {np.percentile(sims, 95):+.2f}) | yesterday's S1 {lag:+.2f}bp")
        print("         other half-hours as signal (gross bp): " + " ".join(f"{k}:{v:+.2f}" for k, v in others.items()) +
              f"  -> share >= real: {rank:.0%}")
        if mk in US:
            plc_ok.append(p_rand < 0.05 and rank <= 0.2)
    print(f"  VERDICT 5: {'PASS' if all(plc_ok) else 'FAIL'} (US markets beating placebos: {sum(plc_ok)}/{len(plc_ok)})")

    # -------------------------------------------------------------------------------- 6. costs & delay
    hdr("6. COSTS x1/x2/x3 and 5-minute execution delay (5m data; stricter than the 1-2 min asked)")
    cost_ok = []
    for mk in MK:
        row = []
        for cm in (1, 2, 3):
            x = strat(data[mk], CORE, MK[mk]["cost"], cm).dropna()
            row.append(f"x{cm} {x.mean():+.2f}bp t={tstat(x):+.2f}")
        xd = strat(data[mk], CORE, MK[mk]["cost"], 1, delay=True).dropna()
        xd2 = strat(data[mk], CORE, MK[mk]["cost"], 2, delay=True).dropna()
        row.append(f"delay5m x1 {xd.mean():+.2f}bp t={tstat(xd):+.2f}")
        row.append(f"delay5m x2 {xd2.mean():+.2f}bp")
        print(f"  {mk:6s} " + " | ".join(row))
        if mk in US:
            cost_ok.append(strat(data[mk], CORE, MK[mk]["cost"], 3).mean() > 0 and xd2.mean() > 0)
    print(f"  VERDICT 6: {'PASS' if all(cost_ok) else 'FAIL'} ({sum(cost_ok)}/{len(cost_ok)} US markets positive at x3 and delay+x2)")

    # -------------------------------------------------------------------------------- 7. walk-forward
    hdr("7. WALK-FORWARD VARIANT CHOICE (expanding window, pick best net t-stat in past years, trade next year)")
    wf_ok, wf_series = [], {}
    for mk in MK:
        years = sorted(set(data[mk].index.year))
        out, picks = [], []
        for y in years[1:]:
            best = max(VARIANTS, key=lambda v: tstat(surf[(mk, v)][surf[(mk, v)].index.year < y]))
            picks.append(f"{y % 100}:{'/'.join(best)}")
            out.append(surf[(mk, best)][surf[(mk, best)].index.year == y])
        w = pd.concat(out)
        wf_series[mk] = w
        print(f"  {mk:6s} WF net {summ(w)} | picks " + " ".join(picks))
        if mk in US:
            wf_ok.append(w.mean() > 0 and tstat(w) > 1.0)
    print(f"  VERDICT 7: {'PASS' if all(wf_ok) else 'FAIL'} ({sum(wf_ok)}/{len(wf_ok)} US markets WF mean>0 & t>1)")

    # -------------------------------------------------------------------------------- 8. cross-market
    hdr("8. CROSS-MARKET (core)")
    cm_ok = []
    for mk in MK:
        x = surf[(mk, CORE)]
        print(f"  {mk:6s} {summ(x)}")
        cm_ok.append(x.mean() > 0)
    print(f"  VERDICT 8: {'PASS' if sum(cm_ok) >= 3 and all(cm_ok[:3]) else 'FAIL'} (net-positive in {sum(cm_ok)}/4 incl. DE40 mechanism check)")

    # -------------------------------------------------------------------------------- 9. bootstrap
    hdr("9. MONTE CARLO / BLOCK BOOTSTRAP (daily returns, 21-day blocks, 252-day years, 1x notional)")
    def boot(r, lev, nsim=4000, years=1):
        r = r.values * lev / 1e4
        n, L = len(r), 21
        ann, dd = [], []
        for _ in range(nsim):
            k = int(np.ceil(252 * years / L))
            st = rng.integers(0, n - L, k)
            s = np.concatenate([r[i:i + L] for i in st])[:252 * years]
            eq = np.cumsum(s)
            ann.append(s.sum() / years)
            dd.append(np.max(np.maximum.accumulate(np.concatenate([[0], eq])) - np.concatenate([[0], eq])))
        return np.array(ann), np.array(dd)
    port_parts = {}
    for mk in MK:
        d = data[mk]
        x = surf[(mk, CORE)].reindex(d.index).fillna(0.0)
        port_parts[mk] = x
    port = pd.concat([port_parts[m] for m in US], axis=1).mean(axis=1)  # 1x notional split across 3 US indices
    series = {"US500 core": port_parts["US500"], "USTEC core": port_parts["USTEC"],
              "US3 equal-split core": port, "DE40 core": port_parts["DE40"]}
    for lab, s in series.items():
        for lev in (1, 2):
            a, _ = boot(s, lev)
            _, dd10 = boot(s, lev, nsim=1000, years=10)
            yrs = s.groupby(s.index.year).sum() * lev / 100
            print(f"  {lab:22s} {lev}x: hist ann {s.mean() * 252 * lev / 100:+.2f}% | boot ann mean {a.mean():+.2%} "
                  f"5th {np.percentile(a, 5):+.2%} | P(losing yr) {np.mean(a < 0):.0%} | 10y maxDD median {np.median(dd10):.1%} "
                  f"95th {np.percentile(dd10, 95):.1%} | yrs+ {np.mean(yrs > 0):.0%}")
    print("  US3 equal-split year-by-year (1x, %): " + " ".join(f"{y % 100}:{v:+.1f}" for y, v in (port.groupby(port.index.year).sum() / 100).items()))
    a, _ = boot(port, 1)
    print(f"  VERDICT 9: {'PASS' if np.mean(a < 0) < 0.25 and a.mean() > 0 else 'FAIL'} (US3 1x P(losing year) {np.mean(a < 0):.0%})")

    # -------------------------------------------------------------------------------- 10. multiple testing
    hdr("10. MULTIPLE-TESTING CORRECTION")
    n = len(TRIALS)
    tb = -ND.inv_cdf(0.05 / 2 / n)
    surv = [(l, t) for l, t in TRIALS if t > tb]
    print(f"  trials counted: {n} (4 markets x {len(VARIANTS)} variants); Bonferroni 5% two-sided t threshold {tb:.2f}; "
          f"Harvey-Liu-Zhu t>3.0")
    best = sorted(TRIALS, key=lambda z: -z[1])[:8]
    print("  top t-stats: " + "; ".join(f"{l} t={t:+.2f}" for l, t in best))
    core_t = {mk: tstat(surf[(mk, CORE)]) for mk in MK}
    print("  core t-stats (pre-registered, 1 test per market): " + " ".join(f"{k}:{v:+.2f}" for k, v in core_t.items()))
    pt = tstat(port.replace(0, np.nan).dropna()) if False else tstat(port[port != 0])
    print(f"  US3 portfolio t (days traded) {pt:+.2f}; survivors of Bonferroni: {len(surv)}")
    print(f"  VERDICT 10: {'PASS' if surv or pt > 3 else 'FAIL'}")


if __name__ == "__main__":
    main()
