"""Adversarial validation battery for stock-index calendar & overnight effects.

Sub-strategies (grids fixed upfront, no tuning):
  a) Turn-of-month (TOM): long from close of trading day -k (k=1..4; -1 = last day of month)
     to close of day +x (x=1..5).                        core: k=2 (holds day -1), x=3
  b) Pre-holiday: long over the trading day before a scheduled market holiday.
     variants: D-1 only / D-2..D-1 / D-1 + D+1            core: D-1
  c) Overnight (close->open) vs intraday (open->close) decomposition; long-overnight-only on CFDs.
  d) Day-of-week (descriptive + DEV->later persistence) and FOMC pre-announcement drift
     (scheduled FOMC dates 1994-2026 transcribed from federalreserve.gov calendars).
  e) Short-term reversal: buy at close when day return < -k*sd20 (k=1.5,2,2.5), hold h=1,2,3,5 days,
     with/without 200-day-trend filter.                  core: k=2, h=3, trend filter

Data: Yahoo daily cash indices (1985+, research/cache_yahoo_indices.csv) and MT5 IC Markets CFD H1
(2016+, data/pro/mt5_*_H1.csv; server time = New York + 7h). Costs from live MT5 symbol_info
(read-only) with snapshot fallback. NO ORDERS. Run:
    .venv/Scripts/python.exe research/index_calendar_battery.py > research/index_calendar_battery.out
"""
import math
import warnings

import numpy as np
import pandas as pd
from statistics import NormalDist

warnings.filterwarnings("ignore")
rng = np.random.default_rng(20261021)
N_PLACEBO = 1000
TRIALS = []  # (label, daily SR) for multiple-testing accounting

SYMS = {
    "US500": dict(yh="^GSPC", tz="America/New_York", oh=10, ch=16, div=1.4, us=True, h1="data/pro/mt5_US500_H1.csv"),
    "USTEC": dict(yh="^NDX", tz="America/New_York", oh=10, ch=16, div=0.7, us=True, h1="data/pro/mt5_USTEC_H1.csv"),
    "US30": dict(yh="^DJI", tz="America/New_York", oh=10, ch=16, div=1.8, us=True, h1="data/pro/mt5_US30_H1.csv"),
    "DE40": dict(yh="^GDAXI", tz="Europe/Berlin", oh=9, ch=17, div=0.0, us=False, h1="data/pro/mt5_DE40_H1.csv"),
    "UK100": dict(yh="^FTSE", tz="Europe/London", oh=8, ch=16, div=3.5, us=False, h1="research/cache_mt5_UK100_H1.csv"),
    "JP225": dict(yh="^N225", tz="Asia/Tokyo", oh=9, ch=15, div=1.8, us=False, h1="data/pro/mt5_JP225_H1.csv"),
}
# symbol_info snapshot taken 2026-10-21 (swap money per 1-contract lot per night; contract = 1 x index)
SNAP = {
    "US500": dict(swl=-2.064, sws=-0.255, px=7722.5),
    "USTEC": dict(swl=-7.925, sws=-0.98, px=29621.4),
    "US30": dict(swl=-13.944, sws=-1.723, px=52472.3),
    "DE40": dict(swl=-5.738, sws=-1.964, px=25357.5),
    "UK100": dict(swl=-2.864, sws=-0.409, px=10725.3),
    "JP225": dict(swl=-12.359, sws=-7.415, px=64460.0),
}
PERIODS = [("DEV 1990-2009", "1990", "2009"), ("VAL 2010-2019", "2010", "2019"), ("RECENT 2020+", "2020", "2030")]
UNSCHED_US = pd.to_datetime(["1985-09-27", "1994-04-27", "2001-09-11", "2001-09-12", "2001-09-13", "2001-09-14",
                             "2004-06-11", "2007-01-02", "2012-10-29", "2012-10-30", "2018-12-05", "2025-01-09"])

FOMC_RAW = {  # scheduled meetings, last day = announcement day (federalreserve.gov fomccalendars / fomchistoricalYYYY)
    1994: "Feb 4,Mar 22,May 17,Jul 6,Aug 16,Sep 27,Nov 15,Dec 20",
    1995: "Feb 1,Mar 28,May 23,Jul 6,Aug 22,Sep 26,Nov 15,Dec 19",
    1996: "Jan 31,Mar 26,May 21,Jul 3,Aug 20,Sep 24,Nov 13,Dec 17",
    1997: "Feb 5,Mar 25,May 20,Jul 2,Aug 19,Sep 30,Nov 12,Dec 16",
    1998: "Feb 4,Mar 31,May 19,Jul 1,Aug 18,Sep 29,Nov 17,Dec 22",
    1999: "Feb 3,Mar 30,May 18,Jun 30,Aug 24,Oct 5,Nov 16,Dec 21",
    2000: "Feb 2,Mar 21,May 16,Jun 28,Aug 22,Oct 3,Nov 15,Dec 19",
    2001: "Jan 31,Mar 20,May 15,Jun 27,Aug 21,Oct 2,Nov 6,Dec 11",
    2002: "Jan 30,Mar 19,May 7,Jun 26,Aug 13,Sep 24,Nov 6,Dec 10",
    2003: "Jan 29,Mar 18,May 6,Jun 25,Aug 12,Sep 16,Oct 28,Dec 9",
    2004: "Jan 28,Mar 16,May 4,Jun 30,Aug 10,Sep 21,Nov 10,Dec 14",
    2005: "Feb 2,Mar 22,May 3,Jun 30,Aug 9,Sep 20,Nov 1,Dec 13",
    2006: "Jan 31,Mar 28,May 10,Jun 29,Aug 8,Sep 20,Oct 25,Dec 12",
    2007: "Jan 31,Mar 21,May 9,Jun 28,Aug 7,Sep 18,Oct 31,Dec 11",
    2008: "Jan 30,Mar 18,Apr 30,Jun 25,Aug 5,Sep 16,Oct 29,Dec 16",
    2009: "Jan 28,Mar 18,Apr 29,Jun 24,Aug 12,Sep 23,Nov 4,Dec 16",
    2010: "Jan 27,Mar 16,Apr 28,Jun 23,Aug 10,Sep 21,Nov 3,Dec 14",
    2011: "Jan 26,Mar 15,Apr 27,Jun 22,Aug 9,Sep 21,Nov 2,Dec 13",
    2012: "Jan 25,Mar 13,Apr 25,Jun 20,Aug 1,Sep 13,Oct 24,Dec 12",
    2013: "Jan 30,Mar 20,May 1,Jun 19,Jul 31,Sep 18,Oct 30,Dec 18",
    2014: "Jan 29,Mar 19,Apr 30,Jun 18,Jul 30,Sep 17,Oct 29,Dec 17",
    2015: "Jan 28,Mar 18,Apr 29,Jun 17,Jul 29,Sep 17,Oct 28,Dec 16",
    2016: "Jan 27,Mar 16,Apr 27,Jun 15,Jul 27,Sep 21,Nov 2,Dec 14",
    2017: "Feb 1,Mar 15,May 3,Jun 14,Jul 26,Sep 20,Nov 1,Dec 13",
    2018: "Jan 31,Mar 21,May 2,Jun 13,Aug 1,Sep 26,Nov 8,Dec 19",
    2019: "Jan 30,Mar 20,May 1,Jun 19,Jul 31,Sep 18,Oct 30,Dec 11",
    2020: "Jan 29,Apr 29,Jun 10,Jul 29,Sep 16,Nov 5,Dec 16",
    2021: "Jan 27,Mar 17,Apr 28,Jun 16,Jul 28,Sep 22,Nov 3,Dec 15",
    2022: "Jan 26,Mar 16,May 4,Jun 15,Jul 27,Sep 21,Nov 2,Dec 14",
    2023: "Feb 1,Mar 22,May 3,Jun 14,Jul 26,Sep 20,Nov 1,Dec 13",
    2024: "Jan 31,Mar 20,May 1,Jun 12,Jul 31,Sep 18,Nov 7,Dec 18",
    2025: "Jan 29,Mar 19,May 7,Jun 18,Jul 30,Sep 17,Oct 29,Dec 10",
    2026: "Jan 28,Mar 18,Apr 29,Jun 17,Jul 29,Sep 16",
}
FOMC = pd.DatetimeIndex(sorted(pd.to_datetime(f"{d} {y}", format="%b %d %Y")
                               for y, s in FOMC_RAW.items() for d in s.split(",")))


class stats:  # minimal scipy.stats replacement (scipy not installed in .venv)
    class norm:
        cdf = staticmethod(lambda x: NormalDist().cdf(x))
        ppf = staticmethod(lambda p: NormalDist().inv_cdf(p))

    @staticmethod
    def skew(x):
        x = np.asarray(x, float)
        return float(((x - x.mean()) ** 3).mean() / x.std() ** 3)

    @staticmethod
    def kurtosis(x, fisher=False):
        x = np.asarray(x, float)
        k = float(((x - x.mean()) ** 4).mean() / x.std() ** 4)
        return k - 3 if fisher else k

    @staticmethod
    def spearmanr(a, b):
        ra = pd.Series(np.asarray(a)).rank()
        rb = pd.Series(np.asarray(b)).rank()
        return (ra.corr(rb), None)


def hdr(s):
    print("\n" + "=" * 110 + f"\n{s}\n" + "=" * 110)


def tstat(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return x.mean() / x.std(ddof=1) * math.sqrt(len(x)) if len(x) > 2 and x.std() > 0 else np.nan


def nw_ols(y, x, lags=5):
    """OLS y = a + b x with Newey-West t for alpha."""
    m = ~(np.isnan(y) | np.isnan(x))
    y, x = y[m], x[m]
    X = np.column_stack([np.ones_like(x), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    n = len(y)
    XtXi = np.linalg.inv(X.T @ X)
    S = (X * e[:, None]).T @ (X * e[:, None])
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None])
        S += w * (G + G.T)
    V = XtXi @ S @ XtXi
    return b[0], b[1], b[0] / math.sqrt(V[0, 0])


# ============================================================ costs
hdr("COSTS: MT5 IC Markets symbol_info (read-only) + H1 spread history")
LIVE = {k: dict(v) for k, v in SNAP.items()}
try:
    import MetaTrader5 as mt5
    if mt5.initialize():
        for s in SYMS:
            i = mt5.symbol_info(s)
            if i is not None:
                LIVE[s].update(swl=i.swap_long, sws=i.swap_short, mode=i.swap_mode, r3=i.swap_rollover3days,
                               cs=i.trade_contract_size, live_spread_pts=i.spread * i.point)
        mt5.shutdown()
        print("live symbol_info read OK")
except Exception as ex:  # noqa
    print("MT5 not available, using snapshot", ex)

H1 = {}
for s, c in SYMS.items():
    d = pd.read_csv(c["h1"])
    srv = pd.to_datetime(d.epoch, unit="s")
    ny = (srv - pd.Timedelta(hours=7)).dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    d["loc"] = ny.dt.tz_convert(c["tz"])
    d = d.dropna(subset=["loc"])
    d = d[d["loc"].dt.year >= 2016]
    d["date"] = d["loc"].dt.tz_localize(None).dt.normalize()
    d["h"] = d["loc"].dt.hour
    d["nyh"] = d["loc"].dt.tz_convert("America/New_York").dt.hour
    H1[s] = d

COST = {}
for s, c in SYMS.items():
    d = H1[s]
    rec = d[d.date >= "2023-01-01"]
    px = rec.close.median()
    sp_o = rec[rec.h == c["oh"]].spread.median() * 0.01
    sp_c = rec[rec.h == c["ch"]].spread.median() * 0.01
    L = LIVE[s]
    price = L.get("px", px)
    long_bp = -L["swl"] / price * 1e4
    short_bp = -L["sws"] / price * 1e4
    m_bp = (long_bp + short_bp) / 2  # broker markup per night
    r_bp = (long_bp - short_bp) / 2  # implied benchmark rate per night
    rt_close = sp_c / px * 1e4  # round trip entered & exited at the close hour
    rt_oc = (sp_o + sp_c) / 2 / px * 1e4
    COST[s] = dict(rt_close=rt_close, rt_oc=rt_oc, long_bp=long_bp, m_bp=m_bp, r_bp=r_bp, div_bp=c["div"] / 365 * 100)
    print(f"{s:6s} px~{price:9.1f} spread@open {sp_o:5.2f}pt @close {sp_c:5.2f}pt -> RT {rt_close:4.2f}bp (close) "
          f"{rt_oc:4.2f}bp (open/close) | swap long {L['swl']:+.3f}/night = {long_bp:4.2f}bp/night "
          f"({long_bp*365/100:4.1f}%/yr incl. triple) markup {m_bp:4.2f}bp implied rate {r_bp*360/100:4.2f}%/yr "
          f"| div credit {c['div']}%/yr={c['div']/365*100:4.2f}bp/night | mode={L.get('mode','?')} triple-day={L.get('r3','?')}")
print("Assumptions: 1 contract = 1 x index; P&L on bid bars, one full spread charged per round trip at the hour used;"
      "\n swap charged per calendar night held (triple-Friday => calendar days); CFD longs receive dividend adjustments"
      "\n (Yahoo/MT5 prices are price indices except DAX = total-return index -> no div credit). For US indices the"
      "\n historical swap is modelled as markup + ^IRX T-bill rate; non-US use today's live swap (flag: their rates differ).")

# ============================================================ data
yall = pd.read_csv("research/cache_yahoo_indices.csv", parse_dates=["Date"])
irx = yall[yall.tk == "^IRX"].set_index("Date").Close.sort_index()


def swap_series(sym, dates):
    c = COST[sym]
    if SYMS[sym]["us"]:
        r = irx.reindex(dates, method="ffill").bfill().clip(lower=0).values
        return c["m_bp"] + r / 100 / 360 * 1e4
    return np.full(len(dates), c["long_bp"])


def mk_daily(dates, O, C, sym):
    df = pd.DataFrame({"O": O, "C": C}, index=pd.DatetimeIndex(dates)).sort_index()
    df = df[~df.index.duplicated()]
    df["pC"] = df.C.shift(1)
    df["rcc"] = (df.C / df.pC - 1) * 1e4
    df["ron"] = (df.O / df.pC - 1) * 1e4
    df["rid"] = (df.C / df.O - 1) * 1e4
    df["gap"] = pd.Series(df.index, index=df.index).diff().dt.days
    df["bull"] = df.C > df.C.rolling(200).mean()  # known at close t
    df["sd20"] = df.rcc.rolling(20).std()  # known at close t
    df["bull_prev"] = df.bull.shift(1)
    df["sd_prev"] = df.sd20.shift(1)
    df = df.iloc[1:].copy()
    df["swap"] = swap_series(sym, df.index)
    df["dv"] = COST[sym]["div_bp"]
    df["year"] = df.index.year
    ym = df.index.year * 100 + df.index.month
    df["ym"] = ym
    g = df.groupby("ym")
    df["dn"] = g.cumcount() + 1
    df["de"] = -(g.cumcount(ascending=False) + 1)
    return df


YH, MT = {}, {}
hdr("DATA + ARTIFACT CHECK (Yahoo open == previous close / open == close)")
for s, c in SYMS.items():
    y = yall[yall.tk == c["yh"]].dropna(subset=["Close"]).set_index("Date").sort_index()
    y = y[y.Close > 0]
    y.loc[y.Open <= 0, "Open"] = np.nan
    pc = y.Close.shift(1)
    stale = (np.abs(y.Open - pc) < 1e-6 * y.Close) | (np.abs(y.Open - y.Close) < 1e-6 * y.Close) | y.Open.isna()
    fr = stale.groupby(y.index.year).mean()
    bad_years = fr[fr > 0.05].index.tolist()
    df = mk_daily(y.index, y.Open.values, y.Close.values, s)
    df["oc_ok"] = ~df.year.isin(bad_years)
    df = df[df.index >= "1989-01-01"]
    df.loc[df.index < "1990-01-01", ["rcc"]] = df.loc[df.index < "1990-01-01", "rcc"]  # warm-up kept for SMA
    YH[s] = df[df.index >= "1990-01-01"].copy()
    good = [yr for yr in range(1990, 2027) if yr not in bad_years]
    print(f"{s:6s} {c['yh']:7s} {YH[s].index[0].date()}..{YH[s].index[-1].date()} n={len(YH[s])}"
          f" | stale-open years (>5% of days): {[b for b in bad_years if b >= 1985]} -> excluded from ON/ID")
    # MT5 daily from H1: open = bar opening at local oh, close = bar opening at local ch (approx cash close)
    d = H1[s]
    O = d[d.h == c["oh"]].drop_duplicates("date").set_index("date").open
    C = d[d.h == c["ch"]].drop_duplicates("date").set_index("date").open
    O9 = d[d.h == c["oh"] - 1].drop_duplicates("date").set_index("date").open
    j = pd.concat([O.rename("O"), C.rename("C"), O9.rename("O9")], axis=1)
    j = j[j.index.isin(y.index)].dropna(subset=["O", "C"])  # only cash-market trading days
    m = mk_daily(j.index, j.O.values, j.C.values, s)
    m["O9"] = j.O9.reindex(m.index)
    m["oc_ok"] = True
    MT[s] = m
    print(f"       MT5 CFD {m.index[0].date()}..{m.index[-1].date()} n={len(m)} (open {c['oh']}:00 / close {c['ch']}:00 {c['tz']})")


# ============================================================ engine
def evaluate(df, pos, sym, cmult=1.0, rt=None):
    pos = np.asarray(pos, bool)
    prev = np.r_[False, pos[:-1]]
    entries = pos & ~prev
    rt = COST[sym]["rt_close"] if rt is None else rt
    g = np.where(pos, df.rcc.values, 0.0)
    carry = np.where(pos, df.gap.values * (df.swap.values - df.dv.values), 0.0)
    net = g - entries * rt * cmult - carry
    return pd.Series(net, index=df.index), pd.Series(g, index=df.index)


def bh_net(df):
    return pd.Series(df.rcc.values - df.gap.values * (df.swap.values - df.dv.values), index=df.index)


def sr(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return x.mean() / x.std() * math.sqrt(252) if x.std() > 0 else np.nan


def line(net, pos=None, idx=None):
    net = net.dropna()
    s = f"ann {net.mean()*252/100:+6.2f}%  SR {sr(net):+5.2f}"
    if pos is not None:
        p = pd.Series(np.asarray(pos, bool), index=idx).reindex(net.index).fillna(False).values
        s += f"  per-inv-day {net[p].mean():+6.2f}bp t={tstat(net[p]):+5.2f} exp={p.mean()*100:4.1f}%"
    yr = net.groupby(net.index.year).sum()
    s += f"  yrs+ {np.mean(yr > 0)*100:3.0f}% ({len(yr)})"
    return s


def alpha_line(net, df):
    a, b, t = nw_ols(net.values, df.rcc.values)
    a2, b2, t2 = nw_ols(net.values, bh_net(df).values)
    return (f"alpha vs gross index {a*252/100:+5.2f}%/yr (NW t={t:+4.2f}) b={b:4.2f} | "
            f"vs net CFD B&H {a2*252/100:+5.2f}%/yr (t={t2:+4.2f})")


def placebo_cal(df, pos, nrep=N_PLACEBO, filt=None):
    """random-block placebo: same number of blocks and lengths placed at random start days (per month for
    monthly patterns, uniformly otherwise); returns p-value of gross mean per invested day."""
    pos = np.asarray(pos, bool)
    r = df.rcc.values
    act = r[pos].mean()
    starts = np.where(pos & ~np.r_[False, pos[:-1]])[0]
    ends = np.where(pos & ~np.r_[pos[1:], False])[0]
    lens = ends - starts + 1
    n = len(r)
    cand = np.arange(n) if filt is None else np.where(filt)[0]
    out = np.empty(nrep)
    for i in range(nrep):
        st = rng.choice(cand, size=len(starts), replace=True)
        pp = np.zeros(n, bool)
        for a, L in zip(st, lens):
            pp[a:a + L] = True
        out[i] = r[pp].mean()
    return act, out.mean(), np.mean(out >= act)


def block_boot(net, years=10000, block=21):
    x = net.dropna().values
    n = len(x)
    days = 252
    nb = days // block + 1
    ann = np.empty(years)
    for i in range(years):
        st = rng.integers(0, n - block, nb)
        ann[i] = np.concatenate([x[s:s + block] for s in st])[:days].sum()
    # 10-year max drawdown distribution
    dds = []
    for i in range(2000):
        st = rng.integers(0, n - block, 10 * nb)
        path = np.cumsum(np.concatenate([x[s:s + block] for s in st])[:10 * days]) / 100  # % of notional
        dds.append((np.maximum.accumulate(path) - path).max())
    return ann / 100, np.array(dds)


def periods(net):
    out = []
    for nm, a, b in PERIODS:
        x = net[a:b]
        if len(x) > 50:
            out.append(f"{nm}: ann {x.mean()*252/100:+5.2f}% SR {sr(x):+4.2f}")
    return " | ".join(out)


def deflated_sr(best_sr_d, n_obs, skew, kurt, sr_var, n_trials):
    g = 0.5772156649
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * math.e))
    sr0 = math.sqrt(sr_var) * ((1 - g) * z1 + g * z2)
    den = math.sqrt(1 - skew * best_sr_d + (kurt - 1) / 4 * best_sr_d ** 2)
    return sr0, stats.norm.cdf((best_sr_d - sr0) * math.sqrt(n_obs - 1) / den)


# ============================================================ position builders
def pos_tom(df, k, x):
    return ((df.de.values >= -(k - 1)) & (k > 1)) | (df.dn.values <= x)


def holidays_for(sym, df):
    days = df.index
    allwd = pd.bdate_range(days[0], days[-1])
    missing = allwd.difference(days)
    if SYMS[sym]["us"] or sym == "US500":
        missing = missing.difference(UNSCHED_US)
    return missing


HOL = {}


def pre_hol_flags(sym, df):
    # use the Yahoo cash calendar for both sources
    miss = HOL[sym]
    d = df.index
    nxt = np.r_[d[1:], d[-1:]]
    flag = np.zeros(len(d), bool)
    ms = set(miss)
    for i in range(len(d) - 1):
        if (nxt[i] - d[i]).days > 1:
            for k in range(1, (nxt[i] - d[i]).days):
                dd = d[i] + pd.Timedelta(days=k)
                if dd.weekday() < 5 and dd in ms:
                    flag[i] = True
                    break
    return flag


def pos_prehol(df, sym, variant):
    f = pre_hol_flags(sym, df)
    if variant == "D-1":
        return f
    if variant == "D-2..D-1":
        return f | np.r_[f[1:], False]
    if variant == "D-1,D+1":
        return f | np.r_[False, f[:-1]]


def pos_rev(df, k, h, trend):
    sig = (df.rcc.values < -k * df.sd_prev.values)
    if trend:
        sig &= df.bull.values.astype(bool)
    pos = np.zeros(len(df), bool)
    for i in np.where(sig)[0]:
        pos[i + 1:i + 1 + h] = True
    return pos


def pos_fomc(df, variant):
    f = df.index.isin(FOMC)
    if variant == "D0":
        return f
    if variant == "D-1..D0":
        return f | np.r_[f[1:], False]


# ============================================================ generic family battery
def family(name, grid, core, builder, need_hol=False, placebo_filter=None):
    hdr(f"SUB-STRATEGY {name}  grid={len(grid)} configs  core={core}")
    res = {}
    # --- 8 cross-market + 3 OOS, core config, Yahoo net (MT5 cost model)
    print("-- core config per index (Yahoo cash, net of MT5 spread + swap - div credit):")
    core_sr = {}
    for s in SYMS:
        df = YH[s]
        p = builder(df, s, core)
        net, g = evaluate(df, p, s)
        core_sr[s] = sr(net)
        print(f"  {s:6s} {line(net, p, df.index)} | {alpha_line(net, df)}")
        print(f"         {periods(net)}  | gross per-inv-day {g[p].mean():+5.2f}bp")
    # --- grid surface (record trials)
    print("-- parameter surface: net SR (Yahoo full) per index; row = config")
    tab = {}
    for prm in grid:
        row = {}
        for s in SYMS:
            df = YH[s]
            p = builder(df, s, prm)
            net, _ = evaluate(df, p, s)
            row[s] = sr(net)
            TRIALS.append((f"{name}{prm}/{s}/YH", net.mean() / net.std() if net.std() > 0 else 0))
        tab[str(prm)] = row
    t = pd.DataFrame(tab).T
    t["mean"] = t.mean(axis=1)
    print(t.round(2).to_string())
    print(f"  share of (config,index) cells with net SR>0: {np.mean(t.drop(columns='mean').values > 0)*100:.0f}%  "
          f"(cells>0.3: {np.mean(t.drop(columns='mean').values > 0.3)*100:.0f}%)")
    res["surface_pos"] = np.mean(t.drop(columns="mean").values > 0)
    # --- 1 cross-source + 6 costs on MT5
    print("-- cross-source 2016+ (MT5 CFD vs Yahoo cash), core config:")
    agree = 0
    for s in SYMS:
        m, y = MT[s], YH[s]
        pm = builder(m, s, core)
        py = builder(y, s, core)
        nm_, gm = evaluate(m, pm, s)
        ny_, gy = evaluate(y, py, s)
        yy = ny_[ny_.index >= m.index[0]]
        ev = pd.concat([gm[pm].rename("mt5"), gy[py].rename("yh")], axis=1).dropna()
        cc = ev.corr().iloc[0, 1] if len(ev) > 5 else np.nan
        agree += (np.sign(nm_.mean()) == np.sign(yy.mean()))
        c3 = " ".join(f"x{c}:{evaluate(m, pm, s, cmult=c)[0].mean()*252/100:+5.2f}%" for c in (1, 2, 3))
        print(f"  {s:6s} MT5 {line(nm_, pm, m.index)}\n         YH same span ann {yy.mean()*252/100:+5.2f}% SR {sr(yy):+4.2f}"
              f" | per-day corr {cc:.2f} (n={len(ev)}) | MT5 cost x1/x2/x3 {c3} | MT5 B&H net {bh_net(m).mean()*252/100:+5.2f}%")
    res["xsource_agree"] = agree
    # --- 4 regimes + 5 placebo/B&H on US500 & pooled
    print("-- regimes (Yahoo, core, gross per invested day vs B&H mean day in same regime):")
    for s in SYMS:
        df = YH[s]
        p = builder(df, s, core)
        r = df.rcc.values
        bull = df.bull_prev.values.astype(bool)
        hv = (df.sd_prev > df.sd_prev.median()).values
        out = []
        for nm, msk in [("bull", bull), ("bear", ~bull), ("hiVol", hv), ("loVol", ~hv)]:
            out.append(f"{nm} {r[p & msk].mean():+6.2f} vs {r[msk].mean():+5.2f} (n={int((p & msk).sum())})")
        print(f"  {s:6s} " + " | ".join(out))
    print("-- placebo (same blocks at random days, 1000 reps; gross bp per invested day) and B&H:")
    pv = {}
    for s in SYMS:
        df = YH[s]
        p = builder(df, s, core)
        filt = placebo_filter(df) if placebo_filter else None
        act, plm, pval = placebo_cal(df, p, filt=filt)
        pv[s] = pval
        bh = bh_net(df)
        net, _ = evaluate(df, p, s)
        print(f"  {s:6s} actual {act:+6.2f} placebo mean {plm:+6.2f} p={pval:.3f} | strat SR {sr(net):+4.2f} vs B&H net SR "
              f"{sr(bh):+4.2f} (B&H {bh.mean()*252/100:+5.2f}%/yr, {df.rcc.mean():+5.2f}bp/day)")
    res["placebo"] = pv
    # --- 7 walk-forward (expanding, choose best net SR in past >= 10y, per index)
    print("-- walk-forward (expanding window, choose config by past net SR, min 10y history):")
    wf_all = []
    for s in SYMS:
        df = YH[s]
        nets = {str(prm): evaluate(df, builder(df, s, prm), s)[0] for prm in grid}
        N = pd.DataFrame(nets)
        yrs = sorted(set(df.year))
        oos = []
        picks = []
        for Y in yrs[10:]:
            past = N[N.index.year < Y]
            best = (past.mean() / past.std()).idxmax()
            picks.append(best)
            oos.append(N.loc[N.index.year == Y, best])
        oos = pd.concat(oos)
        wf_all.append(oos.rename(s))
        print(f"  {s:6s} WF OOS {line(oos)} | last pick {picks[-1]} | distinct picks {len(set(picks))}")
    res["wf"] = wf_all
    res["core_sr"] = core_sr
    return res


# ============================================================ (a) TOM
TOM_GRID = [(k, x) for k in (1, 2, 3, 4) for x in (1, 2, 3, 4, 5)]
resA = family("a) TURN-OF-MONTH (k=entry close of day -k, x=exit close of day +x)", TOM_GRID, (2, 3),
              lambda df, s, p: pos_tom(df, *p))

# ============================================================ (b) pre-holiday
for s in SYMS:
    HOL[s] = holidays_for(s, YH[s])
hdr("PRE-HOLIDAY calendar sanity: inferred scheduled holidays per year (weekday non-trading days)")
for s in SYMS:
    cnt = pd.Series(HOL[s].year).value_counts().sort_index()
    print(f"  {s:6s} median/yr {cnt.median():.0f} min {cnt.min()} max {cnt.max()} (years with >15: {cnt[cnt > 15].index.tolist()})")
resB = family("b) PRE-HOLIDAY", ["D-1", "D-2..D-1", "D-1,D+1"], "D-1", lambda df, s, p: pos_prehol(df, s, p))

# ============================================================ (e) reversal
REV_GRID = [(k, h, tr) for tr in (True, False) for k in (1.5, 2.0, 2.5) for h in (1, 2, 3, 5)]
resE = family("e) SHORT-TERM REVERSAL (k x sd20 drop, hold h days, trend filter)", REV_GRID, (2.0, 3, True),
              lambda df, s, p: pos_rev(df, *p), placebo_filter=lambda df: df.bull.values.astype(bool))

# ============================================================ (c) overnight vs intraday
hdr("SUB-STRATEGY c) OVERNIGHT (close->open) vs INTRADAY (open->close)")
print("Yahoo cash (years with stale opens excluded), gross bp/day, and long-overnight-only on CFD net:")
for s in SYMS:
    df = YH[s][YH[s].oc_ok].dropna(subset=["ron", "rid"])
    for nm, a, b in PERIODS:
        x = df[a:b]
        if len(x) < 100:
            continue
        print(f"  {s:6s} {nm}: ON {x.ron.mean():+5.2f}bp t={tstat(x.ron):+5.2f} | ID {x.rid.mean():+5.2f}bp "
              f"t={tstat(x.rid):+5.2f} | CC {x.rcc.mean():+5.2f} (n={len(x)})")
print("\nMT5 CFD 2016+ decomposition (open = local open-hour bar; costs: 1 RT spread/day + swap per night held):")
for s in SYMS:
    m = MT[s]
    c = COST[s]
    rt = c["rt_oc"]
    carry = m.gap * (m.swap - m.dv)
    on_net = m.ron - rt - carry
    id_net = m.rid - rt
    for nm, ser in [("ON gross", m.ron), ("ON net", on_net), ("ID gross", m.rid), ("ID net", id_net),
                    ("ON net x2", m.ron - 2 * rt - carry), ("ID net x2", m.rid - 2 * rt)]:
        TRIALS.append((f"c){nm}/{s}/MT5", ser.mean() / ser.std()))
    extra = ""
    if "O9" in m and m.O9.notna().mean() > 0.5:
        on9 = (m.O9 / m.pC - 1) * 1e4
        extra = f" | ON to {SYMS[s]['oh']-1}:00 gross {on9.mean():+5.2f}"
    print(f"  {s:6s} ON gross {m.ron.mean():+5.2f}bp t={tstat(m.ron):+4.2f} | ID gross {m.rid.mean():+5.2f} t={tstat(m.rid):+4.2f}"
          f" | carry/night-held avg {carry.mean():4.2f}bp RT {rt:4.2f}bp -> ON net {on_net.mean()*252/100:+6.2f}%/yr SR {sr(on_net):+4.2f}"
          f" | ID net {id_net.mean()*252/100:+6.2f}%/yr SR {sr(id_net):+4.2f} | B&H net {bh_net(m).mean()*252/100:+5.2f}%{extra}")
    print(f"         ON net by period: {periods(on_net)}")

# ============================================================ (d) day of week + FOMC
hdr("SUB-STRATEGY d1) DAY-OF-WEEK (close-close, gross bp)")
dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
for s in SYMS:
    df = YH[s]
    out = []
    for nm, a, b in PERIODS:
        x = df[a:b]
        mm = x.groupby(x.index.weekday).rcc.mean()
        out.append(nm.split()[0] + " " + " ".join(f"{dow_names[i]}{mm.get(i, np.nan):+5.1f}" for i in range(5)))
    dev = df["1990":"2009"].groupby(df["1990":"2009"].index.weekday).rcc.mean()
    later = df["2010":].groupby(df["2010":].index.weekday).rcc.mean()
    print(f"  {s:6s} " + " || ".join(out) + f" || rank-corr DEV vs 2010+: {stats.spearmanr(dev, later)[0]:+.2f}")
TRIALS.extend((f"d)DOW{i}", 0.0) for i in range(5 * 6))  # counted as tried (5 weekdays x 6 indices)

resD = family("d2) FOMC ANNOUNCEMENT DAY (close-close, US indices meaningful; others = placebo-ish)",
              ["D0", "D-1..D0"], "D0", lambda df, s, p: pos_fomc(df, p))
hdr("FOMC pre-announcement drift on MT5 CFD: prior 16:00 ET close -> 14:00 ET (announcement time), 2016+")
for s in ["US500", "USTEC", "US30", "DE40", "UK100", "JP225"]:
    d = H1[s]
    nyd = d["loc"].dt.tz_convert("America/New_York")
    d2 = d.assign(nyts=nyd.dt.tz_localize(None))
    p14 = d2[nyd.dt.hour == 14].assign(dd=lambda z: z.nyts.dt.normalize()).drop_duplicates("dd").set_index("dd").open
    p16 = d2[nyd.dt.hour == 16].assign(dd=lambda z: z.nyts.dt.normalize()).drop_duplicates("dd").set_index("dd").open
    ev = []
    alld = []
    days = p14.index.intersection(YH["US500"].index)
    for D in days:
        prev = YH["US500"].index[YH["US500"].index < D]
        if len(prev) == 0:
            continue
        pdn = prev[-1]
        if pdn in p16.index:
            r = (p14[D] / p16[pdn] - 1) * 1e4
            alld.append((D, r, D in FOMC))
    a = pd.DataFrame(alld, columns=["d", "r", "f"]).set_index("d")
    f = a[a.f].r
    nf = a[~a.f].r
    cst = COST[s]["rt_close"] + (COST[s]["long_bp"] - COST[s]["div_bp"])
    TRIALS.append((f"d)FOMC-pre/{s}/MT5", 0.0))
    print(f"  {s:6s} FOMC days n={len(f)} mean {f.mean():+6.2f}bp median {f.median():+6.2f} t={tstat(f):+4.2f} hit {np.mean(f>0)*100:3.0f}% | "
          f"non-FOMC same window {nf.mean():+5.2f}bp | net of ~{cst:4.2f}bp cost: {f.mean()-cst:+6.2f}bp/event "
          f"=> {(f.mean()-cst)*8/100:+5.2f}%/yr | by year: "
          + " ".join(f"{y%100:02d}:{g.mean():+.0f}" for y, g in f.groupby(f.index.year)))

# ============================================================ multiple testing + best candidate
hdr("MULTIPLE TESTING")
ntr = len(TRIALS)
srs = np.array([t[1] for t in TRIALS if t[1] != 0.0])
print(f"configs x indices x sources evaluated (incl. descriptive DOW/FOMC cells): {ntr}")
cands = []
for nm, res, builder, core in [("TOM", resA, lambda df, s, p: pos_tom(df, *p), (2, 3)),
                               ("PREHOL", resB, lambda df, s, p: pos_prehol(df, s, p), "D-1"),
                               ("REV", resE, lambda df, s, p: pos_rev(df, *p), (2.0, 3, True)),
                               ("FOMC", resD, lambda df, s, p: pos_fomc(df, p), "D0")]:
    for s in SYMS:
        df = YH[s]
        net, _ = evaluate(df, builder(df, s, core), s)
        cands.append((nm, s, net))
for nm, s, net in sorted(cands, key=lambda z: -sr(z[2]))[:6]:
    x = net.dropna()
    srd = x.mean() / x.std()
    t = srd * math.sqrt(len(x))
    p1 = 2 * (1 - stats.norm.cdf(abs(t)))
    sr0, dsr = deflated_sr(srd, len(x), stats.skew(x), stats.kurtosis(x, fisher=False), srs.var(), ntr)
    print(f"  {nm:6s} {s:6s} net SR {sr(x):+4.2f} t={t:+4.2f} p={p1:.1e} Bonferroni p={min(1, p1*ntr):.3f} | "
          f"DSR prob={dsr:.3f} (SR0 ann {sr0*math.sqrt(252):.2f})")


def portfolio_report(label, nets):
    P = pd.concat(nets, axis=1).fillna(0)
    port = P.mean(axis=1)
    hdr(f"BEST-CANDIDATE PORTFOLIO: {label} (equal 1x notional split across {P.shape[1]} indices)")
    print("  " + line(port))
    print("  " + periods(port))
    yr = port.groupby(port.index.year).sum() / 100
    print("  year-by-year %: " + " ".join(f"{y%100:02d}:{v:+.1f}" for y, v in yr.items()))
    ann, dds = block_boot(port)
    print(f"  bootstrap (21d blocks) annual return: mean {ann.mean():+.2f}% median {np.median(ann):+.2f}% "
          f"5th pct {np.percentile(ann,5):+.2f}% | P(losing year) {np.mean(ann<0)*100:.0f}% | "
          f"10y maxDD median {np.median(dds):.1f}% 95th {np.percentile(dds,95):.1f}%")
    return port


us = ["US500", "USTEC", "US30"]
tom_nets = [evaluate(YH[s], pos_tom(YH[s], 2, 3), s)[0].rename(s) for s in SYMS]
portfolio_report("TOM core (k=2,x=3), all 6 indices, Yahoo net of MT5 costs", tom_nets)
portfolio_report("TOM core, MT5 CFD 2016+ actual", [evaluate(MT[s], pos_tom(MT[s], 2, 3), s)[0].rename(s) for s in SYMS])
portfolio_report("TOM walk-forward picks (all 6)", resA["wf"])
rev_nets = [evaluate(YH[s], pos_rev(YH[s], 2.0, 3, True), s)[0].rename(s) for s in SYMS]
portfolio_report("Reversal core (k=2,h=3,trend), all 6, Yahoo net", rev_nets)
portfolio_report("Reversal core, MT5 CFD 2016+", [evaluate(MT[s], pos_rev(MT[s], 2.0, 3, True), s)[0].rename(s) for s in SYMS])
ph = [evaluate(YH[s], pos_prehol(YH[s], s, "D-1"), s)[0].rename(s) for s in us]
portfolio_report("Pre-holiday D-1, US indices, Yahoo net", ph)
fo = [evaluate(YH[s], pos_fomc(YH[s], "D0"), s)[0].rename(s) for s in SYMS]
fport = portfolio_report("FOMC D0 (prior close -> FOMC-day close), all 6, Yahoo net (1994+ meaningful)", fo)
fo94 = [x["1994":] for x in fo]
portfolio_report("FOMC D0, all 6, Yahoo net, 1994+ only", fo94)
portfolio_report("FOMC D0, MT5 CFD 2016+", [evaluate(MT[s], pos_fomc(MT[s], "D0"), s)[0].rename(s) for s in SYMS])
hdr("FOMC D0 event-level robustness (Yahoo gross bp per event, 1994+)")
for s in SYMS:
    df = YH[s]["1994":]
    ev = df.rcc[df.index.isin(FOMC)]
    tm = np.sort(ev.values)[int(len(ev) * .05):int(len(ev) * .95)].mean()
    ex = ev[~ev.index.year.isin([2008, 2020, 2022])]
    pre = ev[ev.index.year <= 2011]
    post = ev[ev.index.year > 2011]
    bs = np.array([rng.choice(ev.values, len(ev)).mean() for _ in range(5000)])
    print(f"  {s:6s} n={len(ev)} mean {ev.mean():+6.2f} median {ev.median():+6.2f} 5%-trim {tm:+6.2f} hit {np.mean(ev>0)*100:3.0f}%"
          f" | ex-2008/20/22 {ex.mean():+6.2f} | 1994-2011 {pre.mean():+6.2f} (t={tstat(pre):+4.2f}) 2012+ {post.mean():+6.2f}"
          f" (t={tstat(post):+4.2f}) | boot P(mean<=0) {np.mean(bs<=0):.4f}")
bh = [bh_net(YH[s]).rename(s) for s in SYMS]
portfolio_report("REFERENCE buy & hold CFD (net swap, div credit), all 6", bh)
print(f"\nTotal trials counted: {len(TRIALS)}")
