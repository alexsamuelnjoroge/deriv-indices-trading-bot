"""Adversarial test battery for the XAUUSD "daily break drift" candidate edge.

Trade: buy shortly before the daily gold halt (00:00 server time = 17:00 New York),
sell some minutes after the 01:00 server reopen. Server time = America/New_York + 7h.

Read-only research. Pulls MT5 ticks read-only (no orders). Run:
    python research/xau_break_drift_battery.py | tee research/xau_break_drift_battery.out
"""
import glob
import json
import math
import datetime as dt

import numpy as np
import pandas as pd
from scipy import stats

rng = np.random.default_rng(12345)
DERIV_COST = 1.2  # bp round trip (0.00012 of notional)
ENTRIES = [60, 30, 15, 5]
EXITS = [5, 30, 60, 120, 180, 240]
DEF_E, DEF_X = 5, 120


def tstat(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return np.nan
    return x.mean() / x.std(ddof=1) * math.sqrt(len(x))


def summ(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return f"n={len(x):5d} mean={x.mean():+6.2f}bp t={tstat(x):+5.2f} hit={np.mean(x > 0)*100:4.1f}%"


def hdr(s):
    print("\n" + "=" * 100 + f"\n{s}\n" + "=" * 100)


# ---------------------------------------------------------------- data
def load_duka(sym="frxXAUUSD"):
    fs = sorted(glob.glob(f"data/duka/{sym}_M5_*.csv"))
    d = pd.concat([pd.read_csv(f) for f in fs])
    ts = pd.to_datetime(d.timestamp, utc=True)
    d["srv"] = ts.dt.tz_convert("America/New_York").dt.tz_localize(None) + pd.Timedelta(hours=7)
    d = d.drop_duplicates("srv").set_index("srv").sort_index()
    return d[["open", "high", "low", "close"]].astype(float)


bars = load_duka()
# price observed at time T = close of the bar labelled T-5min
px_close = pd.Series(bars.close.values, index=bars.index + pd.Timedelta(minutes=5))
px_open = pd.Series(bars.open.values, index=bars.index)
grid = pd.date_range(bars.index[0].floor("D"), bars.index[-1].ceil("D"), freq="5min")
PX = px_close.reindex(grid, method="ffill", tolerance=pd.Timedelta("10min"))
LOW = bars.low.reindex(grid)
HIGH = bars.high.reindex(grid)

days = pd.DatetimeIndex(sorted(set(bars.index.normalize())))
trade_days = days[days.weekday.isin([1, 2, 3, 4])]  # Tue-Fri server mornings (break Mon-Thu nights)


def px_at(times):
    return PX.reindex(times).values


def break_trades(entry_min, exit_min, dset=trade_days, extra_entry_delay=0, extra_exit_delay=0):
    E = dset - pd.Timedelta(minutes=entry_min - extra_entry_delay)
    X = dset + pd.Timedelta(minutes=60 + exit_min + extra_exit_delay)
    pe, px = px_at(E), px_at(X)
    return pd.Series((px / pe - 1) * 1e4, index=dset)


def min_low_between(dset, entry_min, exit_min):
    """worst bid low (as return vs entry) between entry and exit, using M5 bars."""
    out = []
    E = dset - pd.Timedelta(minutes=entry_min)
    X = dset + pd.Timedelta(minutes=60 + exit_min)
    pe = px_at(E)
    for e, x, p in zip(E, X, pe):
        lo = bars.low.loc[e:x - pd.Timedelta(minutes=5)]
        out.append((lo.min() / p - 1) * 1e4 if len(lo) and not np.isnan(p) else np.nan)
    return pd.Series(out, index=dset)


base = break_trades(DEF_E, DEF_X).dropna()
print(f"Dukascopy XAUUSD M5 bid: {bars.index[0]} -> {bars.index[-1]} (server time), {len(bars)} bars")
print("Bars inside 00:00-00:59 server:", int((bars.index.hour == 0).sum()), "(0 confirms the NY+7 mapping)")
hdr("DEFAULT: entry 5 min before break (23:55 srv), exit 120 min after reopen (03:00 srv), Tue-Fri")
print("gross:", summ(base))
print("net Deriv 1.2bp:", summ(base - DERIV_COST))
res = {}

# ---------------------------------------------------------------- 1 cross-source
hdr("TEST 1: cross-source agreement (MT5 H1 vs Dukascopy vs Deriv)")
m = pd.read_csv("data/pro/mt5_XAUUSD_H1.csv")
m["t"] = pd.to_datetime(m.epoch, unit="s")
m = m.set_index("t").sort_index()
mc = m.close
mt5_days = trade_days.union(pd.DatetimeIndex(sorted(set(m.index.normalize()))))
mt5_days = mt5_days[mt5_days.weekday.isin([1, 2, 3, 4])]
mt5_days = mt5_days[mt5_days >= pd.Timestamp("2016-07-01")]
e = mc.reindex(mt5_days - pd.Timedelta(hours=1)).values  # prev-day 23:00 bar close
x = mc.reindex(mt5_days + pd.Timedelta(hours=2)).values  # 02:00 bar close
mt5r = pd.Series((x / e - 1) * 1e4, index=mt5_days).dropna()
print("MT5 H1 23:00c->02:00c full:", summ(mt5r))
for y, g in mt5r.groupby(mt5r.index.year):
    print(f"  MT5 {y}: {summ(g)}")
duk_same = break_trades(0, 120).dropna()  # 24:00 -> 03:00 equivalent on Duka
j = pd.concat([mt5r.rename("mt5"), duk_same.rename("duka")], axis=1).dropna()
print(f"Overlap n={len(j)}  corr(per-trade)={j.corr().iloc[0,1]:.3f}  mean mt5={j.mt5.mean():+.2f} duka={j.duka.mean():+.2f}")
yy = j.groupby(j.index.year).agg(["mean", "count"])
yc = j.groupby(j.index.year).apply(lambda g: g.mt5.corr(g.duka))
print(pd.concat([yy, yc.rename("corr")], axis=1).round(2).to_string())
print(f"corr of yearly means: {yy[('mt5','mean')].corr(yy[('duka','mean')]):.3f}")
# Deriv 5-min
dj = pd.DataFrame(json.load(open("data/pro/deriv_XAUUSD_300.json"))).astype(float)
dj["srv"] = pd.to_datetime(dj.epoch, unit="s", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None) + pd.Timedelta(hours=7)
dpx = pd.Series(dj.close.values, index=dj.srv + pd.Timedelta(minutes=5))
dpx = dpx[~dpx.index.duplicated()].reindex(grid, method="ffill", tolerance=pd.Timedelta("10min"))
ddays = trade_days[(trade_days > dj.srv.min() + pd.Timedelta(days=1))]
derr = pd.Series((dpx.reindex(ddays + pd.Timedelta(minutes=180)).values / dpx.reindex(ddays - pd.Timedelta(minutes=5)).values - 1) * 1e4, index=ddays).dropna()
dj2 = pd.concat([derr.rename("deriv"), base.rename("duka")], axis=1).dropna()
print(f"Deriv 5-min default trade: {summ(derr)};  overlap with Duka n={len(dj2)} corr={dj2.corr().iloc[0,1]:.3f}  means deriv={dj2.deriv.mean():+.2f} duka={dj2.duka.mean():+.2f}")
print("Bars Deriv has inside 00:00-00:59 srv:", int((dj.srv.dt.hour == 0).sum()))

# ---------------------------------------------------------------- 2 timing surface
hdr("TEST 2: timing surface (mean bp / t), Dukascopy, Tue-Fri")
surf_m = pd.DataFrame(index=ENTRIES, columns=EXITS, dtype=float)
surf_t = surf_m.copy()
TR = {}
for en in ENTRIES:
    for ex in EXITS:
        r = break_trades(en, ex)
        TR[(en, ex)] = r
        surf_m.loc[en, ex] = r.mean()
        surf_t.loc[en, ex] = tstat(r)
print("rows = minutes before break, cols = minutes after reopen")
print("mean bp:\n" + surf_m.round(2).to_string())
print("t-stat:\n" + surf_t.round(2).to_string())
print(f"cells positive: {(surf_m>0).sum().sum()}/24, t>2: {(surf_t>2).sum().sum()}/24, net>1.2bp: {(surf_m>DERIV_COST).sum().sum()}/24")
# decompose: pre-break part vs post part
pre = pd.Series((px_at(trade_days) / px_at(trade_days - pd.Timedelta(minutes=60)) - 1) * 1e4, index=trade_days)
print("pre-break 60min->halt alone:", summ(pre))

# ---------------------------------------------------------------- 3 splits
hdr("TEST 3: year / weekday / month / regime splits (default, gross)")
for y, g in base.groupby(base.index.year):
    print(f"  {y}: {summ(g)}")
wd = {1: "Tue(Mon night)", 2: "Wed", 3: "Thu(Wed night)", 4: "Fri(Thu night)"}
for w, g in base.groupby(base.index.weekday):
    print(f"  {wd[w]:15s}: {summ(g)}")
mon = base.groupby(base.index.month).agg(["mean", "count", tstat])
print("month:\n" + mon.round(2).T.to_string())
# regimes from daily server closes (strictly prior info: close of D-2 relative to entry day D)
dclose = px_close[px_close.index.hour == 0].copy()
dclose.index = dclose.index.normalize() - pd.Timedelta(days=1)  # close at 24:00 of server day
dclose = dclose.groupby(level=0).last()
sma = dclose.rolling(120).mean()
trend_up = (dclose > sma).shift(1)
lr = np.log(dclose).diff()
rv = lr.rolling(20).std()
rv_hi = (rv > rv.rolling(252).median()).shift(1)
ret120 = (dclose / dclose.shift(120) - 1).shift(1)
prevday = (trade_days - pd.Timedelta(days=1)).normalize()
reg = pd.DataFrame({"r": base}).assign(
    trend=trend_up.reindex(prevday).values[np.searchsorted(trade_days, base.index)],
    vol=rv_hi.reindex(prevday).values[np.searchsorted(trade_days, base.index)],
    r120=ret120.reindex(prevday).values[np.searchsorted(trade_days, base.index)])
for k in ["trend", "vol"]:
    for v, g in reg.dropna(subset=[k]).groupby(k):
        print(f"  {k}={'UP/HIGH' if v else 'DOWN/LOW'}: {summ(g.r)}")
reg["bear"] = reg.r120 < -0.05
print("  gold 120d return < -5% (bear legs):", summ(reg[reg.bear == True].r))
print("  gold 120d return > +5%            :", summ(reg[reg.r120 > 0.05].r))
for lab, a, b in [("2018 Apr-Sep down leg", "2018-04-15", "2018-09-30"),
                  ("2021 Jan-Mar down leg", "2021-01-06", "2021-03-31"),
                  ("2022 Mar-Oct down leg", "2022-03-09", "2022-11-03"),
                  ("2020 Aug-2021 Mar down", "2020-08-07", "2021-03-31")]:
    print(f"  {lab:24s}: {summ(base[a:b])}")

# ---------------------------------------------------------------- 4 placebo
hdr("TEST 4: placebo - buy at every 5-min start time, hold 185 clock-min (same as default), Mon-Fri, no weekend crossing")
HOLD = 185
P = PX
starts = pd.timedelta_range("0min", "1435min", freq="5min")
rows = []
alld = days
for st in starts:
    E = alld + st
    X = E + pd.Timedelta(minutes=HOLD)
    ok = (E.weekday < 5) & ~((E.weekday == 4) & (X.normalize() > E.normalize()))
    r = (P.reindex(X[ok]).values / P.reindex(E[ok]).values - 1) * 1e4
    r = r[~np.isnan(r)]
    rows.append((str(st)[7:12], len(r), r.mean(), tstat(r)))
pl = pd.DataFrame(rows, columns=["start", "n", "mean", "t"]).set_index("start")
brk = pl.loc["23:55"]
print(f"break-window start 23:55: n={brk.n} mean={brk['mean']:+.2f} t={brk.t:+.2f}")
print(f"rank by t among {len(pl)} start times: {int((pl.t > brk.t).sum())+1}; rank by mean: {int((pl['mean'] > brk['mean']).sum())+1}")
print("top 10 start times by t:\n" + pl.sort_values("t", ascending=False).head(10).round(2).to_string())
print("bottom 5:\n" + pl.sort_values("t").head(5).round(2).to_string())
nonbrk = pl[~pl.index.isin([f"{h:02d}:{m:02d}" for h in [21, 22, 23] for m in range(0, 60, 5)])]
print(f"start times not overlapping the break (start<21:00 or >=00:00): mean of means={nonbrk['mean'].mean():+.2f}  sd of t={nonbrk.t.std():.2f}  max t={nonbrk.t.max():+.2f}")
print("Note: windows starting 21:00-23:55 also contain the break; daily gold drift (~+0.4bp/3h over 2018-26) inflates every window.")
# drift-adjusted: subtract unconditional mean 185-min return
uncond = pl["mean"].mean()
print(f"unconditional avg 185-min return across all start times: {uncond:+.2f}bp -> break excess = {brk['mean']-uncond:+.2f}bp")
# Friday into weekend
fri = days[days.weekday == 4]
mon_ = fri + pd.Timedelta(days=3)
fr = pd.Series((P.reindex(mon_ + pd.Timedelta(minutes=180)).values / P.reindex(fri + pd.Timedelta(minutes=1435)).values - 1) * 1e4, index=fri).dropna()
print("Friday 23:55 -> Monday 03:00 (weekend break):", summ(fr))
# sign-flip and bootstrap
x = base.values
obs = x.mean()
flips = rng.choice([-1, 1], size=(20000, len(x)))
pf = np.mean((flips * x).mean(1) >= obs)
bs = rng.choice(x, size=(20000, len(x))).mean(1)
print(f"sign-flip p (one-sided) = {pf:.5f};  bootstrap 95% CI of mean = [{np.percentile(bs,2.5):+.2f}, {np.percentile(bs,97.5):+.2f}]bp")
# random-day-time placebo: draw random (start time not in 21:00-00:55) each day
res["placebo_rank"] = int((pl.t > brk.t).sum()) + 1

# ---------------------------------------------------------------- 5 execution stress
hdr("TEST 5: execution stress (default 23:55 -> 03:00)")
for k in [1, 2, 3]:
    print(f"  Deriv cost x{k} ({DERIV_COST*k:.1f}bp): {summ(base - DERIV_COST*k)}")
for ed in [5]:
    r = break_trades(DEF_E, DEF_X, extra_entry_delay=ed).dropna()
    print(f"  entry delayed {ed} min (at the halt):      gross {summ(r)} | net {r.mean()-DERIV_COST:+.2f}")
for en in [10, 15]:
    r = break_trades(en, DEF_X).dropna()
    print(f"  entry earlier, {en} min before halt:       gross {summ(r)} | net {r.mean()-DERIV_COST:+.2f}")
for xd in [5, 15]:
    r = break_trades(DEF_E, DEF_X, extra_exit_delay=xd).dropna()
    print(f"  exit delayed {xd} min:                    gross {summ(r)} | net {r.mean()-DERIV_COST:+.2f}")
for sl in [1, 2, 3]:
    print(f"  extra slippage {sl}bp (entry thin pre-halt): net {summ(base - DERIV_COST - sl)}")
# short exit right after reopen: gap slippage matters
r5 = break_trades(DEF_E, 5).dropna()
print(f"  exit 5 min after reopen with 2bp reopen slippage: net {summ(r5 - DERIV_COST - 2)}")
# MT5 costs: spread from ticks + swap
spread_entry = spread_exit = np.nan
tick_rets = None
try:
    import MetaTrader5 as mt5
    mt5.initialize()
    si = mt5.symbol_info("XAUUSD")
    print(f"  MT5 symbol_info: spread={si.spread} points, swap_long={si.swap_long} points/lot/night, triple day={si.swap_rollover3days}")
    rows = []
    for D in trade_days[trade_days >= pd.Timestamp("2025-01-01")]:
        # MT5 python treats request datetimes as UTC; server is UTC+2/+3 -> pad generously
        t = mt5.copy_ticks_range("XAUUSD", (D - pd.Timedelta(hours=1)).to_pydatetime(), (D + pd.Timedelta(hours=8)).to_pydatetime(), mt5.COPY_TICKS_ALL)
        if t is None or len(t) == 0:
            continue
        tk = pd.DataFrame(t)
        tk["s"] = pd.to_datetime(tk.time_msc, unit="ms")
        tk = tk.set_index("s").sort_index()
        e_ = tk.loc[:D - pd.Timedelta(minutes=5)]
        x_ = tk.loc[D + pd.Timedelta(minutes=60):D + pd.Timedelta(minutes=63)]
        x2 = tk.loc[:D + pd.Timedelta(minutes=180)]
        e1 = tk.loc[:D - pd.Timedelta(minutes=4)]
        e2 = tk.loc[:D - pd.Timedelta(minutes=3)]
        e5 = tk.loc[:D - pd.Timedelta(seconds=5)]
        if len(e_) == 0 or len(x2) == 0 or x2.index[-1] < D + pd.Timedelta(minutes=170):
            continue
        a = e_.iloc[-1]
        b = x2.iloc[-1]
        rows.append(dict(D=D, spr_e=(a.ask - a.bid) / a.bid * 1e4, spr_x=(b.ask - b.bid) / b.bid * 1e4,
                         spr_reopen=((x_.ask - x_.bid) / x_.bid * 1e4).median() if len(x_) else np.nan,
                         spr_lastsec=(e5.iloc[-1].ask - e5.iloc[-1].bid) / e5.iloc[-1].bid * 1e4,
                         mid=((b.bid + b.ask) / (a.bid + a.ask) - 1) * 1e4,
                         real=(b.bid / a.ask - 1) * 1e4,
                         real_d1=(b.bid / e1.iloc[-1].ask - 1) * 1e4, real_d2=(b.bid / e2.iloc[-1].ask - 1) * 1e4,
                         real_d5=(b.bid / e5.iloc[-1].ask - 1) * 1e4))
    mt5.shutdown()
    tick_rets = pd.DataFrame(rows).set_index("D")
    if len(tick_rets):
        print(f"  MT5 ticks {tick_rets.index[0].date()}..{tick_rets.index[-1].date()} n={len(tick_rets)}: median spread entry(23:55)={tick_rets.spr_e.median():.2f}bp, "
              f"last-second-before-halt={tick_rets.spr_lastsec.median():.2f}bp, reopen 01:00-01:03={tick_rets.spr_reopen.median():.2f}bp, exit(03:00)={tick_rets.spr_x.median():.2f}bp")
        print(f"   mid:              {summ(tick_rets.mid)}")
        print(f"   ask-in/bid-out:   {summ(tick_rets.real)}")
        for c, lab in [("real_d1", "entry 1 min later (23:56)"), ("real_d2", "entry 2 min later (23:57)"), ("real_d5", "entry at 23:59:55")]:
            print(f"   {lab:26s}: {summ(tick_rets[c])}")
        sw = np.where(tick_rets.index.weekday == 3, 3 * 1.34, 1.34)  # Wed-night rollover (Thu morning) triple
        print(f"   ask/bid minus swap (1.34bp, x3 Wed night): {summ(tick_rets.real - sw)}")
        print(f"   ask/bid minus Deriv 1.2bp (proxy):         {summ(tick_rets.real - DERIV_COST)}")
except Exception as ex:  # noqa
    print("  MT5 tick pull failed:", ex)
sw_long = np.where(base.index.weekday == 3, 3 * 1.34, 1.34)
print(f"  Duka (bid) long history with MT5 costs (0.25bp spread + swap x3 Wed): {summ(base - 0.25 - sw_long)}")

# ---------------------------------------------------------------- 6 walk-forward
hdr("TEST 6: walk-forward - choose best (entry,exit) by t-stat on all PRIOR years, trade next year (net Deriv)")
years = sorted(set(base.index.year))
wf = []
for y in years[1:]:
    best, bt = None, -9
    for k, r in TR.items():
        tr = r[r.index.year < y].dropna()
        tt = tstat(tr)
        if tt > bt:
            best, bt = k, tt
    oos = TR[best][TR[best].index.year == y].dropna() - DERIV_COST
    dfl = TR[(DEF_E, DEF_X)][TR[(DEF_E, DEF_X)].index.year == y].dropna() - DERIV_COST
    wf.append((y, best, round(bt, 2), oos.sum(), oos.mean(), tstat(oos), dfl.sum(), dfl.mean()))
wf = pd.DataFrame(wf, columns=["year", "chosen(e,x)", "IS_t", "OOS_sum_bp", "OOS_mean", "OOS_t", "default_sum_bp", "default_mean"])
print(wf.round(2).to_string(index=False))
print(f"OOS years positive: {(wf.OOS_sum_bp>0).sum()}/{len(wf)};  total OOS {wf.OOS_sum_bp.sum():+.0f}bp vs default {wf.default_sum_bp.sum():+.0f}bp")

# ---------------------------------------------------------------- 7 Monte Carlo
hdr("TEST 7: Monte Carlo on default trade sequence, net Deriv 1.2bp, x100 stop-out at -1%")
lows = min_low_between(base.index, DEF_E, DEF_X)
stop = lows <= -100
print(f"trades where bid low fell >1% below entry before exit: {int(stop.sum())}/{len(base)} ({stop.mean()*100:.2f}%) ; >0.5%: {(lows<=-50).mean()*100:.2f}%")
print("  worst 5 intratrade lows (bp):", lows.nsmallest(5).round(0).to_dict())
rr = base.copy()
rr[stop] = -100.0
net = (rr - DERIV_COST) / 1e4
tpy = len(base) / (len(years) - 0.2)
NT = int(round(tpy))
print(f"trades/year ~ {NT}")
for L in [1, 2, 3]:
    sims = 5000
    # block bootstrap (blocks of 10 trades) to keep any clustering
    nb = NT // 10 + 1
    starts_ = rng.integers(0, len(net) - 10, size=(sims, nb))
    idx = (starts_[:, :, None] + np.arange(10)).reshape(sims, -1)[:, :NT]
    path = net.values[idx] * L
    eq = np.cumprod(1 + path, axis=1)
    ann = eq[:, -1] - 1
    dd = (1 - eq / np.maximum.accumulate(eq, axis=1)).max(1)
    los = path < 0
    streak = np.array([max((len(s) for s in "".join("1" if v else "0" for v in row).split("0")), default=0) for row in los[:500]])
    print(f"  {L}x notional: annual ret median {np.median(ann)*100:+.1f}% [5%: {np.percentile(ann,5)*100:+.1f}%, 95%: {np.percentile(ann,95)*100:+.1f}%], "
          f"P(losing year)={np.mean(ann<0)*100:.1f}%, maxDD median {np.median(dd)*100:.1f}% / 95th {np.percentile(dd,95)*100:.1f}%, "
          f"longest losing streak median {np.median(streak):.0f} (95th {np.percentile(streak,95):.0f})")
# full-history actual equity at 1x
eq = np.cumprod(1 + net.values)
print(f"  actual 2018-2026 path at 1x: total {(eq[-1]-1)*100:+.1f}%, maxDD {(1-eq/np.maximum.accumulate(eq)).max()*100:.1f}%")
ann_act = net.groupby(net.index.year).apply(lambda s: np.prod(1 + s) - 1) * 100
print("  actual yearly % at 1x:", ann_act.round(2).to_dict())

# ---------------------------------------------------------------- 8 multiple testing
hdr("TEST 8: multiple testing")
N_TRIALS = 40 + 24 + 10  # prior time-of-day scans + timing grid + splits/variants
x = base.values
sr = x.mean() / x.std(ddof=1)
T = len(x)
p_raw = 2 * (1 - stats.t.cdf(abs(tstat(x)), T - 1))
print(f"default per-trade SR={sr:.4f} (annualised ~{sr*math.sqrt(NT):.2f}), t={tstat(x):.2f}, raw p={p_raw:.2e}, Bonferroni x{N_TRIALS} p={min(1,p_raw*N_TRIALS):.2e}")
# net
xn = x - DERIV_COST
p_net = 2 * (1 - stats.t.cdf(abs(tstat(xn)), T - 1))
print(f"net of Deriv: t={tstat(xn):.2f}, raw p={p_net:.2e}, Bonferroni p={min(1,p_net*N_TRIALS):.2e}")
# deflated Sharpe: variance of SR across trials taken from placebo distribution (start-time SRs)
sr_trials = (nonbrk.t / np.sqrt(nonbrk.n)).dropna().values
V = np.var(sr_trials, ddof=1)
g = 0.5772
sr0 = math.sqrt(V) * ((1 - g) * stats.norm.ppf(1 - 1 / N_TRIALS) + g * stats.norm.ppf(1 - 1 / (N_TRIALS * math.e)))
sk, ku = stats.skew(x), stats.kurtosis(x, fisher=False)
for lab, xx in [("gross", x), ("net", xn)]:
    s_ = xx.mean() / xx.std(ddof=1)
    dsr = stats.norm.cdf((s_ - sr0) * math.sqrt(T - 1) / math.sqrt(1 - sk * s_ + (ku - 1) / 4 * s_ ** 2))
    print(f"  deflated Sharpe ({lab}): SR0 (expected max of {N_TRIALS} null trials)={sr0:.4f}, SR={s_:.4f}, DSR prob={dsr:.4f}  (skew {sk:.2f}, kurt {ku:.1f})")
# robustness to outliers
tr = np.sort(x)
print(f"  trimmed mean (drop top/bottom 1%): {stats.trim_mean(x,0.01):+.2f}bp; median {np.median(x):+.2f}bp; without top 10 trades: {np.sort(x)[:-10].mean():+.2f}bp")

# ---------------------------------------------------------------- 9 mechanism
hdr("TEST 9: mechanism probes")
D = base.index
p_e = px_at(D - pd.Timedelta(minutes=5))
p_h = px_at(D)  # price at halt
p_o = px_open.reindex(D + pd.Timedelta(minutes=60)).values  # first bar open at reopen
segs = {"23:55->halt": (px_at(D) / p_e - 1) * 1e4,
        "halt->reopen open (gap)": (p_o / p_h - 1) * 1e4,
        "reopen open->01:05": (px_at(D + pd.Timedelta(minutes=65)) / p_o - 1) * 1e4,
        "01:05->01:30": (px_at(D + pd.Timedelta(minutes=90)) / px_at(D + pd.Timedelta(minutes=65)) - 1) * 1e4,
        "01:30->02:00": (px_at(D + pd.Timedelta(minutes=120)) / px_at(D + pd.Timedelta(minutes=90)) - 1) * 1e4,
        "02:00->03:00": (px_at(D + pd.Timedelta(minutes=180)) / px_at(D + pd.Timedelta(minutes=120)) - 1) * 1e4,
        "03:00->05:00 (after exit)": (px_at(D + pd.Timedelta(minutes=300)) / px_at(D + pd.Timedelta(minutes=180)) - 1) * 1e4,
        "05:00->09:00": (px_at(D + pd.Timedelta(minutes=540)) / px_at(D + pd.Timedelta(minutes=300)) - 1) * 1e4}
for k, v in segs.items():
    print(f"  {k:28s}: {summ(v)}")
# Dukascopy bid spread artifact check: the reopen bid may be depressed/elevated by wide spreads
# previous NY session move (15:00->23:55 srv = 08:00->16:55 NY)
ny = (p_e / px_at(D - pd.Timedelta(hours=9)) - 1) * 1e4
dfm = pd.DataFrame({"r": base.values, "ny": ny}, index=D).dropna()
print(f"  corr(prev NY session move, drift) = {dfm.corr().iloc[0,1]:+.3f}")
for lab, msk in [("NY up", dfm.ny > 0), ("NY down", dfm.ny <= 0),
                 ("NY big up (>+1sd)", dfm.ny > dfm.ny.std()), ("NY big down (<-1sd)", dfm.ny < -dfm.ny.std())]:
    print(f"    {lab:20s}: {summ(dfm.r[msk])}")
# break length (holidays): last bar end before halt vs first bar after
lastpre, firstpost = [], []
bi = bars.index
for d0 in D:
    i = bi.searchsorted(d0 + pd.Timedelta(minutes=30))
    lastpre.append(bi[i - 1] + pd.Timedelta(minutes=5))
    firstpost.append(bi[i])
gap = (pd.DatetimeIndex(firstpost) - pd.DatetimeIndex(lastpre)).total_seconds() / 60
print(f"  break length distribution (min): {pd.Series(gap).value_counts().head(5).to_dict()}")
# all days incl. holiday (early-close) breaks: entry = last price before break, exit 120 after reopen
alld2 = days[days.weekday.isin([1, 2, 3, 4])]
rec = []
for d0 in alld2:
    i = bi.searchsorted(d0 + pd.Timedelta(minutes=30))
    if i == 0 or i >= len(bi):
        continue
    lp, fp = bi[i - 1] + pd.Timedelta(minutes=5), bi[i]
    if (fp - lp) > pd.Timedelta("30h"):
        continue
    pe_ = bars.close.iloc[i - 1]
    xt = fp + pd.Timedelta(minutes=120)
    pxx = PX.get(xt, np.nan)
    rec.append((d0, (fp - lp).total_seconds() / 60, (pxx / pe_ - 1) * 1e4))
hb = pd.DataFrame(rec, columns=["D", "gap", "r"]).set_index("D").dropna()
print(f"  normal 60-min breaks: {summ(hb.r[hb.gap == 60])}")
print(f"  longer breaks (>60, holidays/early close): {summ(hb.r[hb.gap > 60])}")
# vs EURUSD dollar move across break (does gold drift = dollar weakness?)
try:
    eu = load_duka("frxEURUSD")
    epx = pd.Series(eu.close.values, index=eu.index + pd.Timedelta(minutes=5)).reindex(grid, method="ffill", tolerance=pd.Timedelta("10min"))
    eur = pd.Series((epx.reindex(D + pd.Timedelta(minutes=180)).values / epx.reindex(D - pd.Timedelta(minutes=5)).values - 1) * 1e4, index=D)
    jj = pd.concat([base.rename("xau"), eur.rename("eur")], axis=1).dropna()
    b_ = np.polyfit(jj.eur, jj.xau, 1)
    resid = jj.xau - b_[0] * jj.eur
    print(f"  EURUSD same window: {summ(jj.eur)}; beta xau on eur={b_[0]:.2f}; gold drift ex-dollar (resid incl. intercept): {summ(resid)}")
except Exception as ex:  # noqa
    print("  EURUSD check failed", ex)

hdr("DONE")
