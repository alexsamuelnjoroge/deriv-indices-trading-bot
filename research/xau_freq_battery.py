"""
XAUUSD frequent-trading research battery (pre-registered 2026-11-10).

Goal: MT5 gold strategies that trade often (>= 150 trades/yr), are net-profitable
after costs, and survive data they were not selected on.

Data : Dukascopy M5 bid 2018 -> 2026-11 (data/duka/frxXAUUSD_M5_*.csv, UTC).
Costs: IC Markets raw MT5 — spread ~$0.09 + commission $0.07/oz round trip
       ~= 0.5bp at $4100. Base charge 0.6bp RT, stress 1.2bp RT.
       Every trade is flat before the daily rollover (20:30-22:30 UTC blocked),
       so no swap is paid. Entry at next bar open, exit at bar close (time exit).
Split: TRAIN 2018-2022 selects; TEST 2023 -> now judges. TEST is not used for any choice.
Pass rule (fixed before running):
       TRAIN net@0.6bp mean > 0, TRAIN daily-clustered t >= 3.5 (~750 configs tested),
       >= 150 trades/yr; then TEST net@0.6bp mean > 0 with t >= 2, and net@1.2bp > 0.
Families:
  F1 overreaction: k-bar move z-score vs trailing vol -> fade or follow
  F2 session-VWAP deviation -> fade or follow
  F3 vol-regime intraday time-series momentum / reversal (hourly entries)
  F4 hour-of-week seasonality (cells chosen on TRAIN)
  F5 gradient boosting, yearly expanding walk-forward on ~20 price features
Gold rose ~+150% over 2023-26, so long and short legs are reported separately,
and "vs drift" removes the unconditional drift over the same holding time.
"""

import glob
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

COST, STRESS = 0.6, 1.2
TRAIN_END = pd.Timestamp("2023-01-01", tz="UTC")
MIN_TPY = 150
T_TRAIN = 3.5

t0 = time.time()


def log(*a):
    print(*a, flush=True)


# ── data ─────────────────────────────────────────────────────────────────────
parts = [pd.read_csv(f, index_col=0, parse_dates=True)
         for f in sorted(glob.glob("data/duka/frxXAUUSD_M5_*.csv"))]
df = pd.concat(parts)
df = df[~df.index.duplicated()].sort_index()
df.index = df.index.tz_convert("UTC")
df = df[df["volume"] > 0]

N = len(df)
ts = (df.index.asi8 // 10**9).astype(np.int64)
lp = np.log(df["close"].to_numpy()) * 1e4
lpo = np.log(df["open"].to_numpy()) * 1e4
hi = np.log(df["high"].to_numpy()) * 1e4
lo = np.log(df["low"].to_numpy()) * 1e4
vol = df["volume"].to_numpy()
hour = df.index.hour.to_numpy()
minute = df.index.minute.to_numpy()
dow = df.index.dayofweek.to_numpy()
mod = hour * 60 + minute
year = df.index.year.to_numpy()
is_train = df.index < TRAIN_END

sess = (ts - int(20.5 * 3600)) // 86400
entry_ok = ~((mod >= 20 * 60 + 30) & (mod < 22 * 60 + 30))

r1 = np.r_[np.nan, np.diff(lp)]
r1[1:][np.diff(ts) > 3600] = np.nan          # don't count gap returns as volatility
sig = pd.Series(r1).rolling(288, min_periods=200).std().to_numpy()
sig48 = pd.Series(r1).rolling(48, min_periods=40).std().to_numpy()

SESS_MASK = {
    "all":  entry_ok,
    "asia": entry_ok & ((mod >= 22 * 60 + 30) | (hour < 7)),
    "eu":   entry_ok & (hour >= 7) & (hour < 12),
    "us":   entry_ok & (hour >= 12) & (hour < 20),
}
log(f"loaded {N} bars {df.index[0]} -> {df.index[-1]}  ({time.time()-t0:.0f}s)")

# unconditional drift per bar, by split (for "vs drift")
drift_tr = np.nanmean(np.where(is_train, r1, np.nan))
drift_te = np.nanmean(np.where(~is_train, r1, np.nan))


# ── backtest core ────────────────────────────────────────────────────────────
def trades(cand_idx, cand_side, H):
    """Greedy non-overlapping time-exit trades. Entry open[i+1], exit close[i+H]."""
    out_i, out_s = [], []
    pos, n = 0, len(cand_idx)
    while pos < n:
        i = cand_idx[pos]
        e = i + H
        if (e < N and sess[e] == sess[i + 1] == sess[i]
                and ts[e] - ts[i] <= H * 300 + 3600):
            out_i.append(i)
            out_s.append(cand_side[pos])
            pos = np.searchsorted(cand_idx, e, "left")
        else:
            pos += 1
    out_i = np.asarray(out_i, dtype=np.int64)
    out_s = np.asarray(out_s, dtype=np.int8)
    gross = out_s * (lp[out_i + H] - lpo[out_i + 1]) if len(out_i) else np.array([])
    return out_i, out_s, gross


def stats(idx, side, gross, H, mask_split, drift):
    m = mask_split[idx] if len(idx) else np.array([], bool)
    i, s, g = idx[m], side[m], gross[m]
    if len(i) < 30:
        return None
    net = g - COST
    days = sess[i]
    dsum = pd.Series(net).groupby(days).sum()
    ndays_span = (ts[i[-1]] - ts[i[0]]) / 86400 / 365.25
    t = dsum.mean() / dsum.std(ddof=1) * np.sqrt(len(dsum)) if dsum.std(ddof=1) > 0 else 0
    yrs = pd.Series(net).groupby(year[i]).mean()
    return {
        "n": len(i), "tpy": len(i) / max(ndays_span, 0.25),
        "net": net.mean(), "net_stress": (g - STRESS).mean(),
        "wr": (net > 0).mean() * 100, "t": t,
        "long": g[s > 0].mean() - COST if (s > 0).any() else np.nan,
        "short": g[s < 0].mean() - COST if (s < 0).any() else np.nan,
        "nl": int((s > 0).sum()), "ns": int((s < 0).sum()),
        "vs_drift": (g - s * drift * H).mean() - COST,
        "yrs_pos": f"{(yrs > 0).sum()}/{len(yrs)}",
        "by_year": " ".join(f"{y % 100:02d}:{v:+.1f}" for y, v in yrs.items()),
    }


results = []


def evaluate(family, cfg, cand_mask, side_arr, H):
    idx = np.flatnonzero(cand_mask)
    if len(idx) < 30:
        return
    idx_, side_, gross = trades(idx, side_arr[idx], H)
    if len(idx_) < 30:
        return
    tr = stats(idx_, side_, gross, H, is_train, drift_tr)
    te = stats(idx_, side_, gross, H, ~is_train, drift_te)
    if tr is None or te is None:
        return
    results.append({"family": family, "cfg": cfg, "H": H, "tr": tr, "te": te})


def first_bar(cond):
    prev = np.r_[False, cond[:-1]]
    return cond & ~prev


# ── F1 overreaction ──────────────────────────────────────────────────────────
log("F1 overreaction ...")
for k in (1, 3, 6, 12):
    rk = lp - np.r_[np.full(k, np.nan), lp[:-k]]
    z = rk / (sig * np.sqrt(k))
    for thr in (2.0, 2.5, 3.0, 4.0):
        big = np.abs(z) > thr
        fire = first_bar(np.nan_to_num(big, nan=False).astype(bool))
        zs = np.sign(np.nan_to_num(z))
        for sname, smask in SESS_MASK.items():
            c = fire & smask
            for mode, sgn in (("fade", -1), ("follow", 1)):
                side = (sgn * zs).astype(np.int8)
                for H in (3, 6, 12, 24):
                    evaluate("F1", f"k{k} z>{thr} {sname} {mode}", c, side, H)
log(f"  {len(results)} configs ({time.time()-t0:.0f}s)")

# ── F2 session VWAP deviation ────────────────────────────────────────────────
log("F2 VWAP ...")
tp = (hi + lo + lp) / 3
s_ser = pd.Series(sess)
cum_pv = pd.Series(tp * vol).groupby(s_ser).cumsum().to_numpy()
cum_v = pd.Series(vol).groupby(s_ser).cumsum().to_numpy()
vwap = cum_pv / cum_v
bars_in = s_ser.groupby(s_ser).cumcount().to_numpy()
dev = (lp - vwap) / (sig * np.sqrt(48))
n_f2 = len(results)
for thr in (1.0, 1.5, 2.0, 3.0):
    fire = first_bar(np.nan_to_num(np.abs(dev) > thr, nan=False).astype(bool)) & (bars_in >= 24)
    ds = np.sign(np.nan_to_num(dev))
    for sname in ("all", "eu", "us"):
        c = fire & SESS_MASK[sname]
        for mode, sgn in (("fade", -1), ("follow", 1)):
            side = (sgn * ds).astype(np.int8)
            for H in (6, 12, 24, 48):
                evaluate("F2", f"vwapdev>{thr} {sname} {mode}", c, side, H)
log(f"  {len(results)-n_f2} configs ({time.time()-t0:.0f}s)")

# ── F3 regime momentum / reversal ────────────────────────────────────────────
log("F3 regime TSMOM ...")
rv_med = pd.Series(sig).rolling(288 * 60, min_periods=288 * 20).median().to_numpy()
regimes = {"all": np.ones(N, bool), "highvol": sig > rv_med, "lowvol": sig <= rv_med}
hourly = (minute == 0) & entry_ok
n_f3 = len(results)
for L in (12, 24, 48, 96):
    rL = lp - np.r_[np.full(L, np.nan), lp[:-L]]
    ms = np.sign(np.nan_to_num(rL))
    for rname, rmask in regimes.items():
        c = hourly & rmask & (ms != 0)
        for mode, sgn in (("mom", 1), ("rev", -1)):
            side = (sgn * ms).astype(np.int8)
            for H in (12, 24, 48):
                evaluate("F3", f"L{L} {rname} {mode}", c, side, H)
log(f"  {len(results)-n_f3} configs ({time.time()-t0:.0f}s)")

# ── F4 hour-of-week seasonality ──────────────────────────────────────────────
log("F4 hour-of-week ...")
H4 = 12
fwd = np.full(N, np.nan)
fwd[:-H4] = lp[H4:] - lpo[np.minimum(np.arange(1, N - H4 + 1), N - 1)]
valid_h = hourly & np.r_[sess[H4:] == sess[:-H4], np.zeros(H4, bool)]
cell = dow * 24 + hour
tr_df = pd.DataFrame({"cell": cell[valid_h & is_train], "r": fwd[valid_h & is_train]}).dropna()
g = tr_df.groupby("cell")["r"]
cell_t = g.mean() / g.std() * np.sqrt(g.count())
f4_cells = {}
for thr in (2.5, 3.0):
    chosen = cell_t[cell_t.abs() >= thr]
    side = np.zeros(N, np.int8)
    for c_, tv in chosen.items():
        side[cell == c_] = 1 if tv > 0 else -1
    f4_cells[thr] = {int(c_): round(float(tv), 2) for c_, tv in chosen.items()}
    evaluate("F4", f"HoW |t|>={thr} ({len(chosen)} cells)", valid_h & (side != 0), side, H4)
log(f"  cells: {f4_cells}")

# ── F5 gradient-boosting walk-forward ────────────────────────────────────────
log("F5 ML walk-forward ...")
def lagret(k):
    return (lp - np.r_[np.full(k, np.nan), lp[:-k]]) / (sig * np.sqrt(k))

sess_hi = pd.Series(hi).groupby(s_ser).max()
sess_lo = pd.Series(lo).groupby(s_ser).min()
prev_hi = s_ser.map(sess_hi.shift(1)).to_numpy()
prev_lo = s_ser.map(sess_lo.shift(1)).to_numpy()
sess_open = pd.Series(lpo).groupby(s_ser).transform("first").to_numpy()
unit = sig * np.sqrt(48)
rng_hi = pd.Series(hi).rolling(48).max().to_numpy()
rng_lo = pd.Series(lo).rolling(48).min().to_numpy()
X = np.column_stack([
    lagret(1), lagret(3), lagret(6), lagret(12), lagret(48), lagret(96), lagret(288),
    sig48 / sig, sig / rv_med,
    np.sin(2 * np.pi * mod / 1440), np.cos(2 * np.pi * mod / 1440), dow,
    dev, (lp - prev_hi) / unit, (lp - prev_lo) / unit, (lp - sess_open) / unit,
    (lp - rng_lo) / np.maximum(rng_hi - rng_lo, 1e-9),
    vol / pd.Series(vol).rolling(288, min_periods=100).mean().to_numpy(),
    bars_in,
])
H5 = 12
y5 = np.full(N, np.nan)
y5[:-H5] = lp[H5:] - lpo[np.minimum(np.arange(1, N - H5 + 1), N - 1)]
ok5 = entry_ok & np.r_[sess[H5:] == sess[:-H5], np.zeros(H5, bool)] & np.isfinite(X).all(1) & np.isfinite(y5)
pred = np.full(N, np.nan)
for Y in range(2020, int(year.max()) + 1):
    trn = ok5 & (year < Y) & (np.arange(N) % 3 == 0)
    tst = ok5 & (year == Y)
    yt = np.clip(y5[trn], -60, 60)
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
                                      min_samples_leaf=1000, l2_regularization=1.0,
                                      random_state=0)
    m.fit(X[trn], yt)
    pred[tst] = m.predict(X[tst])
    log(f"  fit <{Y}: n={trn.sum()} -> predict {tst.sum()} ({time.time()-t0:.0f}s)")
psign = np.sign(np.nan_to_num(pred)).astype(np.int8)
for q in (0.5, 1.0, 2.0, 3.0):
    c = np.isfinite(pred) & (np.abs(np.nan_to_num(pred)) > q)
    evaluate("F5", f"GBR pred>|{q}|bp (OOS 2020+)", c, psign, H5)

# ── report ───────────────────────────────────────────────────────────────────
log(f"\nTOTAL configs: {len(results)}  ({time.time()-t0:.0f}s)")
log(f"drift/bar  train {drift_tr:+.4f}bp  test {drift_te:+.4f}bp")


def fmt(r, part):
    s = r[part]
    return (f"n={s['n']:5d} tpy={s['tpy']:4.0f} net={s['net']:+5.2f} str={s['net_stress']:+5.2f} "
            f"t={s['t']:+5.2f} WR={s['wr']:4.1f} L={s['long']:+5.2f}({s['nl']}) "
            f"S={s['short']:+5.2f}({s['ns']}) vsD={s['vs_drift']:+5.2f} yrs+={s['yrs_pos']}")


log("\n=== PASSING TRAIN SELECTION (net>0, t>=%.1f, tpy>=%d) ===" % (T_TRAIN, MIN_TPY))
sel = [r for r in results if r["tr"]["net"] > 0 and r["tr"]["t"] >= T_TRAIN and r["tr"]["tpy"] >= MIN_TPY]
sel.sort(key=lambda r: -r["tr"]["t"])
for r in sel:
    verdict = ("PASS" if r["te"]["net"] > 0 and r["te"]["t"] >= 2 and r["te"]["net_stress"] > 0
               else "fail")
    log(f"[{verdict}] {r['family']} {r['cfg']} H={r['H']}")
    log(f"   TRAIN {fmt(r, 'tr')}")
    log(f"   TEST  {fmt(r, 'te')}")
    log(f"   TEST years: {r['te']['by_year']}")
if not sel:
    log("  (none)")

log("\n=== TOP 15 BY TRAIN t (any frequency), for context ===")
for r in sorted(results, key=lambda r: -r["tr"]["t"])[:15]:
    log(f"{r['family']} {r['cfg']} H={r['H']}")
    log(f"   TRAIN {fmt(r, 'tr')}")
    log(f"   TEST  {fmt(r, 'te')}")

log("\n=== F5 ML (all OOS; 2020-22 = 'train' column for threshold choice) ===")
for r in results:
    if r["family"] == "F5":
        log(f"{r['cfg']}\n   20-22 {fmt(r, 'tr')}\n   23+   {fmt(r, 'te')}\n   years {r['te']['by_year']}")

log("\n=== GROSS EDGE MAP: best TEST net per family among configs with TRAIN net>0 & tpy>=150 ===")
for fam in ("F1", "F2", "F3", "F4", "F5"):
    pool = [r for r in results if r["family"] == fam and r["tr"]["net"] > 0 and r["tr"]["tpy"] >= MIN_TPY]
    log(f"{fam}: {len(pool)} configs train-positive; "
        f"test-positive {sum(r['te']['net'] > 0 for r in pool)}; "
        f"test t>=2 {sum(r['te']['t'] >= 2 for r in pool)}")

pd.DataFrame([{"family": r["family"], "cfg": r["cfg"], "H": r["H"],
               **{f"tr_{k}": v for k, v in r["tr"].items()},
               **{f"te_{k}": v for k, v in r["te"].items()}} for r in results]
             ).to_csv("research/xau_freq_battery_results.csv", index=False)
log("wrote research/xau_freq_battery_results.csv")
