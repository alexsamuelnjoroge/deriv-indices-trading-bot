"""
XAUUSD cross-asset research battery #2 (pre-registered 2026-11-10).

Battery #1 (xau_freq_battery.py) used gold's own price only: nothing passed.
This battery conditions gold on what drives it intraday.

Factors (Dukascopy M5 bid, aligned to gold bars, ffill <= 1h):
  usd  = equal-weight USD basket from EURUSD GBPUSD AUDUSD USDJPY USDCHF USDCAD
  dxy  = US Dollar Index CFD      xag = silver
  bond = US T-bond CFD (rates)    spx = US500 CFD
Betas: OLS of gold 5-min returns on factor returns over the previous 20 sessions,
refit every session (past-only). Residual e = gold return - beta . factors.

Families:
  G1 single-factor residual: cumulative residual over W bars in vol units; fade
     (gold moved without its driver -> reverts) or follow (driver moved, gold lags)
  G2 multi-factor residual (usd+xag+bond+spx), by session
  G3 confirmation split of gold overreactions: fade moves the factors don't
     explain, follow moves they do (and the two opposite controls)
Costs/split/pass rule identical to battery #1: 0.6bp RT (stress 1.2bp), flat before
rollover, TRAIN 2018-22 selects, TEST 2023+ judges; TRAIN t >= 3.5, >= 150 trades/yr,
then TEST net > 0, t >= 2, stress net > 0.
"""

import glob
import time

import numpy as np
import pandas as pd

COST, STRESS = 0.6, 1.2
TRAIN_END = pd.Timestamp("2023-01-01", tz="UTC")
MIN_TPY, T_TRAIN = 150, 3.5
t0 = time.time()


def log(*a):
    print(*a, flush=True)


def load(prefix):
    fs = sorted(glob.glob(f"data/duka/{prefix}_M5_*.csv"))
    df = pd.concat(pd.read_csv(f, index_col=0, parse_dates=True) for f in fs)
    df = df[~df.index.duplicated()].sort_index()
    df.index = df.index.tz_convert("UTC")
    return df


g = load("frxXAUUSD")
g = g[g["volume"] > 0]
idx = g.index
N = len(g)
ts = (idx.asi8 // 10**9).astype(np.int64)
lp = np.log(g["close"].to_numpy()) * 1e4
lpo = np.log(g["open"].to_numpy()) * 1e4
hour = idx.hour.to_numpy()
minute = idx.minute.to_numpy()
mod = hour * 60 + minute
year = idx.year.to_numpy()
is_train = idx < TRAIN_END
sess = (ts - int(20.5 * 3600)) // 86400
entry_ok = ~((mod >= 20 * 60 + 30) & (mod < 22 * 60 + 30))
gap = np.r_[True, np.diff(ts) > 3600]

rg = np.r_[np.nan, np.diff(lp)]
rg[gap] = np.nan


def fret(prefix, invert=False):
    c = load(prefix)["close"]
    c = c.reindex(idx, method="ffill", limit=12)
    r = np.r_[np.nan, np.diff(np.log(c.to_numpy()) * 1e4)]
    r[gap] = np.nan
    return -r if invert else r


fx = [fret("frxEURUSD", True), fret("frxGBPUSD", True), fret("frxAUDUSD", True),
      fret("frxUSDJPY"), fret("frxUSDCHF"), fret("frxUSDCAD")]
F = {"usd": np.nanmean(np.vstack(fx), axis=0),
     "dxy": fret("DXY"), "xag": fret("XAGUSD"), "bond": fret("USTBOND"), "spx": fret("USA500")}
for k, v in F.items():
    log(f"factor {k}: coverage {np.isfinite(v).mean()*100:.1f}%")
log(f"loaded ({time.time()-t0:.0f}s)")

sess_rank = np.unique(sess, return_inverse=True)[1]
NS = sess_rank.max() + 1
SIG = pd.Series(rg).rolling(288, min_periods=200).std().to_numpy()


def residual(names, lookback=20):
    X = np.column_stack([F[n] for n in names])
    ok = np.isfinite(rg) & np.isfinite(X).all(1)
    Xo = np.where(ok[:, None], X, 0.0)
    yo = np.where(ok, rg, 0.0)
    k = X.shape[1]
    xtx = np.zeros((NS, k, k))
    xty = np.zeros((NS, k))
    np.add.at(xtx, sess_rank, np.einsum("ni,nj->nij", Xo, Xo))
    np.add.at(xty, sess_rank, Xo * yo[:, None])
    cx = np.cumsum(xtx, 0)
    cy = np.cumsum(xty, 0)
    beta = np.full((NS, k), np.nan)
    for s in range(lookback, NS):
        A = cx[s - 1] - cx[s - 1 - lookback]
        b = cy[s - 1] - cy[s - 1 - lookback]
        try:
            beta[s] = np.linalg.solve(A + 1e-6 * np.eye(k), b)
        except np.linalg.LinAlgError:
            pass
    B = beta[sess_rank]
    fit = np.einsum("ni,ni->n", np.nan_to_num(X), B)
    e = np.where(ok, rg - fit, np.nan)
    # out-of-sample R^2 of the past-only betas
    m = ok & np.isfinite(fit)
    r2 = {}
    for nm, part in (("train", is_train), ("test", ~is_train)):
        mm = m & part
        r2[nm] = 1 - np.sum((rg[mm] - fit[mm]) ** 2) / np.sum(rg[mm] ** 2)
    return e, fit, r2


def roll_sum(x, W):
    return pd.Series(np.nan_to_num(x)).rolling(W).sum().to_numpy()


def first_bar(c):
    return c & ~np.r_[False, c[:-1]]


def trades(cand_idx, cand_side, H):
    out_i, out_s = [], []
    pos, n = 0, len(cand_idx)
    while pos < n:
        i = cand_idx[pos]
        e = i + H
        if e < N and sess[e] == sess[i + 1] == sess[i] and ts[e] - ts[i] <= H * 300 + 3600:
            out_i.append(i)
            out_s.append(cand_side[pos])
            pos = np.searchsorted(cand_idx, e, "left")
        else:
            pos += 1
    out_i = np.asarray(out_i, dtype=np.int64)
    out_s = np.asarray(out_s, dtype=np.int8)
    gross = out_s * (lp[out_i + H] - lpo[out_i + 1]) if len(out_i) else np.array([])
    return out_i, out_s, gross


def stats(i_, s_, g_, part):
    m = part[i_]
    i, s, gr = i_[m], s_[m], g_[m]
    if len(i) < 30:
        return None
    net = gr - COST
    d = pd.Series(net).groupby(sess[i]).sum()
    span = (ts[i[-1]] - ts[i[0]]) / 86400 / 365.25
    t = d.mean() / d.std(ddof=1) * np.sqrt(len(d)) if d.std(ddof=1) > 0 else 0
    yrs = pd.Series(net).groupby(year[i]).mean()
    return {"n": len(i), "tpy": len(i) / max(span, 0.25), "net": net.mean(),
            "net_stress": (gr - STRESS).mean(), "wr": (net > 0).mean() * 100, "t": t,
            "long": gr[s > 0].mean() - COST if (s > 0).any() else np.nan,
            "short": gr[s < 0].mean() - COST if (s < 0).any() else np.nan,
            "nl": int((s > 0).sum()), "ns": int((s < 0).sum()),
            "yrs_pos": f"{(yrs > 0).sum()}/{len(yrs)}",
            "by_year": " ".join(f"{y % 100:02d}:{v:+.1f}" for y, v in yrs.items())}


results = []


def evaluate(fam, cfg, cand, side, H):
    ci = np.flatnonzero(cand)
    if len(ci) < 30:
        return
    i_, s_, g_ = trades(ci, side[ci], H)
    if len(i_) < 60:
        return
    tr, te = stats(i_, s_, g_, is_train), stats(i_, s_, g_, ~is_train)
    if tr and te:
        results.append({"family": fam, "cfg": cfg, "H": H, "tr": tr, "te": te})


SESS = {"all": entry_ok,
        "asia": entry_ok & ((mod >= 22 * 60 + 30) | (hour < 7)),
        "eu": entry_ok & (hour >= 7) & (hour < 12),
        "us": entry_ok & (hour >= 12) & (hour < 20)}


def residual_family(fam, names, sessions):
    e, fit, r2 = residual(names)
    log(f"{fam} {'+'.join(names)}: OOS R^2 train {r2['train']*100:.1f}% test {r2['test']*100:.1f}%")
    esd = pd.Series(e).rolling(288, min_periods=200).std().to_numpy()
    for W in (6, 12, 24, 48):
        z = roll_sum(e, W) / (esd * np.sqrt(W))
        zs = np.sign(np.nan_to_num(z)).astype(np.int8)
        for thr in (2.0, 2.5, 3.0):
            fire = first_bar(np.nan_to_num(np.abs(z) > thr, nan=False).astype(bool))
            for sn in sessions:
                c = fire & SESS[sn]
                for mode, sg in (("fade", -1), ("follow", 1)):
                    side = (sg * zs).astype(np.int8)
                    for H in (6, 12, 24):
                        evaluate(fam, f"{'+'.join(names)} W{W} z>{thr} {sn} {mode}", c, side, H)
    return e, fit


log("G1 single-factor residuals ...")
for nm in ("usd", "dxy", "xag", "bond", "spx"):
    residual_family("G1", [nm], ("all",))
log(f"  configs so far {len(results)} ({time.time()-t0:.0f}s)")

log("G2 multi-factor residual ...")
e_mf, fit_mf = residual_family("G2", ["usd", "xag", "bond", "spx"], ("all", "asia", "eu", "us"))
log(f"  configs so far {len(results)} ({time.time()-t0:.0f}s)")

log("G3 confirmation split ...")
for k in (6, 12):
    gm = roll_sum(rg, k)
    fm = roll_sum(fit_mf, k)
    z = gm / (SIG * np.sqrt(k))
    share = np.abs(fm) / np.maximum(np.abs(gm), 1e-9)      # how much of the move the factors explain
    same_dir = np.sign(fm) == np.sign(gm)
    zs = np.sign(np.nan_to_num(z)).astype(np.int8)
    for thr in (2.0, 3.0):
        fire = first_bar(np.nan_to_num(np.abs(z) > thr, nan=False).astype(bool)) & entry_ok
        idio = fire & ((share < 0.3) | ~same_dir)
        conf = fire & (share >= 0.6) & same_dir
        for H in (6, 12, 24):
            evaluate("G3", f"k{k} z>{thr} UNCONFIRMED fade", idio, (-zs).astype(np.int8), H)
            evaluate("G3", f"k{k} z>{thr} CONFIRMED follow", conf, zs, H)
            evaluate("G3", f"k{k} z>{thr} UNCONFIRMED follow", idio, zs, H)
            evaluate("G3", f"k{k} z>{thr} CONFIRMED fade", conf, (-zs).astype(np.int8), H)

log(f"\nTOTAL configs: {len(results)} ({time.time()-t0:.0f}s)")


def fmt(s):
    return (f"n={s['n']:5d} tpy={s['tpy']:4.0f} net={s['net']:+5.2f} str={s['net_stress']:+5.2f} "
            f"t={s['t']:+5.2f} WR={s['wr']:4.1f} L={s['long']:+5.2f}({s['nl']}) "
            f"S={s['short']:+5.2f}({s['ns']}) yrs+={s['yrs_pos']}")


log(f"\n=== PASSING TRAIN SELECTION (net>0, t>={T_TRAIN}, tpy>={MIN_TPY}) ===")
sel = sorted([r for r in results if r["tr"]["net"] > 0 and r["tr"]["t"] >= T_TRAIN
              and r["tr"]["tpy"] >= MIN_TPY], key=lambda r: -r["tr"]["t"])
for r in sel:
    v = "PASS" if r["te"]["net"] > 0 and r["te"]["t"] >= 2 and r["te"]["net_stress"] > 0 else "fail"
    log(f"[{v}] {r['family']} {r['cfg']} H={r['H']}\n   TRAIN {fmt(r['tr'])}\n   TEST  {fmt(r['te'])}"
        f"\n   TEST years {r['te']['by_year']}")
if not sel:
    log("  (none)")

log("\n=== TOP 20 BY TRAIN t (any frequency) ===")
for r in sorted(results, key=lambda r: -r["tr"]["t"])[:20]:
    log(f"{r['family']} {r['cfg']} H={r['H']}\n   TRAIN {fmt(r['tr'])}\n   TEST  {fmt(r['te'])}")

log("\n=== FAMILY SUMMARY (TRAIN net>0 & tpy>=150) ===")
for fam in ("G1", "G2", "G3"):
    pool = [r for r in results if r["family"] == fam and r["tr"]["net"] > 0 and r["tr"]["tpy"] >= MIN_TPY]
    log(f"{fam}: {len([r for r in results if r['family']==fam])} configs, {len(pool)} train-positive, "
        f"test-positive {sum(r['te']['net'] > 0 for r in pool)}, test t>=2 {sum(r['te']['t'] >= 2 for r in pool)}")

pd.DataFrame([{"family": r["family"], "cfg": r["cfg"], "H": r["H"],
               **{f"tr_{k}": v for k, v in r["tr"].items()},
               **{f"te_{k}": v for k, v in r["te"].items()}} for r in results]
             ).to_csv("research/xau_xasset_battery_results.csv", index=False)
log("wrote research/xau_xasset_battery_results.csv")
