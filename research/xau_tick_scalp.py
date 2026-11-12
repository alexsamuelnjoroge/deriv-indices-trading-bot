"""
XAUUSD tick-level scalping study (signal list pre-registered 2026-11-10; run once ~3-4 weeks
of ticks exist). Data: data/ticks/<SYMBOL>/<UTC date>.csv.gz from pro_bot/tick_recorder.py.

Execution model (realistic for a home PC on IC Markets MT5):
  signal at the end of second t, filled on the first quote of second t+LATENCY (default 1s);
  buys pay the ask, sells hit the bid, exits cross the spread again; commission $7/lot RT.
  No position held 20:30-22:30 UTC. One position per config at a time.

Signals (follow and fade of each; hold H in {15, 60, 300} s):
  IMB   tick imbalance over N s (N 5/15/60): (upticks-downticks)/(upticks+downticks) > 0.6/0.8,
        with >= 10 mid changes in the window
  RET   N-second mid move (N 5/15/60) > 3/4 sigma of its trailing-1h distribution
  BURST quote rate over 10 s > 3 sigma of the trailing 30 min, direction = last 10 s move
  XAG   silver N-second move (N 5/15) > 3 sigma while gold's is < 1 sigma -> follow silver
Split: first 60% of days select, last 40% judge (chronological). Day-clustered t-stats.
Pass: TRAIN net mean > 0 with t >= 3 (~130 configs), TEST net mean > 0 with t >= 2.

    python research/xau_tick_scalp.py [--latency 1] [--min-days 10]
"""

import argparse
import glob
import math
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--latency", type=int, default=1, help="seconds from signal to fill")
ap.add_argument("--min-days", type=int, default=10)
args = ap.parse_args()

COMMISSION_PER_OZ = 0.07          # $7 per 100-oz lot round trip


def load_ticks(sym):
    files = sorted(glob.glob(f"data/ticks/{sym}/*.csv.gz"))
    if not files:
        return None
    df = pd.concat(pd.read_csv(f) for f in files)
    df = df[(df.bid > 0) & (df.ask >= df.bid)].sort_values("utc_msc")
    return df


def to_seconds(df):
    """1-second grid (last quote of each second, forward-filled within 60 s) + per-second counts."""
    sec = df.utc_msc // 1000
    mid = (df.bid + df.ask) / 2
    dm = np.sign(mid.diff().fillna(0))
    g = pd.DataFrame({"sec": sec, "bid": df.bid, "ask": df.ask, "mid": mid,
                      "up": (dm > 0).astype(int), "dn": (dm < 0).astype(int)})
    agg = g.groupby("sec").agg(bid=("bid", "last"), ask=("ask", "last"), mid=("mid", "last"),
                               n=("mid", "size"), up=("up", "sum"), dn=("dn", "sum"))
    full = pd.RangeIndex(agg.index.min(), agg.index.max() + 1)
    agg = agg.reindex(full)
    agg[["n", "up", "dn"]] = agg[["n", "up", "dn"]].fillna(0)
    agg[["bid", "ask", "mid"]] = agg[["bid", "ask", "mid"]].ffill(limit=60)
    return agg.dropna(subset=["mid"])


gold = load_ticks("XAUUSD")
if gold is None:
    raise SystemExit("no XAUUSD ticks in data/ticks/ yet — is pro_bot.tick_recorder running?")
s = to_seconds(gold)
ts = pd.to_datetime(s.index, unit="s", utc=True)
days = ts.normalize()
n_days = days.nunique()
print(f"XAUUSD ticks {len(gold):,} | seconds with quotes {len(s):,} | days {n_days} "
      f"({ts[0]:%Y-%m-%d %H:%M} -> {ts[-1]:%Y-%m-%d %H:%M} UTC)")

# ── diagnostics ──────────────────────────────────────────────────────────────
spread_bp = (s.ask - s.bid) / s.mid * 1e4
print("\n=== Spread by UTC hour (median $ | median bp | quotes/s) ===")
hr = ts.hour
for h in range(24):
    m = hr == h
    if m.any():
        print(f"  {h:02d}: ${np.median((s.ask - s.bid)[m]):.3f} | {np.median(spread_bp[m]):.2f}bp | "
              f"{s.n[m].mean():.1f}/s")
lm = np.log(s.mid.to_numpy()) * 1e4
print("\n=== Do short moves reverse or continue? corr(past N s move, next N s move) ===")
for N in (1, 5, 15, 60, 300):
    r = lm[N:] - lm[:-N]
    a, b = r[:-N:N], r[N::N]
    print(f"  N={N:4d}s  corr={np.corrcoef(a, b)[0, 1]:+.3f}  (n={len(a):,}; negative = reversal)")
print(f"\nCost of a round trip at the median spread: {np.median(spread_bp):.2f}bp + commission "
      f"{COMMISSION_PER_OZ / s.mid.median() * 1e4:.2f}bp")

# ── features (all computed from data at or before second t) ──────────────────
mod = ts.hour * 60 + ts.minute
tradable = ~((mod >= 20 * 60 + 30) & (mod < 22 * 60 + 30))
lmS = pd.Series(lm, index=s.index)


def roll(x, w):
    return x.rolling(w, min_periods=1).sum()


feat = {}
for N in (5, 15, 60):
    up, dn = roll(s.up, N), roll(s.dn, N)
    feat[f"imb{N}"] = ((up - dn) / (up + dn).replace(0, np.nan)).where(up + dn >= 10)
    ret = lmS - lmS.shift(N)
    feat[f"z{N}"] = ret / ret.rolling(3600, min_periods=600).std()
rate = roll(s.n, 10)
feat["burst"] = (rate - rate.rolling(1800, min_periods=600).mean()) / rate.rolling(1800, min_periods=600).std()
feat["dir10"] = np.sign(lmS - lmS.shift(10))

xag = load_ticks("XAGUSD")
if xag is not None:
    sx = to_seconds(xag).reindex(s.index).ffill(limit=60)
    lx = np.log(sx.mid) * 1e4
    for N in (5, 15):
        rx = lx - lx.shift(N)
        feat[f"xz{N}"] = rx / rx.rolling(3600, min_periods=600).std()

bid, ask, mid = s.bid.to_numpy(), s.ask.to_numpy(), s.mid.to_numpy()
sec_day = days.to_numpy()
ok_t = tradable.to_numpy() if hasattr(tradable, "to_numpy") else np.asarray(tradable)


def simulate(cand, side, H):
    idx = np.flatnonzero(cand)
    out, nxt = [], -1
    L = args.latency
    for i in idx:
        if i < nxt:
            continue
        e, x = i + L, i + L + H
        if x >= len(mid) or not ok_t[i] or not ok_t[x] or s.index[x] - s.index[i] != L + H:
            continue
        sd = side[i]
        entry = ask[e] if sd > 0 else bid[e]
        exitp = bid[x] if sd > 0 else ask[x]
        net = sd * (exitp - entry) / entry * 1e4 - COMMISSION_PER_OZ / entry * 1e4
        out.append((i, net))
        nxt = x
    return out


def stats(tr):
    if len(tr) < 30:
        return None
    net = np.array([n for _, n in tr])
    d = pd.Series(net).groupby(sec_day[[i for i, _ in tr]]).sum()
    t = d.mean() / d.std(ddof=1) * math.sqrt(len(d)) if len(d) > 1 and d.std(ddof=1) > 0 else 0
    return dict(n=len(net), per_day=len(net) / max(len(d), 1), mean=net.mean(), t=t,
                wr=(net > 0).mean() * 100)


configs = []
for N in (5, 15, 60):
    imb = feat[f"imb{N}"].to_numpy()
    for thr in (0.6, 0.8):
        c = np.nan_to_num(np.abs(imb)) > thr
        configs.append((f"IMB{N} >{thr}", c, np.sign(np.nan_to_num(imb))))
    z = feat[f"z{N}"].to_numpy()
    for thr in (3, 4):
        c = np.nan_to_num(np.abs(z)) > thr
        configs.append((f"RET{N} >{thr}sd", c, np.sign(np.nan_to_num(z))))
b = feat["burst"].to_numpy()
configs.append(("BURST >3sd", np.nan_to_num(b) > 3, feat["dir10"].fillna(0).to_numpy()))
for N in (5, 15):
    if f"xz{N}" in feat:
        xz, gz = feat[f"xz{N}"].to_numpy(), feat[f"z{N}"].to_numpy()
        c = (np.nan_to_num(np.abs(xz)) > 3) & (np.nan_to_num(np.abs(gz), nan=9) < 1)
        configs.append((f"XAG{N} lead", c, np.sign(np.nan_to_num(xz))))

udays = np.unique(sec_day)
cut = udays[int(len(udays) * 0.6)] if len(udays) > 1 else udays[0]
print(f"\n=== Signals (TRAIN days < {pd.Timestamp(cut):%Y-%m-%d} <= TEST) | latency {args.latency}s ===")
if n_days < args.min_days:
    print(f"!! only {n_days} day(s) of ticks — results below are a smoke test, NOT a verdict "
          f"(need >= {args.min_days})")
rows = []
for name, cand, direction in configs:
    for mode, sg in (("follow", 1), ("fade", -1)):
        if name.startswith("XAG") and mode == "fade":
            continue
        side = (sg * direction).astype(int)
        for H in (15, 60, 300):
            tr = simulate(cand & (side != 0), side, H)
            a = stats([x for x in tr if sec_day[x[0]] < cut])
            bb = stats([x for x in tr if sec_day[x[0]] >= cut])
            rows.append((f"{name} {mode} H{H}s", a, bb))


def fmt(x):
    return "n<30" if not x else f"n={x['n']:5d} {x['per_day']:5.0f}/day net={x['mean']:+6.2f}bp t={x['t']:+5.2f} WR={x['wr']:4.1f}%"


for name, a, bb in sorted(rows, key=lambda r: -(r[1]["t"] if r[1] else -99)):
    verdict = ("PASS" if a and bb and a["mean"] > 0 and a["t"] >= 3 and bb["mean"] > 0 and bb["t"] >= 2
               else "fail")
    print(f"[{verdict}] {name:<26} TRAIN {fmt(a):<52} TEST {fmt(bb)}")

# ── real execution from the incubator ────────────────────────────────────────
inc = Path("logs/incubator_trades.csv")
if inc.exists():
    t = pd.read_csv(inc)
    if len(t):
        t["slip_s"] = (pd.to_datetime(t.entry_time_utc) - pd.to_datetime(t.signal_bar_utc)).dt.total_seconds() - 300
        print(f"\n=== Incubator execution: {len(t)} trades, entry delay after bar close "
              f"median {t.slip_s.median():.1f}s, max {t.slip_s.max():.1f}s ===")
