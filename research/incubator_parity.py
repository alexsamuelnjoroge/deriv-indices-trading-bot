"""
Parity check: the frozen incubator rules (pro_bot/incubator_signals.py) run on the
Dukascopy history must reproduce the research batteries' numbers, and rule C without
the T-bond factor must behave like the tested 4-factor version.
"""

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pro_bot.incubator_signals import RULES, FX_USD_SIGN, align, compute_signals, session_id  # noqa: E402

COST, STRESS = 0.6, 1.2
TRAIN_END = pd.Timestamp("2023-01-01", tz="UTC")


def load(prefix):
    fs = sorted(glob.glob(f"data/duka/{prefix}_M5_*.csv"))
    parts = [p for p in (pd.read_csv(f, index_col=0, parse_dates=True) for f in fs) if len(p)]
    df = pd.concat(parts)
    df = df[~df.index.duplicated()].sort_index()
    df.index = df.index.tz_convert("UTC")
    return df


g = load("frxXAUUSD")
g = g[g["volume"] > 0]
others = {s: load(f"frx{s}")["close"] for s in FX_USD_SIGN}
others["XAGUSD"] = load("XAGUSD")["close"]
others["US500"] = load("USA500")["close"]
others["USTBOND"] = load("USTBOND")["close"]
df = align(g["close"], others)

rules = dict(RULES)
rules["C_with_bond"] = dict(RULES["C_mf_resid_eu"], factors=("usd", "xag", "bond", "spx"))
sig = compute_signals(df, rules)

ts = df.index.asi8 // 10**9
sess = session_id(ts)
lp = np.log(g["close"].to_numpy()) * 1e4
lpo = np.log(g["open"].to_numpy()) * 1e4
year = df.index.year.to_numpy()
is_train = df.index < TRAIN_END
N = len(df)


def trades(side_all, H):
    cand = np.flatnonzero(side_all)
    out_i, pos = [], 0
    while pos < len(cand):
        i = cand[pos]
        e = i + H
        if e < N and sess[e] == sess[i + 1] == sess[i] and ts[e] - ts[i] <= H * 300 + 3600:
            out_i.append(i)
            pos = np.searchsorted(cand, e, "left")
        else:
            pos += 1
    i = np.asarray(out_i)
    s = side_all[i]
    return i, s, s * (lp[i + H] - lpo[i + 1])


def summary(i, s, gr, part):
    m = part[i]
    i, s, gr = i[m], s[m], gr[m]
    net = gr - COST
    d = pd.Series(net).groupby(sess[i]).sum()
    t = d.mean() / d.std(ddof=1) * np.sqrt(len(d))
    span = (ts[i[-1]] - ts[i[0]]) / 86400 / 365.25
    yrs = pd.Series(net).groupby(year[i]).mean()
    return (f"n={len(i):4d} tpy={len(i)/span:4.0f} net={net.mean():+5.2f} str={(gr-STRESS).mean():+5.2f} "
            f"t={t:+5.2f} WR={(net>0).mean()*100:4.1f} L={gr[s>0].mean()-COST:+5.2f} S={gr[s<0].mean()-COST:+5.2f}"
            f" | yrs {' '.join(f'{y%100:02d}:{v:+.1f}' for y, v in yrs.items())}")


for name, r in rules.items():
    i, s, gr = trades(sig[f"{name}_side"].to_numpy(), r["H"])
    print(name, r)
    print("   TRAIN", summary(i, s, gr, is_train))
    print("   TEST ", summary(i, s, gr, ~is_train))
