"""
Robustness checks for the session-breakout variant that passed (03:00-08:00 server range,
OCO stop orders, SL = opposite side, TP = 1R). Post-hoc inspection, nothing is re-selected:
  1. year by year in R units (removes the volatility-level effect of bp)
  2. long vs short
  3. extra slippage per trade (stop orders slip in fast markets)
  4. neighbouring range windows and TP multiples (is the winner an isolated spike?)
  5. placebo: coin-flip direction at range end with the same bracket size
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xau_session_breakout_ea import COST, TEST_YEAR, placebo, run  # noqa: E402


def summ(tr, extra=0.0):
    net = np.array([x[2] for x in tr]) - COST - extra
    r = net / np.array([x[3] for x in tr])
    t = net.mean() / net.std(ddof=1) * np.sqrt(len(net))
    return f"n={len(net):4d} {net.mean():+5.2f}bp t={t:+5.2f} | {r.mean():+.3f}R"


base = run(3, 8, "a")
tr = [x for x in base if x[0].year < TEST_YEAR]
te = [x for x in base if x[0].year >= TEST_YEAR]

print("1) Year by year (net of 0.36bp): bp | R | median risk bp")
for y in sorted({x[0].year for x in base}):
    ys = [x for x in base if x[0].year == y]
    net = np.array([x[2] for x in ys]) - COST
    risk = np.array([x[3] for x in ys])
    print(f"   {y}: {net.mean():+6.2f}bp | {np.mean(net / risk):+.3f}R | risk {np.median(risk):5.1f}bp | n={len(ys)}")

print("\n2) Long vs short")
for side, lab in ((1, "long "), (-1, "short")):
    print(f"   {lab} TRAIN {summ([x for x in tr if x[1] == side])}   TEST {summ([x for x in te if x[1] == side])}")

print("\n3) Extra slippage per round trip")
for extra in (0.0, 1.0, 2.0, 3.0, 5.0):
    print(f"   +{extra:.0f}bp  TRAIN {summ(tr, extra)}   TEST {summ(te, extra)}")

print("\n4) Neighbouring windows (SL opposite, TP 1R) and TP multiples (03-08)")
for r0, r1 in ((2, 7), (2, 8), (3, 7), (3, 8), (3, 9), (4, 8), (4, 9), (5, 9)):
    x = run(r0, r1, "a")
    print(f"   {r0:02d}-{r1:02d}  TRAIN {summ([v for v in x if v[0].year < TEST_YEAR])}   "
          f"TEST {summ([v for v in x if v[0].year >= TEST_YEAR])}")
for m in (0.5, 0.75, 1.0, 1.5, 2.0):
    x = run(3, 8, "a", tp_r=m)
    print(f"   TP {m:.2f}R TRAIN {summ([v for v in x if v[0].year < TEST_YEAR])}   "
          f"TEST {summ([v for v in x if v[0].year >= TEST_YEAR])}")

print("\n5) Placebo: random direction at 08:00, same bracket (5 seeds)")
for seed in range(5):
    x = placebo(3, 8, seed)
    print(f"   seed {seed}  TRAIN {summ([v for v in x if v[0].year < TEST_YEAR])}   "
          f"TEST {summ([v for v in x if v[0].year >= TEST_YEAR])}")
