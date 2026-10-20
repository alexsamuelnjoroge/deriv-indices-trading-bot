"""
Jump-index last-digit bias on ticks collected by tick_recorder.py.

Question: is digit 3 (and 7) rare enough on JD indices for DIGITDIFF to beat
Deriv's payout? Differs breaks even when the digit appears <= 1 - 1/payout of
the time (8.88% at 1.0975x — the best ratio, at a $4 stake — or 8.76% at 1.096x).

Usage:
    python analyze_jd_digits.py [--payout 1.0975]
"""

import argparse
import glob
from collections import defaultdict

import numpy as np


def load(sym: str) -> tuple[np.ndarray, list[str]]:
    digits, days = [], []
    for f in sorted(glob.glob(f"data/ticks_live/{sym}_*.csv")):
        with open(f) as fh:
            next(fh)
            d = [int(line.split(",")[1].strip()[-1]) for line in fh if line.count(",") >= 1]
        digits.append(np.array(d))
        days.append(f[-14:-4])
    return digits, days


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--payout", type=float, default=1.0975)
    ap.add_argument("--symbols", nargs="+", default=["JD10", "JD25", "JD50", "JD75"])
    a = ap.parse_args()
    be = 1 - 1 / a.payout
    print(f"Differs break-even: digit frequency <= {be*100:.2f}% at payout {a.payout}x\n")
    for sym in a.symbols:
        per_day, days = load(sym)
        if not per_day:
            print(f"{sym}: no recorded ticks yet")
            continue
        allx = np.concatenate(per_day)
        tot = np.bincount(allx, minlength=10) / len(allx) * 100
        print(f"{sym}: {len(allx):,} ticks over {len(days)} day(s)")
        print("  ALL     " + " ".join(f"{k}:{v:5.2f}" for k, v in enumerate(tot)))
        for d, x in zip(days, per_day):
            f = np.bincount(x, minlength=10) / max(len(x), 1) * 100
            print(f"  {d[5:]}  " + " ".join(f"{k}:{v:5.2f}" for k, v in enumerate(f)) + f"  n={len(x)}")
        for k in range(10):
            pw = 1 - (allx == k).mean()
            se = np.sqrt(pw * (1 - pw) / len(allx))
            ev = pw * a.payout - 1
            z = ev / (se * a.payout)
            days_below = sum((x == k).mean() < be for x in per_day if len(x))
            if ev > -0.003:
                print(f"  DIFF {k}: P(win) {pw*100:.3f}%  EV {ev*100:+.3f}c per $1  z={z:+.1f}  "
                      f"days with digit below break-even: {days_below}/{len(per_day)}")
        print()


if __name__ == "__main__":
    main()
