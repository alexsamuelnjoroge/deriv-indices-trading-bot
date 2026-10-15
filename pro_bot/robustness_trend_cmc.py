"""
Robustness checks for Donchian trend-following on XAU, XAG, BTC, ETH.

1. Parameter grid: entry lookback x ATR stop mult x stop cap -- is the edge
   broad, or does it only exist at a few settings?
2. Stress tests on the classic 55/20 config: leave-one-symbol-out, drop the
   best trades, drop the crypto bull years, and risk-per-trade vs drawdown.

Crypto positions roll weekly (Deriv crypto multipliers expire ~7.6 days).

Usage:
    python pro_bot/robustness_trend_cmc.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from pro_bot.backtest_trend_portfolio import UNIVERSE, ROLL_DAYS, load, simulate

SYMS = ["frxXAUUSD", "frxXAGUSD", "cryBTCUSD", "cryETHUSD"]
T0 = int(pd.Timestamp("2013-01-01").timestamp())
T1 = int(time.time())
PERIODS = np.linspace(T0, T1, 5)

ENTRIES = [20, 30, 40, 55, 70, 100, 150]
ATR_MULTS = [1.5, 2.0, 3.0]
CAPS = [0.005, 0.0075, 0.0095, None]   # None = uncapped (normal broker)


def equity(trs, risk=0.005, t0=T0, t1=T1):
    trs = sorted([t for t in trs if t0 <= t[1] < t1], key=lambda x: x[1])
    if not trs:
        return dict(n=0, ev=0, cagr=0, mdd=0, net=0)
    rs = np.array([t[2] for t in trs])
    eq, peak, mdd = 1.0, 1.0, 0.0
    for r in rs:
        eq *= max(1 + risk * r, 0.0)
        peak = max(peak, eq)
        mdd = max(mdd, 1 - eq / peak if peak else 1)
    yrs = (t1 - t0) / (365.25 * 86400)
    return dict(n=len(rs), ev=rs.mean(), net=rs.sum(), cagr=eq ** (1 / yrs) - 1 if eq > 0 else -1, mdd=mdd)


def run(data, entry, exit_, atr_mult, cap):
    prm = dict(entry=entry, exit=exit_, atr_mult=atr_mult)
    per = {}
    for s in SYMS:
        kw = dict(cap_on=cap is not None, roll_days=ROLL_DAYS.get(s))
        if cap is not None:
            kw["cap_frac"] = cap
        per[s] = [t for t in simulate(data[s], "donchian", prm, UNIVERSE[s][1], **kw) if t[0] >= T0]
    return per


def flat(per, skip=()):
    return [t for s, tr in per.items() if s not in skip for t in tr]


def main():
    data = {s: load(s, UNIVERSE[s][0]) for s in SYMS}

    print("\n" + "=" * 104)
    print("  1. PARAMETER GRID  (exit = entry/2)   cells: CAGR / maxDD / periods+ / symbols+   @0.5% risk")
    print("=" * 104)
    grid = []
    for cap in CAPS:
        cap_lbl = f"cap {cap*100:.2f}%" if cap else "uncapped"
        print(f"\n  {cap_lbl}")
        print("  " + "entry".ljust(7) + "".join(f"ATRx{m:<27}" for m in ATR_MULTS))
        for e in ENTRIES:
            cells = []
            for m in ATR_MULTS:
                per = run(data, e, e // 2, m, cap)
                st = equity(flat(per))
                pp = sum(equity(flat(per), t0=PERIODS[k], t1=PERIODS[k + 1])["ev"] > 0 for k in range(4))
                sp = sum(sum(t[2] for t in tr) > 0 for tr in per.values())
                grid.append((cap, e, m, st, pp, sp))
                cells.append(f"{st['cagr']*100:+6.1f}% DD{st['mdd']*100:3.0f}% {pp}/4 {sp}/4".ljust(31))
            print(f"  {e:<7}" + "".join(cells))

    print("\n  Grid summary:")
    for cap in CAPS:
        g = [x for x in grid if x[0] == cap]
        pos = sum(x[3]["cagr"] > 0 for x in g)
        allp = sum(x[4] == 4 for x in g)
        med = np.median([x[3]["cagr"] for x in g]) * 100
        lbl = f"cap {cap*100:.2f}%" if cap else "uncapped"
        print(f"    {lbl:<12} {pos}/{len(g)} configs profitable   {allp}/{len(g)} positive in all 4 periods   "
              f"median CAGR {med:+.1f}%")

    print("\n" + "=" * 104)
    print("  2. STRESS TESTS  -- Donchian 55/20, ATRx2, cap 0.95% (Deriv x100)")
    print("=" * 104)
    per = run(data, 55, 20, 2.0, 0.0095)
    base = equity(flat(per))
    print(f"  Baseline:                    n={base['n']}  EV {base['ev']:+.3f}R  CAGR {base['cagr']*100:+.1f}%  "
          f"maxDD {base['mdd']*100:.1f}%")

    print("\n  Leave one symbol out:")
    for s in SYMS:
        st = equity(flat(per, skip=(s,)))
        print(f"    without {s:<10}  CAGR {st['cagr']*100:+6.1f}%  maxDD {st['mdd']*100:5.1f}%  EV {st['ev']:+.3f}R")
    for lbl, skip in (("metals only", ("cryBTCUSD", "cryETHUSD")), ("crypto only", ("frxXAUUSD", "frxXAGUSD"))):
        st = equity(flat(per, skip=skip))
        print(f"    {lbl:<18}  CAGR {st['cagr']*100:+6.1f}%  maxDD {st['mdd']*100:5.1f}%  EV {st['ev']:+.3f}R")

    print("\n  Remove the best trades (luck dependence):")
    allt = flat(per)
    ranked = sorted(allt, key=lambda t: -t[2])
    for k in (1, 5, 10, 20):
        st = equity(ranked[k:])
        print(f"    drop top {k:<3} trades       CAGR {st['cagr']*100:+6.1f}%  EV {st['ev']:+.3f}R  "
              f"(top {k} = {sum(t[2] for t in ranked[:k]):.0f}R of {base['net']:.0f}R)")

    print("\n  Drop crypto bull years (2017, 2020-21):")
    bull = [(pd.Timestamp("2017-01-01"), pd.Timestamp("2018-01-01")),
            (pd.Timestamp("2020-07-01"), pd.Timestamp("2021-12-31"))]
    def in_bull(t):
        return any(a.timestamp() <= t[1] < b.timestamp() for a, b in bull)
    st = equity([t for t in allt if not in_bull(t)])
    print(f"    ex-bull                   CAGR {st['cagr']*100:+6.1f}%  maxDD {st['mdd']*100:5.1f}%  EV {st['ev']:+.3f}R")

    print("\n  Last 3 years only (2023-10 -> now):")
    t3 = int(pd.Timestamp("2023-10-01").timestamp())
    for lbl, skip in (("all 4", ()), ("metals", ("cryBTCUSD", "cryETHUSD")), ("crypto", ("frxXAUUSD", "frxXAGUSD"))):
        st = equity(flat(per, skip=skip), t0=t3)
        print(f"    {lbl:<8} n={st['n']:<4} CAGR {st['cagr']*100:+6.1f}%  maxDD {st['mdd']*100:5.1f}%  EV {st['ev']:+.3f}R")

    print("\n  Risk per trade vs outcome:")
    for risk in (0.0025, 0.005, 0.01):
        st = equity(allt, risk=risk)
        print(f"    {risk*100:.2f}% risk           CAGR {st['cagr']*100:+6.1f}%  maxDD {st['mdd']*100:5.1f}%")

    lose_streak = cur = 0
    for t in sorted(allt, key=lambda x: x[1]):
        cur = cur + 1 if t[2] <= 0 else 0
        lose_streak = max(lose_streak, cur)
    wr = np.mean([t[2] > 0 for t in allt])
    print(f"\n  Win rate {wr*100:.1f}%   longest losing streak {lose_streak} trades")
    print()


if __name__ == "__main__":
    main()
