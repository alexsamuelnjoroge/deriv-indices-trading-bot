"""
XAUUSD stacked improvement test.

New baseline: EMA200 ADX0 RSI-adaptive(lb50) PEAK MACRO20 ATR1.5 RR2.0 + 4H EMA20
Tests stacking ATR x2.0 and RSI lookback 30 on top of that baseline.

Usage:
  python pro_bot/backtest_xauusd_stack.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Reuse everything from the existing sweep module
from pro_bot.backtest_xauusd import load, split_all, run, stats, SYM, SPREAD
from pro_bot.backtest import simulate_exits

BASE = dict(
    ema_period=200, slope_bars=3, rsi_period=14, rsi_entry=35.0,
    tp_rr=2.0, atr_mult_sl=1.5, macro_ema_period=20,
    adaptive=True, rsi_lookback=50, direction="both",
    use_4h_filter=True, ema_4h_period=20,
)


def test(label, cfg, train, hold):
    sigs_tr = run(train["5m"], train["1h"], train["4h"], train["1d"], cfg)
    sigs_ho = run(hold["5m"],  hold["1h"],  hold["4h"],  hold["1d"],  cfg)
    tr = stats(simulate_exits(train["5m"], sigs_tr, spread=SPREAD, be_r=1.0), min_n=20)
    ho = stats(simulate_exits(hold["5m"],  sigs_ho, spread=SPREAD, be_r=1.0), min_n=10)

    def fmt(s, tag=""):
        if s is None:
            return "  (too few trades)"
        verdict = "STRONG" if s["ev"] > 0.05 else "OK" if s["ev"] > 0 else "NEG"
        return (f"  n={s['n']:>3}  WR:{s['wr']*100:>5.1f}%  "
                f"EV:{s['ev']:>+.4f}R  Net:{s['net_r']:>+5.1f}R  "
                f"B:{s['buys']}({s['buy_wr']*100:.0f}%) "
                f"S:{s['sells']}({s['sel_wr']*100:.0f}%)  [{verdict}]{tag}")

    ho_good = ho is not None and ho["ev"] > 0
    print(f"\n  {label}")
    print(f"    Train  :{fmt(tr)}")
    print(f"    Holdout:{fmt(ho, '  <<<' if ho_good else '')}")

    return tr, ho


async def main():
    print("=" * 78)
    print("XAUUSD — Stack test: 4H EMA20 baseline + ATR×2.0 + RSI lookback 30")
    print(f"Spread={SPREAD}  BE@1R  80/20 holdout  365 days")
    print("=" * 78)

    print("\n  Loading data...")
    b5, b1h, b4h, b1d = await load()
    train, hold = split_all(b5, b1h, b4h, b1d)
    print(f"  Train {len(train['5m'])} bars | Hold {len(hold['5m'])} bars\n")

    results = []

    def record(label, cfg, train, hold):
        tr, ho = test(label, cfg, train, hold)
        if tr and ho and tr["ev"] > 0 and ho["ev"] > 0:
            results.append((label, cfg, tr, ho))

    # ── Baseline (current live config) ────────────────────────────────────────
    print("── Baseline ────────────────────────────────────────────────────────")
    record("4H EMA20  ATR×1.5  lookback=50  (CURRENT)", BASE, train, hold)

    # ── ATR×2.0 alone ─────────────────────────────────────────────────────────
    print("\n── ATR×2.0 ─────────────────────────────────────────────────────────")
    record("4H EMA20  ATR×2.0  lookback=50",
           {**BASE, "atr_mult_sl": 2.0}, train, hold)

    # ── RSI lookback 30 alone ─────────────────────────────────────────────────
    print("\n── RSI lookback 30 ─────────────────────────────────────────────────")
    record("4H EMA20  ATR×1.5  lookback=30",
           {**BASE, "rsi_lookback": 30}, train, hold)

    # ── Both stacked ──────────────────────────────────────────────────────────
    print("\n── Both stacked ────────────────────────────────────────────────────")
    record("4H EMA20  ATR×2.0  lookback=30",
           {**BASE, "atr_mult_sl": 2.0, "rsi_lookback": 30}, train, hold)

    # ── Stretch: also widen RR ─────────────────────────────────────────────────
    print("\n── Stretch: stack + wider RR ───────────────────────────────────────")
    record("4H EMA20  ATR×2.0  lookback=30  RR=2.5",
           {**BASE, "atr_mult_sl": 2.0, "rsi_lookback": 30, "tp_rr": 2.5}, train, hold)
    record("4H EMA20  ATR×2.0  lookback=30  RR=3.0",
           {**BASE, "atr_mult_sl": 2.0, "rsi_lookback": 30, "tp_rr": 3.0}, train, hold)
    record("4H EMA20  ATR×1.5  lookback=30  RR=2.5",
           {**BASE, "rsi_lookback": 30, "tp_rr": 2.5}, train, hold)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print("SUMMARY — sorted by holdout EV\n")

    for label, cfg, tr, ho in sorted(results, key=lambda x: -x[3]["ev"]):
        changed = {k: v for k, v in cfg.items() if BASE.get(k) != v}
        mark = "  ★ NEW BEST" if cfg != BASE and ho["ev"] > results[0][3]["ev"] else ""
        print(f"  {label}{mark}")
        print(f"    Changes  : {changed if changed else 'none (baseline)'}")
        print(f"    Train    : EV {tr['ev']:+.4f}R  WR {tr['wr']*100:.1f}%  n={tr['n']}")
        print(f"    Holdout  : EV {ho['ev']:+.4f}R  WR {ho['wr']*100:.1f}%  n={ho['n']}  Net {ho['net_r']:+.1f}R")
        print()

    if results:
        best = max(results, key=lambda x: x[3]["ev"])
        if best[1] != BASE:
            print(f"  Recommendation: update config.yaml with changes from '{best[0]}'")
            print(f"  Holdout EV improvement: "
                  f"{results[0][3]['ev']:+.4f}R → {best[3]['ev']:+.4f}R")
        else:
            print("  Current config is already optimal — no stacking improvement found.")


if __name__ == "__main__":
    asyncio.run(main())
