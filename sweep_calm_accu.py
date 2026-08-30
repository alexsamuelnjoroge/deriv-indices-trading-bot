"""
Calm-Period ACCU sweep — inter-spike accumulator entries with barrier-relative calm filter.

Grid:
  spike_cooldown     50, 100, 200   — ticks to avoid after each spike
  calm_atr_ratio     0.7, 1.0, 1.5  — short_atr must be < ratio x long_atr
  calm_barrier_mult  0.0, 0.2, 0.3, 0.5  — max tick as multiple of barrier (0=off)
  calm_lookback      5, 10, 20       — ticks to inspect for barrier-relative filter
  hold_ticks         symbol-specific range
  growth_rate        0.04, 0.05

entry_cooldown is fixed at hold_ticks to avoid stacking entries.
short_atr_period=10, long_atr_period=50, spike_mult=15 (fixed).

Symbols: CRASH500, BOOM150N, CRASH1000, BOOM1000

Usage:
  python sweep_calm_accu.py
  python sweep_calm_accu.py --symbol CRASH500
"""

import argparse
import sys

from loguru import logger

from src.backtest.engine import BacktestEngine
from src.data.history import fetch_ticks
from src.strategies.calm_accu import CalmAccuStrategy

logger.remove()
logger.add(sys.stderr, level="ERROR", format="{time:HH:mm:ss} | {level} | {message}")

# Barriers from check_contracts.py (2026-08-26) at 4% growth rate
SYMBOLS = {
    "CRASH500":  {"symbol_type": "crash", "barrier_pct": 4.72e-6, "hold_range": [10, 15, 20]},
    "BOOM150N":  {"symbol_type": "boom",  "barrier_pct": 1.61e-6, "hold_range": [8, 10, 15]},
    "CRASH1000": {"symbol_type": "crash", "barrier_pct": 2.27e-6, "hold_range": [8, 12, 15]},
    "BOOM1000":  {"symbol_type": "boom",  "barrier_pct": 2.25e-6, "hold_range": [8, 12, 15]},
}

SPIKE_COOLDOWNS    = [50, 100, 200]
CALM_ATR_RATIOS    = [0.7, 1.0, 1.5]
CALM_BARRIER_MULTS = [0.0, 0.2, 0.3, 0.5]   # 0.0 = barrier filter off
CALM_LOOKBACKS     = [5, 10, 20]
GROWTH_RATES       = [0.04, 0.05]

SHORT_ATR    = 10
LONG_ATR     = 50
SPIKE_MULT   = 15.0
LOSS_COOLDOWN = 2

WINDOWS     = 4
WINDOW_SIZE = 21_500   # 4 x 21.5k = 86k ticks

RISK_BASE = {
    "stake_percent":      2.0,
    "max_stake":          20.0,
    "min_stake":          1.00,
    "daily_loss_limit":   100.0,
    "use_kelly":          False,
    "max_open_contracts": 1,
}

MIN_TRADES = 6
SEP = "=" * 130


def payout(gr, ht):
    return (1 + gr) ** ht - 1


def be(gr, ht):
    return 1.0 / (1.0 + payout(gr, ht))


def run_combo(ticks, meta, sc, ratio, cbm, cl, ht, gr):
    pay = payout(gr, ht)
    brk = be(gr, ht)

    strategy_cfg = {
        "symbol_type":      meta["symbol_type"],
        "long_atr_period":  LONG_ATR,
        "short_atr_period": SHORT_ATR,
        "spike_mult":       SPIKE_MULT,
        "spike_cooldown":   sc,
        "calm_atr_ratio":   ratio,
        "calm_barrier_mult": cbm,
        "calm_lookback":    cl,
        "entry_cooldown":   max(ht, 5),
        "loss_cooldown":    LOSS_COOLDOWN,
        "hold_ticks":       ht,
        "growth_rate":      gr,
        "barrier_pct":      meta["barrier_pct"],
    }
    risk_cfg = {
        **RISK_BASE,
        "payout_pct":  pay,
        "barrier_pct": meta["barrier_pct"],
    }

    wins = losses = trades = passes = 0
    for w in range(WINDOWS):
        seg = ticks[w * WINDOW_SIZE: (w + 1) * WINDOW_SIZE]
        if len(seg) < 100:
            continue
        engine = BacktestEngine(
            strategy_cfg=strategy_cfg,
            risk_cfg=risk_cfg,
            payout_pct=pay,
            strategy_class=CalmAccuStrategy,
        )
        r = engine.run(seg, starting_balance=1000.0)
        if r.total_trades >= 3 and r.win_rate >= brk * 100:
            passes += 1
        wins   += r.wins
        losses += r.losses
        trades += r.total_trades

    wr = wins / trades * 100 if trades > 0 else 0.0
    ev = (wr / 100 - brk) * pay * 100
    return {
        "sc": sc, "ratio": ratio, "cbm": cbm, "cl": cl, "ht": ht, "gr": gr,
        "pay": pay * 100, "be": brk * 100,
        "wr": wr, "ev": ev, "trades": trades, "passes": passes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="ALL")
    args = parser.parse_args()

    symbols = list(SYMBOLS.keys()) if args.symbol.upper() == "ALL" else [args.symbol.upper()]
    total   = WINDOWS * WINDOW_SIZE

    for symbol in symbols:
        meta = SYMBOLS[symbol]

        print()
        print(SEP)
        print(f"  {symbol}  |  Calm-Period ACCU Sweep (barrier-relative filter)  |  {WINDOWS}x{WINDOW_SIZE:,} ticks")
        print(f"  Fixed: spike_mult={SPIKE_MULT:.0f}x  short_atr={SHORT_ATR}  long_atr={LONG_ATR}")
        print(f"  Sweeping: spike_cooldown  calm_atr_ratio  calm_barrier_mult  calm_lookback  hold_ticks  growth_rate")
        print(SEP)

        ticks = fetch_ticks(symbol, count=total + 5_000)
        ticks = ticks[-total:]
        print(f"  Using {len(ticks):,} ticks\n")

        results = []
        hold_ticks_range = meta["hold_range"]
        total_combos = (len(SPIKE_COOLDOWNS) * len(CALM_ATR_RATIOS) *
                        len(CALM_BARRIER_MULTS) * len(CALM_LOOKBACKS) *
                        len(hold_ticks_range) * len(GROWTH_RATES))
        done = 0

        for sc in SPIKE_COOLDOWNS:
            for ratio in CALM_ATR_RATIOS:
                for cbm in CALM_BARRIER_MULTS:
                    for cl in CALM_LOOKBACKS:
                        for ht in hold_ticks_range:
                            for gr in GROWTH_RATES:
                                r = run_combo(ticks, meta, sc, ratio, cbm, cl, ht, gr)
                                results.append(r)
                                done += 1
                                if done % 20 == 0:
                                    print(f"  ... {done}/{total_combos}")

        results.sort(key=lambda x: (x["passes"], x["ev"]), reverse=True)

        print(f"\n  {'sc':>4}  {'ratio':>5}  {'cbm':>4}  {'cl':>3}  {'ht':>3}  {'gr':>4}  "
              f"{'pay%':>6}  {'BE%':>6}  {'WR%':>6}  {'EV%':>8}  {'trades':>6}  pass")
        print(f"  {'-'*4}  {'-'*5}  {'-'*4}  {'-'*3}  {'-'*3}  {'-'*4}  "
              f"{'-'*6}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*6}  ----")
        shown = 0
        for r in results:
            if r["trades"] < MIN_TRADES:
                continue
            flag = " ***" if r["passes"] >= 4 and r["ev"] > 0 else (
                   " **"  if r["passes"] == 3 and r["ev"] > 0 else (
                   " *"   if r["passes"] == 2 and r["ev"] > 0 else ""))
            print(f"  {r['sc']:>4}  {r['ratio']:>5.1f}  {r['cbm']:>4.1f}  {r['cl']:>3}  "
                  f"{r['ht']:>3}  {r['gr']*100:>3.0f}%  "
                  f"{r['pay']:>6.1f}%  {r['be']:>6.1f}%  {r['wr']:>6.1f}%  "
                  f"{r['ev']:>+8.3f}%  {r['trades']:>6}  {r['passes']}/{WINDOWS}{flag}")
            shown += 1
            if shown >= 40:
                remaining = sum(1 for x in results if x["trades"] >= MIN_TRADES) - shown
                if remaining > 0:
                    print(f"  ... ({remaining} more rows not shown)")
                break

        valid = [r for r in results if r["trades"] >= MIN_TRADES]
        if valid:
            best = valid[0]
            print(f"\n  Best: sc={best['sc']}t  ratio={best['ratio']}  cbm={best['cbm']}  "
                  f"cl={best['cl']}  ht={best['ht']}  gr={best['gr']*100:.0f}%  "
                  f"WR={best['wr']:.1f}%  BE={best['be']:.1f}%  "
                  f"EV={best['ev']:+.3f}%  passes={best['passes']}/{WINDOWS}")
        else:
            print("  No combinations with enough trades.")

    print()
    print(SEP)
    print("  *** = 4/4 passes AND EV > 0   ** = 3/4   * = 2/4")
    print("  sc=spike_cooldown  ratio=calm_atr_ratio  cbm=calm_barrier_mult  cl=calm_lookback")
    print("  ht=hold_ticks  gr=growth_rate")


if __name__ == "__main__":
    main()
