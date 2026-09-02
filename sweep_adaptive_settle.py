"""
Adaptive ATR settle sweep for CrashBoomRecoilStrategy.

Instead of waiting a fixed settle_ticks after the confirm gate, the adaptive
mode waits until the short-term ATR drops below settle_atr_ratio × barrier_width.
This targets barrier breach probability directly rather than guessing via a timer.

Sweeps:
  settle_atr_ratio     0.3 → 2.0  (how tight the ATR gate is)
  settle_short_period  3, 5, 8    (window for short ATR measurement)
  max_settle_ticks     15, 30, 50 (cap on wait before abandoning entry)
  hold_ticks           symbol-specific
  growth_rate          0.04, 0.05

Each symbol also runs its current fixed-settle baseline for direct comparison.

Symbols: BOOM1000, CRASH1000, CRASH150N, CRASH600

Usage:
  python sweep_adaptive_settle.py
  python sweep_adaptive_settle.py --symbol BOOM1000
"""

import argparse
import sys

from loguru import logger

from src.backtest.engine import BacktestEngine
from src.data.history import fetch_ticks
from src.strategies.crash_boom_recoil import CrashBoomRecoilStrategy

logger.remove()
logger.add(sys.stderr, level="ERROR", format="{time:HH:mm:ss} | {level} | {message}")

SYMBOLS = {
    # barrier_pct at ~4% growth (1% growth value × 0.849, verified on BOOM1000/150N/300N)
    # hold_range chosen to keep P(in-hold spike) < 15% for the symbol's rate
    "BOOM50":    {"symbol_type": "boom",  "barrier_pct": 1.55e-5, "hold_range": [3, 5, 8],    "growth_rates": [0.04, 0.05], "baseline_settle": 0},
    "CRASH50":   {"symbol_type": "crash", "barrier_pct": 1.55e-5, "hold_range": [3, 5, 8],    "growth_rates": [0.04, 0.05], "baseline_settle": 0},
    "BOOM150N":  {"symbol_type": "boom",  "barrier_pct": 1.61e-6, "hold_range": [8, 10, 12],  "growth_rates": [0.04, 0.05], "baseline_settle": 0},
    "CRASH150N": {"symbol_type": "crash", "barrier_pct": 1.61e-6, "hold_range": [8, 10, 12],  "growth_rates": [0.04, 0.05], "baseline_settle": 0},
    "BOOM300N":  {"symbol_type": "boom",  "barrier_pct": 2.07e-5, "hold_range": [10, 15, 20], "growth_rates": [0.04, 0.05], "baseline_settle": 0},
    "CRASH300N": {"symbol_type": "crash", "barrier_pct": 2.07e-5, "hold_range": [10, 15, 20], "growth_rates": [0.04, 0.05], "baseline_settle": 0},
    "BOOM500":   {"symbol_type": "boom",  "barrier_pct": 4.72e-6, "hold_range": [12, 15, 20], "growth_rates": [0.04, 0.05], "baseline_settle": 0},
    "CRASH500":  {"symbol_type": "crash", "barrier_pct": 4.72e-6, "hold_range": [12, 15, 20], "growth_rates": [0.04, 0.05], "baseline_settle": 0},
    "BOOM600":   {"symbol_type": "boom",  "barrier_pct": 3.92e-6, "hold_range": [6, 8, 10],   "growth_rates": [0.04, 0.05], "baseline_settle": 0},
    "CRASH600":  {"symbol_type": "crash", "barrier_pct": 3.92e-6, "hold_range": [6, 8, 10],   "growth_rates": [0.04, 0.05], "baseline_settle": 15},
    "BOOM900":   {"symbol_type": "boom",  "barrier_pct": 2.60e-6, "hold_range": [12, 15, 18], "growth_rates": [0.04, 0.05], "baseline_settle": 0},
    "CRASH900":  {"symbol_type": "crash", "barrier_pct": 2.60e-6, "hold_range": [12, 15, 18], "growth_rates": [0.04, 0.05], "baseline_settle": 0},
    "BOOM1000":  {"symbol_type": "boom",  "barrier_pct": 2.35e-6, "hold_range": [12, 15, 18], "growth_rates": [0.04, 0.05], "baseline_settle": 3},
    "CRASH1000": {"symbol_type": "crash", "barrier_pct": 2.33e-6, "hold_range": [8, 10, 12],  "growth_rates": [0.04, 0.05], "baseline_settle": 0},
}

SETTLE_ATR_RATIOS   = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
SETTLE_SHORT_PERIODS = [3, 5, 8]
MAX_SETTLE_TICKS    = [15, 30, 50]

SPIKE_MULT     = 12.0
ATR_PERIOD     = 50
COOLDOWN_TICKS = 5
LOSS_COOLDOWN  = 2
CONFIRM_THRESH = 0.5
CLUSTER_WINDOW = 1000
MAX_CLUSTER_SPIKES = 3

WINDOWS     = 4
WINDOW_SIZE = 21_500

RISK_BASE = {
    "stake_percent":      2.0,
    "max_stake":          2.0,
    "min_stake":          1.00,
    "daily_loss_limit":   100.0,
    "use_kelly":          False,
    "max_open_contracts": 1,
}

MIN_TRADES = 4
SEP = "=" * 140


def payout(gr, ht):
    return (1 + gr) ** ht - 1


def be(gr, ht):
    return 1.0 / (1.0 + payout(gr, ht))


def run_combo(ticks, meta, strategy_cfg, gr, ht):
    pay = payout(gr, ht)
    brk = be(gr, ht)

    cfg = {
        **strategy_cfg,
        "growth_rate": gr,
        "hold_ticks":  ht,
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
            strategy_cfg=cfg,
            risk_cfg=risk_cfg,
            payout_pct=pay,
            strategy_class=CrashBoomRecoilStrategy,
        )
        r = engine.run(seg, starting_balance=1000.0)
        if r.total_trades >= 2 and r.win_rate >= brk * 100:
            passes += 1
        wins   += r.wins
        losses += r.losses
        trades += r.total_trades

    wr = wins / trades * 100 if trades > 0 else 0.0
    ev = (wr / 100 - brk) * pay * 100
    return {
        "wr": wr, "ev": ev, "trades": trades, "passes": passes,
        "pay": pay * 100, "be": brk * 100,
    }


def baseline_cfg(meta, settle):
    return {
        "symbol_type":     meta["symbol_type"],
        "spike_mult":      SPIKE_MULT,
        "atr_period":      ATR_PERIOD,
        "cooldown_ticks":  COOLDOWN_TICKS,
        "loss_cooldown":   LOSS_COOLDOWN,
        "barrier_pct":     meta["barrier_pct"],
        "confirm_threshold": CONFIRM_THRESH,
        "cluster_window":  CLUSTER_WINDOW,
        "max_cluster_spikes": MAX_CLUSTER_SPIKES,
        "min_spike_ratio": 0,
        "settle_ticks":    settle,
        "adaptive_settle": False,
    }


def adaptive_cfg(meta, ratio, period, max_wait):
    return {
        "symbol_type":      meta["symbol_type"],
        "spike_mult":       SPIKE_MULT,
        "atr_period":       ATR_PERIOD,
        "cooldown_ticks":   COOLDOWN_TICKS,
        "loss_cooldown":    LOSS_COOLDOWN,
        "barrier_pct":      meta["barrier_pct"],
        "confirm_threshold": CONFIRM_THRESH,
        "cluster_window":   CLUSTER_WINDOW,
        "max_cluster_spikes": MAX_CLUSTER_SPIKES,
        "min_spike_ratio":  0,
        "settle_ticks":     0,
        "adaptive_settle":  True,
        "settle_atr_ratio": ratio,
        "settle_short_period": period,
        "max_settle_ticks": max_wait,
    }


def print_header(label):
    print(f"\n  {'mode':<22}  {'ratio':>5}  {'per':>3}  {'max':>3}  "
          f"{'ht':>3}  {'gr':>4}  {'pay%':>6}  {'BE%':>6}  "
          f"{'WR%':>6}  {'EV%':>8}  {'trades':>6}  pass")
    print(f"  {'-'*22}  {'-'*5}  {'-'*3}  {'-'*3}  "
          f"{'-'*3}  {'-'*4}  {'-'*6}  {'-'*6}  "
          f"{'-'*6}  {'-'*8}  {'-'*6}  ----")


def format_row(mode, ratio, period, max_wait, ht, gr, r, flag=""):
    return (
        f"  {mode:<22}  {ratio:>5}  {period:>3}  {max_wait:>3}  "
        f"{ht:>3}  {gr*100:>3.0f}%  "
        f"{r['pay']:>6.1f}%  {r['be']:>6.1f}%  {r['wr']:>6.1f}%  "
        f"{r['ev']:>+8.3f}%  {r['trades']:>6}  {r['passes']}/{WINDOWS}{flag}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="ALL")
    args = parser.parse_args()

    symbols = list(SYMBOLS.keys()) if args.symbol.upper() == "ALL" else [args.symbol.upper()]
    total_ticks = WINDOWS * WINDOW_SIZE

    for symbol in symbols:
        meta = SYMBOLS[symbol]

        print()
        print(SEP)
        print(f"  {symbol}  |  Adaptive ATR Settle Sweep  |  {WINDOWS}×{WINDOW_SIZE:,} ticks")
        print(f"  Fixed: spike_mult={SPIKE_MULT:.0f}x  atr_period={ATR_PERIOD}  "
              f"confirm={CONFIRM_THRESH}  barrier={meta['barrier_pct']:.2e}")
        print(f"  Sweeping: settle_atr_ratio  settle_short_period  max_settle_ticks  hold_ticks  growth_rate")
        print(SEP)

        ticks = fetch_ticks(symbol, count=total_ticks + 5_000)
        ticks = ticks[-total_ticks:]
        print(f"  Using {len(ticks):,} ticks\n")

        # ── Baseline: fixed settle_ticks ──────────────────────────────────
        print(f"  BASELINE (fixed settle_ticks={meta['baseline_settle']}):")
        print_header("BASELINE")
        for gr in meta["growth_rates"]:
            for ht in meta["hold_range"]:
                cfg = baseline_cfg(meta, meta["baseline_settle"])
                r = run_combo(ticks, meta, cfg, gr, ht)
                if r["trades"] < MIN_TRADES:
                    continue
                flag = " ***" if r["passes"] == 4 else " **" if r["passes"] == 3 else " *" if r["passes"] == 2 else ""
                print(format_row("fixed", meta["baseline_settle"], "-", "-", ht, gr, r, flag))

        # ── Adaptive settle sweep ─────────────────────────────────────────
        print(f"\n  ADAPTIVE (settle_atr_ratio × settle_short_period × max_settle_ticks):")
        total_combos = (len(SETTLE_ATR_RATIOS) * len(SETTLE_SHORT_PERIODS) *
                        len(MAX_SETTLE_TICKS) * len(meta["hold_range"]) * len(meta["growth_rates"]))
        print(f"  Running {total_combos} combinations...")

        results = []
        done = 0
        for ratio in SETTLE_ATR_RATIOS:
            for period in SETTLE_SHORT_PERIODS:
                for max_wait in MAX_SETTLE_TICKS:
                    for gr in meta["growth_rates"]:
                        for ht in meta["hold_range"]:
                            cfg = adaptive_cfg(meta, ratio, period, max_wait)
                            r = run_combo(ticks, meta, cfg, gr, ht)
                            results.append({
                                **r,
                                "ratio": ratio, "period": period,
                                "max_wait": max_wait, "gr": gr, "ht": ht,
                            })
                            done += 1
                            if done % 30 == 0:
                                print(f"  ... {done}/{total_combos}")

        results.sort(key=lambda x: (x["passes"], x["ev"]), reverse=True)

        print_header("ADAPTIVE")
        shown = 0
        for r in results:
            if r["trades"] < MIN_TRADES:
                continue
            flag = " ***" if r["passes"] == 4 else " **" if r["passes"] == 3 else " *" if r["passes"] == 2 else ""
            print(format_row(
                "adaptive", r["ratio"], r["period"], r["max_wait"],
                r["ht"], r["gr"], r, flag,
            ))
            shown += 1
            if shown >= 30:
                remaining = sum(1 for x in results if x["trades"] >= MIN_TRADES) - shown
                if remaining > 0:
                    print(f"  ... ({remaining} more not shown)")
                break

        valid = [r for r in results if r["trades"] >= MIN_TRADES]
        if valid:
            best = valid[0]
            print(f"\n  Best adaptive: ratio={best['ratio']}  period={best['period']}  "
                  f"max_wait={best['max_wait']}  ht={best['ht']}  gr={best['gr']*100:.0f}%  "
                  f"WR={best['wr']:.1f}%  BE={best['be']:.1f}%  "
                  f"EV={best['ev']:+.3f}%  passes={best['passes']}/{WINDOWS}")

    print()
    print(SEP)
    print("  *** = 4/4 passes AND EV > 0   ** = 3/4   * = 2/4")
    print("  ratio=settle_atr_ratio  per=settle_short_period  max=max_settle_ticks")
    print("  Adaptive wins if WR or EV improves over baseline at same passes level")


if __name__ == "__main__":
    main()
