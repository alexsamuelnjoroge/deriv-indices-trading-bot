"""
Max-compound ACCU sweep — removes early_sell, tests large hold_ticks.

Tests whether holding longer (no early sell, ride to max duration or next spike)
gives better EV than the current early-exit configs.

Walk-forward: 4x30k ticks (120k cache). Need >= 3/4 to mark as robust.

Current live config (baseline):
  CRASH1000: gr=5%, ht=8,  early_sell_pct=0.25 → exits ~tick 5
  BOOM1000:  gr=5%, ht=15, early_sell_pct=0.25 → exits ~tick 5

Usage:
  python sweep_max_compound.py
  python sweep_max_compound.py --symbol CRASH1000
"""

import argparse
import json
import math
import sys
from pathlib import Path

from src.strategies.crash_boom_recoil import CrashBoomRecoilStrategy
from src.data.tick_store import TickStore

CACHE_DIR   = Path("data")
TICK_COUNT  = 120_000
WINDOWS     = 4
WINDOW_SIZE = 30_000
MIN_TRADES  = 4

SYMBOLS_CFG = {
    "CRASH1000": {
        "symbol_type":       "crash",
        "spike_mult":        12.0,
        "atr_period":        50,
        "settle_ticks":      0,
        "barrier_pct":       0.00000227,
        "cooldown_ticks":    5,
        "loss_cooldown":     2,
        "confirm_threshold": 0.5,
        "mean_interval":     1143,   # from spike interval analysis
    },
    "BOOM1000": {
        "symbol_type":       "boom",
        "spike_mult":        12.0,
        "atr_period":        50,
        "settle_ticks":      3,
        "barrier_pct":       0.00000225,
        "cooldown_ticks":    5,
        "loss_cooldown":     2,
        "confirm_threshold": 0.5,
        "mean_interval":     1165,
    },
}

# hold_ticks to sweep (no early sell in any of these)
HOLD_TICKS_OPTIONS  = [5, 8, 15, 20, 30, 55, 90, 120, 200]
GROWTH_RATES        = [0.03, 0.04, 0.05]

# Current live baselines (for comparison row)
BASELINES = {
    "CRASH1000": {"gr": 0.05, "ht": 8,  "early_sell_pct": 0.25, "settle": 0},
    "BOOM1000":  {"gr": 0.05, "ht": 15, "early_sell_pct": 0.25, "settle": 3},
}


def payout_pct(gr: float, ht: int) -> float:
    return (1 + gr) ** ht - 1


def be_pct(gr: float, ht: int) -> float:
    pay = payout_pct(gr, ht)
    return 1.0 / (1.0 + pay)


def early_sell_tick(gr: float, esp: float) -> int:
    """How many ticks until early_sell_pct is reached."""
    if esp <= 0:
        return 9999
    return math.ceil(math.log(1 + esp) / math.log(1 + gr))


def simulate_window(ticks, cfg, gr, ht, bar, early_sell=0.0):
    """
    Simulate one walk-forward window.
    Returns (wins, total_trades).

    early_sell: fraction profit at which to exit early (0 = disabled).
    """
    strategy = CrashBoomRecoilStrategy({
        **cfg,
        "growth_rate":  gr,
        "hold_ticks":   ht,
        "barrier_pct":  bar,
    })
    store    = TickStore(max_ticks=500)
    hold_end = -1
    wins = trades = 0
    esp_tick = early_sell_tick(gr, early_sell)

    for i, tick in enumerate(ticks):
        store.add(tick)
        sig = strategy.evaluate(store)

        if sig.action == "BUY_ACCU" and i > hold_end:
            # Effective hold is capped by early sell tick
            effective_ht = min(ht, esp_tick)
            survived = True
            for j in range(i + 1, min(i + 1 + effective_ht, len(ticks))):
                prev = float(ticks[j - 1]["quote"])
                curr = float(ticks[j]["quote"])
                if prev > 0 and abs(curr - prev) / prev > bar:
                    survived = False
                    break
            if survived:
                wins += 1
            trades += 1
            hold_end = i + effective_ht

    return wins, trades


def run_combo(ticks, cfg, gr, ht, early_sell=0.0):
    bar = cfg["barrier_pct"]
    be  = be_pct(gr, ht if early_sell <= 0 else early_sell_tick(gr, early_sell))
    pay = payout_pct(gr, ht if early_sell <= 0 else early_sell_tick(gr, early_sell))

    window_wins   = []
    window_trades = []
    passes = 0

    for w in range(WINDOWS):
        seg = ticks[w * WINDOW_SIZE: (w + 1) * WINDOW_SIZE]
        wins, trades = simulate_window(seg, cfg, gr, ht, bar, early_sell)
        window_wins.append(wins)
        window_trades.append(trades)
        if trades >= MIN_TRADES and (wins / trades if trades else 0) >= be:
            passes += 1

    total_wins   = sum(window_wins)
    total_trades = sum(window_trades)
    wr   = total_wins / total_trades * 100 if total_trades else 0
    ev   = (wr / 100 - be) * pay * 100
    return wr, ev, total_trades, passes, [w / t * 100 if t else 0
                                          for w, t in zip(window_wins, window_trades)]


def analyze(symbol: str) -> None:
    cfg    = SYMBOLS_CFG[symbol]
    path   = CACHE_DIR / f"{symbol}_{TICK_COUNT}.json"
    if not path.exists():
        print(f"  {symbol}: no {TICK_COUNT}-tick cache")
        return

    with open(path) as f:
        ticks = json.load(f)
    print(f"  Using {len(ticks):,} ticks  |  4x{WINDOW_SIZE:,} walk-forward")

    SEP = "=" * 110
    print()
    print(SEP)
    print(f"  {symbol}  —  Max-Compound Sweep  (no early sell unless marked [baseline])")
    print(SEP)
    print(f"  {'gr':>4}  {'ht':>4}  {'eff_ht':>6}  {'pay%':>7}  {'BE%':>5}  "
          f"{'WR%':>6}  {'EV%':>8}  {'trades':>7}  pass  note")
    print(f"  {'-'*4}  {'-'*4}  {'-'*6}  {'-'*7}  {'-'*5}  "
          f"{'-'*6}  {'-'*8}  {'-'*7}  ----  ----")

    baseline = BASELINES[symbol]
    bl_gr    = baseline["gr"]
    bl_ht    = baseline["ht"]
    bl_esp   = baseline["early_sell_pct"]
    bl_eff   = early_sell_tick(bl_gr, bl_esp)

    # Print baseline first
    wr, ev, trades, passes, wrs = run_combo(ticks, cfg, bl_gr, bl_ht, bl_esp)
    pay = payout_pct(bl_gr, bl_eff)
    be  = be_pct(bl_gr, bl_eff)
    mark = "***" if passes >= 3 and ev > 0 else ("*" if passes >= 2 and ev > 0 else "")
    print(f"  {bl_gr*100:.0f}%  {bl_ht:>4}  {bl_eff:>6}  {pay*100:>6.1f}%  {be*100:>5.1f}%  "
          f"{wr:>5.1f}%  {ev:>+8.3f}%  {trades:>7}  {passes}/{WINDOWS} {mark:<3}  [BASELINE]")

    best_ev = ev
    best_cfg = None

    for gr in GROWTH_RATES:
        for ht in HOLD_TICKS_OPTIONS:
            if ht == bl_ht and gr == bl_gr:
                continue   # already shown as baseline
            wr, ev, trades, passes, wrs = run_combo(ticks, cfg, gr, ht, early_sell=0.0)
            pay = payout_pct(gr, ht)
            be  = be_pct(gr, ht)
            mark = "***" if passes >= 3 and ev > 0 else ("*" if passes >= 2 and ev > 0 else "")
            # P(survive) using exponential approximation
            p_surv = math.exp(-ht / cfg["mean_interval"]) * 100
            print(f"  {gr*100:.0f}%  {ht:>4}  {ht:>6}  {pay*100:>6.1f}%  {be*100:>5.1f}%  "
                  f"{wr:>5.1f}%  {ev:>+8.3f}%  {trades:>7}  {passes}/{WINDOWS} {mark:<3}  "
                  f"P(no spike)~{p_surv:.0f}%")
            if passes >= 3 and ev > best_ev:
                best_ev  = ev
                best_cfg = (gr, ht, wr, ev, pay, be, passes)

    print()
    if best_cfg:
        gr2, ht2, wr2, ev2, pay2, be2, ps2 = best_cfg
        p_surv = math.exp(-ht2 / cfg["mean_interval"]) * 100
        print(f"  Best: gr={gr2*100:.0f}%  ht={ht2}  WR={wr2:.1f}%  BE={be2*100:.1f}%  "
              f"EV={ev2:+.3f}%  passes={ps2}/{WINDOWS}")
        print(f"  At $1 stake: win=${pay2:.2f}  P(survive to ht)~{p_surv:.0f}%")
        print(f"  EV improvement vs baseline: {ev2 - BASELINES[symbol]['__ev'] if '__ev' in BASELINES[symbol] else 'see above'}x")
    else:
        print(f"  No config beats baseline with >= 3/4 passes.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", help="CRASH1000 or BOOM1000 (default: both)")
    args = parser.parse_args()

    symbols = ([args.symbol] if args.symbol and args.symbol in SYMBOLS_CFG
               else list(SYMBOLS_CFG.keys()))

    print("Max-Compound ACCU sweep — removing early_sell_pct")
    print("Comparing current early-exit baseline vs longer holds.")
    print("Walk-forward: 4x30k. Need >= 3/4 passes for robust.\n")

    for sym in symbols:
        analyze(sym)
        print()


if __name__ == "__main__":
    main()
