"""
Advanced CRASH/BOOM sweep: volatility filter, trend filter, and binary mode.

Tests two improvement approaches on all 4 active CRASH/BOOM symbols:

  1. FILTER SWEEP (ACCU mode):
     Keeps current best params per symbol and varies:
       - volatility_filter_window (pre-spike tick-range calm gate)
       - trend_filter_window (pre-spike trend direction gate)
     These params are already coded in crash_boom_recoil.py but never swept.

  2. BINARY MODE:
     After a CRASH spike -> BUY_RISE (CALL). After a BOOM spike -> BUY_FALL (PUT).
     Sweeps contract_duration: 5, 10, 15, 20 ticks.
     Binary payout ~87% on synthetics; BE = 53.5%.
     No confirm gate (direction, not size, matters for binary).

4-fold walk-forward on 60k ticks per symbol (4 x 15k).
Reports passes >= 3/4 or any result beating current ACCU baseline.

Usage:
  python sweep_crash_boom_advanced.py                    # all symbols, both modes
  python sweep_crash_boom_advanced.py --symbol CRASH1000
  python sweep_crash_boom_advanced.py --mode binary
  python sweep_crash_boom_advanced.py --mode filters
  python sweep_crash_boom_advanced.py --fresh            # re-download tick data
"""

import argparse
import sys

from loguru import logger

from src.backtest.engine import BacktestEngine
from src.data.history import fetch_ticks
from src.strategies.crash_boom_recoil import CrashBoomRecoilStrategy

logger.remove()
logger.add(sys.stderr, level="ERROR", format="{time:HH:mm:ss} | {level} | {message}")

# ── Current best params per symbol (from validated sweeps) ──────────────────
# barrier_pct = real Deriv API value at the specified growth_rate (from config.yaml).
BEST_PARAMS = {
    "CRASH1000": {"growth_rate": 0.05, "hold_ticks": 8,  "spike_mult": 12.0, "confirm_threshold": 0.5, "symbol_type": "crash", "barrier_pct": 2.27e-6},
    "CRASH900":  {"growth_rate": 0.04, "hold_ticks": 20, "spike_mult": 12.0, "confirm_threshold": 0.5, "symbol_type": "crash", "barrier_pct": 2.61e-6},
    "BOOM1000":  {"growth_rate": 0.05, "hold_ticks": 15, "spike_mult": 12.0, "confirm_threshold": 0.5, "symbol_type": "boom",  "barrier_pct": 2.25e-6},
    "BOOM900":   {"growth_rate": 0.05, "hold_ticks": 12, "spike_mult": 12.0, "confirm_threshold": 0.5, "symbol_type": "boom",  "barrier_pct": 2.51e-6},
}

# ── Known EV baselines (no filters) ─────────────────────────────────────────
BASELINE_EV = {
    "CRASH1000": +0.84,
    "CRASH900":  +15.46,
    "BOOM1000":  +9.76,
    "BOOM900":   +7.08,
}

# ── Filter sweep grid ────────────────────────────────────────────────────────
# volatility_filter_window=0 disables the filter (baseline).
VFILTER_WINDOWS = [0, 30, 60, 100, 150]
VFILTER_MULTS   = [1.5, 2.0, 2.5, 3.0]
TREND_WINDOWS   = [0, 30, 60, 100, 200]

# ── Binary sweep grid ────────────────────────────────────────────────────────
BINARY_DURATIONS = [5, 10, 15, 20]
BINARY_PAYOUT    = 0.87   # ~87% on Deriv synthetic binaries
BINARY_BE        = round(100 / (1 + BINARY_PAYOUT), 1)

ATR_PERIOD = 50
COOLDOWN   = 5

WINDOWS     = 4
WINDOW_SIZE = 15_000

RISK_BASE = {
    "stake_percent":      2.0,
    "max_stake":          20.0,
    "min_stake":          0.35,
    "daily_loss_limit":   100.0,
    "use_kelly":          False,
    "max_open_contracts": 1,
}

SEP  = "=" * 95
THIN = "-" * 95


# ── Helpers ──────────────────────────────────────────────────────────────────

def _accu_payout(growth_rate, hold_ticks):
    return (1 + growth_rate) ** hold_ticks - 1


def _accu_be(growth_rate, hold_ticks):
    p = _accu_payout(growth_rate, hold_ticks)
    return 1.0 / (1.0 + p)


def run_accu_combo(ticks, symbol, vf_window, vf_mult, tr_window):
    """Run ACCU with current best params + given filter params."""
    bp          = BEST_PARAMS[symbol]
    payout      = _accu_payout(bp["growth_rate"], bp["hold_ticks"])
    be          = _accu_be(bp["growth_rate"], bp["hold_ticks"])
    barrier_pct = bp["barrier_pct"]

    strategy_cfg = {
        "symbol_type":              bp["symbol_type"],
        "spike_mult":               bp["spike_mult"],
        "atr_period":               ATR_PERIOD,
        "cooldown_ticks":           COOLDOWN,
        "loss_cooldown":            2,
        "hold_ticks":               bp["hold_ticks"],
        "growth_rate":              bp["growth_rate"],
        "barrier_pct":              barrier_pct,
        "confirm_threshold":        bp["confirm_threshold"],
        "volatility_filter_window": vf_window,
        "volatility_filter_mult":   vf_mult,
        "trend_filter_window":      tr_window,
        "use_binary":               False,
    }
    risk_cfg = {**RISK_BASE, "barrier_pct": barrier_pct, "payout_pct": payout}

    wins = losses = trades = 0
    passes = 0
    for w in range(WINDOWS):
        seg    = ticks[w * WINDOW_SIZE: (w + 1) * WINDOW_SIZE]
        engine = BacktestEngine(
            strategy_cfg=strategy_cfg,
            risk_cfg=risk_cfg,
            payout_pct=payout,
            strategy_class=CrashBoomRecoilStrategy,
        )
        r = engine.run(seg, starting_balance=1000.0)
        if r.win_rate >= be * 100 and r.total_trades >= 5:
            passes += 1
        wins   += r.wins
        losses += r.losses
        trades += r.total_trades

    wr   = wins / trades * 100 if trades > 0 else 0.0
    ev   = (wr - be * 100) / 100 * payout * 100
    return {
        "mode": "ACCU",
        "vf_window": vf_window,
        "vf_mult":   vf_mult,
        "tr_window": tr_window,
        "payout":    payout * 100,
        "be":        be * 100,
        "wr":        wr,
        "ev":        ev,
        "trades":    trades,
        "passes":    passes,
    }


def run_binary_combo(ticks, symbol, duration):
    """Run binary (CALL/PUT) post-spike mode with given hold duration."""
    bp  = BEST_PARAMS[symbol]
    be  = BINARY_BE

    strategy_cfg = {
        "symbol_type":       bp["symbol_type"],
        "spike_mult":        bp["spike_mult"],
        "atr_period":        ATR_PERIOD,
        "cooldown_ticks":    COOLDOWN,
        "loss_cooldown":     2,
        "barrier_pct":       0.0,       # no barrier in binary mode
        "confirm_threshold": 0.0,       # no confirm gate
        "contract_duration": duration,
        "use_binary":        True,
    }
    risk_cfg = {**RISK_BASE, "payout_pct": BINARY_PAYOUT}

    wins = losses = trades = 0
    passes = 0
    for w in range(WINDOWS):
        seg    = ticks[w * WINDOW_SIZE: (w + 1) * WINDOW_SIZE]
        engine = BacktestEngine(
            strategy_cfg=strategy_cfg,
            risk_cfg=risk_cfg,
            payout_pct=BINARY_PAYOUT,
            strategy_class=CrashBoomRecoilStrategy,
        )
        r = engine.run(seg, starting_balance=1000.0)
        if r.win_rate >= be and r.total_trades >= 5:
            passes += 1
        wins   += r.wins
        losses += r.losses
        trades += r.total_trades

    wr = wins / trades * 100 if trades > 0 else 0.0
    ev = (wr - be) / 100 * BINARY_PAYOUT * 100
    return {
        "mode":     "BINARY",
        "duration": duration,
        "payout":   BINARY_PAYOUT * 100,
        "be":       be,
        "wr":       wr,
        "ev":       ev,
        "trades":   trades,
        "passes":   passes,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Advanced CRASH/BOOM strategy sweep")
    parser.add_argument("--symbol", default="ALL", help="Symbol or ALL")
    parser.add_argument("--mode",   default="both", choices=["both", "filters", "binary"])
    parser.add_argument("--fresh",  action="store_true")
    args = parser.parse_args()

    if args.symbol.upper() == "ALL":
        symbols = list(BEST_PARAMS.keys())
    else:
        symbols = [args.symbol.upper()]

    total_ticks_needed = WINDOWS * WINDOW_SIZE

    for symbol in symbols:
        if symbol not in BEST_PARAMS:
            print(f"ERROR: {symbol} not in BEST_PARAMS")
            continue

        print()
        print(SEP)
        print(f"  {symbol} — Advanced Strategy Sweep  |  {WINDOWS}x{WINDOW_SIZE:,} ticks")
        bp = BEST_PARAMS[symbol]
        payout = _accu_payout(bp["growth_rate"], bp["hold_ticks"])
        be     = _accu_be(bp["growth_rate"], bp["hold_ticks"])
        print(f"  ACCU baseline: gr={bp['growth_rate']*100:.0f}% ht={bp['hold_ticks']} "
              f"payout={payout*100:.1f}% BE={be*100:.1f}% | known EV={BASELINE_EV[symbol]:+.2f}%")
        print(f"  BINARY baseline: payout={BINARY_PAYOUT*100:.0f}% BE={BINARY_BE:.1f}%")
        print(SEP)

        print(f"  Fetching {total_ticks_needed:,} ticks ...")
        ticks = fetch_ticks(symbol, count=total_ticks_needed, fresh=args.fresh)
        if len(ticks) < total_ticks_needed:
            print(f"  Only {len(ticks)} ticks available — using all.")
        ticks = ticks[-total_ticks_needed:]

        # ── Filter sweep ──────────────────────────────────────────────────
        if args.mode in ("both", "filters"):
            print()
            print(f"  --- ACCU Filter Sweep (volatility_filter x trend_filter) ---")
            filter_results = []

            # Baseline (no filters)
            r = run_accu_combo(ticks, symbol, 0, 0.0, 0)
            r["label"] = "BASELINE (no filter)"
            filter_results.append(r)

            # Volatility filter only
            for vfw in VFILTER_WINDOWS[1:]:    # skip 0 (already baseline)
                for vfm in VFILTER_MULTS:
                    r = run_accu_combo(ticks, symbol, vfw, vfm, 0)
                    r["label"] = f"vfilter w={vfw} m={vfm:.1f}"
                    filter_results.append(r)

            # Trend filter only
            for trw in TREND_WINDOWS[1:]:      # skip 0 (already baseline)
                r = run_accu_combo(ticks, symbol, 0, 0.0, trw)
                r["label"] = f"trend w={trw}"
                filter_results.append(r)

            # Best combo: top vfilter + best trend
            best_vf = max(filter_results[1:len(VFILTER_WINDOWS)*len(VFILTER_MULTS)],
                          key=lambda x: (x["passes"], x["ev"]))
            if best_vf["passes"] >= 3:
                for trw in TREND_WINDOWS[1:]:
                    r = run_accu_combo(ticks, symbol, best_vf["vf_window"], best_vf["vf_mult"], trw)
                    r["label"] = f"vfilter w={best_vf['vf_window']} m={best_vf['vf_mult']:.1f} + trend w={trw}"
                    filter_results.append(r)

            filter_results.sort(key=lambda x: (x["passes"], x["ev"]), reverse=True)
            print(f"  {'Label':<42}  {'WR%':>5}  {'BE%':>5}  {'EV%':>7}  {'trades':>6}  pass")
            print(f"  {'-'*42}  {'---':>5}  {'---':>5}  {'---':>7}  {'------':>6}  ----")
            for r in filter_results[:20]:
                flag = " ***" if r["passes"] == WINDOWS and r["ev"] > BASELINE_EV[symbol] else ""
                print(f"  {r['label']:<42}  {r['wr']:>5.1f}%  {r['be']:>5.1f}%  {r['ev']:>+7.3f}%  "
                      f"{r['trades']:>6}  {r['passes']}/{WINDOWS}{flag}")

        # ── Binary mode sweep ─────────────────────────────────────────────
        if args.mode in ("both", "binary"):
            print()
            print(f"  --- Binary Mode Sweep (CALL/PUT post-spike, payout={BINARY_PAYOUT*100:.0f}%) ---")
            bin_results = []
            for dur in BINARY_DURATIONS:
                r = run_binary_combo(ticks, symbol, dur)
                r["label"] = f"hold={dur}t"
                bin_results.append(r)

            bin_results.sort(key=lambda x: (x["passes"], x["ev"]), reverse=True)
            print(f"  {'Label':<12}  {'WR%':>5}  {'BE%':>5}  {'EV%':>7}  {'trades':>6}  pass")
            print(f"  {'-'*12}  {'---':>5}  {'---':>5}  {'---':>7}  {'------':>6}  ----")
            for r in bin_results:
                flag = " ***" if r["passes"] == WINDOWS and r["ev"] > 0 else ""
                print(f"  {r['label']:<12}  {r['wr']:>5.1f}%  {r['be']:>5.1f}%  {r['ev']:>+7.3f}%  "
                      f"{r['trades']:>6}  {r['passes']}/{WINDOWS}{flag}")

    print()
    print(SEP)
    print("  *** = all passes AND EV beats baseline (ACCU) or > 0 (binary)")
    print(f"  Filter sweep: volatility gate checks pre-spike tick range vs ATR.")
    print(f"  Trend filter: CRASH needs uptrend before spike; BOOM needs downtrend.")
    print(f"  Binary mode: BUY_RISE after CRASH spike / BUY_FALL after BOOM spike.")


if __name__ == "__main__":
    main()
