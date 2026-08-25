"""
Jump indices spike-recoil binary sweep.

Tests CALL/PUT binary contracts after algorithmic spikes on Jump indices.
Jump indices support CALL/PUT (confirmed via check_contracts.py).

Edge thesis: after a large algorithmic jump, price quickly reverts.
  Down-jump -> BUY_RISE (CALL).  Up-jump -> BUY_FALL (PUT).
  Same logic as CRASH/BOOM recoil, but using CALL/PUT (which IS supported).

Sweep parameters:
  Symbols:    JD10, JD25, JD50, JD75, JD100
  spike_mult: 5, 8, 10, 12, 15, 20
  duration:   5t, 10t, 15t, 20t

4-fold walk-forward on 60k ticks per symbol.

Usage:
  python sweep_jump_binary.py                    # all symbols
  python sweep_jump_binary.py --symbol JD10
  python sweep_jump_binary.py --fresh            # re-download tick data
"""

import argparse
import sys

from loguru import logger

from src.backtest.engine import BacktestEngine
from src.data.history import fetch_ticks
from src.strategies.crash_boom_recoil import CrashBoomRecoilStrategy

logger.remove()
logger.add(sys.stderr, level="ERROR", format="{time:HH:mm:ss} | {level} | {message}")

SYMBOLS_JUMP = ["JD10", "JD25", "JD50", "JD75", "JD100"]
SYMBOLS_VOL  = ["R_10", "R_25", "R_50", "R_75", "R_100"]

# default: all families; override with --family jump|vol
SYMBOLS = SYMBOLS_JUMP + SYMBOLS_VOL

SPIKE_MULTS = [5.0, 8.0, 10.0, 12.0, 15.0, 20.0]
DURATIONS   = [5, 10, 15, 20]  # ticks

BINARY_PAYOUT = 0.87
BINARY_BE     = round(100 / (1 + BINARY_PAYOUT), 1)   # 53.5%

ATR_PERIOD = 50
COOLDOWN   = 5
MIN_TRADES = 10   # minimum total trades (all windows) to show a result

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

SEP = "=" * 80


def run_combo(ticks, spike_mult, duration):
    strategy_cfg = {
        "symbol_type":       "jump",
        "spike_mult":        spike_mult,
        "atr_period":        ATR_PERIOD,
        "cooldown_ticks":    COOLDOWN,
        "loss_cooldown":     2,
        "barrier_pct":       0.0,
        "confirm_threshold": 0.0,
        "contract_duration": duration,
        "use_binary":        True,
    }
    risk_cfg = {**RISK_BASE, "payout_pct": BINARY_PAYOUT}

    wins = losses = trades = passes = 0
    for w in range(WINDOWS):
        seg    = ticks[w * WINDOW_SIZE: (w + 1) * WINDOW_SIZE]
        engine = BacktestEngine(
            strategy_cfg=strategy_cfg,
            risk_cfg=risk_cfg,
            payout_pct=BINARY_PAYOUT,
            strategy_class=CrashBoomRecoilStrategy,
        )
        r = engine.run(seg, starting_balance=1000.0)
        if r.total_trades >= 3 and r.win_rate >= BINARY_BE:
            passes += 1
        wins   += r.wins
        losses += r.losses
        trades += r.total_trades

    wr = wins / trades * 100 if trades > 0 else 0.0
    ev = (wr - BINARY_BE) / 100 * BINARY_PAYOUT * 100
    return {
        "spike_mult": spike_mult,
        "duration":   duration,
        "wr":         wr,
        "ev":         ev,
        "wins":       wins,
        "trades":     trades,
        "passes":     passes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="ALL", help="Symbol or ALL")
    parser.add_argument("--family", default="all", choices=["all", "jump", "vol"],
                        help="Symbol family: jump (JD*), vol (R_*), or all")
    parser.add_argument("--fresh",  action="store_true", help="Re-download tick data")
    args = parser.parse_args()

    if args.symbol.upper() != "ALL":
        symbols = [args.symbol.upper()]
    elif args.family == "jump":
        symbols = SYMBOLS_JUMP
    elif args.family == "vol":
        symbols = SYMBOLS_VOL
    else:
        symbols = SYMBOLS
    total_ticks = WINDOWS * WINDOW_SIZE

    for symbol in symbols:
        print()
        print(SEP)
        print(f"  {symbol}  |  Binary spike-recoil sweep  |  {WINDOWS}x{WINDOW_SIZE:,} ticks")
        print(f"  payout={BINARY_PAYOUT*100:.0f}%  BE={BINARY_BE}%  atr_period={ATR_PERIOD}  cooldown={COOLDOWN}t")
        print(SEP)

        print(f"  Fetching {total_ticks:,} ticks ...")
        ticks = fetch_ticks(symbol, count=total_ticks, fresh=args.fresh)
        if len(ticks) < 1000:
            print(f"  ERROR: only {len(ticks)} ticks returned — skipping")
            continue
        if len(ticks) < total_ticks:
            print(f"  Only {len(ticks)} ticks available — using all.")
        ticks = ticks[-total_ticks:]

        results = []
        for spike_mult in SPIKE_MULTS:
            for duration in DURATIONS:
                r = run_combo(ticks, spike_mult, duration)
                r["label"] = f"spike={spike_mult:.0f}x  dur={duration:>2}t"
                if r["trades"] >= MIN_TRADES:
                    results.append(r)

        if not results:
            print(f"  No combos produced >= {MIN_TRADES} trades")
            continue

        results.sort(key=lambda x: (x["passes"], x["ev"]), reverse=True)

        print(f"  {'Label':<22}  {'WR%':>6}  {'BE%':>5}  {'EV%':>8}  {'trades':>6}  {'passes':>6}")
        print(f"  {'-'*22}  {'-'*6}  {'-'*5}  {'-'*8}  {'-'*6}  {'-'*6}")
        for r in results:
            flag = " ***" if r["passes"] == WINDOWS and r["ev"] > 0 else ""
            print(
                f"  {r['label']:<22}  {r['wr']:>6.1f}%  {BINARY_BE:>5.1f}%  "
                f"{r['ev']:>+8.3f}%  {r['trades']:>6}  {r['passes']}/{WINDOWS}{flag}"
            )

    print()
    print(SEP)
    print(f"  *** = 4/4 passes AND EV > 0")
    print(f"  spike_mult = how many x ATR the jump must exceed to trigger entry")
    print(f"  dur = ticks to hold the CALL/PUT after the recoil entry")


if __name__ == "__main__":
    main()
