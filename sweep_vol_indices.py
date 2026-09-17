"""
Volatility Indices Research — V10, V25, V75 (and V100).

Volatility indices are synthetic tick streams with a defined annualised
volatility percentage. They look like random-walk price series with no
algorithmic spikes, so the crash/boom recoil strategy doesn't apply.

This script tests two complementary approaches:

APPROACH A — RSI Reversal (binary)
  Mean-reversion: buy when RSI oversold, sell when overbought.
  Standard binary contract, payout ~87%.
  V10/V25 are lower-vol → RSI extremes are shallower → wider thresholds.
  V75/V100 are higher-vol → stronger trends → reversal harder, but bigger swings.

APPROACH B — CalmAccu (ACCU)
  Enter accumulator during low-volatility (calm) periods.
  Uses short-ATR < calm_ratio × long-ATR filter.
  No real spike_cooldown needed (no algorithmic spikes), so spike_cooldown=0.

Each approach is swept over 3 × 10k walk-forward windows. Results printed
side by side so you can quickly see if either approach has any edge.

Payout assumptions (binary / ACCU):
  V10:  binary ~87%   ACCU ~4% growth (BE 52.1% / 96.2% stay-alive)
  V25:  binary ~87%   ACCU ~4% growth
  V75:  binary ~87%   ACCU ~4% growth (harder — bigger moves)
  V100: binary ~87%   ACCU ~4% growth

NOTE: Get real payout from Deriv UI. ACCU barriers need check_contracts.py.

Usage:
  python sweep_vol_indices.py
  python sweep_vol_indices.py --symbol V75
  python sweep_vol_indices.py --approach accu
  python sweep_vol_indices.py --approach rsi
"""

import argparse
import sys

from loguru import logger

from src.backtest.engine import BacktestEngine
from src.data.history import fetch_ticks
from src.strategies.rsi_reversal import RSIReversalStrategy
from src.strategies.calm_accu import CalmAccuStrategy

logger.remove()
logger.add(sys.stderr, level="ERROR", format="{time:HH:mm:ss} | {level} | {message}")

SYMBOLS = ["V10", "V25", "V75", "V100"]

# Binary payout (same for all V indices on Deriv)
BINARY_PAYOUT = 0.87
BINARY_BE     = 1.0 / (1.0 + BINARY_PAYOUT) * 100

# ACCU placeholder barriers — update from check_contracts.py on VPS
ACCU_BARRIERS: dict[str, dict[float, float]] = {
    "V10":  {0.03: 1.0e-4, 0.04: 9.5e-5, 0.05: 9.0e-5},
    "V25":  {0.03: 2.5e-4, 0.04: 2.4e-4, 0.05: 2.3e-4},
    "V75":  {0.03: 7.5e-4, 0.04: 7.2e-4, 0.05: 6.9e-4},
    "V100": {0.03: 1.0e-3, 0.04: 9.6e-4, 0.05: 9.2e-4},
}

# Walk-forward config
WINDOWS     = 3
WINDOW_SIZE = 10_000
TOTAL_TICKS = WINDOWS * WINDOW_SIZE
MIN_TRADES  = 10
SEP = "=" * 110

RISK_BASE = {
    "stake_percent":      2.0,
    "max_stake":          20.0,
    "min_stake":          0.35,
    "daily_loss_limit":   100.0,
    "use_kelly":          False,
    "max_open_contracts": 1,
}

# ── RSI sweep params ──────────────────────────────────────────────────────────
RSI_CONFIGS = [
    # (rsi_fast, oversold, overbought, atr_ratio, confirm_ticks)
    (7,  20, 80, 1.2, 1),
    (7,  25, 75, 1.2, 1),
    (7,  25, 75, 1.2, 2),
    (14, 25, 75, 1.2, 1),
    (14, 30, 70, 1.0, 1),
    (14, 30, 70, 1.0, 2),
    (14, 35, 65, 1.0, 1),
    (21, 30, 70, 1.0, 1),
    (21, 30, 70, 1.2, 2),
]

# ── ACCU sweep params ─────────────────────────────────────────────────────────
ACCU_GROWTH_RATES  = [0.03, 0.04, 0.05]
ACCU_HOLD_RANGE    = [4, 6, 8, 10, 15]
ACCU_CALM_RATIOS   = [0.6, 0.75, 0.9, 1.0]


def payout_accu(gr: float, ht: int) -> float:
    return (1 + gr) ** ht - 1


def be_accu(gr: float, ht: int) -> float:
    return 1.0 / (1.0 + payout_accu(gr, ht))


# ── RSI combos ────────────────────────────────────────────────────────────────

def run_rsi_combo(ticks_split: list[list[dict]], cfg_tuple: tuple) -> dict:
    rsi_p, ov, ob, atr_r, confirm = cfg_tuple
    brk = 1.0 / (1.0 + BINARY_PAYOUT)

    strategy_cfg = {
        "rsi_period":          rsi_p,
        "rsi_period_fast":     rsi_p,
        "rsi_period_slow":     rsi_p * 2,
        "rsi_oversold":        ov,
        "rsi_overbought":      ob,
        "use_atr_filter":      True,
        "atr_filter_ratio":    atr_r,
        "rsi_slope_confirm":   True,
        "price_confirm":       True,
        "confirm_ticks":       confirm,
        "contract_duration":   3,
        "use_binary":          True,
    }
    risk_cfg = {**RISK_BASE, "payout_pct": BINARY_PAYOUT}

    wins = losses = trades = passes = 0
    for seg in ticks_split:
        if len(seg) < 50:
            continue
        engine = BacktestEngine(
            strategy_cfg=strategy_cfg,
            risk_cfg=risk_cfg,
            payout_pct=BINARY_PAYOUT,
            strategy_class=RSIReversalStrategy,
        )
        r = engine.run(seg, starting_balance=1000.0)
        if r.total_trades >= MIN_TRADES and r.win_rate >= brk * 100:
            passes += 1
        wins   += r.wins
        losses += r.losses
        trades += r.total_trades

    wr = wins / trades * 100 if trades > 0 else 0.0
    ev = (wr / 100 - brk) * BINARY_PAYOUT * 100
    return {
        "rsi": rsi_p, "ov": ov, "ob": ob, "atr_r": atr_r, "confirm": confirm,
        "wr": wr, "ev": ev, "trades": trades,
        "passes": passes, "n_windows": len(ticks_split),
    }


# ── ACCU combos ───────────────────────────────────────────────────────────────

def run_accu_combo(
    ticks_split: list[list[dict]],
    symbol: str,
    gr: float,
    ht: int,
    calm: float,
) -> dict:
    pay = payout_accu(gr, ht)
    brk = be_accu(gr, ht)
    bar = ACCU_BARRIERS[symbol][gr]

    strategy_cfg = {
        "symbol_type":        "crash",
        "long_atr_period":    50,
        "short_atr_period":   10,
        "spike_mult":         99.0,     # effectively disabled — no spikes on V indices
        "spike_cooldown":     0,
        "calm_atr_ratio":     calm,
        "entry_cooldown":     5,
        "hold_ticks":         ht,
        "growth_rate":        gr,
        "barrier_pct":        bar,
        "calm_barrier_mult":  0.0,
    }
    risk_cfg = {**RISK_BASE, "payout_pct": pay, "barrier_pct": bar}

    wins = losses = trades = passes = 0
    for seg in ticks_split:
        if len(seg) < 50:
            continue
        engine = BacktestEngine(
            strategy_cfg=strategy_cfg,
            risk_cfg=risk_cfg,
            payout_pct=pay,
            strategy_class=CalmAccuStrategy,
        )
        r = engine.run(seg, starting_balance=1000.0)
        if r.total_trades >= MIN_TRADES and r.win_rate >= brk * 100:
            passes += 1
        wins   += r.wins
        losses += r.losses
        trades += r.total_trades

    wr = wins / trades * 100 if trades > 0 else 0.0
    ev = (wr / 100 - brk) * pay * 100
    return {
        "gr": gr, "ht": ht, "calm": calm, "bar": bar,
        "pay": pay * 100, "be": brk * 100,
        "wr": wr, "ev": ev, "trades": trades,
        "passes": passes, "n_windows": len(ticks_split),
    }


# ── Symbol sweep ──────────────────────────────────────────────────────────────

def sweep_rsi(symbol: str, splits: list[list[dict]]) -> None:
    print(f"\n  [RSI REVERSAL]  payout={BINARY_PAYOUT*100:.0f}%  BE={BINARY_BE:.1f}%")
    print(f"  {'rsi':>4}  {'ov':>3}  {'ob':>3}  {'atrR':>5}  {'cfm':>3}  "
          f"{'WR%':>6}  {'EV%':>8}  {'trades':>6}  pass")
    print(f"  {'-'*4}  {'-'*3}  {'-'*3}  {'-'*5}  {'-'*3}  {'-'*6}  {'-'*8}  {'-'*6}  ----")

    results = []
    for cfg in RSI_CONFIGS:
        r = run_rsi_combo(splits, cfg)
        results.append(r)
    results.sort(key=lambda x: (x["passes"], x["ev"]), reverse=True)

    for r in results:
        if r["trades"] < MIN_TRADES:
            continue
        flag = f" ***" if r["passes"] >= r["n_windows"] and r["ev"] > 0 else (
               f" *"   if r["passes"] >= r["n_windows"] - 1 and r["ev"] > 0 else "")
        print(f"  {r['rsi']:>4}  {r['ov']:>3}  {r['ob']:>3}  {r['atr_r']:>5.2f}  {r['confirm']:>3}  "
              f"{r['wr']:>6.1f}%  {r['ev']:>+8.3f}%  {r['trades']:>6}  "
              f"{r['passes']}/{r['n_windows']}{flag}")


def sweep_accu(symbol: str, splits: list[list[dict]]) -> None:
    print(f"\n  [ACCU]  barriers are PLACEHOLDERS — run check_contracts.py for exact values")
    print(f"  {'gr':>4}  {'ht':>3}  {'calm':>5}  {'pay%':>6}  {'BE%':>6}  "
          f"{'WR%':>6}  {'EV%':>8}  {'trades':>6}  pass")
    print(f"  {'-'*4}  {'-'*3}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*6}  ----")

    results = []
    for gr in ACCU_GROWTH_RATES:
        for ht in ACCU_HOLD_RANGE:
            for calm in ACCU_CALM_RATIOS:
                r = run_accu_combo(splits, symbol, gr, ht, calm)
                results.append(r)
    results.sort(key=lambda x: (x["passes"], x["ev"]), reverse=True)

    shown = 0
    for r in results:
        if r["trades"] < MIN_TRADES:
            continue
        if shown >= 15 and r["ev"] <= 0:
            break
        flag = f" ***" if r["passes"] >= r["n_windows"] and r["ev"] > 0 else (
               f" *"   if r["passes"] >= r["n_windows"] - 1 and r["ev"] > 0 else "")
        print(f"  {r['gr']*100:>3.0f}%  {r['ht']:>3}  {r['calm']:>5.2f}  "
              f"{r['pay']:>6.1f}%  {r['be']:>6.1f}%  {r['wr']:>6.1f}%  "
              f"{r['ev']:>+8.3f}%  {r['trades']:>6}  {r['passes']}/{r['n_windows']}{flag}")
        shown += 1


def sweep_symbol(symbol: str, approach: str) -> None:
    print()
    print(SEP)
    print(f"  {symbol}  |  {WINDOWS}x{WINDOW_SIZE:,} ticks")
    print(SEP)

    ticks = fetch_ticks(symbol, count=TOTAL_TICKS)
    ticks = ticks[-TOTAL_TICKS:]
    if len(ticks) < WINDOW_SIZE:
        print(f"  Not enough ticks ({len(ticks)}), skipping.")
        return
    print(f"  Using {len(ticks):,} ticks")

    splits = [ticks[i * WINDOW_SIZE:(i + 1) * WINDOW_SIZE] for i in range(WINDOWS)]

    if approach in ("rsi", "both"):
        sweep_rsi(symbol, splits)

    if approach in ("accu", "both"):
        sweep_accu(symbol, splits)


def main() -> None:
    parser = argparse.ArgumentParser(description="Volatility indices strategy research")
    parser.add_argument("--symbol",   default=None, help="Single symbol (default: all)")
    parser.add_argument("--approach", default="both", choices=["rsi", "accu", "both"],
                        help="Which strategy approach to test")
    args = parser.parse_args()

    symbols = [args.symbol.upper()] if args.symbol else SYMBOLS

    print()
    print("  VOLATILITY INDICES RESEARCH — V10 / V25 / V75 / V100")
    print("  Approach A: RSI Reversal binary (mean-reversion)")
    print("  Approach B: CalmAccu — ACCU during low-volatility periods")

    for sym in symbols:
        if sym not in ACCU_BARRIERS:
            print(f"  Unknown symbol: {sym}")
            continue
        sweep_symbol(sym, args.approach)

    print()
    print(SEP)
    print("  *** = all windows pass AND EV>0   * = all-but-one pass AND EV>0")
    print("  ACCU barriers are placeholders — actual EV estimates unreliable until updated")


if __name__ == "__main__":
    main()
