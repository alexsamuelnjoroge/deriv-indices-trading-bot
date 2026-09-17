"""
Forex Accumulator (ACCU) Research — frxGBPUSD, frxAUDUSD, frxEURUSD, frxUSDJPY.

Deriv's ACCU contract on forex works the same way: stake compounds at growth_rate
each tick as long as price doesn't move more than barrier_pct from the previous tick.
On forex, spikes are rarer than Crash/Boom so the CalmAccuStrategy inter-spike
logic becomes a general low-vol filter.

Barriers from Deriv API (run check_contracts.py to refresh):
  frxGBPUSD:  growth=4% → 0.00027  growth=5% → 0.00026  (approx 0.027% per tick)
  frxAUDUSD:  growth=4% → 0.00027  growth=5% → 0.00026
  frxEURUSD:  growth=4% → 0.00027  growth=5% → 0.00026
  frxUSDJPY:  growth=4% → 0.00027  growth=5% → 0.00026

NOTE: These are placeholder barriers. Run check_contracts.py on the VPS to get
exact live values before trusting EV estimates.

Sweep:
  growth_rate    0.03, 0.04, 0.05
  hold_ticks     4, 6, 8, 10, 15, 20
  calm_atr_ratio 0.7, 0.85, 1.0
  spike_mult     8, 12, 18

Walk-forward: 3 × 10k ticks.

Usage:
  python sweep_forex_accu.py
  python sweep_forex_accu.py --symbol frxGBPUSD
"""

import argparse
import sys

from loguru import logger

from src.backtest.engine import BacktestEngine
from src.data.history import fetch_ticks
from src.strategies.calm_accu import CalmAccuStrategy

logger.remove()
logger.add(sys.stderr, level="ERROR", format="{time:HH:mm:ss} | {level} | {message}")

# Placeholder barriers — update from check_contracts.py on VPS
LIVE_BARRIERS: dict[str, dict[float, float]] = {
    "frxGBPUSD": {0.03: 2.7e-4, 0.04: 2.7e-4, 0.05: 2.6e-4},
    "frxAUDUSD": {0.03: 2.7e-4, 0.04: 2.7e-4, 0.05: 2.6e-4},
    "frxEURUSD": {0.03: 2.7e-4, 0.04: 2.7e-4, 0.05: 2.6e-4},
    "frxUSDJPY": {0.03: 2.7e-4, 0.04: 2.7e-4, 0.05: 2.6e-4},
}

SYMBOLS = list(LIVE_BARRIERS.keys())

GROWTH_RATES  = [0.03, 0.04, 0.05]
HOLD_RANGE    = [4, 6, 8, 10, 15, 20]
CALM_RATIOS   = [0.7, 0.85, 1.0]
SPIKE_MULTS   = [8.0, 12.0, 18.0]

WINDOWS     = 3
WINDOW_SIZE = 10_000
TOTAL_TICKS = WINDOWS * WINDOW_SIZE
MIN_TRADES  = 4
SEP = "=" * 120

RISK_BASE = {
    "stake_percent":      2.0,
    "max_stake":          20.0,
    "min_stake":          0.35,
    "daily_loss_limit":   100.0,
    "use_kelly":          False,
    "max_open_contracts": 1,
}


def payout(gr: float, ht: int) -> float:
    return (1 + gr) ** ht - 1


def be(gr: float, ht: int) -> float:
    return 1.0 / (1.0 + payout(gr, ht))


def run_combo(
    ticks_split: list[list[dict]],
    symbol: str,
    gr: float,
    ht: int,
    calm: float,
    spike: float,
) -> dict:
    pay  = payout(gr, ht)
    brk  = be(gr, ht)
    bar  = LIVE_BARRIERS[symbol][gr]

    strategy_cfg = {
        "symbol_type":         "crash",   # not used in CalmAccu but required
        "long_atr_period":     50,
        "short_atr_period":    10,
        "spike_mult":          spike,
        "spike_cooldown":      10,
        "calm_atr_ratio":      calm,
        "entry_cooldown":      5,
        "hold_ticks":          ht,
        "growth_rate":         gr,
        "barrier_pct":         bar,
        "calm_barrier_mult":   0.0,
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
        "gr": gr, "ht": ht, "calm": calm, "spike": spike,
        "pay": pay * 100, "be": brk * 100, "bar": bar,
        "wr": wr, "ev": ev, "trades": trades,
        "passes": passes, "n_windows": len(ticks_split),
    }


def sweep_symbol(symbol: str) -> None:
    print()
    print(SEP)
    print(f"  {symbol}  |  {WINDOWS}x{WINDOW_SIZE:,} ticks")
    for gr in GROWTH_RATES:
        print(f"    Barrier gr={gr*100:.0f}%: {LIVE_BARRIERS[symbol][gr]:.2e}")
    print(SEP)

    ticks = fetch_ticks(symbol, count=TOTAL_TICKS)
    ticks = ticks[-TOTAL_TICKS:]
    if len(ticks) < WINDOW_SIZE:
        print(f"  Not enough ticks ({len(ticks)}), skipping.")
        return
    print(f"  Using {len(ticks):,} ticks\n")

    splits = [ticks[i * WINDOW_SIZE:(i + 1) * WINDOW_SIZE] for i in range(WINDOWS)]

    results = []
    for gr in GROWTH_RATES:
        for ht in HOLD_RANGE:
            for calm in CALM_RATIOS:
                for spike in SPIKE_MULTS:
                    r = run_combo(splits, symbol, gr, ht, calm, spike)
                    results.append(r)

    results.sort(key=lambda x: (x["passes"], x["ev"]), reverse=True)
    n_win = WINDOWS

    hdr = (f"  {'gr':>4}  {'ht':>3}  {'calm':>5}  {'spk':>5}  {'pay%':>6}  "
           f"{'BE%':>6}  {'WR%':>6}  {'EV%':>8}  {'trades':>6}  pass")
    print(hdr)
    print(f"  {'-'*4}  {'-'*3}  {'-'*5}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*6}  ----")

    shown = 0
    for r in results:
        if r["trades"] < MIN_TRADES:
            continue
        if shown >= 20 and r["ev"] <= 0:
            break
        flag = f" ***" if r["passes"] >= n_win and r["ev"] > 0 else (
               f" *"   if r["passes"] >= n_win - 1 and r["ev"] > 0 else "")
        print(f"  {r['gr']*100:>3.0f}%  {r['ht']:>3}  {r['calm']:>5.2f}  {r['spike']:>5.0f}  "
              f"{r['pay']:>6.1f}%  {r['be']:>6.1f}%  {r['wr']:>6.1f}%  "
              f"{r['ev']:>+8.3f}%  {r['trades']:>6}  {r['passes']}/{n_win}{flag}")
        shown += 1

    valid = [r for r in results if r["trades"] >= MIN_TRADES]
    if valid:
        b = valid[0]
        print(f"\n  Best: gr={b['gr']*100:.0f}%  ht={b['ht']}  calm={b['calm']}  spike={b['spike']:.0f}x  "
              f"WR={b['wr']:.1f}%  BE={b['be']:.1f}%  EV={b['ev']:+.3f}%  passes={b['passes']}/{b['n_windows']}")
    else:
        print("  (no combos with enough trades)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Forex ACCU sweep")
    parser.add_argument("--symbol", default=None, help="Single symbol to test (default: all)")
    args = parser.parse_args()

    symbols = [args.symbol.upper()] if args.symbol else SYMBOLS

    print()
    print("  FOREX ACCUMULATOR RESEARCH")
    print("  NOTE: barriers are placeholders — run check_contracts.py on VPS to get exact values")

    for sym in symbols:
        if sym not in LIVE_BARRIERS:
            print(f"  Unknown symbol: {sym}")
            continue
        sweep_symbol(sym)

    print()
    print(SEP)
    print("  *** = all windows pass AND EV>0   * = all-but-one windows pass AND EV>0")
    print("  calm = calm_atr_ratio (short ATR must be < calm × long ATR to enter)")
    print("  spk  = spike_mult (ATR multiple that counts as a spike)")


if __name__ == "__main__":
    main()
