"""
BOOM1000 ACCU parameter sweep.

Sweeps hold_ticks × growth_rate × spike_mult × confirm_threshold
and ranks combinations by net EV = (WR - BE) × payout per trade.

Usage:
  python sweep_boom1000.py                 # full sweep
  python sweep_boom1000.py --fresh         # re-download tick data
  python sweep_boom1000.py --windows 4    # walk-forward windows (default 4)
  python sweep_boom1000.py --count 15000  # ticks per window (default 15000)
"""

import argparse
import sys

from loguru import logger

from src.backtest.engine import BacktestEngine
from src.data.history import fetch_ticks
from src.strategies.crash_boom_recoil import CrashBoomRecoilStrategy

logger.remove()
logger.add(sys.stderr, level="WARNING", format="{time:HH:mm:ss} | {level} | {message}")

SYMBOL      = "BOOM1000"
BARRIER_PCT = 2.35e-6    # real Deriv ACCU barrier at 4% growth (from check_contracts.py)
ATR_PERIOD  = 50
COOLDOWN    = 5

# Parameter sweep ranges
HOLD_TICKS_VALUES     = [6, 8, 10, 12, 15, 20]
GROWTH_RATE_VALUES    = [0.03, 0.04]
SPIKE_MULT_VALUES     = [12.0, 15.0, 20.0, 25.0]
CONFIRM_THRESH_VALUES = [0.5, 1.0, 2.0]

RISK_BASE = {
    "stake_percent":      2.0,
    "max_stake":          20.0,
    "min_stake":          0.35,
    "daily_loss_limit":   100.0,
    "use_kelly":          False,
    "max_open_contracts": 1,
    "barrier_pct":        BARRIER_PCT,
}

SEP  = "=" * 90
THIN = "-" * 90


def run_combo(ticks, windows, window_size, hold_ticks, growth_rate, spike_mult, confirm_threshold):
    payout   = (1 + growth_rate) ** hold_ticks - 1
    be       = 1.0 / (1.0 + payout)
    strategy_cfg = {
        "symbol_type":       "boom",
        "spike_mult":        spike_mult,
        "atr_period":        ATR_PERIOD,
        "cooldown_ticks":    COOLDOWN,
        "loss_cooldown":     0,
        "hold_ticks":        hold_ticks,
        "growth_rate":       growth_rate,
        "barrier_pct":       BARRIER_PCT,
        "confirm_threshold": confirm_threshold,
    }
    risk_cfg = {**RISK_BASE, "payout_pct": payout}

    wins_total = losses_total = trades_total = 0
    net_total  = 0.0
    passes     = 0

    for w in range(windows):
        seg    = ticks[w * window_size: (w + 1) * window_size]
        engine = BacktestEngine(
            strategy_cfg=strategy_cfg,
            risk_cfg=risk_cfg,
            payout_pct=payout,
            strategy_class=CrashBoomRecoilStrategy,
        )
        r = engine.run(seg, starting_balance=1000.0)
        if r.win_rate >= be * 100 and r.total_trades >= 5:
            passes += 1
        wins_total   += r.wins
        trades_total += r.total_trades
        net_total    += r.net_profit

    wr   = wins_total / trades_total * 100 if trades_total > 0 else 0.0
    edge = wr - be * 100
    ev   = (edge / 100) * payout * 100   # edge × payout in %, EV per $1 stake

    return {
        "hold_ticks":        hold_ticks,
        "growth_rate":       growth_rate,
        "spike_mult":        spike_mult,
        "confirm_threshold": confirm_threshold,
        "payout":            payout * 100,
        "be":                be * 100,
        "wr":                wr,
        "edge":              edge,
        "ev_pct":            ev,
        "trades":            trades_total,
        "net":               net_total,
        "passes":            passes,
        "windows":           windows,
    }


def main():
    parser = argparse.ArgumentParser(description="BOOM1000 ACCU parameter sweep")
    parser.add_argument("--fresh",   action="store_true")
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--count",   type=int, default=15000)
    args = parser.parse_args()

    total_ticks = args.count * args.windows
    print(f"\nBOOM1000 ACCU Parameter Sweep  |  {args.windows} windows x {args.count:,} ticks")
    print(f"barrier_pct={BARRIER_PCT:.2e}  |  symbol_type=boom")
    print(f"Fetching {total_ticks:,} ticks ...")
    print(SEP)

    ticks = fetch_ticks(SYMBOL, count=total_ticks, fresh=args.fresh)
    if len(ticks) < args.count:
        print(f"ERROR: Only {len(ticks)} ticks available.")
        return
    ticks = ticks[-args.windows * args.count:]

    combos = [
        (ht, gr, sm, ct)
        for ht in HOLD_TICKS_VALUES
        for gr in GROWTH_RATE_VALUES
        for sm in SPIKE_MULT_VALUES
        for ct in CONFIRM_THRESH_VALUES
    ]
    print(f"Running {len(combos)} combinations ...\n")

    results = []
    for i, (ht, gr, sm, ct) in enumerate(combos, 1):
        r = run_combo(ticks, args.windows, args.count, ht, gr, sm, ct)
        results.append(r)
        if i % 10 == 0:
            print(f"  ... {i}/{len(combos)} done", flush=True)

    # Sort by passes desc, then ev_pct desc
    results.sort(key=lambda x: (x["passes"], x["ev_pct"]), reverse=True)

    print()
    print(SEP)
    print(f"  {'ht':>2}  {'gr':>3}  {'sm':>4}  {'ct':>3}  {'payout':>6}  {'BE%':>5}  "
          f"{'WR%':>5}  {'edge%':>6}  {'EV%/trade':>9}  {'trades':>6}  {'net$':>8}  pass")
    print(THIN)

    for r in results[:40]:    # top 40
        flag = " <<" if r["passes"] == r["windows"] and r["ev_pct"] > 3 else ""
        print(
            f"  {r['hold_ticks']:>2}  {r['growth_rate']*100:>3.0f}%  {r['spike_mult']:>4.0f}  "
            f"{r['confirm_threshold']:>3.1f}  {r['payout']:>6.1f}%  {r['be']:>5.1f}%  "
            f"{r['wr']:>5.1f}%  {r['edge']:>+6.2f}%  {r['ev_pct']:>+9.3f}%  "
            f"{r['trades']:>6}  {r['net']:>+8.2f}  {r['passes']}/{r['windows']}{flag}"
        )

    print(THIN)
    best = results[0]
    print(f"\nBest: hold_ticks={best['hold_ticks']} growth_rate={best['growth_rate']*100:.0f}% "
          f"spike_mult={best['spike_mult']:.0f} confirm_threshold={best['confirm_threshold']:.1f}")
    print(f"      WR={best['wr']:.1f}%  BE={best['be']:.1f}%  edge={best['edge']:+.2f}%  "
          f"EV={best['ev_pct']:+.3f}%/trade  passes={best['passes']}/{best['windows']}")

    # Summary by hold_ticks (averaged over other params)
    print()
    print(SEP)
    print("Hold-ticks summary (best combo per hold_ticks):")
    print(THIN)
    seen = set()
    for r in results:
        if r["hold_ticks"] not in seen:
            seen.add(r["hold_ticks"])
            print(
                f"  ht={r['hold_ticks']:>2}  gr={r['growth_rate']*100:.0f}%  "
                f"sm={r['spike_mult']:.0f}  ct={r['confirm_threshold']:.1f}  "
                f"WR={r['wr']:.1f}%  BE={r['be']:.1f}%  "
                f"EV={r['ev_pct']:+.3f}%/trade  passes={r['passes']}/{r['windows']}"
            )

    print()
    print("Column guide:")
    print("  ht=hold_ticks  gr=growth_rate  sm=spike_mult  ct=confirm_threshold")
    print("  EV%/trade = (WR - BE) x payout  -- dollar edge per $1 stake per trade")


if __name__ == "__main__":
    main()
