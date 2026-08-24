"""
Test 5% growth rate for active ACCU symbols.

Uses real barrier_pct values fetched from Deriv API (check_contracts.py, 2026-08-24):
  BOOM900   5%: 2.51e-6
  BOOM1000  5%: 2.25e-6
  CRASH1000 5%: 2.27e-6
  CRASH900  5%: 2.50e-6

Walk-forward: 4 x 30,000 ticks per symbol.
Hold-ticks tested: 8, 10, 12, 15.

Usage:
  python test_5pct_growth.py
  python test_5pct_growth.py --fresh
"""

import argparse
import sys

from loguru import logger
from src.data.history import fetch_ticks
from src.backtest.engine import BacktestEngine
from src.strategies.crash_boom_recoil import CrashBoomRecoilStrategy

logger.remove()
logger.add(sys.stderr, level="ERROR", format="{time:HH:mm:ss} | {level} | {message}")

GROWTH_RATE   = 0.05
WINDOWS       = 4
WINDOW_SIZE   = 20_000   # 4 x 20k = 80k (within Deriv's ~86k historical limit)
SPIKE_MULT    = 12.0
CONFIRM_THRESH = 0.5
ATR_PERIOD    = 50
COOLDOWN      = 5

# Real barrier_pcts at 5% growth from Deriv API (2026-08-24)
SYMBOLS = {
    "BOOM900":   {"symbol_type": "boom",  "barrier_pct": 2.51e-6},
    "BOOM1000":  {"symbol_type": "boom",  "barrier_pct": 2.25e-6},
    "CRASH1000": {"symbol_type": "crash", "barrier_pct": 2.27e-6},
    "CRASH900":  {"symbol_type": "crash", "barrier_pct": 2.50e-6},
}

HOLD_TICKS_TO_TEST = [8, 10, 12, 15]

RISK_BASE = {
    "stake_percent":      2.0,
    "max_stake":          20.0,
    "min_stake":          0.35,
    "daily_loss_limit":   100.0,
    "use_kelly":          False,
    "max_open_contracts": 1,
}

SEP  = "=" * 80
THIN = "-" * 80


def payout(ht: int) -> float:
    return (1 + GROWTH_RATE) ** ht - 1


def run_window(ticks, ht, symbol_type, barrier_pct):
    pay = payout(ht)
    be  = 1.0 / (1.0 + pay)

    strategy_cfg = {
        "symbol_type":       symbol_type,
        "spike_mult":        SPIKE_MULT,
        "atr_period":        ATR_PERIOD,
        "cooldown_ticks":    COOLDOWN,
        "loss_cooldown":     0,
        "hold_ticks":        ht,
        "growth_rate":       GROWTH_RATE,
        "barrier_pct":       barrier_pct,
        "confirm_threshold": CONFIRM_THRESH,
    }
    risk_cfg = {**RISK_BASE, "barrier_pct": barrier_pct, "payout_pct": pay}

    engine = BacktestEngine(
        strategy_cfg=strategy_cfg,
        risk_cfg=risk_cfg,
        payout_pct=pay,
        strategy_class=CrashBoomRecoilStrategy,
    )
    r = engine.run(ticks, starting_balance=1000.0)
    won = r.win_rate >= be * 100 and r.total_trades >= 5
    return r.wins, r.losses, r.total_trades, r.net_profit, won, pay, be


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    print(f"\n5% Growth Rate ACCU Sweep  |  {WINDOWS} x {WINDOW_SIZE:,} ticks")
    print(f"spike_mult={SPIKE_MULT}  confirm_threshold={CONFIRM_THRESH}")
    print(SEP)

    for symbol, cfg in SYMBOLS.items():
        symbol_type = cfg["symbol_type"]
        barrier_pct = cfg["barrier_pct"]

        print(f"\n{symbol}  |  5% growth  |  barrier_pct={barrier_pct:.2e}")
        print(THIN)

        total_needed = WINDOWS * WINDOW_SIZE
        ticks = fetch_ticks(symbol, count=total_needed, fresh=args.fresh)
        if len(ticks) < total_needed:
            print(f"  Only {len(ticks)} ticks available (need {total_needed}) — skipping.")
            continue
        ticks = ticks[-total_needed:]

        print(f"  {'ht':>2}  {'payout':>7}  {'BE%':>5}  {'WR%':>6}  "
              f"{'edge%':>7}  {'EV%/trade':>10}  {'trades':>6}  {'net$':>8}  pass")
        print(THIN)

        for ht in HOLD_TICKS_TO_TEST:
            total_wins = total_trades = 0
            total_net  = 0.0
            passes     = 0

            for w in range(WINDOWS):
                seg = ticks[w * WINDOW_SIZE: (w + 1) * WINDOW_SIZE]
                wins, losses, trades, net, passed, pay, be = run_window(
                    seg, ht, symbol_type, barrier_pct
                )
                total_wins   += wins
                total_trades += trades
                total_net    += net
                if passed:
                    passes += 1

            pay = payout(ht)
            be  = 1.0 / (1.0 + pay)
            wr  = total_wins / total_trades * 100 if total_trades else 0.0
            edge = wr - be * 100
            ev   = (edge / 100) * pay * 100
            flag = " <<" if passes == WINDOWS and ev > 2 else ""
            verdict = f"{passes}/{WINDOWS} ROBUST" if passes == WINDOWS else f"{passes}/{WINDOWS}"

            print(
                f"  {ht:>2}  {pay*100:>6.1f}%  {be*100:>5.1f}%  {wr:>5.1f}%  "
                f"{edge:>+6.2f}%  {ev:>+9.3f}%  {total_trades:>6}  "
                f"{total_net:>+8.2f}  {verdict}{flag}"
            )

        # Comparison: show current config win at 4% growth for reference
        print(THIN)
        ht_curr = {"BOOM900": 10, "BOOM1000": 20, "CRASH1000": 8, "CRASH900": 10}.get(symbol, 10)
        pay4    = (1.04 ** ht_curr) - 1
        pay5    = (1.05 ** ht_curr) - 1
        print(f"  Reference: current ht={ht_curr} at 4% -> win=${pay4:.3f}  |  5% -> win=${pay5:.3f}  "
              f"(+{(pay5-pay4)/pay4*100:.1f}%)")

    print(f"\n{SEP}")
    print(f"  ACCU win = stake x ((1 + growth_rate)^hold_ticks - 1)")
    print(f"  EV%/trade = (WR - BE) x payout")
    print(f"  Barriers from Deriv API at growth=5% (2026-08-24)")


if __name__ == "__main__":
    main()
