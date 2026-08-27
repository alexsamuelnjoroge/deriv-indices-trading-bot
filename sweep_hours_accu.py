"""
Hour-of-day analysis for CRASH/BOOM ACCU strategies.

Runs each active symbol's live-configured strategy on the full 60k tick cache,
records each trade's EAT hour and outcome, and reports WR% + EV% per hour.

Hours that consistently fall below breakeven are candidates for blocked_hours_eat.

Usage:
  python sweep_hours_accu.py
  python sweep_hours_accu.py --symbol CRASH500
"""

import argparse
import sys

from loguru import logger

from src.data.history import fetch_ticks
from src.data.tick_store import TickStore
from src.strategies.calm_accu import CalmAccuStrategy
from src.strategies.crash_boom_recoil import CrashBoomRecoilStrategy

logger.remove()
logger.add(sys.stderr, level="ERROR", format="{time:HH:mm:ss} | {level} | {message}")

EAT_OFFSET  = 3     # EAT = UTC+3
TICK_COUNT  = 60_000
MIN_TRADES  = 3     # min trades in an hour bucket to report it

# ── Per-symbol live config ─────────────────────────────────────────────────────
SYMBOLS = [
    {
        "symbol":        "BOOM600",
        "strategy_type": "crash_boom_recoil",
        "growth_rate":   0.04,
        "hold_ticks":    15,
        "barrier_pct":   0.00000394,
        "payout":        0.801,
        "cfg": {
            "symbol_type":       "boom",
            "spike_mult":        12.0,
            "atr_period":        50,
            "cooldown_ticks":    5,
            "loss_cooldown":     2,
            "confirm_threshold": 0.5,
            "barrier_pct":       0.00000394,
        },
    },
    {
        "symbol":        "CRASH150N",
        "strategy_type": "crash_boom_recoil",
        "growth_rate":   0.04,
        "hold_ticks":    10,
        "barrier_pct":   0.00000161,
        "payout":        0.480,
        "cfg": {
            "symbol_type":       "crash",
            "spike_mult":        12.0,
            "atr_period":        50,
            "cooldown_ticks":    5,
            "loss_cooldown":     2,
            "confirm_threshold": 0.5,
            "barrier_pct":       0.00000161,
        },
    },
    {
        "symbol":        "CRASH600",
        "strategy_type": "crash_boom_recoil",
        "growth_rate":   0.04,
        "hold_ticks":    8,
        "barrier_pct":   0.00000392,
        "payout":        0.369,
        "cfg": {
            "symbol_type":       "crash",
            "spike_mult":        12.0,
            "atr_period":        50,
            "cooldown_ticks":    5,
            "loss_cooldown":     2,
            "confirm_threshold": 0.5,
            "barrier_pct":       0.00000392,
            "settle_ticks":      15,
        },
    },
    {
        "symbol":        "CRASH500",
        "strategy_type": "calm_accu",
        "growth_rate":   0.04,
        "hold_ticks":    20,
        "barrier_pct":   0.00000472,
        "payout":        1.191,
        "cfg": {
            "symbol_type":      "crash",
            "long_atr_period":  50,
            "short_atr_period": 10,
            "spike_mult":       15.0,
            "spike_cooldown":   200,
            "calm_atr_ratio":   1.5,
            "entry_cooldown":   20,
            "loss_cooldown":    2,
            "barrier_pct":      0.00000472,
        },
    },
    {
        "symbol":        "BOOM150N",
        "strategy_type": "calm_accu",
        "growth_rate":   0.04,
        "hold_ticks":    10,
        "barrier_pct":   0.00000161,
        "payout":        0.480,
        "cfg": {
            "symbol_type":      "boom",
            "long_atr_period":  50,
            "short_atr_period": 10,
            "spike_mult":       15.0,
            "spike_cooldown":   100,
            "calm_atr_ratio":   0.7,
            "entry_cooldown":   10,
            "loss_cooldown":    2,
            "barrier_pct":      0.00000161,
        },
    },
]


def make_strategy(sym: dict):
    cfg = {**sym["cfg"], "hold_ticks": sym["hold_ticks"], "growth_rate": sym["growth_rate"]}
    if sym["strategy_type"] == "crash_boom_recoil":
        return CrashBoomRecoilStrategy(cfg)
    return CalmAccuStrategy(cfg)


def sim_accu(ticks: list[dict], strategy, hold_ticks: int, barrier_pct: float) -> list[tuple[int, bool]]:
    """
    Run strategy on ticks, simulate ACCU outcomes.
    Returns list of (hour_eat, won) for each trade.
    """
    store = TickStore(max_ticks=500)
    trades: list[tuple[int, bool]] = []
    hold_end = -1  # tick index when current hold expires (no overlapping positions)

    for i, tick in enumerate(ticks):
        store.add(tick)
        sig = strategy.evaluate(store)

        if sig.action == "BUY_ACCU" and i > hold_end:
            hour = (tick["epoch"] // 3600 + EAT_OFFSET) % 24
            # Simulate barrier knockout over hold_ticks ticks
            won = True
            for j in range(i + 1, min(i + 1 + hold_ticks, len(ticks))):
                prev = float(ticks[j - 1]["quote"])
                curr = float(ticks[j]["quote"])
                if prev > 0 and abs(curr - prev) / prev > barrier_pct:
                    won = False
                    break
            trades.append((hour, won))
            hold_end = i + hold_ticks

    return trades


def report(sym: dict, trades: list[tuple[int, bool]]):
    payout  = sym["payout"]
    be_pct  = 100.0 / (1.0 + payout)
    name    = sym["symbol"]
    total   = len(trades)

    SEP = "=" * 72
    print()
    print(SEP)
    print(f"  {name}  |  {sym['strategy_type']}  |  {total} trades  |  "
          f"payout={payout*100:.0f}%  BE={be_pct:.1f}%")
    print(f"  hold={sym['hold_ticks']}t  barrier={sym['barrier_pct']:.2e}  gr={sym['growth_rate']*100:.0f}%")
    print(SEP)

    if total < MIN_TRADES:
        print("  Too few trades to report.")
        return

    # Aggregate by hour
    by_hour: dict[int, list[bool]] = {}
    for h, won in trades:
        by_hour.setdefault(h, []).append(won)

    print(f"  {'Hour (EAT)':>12}  {'Trades':>6}  {'Wins':>5}  {'WR%':>7}  {'EV%':>8}  verdict")
    print(f"  {'-'*12}  {'-'*6}  {'-'*5}  {'-'*7}  {'-'*8}  -------")

    losing_hours = []
    winning_hours = []
    for h in range(24):
        outcomes = by_hour.get(h, [])
        if len(outcomes) < MIN_TRADES:
            continue
        wr    = sum(outcomes) / len(outcomes) * 100
        ev    = (wr - be_pct) / 100 * payout * 100
        label = "BLOCK" if wr < be_pct else ""
        if wr < be_pct:
            losing_hours.append(h)
        else:
            winning_hours.append(h)
        print(f"  {h:>10}:00  {len(outcomes):>6}  {sum(outcomes):>5}  {wr:>6.1f}%  {ev:>+8.3f}%  {label}")

    print(SEP)
    wr_all = sum(w for _, w in trades) / total * 100 if total else 0
    ev_all = (wr_all - be_pct) / 100 * payout * 100
    print(f"  Overall: WR={wr_all:.1f}%  BE={be_pct:.1f}%  EV={ev_all:+.3f}%  ({total} trades)")

    if losing_hours:
        print(f"  Candidate blocked hours (EAT): {sorted(losing_hours)}")
        print(f"  Add to blocked_hours_eat in config.yaml if pattern is consistent.")
    else:
        print("  No hours consistently below breakeven.")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", help="Run only this symbol (e.g. CRASH500)")
    args = parser.parse_args()

    targets = [s for s in SYMBOLS if not args.symbol or s["symbol"] == args.symbol]

    for sym in targets:
        name = sym["symbol"]
        print(f"\nLoading {TICK_COUNT} ticks for {name}...", end=" ", flush=True)
        ticks = fetch_ticks(name, TICK_COUNT)
        print(f"{len(ticks)} ticks loaded")

        strategy = make_strategy(sym)
        trades   = sim_accu(ticks, strategy, sym["hold_ticks"], sym["barrier_pct"])
        report(sym, trades)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
