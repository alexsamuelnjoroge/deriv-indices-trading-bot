"""
CRASH50 extended hold sweep: gr=4-5%, hold=8-20, settle=0/3.

BOOM50 sweep showed EV keeps rising with hold (hold=20 → EV=+53.6%, 4/4 ROBUST).
Testing whether CRASH50 has the same pattern at gr=5% — currently live at hold=12.

Usage:
    python sweep_crash50_extended.py
"""

import sys

from loguru import logger

from src.backtest.engine import BacktestEngine
from src.data.history import fetch_ticks
from src.strategies.crash_boom_recoil import CrashBoomRecoilStrategy

logger.remove()
logger.add(sys.stderr, level="ERROR", format="{time:HH:mm:ss} | {level} | {message}")

# ASCII-safe output to avoid cp1252 issues on Windows when piping
LIVE_BARRIERS = {
    0.04: 2.15e-6,
    0.05: 2.03e-6,
}

SYMBOL       = "CRASH50"
SYMBOL_TYPE  = "crash"
FREQ         = 50
GROWTH_RATES = [0.04, 0.05]
HOLD_RANGE   = [8, 10, 12, 15, 18, 20]
SETTLE_TICKS = [0, 3]

SPIKE_MULT    = 12.0
ATR_PERIOD    = 50
COOLDOWN      = 5
LOSS_COOLDOWN = 2
WINDOWS       = 4
WINDOW_SIZE   = 21_500
MIN_TRADES    = 4

RISK_BASE = {
    "stake_percent":      2.0,
    "max_stake":          20.0,
    "min_stake":          0.35,
    "daily_loss_limit":   100.0,
    "use_kelly":          False,
    "max_open_contracts": 1,
}


def payout(gr, ht):
    return (1 + gr) ** ht - 1


def be_wr(gr, ht):
    return 1.0 / (1.0 + payout(gr, ht))


def run_combo(ticks, gr, ht, st):
    pay = payout(gr, ht)
    brk = be_wr(gr, ht)
    bar = LIVE_BARRIERS[gr]

    strategy_cfg = {
        "symbol_type":        SYMBOL_TYPE,
        "spike_mult":         SPIKE_MULT,
        "atr_period":         ATR_PERIOD,
        "cooldown_ticks":     COOLDOWN,
        "loss_cooldown":      LOSS_COOLDOWN,
        "hold_ticks":         ht,
        "growth_rate":        gr,
        "barrier_pct":        bar,
        "confirm_threshold":  0.5,
        "settle_ticks":       st,
        "adaptive_settle":    False,
        "use_binary":         False,
        "min_spike_ratio":    0,
        "cluster_window":     FREQ,
        "max_cluster_spikes": 3,
    }
    risk_cfg = {**RISK_BASE, "payout_pct": pay, "barrier_pct": bar}

    wins = losses = trades = passes = 0
    for w in range(WINDOWS):
        seg = ticks[w * WINDOW_SIZE: (w + 1) * WINDOW_SIZE]
        if len(seg) < 100:
            continue
        engine = BacktestEngine(
            strategy_cfg=strategy_cfg,
            risk_cfg=risk_cfg,
            payout_pct=pay,
            strategy_class=CrashBoomRecoilStrategy,
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
        "gr": gr, "ht": ht, "st": st,
        "bar": bar, "pay": pay * 100, "be": brk * 100,
        "wr": wr, "ev": ev, "trades": trades, "passes": passes,
    }


def main():
    print()
    print("=" * 110)
    print(f"  {SYMBOL}  |  spike every ~{FREQ}t  |  {WINDOWS}x{WINDOW_SIZE:,} ticks ({WINDOWS*WINDOW_SIZE//1000}k total)")
    print(f"  growth_rates={[f'{g*100:.0f}%' for g in GROWTH_RATES]}  "
          f"hold_range={HOLD_RANGE}  settle_ticks={SETTLE_TICKS}")
    print("=" * 110)

    ticks = fetch_ticks(SYMBOL, count=WINDOWS * WINDOW_SIZE)
    ticks = ticks[-(WINDOWS * WINDOW_SIZE):]
    if len(ticks) < WINDOW_SIZE:
        print(f"  Not enough ticks ({len(ticks)}), aborting.")
        return
    print(f"  Using {len(ticks):,} ticks\n")

    results = []
    for st in SETTLE_TICKS:
        for gr in GROWTH_RATES:
            for ht in HOLD_RANGE:
                r = run_combo(ticks, gr, ht, st)
                results.append(r)

    results.sort(key=lambda x: (x["passes"], x["ev"]), reverse=True)

    hdr = (f"  {'gr':>4}  {'ht':>4}  {'st':>3}  {'bar':>9}  "
           f"{'pay%':>6}  {'BE%':>6}  {'WR%':>6}  {'EV%':>9}  {'trades':>6}  pass")
    print(hdr)
    print(f"  {'-'*4}  {'-'*4}  {'-'*3}  {'-'*9}  {'-'*6}  {'-'*6}  "
          f"{'-'*6}  {'-'*9}  {'-'*6}  ----")

    for r in results:
        if r["trades"] < MIN_TRADES:
            continue
        flag = ("  ROBUST" if r["passes"] == WINDOWS and r["ev"] > 0 else
                "  OK"     if r["passes"] >= WINDOWS - 1 and r["ev"] > 0 else "")
        print(f"  {r['gr']*100:>3.0f}%  {r['ht']:>4}  {r['st']:>3}  "
              f"{r['bar']:.2e}  {r['pay']:>6.1f}%  {r['be']:>6.1f}%  "
              f"{r['wr']:>6.1f}%  {r['ev']:>+9.3f}%  {r['trades']:>6}  "
              f"{r['passes']}/{WINDOWS}{flag}")

    print()

    # Summary: best 4/4 ROBUST configs
    robust = [r for r in results if r["passes"] == WINDOWS and r["ev"] > 0
              and r["trades"] >= MIN_TRADES]
    if robust:
        best = robust[0]
        print(f"  Best 4/4 ROBUST: gr={best['gr']*100:.0f}%  hold={best['ht']}  "
              f"settle={best['st']}  WR={best['wr']:.1f}%  BE={best['be']:.1f}%  "
              f"EV={best['ev']:+.3f}%  trades={best['trades']}")
    print()


if __name__ == "__main__":
    main()
