"""
Jump Indices Binary Research — JD10, JD25, JD50, JD75, JD100.

JD indices have bidirectional algorithmic spikes (both up and down). The
JDBinaryStrategy fires a directional binary after each spike settles:
  down-spike → BUY_RISE (CALL)
  up-spike   → BUY_FALL (PUT)

JD payout on Deriv is typically 92% for JD10/JD25/JD50/JD75, 96% for JD100.
  BE at 92%  = 52.1%
  BE at 96%  = 51.0%

JD50 and JD75 were previously run live and disabled (insufficient edge found).
This sweep covers all 5 symbols to find if any config passes walk-forward.

Sweep parameters:
  spike_mult     8, 10, 12, 15
  settle_ticks   0, 1, 2
  contract_dur   1, 3, 5  ticks
  require_recoil True, False

Walk-forward: 3 × 5k ticks.

Usage:
  python sweep_jump_indices.py
  python sweep_jump_indices.py --symbol JD25
"""

import argparse
import sys

from loguru import logger

from src.backtest.engine import BacktestEngine
from src.data.history import fetch_ticks
from src.strategies.jd_binary import JDBinaryStrategy

logger.remove()
logger.add(sys.stderr, level="ERROR", format="{time:HH:mm:ss} | {level} | {message}")

# Payout per symbol — update from Deriv UI if changed
# JD50/JD75 were previously live (both disabled after losses); included here for research
PAYOUTS = {
    "JD10":  0.92,
    "JD25":  0.92,
    "JD50":  0.92,
    "JD75":  0.92,
    "JD100": 0.96,
}
SYMBOLS = list(PAYOUTS.keys())

SPIKE_MULTS    = [8.0, 10.0, 12.0, 15.0]
SETTLE_OPTIONS = [0, 1, 2]
DURATIONS      = [1, 3, 5]
RECOIL_OPTIONS = [True, False]

ATR_PERIOD   = 30
SPIKE_COOLDOWN = 5
LOSS_COOLDOWN  = 2

WINDOWS     = 3
WINDOW_SIZE = 5_000
TOTAL_TICKS = WINDOWS * WINDOW_SIZE
MIN_TRADES  = 10
SEP = "=" * 120

RISK_BASE = {
    "stake_percent":      2.0,
    "max_stake":          20.0,
    "min_stake":          0.35,
    "daily_loss_limit":   100.0,
    "use_kelly":          False,
    "max_open_contracts": 1,
}


def run_combo(
    ticks_split: list[list[dict]],
    payout_pct: float,
    spike_mult: float,
    settle: int,
    dur: int,
    recoil: bool,
) -> dict:
    brk = 1.0 / (1.0 + payout_pct)

    strategy_cfg = {
        "atr_period":               ATR_PERIOD,
        "spike_mult":               spike_mult,
        "settle_ticks":             settle,
        "spike_cooldown":           SPIKE_COOLDOWN,
        "loss_cooldown":            LOSS_COOLDOWN,
        "contract_duration":        dur,
        "require_recoil_confirm":   recoil,
        "min_spike_ratio":          0.0,
        "use_binary":               True,
    }
    risk_cfg = {**RISK_BASE, "payout_pct": payout_pct}

    wins = losses = trades = passes = 0
    for seg in ticks_split:
        if len(seg) < 50:
            continue
        engine = BacktestEngine(
            strategy_cfg=strategy_cfg,
            risk_cfg=risk_cfg,
            payout_pct=payout_pct,
            strategy_class=JDBinaryStrategy,
        )
        r = engine.run(seg, starting_balance=1000.0)
        if r.total_trades >= MIN_TRADES and r.win_rate >= brk * 100:
            passes += 1
        wins   += r.wins
        losses += r.losses
        trades += r.total_trades

    wr = wins / trades * 100 if trades > 0 else 0.0
    ev = (wr / 100 - brk) * payout_pct * 100
    return {
        "spike": spike_mult, "settle": settle, "dur": dur, "recoil": recoil,
        "pay": payout_pct * 100, "be": brk * 100,
        "wr": wr, "ev": ev, "trades": trades,
        "passes": passes, "n_windows": len(ticks_split),
    }


def sweep_symbol(symbol: str) -> None:
    payout_pct = PAYOUTS[symbol]
    be_pct     = 1.0 / (1.0 + payout_pct) * 100

    print()
    print(SEP)
    print(f"  {symbol}  |  payout={payout_pct*100:.0f}%  BE={be_pct:.1f}%  |  {WINDOWS}x{WINDOW_SIZE:,} ticks")
    print(SEP)

    ticks = fetch_ticks(symbol, count=TOTAL_TICKS)
    ticks = ticks[-TOTAL_TICKS:]
    if len(ticks) < WINDOW_SIZE:
        print(f"  Not enough ticks ({len(ticks)}), skipping.")
        return
    print(f"  Using {len(ticks):,} ticks\n")

    splits = [ticks[i * WINDOW_SIZE:(i + 1) * WINDOW_SIZE] for i in range(WINDOWS)]

    results = []
    for spike in SPIKE_MULTS:
        for settle in SETTLE_OPTIONS:
            for dur in DURATIONS:
                for recoil in RECOIL_OPTIONS:
                    r = run_combo(splits, payout_pct, spike, settle, dur, recoil)
                    results.append(r)

    results.sort(key=lambda x: (x["passes"], x["ev"]), reverse=True)
    n_win = WINDOWS

    hdr = (f"  {'spk':>5}  {'stl':>3}  {'dur':>3}  {'rcl':>5}  "
           f"{'pay%':>5}  {'BE%':>5}  {'WR%':>6}  {'EV%':>8}  {'trades':>6}  pass")
    print(hdr)
    print(f"  {'-'*5}  {'-'*3}  {'-'*3}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*6}  {'-'*8}  {'-'*6}  ----")

    shown = 0
    for r in results:
        if r["trades"] < MIN_TRADES:
            continue
        if shown >= 25 and r["ev"] <= 0:
            break
        flag = f" ***" if r["passes"] >= n_win and r["ev"] > 0 else (
               f" *"   if r["passes"] >= n_win - 1 and r["ev"] > 0 else "")
        rcl_str = "Y" if r["recoil"] else "N"
        print(f"  {r['spike']:>5.0f}  {r['settle']:>3}  {r['dur']:>3}  {rcl_str:>5}  "
              f"{r['pay']:>5.0f}%  {r['be']:>5.1f}%  {r['wr']:>6.1f}%  "
              f"{r['ev']:>+8.3f}%  {r['trades']:>6}  {r['passes']}/{n_win}{flag}")
        shown += 1

    valid = [r for r in results if r["trades"] >= MIN_TRADES]
    if valid:
        b = valid[0]
        rcl_str = "Y" if b["recoil"] else "N"
        print(f"\n  Best: spike={b['spike']:.0f}x  settle={b['settle']}t  dur={b['dur']}t  "
              f"recoil={rcl_str}  WR={b['wr']:.1f}%  BE={b['be']:.1f}%  "
              f"EV={b['ev']:+.3f}%  passes={b['passes']}/{b['n_windows']}")
    else:
        print("  (no combos with enough trades)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Jump Indices binary sweep")
    parser.add_argument("--symbol", default=None, help="Single symbol (default: all)")
    args = parser.parse_args()

    symbols = [args.symbol.upper()] if args.symbol else SYMBOLS

    print()
    print("  JUMP INDICES BINARY RESEARCH — post-spike directional binaries")
    print("  down-spike → BUY_RISE (CALL)   up-spike → BUY_FALL (PUT)")

    for sym in symbols:
        if sym not in PAYOUTS:
            print(f"  Unknown symbol: {sym}")
            continue
        sweep_symbol(sym)

    print()
    print(SEP)
    print("  *** = all windows pass AND EV>0   * = all-but-one pass AND EV>0")
    print("  spk=spike_mult  stl=settle_ticks  dur=contract_duration  rcl=require_recoil_confirm")


if __name__ == "__main__":
    main()
