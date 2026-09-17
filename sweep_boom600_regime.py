"""
Crash/Boom Regime Filter Research — all symbols.

All disabled symbols (BOOM600, BOOM500, JD75) and active ones failed on raw
WR vs break-even. This script asks: is there a time-of-day or volatility-regime
slice where ACCU actually clears break-even across walk-forward windows?

Approach:
  1. Load 60k ticks per symbol (2×30k windows).
  2. Slice ticks by UTC hour bucket (00-04, 04-08, 08-12, 12-16, 16-20, 20-24).
  3. Within each bucket, sweep hold_ticks × growth_rate × settle_ticks.
  4. Separately, split by calm vs volatile ATR regime and sweep same params.

Usage:
  python sweep_boom600_regime.py                         # all symbols, both modes
  python sweep_boom600_regime.py --symbol BOOM600        # one symbol
  python sweep_boom600_regime.py --group mid             # fast/mid/slow group
  python sweep_boom600_regime.py --mode hours            # time-of-day only
  python sweep_boom600_regime.py --mode vol              # volatility regime only
  python sweep_boom600_regime.py --symbol CRASH500 --mode hours
"""

import argparse
import sys
from datetime import datetime, timezone

from loguru import logger

from src.backtest.engine import BacktestEngine
from src.data.history import fetch_ticks
from src.strategies.crash_boom_recoil import CrashBoomRecoilStrategy

logger.remove()
logger.add(sys.stderr, level="ERROR", format="{time:HH:mm:ss} | {level} | {message}")

# ── Symbol registry (same barriers as sweep_all_crash_boom.py) ───────────────
LIVE_BARRIERS = {
    "CRASH1000": {0.04: 2.36e-6, 0.05: 2.26e-6},
    "CRASH900":  {0.04: 2.61e-6, 0.05: 2.50e-6},
    "CRASH600":  {0.04: 3.92e-6, 0.05: 3.76e-6},
    "CRASH500":  {0.04: 4.72e-6, 0.05: 4.55e-6},
    "CRASH150N": {0.04: 1.61e-6, 0.05: 1.54e-6},
    "CRASH50":   {0.04: 2.15e-6, 0.05: 2.03e-6},
    "BOOM1000":  {0.04: 2.35e-6, 0.05: 2.25e-6},
    "BOOM900":   {0.04: 2.62e-6, 0.05: 2.51e-6},
    "BOOM600":   {0.04: 3.94e-6, 0.05: 3.77e-6},
    "BOOM500":   {0.04: 4.73e-6, 0.05: 4.54e-6},
    "BOOM150N":  {0.04: 1.61e-6, 0.05: 1.54e-6},
    "BOOM50":    {0.04: 2.15e-6, 0.05: 2.03e-6},
}

SYMBOLS = {
    "CRASH50":   {"symbol_type": "crash", "freq": 50,   "hold_range": [3, 4, 5, 6, 8]},
    "BOOM50":    {"symbol_type": "boom",  "freq": 50,   "hold_range": [3, 4, 5, 6, 8]},
    "CRASH150N": {"symbol_type": "crash", "freq": 150,  "hold_range": [5, 8, 10, 12, 15]},
    "BOOM150N":  {"symbol_type": "boom",  "freq": 150,  "hold_range": [5, 8, 10, 12, 15]},
    "CRASH600":  {"symbol_type": "crash", "freq": 600,  "hold_range": [6, 8, 10, 12, 15]},
    "BOOM600":   {"symbol_type": "boom",  "freq": 600,  "hold_range": [6, 8, 10, 12, 15]},
    "CRASH500":  {"symbol_type": "crash", "freq": 500,  "hold_range": [6, 8, 10, 12, 15]},
    "BOOM500":   {"symbol_type": "boom",  "freq": 500,  "hold_range": [6, 8, 10, 12, 15]},
    "CRASH1000": {"symbol_type": "crash", "freq": 1000, "hold_range": [6, 8, 10, 12]},
    "BOOM1000":  {"symbol_type": "boom",  "freq": 1000, "hold_range": [12, 15, 20]},
    "CRASH900":  {"symbol_type": "crash", "freq": 900,  "hold_range": [8, 12, 15, 20]},
    "BOOM900":   {"symbol_type": "boom",  "freq": 900,  "hold_range": [8, 12, 15, 20]},
}

GROUPS = {
    "fast": ["CRASH50", "BOOM50", "CRASH150N", "BOOM150N"],
    "mid":  ["CRASH600", "BOOM600", "CRASH500", "BOOM500"],
    "slow": ["CRASH1000", "BOOM1000", "CRASH900", "BOOM900"],
    "all":  list(SYMBOLS.keys()),
}

GROWTH_RATES = [0.04, 0.05]
SETTLE_TICKS = [0, 3]
WINDOWS      = 2
WINDOW_SIZE  = 30_000
TOTAL_TICKS  = WINDOWS * WINDOW_SIZE
ATR_PERIOD   = 50
SPIKE_MULT   = 12.0
COOLDOWN     = 5
MIN_TRADES   = 3
SEP = "=" * 110

RISK_BASE = {
    "stake_percent":      2.0,
    "max_stake":          20.0,
    "min_stake":          0.35,
    "daily_loss_limit":   100.0,
    "use_kelly":          False,
    "max_open_contracts": 1,
}

HOUR_BUCKETS = [
    ("00-04", 0,  4),
    ("04-08", 4,  8),
    ("08-12", 8,  12),
    ("12-16", 12, 16),
    ("16-20", 16, 20),
    ("20-24", 20, 24),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def payout(gr: float, ht: int) -> float:
    return (1 + gr) ** ht - 1


def be(gr: float, ht: int) -> float:
    return 1.0 / (1.0 + payout(gr, ht))


def _tick_hour(tick: dict) -> int:
    epoch = tick.get("epoch", tick.get("epoch_ms", 0))
    if epoch > 1e12:
        epoch /= 1000.0
    return datetime.fromtimestamp(epoch, tz=timezone.utc).hour


def _compute_atr(prices: list[float], period: int) -> float | None:
    hist = prices[-(period + 1):]
    if len(hist) < period + 1:
        return None
    ranges = [abs(hist[i] - hist[i - 1]) for i in range(1, len(hist))]
    return sum(ranges) / len(ranges) if ranges else None


# ── Core sweep ────────────────────────────────────────────────────────────────

def run_combo(
    ticks_split: list[list[dict]],
    symbol: str,
    meta: dict,
    gr: float,
    ht: int,
    st: int,
) -> dict:
    pay = payout(gr, ht)
    brk = be(gr, ht)
    bar = LIVE_BARRIERS[symbol][gr]

    strategy_cfg = {
        "symbol_type":              meta["symbol_type"],
        "spike_mult":               SPIKE_MULT,
        "atr_period":               ATR_PERIOD,
        "cooldown_ticks":           COOLDOWN,
        "loss_cooldown":            2,
        "hold_ticks":               ht,
        "growth_rate":              gr,
        "barrier_pct":              bar,
        "confirm_threshold":        0.5,
        "settle_ticks":             st,
        "volatility_filter_window": 0,
        "trend_filter_window":      0,
        "use_binary":               False,
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
        "pay": pay * 100, "be": brk * 100, "bar": bar,
        "wr": wr, "ev": ev, "trades": trades,
        "passes": passes, "n_windows": len(ticks_split),
    }


def sweep_bucket(
    label: str,
    ticks_split: list[list[dict]],
    symbol: str,
    meta: dict,
) -> None:
    total = sum(len(s) for s in ticks_split)
    print(f"\n  Bucket: {label}  |  ticks={total:,}")
    print(f"  {'st':>3}  {'gr':>4}  {'ht':>3}  {'pay%':>6}  {'BE%':>6}  {'WR%':>6}  {'EV%':>8}  {'trades':>6}  pass")
    print(f"  {'-'*3}  {'-'*4}  {'-'*3}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*6}  ----")

    results = []
    for st in SETTLE_TICKS:
        for gr in GROWTH_RATES:
            for ht in meta["hold_range"]:
                r = run_combo(ticks_split, symbol, meta, gr, ht, st)
                results.append(r)

    results.sort(key=lambda x: (x["passes"], x["ev"]), reverse=True)
    n_win = results[0]["n_windows"] if results else 1

    for r in results:
        if r["trades"] < MIN_TRADES:
            continue
        flag = " ***" if r["passes"] >= n_win and r["ev"] > 0 else (
               " *"   if r["passes"] >= max(1, n_win - 1) and r["ev"] > 0 else "")
        print(f"  {r['st']:>3}  {r['gr']*100:>3.0f}%  {r['ht']:>3}  "
              f"{r['pay']:>6.1f}%  {r['be']:>6.1f}%  "
              f"{r['wr']:>6.1f}%  {r['ev']:>+8.3f}%  {r['trades']:>6}  "
              f"{r['passes']}/{n_win}{flag}")

    valid = [r for r in results if r["trades"] >= MIN_TRADES]
    if valid:
        b = valid[0]
        print(f"\n  Best: st={b['st']}t  gr={b['gr']*100:.0f}%  ht={b['ht']}  "
              f"WR={b['wr']:.1f}%  BE={b['be']:.1f}%  EV={b['ev']:+.3f}%  "
              f"passes={b['passes']}/{b['n_windows']}")
    else:
        print("  (no combos with enough trades)")


# ── Mode: hours ───────────────────────────────────────────────────────────────

def mode_hours(ticks: list[dict], symbol: str, meta: dict) -> None:
    print(f"\n  MODE: time-of-day  ({len(ticks):,} ticks sliced by UTC hour)")
    print(SEP)
    for label, h0, h1 in HOUR_BUCKETS:
        bucket = [t for t in ticks if h0 <= _tick_hour(t) < h1]
        if not bucket:
            print(f"\n  Bucket: UTC {label}  — no ticks")
            continue
        mid = len(bucket) // 2
        sweep_bucket(f"UTC {label}", [bucket[:mid], bucket[mid:]], symbol, meta)


# ── Mode: vol ─────────────────────────────────────────────────────────────────

def mode_vol(ticks: list[dict], symbol: str, meta: dict) -> None:
    print(f"\n  MODE: volatility regime  ({len(ticks):,} ticks)")
    print(SEP)
    SHORT_ATR = 20
    LONG_ATR  = 100

    prices = [float(t["quote"]) for t in ticks]
    calm_idx   = []
    active_idx = []

    for i in range(LONG_ATR + 1, len(prices)):
        s = _compute_atr(prices[:i+1], SHORT_ATR)
        l = _compute_atr(prices[:i+1], LONG_ATR)
        if s is None or l is None or l == 0:
            continue
        (calm_idx if s < l else active_idx).append(i)

    for label, idx in [("calm (short<long ATR)", calm_idx), ("active (short≥long ATR)", active_idx)]:
        bucket = [ticks[i] for i in idx]
        if not bucket:
            print(f"\n  Regime: {label}  — no ticks")
            continue
        mid = len(bucket) // 2
        sweep_bucket(label, [bucket[:mid], bucket[mid:]], symbol, meta)


# ── Symbol entry point ────────────────────────────────────────────────────────

def sweep_symbol(symbol: str, mode: str) -> None:
    meta = SYMBOLS[symbol]
    print()
    print(SEP)
    print(f"  {symbol}  |  spike ~{meta['freq']}t  |  {WINDOWS}×{WINDOW_SIZE:,} ticks")
    for gr in GROWTH_RATES:
        print(f"    Barrier gr={gr*100:.0f}%: {LIVE_BARRIERS[symbol][gr]:.2e}")
    print(SEP)

    ticks = fetch_ticks(symbol, count=TOTAL_TICKS)
    ticks = ticks[-TOTAL_TICKS:]
    if len(ticks) < 1000:
        print(f"  Not enough ticks ({len(ticks)}), skipping.")
        return
    print(f"  Loaded {len(ticks):,} ticks")

    if mode in ("hours", "both"):
        mode_hours(ticks, symbol, meta)

    if mode in ("vol", "both"):
        mode_vol(ticks, symbol, meta)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Crash/Boom regime filter research")
    parser.add_argument("--symbol", default=None, help="Single symbol (default: all)")
    parser.add_argument("--group",  default="all", choices=list(GROUPS.keys()))
    parser.add_argument("--mode",   default="both", choices=["hours", "vol", "both"])
    args = parser.parse_args()

    if args.symbol:
        symbols = [args.symbol.upper()]
    else:
        symbols = GROUPS[args.group]

    print()
    print("  CRASH/BOOM REGIME FILTER RESEARCH")
    print("  Looks for time-of-day or calm/volatile windows where ACCU clears break-even")

    for sym in symbols:
        if sym not in SYMBOLS:
            print(f"  Unknown symbol: {sym}")
            continue
        sweep_symbol(sym, args.mode)

    print()
    print(SEP)
    print("  *** = all windows pass AND EV > 0   * = all-but-one windows pass AND EV > 0")
    print("  st=settle_ticks  gr=growth_rate  ht=hold_ticks")


if __name__ == "__main__":
    main()
