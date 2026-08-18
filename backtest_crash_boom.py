"""
Walk-forward backtest for Crash/Boom Post-Spike Recoil strategy.

Fetches historical ticks from the Deriv API (cached to data/), runs a
4-window walk-forward validation, and prints a pass/fail verdict per symbol.

Key output: "detected vs expected spikes" shows if spike_mult needs tuning.
  detected ~= expected  ->  spike_mult is well calibrated
  detected >> expected  ->  spike_mult too LOW  (trading noise)
  detected << expected  ->  spike_mult too HIGH (missing real spikes)

Usage:
  python backtest_crash_boom.py                         # all 4 symbols, 60k ticks each
  python backtest_crash_boom.py --symbol CRASH1000      # single symbol
  python backtest_crash_boom.py --count 20000           # ticks per window (default 15000)
  python backtest_crash_boom.py --windows 4             # walk-forward windows (default 4)
  python backtest_crash_boom.py --fresh                 # re-download (ignore cache)
  python backtest_crash_boom.py --spike-mult 25         # override spike_mult for all
"""

import argparse
import sys

from loguru import logger

from src.data.history import fetch_ticks
from src.backtest.engine import BacktestEngine
from src.strategies.crash_boom_recoil import CrashBoomRecoilStrategy

logger.remove()
logger.add(sys.stderr, level="WARNING", format="{time:HH:mm:ss} | {level} | {message}")

# Per-symbol configs. spike_mult: threshold in multiples of pre-spike ATR.
# The strategy uses pre-spike ATR (excludes current tick), so real spikes
# typically appear as 50-500x. A threshold of 15 catches most real ones
# while ignoring the rare outlier normal tick.
# CRASH300 / BOOM300 are not available in all Deriv regions -- add if you have access.

CONFIGS = {
    "CRASH1000": {"symbol_type": "crash", "spike_mult": 15.0, "expected_rate": 1000},
    "CRASH500":  {"symbol_type": "crash", "spike_mult": 15.0, "expected_rate": 500},
    "BOOM1000":  {"symbol_type": "boom",  "spike_mult": 15.0, "expected_rate": 1000},
    "BOOM500":   {"symbol_type": "boom",  "spike_mult": 15.0, "expected_rate": 500},
}

PAYOUT_PCT   = 0.87    # Deriv synthetic binary payout (verify on your account)
STARTING_BAL = 1000.0

RISK_CFG = {
    "stake_percent":      2.0,
    "max_stake":          20.0,
    "min_stake":          0.35,
    "daily_loss_limit":   100.0,   # 100 = effectively disabled for backtesting
    "use_kelly":          False,
    "payout_pct":         PAYOUT_PCT,
    "max_open_contracts": 1,
}

SEP_WIDE = "=" * 70
SEP_THIN = "-" * 70


def _verdict(passes: int, total: int) -> str:
    if passes == total:
        return "ROBUST"
    elif passes >= total * 3 // 4:
        return "MOSTLY OK"
    elif passes >= total // 2:
        return "MARGINAL"
    else:
        return "OVERFIT / NO EDGE"


def run_walk_forward(
    symbol: str,
    sym_cfg: dict,
    num_windows: int,
    window_size: int,
    fresh: bool,
    spike_mult_override,
):
    expected_rate = sym_cfg["expected_rate"]
    atr_period    = 50
    cooldown      = 5
    contract_dur  = 3

    spike_mult = spike_mult_override if spike_mult_override is not None else sym_cfg["spike_mult"]

    strategy_cfg = {
        "symbol_type":       sym_cfg["symbol_type"],
        "spike_mult":        spike_mult,
        "atr_period":        atr_period,
        "cooldown_ticks":    cooldown,
        "loss_cooldown":     0,
        "contract_duration": contract_dur,
        "bar_size":          1,
        "rsi_period":        14,
    }

    total_ticks = window_size * num_windows

    print()
    print(SEP_WIDE)
    print(f"  {symbol}  ({sym_cfg['symbol_type'].upper()})  |  spike_mult={spike_mult}")
    print(f"  expected 1 spike per {expected_rate} ticks")
    print(f"  Fetching {total_ticks:,} ticks ({num_windows} x {window_size:,}) ...")
    print(SEP_THIN)

    try:
        ticks = fetch_ticks(symbol, count=total_ticks, fresh=fresh)
    except RuntimeError as e:
        print(f"  [ERROR] Could not fetch {symbol}: {e}")
        print("  This symbol may not be available on your Deriv account/region.")
        return

    if len(ticks) < window_size:
        print(f"  [WARN] Only {len(ticks)} ticks available -- skipping.")
        return

    ticks = ticks[-num_windows * window_size:]   # use most recent data

    window_results = []
    for w in range(num_windows):
        segment = ticks[w * window_size: (w + 1) * window_size]

        engine = BacktestEngine(
            strategy_cfg=strategy_cfg,
            risk_cfg=RISK_CFG,
            payout_pct=PAYOUT_PCT,
            strategy_class=CrashBoomRecoilStrategy,
        )
        result = engine.run(segment, starting_balance=STARTING_BAL)

        be          = result.breakeven_win_rate
        expected_n  = window_size / expected_rate
        spike_ratio = (result.total_trades / expected_n) if expected_n > 0 else 0
        passes      = result.win_rate >= be and result.total_trades >= 5
        verdict_str = "PASS" if passes else "FAIL"

        ratio_warn = ""
        if spike_ratio < 0.5:
            ratio_warn = f" [spike_mult too HIGH -- only {spike_ratio:.1f}x expected]"
        elif spike_ratio > 2.5:
            ratio_warn = f" [spike_mult too LOW  -- {spike_ratio:.1f}x expected, detecting noise]"

        print(
            f"  W{w + 1}: {result.total_trades:>4} trades"
            f" (~{expected_n:.0f} exp, {spike_ratio:.1f}x) |"
            f" WR {result.win_rate:>5.1f}% (BE {be:.1f}%) |"
            f" Net {result.net_profit:>+7.2f} | {verdict_str}{ratio_warn}"
        )
        window_results.append(passes)

    passes_total = sum(window_results)
    verdict      = _verdict(passes_total, num_windows)

    print(SEP_THIN)
    print(f"  => {passes_total}/{num_windows} windows pass | {verdict}")
    print(
        f"     config_hint: spike_mult={spike_mult}, atr_period={atr_period},"
        f" cooldown={cooldown}, contract_dur={contract_dur}t"
    )

    if verdict == "ROBUST":
        print(f"  ACTION: enable {symbol} in config.yaml with crash_boom_recoil strategy")
    elif verdict in ("MOSTLY OK", "MARGINAL"):
        print("  ACTION: tune spike_mult and re-run before enabling")
    else:
        print("  ACTION: no exploitable edge found -- leave disabled")


def main():
    parser = argparse.ArgumentParser(description="Crash/Boom Recoil Strategy Backtest")
    parser.add_argument("--symbol",     default=None,
                        help="Single symbol to test (e.g. CRASH1000)")
    parser.add_argument("--count",      type=int, default=15000,
                        help="Ticks per window (default 15000)")
    parser.add_argument("--windows",    type=int, default=4,
                        help="Number of walk-forward windows (default 4)")
    parser.add_argument("--fresh",      action="store_true",
                        help="Re-download tick data (ignore cache)")
    parser.add_argument("--spike-mult", type=float, default=None, dest="spike_mult",
                        help="Override spike_mult for all symbols")
    args = parser.parse_args()

    if args.symbol:
        if args.symbol not in CONFIGS:
            print(f"Unknown symbol '{args.symbol}'. Available: {list(CONFIGS)}")
            return
        symbols = {args.symbol: CONFIGS[args.symbol]}
    else:
        symbols = CONFIGS

    be_pct = round(100 / (1 + PAYOUT_PCT), 1)
    print(f"\nCrash/Boom Recoil Backtest -- {args.windows} windows x {args.count:,} ticks")
    print(f"Payout: {PAYOUT_PCT * 100:.0f}%  |  Breakeven WR: {be_pct}%")

    for symbol, cfg in symbols.items():
        run_walk_forward(
            symbol=symbol,
            sym_cfg=cfg,
            num_windows=args.windows,
            window_size=args.count,
            fresh=args.fresh,
            spike_mult_override=args.spike_mult,
        )

    print()
    print(SEP_WIDE)
    print("Done.")
    print("Re-run with --spike-mult N to tune the spike detection threshold.")
    print("Enable passing symbols in config.yaml with strategy_type: crash_boom_recoil")


if __name__ == "__main__":
    main()
