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

# barrier_pct: real ACCU barrier at 4% growth, fetched from Deriv API via check_contracts.py
CONFIGS = {
    "CRASH1000": {"symbol_type": "crash", "spike_mult": 15.0, "expected_rate": 1000, "barrier_pct": 2.35e-6},
    "CRASH900":  {"symbol_type": "crash", "spike_mult": 15.0, "expected_rate": 900,  "barrier_pct": 2.61e-6},
    "CRASH600":  {"symbol_type": "crash", "spike_mult": 15.0, "expected_rate": 600,  "barrier_pct": 3.93e-6},
    "CRASH500":  {"symbol_type": "crash", "spike_mult": 15.0, "expected_rate": 500,  "barrier_pct": 4.73e-6},
    "CRASH300N": {"symbol_type": "crash", "spike_mult": 15.0, "expected_rate": 300,  "barrier_pct": 2.06e-5},
    "CRASH150N": {"symbol_type": "crash", "spike_mult": 15.0, "expected_rate": 150,  "barrier_pct": 1.61e-6},
    "CRASH50":   {"symbol_type": "crash", "spike_mult": 15.0, "expected_rate": 50,   "barrier_pct": 2.15e-6},
    "BOOM1000":  {"symbol_type": "boom",  "spike_mult": 15.0, "expected_rate": 1000, "barrier_pct": 2.35e-6},
    "BOOM900":   {"symbol_type": "boom",  "spike_mult": 15.0, "expected_rate": 900,  "barrier_pct": 2.62e-6},
    "BOOM600":   {"symbol_type": "boom",  "spike_mult": 15.0, "expected_rate": 600,  "barrier_pct": 3.95e-6},
    "BOOM500":   {"symbol_type": "boom",  "spike_mult": 15.0, "expected_rate": 500,  "barrier_pct": 4.73e-6},
    "BOOM300N":  {"symbol_type": "boom",  "spike_mult": 15.0, "expected_rate": 300,  "barrier_pct": 2.08e-5},
    "BOOM150N":  {"symbol_type": "boom",  "spike_mult": 15.0, "expected_rate": 150,  "barrier_pct": 1.61e-6},
    "BOOM50":    {"symbol_type": "boom",  "spike_mult": 15.0, "expected_rate": 50,   "barrier_pct": 2.15e-6},
}

GROWTH_RATE  = 0.04    # ACCU growth rate per tick (matches config.yaml)
HOLD_TICKS   = 8      # ACCU hold period in ticks (matches config.yaml)
BARRIER_PCT  = 2.4e-6 # Default — run check_contracts.py to get the real value per symbol.
STARTING_BAL = 1000.0

# Effective ACCU payout: compound growth over hold period
ACCU_PAYOUT = (1 + GROWTH_RATE) ** HOLD_TICKS - 1  # e.g. 1.04^8 - 1 = 36.9%

RISK_CFG = {
    "stake_percent":      2.0,
    "max_stake":          20.0,
    "min_stake":          0.35,
    "daily_loss_limit":   100.0,   # 100 = effectively disabled for backtesting
    "use_kelly":          False,
    "payout_pct":         ACCU_PAYOUT,
    "max_open_contracts": 1,
    "barrier_pct":        BARRIER_PCT,
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
    trend_filter_window: int = 0,
    volatility_filter_window: int = 0,
    volatility_filter_mult: float = 3.0,
    barrier_pct_override=None,
    confirm_threshold: float = 1.0,
    growth_rate_override: float = GROWTH_RATE,
    hold_ticks_override: int = HOLD_TICKS,
    use_binary: bool = False,
    binary_payout: float = 0.87,
    binary_duration: int = 5,
):
    expected_rate = sym_cfg["expected_rate"]
    atr_period    = 50
    cooldown      = 5
    contract_dur  = 3

    spike_mult  = spike_mult_override if spike_mult_override is not None else sym_cfg["spike_mult"]
    barrier_pct = barrier_pct_override if barrier_pct_override is not None else sym_cfg.get("barrier_pct", BARRIER_PCT)
    growth_rate = growth_rate_override
    hold_ticks  = hold_ticks_override
    accu_payout = (1 + growth_rate) ** hold_ticks - 1

    # Binary mode overrides payout and contract duration
    effective_payout = binary_payout if use_binary else accu_payout
    effective_dur    = binary_duration if use_binary else contract_dur

    strategy_cfg = {
        "symbol_type":              sym_cfg["symbol_type"],
        "spike_mult":               spike_mult,
        "atr_period":               atr_period,
        "cooldown_ticks":           cooldown,
        "loss_cooldown":            0,
        "contract_duration":        effective_dur,
        "bar_size":                 1,
        "rsi_period":               14,
        "hold_ticks":               hold_ticks,
        "growth_rate":              growth_rate,
        "barrier_pct":              barrier_pct,
        "confirm_threshold":        confirm_threshold,
        "trend_filter_window":      trend_filter_window,
        "volatility_filter_window": volatility_filter_window,
        "volatility_filter_mult":   volatility_filter_mult,
        "use_binary":               use_binary,
    }

    total_ticks = window_size * num_windows

    print()
    print(SEP_WIDE)
    be_pct = round(100 / (1 + effective_payout), 1)
    mode_str = f"BINARY {binary_duration}t ({binary_payout*100:.0f}% payout)" if use_binary else f"ACCU growth={growth_rate*100:.0f}%/tick x {hold_ticks}t"
    print(f"  {symbol}  ({sym_cfg['symbol_type'].upper()})  |  spike_mult={spike_mult}  |  barrier_pct={barrier_pct:.2e}")
    print(f"  {mode_str}  |  payout={effective_payout*100:.1f}%  |  BE={be_pct:.1f}%")
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

        risk_cfg = {**RISK_CFG, "barrier_pct": barrier_pct}
        engine = BacktestEngine(
            strategy_cfg=strategy_cfg,
            risk_cfg=risk_cfg,
            payout_pct=effective_payout,
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

    filters = []
    if trend_filter_window:
        filters.append(f"trend_filter={trend_filter_window}t")
    if volatility_filter_window:
        filters.append(f"vol_filter={volatility_filter_window}t(x{volatility_filter_mult})")
    filter_str = ", ".join(filters) if filters else "none"

    print(SEP_THIN)
    print(f"  => {passes_total}/{num_windows} windows pass | {verdict}")
    print(
        f"     config_hint: spike_mult={spike_mult}, atr_period={atr_period},"
        f" cooldown={cooldown}, contract_dur={contract_dur}t"
    )
    print(f"     filters: {filter_str}")

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
    parser.add_argument("--spike-mult",  type=float, default=None,  dest="spike_mult",
                        help="Override spike_mult for all symbols")
    parser.add_argument("--trend-filter", type=int, default=0, dest="trend_filter",
                        help="Pre-spike trend window in ticks; 0=off (default 0). "
                             "CRASH: only enter if trend was up. BOOM: only if down.")
    parser.add_argument("--vol-filter",   type=int, default=0, dest="vol_filter",
                        help="1m volatility window in ticks; 0=off (default 0). "
                             "Skips entry when recent tick range > mult x ATR.")
    parser.add_argument("--vol-mult",     type=float, default=3.0, dest="vol_mult",
                        help="Max allowed range in ATR multiples for vol filter (default 3.0)")
    parser.add_argument("--barrier-pct",      type=float, default=None,  dest="barrier_pct",
                        help="ACCU barrier as fraction of price per tick. "
                             "Get real value from check_contracts.py. Default: per-symbol from CONFIGS.")
    parser.add_argument("--confirm-threshold", type=float, default=1.0,   dest="confirm_threshold",
                        help="Max barrier fraction the confirm tick may consume (default 1.0 = 100%%). "
                             "Set 0 to disable confirmation gate entirely.")
    parser.add_argument("--growth-rate", type=float, default=None, dest="growth_rate",
                        help="ACCU growth rate per tick (default 0.04 = 4%%). "
                             "Higher rate = better payout but tighter barrier.")
    parser.add_argument("--hold-ticks", type=int, default=None, dest="hold_ticks",
                        help="Ticks to hold ACCU before selling (default 8). "
                             "More ticks = higher payout but more knockout exposure.")
    parser.add_argument("--binary", action="store_true",
                        help="Test binary CALL/PUT after spike instead of ACCU. "
                             "CRASH spike -> CALL (recoil up). BOOM spike -> PUT (recoil down). "
                             "Payout 87%%, breakeven 53.5%%.")
    parser.add_argument("--binary-payout", type=float, default=0.87, dest="binary_payout",
                        help="Binary payout fraction (default 0.87 = 87%%).")
    parser.add_argument("--binary-duration", type=int, default=5, dest="binary_duration",
                        help="Binary contract duration in ticks (default 5).")
    args = parser.parse_args()

    if args.symbol:
        if args.symbol not in CONFIGS:
            print(f"Unknown symbol '{args.symbol}'. Available: {list(CONFIGS)}")
            return
        symbols = {args.symbol: CONFIGS[args.symbol]}
    else:
        symbols = CONFIGS

    barrier_override  = args.barrier_pct   # None = per-symbol from CONFIGS
    growth_rate_arg   = args.growth_rate if args.growth_rate is not None else GROWTH_RATE
    hold_ticks_arg    = args.hold_ticks  if args.hold_ticks  is not None else HOLD_TICKS
    accu_payout_arg   = (1 + growth_rate_arg) ** hold_ticks_arg - 1

    if args.binary:
        eff_payout = args.binary_payout
        be_pct     = round(100 / (1 + eff_payout), 1)
        print(f"\nCrash/Boom Binary CALL/PUT Backtest -- {args.windows} windows x {args.count:,} ticks")
        print(f"Binary: {args.binary_duration}t contract  |  Payout: {eff_payout*100:.0f}%  |  Breakeven WR: {be_pct:.1f}%")
        print("Signal: CRASH spike -> CALL (recoil up)  |  BOOM spike -> PUT (recoil down)")
    else:
        eff_payout = accu_payout_arg
        be_pct     = round(100 / (1 + eff_payout), 1)
        print(f"\nCrash/Boom Recoil Backtest -- {args.windows} windows x {args.count:,} ticks")
        print(f"ACCU: growth={growth_rate_arg*100:.0f}%/tick x {hold_ticks_arg}t  |  Win payout: {eff_payout*100:.1f}%  |  Breakeven WR: {be_pct:.1f}%")
    if barrier_override is not None:
        print(f"Barrier override: {barrier_override:.2e} ({barrier_override*100:.6f}% of price per tick)")
    else:
        print("Barrier: per-symbol from CONFIGS (real values from Deriv API)")

    for symbol, cfg in symbols.items():
        run_walk_forward(
            symbol=symbol,
            sym_cfg=cfg,
            num_windows=args.windows,
            window_size=args.count,
            fresh=args.fresh,
            spike_mult_override=args.spike_mult,
            trend_filter_window=args.trend_filter,
            volatility_filter_window=args.vol_filter,
            volatility_filter_mult=args.vol_mult,
            barrier_pct_override=barrier_override,
            confirm_threshold=args.confirm_threshold,
            growth_rate_override=growth_rate_arg,
            hold_ticks_override=hold_ticks_arg,
            use_binary=args.binary,
            binary_payout=args.binary_payout,
            binary_duration=args.binary_duration,
        )

    print()
    print(SEP_WIDE)
    print("Done.")
    print("Re-run with --spike-mult N to tune the spike detection threshold.")
    print("Enable passing symbols in config.yaml with strategy_type: crash_boom_recoil")


if __name__ == "__main__":
    main()
