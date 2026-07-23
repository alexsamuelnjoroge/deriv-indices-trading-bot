"""
Multi-window parameter sweep.
Tests all combinations across N independent windows and ranks by
min_win_rate (the weakest window) — only truly robust params will
stay above breakeven in every window.

Usage:
  python sweep_multi.py                        # R_25, 5 x 5k ticks
  python sweep_multi.py --dur10                # R_25 focused 10-tick sweep, 5 x 10k ticks
  python sweep_multi.py --symbol R_50 --dur10  # sweep for R_50
  python sweep_multi.py --symbol R_75 --dur10  # sweep for R_75
  python sweep_multi.py --fresh                # re-fetch ticks (ignore cache)
"""
import argparse, copy, itertools, json, os
from loguru import logger
logger.remove()

import yaml
from dotenv import load_dotenv
from src.data.history import fetch_ticks
from src.backtest.engine import BacktestEngine
from src.strategies.zscore_reversal import ZScoreReversalStrategy
from src.strategies.streak_reversal import StreakReversalStrategy

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--symbol",    type=str,  default=None,  help="Symbol to sweep (default: first enabled in config)")
parser.add_argument("--dur10",     action="store_true",       help="Focused 10-tick sweep on 50k ticks (5 x 10k windows)")
parser.add_argument("--fresh",     action="store_true",       help="Re-fetch ticks from Deriv (ignore cache)")
parser.add_argument("--windows",   type=int,  default=5,      help="Number of windows (default: 5)")
parser.add_argument("--crossover", action="store_true",       help="Test RSI 50 midline crossover strategy instead of reversal")
parser.add_argument("--momentum",  action="store_true",       help="Test RSI momentum strategy (RSI>OB=BUY_RISE, RSI<OS=BUY_FALL) with asymmetric thresholds")
parser.add_argument("--filters",   action="store_true",       help="Test optional signal filters (slope/price confirm, dual RSI) on top of proven 80/20 reversal base params")
parser.add_argument("--barsize",    action="store_true",       help="Sweep bar_size (tick aggregation) on fixed 80/20 RSI10 base params")
parser.add_argument("--divergence", action="store_true",       help="Sweep divergence lookback on fixed 80/20 RSI10 base params")
parser.add_argument("--adaptive",   action="store_true",       help="Sweep adaptive RSI thresholds (percentile-based) vs fixed 80/20")
parser.add_argument("--depth",      action="store_true",       help="Sweep RSI-depth contract duration (longer contract when RSI is extreme)")
parser.add_argument("--zscore",     action="store_true",       help="Test Z-score mean reversion strategy (alternative to RSI reversal)")
parser.add_argument("--streak",     action="store_true",       help="Test consecutive tick streak reversal strategy")
args = parser.parse_args()

# ── Load base config ───────────────────────────────────────────
with open("config.yaml") as f:
    base_cfg = yaml.safe_load(f)

# Resolve symbol: CLI flag > first enabled in symbols list > fallback
if args.symbol:
    SYMBOL = args.symbol.upper()
else:
    sym_list = base_cfg.get("symbols", [])
    enabled  = [s for s in sym_list if s.get("enabled", True)]
    SYMBOL   = enabled[0]["symbol"] if enabled else "R_25"

PAYOUT   = base_cfg["risk"].get("payout_pct", 0.9232)
BALANCE  = 1000.0
BE       = round((1 / (1 + PAYOUT)) * 100, 1)   # 52.0% at 92.32% payout
N_WINDOWS   = args.windows
WINDOW_SIZE = 10000 if args.dur10 else 5000

total_fetch = N_WINDOWS * WINDOW_SIZE

# ── Fetch ticks ────────────────────────────────────────────────
print(f"Loading {total_fetch:,} ticks ({N_WINDOWS} windows x {WINDOW_SIZE:,} ticks)...")
ticks = fetch_ticks(
    SYMBOL,
    count=total_fetch,
    fresh=args.fresh,
    app_id=os.getenv("DERIV_APP_ID", "1089"),
    api_token=os.getenv("DERIV_API_TOKEN"),
)
print(f"  Got {len(ticks):,} ticks")

window_size = len(ticks) // N_WINDOWS
windows = [ticks[i * window_size: (i + 1) * window_size] for i in range(N_WINDOWS)]

# ── Parameter grids ────────────────────────────────────────────
if args.streak:
    GRID = {
        "streak_length":     [4, 5, 6, 7, 8],
        "contract_duration": [10],
        "use_atr_filter":    [False, True],
        "atr_filter_ratio":  [1.0, 1.2],
        "loss_cooldown":     [0, 2, 3],
    }
elif args.zscore:
    GRID = {
        "zscore_period":    [10, 15, 20, 30],
        "zscore_threshold": [2.0, 2.5, 3.0],
        "confirm_ticks":    [1, 2, 3],
        "use_atr_filter":   [False, True],
        "atr_filter_ratio": [1.0, 1.2],
        "loss_cooldown":    [0, 2, 3],
        "contract_duration": [10],
    }
elif args.momentum:
    # Momentum grid: asymmetric thresholds, signal direction flipped
    # RSI > overbought → BUY_RISE, RSI < oversold → BUY_FALL
    # confirm_ticks fixed at 2 per user preference
    GRID = {
        "rsi_period":       [5, 7, 10, 14],
        "rsi_overbought":   [45, 50, 55, 60],
        "rsi_oversold":     [25, 30, 35, 40],
        "contract_duration":[10],
        "use_atr_filter":   [False, True],
        "atr_filter_ratio": [1.0, 1.2],
        "loss_cooldown":    [0, 2, 3],
        "confirm_ticks":    [2],
        "momentum_mode":    [True],
    }
elif args.crossover:
    # Crossover grid: RSI 50 midline momentum strategy
    # rsi_overbought/oversold are unused in crossover mode (always 50)
    GRID = {
        "rsi_period":       [5, 7, 10, 14],
        "rsi_overbought":   [80],           # placeholder — ignored in crossover mode
        "contract_duration":[10],
        "use_atr_filter":   [False, True],
        "atr_filter_ratio": [0.8, 1.0, 1.2, 1.5],
        "loss_cooldown":    [0, 2, 3, 5],
        "confirm_ticks":    [1, 2, 3],
        "use_midline_cross":[True],
    }
elif args.adaptive:
    # Adaptive threshold grid: dynamic overbought/oversold via RSI percentile.
    # Includes a fixed-threshold baseline row (adaptive_threshold=False) for comparison.
    GRID = {
        "rsi_period":         [10],
        "rsi_overbought":     [80],   # fallback when not enough history
        "contract_duration":  [10],
        "confirm_ticks":      [2],
        "use_atr_filter":     [False],
        "atr_filter_ratio":   [1.0],
        "loss_cooldown":      [0, 2, 3],
        "adaptive_threshold": [False, True],
        "adaptive_window":    [200, 500],
        "adaptive_pct":       [92.5, 95.0, 97.5],
    }
elif args.depth:
    # RSI-depth duration grid: longer contract for deeper RSI extremes.
    # Baseline row: rsi_extreme_threshold=0 (disabled, standard 10-tick).
    GRID = {
        "rsi_period":              [10],
        "rsi_overbought":          [80],
        "contract_duration":       [10],
        "confirm_ticks":           [2],
        "use_atr_filter":          [False],
        "atr_filter_ratio":        [1.0],
        "loss_cooldown":           [0, 2, 3],
        "rsi_extreme_threshold":   [0, 83, 85, 87, 90],
        "rsi_extreme_duration":    [15, 20],
    }
elif args.divergence:
    # Divergence grid: base params fixed, sweep divergence lookback window.
    GRID = {
        "rsi_period":          [10],
        "rsi_overbought":      [80],
        "contract_duration":   [10],
        "confirm_ticks":       [2],
        "use_atr_filter":      [False],
        "atr_filter_ratio":    [1.0],
        "loss_cooldown":       [0, 2, 3],
        "use_divergence":      [True],
        "divergence_lookback": [10, 20, 30, 50],
    }
elif args.barsize:
    # Bar-size grid: all other params fixed at proven best, sweep aggregation granularity.
    GRID = {
        "rsi_period":       [10],
        "rsi_overbought":   [80],
        "contract_duration":[10],
        "confirm_ticks":    [2],
        "use_atr_filter":   [False],
        "atr_filter_ratio": [1.0],
        "loss_cooldown":    [0, 2, 3],
        "bar_size":         [1, 3, 5, 10],
    }
elif args.filters:
    # Filters grid: base params fixed (RSI 10, 80/20, cf=2, dur=10), sweep optional signal layers.
    # Determines whether slope/price/dual-RSI filters improve on the raw reversal.
    GRID = {
        "rsi_period":          [10],
        "rsi_overbought":      [80],
        "contract_duration":   [10],
        "confirm_ticks":       [2],
        "use_atr_filter":      [False, True],
        "atr_filter_ratio":    [1.0, 1.2],
        "loss_cooldown":       [0, 2, 3],
        "rsi_slope_confirm":   [False, True],
        "price_confirm":       [False, True],
        "dual_rsi_period":     [0, 14, 20],
        "dual_rsi_threshold":  [45.0, 50.0],
    }
elif args.dur10:
    # Focused grid: 10-tick contracts only — vary RSI, thresholds, ATR, cooldown
    GRID = {
        "rsi_period":       [7, 10, 14],
        "rsi_overbought":   [70, 75, 80, 82, 85],
        "contract_duration":[10],
        "use_atr_filter":   [False, True],
        "atr_filter_ratio": [0.8, 1.0, 1.2, 1.5],
        "loss_cooldown":    [0, 2, 3, 5],
        "confirm_ticks":    [1, 2, 3],
        "use_midline_cross":[False],
    }
else:
    # Standard grid across all durations
    GRID = {
        "rsi_period":       [7, 10, 14],
        "rsi_overbought":   [70, 72, 75, 80],
        "contract_duration":[3, 5, 10],
        "use_atr_filter":   [False, True],
        "atr_filter_ratio": [0.8, 1.0, 1.2],
        "loss_cooldown":    [0, 3, 5],
        "confirm_ticks":    [1],
        "use_midline_cross":[False],
    }

def combos():
    keys = list(GRID.keys())
    for vals in itertools.product(*GRID.values()):
        d = dict(zip(keys, vals))
        if not d["use_atr_filter"] and d["atr_filter_ratio"] != 1.0:
            continue   # skip redundant ratio variants when filter is off
        # dual_rsi_threshold is meaningless when dual_rsi_period=0 — deduplicate
        if d.get("dual_rsi_period", 1) == 0 and d.get("dual_rsi_threshold", 45.0) != 45.0:
            continue
        # Momentum mode has rsi_oversold in the grid directly (asymmetric)
        if "rsi_oversold" not in d and "rsi_overbought" in d:
            d["rsi_oversold"] = 100 - d["rsi_overbought"]
        yield d

# ── Run sweep ──────────────────────────────────────────────────
results = []
total_combos = sum(1 for _ in combos())
print(f"Testing {total_combos} combinations across {N_WINDOWS} windows of {WINDOW_SIZE:,} ticks...\n")

STRATEGY_CLASS = (StreakReversalStrategy if args.streak else
                  ZScoreReversalStrategy if args.zscore else
                  None)  # None → engine defaults to RSIReversalStrategy

for idx, params in enumerate(combos(), 1):
    s_cfg = copy.deepcopy(base_cfg["strategy"])
    r_cfg = copy.deepcopy(base_cfg["risk"])

    if args.streak:
        s_cfg.update({
            "symbol":            SYMBOL,
            "streak_length":     params["streak_length"],
            "use_atr_filter":    params["use_atr_filter"],
            "atr_filter_ratio":  params["atr_filter_ratio"],
            "loss_cooldown":     params["loss_cooldown"],
            "contract_duration": params["contract_duration"],
            "atr_period":        14,
            "atr_baseline_period": 50,
        })
    elif args.zscore:
        s_cfg.update({
            "symbol":            SYMBOL,
            "zscore_period":     params["zscore_period"],
            "zscore_threshold":  params["zscore_threshold"],
            "confirm_ticks":     params["confirm_ticks"],
            "use_atr_filter":    params["use_atr_filter"],
            "atr_filter_ratio":  params["atr_filter_ratio"],
            "loss_cooldown":     params["loss_cooldown"],
            "contract_duration": params["contract_duration"],
            "atr_period":        14,
            "atr_baseline_period": 50,
        })
    else:
        s_cfg.update({
            "symbol":             SYMBOL,
            "rsi_period":         params["rsi_period"],
            "rsi_overbought":     params["rsi_overbought"],
            "rsi_oversold":       params["rsi_oversold"],
            "contract_duration":  params["contract_duration"],
            "use_atr_filter":     params["use_atr_filter"],
            "atr_filter_ratio":   params["atr_filter_ratio"],
            "loss_cooldown":      params["loss_cooldown"],
            "confirm_ticks":      params["confirm_ticks"],
            "bar_size":           params.get("bar_size", 1),
            "use_divergence":          params.get("use_divergence", False),
            "divergence_lookback":     params.get("divergence_lookback", 20),
            "adaptive_threshold":      params.get("adaptive_threshold", False),
            "adaptive_window":         params.get("adaptive_window", 500),
            "adaptive_pct":            params.get("adaptive_pct", 95.0),
            "rsi_extreme_threshold":   params.get("rsi_extreme_threshold", 0.0),
            "rsi_extreme_duration":    params.get("rsi_extreme_duration", 15),
            "use_midline_cross":  params.get("use_midline_cross", False),
            "momentum_mode":      params.get("momentum_mode", False),
            "rsi_slope_confirm":  params.get("rsi_slope_confirm", False),
            "price_confirm":      params.get("price_confirm", False),
            "dual_rsi_period":    params.get("dual_rsi_period", 0),
            "dual_rsi_threshold": params.get("dual_rsi_threshold", 45.0),
            "ema_trend_period":   0,
            "use_bb_filter":      False,
        })

    win_rates        = []
    total_trades_sum = 0

    for w_ticks in windows:
        eng = BacktestEngine(strategy_cfg=s_cfg, risk_cfg=r_cfg, payout_pct=PAYOUT,
                             strategy_class=STRATEGY_CLASS)
        res = eng.run(w_ticks, starting_balance=BALANCE)
        win_rates.append(res.win_rate)
        total_trades_sum += res.total_trades

    if total_trades_sum < N_WINDOWS * 5:   # skip combos with < 5 trades per window on average
        continue

    avg_wr  = round(sum(win_rates) / len(win_rates), 1)
    min_wr  = round(min(win_rates), 1)
    wins_ge = sum(1 for w in win_rates if w >= BE)

    results.append({
        **params,
        "avg_wr":           avg_wr,
        "min_wr":           min_wr,
        "wins_ge":          wins_ge,
        "per_window_wrs":   [round(w, 1) for w in win_rates],
        "total_trades":     total_trades_sum,
    })

    if idx % 50 == 0:
        print(f"  {idx}/{total_combos}...", end="\r", flush=True)

# ── Sort: windows_profitable DESC → min_wr DESC → avg_wr DESC ──
results.sort(key=lambda x: (x["wins_ge"], x["min_wr"], x["avg_wr"]), reverse=True)

print("\n")
if args.streak:
    hdr = (f"{'WG':>3} {'MinWR':>6} {'AvgWR':>6} {'Trd':>5}  "
           f"{'Streak':>7} {'ATR':>4} {'Ratio':>5} {'Cd':>3}  Per-window WRs")
elif args.zscore:
    hdr = (f"{'WG':>3} {'MinWR':>6} {'AvgWR':>6} {'Trd':>5}  "
           f"{'Period':>7} {'Thr':>5} {'ATR':>4} {'Ratio':>5} {'Cd':>3} {'Cf':>3}  Per-window WRs")
elif args.filters:
    hdr = (f"{'WG':>3} {'MinWR':>6} {'AvgWR':>6} {'Trd':>5}  "
           f"{'ATR':>4} {'Ratio':>5} {'Cd':>3}  "
           f"{'Slp':>4} {'Prc':>4} {'DRSI':>5} {'DThr':>5}  Per-window WRs")
elif args.adaptive:
    hdr = (f"{'WG':>3} {'MinWR':>6} {'AvgWR':>6} {'Trd':>5}  "
           f"{'Adp':>4} {'Win':>5} {'Pct':>5} {'Cd':>3}  Per-window WRs")
elif args.depth:
    hdr = (f"{'WG':>3} {'MinWR':>6} {'AvgWR':>6} {'Trd':>5}  "
           f"{'Thr':>5} {'XDur':>5} {'Cd':>3}  Per-window WRs")
elif args.divergence:
    hdr = (f"{'WG':>3} {'MinWR':>6} {'AvgWR':>6} {'Trd':>5}  "
           f"{'Lkbk':>5} {'Cd':>3}  Per-window WRs")
elif args.barsize:
    hdr = (f"{'WG':>3} {'MinWR':>6} {'AvgWR':>6} {'Trd':>5}  "
           f"{'Bar':>4} {'Cd':>3}  Per-window WRs")
else:
    hdr = (f"{'WG':>3} {'MinWR':>6} {'AvgWR':>6} {'Trd':>5}  "
           f"{'RSI':>4} {'Dur':>4} {'OB/OS':>6} "
           f"{'ATR':>4} {'Ratio':>5} {'Cd':>3} {'Cf':>3}  Per-window WRs")
print(hdr)
print("-" * (len(hdr) + 10))

for r in results[:50]:
    atr_flag = "Y" if r["use_atr_filter"] else "N"
    wrs = "  ".join(f"{w:.1f}" for w in r["per_window_wrs"])
    if args.streak:
        print(
            f"{r['wins_ge']:>3} {r['min_wr']:>6.1f} {r['avg_wr']:>6.1f} {r['total_trades']:>5}  "
            f"{r['streak_length']:>7} {atr_flag:>4} {r['atr_filter_ratio']:>5.1f} {r['loss_cooldown']:>3}  {wrs}"
        )
    elif args.zscore:
        print(
            f"{r['wins_ge']:>3} {r['min_wr']:>6.1f} {r['avg_wr']:>6.1f} {r['total_trades']:>5}  "
            f"Z{r['zscore_period']:<3} >{r['zscore_threshold']:.1f} "
            f"{atr_flag:>4} {r['atr_filter_ratio']:>5.1f} {r['loss_cooldown']:>3} {r['confirm_ticks']:>3}  {wrs}"
        )
    elif args.filters:
        slp  = "Y" if r.get("rsi_slope_confirm") else "N"
        prc  = "Y" if r.get("price_confirm") else "N"
        drsi = r.get("dual_rsi_period", 0)
        dthr = r.get("dual_rsi_threshold", 45.0)
        print(
            f"{r['wins_ge']:>3} {r['min_wr']:>6.1f} {r['avg_wr']:>6.1f} {r['total_trades']:>5}  "
            f"{atr_flag:>4} {r['atr_filter_ratio']:>5.1f} {r['loss_cooldown']:>3}  "
            f"{slp:>4} {prc:>4} {drsi:>5} {dthr:>5.0f}  {wrs}"
        )
    elif args.adaptive:
        adp = "Y" if r.get("adaptive_threshold") else "N"
        win = r.get("adaptive_window", 500)
        pct = r.get("adaptive_pct", 95.0)
        print(
            f"{r['wins_ge']:>3} {r['min_wr']:>6.1f} {r['avg_wr']:>6.1f} {r['total_trades']:>5}  "
            f"{adp:>4} {win:>5} {pct:>5.1f} {r['loss_cooldown']:>3}  {wrs}"
        )
    elif args.depth:
        thr  = r.get("rsi_extreme_threshold", 0)
        xdur = r.get("rsi_extreme_duration", 15)
        print(
            f"{r['wins_ge']:>3} {r['min_wr']:>6.1f} {r['avg_wr']:>6.1f} {r['total_trades']:>5}  "
            f"{thr:>5} {xdur:>5} {r['loss_cooldown']:>3}  {wrs}"
        )
    elif args.divergence:
        print(
            f"{r['wins_ge']:>3} {r['min_wr']:>6.1f} {r['avg_wr']:>6.1f} {r['total_trades']:>5}  "
            f"{r.get('divergence_lookback', 20):>5} {r['loss_cooldown']:>3}  {wrs}"
        )
    elif args.barsize:
        print(
            f"{r['wins_ge']:>3} {r['min_wr']:>6.1f} {r['avg_wr']:>6.1f} {r['total_trades']:>5}  "
            f"{r.get('bar_size', 1):>4} {r['loss_cooldown']:>3}  {wrs}"
        )
    else:
        print(
            f"{r['wins_ge']:>3} {r['min_wr']:>6.1f} {r['avg_wr']:>6.1f} {r['total_trades']:>5}  "
            f"{r['rsi_period']:>4} {r['contract_duration']:>4} "
            f"{r['rsi_overbought']}/{r['rsi_oversold']:>3}  "
            f"{atr_flag:>4} {r['atr_filter_ratio']:>5.1f} {r['loss_cooldown']:>3} "
            f"{r['confirm_ticks']:>3}  {wrs}"
        )

# ── Save full results ──────────────────────────────────────────
os.makedirs("data", exist_ok=True)
suffix   = ("streak" if args.streak else
            "zscore" if args.zscore else
            "momentum"  if args.momentum  else
            "crossover" if args.crossover else
            "filters" if args.filters else
            "adaptive" if args.adaptive else
            "depth" if args.depth else
            "divergence" if args.divergence else
            "barsize" if args.barsize else
            "dur10" if args.dur10 else
            "std")
out_file = f"data/sweep_{SYMBOL}_{suffix}.json"
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nFull results saved to {out_file}")
print(f"Total combinations tested: {len(results)} (filtered from {total_combos})")
