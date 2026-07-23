"""
Walk-forward out-of-sample validation.

Fetches a long stretch of tick history (default: 150k ticks, ~42 hours),
splits it into sequential non-overlapping windows, and evaluates a fixed
set of finalist configurations on each window — no parameter search.

Usage:
  python validate.py                     # 150k ticks, 15 windows, R_25
  python validate.py --count 300000      # ~3.5 days of data, 30 windows
  python validate.py --symbol R_75       # different symbol
  python validate.py --windows 20        # more windows (smaller each)
  python validate.py --no-fresh          # reuse existing holdout cache

Philosophy:
  Training data  → sweep_multi.py → pick parameters (done once)
  Walk-forward   → validate.py    → confirm generalisation across time
  Live trading   → main.py        → the ultimate forward test (ongoing)

Never tune parameters based on validate.py output. Read it as a verdict.
"""

import argparse
import copy
import os

import yaml
from dotenv import load_dotenv
from loguru import logger

logger.remove()

from src.data.history import fetch_ticks
from src.backtest.engine import BacktestEngine
from src.strategies.zscore_reversal import ZScoreReversalStrategy
from src.strategies.streak_reversal import StreakReversalStrategy

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--symbol",   type=str, default=None)
parser.add_argument("--windows",  type=int, default=15,    help="Number of sequential windows (default: 15)")
parser.add_argument("--count",    type=int, default=150000, help="Total ticks to fetch (default: 150000 = ~42h)")
parser.add_argument("--no-fresh", action="store_true",     help="Reuse cached holdout data if available")
args = parser.parse_args()

with open("config.yaml") as f:
    base_cfg = yaml.safe_load(f)

if args.symbol:
    SYMBOL = args.symbol.upper()
else:
    enabled = [s for s in base_cfg.get("symbols", []) if s.get("enabled", True)]
    SYMBOL  = enabled[0]["symbol"] if enabled else "R_25"

PAYOUT    = base_cfg["risk"].get("payout_pct", 0.9232)
BALANCE   = 1000.0
BE        = round((1 / (1 + PAYOUT)) * 100, 1)
N_WINDOWS = args.windows

# ── Holdout dataset ────────────────────────────────────────────────────────────────
# Count must differ from sweep_multi.py's 50k training cache to avoid file collision.
HOLDOUT_COUNT = args.count

use_fresh = not args.no_fresh
action    = "Fetching" if use_fresh else "Loading"
print(f"{action} {HOLDOUT_COUNT:,} holdout ticks for {SYMBOL}...")
ticks = fetch_ticks(
    SYMBOL,
    count=HOLDOUT_COUNT,
    fresh=use_fresh,
    app_id=os.getenv("DERIV_APP_ID", "1089"),
    api_token=os.getenv("DERIV_API_TOKEN"),
)
print(f"  Got {len(ticks):,} ticks\n")

window_size = len(ticks) // N_WINDOWS
windows     = [ticks[i * window_size: (i + 1) * window_size] for i in range(N_WINDOWS)]

# ── Finalist configurations — copy exactly from the live config / best sweep results ──
# Do NOT change these based on validate.py output.
r_cfg = copy.deepcopy(base_cfg["risk"])

FINALISTS = [
    {
        "name": "RSI(10) 80/20  cf=2  [LIVE]",
        "strategy_class": None,   # None → BacktestEngine defaults to RSIReversalStrategy
        "cfg": {
            **copy.deepcopy(base_cfg["strategy"]),
            "symbol":             SYMBOL,
            "rsi_period":         10,
            "rsi_overbought":     80,
            "rsi_oversold":       20,
            "contract_duration":  10,
            "confirm_ticks":      2,
            "use_atr_filter":     False,
            "loss_cooldown":      0,
            "bar_size":           1,
            "rsi_extreme_threshold": 0,
            "use_divergence":     False,
            "adaptive_threshold": False,
            "use_midline_cross":  False,
            "momentum_mode":      False,
            "rsi_slope_confirm":  False,
            "price_confirm":      False,
            "dual_rsi_period":    0,
            "ema_trend_period":   0,
            "use_bb_filter":      False,
            "atr_period":         14,
            "atr_baseline_period": 50,
        },
    },
    {
        "name": "RSI(10) 80/20  cf=2  cd=3  [cooldown]",
        "strategy_class": None,
        "cfg": {
            **copy.deepcopy(base_cfg["strategy"]),
            "symbol":             SYMBOL,
            "rsi_period":         10,
            "rsi_overbought":     80,
            "rsi_oversold":       20,
            "contract_duration":  10,
            "confirm_ticks":      2,
            "use_atr_filter":     False,
            "loss_cooldown":      3,
            "bar_size":           1,
            "rsi_extreme_threshold": 0,
            "use_divergence":     False,
            "adaptive_threshold": False,
            "use_midline_cross":  False,
            "momentum_mode":      False,
            "rsi_slope_confirm":  False,
            "price_confirm":      False,
            "dual_rsi_period":    0,
            "ema_trend_period":   0,
            "use_bb_filter":      False,
            "atr_period":         14,
            "atr_baseline_period": 50,
        },
    },
    {
        "name": "Streak(7)      cd=2  [backtest best]",
        "strategy_class": StreakReversalStrategy,
        "cfg": {
            "symbol":             SYMBOL,
            "streak_length":      7,
            "contract_duration":  10,
            "use_atr_filter":     False,
            "atr_filter_ratio":   1.0,
            "loss_cooldown":      2,
            "atr_period":         14,
            "atr_baseline_period": 50,
        },
    },
]

# ── Training results for comparison (from last sweep_multi.py --dur10 run) ──
TRAINING_RESULTS = {
    "RSI(10) 80/20  cf=2  [LIVE]":              {"wg": 4, "min_wr": 43.8, "avg_wr": 54.9, "trades": 91},
    "RSI(10) 80/20  cf=2  cd=3  [cooldown]":    {"wg": 4, "min_wr": 47.4, "avg_wr": 52.5, "trades": 232},
    "Streak(7)      cd=2  [backtest best]":      {"wg": 4, "min_wr": 46.7, "avg_wr": 55.7, "trades": 261},
}

# ── Run validation ──────────────────────────────────────────────
print(f"Evaluating {len(FINALISTS)} configs × {N_WINDOWS} windows of "
      f"{window_size:,} ticks each\n")
print(f"Breakeven WR: {BE}%  |  Payout: {PAYOUT*100:.2f}%\n")

header_width = max(len(f["name"]) for f in FINALISTS) + 2
col = f"{'Config':<{header_width}} {'WG':>3} {'MinWR':>6} {'AvgWR':>6} {'Trd':>5}   Train->Hold  Per-window WRs"
print(col)
print("-" * len(col))

all_results = []

for finalist in FINALISTS:
    name    = finalist["name"]
    s_cfg   = finalist["cfg"]
    s_class = finalist["strategy_class"]

    win_rates = []
    trade_sum = 0

    for w_idx, w_ticks in enumerate(windows):
        eng = BacktestEngine(strategy_cfg=s_cfg, risk_cfg=r_cfg,
                             payout_pct=PAYOUT, strategy_class=s_class)
        res = eng.run(w_ticks, starting_balance=BALANCE)
        win_rates.append(res.win_rate)
        trade_sum += res.total_trades
        print(f"  {name}  w{w_idx+1}/{N_WINDOWS}: {res.win_rate:.1f}% ({res.total_trades} trades)",
              end="\r", flush=True)

    avg_wr  = round(sum(win_rates) / len(win_rates), 1)
    min_wr  = round(min(win_rates), 1)
    max_wr  = round(max(win_rates), 1)
    wins_ge = sum(1 for w in win_rates if w >= BE)
    profitable_pct = round(wins_ge / N_WINDOWS * 100)

    train = TRAINING_RESULTS.get(name, {})
    trend = ""
    if train:
        delta = avg_wr - train["avg_wr"]
        trend = f"{train['avg_wr']:.1f}->{avg_wr:.1f} ({delta:+.1f})"
    else:
        trend = f"?->{avg_wr:.1f}"

    all_results.append({
        "name": name, "win_rates": win_rates, "avg_wr": avg_wr,
        "min_wr": min_wr, "max_wr": max_wr, "wins_ge": wins_ge,
        "trade_sum": trade_sum, "trend": trend, "profitable_pct": profitable_pct,
    })

    wrs_str = "  ".join(
        f"[{w:.1f}]" if w < BE else f" {w:.1f} " for w in win_rates
    )
    print(f"{name:<{header_width}} {wins_ge:>3}/{N_WINDOWS} {min_wr:>6.1f} {avg_wr:>6.1f} "
          f"{trade_sum:>5}   {trend:<22}  {wrs_str}")

print()
print("-" * (header_width + 90))
print("WRs in [brackets] are below breakeven. Train->Hold shows avg_wr drift vs training set.")

# ── Distribution summary ───────────────────────────────────────────────────────────
print()
print(f"  {'Config':<{header_width}} {'% profitable':>13}  {'Min':>6}  {'Avg':>6}  {'Max':>6}  {'Trades/window':>14}")
print(f"  {'-'*(header_width+60)}")
for r in all_results:
    trades_per = round(r["trade_sum"] / N_WINDOWS, 1)
    print(f"  {r['name']:<{header_width}} {r['profitable_pct']:>12}%  "
          f"{r['min_wr']:>6.1f}  {r['avg_wr']:>6.1f}  {r['max_wr']:>6.1f}  {trades_per:>14.1f}")
print()
print(f"  Breakeven: {BE}%  |  Total windows: {N_WINDOWS}  |  Ticks/window: ~{window_size:,}")
print(f"  Verdict guide: >= 60% profitable windows = robust  |  < 50% = no reliable edge")
