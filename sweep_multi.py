"""
Multi-window parameter sweep.
Tests all combinations across N independent windows and ranks by
min_win_rate (the weakest window) — only truly robust params will
stay above breakeven in every window.

Usage:
  python sweep_multi.py                   # 5 windows x 5k ticks (25k total)
  python sweep_multi.py --dur10           # focused 10-tick sweep: 5 x 10k ticks (50k total)
  python sweep_multi.py --fresh           # re-fetch ticks from Deriv
"""
import argparse, copy, itertools, json, os
from loguru import logger
logger.remove()

import yaml
from dotenv import load_dotenv
from src.data.history import fetch_ticks
from src.backtest.engine import BacktestEngine

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--dur10",  action="store_true", help="Focused 10-tick sweep on 50k ticks (5 x 10k windows)")
parser.add_argument("--fresh",  action="store_true", help="Re-fetch ticks from Deriv (ignore cache)")
parser.add_argument("--windows", type=int, default=5, help="Number of windows (default: 5)")
args = parser.parse_args()

# ── Load base config ───────────────────────────────────────────
with open("config.yaml") as f:
    base_cfg = yaml.safe_load(f)

SYMBOL   = base_cfg["strategy"]["symbol"]
PAYOUT   = 0.87
BALANCE  = 1000.0
BE       = round((1 / (1 + PAYOUT)) * 100, 1)   # 53.5%
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
if args.dur10:
    # Focused grid: 10-tick contracts only — vary RSI, thresholds, ATR, cooldown
    GRID = {
        "rsi_period":       [7, 10, 14],
        "rsi_overbought":   [70, 72, 75, 80],
        "contract_duration":[10],
        "use_atr_filter":   [False, True],
        "atr_filter_ratio": [0.8, 1.0, 1.2, 1.5],
        "loss_cooldown":    [0, 2, 3, 5],
        "confirm_ticks":    [1, 2],
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
    }

def combos():
    keys = list(GRID.keys())
    for vals in itertools.product(*GRID.values()):
        d = dict(zip(keys, vals))
        if not d["use_atr_filter"] and d["atr_filter_ratio"] != 1.0:
            continue   # skip redundant ratio variants when filter is off
        d["rsi_oversold"] = 100 - d["rsi_overbought"]
        yield d

# ── Run sweep ──────────────────────────────────────────────────
results = []
total_combos = sum(1 for _ in combos())
print(f"Testing {total_combos} combinations across {N_WINDOWS} windows of {WINDOW_SIZE:,} ticks...\n")

for idx, params in enumerate(combos(), 1):
    s_cfg = copy.deepcopy(base_cfg["strategy"])
    r_cfg = copy.deepcopy(base_cfg["risk"])

    s_cfg.update({
        "rsi_period":        params["rsi_period"],
        "rsi_overbought":    params["rsi_overbought"],
        "rsi_oversold":      params["rsi_oversold"],
        "contract_duration": params["contract_duration"],
        "use_atr_filter":    params["use_atr_filter"],
        "atr_filter_ratio":  params["atr_filter_ratio"],
        "loss_cooldown":     params["loss_cooldown"],
        "confirm_ticks":     params["confirm_ticks"],
        # keep disabled filters off
        "rsi_slope_confirm": False,
        "price_confirm":     False,
        "dual_rsi_period":   0,
        "ema_trend_period":  0,
        "use_bb_filter":     False,
    })

    win_rates        = []
    total_trades_sum = 0

    for w_ticks in windows:
        eng = BacktestEngine(strategy_cfg=s_cfg, risk_cfg=r_cfg, payout_pct=PAYOUT)
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
hdr = (f"{'WG':>3} {'MinWR':>6} {'AvgWR':>6} {'Trd':>5}  "
       f"{'RSI':>4} {'Dur':>4} {'OB/OS':>6} "
       f"{'ATR':>4} {'Ratio':>5} {'Cd':>3} {'Cf':>3}  Per-window WRs")
print(hdr)
print("-" * (len(hdr) + 10))

for r in results[:50]:
    atr_flag = "Y" if r["use_atr_filter"] else "N"
    wrs = "  ".join(f"{w:.1f}" for w in r["per_window_wrs"])
    print(
        f"{r['wins_ge']:>3} {r['min_wr']:>6.1f} {r['avg_wr']:>6.1f} {r['total_trades']:>5}  "
        f"{r['rsi_period']:>4} {r['contract_duration']:>4} "
        f"{r['rsi_overbought']}/{r['rsi_oversold']:>3}  "
        f"{atr_flag:>4} {r['atr_filter_ratio']:>5.1f} {r['loss_cooldown']:>3} "
        f"{r['confirm_ticks']:>3}  {wrs}"
    )

# ── Save full results ──────────────────────────────────────────
os.makedirs("data", exist_ok=True)
out_file = "data/sweep_dur10_results.json" if args.dur10 else "data/sweep_multi_results.json"
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nFull results saved to {out_file}")
print(f"Total combinations tested: {len(results)} (filtered from {total_combos})")
