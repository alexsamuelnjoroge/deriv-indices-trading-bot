"""
Overlap test — how does allowing multiple simultaneous open positions per symbol
affect EV and win rate?

Tests the top validated configs from research_strategies.py with
max_open = 1, 2, 3 positions per symbol.

Usage:
  python backtest_overlap.py
"""

import json, os, asyncio
from datetime import datetime, timezone

MULTIPLIER     = 100
COMMISSION_PCT = 0.02
GRANULARITY    = 3600

# ── Top configs from research_strategies.py ───────────────────────────────────
CONFIGS = [
    {
        "label":    "Gold MACD(12,26,9)",
        "symbol":   "frxXAUUSD",
        "strategy": "macd",
        "params":   {"fast": 12, "slow": 26, "sig": 9},
        "sl_pct":   0.0075,
        "tp_pct":   0.020,
        "max_bars": 96,
    },
    {
        "label":    "Gold Donchian(30)",
        "symbol":   "frxXAUUSD",
        "strategy": "donchian",
        "params":   {"period": 30},
        "sl_pct":   0.010,
        "tp_pct":   0.020,
        "max_bars": 24,
    },
    {
        "label":    "Silver MACD(12,26,9)",
        "symbol":   "frxXAGUSD",
        "strategy": "macd",
        "params":   {"fast": 12, "slow": 26, "sig": 9},
        "sl_pct":   0.010,
        "tp_pct":   0.020,
        "max_bars": 24,
    },
    {
        "label":    "Silver BB Squeeze(50,1.5,30%)",
        "symbol":   "frxXAGUSD",
        "strategy": "bb_squeeze",
        "params":   {"period": 50, "num_std": 1.5, "squeeze_pct": 30},
        "sl_pct":   0.010,
        "tp_pct":   0.020,
        "max_bars": 48,
    },
    {
        "label":    "GBP/USD EMA(20x50)",
        "symbol":   "frxGBPUSD",
        "strategy": "ema_cross",
        "params":   {"fast": 20, "slow": 50},
        "sl_pct":   0.0075,
        "tp_pct":   0.020,
        "max_bars": 96,
    },
    {
        "label":    "GBP/USD MACD(5,13,5)",
        "symbol":   "frxGBPUSD",
        "strategy": "macd",
        "params":   {"fast": 5, "slow": 13, "sig": 5},
        "sl_pct":   0.0075,
        "tp_pct":   0.010,
        "max_bars": 96,
    },
]

# ── Candle loading ─────────────────────────────────────────────────────────────
def load_candles(symbol):
    cache = f"cache_{symbol}_{GRANULARITY}s_candles.json"
    with open(cache) as f:
        return json.load(f)

# ── Indicator helpers ──────────────────────────────────────────────────────────
def ema_series(values, period):
    result, alpha = [None] * len(values), 2 / (period + 1)
    val, warmup = None, 0
    for i, v in enumerate(values):
        if v is None: continue
        if val is None: val = v; warmup = i
        else: val = alpha * v + (1 - alpha) * val
        result[i] = val
    for i in range(warmup, min(warmup + period - 1, len(result))):
        result[i] = None
    return result


def get_signals(candles, strategy, params):
    if strategy == "macd":
        closes = [c["close"] for c in candles]
        fe = ema_series(closes, params["fast"])
        se = ema_series(closes, params["slow"])
        macd = [f - s if f is not None and s is not None else None for f, s in zip(fe, se)]
        sig  = ema_series(macd, params["sig"])
        hist = [m - s if m is not None and s is not None else None for m, s in zip(macd, sig)]
        out = []
        for i in range(1, len(candles)):
            if hist[i] is None or hist[i-1] is None or macd[i] is None: continue
            if hist[i] > 0 and hist[i-1] <= 0 and macd[i] > 0: out.append((i, +1))
            elif hist[i] < 0 and hist[i-1] >= 0 and macd[i] < 0: out.append((i, -1))
        return out

    elif strategy == "donchian":
        period = params["period"]
        highs = [c["high"] for c in candles]
        lows  = [c["low"]  for c in candles]
        out = []
        for i in range(period, len(candles)):
            if candles[i]["high"] > max(highs[i-period:i]): out.append((i, +1))
            elif candles[i]["low"] < min(lows[i-period:i]):  out.append((i, -1))
        return out

    elif strategy == "ema_cross":
        closes = [c["close"] for c in candles]
        fe = ema_series(closes, params["fast"])
        se = ema_series(closes, params["slow"])
        out = []
        for i in range(1, len(candles)):
            if fe[i] is None or se[i] is None or fe[i-1] is None or se[i-1] is None: continue
            if fe[i] > se[i] and fe[i-1] <= se[i-1]: out.append((i, +1))
            elif fe[i] < se[i] and fe[i-1] >= se[i-1]: out.append((i, -1))
        return out

    elif strategy == "bb_squeeze":
        closes = [c["close"] for c in candles]
        period, num_std, squeeze_pct = params["period"], params["num_std"], params["squeeze_pct"]
        out, width_history = [], []
        for i in range(period, len(candles)):
            window = closes[i-period:i]
            mid = sum(window) / period
            std = (sum((p-mid)**2 for p in window) / period) ** 0.5
            width = (2 * num_std * std) / mid if mid > 0 else 0
            width_history.append(width)
            if len(width_history) < 20: continue
            recent = sorted(width_history[-100:])
            thresh = recent[max(0, int(len(recent) * squeeze_pct / 100) - 1)]
            was_sq = width_history[-2] <= thresh if len(width_history) >= 2 else False
            is_sq  = width <= thresh
            if was_sq and not is_sq:
                out.append((i, +1 if closes[i] > mid else -1))
        return out

    return []


# ── Multi-position simulator ───────────────────────────────────────────────────
def simulate(candles, signals, sl_pct, tp_pct, max_bars, max_open):
    """
    Simulate trading with up to max_open simultaneous open positions.
    New signals are taken if len(open) < max_open; otherwise skipped.
    """
    signal_dict = dict(signals)   # bar_i -> direction (last signal wins if dupe)
    open_positions = []
    wins = losses = timeouts = 0
    total_ev = 0.0

    for i in range(len(candles)):
        # Resolve open positions at bar i
        still_open = []
        for pos in open_positions:
            elapsed = i - pos["entry_bar"]
            if elapsed <= 0:
                still_open.append(pos)
                continue

            h, l = candles[i]["high"], candles[i]["low"]
            ep, direction = pos["entry_price"], pos["direction"]
            tp_p = ep * (1 + direction * tp_pct)
            sl_p = ep * (1 - direction * sl_pct)
            resolved = False

            if direction == 1:
                if l <= sl_p:    # SL checked first (conservative)
                    losses += 1; total_ev -= (MULTIPLIER * sl_pct + COMMISSION_PCT); resolved = True
                elif h >= tp_p:
                    wins   += 1; total_ev += (MULTIPLIER * tp_pct - COMMISSION_PCT); resolved = True
            else:
                if h >= sl_p:
                    losses += 1; total_ev -= (MULTIPLIER * sl_pct + COMMISSION_PCT); resolved = True
                elif l <= tp_p:
                    wins   += 1; total_ev += (MULTIPLIER * tp_pct - COMMISSION_PCT); resolved = True

            if not resolved and elapsed >= max_bars:
                pct = (candles[i]["close"] - ep) / ep * direction
                timeouts += 1; total_ev += MULTIPLIER * pct - COMMISSION_PCT; resolved = True

            if not resolved:
                still_open.append(pos)

        open_positions = still_open

        # Open new position if signal fires and slots available
        if i in signal_dict and len(open_positions) < max_open:
            entry_bar = i + 1
            if entry_bar < len(candles):
                open_positions.append({
                    "entry_bar":   entry_bar,
                    "entry_price": candles[entry_bar]["open"],
                    "direction":   signal_dict[i],
                })

    # Force-close anything still open at end of data
    for pos in open_positions:
        ep, direction = pos["entry_price"], pos["direction"]
        pct = (candles[-1]["close"] - ep) / ep * direction
        timeouts += 1
        total_ev += MULTIPLIER * pct - COMMISSION_PCT

    n = wins + losses + timeouts
    if n == 0:
        return None
    wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    be_wr = (MULTIPLIER * sl_pct + COMMISSION_PCT) / (
        MULTIPLIER * tp_pct - COMMISSION_PCT + MULTIPLIER * sl_pct + COMMISSION_PCT) * 100
    return {"n": n, "wins": wins, "losses": losses, "timeouts": timeouts,
            "wr": wr, "be_wr": be_wr, "ev": total_ev / n, "total_ev": total_ev}


# ── Walk-forward with max_open support ────────────────────────────────────────
def walk_forward(candles, signals_fn, sl_pct, tp_pct, max_bars, max_open, n_splits=3):
    sz = len(candles) // n_splits
    evs, passes = [], 0
    rows = []
    for fold in range(n_splits):
        s = fold * sz
        e = (fold + 1) * sz if fold < n_splits - 1 else len(candles)
        chunk = candles[s:e]
        sigs  = signals_fn(chunk)
        r = simulate(chunk, sigs, sl_pct, tp_pct, max_bars, max_open)
        t0 = datetime.fromtimestamp(chunk[0]["epoch"],  tz=timezone.utc).strftime("%m/%d")
        t1 = datetime.fromtimestamp(chunk[-1]["epoch"], tz=timezone.utc).strftime("%m/%d")
        rows.append((fold + 1, t0, t1, r))
        if r and r["ev"] > 0:
            passes += 1
            evs.append(r["ev"])
    mean_ev = sum(evs) / len(evs) if evs else None
    return rows, passes, mean_ev


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("OVERLAP TEST — Max simultaneous positions per symbol: 1 vs 2 vs 3")
    print("=" * 72)

    # Pre-load candle files
    candle_cache = {}
    for cfg in CONFIGS:
        sym = cfg["symbol"]
        if sym not in candle_cache:
            candle_cache[sym] = load_candles(sym)

    summary = []

    for cfg in CONFIGS:
        candles   = candle_cache[cfg["symbol"]]
        sig_fn    = lambda c, cfg=cfg: get_signals(c, cfg["strategy"], cfg["params"])
        sl, tp, mb = cfg["sl_pct"], cfg["tp_pct"], cfg["max_bars"]
        be_wr = (MULTIPLIER * sl + COMMISSION_PCT) / (
            MULTIPLIER * tp - COMMISSION_PCT + MULTIPLIER * sl + COMMISSION_PCT) * 100

        print(f"\n{'─'*72}")
        print(f"{cfg['label']}  |  {cfg['symbol']}  |  BE WR = {be_wr:.1f}%")
        print(f"  SL={sl*100:.2f}%  TP={tp*100:.2f}%  max_hold={mb}h")
        print(f"{'─'*72}")
        print(f"  {'max_open':>8}  {'N':>5}  {'WR':>6}  {'EV/tr':>8}  {'TotEV':>8}  {'WF':>5}  Result")
        print(f"  {'-'*65}")

        for max_open in [1, 2, 3]:
            # Full-sample stats
            all_sigs = sig_fn(candles)
            r = simulate(candles, all_sigs, sl, tp, mb, max_open)
            # Walk-forward
            wf_rows, wf_passes, wf_mean_ev = walk_forward(
                candles, sig_fn, sl, tp, mb, max_open)

            if r is None:
                print(f"  {max_open:>8}  no trades")
                summary.append((cfg["label"], max_open, None, None, None, 0))
                continue

            verdict = "PASS" if wf_passes == 3 else f"{wf_passes}/3"
            ev_str  = f"{r['ev']:+.4f}"
            tot_str = f"{r['total_ev']:+.3f}"
            wf_ev   = f"{wf_mean_ev:+.4f}" if wf_mean_ev else "  n/a  "
            print(f"  {max_open:>8}  {r['n']:>5}  {r['wr']:>5.1f}%  {ev_str:>8}  "
                  f"{tot_str:>8}  {wf_ev}  {verdict}")
            summary.append((cfg["label"], max_open, r["n"], r["ev"],
                            r["total_ev"], wf_passes))

        # Print walk-forward detail for max_open=1 vs max_open=2
        print()
        for max_open in [1, 2]:
            wf_rows, wf_passes, _ = walk_forward(candles, sig_fn, sl, tp, mb, max_open)
            print(f"  Walk-forward (max_open={max_open}):")
            for fold, t0, t1, r in wf_rows:
                if r is None:
                    print(f"    Fold {fold} {t0}-{t1}: no trades")
                else:
                    res = "PASS" if r["ev"] > 0 else "FAIL"
                    print(f"    Fold {fold} {t0}-{t1}: N={r['n']:>3}  "
                          f"WR={r['wr']:>5.1f}%  EV={r['ev']:>+.4f}  {res}")
            print()

    # Summary comparison
    print(f"\n{'='*72}")
    print("SUMMARY — Does allowing more positions per symbol help?")
    print(f"{'='*72}")
    print(f"  {'Config':30} {'max':>4}  {'N':>5}  {'EV/tr':>8}  {'TotEV':>8}  WF")
    print(f"  {'-'*65}")
    for label, max_open, n, ev, tot_ev, wf_passes in summary:
        if n is None:
            continue
        ev_str  = f"{ev:+.4f}"
        tot_str = f"{tot_ev:+.3f}"
        print(f"  {label:30} {max_open:>4}  {n:>5}  {ev_str:>8}  {tot_str:>8}  {wf_passes}/3")

    print("\nKey: TotEV = total expected profit across all trades (in $ per $1 stake)")
    print("     EV/tr = expected value per individual trade")


if __name__ == "__main__":
    main()
