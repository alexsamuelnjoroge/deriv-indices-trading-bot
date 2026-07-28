"""
Strategy Research — alternative multiplier strategies for Gold, Silver, EUR/USD, GBP/USD.

Tests 4 strategies, each with a parameter sweep + 3-fold walk-forward:
  1. Donchian Channel Breakout  — trade price breaking to N-bar high/low
  2. EMA Crossover              — fast EMA crosses above/below slow EMA
  3. Bollinger Band Squeeze     — trade the breakout after a volatility squeeze
  4. MACD Momentum              — MACD histogram crosses zero with trend confirmation

Usage:
  python research_strategies.py                        # all instruments
  python research_strategies.py --symbol frxXAUUSD     # one instrument only
  python research_strategies.py --strategy donchian    # one strategy only
"""

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from itertools import product

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--symbol",   default=None,
                    help="Single symbol to test (default: all four)")
parser.add_argument("--strategy", default=None,
                    choices=["donchian", "ema_cross", "bb_squeeze", "macd"],
                    help="Single strategy to test (default: all four)")
parser.add_argument("--no-fresh", action="store_true",
                    help="Use cached candles if available")
args = parser.parse_args()

MULTIPLIER     = 100
COMMISSION_PCT = 0.02
GRANULARITY    = 3600   # 1h bars

SYMBOLS = [args.symbol] if args.symbol else [
    "frxXAUUSD", "frxXAGUSD", "frxEURUSD", "frxGBPUSD"
]
STRATEGIES = [args.strategy] if args.strategy else [
    "donchian", "ema_cross", "bb_squeeze", "macd"
]

# ── Candle fetching ────────────────────────────────────────────────────────────
async def fetch_candles(symbol):
    import websockets, json as _j
    app_id = os.getenv("DERIV_APP_ID", "1089")
    url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
    all_candles, seen, end = [], set(), "latest"
    async with websockets.connect(url) as ws:
        for chunk_i in range(8):
            await ws.send(_j.dumps({
                "ticks_history": symbol, "style": "candles",
                "granularity": GRANULARITY, "count": 5000,
                "end": end, "req_id": chunk_i + 1,
            }))
            msg = _j.loads(await asyncio.wait_for(ws.recv(), timeout=30))
            if msg.get("error"):
                break
            candles = msg.get("candles", [])
            if not candles:
                break
            new = [c for c in candles if c["epoch"] not in seen]
            for c in new: seen.add(c["epoch"])
            all_candles = new + all_candles
            end = candles[0]["epoch"] - 1
            if len(new) < 5000:
                break
    all_candles.sort(key=lambda c: c["epoch"])
    return all_candles


def load_candles(symbol):
    cache = f"cache_{symbol}_{GRANULARITY}s_candles.json"
    if args.no_fresh and os.path.exists(cache):
        with open(cache) as f:
            data = json.load(f)
        print(f"  {symbol}: {len(data):,} bars from cache")
        return data
    print(f"  {symbol}: fetching...")
    data = asyncio.run(fetch_candles(symbol))
    with open(cache, "w") as f:
        json.dump(data, f)
    print(f"  {symbol}: {len(data):,} bars fetched")
    return data


# ── Shared helpers ─────────────────────────────────────────────────────────────
def ema_series(values, period):
    result = [None] * len(values)
    alpha = 2 / (period + 1)
    val = None
    warmup = 0
    for i, v in enumerate(values):
        if v is None:
            continue
        if val is None:
            val = v
            warmup = i
        else:
            val = alpha * v + (1 - alpha) * val
        result[i] = val
    for i in range(warmup, min(warmup + period - 1, len(result))):
        result[i] = None
    return result


def simulate_trade(candles, entry_idx, direction, sl_pct, tp_pct, max_bars):
    entry = candles[entry_idx]["open"]
    tp_price = entry * (1 + direction * tp_pct)
    sl_price = entry * (1 - direction * sl_pct)
    for i in range(entry_idx, min(entry_idx + max_bars, len(candles))):
        h, l = candles[i]["high"], candles[i]["low"]
        if direction == 1:
            if l <= sl_price: return "sl",      (sl_price - entry) / entry
            if h >= tp_price: return "tp",      (tp_price - entry) / entry
        else:
            if h >= sl_price: return "sl",      (sl_price - entry) / entry
            if l <= tp_price: return "tp",      (tp_price - entry) / entry
    close = candles[min(entry_idx + max_bars - 1, len(candles) - 1)]["close"]
    return "timeout", (close - entry) / entry


def ev_stats(candles, signals, sl_pct, tp_pct, max_bars, min_n=10):
    """
    signals: list of (bar_index, direction) pairs.
    Returns stats dict or None if too few trades.
    """
    wins = losses = timeouts = 0
    total_ev = 0.0
    last_exit = -1

    for bar_i, direction in signals:
        if bar_i <= last_exit:
            continue
        entry_bar = bar_i + 1
        if entry_bar >= len(candles):
            continue
        outcome, pct_move = simulate_trade(
            candles, entry_bar, direction, sl_pct, tp_pct, max_bars)
        if outcome == "tp":
            wins += 1
            ev = MULTIPLIER * tp_pct - COMMISSION_PCT
        elif outcome == "sl":
            losses += 1
            ev = -(MULTIPLIER * sl_pct + COMMISSION_PCT)
        else:
            timeouts += 1
            ev = MULTIPLIER * pct_move * direction - COMMISSION_PCT
        total_ev += ev
        last_exit = entry_bar + max_bars if outcome == "timeout" else entry_bar

    n = wins + losses + timeouts
    if n < min_n:
        return None
    wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    be_wr = (MULTIPLIER * sl_pct + COMMISSION_PCT) / (
        MULTIPLIER * tp_pct - COMMISSION_PCT +
        MULTIPLIER * sl_pct + COMMISSION_PCT) * 100
    return {"n": n, "wins": wins, "losses": losses, "timeouts": timeouts,
            "wr": wr, "be_wr": be_wr, "ev": total_ev / n}


def walk_forward(candles, signal_fn, sl_pct, tp_pct, max_bars, n_splits=3, min_n=5):
    sz = len(candles) // n_splits
    evs, passes = [], 0
    rows = []
    for fold in range(n_splits):
        s = fold * sz
        e = (fold + 1) * sz if fold < n_splits - 1 else len(candles)
        chunk = candles[s:e]
        sigs = signal_fn(chunk)
        r = ev_stats(chunk, sigs, sl_pct, tp_pct, max_bars, min_n=min_n)
        t0 = datetime.fromtimestamp(chunk[0]["epoch"],  tz=timezone.utc).strftime("%m/%d")
        t1 = datetime.fromtimestamp(chunk[-1]["epoch"], tz=timezone.utc).strftime("%m/%d")
        if r is None:
            rows.append((fold + 1, t0, t1, None))
            continue
        evs.append(r["ev"])
        if r["ev"] > 0:
            passes += 1
        rows.append((fold + 1, t0, t1, r))
    return rows, evs, passes


def print_wf(rows, evs, passes, label):
    n_folds = len(rows)
    print(f"\n  Walk-forward ({label}):")
    print(f"  {'Fold':>4}  {'Period':13}  {'N':>4}  {'WR':>6}  {'EV/tr':>8}  Result")
    print(f"  {'-'*58}")
    for fold, t0, t1, r in rows:
        if r is None:
            print(f"  {fold:>4}  {t0}-{t1:11}  too few signals")
        else:
            res = "PASS" if r["ev"] > 0 else "FAIL"
            print(f"  {fold:>4}  {t0}-{t1:11}  {r['n']:>4}  {r['wr']:>5.1f}%  "
                  f"{r['ev']:>+8.4f}  {res}")
    if evs:
        mean_ev = sum(evs) / len(evs)
        print(f"  Mean EV: {mean_ev:+.4f}  |  {passes}/{n_folds} folds passing")
    return passes, len(rows), sum(evs) / len(evs) if evs else None


# ── Strategy 1: Donchian Channel Breakout ─────────────────────────────────────
def donchian_signals(candles, period):
    """Break above N-bar channel high -> MULTUP; below low -> MULTDOWN."""
    signals = []
    highs = [c["high"]  for c in candles]
    lows  = [c["low"]   for c in candles]
    for i in range(period, len(candles)):
        ch_high = max(highs[i - period:i])
        ch_low  = min(lows [i - period:i])
        if candles[i]["high"] > ch_high:
            signals.append((i, +1))
        elif candles[i]["low"] < ch_low:
            signals.append((i, -1))
    return signals


def run_donchian(candles):
    print("\n  [Donchian Channel Breakout]")
    best_results = []
    for period in [10, 20, 30, 50]:
        for sl_pct in [0.003, 0.005, 0.0075, 0.01]:
            for tp_pct in [0.005, 0.0075, 0.01, 0.015, 0.02]:
                if tp_pct <= sl_pct: continue
                for max_bars in [12, 24, 48, 96]:
                    sigs = donchian_signals(candles, period)
                    r = ev_stats(candles, sigs, sl_pct, tp_pct, max_bars)
                    if r and r["ev"] > 0:
                        best_results.append({
                            **r, "period": period,
                            "sl": sl_pct, "tp": tp_pct, "max_bars": max_bars,
                            "label": f"Don({period}) SL{sl_pct*100:.2f}%/TP{tp_pct*100:.2f}% max{max_bars}b"
                        })
    if not best_results:
        print("  No profitable configurations found.")
        return None
    best_results.sort(key=lambda x: x["ev"], reverse=True)
    b = best_results[0]
    print(f"  Best: {b['label']}  N={b['n']}  WR={b['wr']:.1f}%  "
          f"BE={b['be_wr']:.1f}%  EV={b['ev']:+.4f}")

    # Walk-forward on best config
    fn = lambda c: donchian_signals(c, b["period"])
    rows, evs, passes = walk_forward(candles, fn, b["sl"], b["tp"], b["max_bars"])
    return print_wf(rows, evs, passes, b["label"]), b


# ── Strategy 2: EMA Crossover ─────────────────────────────────────────────────
def ema_cross_signals(candles, fast, slow):
    """Fast EMA crosses above slow EMA -> MULTUP; below -> MULTDOWN."""
    closes = [c["close"] for c in candles]
    fast_e = ema_series(closes, fast)
    slow_e = ema_series(closes, slow)
    signals = []
    for i in range(1, len(candles)):
        if fast_e[i] is None or slow_e[i] is None: continue
        if fast_e[i-1] is None or slow_e[i-1] is None: continue
        cross_up   = fast_e[i] > slow_e[i] and fast_e[i-1] <= slow_e[i-1]
        cross_down = fast_e[i] < slow_e[i] and fast_e[i-1] >= slow_e[i-1]
        if cross_up:   signals.append((i, +1))
        elif cross_down: signals.append((i, -1))
    return signals


def run_ema_cross(candles):
    print("\n  [EMA Crossover]")
    best_results = []
    pairs = [(5,20),(5,50),(10,50),(10,100),(20,50),(20,100),(20,200),(50,200)]
    for fast, slow in pairs:
        for sl_pct in [0.003, 0.005, 0.0075, 0.01]:
            for tp_pct in [0.005, 0.0075, 0.01, 0.015, 0.02]:
                if tp_pct <= sl_pct: continue
                for max_bars in [12, 24, 48, 96]:
                    sigs = ema_cross_signals(candles, fast, slow)
                    r = ev_stats(candles, sigs, sl_pct, tp_pct, max_bars)
                    if r and r["ev"] > 0:
                        best_results.append({
                            **r, "fast": fast, "slow": slow,
                            "sl": sl_pct, "tp": tp_pct, "max_bars": max_bars,
                            "label": f"EMA({fast}x{slow}) SL{sl_pct*100:.2f}%/TP{tp_pct*100:.2f}% max{max_bars}b"
                        })
    if not best_results:
        print("  No profitable configurations found.")
        return None
    best_results.sort(key=lambda x: x["ev"], reverse=True)
    b = best_results[0]
    print(f"  Best: {b['label']}  N={b['n']}  WR={b['wr']:.1f}%  "
          f"BE={b['be_wr']:.1f}%  EV={b['ev']:+.4f}")

    fn = lambda c: ema_cross_signals(c, b["fast"], b["slow"])
    rows, evs, passes = walk_forward(candles, fn, b["sl"], b["tp"], b["max_bars"])
    return print_wf(rows, evs, passes, b["label"]), b


# ── Strategy 3: Bollinger Band Squeeze Breakout ────────────────────────────────
def bb_signals(candles, period, num_std, squeeze_pct):
    """
    Squeeze: BB width in lowest squeeze_pct% of recent N-bar widths.
    Release: width expands back; trade direction of price vs midline.
    """
    closes = [c["close"] for c in candles]
    signals = []
    width_history = []

    for i in range(period, len(candles)):
        window = closes[i - period:i]
        mid    = sum(window) / period
        std    = (sum((p - mid) ** 2 for p in window) / period) ** 0.5
        upper  = mid + num_std * std
        lower  = mid - num_std * std
        width  = (upper - lower) / mid if mid > 0 else 0
        width_history.append(width)

        if len(width_history) < 20:
            continue

        recent_widths = sorted(width_history[-100:])
        threshold_idx = max(0, int(len(recent_widths) * squeeze_pct / 100) - 1)
        squeeze_threshold = recent_widths[threshold_idx]

        was_squeeze = width_history[-2] <= squeeze_threshold if len(width_history) >= 2 else False
        is_squeeze  = width <= squeeze_threshold

        if was_squeeze and not is_squeeze:
            direction = +1 if closes[i] > mid else -1
            signals.append((i, direction))

    return signals


def run_bb_squeeze(candles):
    print("\n  [Bollinger Band Squeeze Breakout]")
    best_results = []
    for period in [20, 30, 50]:
        for num_std in [1.5, 2.0, 2.5]:
            for squeeze_pct in [20, 30]:
                for sl_pct in [0.003, 0.005, 0.0075, 0.01]:
                    for tp_pct in [0.005, 0.0075, 0.01, 0.015, 0.02]:
                        if tp_pct <= sl_pct: continue
                        for max_bars in [12, 24, 48, 96]:
                            sigs = bb_signals(candles, period, num_std, squeeze_pct)
                            r = ev_stats(candles, sigs, sl_pct, tp_pct, max_bars)
                            if r and r["ev"] > 0:
                                best_results.append({
                                    **r, "period": period, "std": num_std,
                                    "sqz": squeeze_pct, "sl": sl_pct,
                                    "tp": tp_pct, "max_bars": max_bars,
                                    "label": f"BB({period},{num_std},{squeeze_pct}%) "
                                             f"SL{sl_pct*100:.2f}%/TP{tp_pct*100:.2f}% max{max_bars}b"
                                })
    if not best_results:
        print("  No profitable configurations found.")
        return None
    best_results.sort(key=lambda x: x["ev"], reverse=True)
    b = best_results[0]
    print(f"  Best: {b['label']}  N={b['n']}  WR={b['wr']:.1f}%  "
          f"BE={b['be_wr']:.1f}%  EV={b['ev']:+.4f}")

    fn = lambda c: bb_signals(c, b["period"], b["std"], b["sqz"])
    rows, evs, passes = walk_forward(candles, fn, b["sl"], b["tp"], b["max_bars"])
    return print_wf(rows, evs, passes, b["label"]), b


# ── Strategy 4: MACD Momentum ──────────────────────────────────────────────────
def macd_signals(candles, fast=12, slow=26, signal=9):
    """
    MACD histogram crosses above 0 while MACD line > 0 -> MULTUP.
    MACD histogram crosses below 0 while MACD line < 0 -> MULTDOWN.
    """
    closes  = [c["close"] for c in candles]
    fast_e  = ema_series(closes, fast)
    slow_e  = ema_series(closes, slow)
    macd    = [f - s if f is not None and s is not None else None
               for f, s in zip(fast_e, slow_e)]
    sig_e   = ema_series(macd, signal)
    hist    = [m - s if m is not None and s is not None else None
               for m, s in zip(macd, sig_e)]

    signals = []
    for i in range(1, len(candles)):
        if hist[i] is None or hist[i-1] is None: continue
        if macd[i] is None: continue
        cross_up   = hist[i] > 0 and hist[i-1] <= 0 and macd[i] > 0
        cross_down = hist[i] < 0 and hist[i-1] >= 0 and macd[i] < 0
        if cross_up:    signals.append((i, +1))
        elif cross_down: signals.append((i, -1))
    return signals


def run_macd(candles):
    print("\n  [MACD Momentum]")
    best_results = []
    macd_params = [(12,26,9), (8,21,5), (5,13,5), (12,26,5)]
    for fast, slow, sig in macd_params:
        for sl_pct in [0.003, 0.005, 0.0075, 0.01]:
            for tp_pct in [0.005, 0.0075, 0.01, 0.015, 0.02]:
                if tp_pct <= sl_pct: continue
                for max_bars in [12, 24, 48, 96]:
                    sigs = macd_signals(candles, fast, slow, sig)
                    r = ev_stats(candles, sigs, sl_pct, tp_pct, max_bars)
                    if r and r["ev"] > 0:
                        best_results.append({
                            **r, "fast": fast, "slow": slow, "sig": sig,
                            "sl": sl_pct, "tp": tp_pct, "max_bars": max_bars,
                            "label": f"MACD({fast},{slow},{sig}) "
                                     f"SL{sl_pct*100:.2f}%/TP{tp_pct*100:.2f}% max{max_bars}b"
                        })
    if not best_results:
        print("  No profitable configurations found.")
        return None
    best_results.sort(key=lambda x: x["ev"], reverse=True)
    b = best_results[0]
    print(f"  Best: {b['label']}  N={b['n']}  WR={b['wr']:.1f}%  "
          f"BE={b['be_wr']:.1f}%  EV={b['ev']:+.4f}")

    fn = lambda c: macd_signals(c, b["fast"], b["slow"], b["sig"])
    rows, evs, passes = walk_forward(candles, fn, b["sl"], b["tp"], b["max_bars"])
    return print_wf(rows, evs, passes, b["label"]), b


# ── Main ──────────────────────────────────────────────────────────────────────
STRATEGY_FNS = {
    "donchian":  run_donchian,
    "ema_cross": run_ema_cross,
    "bb_squeeze": run_bb_squeeze,
    "macd":      run_macd,
}

def main():
    print("=" * 65)
    print("STRATEGY RESEARCH — Multiplier Strategies")
    print("=" * 65)

    # Load all candles
    print("\nLoading candles...")
    all_candles = {}
    for sym in SYMBOLS:
        all_candles[sym] = load_candles(sym)

    # Summary table
    summary = []   # (symbol, strategy, passes, n_folds, mean_ev, label)

    for sym in SYMBOLS:
        candles = all_candles[sym]
        t0 = datetime.fromtimestamp(candles[0]["epoch"],  tz=timezone.utc).strftime("%Y-%m-%d")
        t1 = datetime.fromtimestamp(candles[-1]["epoch"], tz=timezone.utc).strftime("%Y-%m-%d")
        print(f"\n{'='*65}")
        print(f"{sym}  |  {len(candles):,} bars  |  {t0} to {t1}")
        print(f"{'='*65}")

        for strat_name in STRATEGIES:
            result = STRATEGY_FNS[strat_name](candles)
            if result is not None:
                (passes, n_folds, mean_ev), b = result
                summary.append((sym, strat_name, passes, n_folds, mean_ev,
                                 b.get("label", strat_name)))
            else:
                summary.append((sym, strat_name, 0, 3, None, "-"))

    # Final summary
    print(f"\n\n{'='*65}")
    print("SUMMARY — Walk-forward results across all instruments & strategies")
    print(f"{'='*65}")
    print(f"{'Symbol':12} {'Strategy':12} {'WF':>6}  {'MeanEV':>8}  Verdict")
    print("-" * 65)
    for sym, strat, passes, n_folds, mean_ev, label in sorted(
            summary, key=lambda x: (x[2], x[4] or -99), reverse=True):
        verdict = "STRONG" if passes == n_folds and mean_ev and mean_ev > 0.05 else \
                  "PASS"   if passes == n_folds else \
                  "WEAK"   if passes >= n_folds - 1 and mean_ev and mean_ev > 0 else \
                  "FAIL"
        ev_str = f"{mean_ev:+.4f}" if mean_ev is not None else "  n/a  "
        print(f"{sym:12} {strat:12} {passes}/{n_folds}     {ev_str}  {verdict}")

    print("\nDone. STRONG = 3/3 folds + mean EV > 0.05")


if __name__ == "__main__":
    main()
