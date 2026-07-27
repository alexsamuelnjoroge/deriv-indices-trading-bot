"""
Multiplier strategy backtest — EMA trend filter + RSI pullback entry.

Strategy logic:
  - Compute EMA(ema_period) on bar closes as the trend direction signal.
  - EMA slope = EMA[i] - EMA[i - slope_bars] determines if trend is up or down.
  - Entry signal (uptrend):  EMA rising  AND RSI(rsi_period) < rsi_entry → MULTUP
  - Entry signal (downtrend): EMA falling AND RSI(rsi_period) > (100-rsi_entry) → MULTDOWN
  - SL and TP expressed as % of entry price — instrument-agnostic, works for
    EUR/USD, Gold, or any symbol.
  - Commission = 2% of stake (flat, paid regardless of outcome).

Economics with x100 multiplier:
  - 1% price move = 100% of stake profit (at x100)
  - Commission = 2% of stake ($0.20 on $10 stake)
  - With 1.0% TP / 0.5% SL: net win = 98% stake, net loss = 52% stake
  - Breakeven WR = 52/(98+52) = 34.7%

Usage:
  python backtest_multiplier.py                          # frxXAUUSD 1h, default sweep
  python backtest_multiplier.py --symbol frxEURUSD
  python backtest_multiplier.py --granularity 3600       # 1-hour bars (default)
  python backtest_multiplier.py --granularity 14400      # 4-hour bars
  python backtest_multiplier.py --no-fresh               # reuse cached candles
"""

import argparse
import asyncio
import json
import math
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--symbol",      default="frxXAUUSD")
parser.add_argument("--granularity", type=int, default=3600,
                    help="Bar size in seconds: 3600=1h (default), 14400=4h, 86400=daily")
parser.add_argument("--no-fresh",    action="store_true", help="Reuse cached candles")
args = parser.parse_args()

SYMBOL      = args.symbol
GRANULARITY = args.granularity
CACHE_FILE  = f"cache_{SYMBOL}_{GRANULARITY}s_candles.json"

# ── Multiplier contract parameters ────────────────────────────────────────────
MULTIPLIER     = 100     # x100 multiplier (available on EUR/USD and Gold)
COMMISSION_PCT = 0.02    # 2% of stake, paid upfront regardless of outcome

# ── Fetch candles via legacy Deriv WS (more history than new OTP endpoint) ────
async def fetch_candles(symbol, granularity, count_per_call=5000, n_calls=8):
    import websockets, json as _j
    app_id = os.getenv("DERIV_APP_ID", "1089")
    ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"

    all_candles, seen, end = [], set(), "latest"
    async with websockets.connect(ws_url) as ws:
        for chunk_i in range(n_calls):
            await ws.send(_j.dumps({
                "ticks_history": symbol, "style": "candles",
                "granularity": granularity, "count": count_per_call,
                "end": end, "req_id": chunk_i + 1,
            }))
            msg = _j.loads(await asyncio.wait_for(ws.recv(), timeout=30))
            if msg.get("error"):
                print(f"  API error: {msg['error'].get('message')}")
                break
            candles = msg.get("candles", [])
            if not candles:
                break
            new = [c for c in candles if c["epoch"] not in seen]
            for c in new:
                seen.add(c["epoch"])
            all_candles = new + all_candles
            end = candles[0]["epoch"] - 1
            print(f"  Chunk {chunk_i+1}: +{len(new)} bars  (total {len(all_candles):,})")
            if len(new) < count_per_call:
                break
    all_candles.sort(key=lambda c: c["epoch"])
    return all_candles


def load_candles():
    if args.no_fresh and os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            data = json.load(f)
        print(f"  Loaded {len(data):,} bars from cache")
        return data
    print(f"Fetching {SYMBOL} {GRANULARITY}s candles…")
    candles = asyncio.run(fetch_candles(SYMBOL, GRANULARITY))
    with open(CACHE_FILE, "w") as f:
        json.dump(candles, f)
    return candles


# ── Indicators ─────────────────────────────────────────────────────────────────
def compute_ema(closes, period):
    ema = [None] * len(closes)
    alpha = 2 / (period + 1)
    for i, p in enumerate(closes):
        if i == 0:
            ema[i] = p
        else:
            ema[i] = alpha * p + (1 - alpha) * ema[i - 1]
    # Mark the first (period-1) values as None (not yet warmed up)
    for i in range(period - 1):
        ema[i] = None
    return ema


def compute_rsi(closes, period):
    rsi = [None] * len(closes)
    if len(closes) < period + 1:
        return rsi
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains  = [max(0, d) for d in deltas]
    losses = [max(0, -d) for d in deltas]
    avg_g = sum(gains[:period]) / period
    avg_l = sum(losses[:period]) / period
    for i in range(period, len(closes)):
        g = (avg_g * (period - 1) + gains[i-1]) / period
        l = (avg_l * (period - 1) + losses[i-1]) / period
        avg_g, avg_l = g, l
        rs = g / l if l > 0 else float("inf")
        rsi[i] = 100 - 100 / (1 + rs)
    return rsi


# ── Trade simulation ──────────────────────────────────────────────────────────
def simulate_trade(candles, entry_idx, direction, sl_pct, tp_pct, max_bars):
    """
    direction: +1 = MULTUP (long), -1 = MULTDOWN (short)
    sl_pct / tp_pct: decimal fractions of entry price (e.g. 0.005 = 0.5%)
    Returns ('tp', pct_move), ('sl', pct_move), or ('timeout', pct_move)
    where pct_move is signed (positive = price went up).
    """
    entry = candles[entry_idx]["open"]
    tp_price = entry * (1 + direction * tp_pct)
    sl_price = entry * (1 - direction * sl_pct)

    for i in range(entry_idx, min(entry_idx + max_bars, len(candles))):
        high = candles[i]["high"]
        low  = candles[i]["low"]

        if direction == 1:   # long: SL below, TP above
            if low  <= sl_price: return "sl",      (sl_price - entry) / entry
            if high >= tp_price: return "tp",      (tp_price - entry) / entry
        else:                 # short: SL above, TP below
            if high >= sl_price: return "sl",      (sl_price - entry) / entry
            if low  <= tp_price: return "tp",      (tp_price - entry) / entry

    close = candles[min(entry_idx + max_bars - 1, len(candles) - 1)]["close"]
    return "timeout", (close - entry) / entry


# ── Core backtest ─────────────────────────────────────────────────────────────
def run_backtest(candles, ema_period, slope_bars, rsi_period, rsi_entry,
                 sl_pct, tp_pct, max_bars, min_n=20):
    """
    rsi_entry: RSI level below which we enter MULTUP in uptrend (e.g. 40).
               Mirror is (100 - rsi_entry) for MULTDOWN in downtrend.
    Returns dict with trade stats, or None if too few trades.
    """
    closes = [c["close"] for c in candles]
    ema    = compute_ema(closes, ema_period)
    rsi    = compute_rsi(closes, rsi_period)

    wins = losses = timeouts = 0
    total_ev = 0.0
    last_exit_bar = -1

    warmup = max(ema_period, rsi_period, slope_bars) + 2

    for i in range(warmup, len(candles) - max_bars - 1):
        if i <= last_exit_bar:
            continue

        e   = ema[i]
        r   = rsi[i]
        e_prev = ema[i - slope_bars]
        if e is None or r is None or e_prev is None:
            continue

        slope_up   = e > e_prev
        slope_down = e < e_prev

        direction = None
        if slope_up   and r < rsi_entry:
            direction = +1   # uptrend pullback → MULTUP
        elif slope_down and r > (100 - rsi_entry):
            direction = -1   # downtrend bounce → MULTDOWN

        if direction is None:
            continue

        entry_bar = i + 1
        if entry_bar >= len(candles):
            continue

        outcome, pct_move = simulate_trade(
            candles, entry_bar, direction, sl_pct, tp_pct, max_bars)

        # P&L as fraction of stake
        if outcome == "tp":
            wins += 1
            ev = MULTIPLIER * tp_pct - COMMISSION_PCT
        elif outcome == "sl":
            losses += 1
            ev = -(MULTIPLIER * sl_pct + COMMISSION_PCT)
        else:
            timeouts += 1
            # Actual P&L from price at timeout close
            ev = MULTIPLIER * pct_move * direction - COMMISSION_PCT

        total_ev += ev
        last_exit_bar = entry_bar + max_bars if outcome == "timeout" else entry_bar

    n = wins + losses + timeouts
    if n < min_n:
        return None
    wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    return {
        "n": n, "wins": wins, "losses": losses, "timeouts": timeouts,
        "wr": wr, "ev_per_trade": total_ev / n,
    }


# ── Parameter sweep ───────────────────────────────────────────────────────────
def sweep(candles):
    results = []

    for ema_period in [20, 50, 100, 200]:
        for slope_bars in [5, 10]:
            for rsi_period in [7, 10, 14]:
                for rsi_entry in [30, 35, 40, 45]:
                    for sl_pct in [0.003, 0.005, 0.0075, 0.01]:
                        for tp_pct in [0.005, 0.0075, 0.01, 0.015, 0.02]:
                            if tp_pct <= sl_pct:
                                continue
                            for max_bars in [12, 24, 48, 96]:
                                r = run_backtest(candles, ema_period, slope_bars,
                                                 rsi_period, rsi_entry,
                                                 sl_pct, tp_pct, max_bars)
                                if r is None:
                                    continue
                                rr = tp_pct / sl_pct
                                be_wr = (MULTIPLIER * sl_pct + COMMISSION_PCT) / (
                                    MULTIPLIER * tp_pct - COMMISSION_PCT +
                                    MULTIPLIER * sl_pct + COMMISSION_PCT) * 100
                                results.append({
                                    **r,
                                    "ema": ema_period, "slope_bars": slope_bars,
                                    "rsi_p": rsi_period, "rsi_entry": rsi_entry,
                                    "sl_pct": sl_pct, "tp_pct": tp_pct,
                                    "max_bars": max_bars,
                                    "rr": rr, "be_wr": be_wr,
                                })
    results.sort(key=lambda x: x["ev_per_trade"], reverse=True)
    return results


def print_table(results, top_n=25):
    print(f"\n{'EMA':>4} {'Sl':>2} {'RSI':>3} {'Entry':>5}  "
          f"{'SL%':>5} {'TP%':>5} {'R:R':>4} {'MaxB':>4}  "
          f"{'N':>4} {'WR':>6} {'BEwr':>6} {'EV/tr':>8}  Notes")
    print("-" * 90)
    for r in results[:top_n]:
        notes = "PROFITABLE" if r["ev_per_trade"] > 0 else (
                "near b/e"  if r["ev_per_trade"] > -0.05 else "")
        above = "**" if r["wr"] > r["be_wr"] else "  "
        print(
            f"{r['ema']:>4} {r['slope_bars']:>2} {r['rsi_p']:>3} {r['rsi_entry']:>5}  "
            f"{r['sl_pct']*100:>4.2f}% {r['tp_pct']*100:>4.2f}% {r['rr']:>4.1f} {r['max_bars']:>4}  "
            f"{r['n']:>4} {r['wr']:>5.1f}% {r['be_wr']:>5.1f}% {r['ev_per_trade']:>+8.4f}  "
            f"{above}{notes}"
        )


# ── Walk-forward validation ───────────────────────────────────────────────────
def walk_forward(candles, best_params, n_splits=3):
    """Split data into n_splits folds and validate best params on each."""
    split_size = len(candles) // n_splits
    print(f"\nWalk-forward validation ({n_splits} folds of ~{split_size} bars each):")
    print(f"{'Fold':>5}  {'Period':20}  {'N':>4}  {'WR':>6}  {'EV/tr':>8}  {'Result':10}")
    print("-" * 65)
    all_ev = []
    for fold in range(n_splits):
        start = fold * split_size
        end   = (fold + 1) * split_size if fold < n_splits - 1 else len(candles)
        chunk = candles[start:end]
        t0 = datetime.fromtimestamp(chunk[0]["epoch"],  tz=timezone.utc).strftime("%m/%d")
        t1 = datetime.fromtimestamp(chunk[-1]["epoch"], tz=timezone.utc).strftime("%m/%d")
        r = run_backtest(chunk, min_n=5, **best_params)
        if r is None:
            print(f"{fold+1:>5}  {t0}–{t1:20}  too few signals (<5)")
            continue
        result_str = "PASS" if r["ev_per_trade"] > 0 else "FAIL"
        print(f"{fold+1:>5}  {t0}–{t1:20}  {r['n']:>4}  {r['wr']:>5.1f}%  "
              f"{r['ev_per_trade']:>+8.4f}  {result_str}")
        all_ev.append(r["ev_per_trade"])
    if all_ev:
        print(f"\n  Mean EV across folds: {sum(all_ev)/len(all_ev):+.4f}")
        print(f"  Folds with positive EV: {sum(1 for e in all_ev if e > 0)}/{len(all_ev)}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    candles = load_candles()
    n = len(candles)
    if n < 200:
        print("Not enough data.")
        return

    t0 = datetime.fromtimestamp(candles[0]["epoch"],  tz=timezone.utc)
    t1 = datetime.fromtimestamp(candles[-1]["epoch"], tz=timezone.utc)

    bar_label = (f"{GRANULARITY//3600}h" if GRANULARITY >= 3600
                 else f"{GRANULARITY//60}m")
    print(f"\n{SYMBOL}  {bar_label} bars  |  {n:,} bars")
    print(f"Period: {t0.strftime('%Y-%m-%d')} to {t1.strftime('%Y-%m-%d')}")
    print(f"Commission: {COMMISSION_PCT*100:.0f}% of stake (flat per trade)")
    print(f"Multiplier: x{MULTIPLIER}")
    print()

    # ── Economics reminder ──
    sl_ex, tp_ex = 0.005, 0.01
    net_win  = MULTIPLIER * tp_ex  - COMMISSION_PCT
    net_loss = MULTIPLIER * sl_ex  + COMMISSION_PCT
    be_wr    = net_loss / (net_win + net_loss) * 100
    print(f"Example: SL={sl_ex*100:.1f}%  TP={tp_ex*100:.1f}%  =>  "
          f"net win={net_win*100:.0f}% stake  |  net loss={net_loss*100:.0f}% stake  |  "
          f"breakeven WR = {be_wr:.1f}%")

    # ── Baseline ──
    print("\nBaseline (EMA50/slope10, RSI10<40, SL0.5%, TP1.0%, max24bars):")
    r = run_backtest(candles, 50, 10, 10, 40, 0.005, 0.010, 24)
    if r:
        net_w  = MULTIPLIER * 0.010 - COMMISSION_PCT
        net_l  = MULTIPLIER * 0.005 + COMMISSION_PCT
        be     = net_l / (net_w + net_l) * 100
        print(f"  N={r['n']}  Wins={r['wins']}  Losses={r['losses']}  Timeouts={r['timeouts']}")
        print(f"  WR={r['wr']:.1f}%  (breakeven {be:.1f}%)  EV/trade={r['ev_per_trade']:+.4f}")
        if r["ev_per_trade"] > 0:
            print("  ** PROFITABLE at baseline parameters **")
        else:
            print(f"  Gap to breakeven: {be - r['wr']:.1f}pp")
    else:
        print("  Too few signals at baseline.")

    # ── Full sweep ──
    print("\nRunning parameter sweep…  (this may take 30-60s)")
    all_results = sweep(candles)

    profitable = [x for x in all_results if x["ev_per_trade"] > 0]
    print(f"\nSweep complete: {len(all_results)} combinations  |  "
          f"profitable: {len(profitable)}")
    print_table(all_results)

    if not profitable:
        print("\nNo profitable combinations found.")
        print("Conclusion: EMA+RSI pullback does NOT show reliable edge on this")
        print("symbol/timeframe with these parameter ranges.")
        return

    # ── Best config summary ──
    best = profitable[0]
    print(f"\n{'='*60}")
    print("BEST CONFIGURATION")
    print(f"{'='*60}")
    print(f"  Strategy : EMA({best['ema']}) slope/{best['slope_bars']}bars  +  "
          f"RSI({best['rsi_p']}) entry < {best['rsi_entry']}")
    print(f"  SL / TP  : {best['sl_pct']*100:.2f}% / {best['tp_pct']*100:.2f}%  "
          f"({best['rr']:.1f}:1 R:R)")
    hold_min = best['max_bars'] * GRANULARITY // 60
    hold_str = f"{hold_min//60}h" if hold_min % 60 == 0 else f"{hold_min}m"
    print(f"  Max hold : {best['max_bars']} bars  ({hold_str})")
    print(f"  N={best['n']}  WR={best['wr']:.1f}%  (breakeven {best['be_wr']:.1f}%)  "
          f"EV={best['ev_per_trade']:+.4f}/trade")
    sl_usd = best["sl_pct"]  * MULTIPLIER * 10
    tp_usd = best["tp_pct"]  * MULTIPLIER * 10
    print(f"\n  In dollars ($10 stake, x{MULTIPLIER}):")
    print(f"  SL=${sl_usd:.2f}  |  TP=${tp_usd:.2f}  |  Commission=$0.20")
    ev_usd = best["ev_per_trade"] * 10
    print(f"  Expected return per trade: ${ev_usd:+.2f}  "
          f"({best['ev_per_trade']*100:+.1f}% of stake)")

    # ── Walk-forward check ──
    # Use the highest-N profitable config for WF — best EV often has too few signals
    high_n_profitable = sorted(profitable, key=lambda x: x["n"], reverse=True)
    wf_cfg = high_n_profitable[0]
    wf_params = {
        "ema_period": wf_cfg["ema"], "slope_bars": wf_cfg["slope_bars"],
        "rsi_period": wf_cfg["rsi_p"], "rsi_entry": wf_cfg["rsi_entry"],
        "sl_pct": wf_cfg["sl_pct"], "tp_pct": wf_cfg["tp_pct"],
        "max_bars": wf_cfg["max_bars"],
    }
    print(f"\n(Walk-forward uses highest-N profitable config: "
          f"EMA{wf_cfg['ema']}/RSI{wf_cfg['rsi_p']}<{wf_cfg['rsi_entry']} "
          f"SL{wf_cfg['sl_pct']*100:.2f}%/TP{wf_cfg['tp_pct']*100:.2f}% "
          f"N={wf_cfg['n']})")
    walk_forward(candles, wf_params, n_splits=3)
    print()


if __name__ == "__main__":
    main()
