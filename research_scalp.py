"""
Scalp research: binary CALL/PUT on real forex + metals at 1-min and 5-min bars.

Tests 4 signal types:
  1. RSI Reversal   — RSI exits oversold/overbought zone
  2. EMA Crossover  — fast EMA crosses slow EMA
  3. RSI+EMA Filter — RSI pullback filtered by EMA slope direction
  4. BB Bounce      — price touches Bollinger Band, expect reversion

Binary sim: buy CALL/PUT at bar close, win if price moved in signal direction
            after hold_bars bars (1 bar = 1min or 5min contract expiry).

EV @ 80% payout: WR × 1.80 - 1  (need WR > 55.6% to be profitable)

Symbols: Gold, Silver, GBP/USD, EUR/USD, USD/JPY, AUD/USD
Cache: data/scalp/<symbol>_<gran>.json

Usage:
  python research_scalp.py          # runs 1-min + 5-min
  python research_scalp.py --1min   # only 1-min bars
  python research_scalp.py --5min   # only 5-min bars
"""

import asyncio
import json
import sys
from pathlib import Path

import websockets

LEGACY_WS  = "wss://ws.derivws.com/websockets/v3?app_id=1089"
CACHE_DIR  = Path("data/scalp")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PAYOUT     = 0.80    # conservative binary payout for real market pairs
FOLDS      = 3
MIN_TRADES = 10      # minimum trades per fold — below this we skip (noise)

SYMBOLS = [
    ("frxXAUUSD", "Gold    "),
    ("frxXAGUSD", "Silver  "),
    ("frxGBPUSD", "GBP/USD "),
    ("frxEURUSD", "EUR/USD "),
    ("frxUSDJPY", "USD/JPY "),
    ("frxAUDUSD", "AUD/USD "),
]

# Run both granularities unless overridden by CLI flag
RUN_1MIN = "--5min" not in sys.argv
RUN_5MIN = "--1min" not in sys.argv

GRAN_CONFIGS = []
if RUN_1MIN:
    GRAN_CONFIGS.append((60,  5000, "1-min"))
if RUN_5MIN:
    GRAN_CONFIGS.append((300, 5000, "5-min"))


# ── Data ───────────────────────────────────────────────────────────────────

async def fetch_closes(symbol: str, granularity: int) -> list[float]:
    cache = CACHE_DIR / f"{symbol}_{granularity}.json"
    if cache.exists():
        with open(cache) as f:
            return json.load(f)

    async with websockets.connect(LEGACY_WS, open_timeout=15) as ws:
        await ws.send(json.dumps({
            "ticks_history": symbol,
            "style":         "candles",
            "granularity":   granularity,
            "count":         5000,
            "end":           "latest",
            "req_id":        1,
        }))
        raw = await asyncio.wait_for(ws.recv(), timeout=30)
        msg = json.loads(raw)

    if msg.get("error"):
        raise RuntimeError(msg["error"]["message"])

    closes = [float(c["close"]) for c in msg["candles"]]
    with open(cache, "w") as f:
        json.dump(closes, f)
    return closes


# ── Indicators ─────────────────────────────────────────────────────────────

def rsi_series(closes, period):
    out = [None] * len(closes)
    if len(closes) < period + 1:
        return out
    ch = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    g  = sum(c for c in ch[:period] if c > 0) / period
    l  = sum(-c for c in ch[:period] if c < 0) / period
    for i in range(period, len(closes)):
        d = ch[i-1]
        g = (g * (period-1) + max(d, 0)) / period
        l = (l * (period-1) + max(-d, 0)) / period
        out[i] = 100.0 if l == 0 else round(100 - 100 / (1 + g/l), 2)
    return out


def ema_series(closes, period):
    out = [None] * len(closes)
    if len(closes) < period:
        return out
    k   = 2 / (period + 1)
    val = sum(closes[:period]) / period
    out[period-1] = val
    for i in range(period, len(closes)):
        val    = closes[i] * k + val * (1 - k)
        out[i] = val
    return out


def bb_series(closes, period, n_std):
    out = [None] * len(closes)
    for i in range(period - 1, len(closes)):
        w    = closes[i - period + 1: i + 1]
        mean = sum(w) / period
        std  = (sum((x - mean)**2 for x in w) / period) ** 0.5
        out[i] = (mean + n_std * std, mean - n_std * std)
    return out


# ── Signal generators ──────────────────────────────────────────────────────

def sig_rsi_reversal(closes, rsi_period, os_lvl):
    ob  = 100 - os_lvl
    rsi = rsi_series(closes, rsi_period)
    out = []
    for i in range(rsi_period + 1, len(closes)):
        if rsi[i] is None or rsi[i-1] is None:
            continue
        if rsi[i-1] < os_lvl <= rsi[i]:
            out.append((i, +1))
        elif rsi[i-1] > ob >= rsi[i]:
            out.append((i, -1))
    return out


def sig_ema_cross(closes, fast, slow):
    ef  = ema_series(closes, fast)
    es  = ema_series(closes, slow)
    out = []
    for i in range(slow + 1, len(closes)):
        if any(x is None for x in [ef[i], es[i], ef[i-1], es[i-1]]):
            continue
        if ef[i-1] <= es[i-1] and ef[i] > es[i]:
            out.append((i, +1))
        elif ef[i-1] >= es[i-1] and ef[i] < es[i]:
            out.append((i, -1))
    return out


def sig_rsi_ema(closes, rsi_period, rsi_entry, ema_period, slope_bars):
    rsi  = rsi_series(closes, rsi_period)
    ema  = ema_series(closes, ema_period)
    ob   = 100 - rsi_entry
    out  = []
    start = max(rsi_period, ema_period) + slope_bars + 1
    for i in range(start, len(closes)):
        if rsi[i] is None or ema[i] is None or ema[i - slope_bars] is None:
            continue
        up = ema[i] > ema[i - slope_bars]
        if up and rsi[i-1] >= rsi_entry > rsi[i]:
            out.append((i, +1))
        elif not up and rsi[i-1] <= ob < rsi[i]:
            out.append((i, -1))
    return out


def sig_bb_bounce(closes, bb_period, bb_std):
    bb  = bb_series(closes, bb_period, bb_std)
    out = []
    for i in range(bb_period, len(closes)):
        if bb[i] is None:
            continue
        upper, lower = bb[i]
        if closes[i] <= lower:
            out.append((i, +1))
        elif closes[i] >= upper:
            out.append((i, -1))
    return out


# ── Simulation ─────────────────────────────────────────────────────────────

def sim_binary(closes, signals, hold_bars):
    wins = total = 0
    for idx, direction in signals:
        if idx + hold_bars >= len(closes):
            continue
        total += 1
        if direction == +1 and closes[idx + hold_bars] > closes[idx]:
            wins += 1
        elif direction == -1 and closes[idx + hold_bars] < closes[idx]:
            wins += 1
    if total < MIN_TRADES:
        return None
    wr = wins / total
    return wr, total, wr * (1 + PAYOUT) - 1


def walk_forward(closes, signal_fn, hold_bars):
    fs = len(closes) // FOLDS
    out = []
    for f in range(FOLDS):
        s    = f * fs
        e    = s + fs if f < FOLDS - 1 else len(closes)
        fold = closes[s:e]
        out.append(sim_binary(fold, signal_fn(fold), hold_bars))
    return out


def classify(folds):
    valid = [r for r in folds if r is not None]
    if len(valid) < FOLDS:
        return "SKIP"
    passes = sum(1 for r in valid if r[2] > 0.05)
    if passes == FOLDS:
        return "STRONG"
    if passes >= FOLDS - 1:
        return "WEAK"
    return "FAIL"


# ── Sweep ──────────────────────────────────────────────────────────────────

def sweep(closes, granularity):
    results = []
    # Estimate trading days: forex ≈ 22 active hours/day
    active_secs = len(closes) * granularity
    trading_days = active_secs / (22 * 3600)

    def record(label, hold_b, folds):
        status = classify(folds)
        if status not in ("STRONG", "WEAK"):
            return
        valid = [r for r in folds if r]
        results.append({
            "signal":       label,
            "hold":         hold_b,
            "hold_min":     hold_b * granularity // 60,
            "folds":        folds,
            "mean_ev":      sum(r[2] for r in valid) / len(valid),
            "total_trades": sum(r[1] for r in valid),
            "per_day":      sum(r[1] for r in valid) / trading_days,
            "status":       status,
        })

    # RSI Reversal
    for rp in [7, 10, 14]:
        for os in [20, 25, 30]:
            for hb in [1, 2, 3]:
                folds = walk_forward(closes, lambda c, r=rp, o=os: sig_rsi_reversal(c, r, o), hb)
                record(f"RSI({rp:2d}) OS={os}", hb, folds)

    # EMA Crossover
    for fast, slow in [(5, 20), (5, 30), (8, 21), (10, 30), (10, 50)]:
        for hb in [1, 2, 3]:
            folds = walk_forward(closes, lambda c, f=fast, s=slow: sig_ema_cross(c, f, s), hb)
            record(f"EMA({fast:2d}/{slow:2d})", hb, folds)

    # RSI + EMA Filter
    for rp in [7, 10, 14]:
        for re in [40, 45]:
            for ep in [20, 50]:
                for sb in [3, 5]:
                    for hb in [1, 2, 3]:
                        folds = walk_forward(
                            closes,
                            lambda c, r=rp, e=re, p=ep, s=sb: sig_rsi_ema(c, r, e, p, s),
                            hb,
                        )
                        record(f"RSI+EMA({rp}/{re}/EMA{ep}/s{sb})", hb, folds)

    # BB Bounce
    for bp in [10, 20]:
        for bs in [1.5, 2.0]:
            for hb in [1, 2, 3]:
                folds = walk_forward(closes, lambda c, b=bp, s=bs: sig_bb_bounce(c, b, s), hb)
                record(f"BB({bp:2d}/{bs})", hb, folds)

    return results


# ── Main ───────────────────────────────────────────────────────────────────

async def main():
    print("=== Scalp Research: Real Forex + Metals — Binary CALL/PUT ===")
    print(f"Payout: {PAYOUT*100:.0f}% | Breakeven WR: {100/(1+PAYOUT):.1f}% | Folds: {FOLDS}\n")

    all_results = []

    for gran, candle_count, gran_label in GRAN_CONFIGS:
        print(f"\n{'='*60}")
        print(f"Granularity: {gran_label} bars ({candle_count} candles ≈ "
              f"{candle_count * gran // 3600:.0f}h of data)")
        print('='*60)

        for sym, label in SYMBOLS:
            print(f"\n[{sym}] {label.strip()} — fetching...")
            try:
                closes = await fetch_closes(sym, gran)
                print(f"  {len(closes)} candles | running sweep...")
            except Exception as e:
                print(f"  SKIP — {e}")
                continue

            results = sweep(closes, gran)
            for r in results:
                r["symbol"] = sym
                r["label"]  = label.strip()
                r["gran"]   = gran_label
            all_results.extend(results)

            strong = sum(1 for r in results if r["status"] == "STRONG")
            weak   = sum(1 for r in results if r["status"] == "WEAK")
            print(f"  → {strong} STRONG  {weak} WEAK")

    # ── Report ────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 90)
    print("SUMMARY — STRONG results only, sorted by mean EV")
    print("=" * 90)

    strong_all = [r for r in all_results if r["status"] == "STRONG"]
    strong_all.sort(key=lambda x: -x["mean_ev"])

    if not strong_all:
        print("\nNo STRONG strategies found.")
        print("Try checking WEAK results or adjusting MIN_TRADES threshold.")
        weak_all = [r for r in all_results if r["status"] == "WEAK"]
        weak_all.sort(key=lambda x: -x["mean_ev"])
        for r in weak_all[:20]:
            valid = [f for f in r["folds"] if f]
            wrs   = [f"{f[0]*100:.1f}%" for f in valid]
            print(f"  {r['gran']} {r['symbol']:12s} {r['signal']:30s} "
                  f"hold={r['hold_min']}min | WRs=[{', '.join(wrs)}] "
                  f"| MeanEV {r['mean_ev']:+.4f} | ~{r['per_day']:.1f}/day | WEAK")
    else:
        prev_sym = None
        for r in strong_all:
            if r["symbol"] != prev_sym:
                print(f"\n── {r['label']} ({r['symbol']}) ──")
                prev_sym = r["symbol"]
            valid = [f for f in r["folds"] if f]
            wrs   = [f"{f[0]*100:.1f}%" for f in valid]
            print(f"  {r['gran']:5s} {r['signal']:30s} hold={r['hold_min']:2d}min "
                  f"| WRs=[{', '.join(wrs)}] "
                  f"| MeanEV {r['mean_ev']:+.4f} "
                  f"| ~{r['per_day']:.1f}/day "
                  f"| {r['status']}")

    strong_count = len(strong_all)
    weak_count   = sum(1 for r in all_results if r["status"] == "WEAK")
    print(f"\n✓ {strong_count} STRONG, {weak_count} WEAK across all symbols and timeframes")

    if strong_count > 0:
        top = strong_all[0]
        print(f"\nTop pick: {top['label']} {top['signal']} "
              f"{top['gran']} hold={top['hold_min']}min | "
              f"MeanEV {top['mean_ev']:+.4f} | ~{top['per_day']:.1f} trades/day")


if __name__ == "__main__":
    asyncio.run(main())
