"""
Phase 3 research: Jump indices + 1HZ Volatility indices.

Tests RSI mean reversion (binary option, CALL/PUT) on:
  Jump 10, 25, 50, 75, 100 (JD10 / JD25 / JD50 / JD75 / JD100)
  Volatility 10/25/50/75/100 (1s) (1HZ10V / 1HZ25V / 1HZ50V / 1HZ75V / 1HZ100V)
  Crash 300/500/1000 + Boom 300/500/1000

Strategy: RSI exits extreme zone (RSI[i-1] < OS ≤ RSI[i] → BUY_RISE, vice versa)
Binary sim: win = price moved in signal direction after hold_bars bars
EV per trade @ 80% payout: WR × 1.80 - 1  (need WR > 55.6%)

Outputs: symbol, params, fold WRs, mean EV, trade count, status
Cache: data/jump/<symbol>_<gran>.json
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import websockets

LEGACY_WS = "wss://ws.derivws.com/websockets/v3?app_id=1089"
CACHE_DIR  = Path("data/jump")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PAYOUT_PCT = 0.80      # conservative binary payout for unknown symbols

SYMBOLS = [
    ("JD10",    "Jump 10"),
    ("JD25",    "Jump 25"),
    ("JD50",    "Jump 50"),
    ("JD75",    "Jump 75"),
    ("JD100",   "Jump 100"),
    ("1HZ10V",  "Vol 10 (1s)"),
    ("1HZ25V",  "Vol 25 (1s)"),
    ("1HZ50V",  "Vol 50 (1s)"),
    ("1HZ75V",  "Vol 75 (1s)"),
    ("1HZ100V", "Vol 100 (1s)"),
    ("CRASH300N","Crash 300"),
    ("CRASH500", "Crash 500"),
    ("CRASH1000","Crash 1000"),
    ("BOOM300N", "Boom 300"),
    ("BOOM500",  "Boom 500"),
    ("BOOM1000", "Boom 1000"),
]

# Granularity to use for candle data (5-min = 300)
GRAN = 300
CANDLE_COUNT = 5000   # ~17 days at 5-min bars

FOLDS = 3

# Param sweep
RSI_PERIODS    = [7, 10, 14]
OS_LEVELS      = [20, 25, 30]   # oversold (paired with 100-OS for overbought)
HOLD_BARS_LIST = [1, 2, 3]      # bars to hold the position (1=5min, 2=10min, 3=15min)


# ── Data fetching ──────────────────────────────────────────────────────────

async def _fetch_candles_ws(symbol: str, count: int, granularity: int) -> list[float]:
    cache_path = CACHE_DIR / f"{symbol}_{granularity}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    async with websockets.connect(LEGACY_WS, open_timeout=15) as ws:
        await ws.send(json.dumps({
            "ticks_history": symbol,
            "style": "candles",
            "granularity": granularity,
            "count": count,
            "end": "latest",
            "req_id": 1,
        }))
        raw = await asyncio.wait_for(ws.recv(), timeout=30)
        msg = json.loads(raw)

    if msg.get("error"):
        raise RuntimeError(msg["error"]["message"])

    closes = [float(c["close"]) for c in msg["candles"]]
    with open(cache_path, "w") as f:
        json.dump(closes, f)
    return closes


# ── Indicators ─────────────────────────────────────────────────────────────

def _rsi_series(closes: list[float], period: int) -> list[float | None]:
    out = [None] * len(closes)
    if len(closes) < period + 1:
        return out
    changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    avg_gain = sum(c for c in changes[:period] if c > 0) / period
    avg_loss = sum(abs(c) for c in changes[:period] if c < 0) / period
    for i in range(period, len(closes)):
        ch = changes[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(ch, 0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-ch, 0)) / period
        if avg_loss == 0:
            out[i] = 100.0
        else:
            out[i] = round(100 - 100 / (1 + avg_gain / avg_loss), 2)
    return out


# ── Signal generation ──────────────────────────────────────────────────────

def _signals(closes, rsi_period, os_lvl):
    """RSI exits oversold/overbought → (bar_index, direction +1/-1)."""
    ob_lvl = 100.0 - os_lvl
    rsi = _rsi_series(closes, rsi_period)
    sigs = []
    for i in range(rsi_period + 1, len(closes)):
        if rsi[i] is None or rsi[i - 1] is None:
            continue
        if rsi[i - 1] < os_lvl <= rsi[i]:
            sigs.append((i, +1))
        elif rsi[i - 1] > ob_lvl >= rsi[i]:
            sigs.append((i, -1))
    return sigs


# ── Walk-forward evaluation ────────────────────────────────────────────────

def _fold_result(closes, sigs_fn, hold_bars):
    """Run sigs on closes, simulate binary outcome, return (wins, total)."""
    sigs = sigs_fn(closes)
    wins = total = 0
    for idx, direction in sigs:
        if idx + hold_bars >= len(closes):
            continue
        total += 1
        entry = closes[idx]
        exit_ = closes[idx + hold_bars]
        if direction == +1 and exit_ > entry:
            wins += 1
        elif direction == -1 and exit_ < entry:
            wins += 1
    return wins, total


def walk_forward(closes, rsi_period, os_lvl, hold_bars, n_folds=3):
    fold_size = len(closes) // n_folds
    results = []
    for f in range(n_folds):
        start = f * fold_size
        end   = start + fold_size if f < n_folds - 1 else len(closes)
        fold  = closes[start:end]
        sigs_fn = lambda c, rp=rsi_period, os=os_lvl: _signals(c, rp, os)
        wins, total = _fold_result(fold, sigs_fn, hold_bars)
        if total == 0:
            results.append(None)
        else:
            wr = wins / total
            ev = wr * (1 + PAYOUT_PCT) - 1
            results.append((wr, total, ev))
    return results


# ── Main ───────────────────────────────────────────────────────────────────

async def research_symbol(symbol: str, label: str) -> list[dict]:
    try:
        closes = await _fetch_candles_ws(symbol, CANDLE_COUNT, GRAN)
    except Exception as e:
        print(f"  [{symbol}] SKIP — {e}")
        return []

    print(f"  [{symbol}] {len(closes)} candles loaded")
    best = []

    for rsi_p in RSI_PERIODS:
        for os_lvl in OS_LEVELS:
            for hold_b in HOLD_BARS_LIST:
                folds = walk_forward(closes, rsi_p, os_lvl, hold_b, FOLDS)
                valid = [r for r in folds if r is not None]
                if len(valid) < FOLDS:
                    continue

                mean_ev   = sum(r[2] for r in valid) / len(valid)
                total_trades = sum(r[1] for r in valid)
                pass_count   = sum(1 for r in valid if r[2] > 0.05)
                fold_wrs     = [round(r[0] * 100, 1) for r in valid]

                if pass_count == FOLDS:
                    status = "STRONG"
                elif pass_count >= FOLDS - 1:
                    status = "WEAK"
                else:
                    status = "FAIL"

                if status != "FAIL":
                    best.append({
                        "symbol": symbol,
                        "label": label,
                        "rsi_p": rsi_p,
                        "os": os_lvl,
                        "hold": hold_b,
                        "fold_wrs": fold_wrs,
                        "mean_ev": round(mean_ev, 4),
                        "total_trades": total_trades,
                        "status": status,
                    })

    return best


async def main():
    print(f"=== Phase 3: Jump / 1HZ / Crash / Boom Index Research ===")
    print(f"Granularity: {GRAN}s ({GRAN//60}min bars) | {CANDLE_COUNT} candles ≈ "
          f"{CANDLE_COUNT * GRAN // 86400} days | Folds: {FOLDS}")
    print(f"Payout assumption: {PAYOUT_PCT*100:.0f}% (conservative) | "
          f"Breakeven WR: {100/(1+PAYOUT_PCT):.1f}%\n")

    all_results = []
    for sym, label in SYMBOLS:
        print(f"\n[{sym}] {label}")
        results = await research_symbol(sym, label)
        all_results.extend(results)

    print("\n\n" + "=" * 80)
    print("RESULTS SUMMARY (non-FAIL only, sorted by mean EV)")
    print("=" * 80)

    strong = [r for r in all_results if r["status"] == "STRONG"]
    weak   = [r for r in all_results if r["status"] == "WEAK"]

    for group, label in [(strong, "STRONG — 3/3 folds EV>0.05"), (weak, "WEAK — 2/3 folds")]:
        if not group:
            continue
        print(f"\n── {label} ──")
        group.sort(key=lambda x: -x["mean_ev"])
        for r in group:
            trades_per_day = r["total_trades"] / (CANDLE_COUNT * GRAN / 86400) / FOLDS
            print(
                f"  {r['symbol']:12s} RSI({r['rsi_p']:2d}) OS={r['os']:2d} hold={r['hold']}bars "
                f"| WRs={r['fold_wrs']} | MeanEV {r['mean_ev']:+.4f} "
                f"| {r['total_trades']} trades (~{trades_per_day:.1f}/day) | {r['status']}"
            )

    if not strong and not weak:
        print("\nNo viable strategies found across all symbols.")
        print("Possible causes:")
        print("  - Symbols unavailable on Deriv (skipped above)")
        print("  - All RSI sweeps below breakeven WR for these indices")
    else:
        print(f"\n✓ {len(strong)} STRONG, {len(weak)} WEAK strategies found")
        if strong:
            best = strong[0]
            hold_min = best["hold"] * GRAN // 60
            print(f"\nTop pick: {best['symbol']} RSI({best['rsi_p']}) OS={best['os']} "
                  f"hold={hold_min}min | MeanEV {best['mean_ev']:+.4f}")
            print(f"  → Binary CALL/PUT | SL: none (duration-based) | Payout: verify in live API")


if __name__ == "__main__":
    asyncio.run(main())
