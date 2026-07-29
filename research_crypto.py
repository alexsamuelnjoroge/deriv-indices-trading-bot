"""
Crypto research — Deriv binary CALL/PUT on crypto pairs.

Phase 1: Discover available crypto symbols, test valid durations + actual payouts.
Phase 2: Fetch historical candles and sweep RSI/EMA strategies via walk-forward.

Usage:
  python3 research_crypto.py            # full run (discovery + sweep)
  python3 research_crypto.py --discover # phase 1 only
  python3 research_crypto.py --sweep    # phase 2 only (uses cached data)
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import aiohttp
import websockets
from dotenv import load_dotenv

load_dotenv()

TOKEN     = os.getenv("DERIV_API_TOKEN")
APP_KEY   = os.getenv("DERIV_APP_KEY", "")
REST_BASE = "https://api.derivws.com"
LEGACY_WS = "wss://ws.derivws.com/websockets/v3?app_id=1089"

CACHE_DIR = Path("data/crypto")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

GRAN       = 300    # 5-min bars
CANDLES    = 5000
FOLDS      = 3
MIN_TRADES = 10

# Test these durations for each symbol
TEST_DURATIONS = [
    (5,  "m", "5min"),
    (10, "m", "10min"),
    (15, "m", "15min"),
    (30, "m", "30min"),
    (60, "m", "60min"),
]

RUN_DISCOVER = "--sweep"  not in sys.argv
RUN_SWEEP    = "--discover" not in sys.argv


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1 — Discovery
# ═══════════════════════════════════════════════════════════════════════════

async def get_ws_url() -> str:
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type":  "application/json",
        "Deriv-App-ID":  APP_KEY,
    }
    async with aiohttp.ClientSession() as s:
        async with s.get(f"{REST_BASE}/trading/v1/options/accounts", headers=headers) as r:
            data = await r.json()
            account_id = data["data"][0]["account_id"]
        async with s.post(
            f"{REST_BASE}/trading/v1/options/accounts/{account_id}/otp",
            headers=headers,
        ) as r:
            data = await r.json()
            return data["data"]["url"]


async def fetch_active_crypto(ws) -> list[dict]:
    """Return active_symbols filtered to crypto pairs."""
    await ws.send(json.dumps({
        "active_symbols": "brief",
        "product_type":   "basic",
        "req_id":         1,
    }))
    raw = await asyncio.wait_for(ws.recv(), timeout=30)
    msg = json.loads(raw)

    if msg.get("error"):
        raise RuntimeError(msg["error"]["message"])

    symbols = msg.get("active_symbols", [])
    # Deriv crypto symbols have market = "cryptocurrency"
    crypto = [s for s in symbols if s.get("market") == "cryptocurrency"]
    return crypto


async def test_proposal(ws, symbol: str, duration: int, unit: str, req_id: int):
    await ws.send(json.dumps({
        "proposal":          1,
        "req_id":            req_id,
        "amount":            1.0,
        "basis":             "stake",
        "contract_type":     "CALL",
        "currency":          "USD",
        "duration":          duration,
        "duration_unit":     unit,
        "underlying_symbol": symbol,
    }))
    raw = await asyncio.wait_for(ws.recv(), timeout=30)
    msg = json.loads(raw)

    if msg.get("error"):
        return None, msg["error"].get("code", "ERROR")

    p = msg.get("proposal", {})
    try:
        payout_pct = round(
            (float(p["payout"]) - float(p["ask_price"])) / float(p["ask_price"]) * 100, 1
        )
    except Exception:
        payout_pct = None
    return payout_pct, None


async def phase1_discover() -> dict:
    """
    Returns {symbol: {"display": str, "payout": float, "min_duration_m": int}}
    for all crypto symbols that accept at least one duration.
    """
    print("=" * 70)
    print("PHASE 1 — Discovering crypto symbols + valid durations")
    print("=" * 70)

    ws_url = await get_ws_url()

    viable: dict[str, dict] = {}

    async with websockets.connect(ws_url, open_timeout=20) as ws:
        crypto_syms = await fetch_active_crypto(ws)
        print(f"Found {len(crypto_syms)} crypto symbols on Deriv\n")

        req_id = 10
        for sym_info in crypto_syms:
            sym     = sym_info["symbol"]
            display = sym_info.get("display_name", sym)
            print(f"  {display:30s} ({sym})")

            best_payout   = None
            min_duration  = None

            for dur, unit, label in TEST_DURATIONS:
                payout_pct, err = await test_proposal(ws, sym, dur, unit, req_id)
                req_id += 1
                if payout_pct is not None:
                    print(f"    OK   {label:6s} | payout {payout_pct:.1f}%")
                    if best_payout is None or payout_pct > best_payout:
                        best_payout = payout_pct
                    if min_duration is None:
                        min_duration = dur
                else:
                    print(f"    FAIL {label:6s} | {err}")

            if best_payout is not None:
                viable[sym] = {
                    "display":      display,
                    "payout":       round(best_payout / 100, 4),
                    "min_dur_m":    min_duration,
                }
                print(f"  → VIABLE: min={min_duration}min payout={best_payout:.1f}%\n")
            else:
                print(f"  → NO valid duration\n")

    # Cache discovery results
    cache = CACHE_DIR / "discovery.json"
    with open(cache, "w") as f:
        json.dump(viable, f, indent=2)
    print(f"Discovery saved to {cache}")
    return viable


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 — Research sweep
# ═══════════════════════════════════════════════════════════════════════════

async def fetch_closes(symbol: str) -> list[float]:
    cache = CACHE_DIR / f"{symbol}_{GRAN}.json"
    if cache.exists():
        with open(cache) as f:
            return json.load(f)

    async with websockets.connect(LEGACY_WS, open_timeout=15) as ws:
        await ws.send(json.dumps({
            "ticks_history": symbol,
            "style":         "candles",
            "granularity":   GRAN,
            "count":         CANDLES,
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


def sig_rsi_ema(closes, rsi_period, rsi_entry, ema_period, slope_bars):
    rsi   = rsi_series(closes, rsi_period)
    ema   = ema_series(closes, ema_period)
    ob    = 100 - rsi_entry
    out   = []
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


# ── Walk-forward simulation ─────────────────────────────────────────────────

def sim_binary(closes, signals, hold_bars, payout):
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
    return wr, total, wr * (1 + payout) - 1


def walk_forward(closes, signal_fn, hold_bars, payout):
    fs  = len(closes) // FOLDS
    out = []
    for f in range(FOLDS):
        s    = f * fs
        e    = s + fs if f < FOLDS - 1 else len(closes)
        fold = closes[s:e]
        out.append(sim_binary(fold, signal_fn(fold), hold_bars, payout))
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


def sweep(closes, payout, min_dur_m):
    results      = []
    trading_days = len(closes) * GRAN / (24 * 3600)  # crypto = 24h/day

    # min hold bars = how many 5-min bars = min_duration_minutes / 5
    min_hb = max(1, min_dur_m // (GRAN // 60))
    # test up to 3× the minimum (e.g. if min=15min → 3/6/9 bars = 15/30/45min)
    hold_range = [min_hb, min_hb * 2, min_hb * 3]

    def record(label, hold_b, folds):
        status = classify(folds)
        if status not in ("STRONG", "WEAK"):
            return
        valid = [r for r in folds if r]
        results.append({
            "signal":    label,
            "hold_bars": hold_b,
            "hold_min":  hold_b * GRAN // 60,
            "folds":     folds,
            "mean_ev":   sum(r[2] for r in valid) / len(valid),
            "n_trades":  sum(r[1] for r in valid),
            "per_day":   sum(r[1] for r in valid) / trading_days,
            "status":    status,
            "payout":    payout,
        })

    # RSI reversal — wider bands for volatile crypto
    for rp in [7, 10, 14]:
        for os in [20, 25, 30]:
            for hb in hold_range:
                folds = walk_forward(closes,
                    lambda c, r=rp, o=os: sig_rsi_reversal(c, r, o), hb, payout)
                record(f"RSI({rp:2d}) OS={os}", hb, folds)

    # RSI + EMA slope filter
    for rp in [7, 10, 14]:
        for re in [40, 45, 50]:
            for ep in [20, 50, 100]:
                for sb in [3, 5]:
                    for hb in hold_range:
                        folds = walk_forward(closes,
                            lambda c, r=rp, e=re, p=ep, s=sb:
                                sig_rsi_ema(c, r, e, p, s), hb, payout)
                        record(f"RSI+EMA({rp}/{re}/EMA{ep}/s{sb})", hb, folds)

    # BB bounce
    for bp in [10, 20]:
        for bs in [1.5, 2.0, 2.5]:
            for hb in hold_range:
                folds = walk_forward(closes,
                    lambda c, b=bp, s=bs: sig_bb_bounce(c, b, s), hb, payout)
                record(f"BB({bp:2d}/{bs})", hb, folds)

    return results


async def phase2_sweep(viable: dict):
    print("\n" + "=" * 70)
    print("PHASE 2 — Strategy sweep on viable crypto symbols")
    print("=" * 70 + "\n")

    all_results = []

    for sym, info in viable.items():
        display  = info["display"]
        payout   = info["payout"]
        min_dur  = info["min_dur_m"]
        be_wr    = round(100 / (1 + payout), 1)

        print(f"[{sym}] {display}  payout={payout*100:.1f}%  BE WR={be_wr}%  min={min_dur}min")
        try:
            closes = await fetch_closes(sym)
            print(f"  {len(closes)} candles (~{len(closes)*GRAN//86400:.0f} days) — sweeping...")
        except Exception as e:
            print(f"  SKIP — {e}\n")
            continue

        results = sweep(closes, payout, min_dur)
        for r in results:
            r["symbol"]  = sym
            r["display"] = display

        all_results.extend(results)

        strong = sum(1 for r in results if r["status"] == "STRONG")
        weak   = sum(1 for r in results if r["status"] == "WEAK")
        print(f"  → {strong} STRONG  {weak} WEAK\n")

    # ── Report ─────────────────────────────────────────────────────────────
    print("=" * 90)
    print("RESULTS — STRONG first, sorted by MeanEV")
    print("=" * 90)

    strong_all = sorted(
        [r for r in all_results if r["status"] == "STRONG"],
        key=lambda x: -x["mean_ev"],
    )
    weak_top = sorted(
        [r for r in all_results if r["status"] == "WEAK"],
        key=lambda x: -x["mean_ev"],
    )[:15]

    for group, title in [(strong_all, "STRONG — 3/3 folds EV>0.05"),
                         (weak_top,   "WEAK   — 2/3 folds (top 15)")]:
        if not group:
            continue
        print(f"\n── {title} ──")
        prev = None
        for r in group:
            if r["symbol"] != prev:
                be = round(100 / (1 + r["payout"]), 1)
                print(f"\n  {r['display']} ({r['symbol']})  "
                      f"payout={r['payout']*100:.1f}%  BE={be}%")
                prev = r["symbol"]
            valid = [f for f in r["folds"] if f]
            wrs   = " / ".join(f"{f[0]*100:.1f}%" for f in valid)
            print(f"    {r['signal']:38s} hold={r['hold_min']:3d}min "
                  f"| [{wrs}] "
                  f"| MeanEV {r['mean_ev']:+.4f} "
                  f"| ~{r['per_day']:.1f}/day "
                  f"| {r['status']}")

    total_strong = len(strong_all)
    total_weak   = len([r for r in all_results if r["status"] == "WEAK"])
    print(f"\n✓ {total_strong} STRONG  {total_weak} WEAK  across all crypto symbols")

    if strong_all:
        top = strong_all[0]
        print(f"\nTop pick: {top['display']} — {top['signal']} "
              f"hold={top['hold_min']}min "
              f"| MeanEV {top['mean_ev']:+.4f} "
              f"| ~{top['per_day']:.1f}/day "
              f"| payout {top['payout']*100:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    if RUN_DISCOVER:
        viable = await phase1_discover()
    else:
        cache = CACHE_DIR / "discovery.json"
        if not cache.exists():
            print("No discovery cache found — run without --sweep first.")
            return
        with open(cache) as f:
            viable = json.load(f)
        print(f"Loaded {len(viable)} viable symbols from cache")

    if not viable:
        print("No viable crypto symbols found.")
        return

    if RUN_SWEEP:
        await phase2_sweep(viable)


if __name__ == "__main__":
    asyncio.run(main())
