"""
Scalp research v2: actual payout rates + valid hold durations per symbol.

From test_durations.py:
  frxXAUUSD  min=5min   payout=80%  → hold 1/2/3 bars valid (5/10/15min)
  frxXAGUSD  min=5min   payout=63%  → hold 1/2/3 bars valid (needs WR>61.3% to profit)
  frxUSDJPY  min=15min  payout=90%  → hold 3 bars only (15min)
  frxGBPUSD  min=15min  payout=88%  → hold 3 bars only (15min)
  frxEURUSD  min=15min  payout=85%  → hold 3 bars only (15min)
  frxAUDUSD  min=15min  payout=85%  → hold 3 bars only (estimated, not tested)

Granularity: 5-min bars only (1-min strategies don't apply — min contract is 5+ min)
Signals: RSI reversal, EMA cross, RSI+EMA filter, BB bounce
Cache: data/scalp/<symbol>_300.json (reuses prior fetch)
"""

import asyncio
import json
from pathlib import Path

import websockets

LEGACY_WS = "wss://ws.derivws.com/websockets/v3?app_id=1089"
CACHE_DIR = Path("data/scalp")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

GRAN       = 300   # 5-min bars
FOLDS      = 3
MIN_TRADES = 10    # minimum per fold to count result

# Actual payouts from test_durations.py + minimum hold bars (bars × 5min = contract duration)
SYMBOL_CONFIG = {
    "frxXAUUSD": {"label": "Gold    ", "payout": 0.80, "min_hold": 1, "max_hold": 3},
    "frxXAGUSD": {"label": "Silver  ", "payout": 0.63, "min_hold": 1, "max_hold": 3},
    "frxUSDJPY": {"label": "USD/JPY ", "payout": 0.90, "min_hold": 3, "max_hold": 3},
    "frxGBPUSD": {"label": "GBP/USD ", "payout": 0.88, "min_hold": 3, "max_hold": 3},
    "frxEURUSD": {"label": "EUR/USD ", "payout": 0.85, "min_hold": 3, "max_hold": 3},
    "frxAUDUSD": {"label": "AUD/USD ", "payout": 0.85, "min_hold": 3, "max_hold": 3},
}


# ── Data ───────────────────────────────────────────────────────────────────

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
            "count":         5000,
            "end":           "latest",
            "req_id":        1,
        }))
        raw  = await asyncio.wait_for(ws.recv(), timeout=30)
        msg  = json.loads(raw)

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


# ── Signals ────────────────────────────────────────────────────────────────

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


# ── Simulation + walk-forward ──────────────────────────────────────────────

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


# ── Per-symbol sweep ────────────────────────────────────────────────────────

def sweep(closes, payout, min_hold, max_hold):
    results     = []
    trading_days = len(closes) * GRAN / (22 * 3600)   # ~22 active hours/day

    def record(label, hold_b, folds):
        status = classify(folds)
        if status not in ("STRONG", "WEAK"):
            return
        valid = [r for r in folds if r]
        results.append({
            "signal":       label,
            "hold_bars":    hold_b,
            "hold_min":     hold_b * GRAN // 60,
            "folds":        folds,
            "mean_ev":      sum(r[2] for r in valid) / len(valid),
            "total_trades": sum(r[1] for r in valid),
            "per_day":      sum(r[1] for r in valid) / trading_days,
            "status":       status,
            "payout":       payout,
        })

    hold_range = range(min_hold, max_hold + 1)

    for rp in [7, 10, 14]:
        for os in [20, 25, 30]:
            for hb in hold_range:
                folds = walk_forward(closes,
                    lambda c, r=rp, o=os: sig_rsi_reversal(c, r, o), hb, payout)
                record(f"RSI({rp:2d}) OS={os}", hb, folds)

    for fast, slow in [(5, 20), (5, 30), (8, 21), (10, 30), (10, 50)]:
        for hb in hold_range:
            folds = walk_forward(closes,
                lambda c, f=fast, s=slow: sig_ema_cross(c, f, s), hb, payout)
            record(f"EMA({fast:2d}/{slow:2d})", hb, folds)

    for rp in [7, 10, 14]:
        for re in [40, 45]:
            for ep in [20, 50]:
                for sb in [3, 5]:
                    for hb in hold_range:
                        folds = walk_forward(closes,
                            lambda c, r=rp, e=re, p=ep, s=sb:
                                sig_rsi_ema(c, r, e, p, s), hb, payout)
                        record(f"RSI+EMA({rp}/{re}/EMA{ep}/s{sb})", hb, folds)

    for bp in [10, 20]:
        for bs in [1.5, 2.0]:
            for hb in hold_range:
                folds = walk_forward(closes,
                    lambda c, b=bp, s=bs: sig_bb_bounce(c, b, s), hb, payout)
                record(f"BB({bp:2d}/{bs})", hb, folds)

    return results


# ── Main ───────────────────────────────────────────────────────────────────

async def main():
    be = {sym: round(100 / (1 + cfg["payout"]), 1)
          for sym, cfg in SYMBOL_CONFIG.items()}

    print("=== Scalp Research v2 — Actual Payouts + Valid Durations ===")
    print(f"Granularity: 5-min bars | Folds: {FOLDS} | Min trades/fold: {MIN_TRADES}")
    print()
    for sym, cfg in SYMBOL_CONFIG.items():
        hold_mins = [h * GRAN // 60
                     for h in range(cfg["min_hold"], cfg["max_hold"] + 1)]
        print(f"  {sym:12s}  payout={cfg['payout']*100:.0f}%  "
              f"BE WR={be[sym]}%  "
              f"valid durations={hold_mins}min")
    print()

    all_results = []

    for sym, cfg in SYMBOL_CONFIG.items():
        print(f"[{sym}] {cfg['label'].strip()} — loading data...")
        try:
            closes = await fetch_closes(sym)
            print(f"  {len(closes)} candles | sweeping at "
                  f"payout={cfg['payout']*100:.0f}% BE={be[sym]}%...")
        except Exception as e:
            print(f"  SKIP — {e}")
            continue

        results = sweep(closes, cfg["payout"], cfg["min_hold"], cfg["max_hold"])
        for r in results:
            r["symbol"] = sym
            r["label"]  = cfg["label"].strip()
        all_results.extend(results)

        strong = sum(1 for r in results if r["status"] == "STRONG")
        weak   = sum(1 for r in results if r["status"] == "WEAK")
        print(f"  → {strong} STRONG  {weak} WEAK")

    # ── Report ────────────────────────────────────────────────────────────
    print("\n" + "=" * 95)
    print("RESULTS — STRONG first, then WEAK, sorted by mean EV within each group")
    print("=" * 95)

    strong_all = sorted(
        [r for r in all_results if r["status"] == "STRONG"],
        key=lambda x: -x["mean_ev"]
    )
    weak_all = sorted(
        [r for r in all_results if r["status"] == "WEAK"],
        key=lambda x: -x["mean_ev"]
    )

    for group, title in [(strong_all, "STRONG — 3/3 folds EV>0.05"),
                         (weak_all[:20],  "WEAK — 2/3 folds (top 20)")]:
        if not group:
            continue
        print(f"\n── {title} ──")
        prev = None
        for r in group:
            if r["symbol"] != prev:
                print(f"\n  {r['label']} ({r['symbol']})  "
                      f"payout={r['payout']*100:.0f}%  "
                      f"BE WR={round(100/(1+r['payout']),1)}%")
                prev = r["symbol"]
            valid    = [f for f in r["folds"] if f]
            wrs      = " / ".join(f"{f[0]*100:.1f}%" for f in valid)
            print(f"    {r['signal']:35s} hold={r['hold_min']:2d}min "
                  f"| [{wrs}] "
                  f"| MeanEV {r['mean_ev']:+.4f} "
                  f"| ~{r['per_day']:.1f}/day "
                  f"| {r['status']}")

    print(f"\n✓ {len(strong_all)} STRONG  {len(weak_all)} WEAK total")

    if strong_all:
        top = strong_all[0]
        print(f"\nTop pick: {top['label']} — {top['signal']} "
              f"hold={top['hold_min']}min "
              f"| MeanEV {top['mean_ev']:+.4f} "
              f"| ~{top['per_day']:.1f} trades/day "
              f"| payout {top['payout']*100:.0f}%")


if __name__ == "__main__":
    asyncio.run(main())
