"""
Binary filter sweep: ATR volatility gate + session window.

Tests the five active binary strategies from config.yaml with two filters:
  1. ATR gate    — skip entry when market is too quiet (below N x rolling mean ATR)
  2. Session     — restrict to high-quality trading hours (EAT = UTC+3)

Signals are exact replicas of current config.yaml params so results are
directly comparable to validated backtest numbers.

Walk-forward: 4 windows of equal size over all available candle history.

Usage:
  python sweep_binary_filters.py            # use cached OHLC data
  python sweep_binary_filters.py --fresh    # re-fetch from Deriv API
"""

import argparse, asyncio, json, sys
from pathlib import Path
import websockets

WS_URL    = "wss://ws.derivws.com/websockets/v3?app_id=1089"
CACHE_DIR = Path("data/scalp")

GRAN       = 300    # 5-min bars
WINDOWS    = 4
MIN_TRADES = 8      # minimum per window to count
ATR_PERIOD = 14     # TR smoothing (bars)
ATR_WINDOW = 100    # rolling mean window for relative ATR threshold

# ── Active binary strategies (mirrors config.yaml exactly) ───────────────────
STRATEGIES = [
    {"symbol": "frxXAUUSD", "label": "XAU primary",
     "payout": 0.80, "hold_bars": 1,
     "sig": "rsi_ema", "rsi_period": 10, "ema_period": 20, "slope_bars": 3, "rsi_entry": 45},
    {"symbol": "frxXAUUSD", "label": "XAU backup",
     "payout": 0.80, "hold_bars": 1,
     "sig": "rsi_ema", "rsi_period": 7,  "ema_period": 20, "slope_bars": 3, "rsi_entry": 40},
    {"symbol": "frxUSDJPY", "label": "JPY primary",
     "payout": 0.90, "hold_bars": 3,
     "sig": "rsi_ema", "rsi_period": 10, "ema_period": 20, "slope_bars": 3, "rsi_entry": 45},
    {"symbol": "frxUSDJPY", "label": "JPY hi-freq",
     "payout": 0.90, "hold_bars": 3,
     "sig": "rsi_ema", "rsi_period": 10, "ema_period": 50, "slope_bars": 5, "rsi_entry": 40},
    {"symbol": "frxAUDUSD", "label": "AUD/USD",
     "payout": 0.85, "hold_bars": 3,
     "sig": "rsi_reversal", "rsi_period": 10, "rsi_os": 25},
]

# ATR threshold: 0.0 = off; 0.75 = only trade when ATR >= 75% of rolling mean
ATR_THRESHOLDS = [0.0, 0.75, 1.0, 1.25]

# Session windows in EAT (UTC+3) — None means all hours (no filter)
# London=07-12UTC, NY-overlap=12-17UTC
SESSION_WINDOWS = {
    "all":        None,
    "London":     set(range(10, 16)),   # 10-15 EAT = 07-12 UTC
    "NY-overlap": set(range(15, 21)),   # 15-20 EAT = 12-17 UTC
    "London+NY":  set(range(10, 21)),   # 10-20 EAT = 07-17 UTC
}


# ── Data fetch ────────────────────────────────────────────────────────────────

async def fetch_ohlc(symbol: str, fresh: bool) -> list[dict]:
    cache = CACHE_DIR / f"{symbol}_{GRAN}_ohlc.json"
    if cache.exists() and not fresh:
        with open(cache) as f:
            return json.load(f)

    async with websockets.connect(WS_URL, open_timeout=15) as ws:
        await ws.send(json.dumps({
            "ticks_history": symbol,
            "style":         "candles",
            "granularity":   GRAN,
            "count":         5000,
            "end":           "latest",
            "req_id":        1,
        }))
        raw = await asyncio.wait_for(ws.recv(), timeout=30)
        msg = json.loads(raw)

    if msg.get("error"):
        raise RuntimeError(msg["error"]["message"])

    candles = [
        {"epoch": c["open_time"],
         "high":  float(c["high"]),
         "low":   float(c["low"]),
         "close": float(c["close"])}
        for c in msg["candles"]
    ]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache, "w") as f:
        json.dump(candles, f)
    return candles


# ── Indicators ────────────────────────────────────────────────────────────────

def compute_atr(candles: list[dict]) -> list:
    n = len(candles)
    trs = [None] * n
    for i in range(1, n):
        h, l, pc = candles[i]["high"], candles[i]["low"], candles[i-1]["close"]
        trs[i] = max(h - l, abs(h - pc), abs(l - pc))

    atr = [None] * n
    for i in range(ATR_PERIOD, n):
        vals = [t for t in trs[i - ATR_PERIOD + 1: i + 1] if t is not None]
        if len(vals) == ATR_PERIOD:
            atr[i] = sum(vals) / ATR_PERIOD
    return atr


def compute_atr_mean(atr: list) -> list:
    n    = len(atr)
    mean = [None] * n
    for i in range(ATR_WINDOW, n):
        vals = [v for v in atr[i - ATR_WINDOW: i] if v is not None]
        if vals:
            mean[i] = sum(vals) / len(vals)
    return mean


def eat_hours(candles: list[dict]) -> list[int]:
    return [(c["epoch"] // 3600 + 3) % 24 for c in candles]


def rsi_series(closes: list, period: int) -> list:
    out = [None] * len(closes)
    if len(closes) < period + 1:
        return out
    ch = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    g  = sum(c for c in ch[:period] if c > 0) / period
    l  = sum(-c for c in ch[:period] if c < 0) / period
    for i in range(period, len(closes)):
        d = ch[i - 1]
        g = (g * (period - 1) + max(d, 0)) / period
        l = (l * (period - 1) + max(-d, 0)) / period
        out[i] = 100.0 if l == 0 else round(100 - 100 / (1 + g / l), 2)
    return out


def ema_series(closes: list, period: int) -> list:
    out = [None] * len(closes)
    if len(closes) < period:
        return out
    k   = 2 / (period + 1)
    val = sum(closes[:period]) / period
    out[period - 1] = val
    for i in range(period, len(closes)):
        val    = closes[i] * k + val * (1 - k)
        out[i] = val
    return out


# ── Signal generators ─────────────────────────────────────────────────────────

def gen_rsi_ema(closes, rsi_period, rsi_entry, ema_period, slope_bars):
    rsi   = rsi_series(closes, rsi_period)
    ema   = ema_series(closes, ema_period)
    ob    = 100 - rsi_entry
    sigs  = []
    start = max(rsi_period, ema_period) + slope_bars + 1
    for i in range(start, len(closes)):
        if rsi[i] is None or ema[i] is None or ema[i - slope_bars] is None:
            continue
        if rsi[i - 1] is None:
            continue
        up = ema[i] > ema[i - slope_bars]
        if up and rsi[i - 1] >= rsi_entry > rsi[i]:
            sigs.append((i, +1))
        elif not up and rsi[i - 1] <= ob < rsi[i]:
            sigs.append((i, -1))
    return sigs


def gen_rsi_reversal(closes, rsi_period, rsi_os):
    rsi = rsi_series(closes, rsi_period)
    ob  = 100 - rsi_os
    out = []
    for i in range(rsi_period + 1, len(closes)):
        if rsi[i] is None or rsi[i - 1] is None:
            continue
        if rsi[i - 1] < rsi_os <= rsi[i]:
            out.append((i, +1))
        elif rsi[i - 1] > ob >= rsi[i]:
            out.append((i, -1))
    return out


# ── Filtering ─────────────────────────────────────────────────────────────────

def apply_filters(signals, atr, atr_mean, hours, atr_thresh, session):
    out = []
    for i, d in signals:
        if atr_thresh > 0:
            a, m = atr[i], atr_mean[i]
            if a is None or m is None or a < atr_thresh * m:
                continue
        if session is not None and hours[i] not in session:
            continue
        out.append((i, d))
    return out


# ── Simulation ────────────────────────────────────────────────────────────────

def sim_binary(closes, signals, hold_bars, payout):
    wins = total = 0
    for i, d in signals:
        if i + hold_bars >= len(closes):
            continue
        total += 1
        if d == +1 and closes[i + hold_bars] > closes[i]:
            wins += 1
        elif d == -1 and closes[i + hold_bars] < closes[i]:
            wins += 1
    if total < MIN_TRADES:
        return None
    wr = wins / total
    return wins, total, wr


# ── Walk-forward ──────────────────────────────────────────────────────────────

def walk_forward(strat, candles, atr_all, atr_mean_all, hours_all, atr_thresh, session):
    closes = [c["close"] for c in candles]
    n, ws  = len(candles), len(candles) // WINDOWS
    results = []
    for w in range(WINDOWS):
        s = w * ws
        e = s + ws if w < WINDOWS - 1 else n
        seg_c  = closes[s:e]
        seg_a  = atr_all[s:e]
        seg_m  = atr_mean_all[s:e]
        seg_h  = hours_all[s:e]

        if strat["sig"] == "rsi_ema":
            sigs = gen_rsi_ema(
                seg_c, strat["rsi_period"], strat["rsi_entry"],
                strat["ema_period"], strat["slope_bars"],
            )
        else:
            sigs = gen_rsi_reversal(seg_c, strat["rsi_period"], strat["rsi_os"])

        sigs = apply_filters(sigs, seg_a, seg_m, seg_h, atr_thresh, session)
        results.append(sim_binary(seg_c, sigs, strat["hold_bars"], strat["payout"]))
    return results


def summarize(results, payout):
    valid = [r for r in results if r is not None]
    if not valid:
        return None
    be = 1.0 / (1 + payout)
    passes     = sum(1 for r in valid if r[2] >= be)
    total_wins  = sum(r[0] for r in valid)
    total_trades = sum(r[1] for r in valid)
    wr = total_wins / total_trades if total_trades else 0.0
    ev = (wr - be) * payout
    return wr, total_trades, ev, passes, len(valid)


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="Re-fetch candle data from Deriv API")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Fetch candles once per unique symbol
    symbol_data: dict[str, tuple] = {}
    for s in STRATEGIES:
        sym = s["symbol"]
        if sym not in symbol_data:
            print(f"Loading {sym}...", end=" ", flush=True)
            try:
                candles = await fetch_ohlc(sym, args.fresh)
                atr     = compute_atr(candles)
                am      = compute_atr_mean(atr)
                hours   = eat_hours(candles)
                symbol_data[sym] = (candles, atr, am, hours)
                print(f"{len(candles)} candles")
            except Exception as e:
                print(f"FAILED: {e}")
                symbol_data[sym] = None

    SEP  = "=" * 108
    THIN = "-" * 108

    print(f"\nBinary Filter Sweep — ATR gate + session window")
    print(f"Walk-forward: {WINDOWS} windows | Min trades/window: {MIN_TRADES} | ATR period: {ATR_PERIOD}")
    print(f"ATR threshold = min ATR as fraction of {ATR_WINDOW}-bar rolling mean ATR")
    print(f"Session EAT (UTC+3): London=10-15, NY-overlap=15-20, London+NY=10-20")
    print(SEP)

    for strat in STRATEGIES:
        sym  = strat["symbol"]
        data = symbol_data.get(sym)
        if data is None:
            print(f"\nSKIP {sym} — data unavailable")
            continue

        candles, atr_all, atr_mean_all, hours_all = data
        payout = strat["payout"]
        be     = 1.0 / (1 + payout)

        if strat["sig"] == "rsi_ema":
            sig_desc = (f"RSI({strat['rsi_period']})+EMA({strat['ema_period']}) "
                        f"slope={strat['slope_bars']} entry={strat['rsi_entry']}")
        else:
            sig_desc = f"RSI({strat['rsi_period']}) OS={strat['rsi_os']}"

        hold_min = strat["hold_bars"] * GRAN // 60
        print(f"\n{sym} — {strat['label']}  |  payout={payout*100:.0f}%  BE={be*100:.1f}%  "
              f"hold={hold_min}min  {sig_desc}")
        print(THIN)
        print(f"  {'Session':12s}  {'ATR':5s}  {'WR':6s}  {'Chg':6s}  {'EV%':7s}  "
              f"{'Trades':7s}  {'Pass':4s}  note")
        print(THIN)

        baseline_wr = None

        for sess_name, session in SESSION_WINDOWS.items():
            for atr_thresh in ATR_THRESHOLDS:
                results = walk_forward(
                    strat, candles, atr_all, atr_mean_all, hours_all,
                    atr_thresh, session,
                )
                summ = summarize(results, payout)
                if summ is None:
                    print(f"  {sess_name:12s}  {atr_thresh:.2f}   -- (too few trades)")
                    continue

                wr, trades, ev, passes, valid = summ
                wr_pct = wr * 100
                ev_pct = ev * 100

                is_baseline = sess_name == "all" and atr_thresh == 0.0
                if is_baseline:
                    baseline_wr = wr_pct
                    chg = "  base"
                elif baseline_wr is not None:
                    diff = wr_pct - baseline_wr
                    chg  = f"{diff:+.1f}pp"
                else:
                    chg = "     -"

                all_pass = passes == valid
                good     = all_pass and ev_pct > 1.0
                flag     = " <<" if good else ""
                print(f"  {sess_name:12s}  {atr_thresh:.2f}   {wr_pct:5.1f}%  {chg:6s}  "
                      f"{ev_pct:+6.2f}%  {trades:7d}  {passes}/{valid}{flag}")

    print(f"\n{SEP}")
    print(f"<< = all {WINDOWS} windows pass BE AND EV > 1%")
    print(f"Chg = WR change vs no-filter baseline for this strategy")


if __name__ == "__main__":
    asyncio.run(main())
