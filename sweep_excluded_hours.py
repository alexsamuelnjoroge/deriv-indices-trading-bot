"""
Excluded-hours sweep: were the filtered-out hours actually losing?

For each forex binary strategy with a session filter in config.yaml, tests:
  A) ALL hours   (baseline — no session filter)
  B) KEPT hours  (what the live bot currently trades)
  C) EXCLUDED hours (the sessions the bot currently skips)

Directly answers: was excluding those hours correct, or were they profitable too?

Strategies tested (current config params):
  1. frxUSDJPY hi-freq  rsi_ema(ema=50, rsi=10, entry=40)  NY-overlap kept (15-20 EAT)
  2. frxAUDUSD          rsi_reversal(rsi=10, os=25)         London kept (10-15 EAT)
  3. frxXAUUSD BB-touch bb_touch(bb=10, std=2.0)            London kept (10-15 EAT)

ATR gate tested at both off (0.0) and 1.25x (current live config).
Walk-forward: 4 equal-size windows over all available 5-min candle history.

Usage:
  python sweep_excluded_hours.py            # use cached OHLC data
  python sweep_excluded_hours.py --fresh    # re-fetch from Deriv API
"""

import argparse, asyncio, json
from pathlib import Path
import websockets

WS_URL    = "wss://ws.derivws.com/websockets/v3?app_id=1089"
CACHE_DIR = Path("data/scalp")

GRAN       = 300    # 5-min bars
WINDOWS    = 4
MIN_TRADES = 8      # minimum trades per window to count that window as valid
ATR_PERIOD = 14
ATR_WINDOW = 100

# ── Strategies with session filters ───────────────────────────────────────────
STRATEGIES = [
    {
        "label":      "USD/JPY hi-freq",
        "symbol":     "frxUSDJPY",
        "sig":        "rsi_ema",
        "rsi_period": 10,
        "ema_period": 50,
        "slope_bars": 5,
        "rsi_entry":  40,
        "payout":     0.90,
        "hold_bars":  3,
        "atr_mult":   1.25,
        "kept_hours": set(range(15, 21)),   # 15-20 EAT = NY-overlap (12-17 UTC)
    },
    {
        "label":      "AUD/USD",
        "symbol":     "frxAUDUSD",
        "sig":        "rsi_reversal",
        "rsi_period": 10,
        "rsi_os":     25,
        "payout":     0.85,
        "hold_bars":  3,
        "atr_mult":   1.25,
        "kept_hours": set(range(10, 16)),   # 10-15 EAT = London (07-12 UTC)
    },
    {
        "label":      "XAU BB-touch",
        "symbol":     "frxXAUUSD",
        "sig":        "bb_touch",
        "bb_period":  10,
        "bb_std":     2.0,
        "payout":     0.80,
        "hold_bars":  1,
        "atr_mult":   1.25,
        "kept_hours": set(range(10, 16)),   # 10-15 EAT = London (07-12 UTC)
    },
]


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
        {"epoch": c["open_time"], "high": float(c["high"]),
         "low":   float(c["low"]), "close": float(c["close"])}
        for c in msg["candles"]
    ]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache, "w") as f:
        json.dump(candles, f)
    return candles


# ── Indicators ────────────────────────────────────────────────────────────────

def compute_atr(candles: list[dict]) -> list:
    n, trs = len(candles), [None] * len(candles)
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
    n = len(atr)
    mean = [None] * n
    for i in range(ATR_WINDOW, n):
        vals = [v for v in atr[i - ATR_WINDOW: i] if v is not None]
        if vals:
            mean[i] = sum(vals) / len(vals)
    return mean


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


def _bb_bands(window: list, n_std: float):
    n    = len(window)
    mean = sum(window) / n
    var  = sum((x - mean) ** 2 for x in window) / n
    std  = var ** 0.5
    return mean + n_std * std, mean - n_std * std


# ── Signal generators ─────────────────────────────────────────────────────────

def gen_rsi_ema(closes, rsi_period, rsi_entry, ema_period, slope_bars):
    """EMA slope + RSI entry-crossing signal (mirrors GoldTrendStrategy)."""
    rsi  = rsi_series(closes, rsi_period)
    ema  = ema_series(closes, ema_period)
    ob   = 100 - rsi_entry
    sigs = []
    start = max(rsi_period, ema_period) + slope_bars + 1
    for i in range(start, len(closes)):
        if rsi[i] is None or rsi[i - 1] is None:
            continue
        if ema[i] is None or ema[i - slope_bars] is None:
            continue
        up = ema[i] > ema[i - slope_bars]
        if up and rsi[i - 1] >= rsi_entry > rsi[i]:
            sigs.append((i, +1))   # BUY_RISE
        elif not up and rsi[i - 1] <= ob < rsi[i]:
            sigs.append((i, -1))   # BUY_FALL
    return sigs


def gen_rsi_reversal(closes, rsi_period, rsi_os):
    """RSI exits OS/OB zone (mirrors RSIBinaryStrategy)."""
    rsi = rsi_series(closes, rsi_period)
    ob  = 100 - rsi_os
    out = []
    for i in range(rsi_period + 1, len(closes)):
        if rsi[i] is None or rsi[i - 1] is None:
            continue
        if rsi[i - 1] < rsi_os <= rsi[i]:
            out.append((i, +1))    # BUY_RISE
        elif rsi[i - 1] > ob >= rsi[i]:
            out.append((i, -1))    # BUY_FALL
    return out


def gen_bb_touch(closes, bb_period, bb_std):
    """Close touches or crosses BB band (mirrors BBBinaryStrategy touch mode)."""
    out = []
    for i in range(bb_period - 1, len(closes)):
        window = closes[i - bb_period + 1: i + 1]
        upper, lower = _bb_bands(window, bb_std)
        c = closes[i]
        if c <= lower:
            out.append((i, +1))    # BUY_RISE (mean reversion expected)
        elif c >= upper:
            out.append((i, -1))    # BUY_FALL
    return out


def gen_signals(strat: dict, closes: list) -> list:
    sig = strat["sig"]
    if sig == "rsi_ema":
        return gen_rsi_ema(
            closes, strat["rsi_period"], strat["rsi_entry"],
            strat["ema_period"], strat["slope_bars"],
        )
    elif sig == "rsi_reversal":
        return gen_rsi_reversal(closes, strat["rsi_period"], strat["rsi_os"])
    elif sig == "bb_touch":
        return gen_bb_touch(closes, strat["bb_period"], strat["bb_std"])
    raise ValueError(f"Unknown sig: {sig}")


# ── Filtering + simulation ────────────────────────────────────────────────────

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


def sim_binary(closes, signals, hold_bars):
    wins = total = 0
    for i, d in signals:
        if i + hold_bars >= len(closes):
            continue
        total += 1
        if d == +1 and closes[i + hold_bars] > closes[i]:
            wins += 1
        elif d == -1 and closes[i + hold_bars] < closes[i]:
            wins += 1
    return wins, total


# ── Walk-forward ──────────────────────────────────────────────────────────────

def walk_forward(strat, candles, atr_all, atr_mean_all, session, atr_thresh):
    closes    = [c["close"] for c in candles]
    hours_all = [(c["epoch"] // 3600 + 3) % 24 for c in candles]
    n, ws     = len(candles), len(candles) // WINDOWS
    be        = 1.0 / (1 + strat["payout"])

    total_wins = total_trades = total_passes = 0
    for w in range(WINDOWS):
        s = w * ws
        e = s + ws if w < WINDOWS - 1 else n

        sigs = gen_signals(strat, closes[s:e])
        sigs = apply_filters(sigs, atr_all[s:e], atr_mean_all[s:e],
                              hours_all[s:e], atr_thresh, session)

        wins, trades = sim_binary(closes[s:e], sigs, strat["hold_bars"])
        if trades >= MIN_TRADES and (wins / trades if trades else 0) >= be:
            total_passes += 1
        total_wins   += wins
        total_trades += trades

    return total_wins, total_trades, total_passes


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="Re-fetch candle data from Deriv API")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    symbol_data: dict[str, tuple] = {}
    for strat in STRATEGIES:
        sym = strat["symbol"]
        if sym in symbol_data:
            continue
        print(f"Loading {sym}...", end=" ", flush=True)
        try:
            candles = await fetch_ohlc(sym, args.fresh)
            atr     = compute_atr(candles)
            am      = compute_atr_mean(atr)
            symbol_data[sym] = (candles, atr, am)
            print(f"{len(candles)} candles")
        except Exception as e:
            print(f"FAILED: {e}")
            symbol_data[sym] = None

    SEP = "=" * 85

    for strat in STRATEGIES:
        sym  = strat["symbol"]
        data = symbol_data.get(sym)
        if data is None:
            print(f"\nSkipping {strat['label']}: no data")
            continue

        candles, atr_all, atr_mean_all = data
        payout    = strat["payout"]
        be_pct    = 100.0 / (1 + payout)
        kept      = strat["kept_hours"]
        excluded  = set(range(24)) - kept

        kept_sorted     = sorted(kept)
        excluded_sorted = sorted(excluded)

        print()
        print(SEP)
        print(f"  {strat['label']}  |  {sym}  |  payout={payout*100:.0f}%  BE={be_pct:.1f}%")
        print(f"  Kept hours (EAT):     {kept_sorted[0]}-{kept_sorted[-1]}")
        print(f"  Excluded hours (EAT): 0-{kept_sorted[0]-1} and {kept_sorted[-1]+1}-23")
        print(f"  Hold: {strat['hold_bars']}x5min bar(s)  |  {len(candles)} candles  |  {WINDOWS}-fold walk-forward")
        print(SEP)

        sessions = [
            ("ALL hours",      None),
            ("KEPT hours",     kept),
            ("EXCLUDED hours", excluded),
        ]

        print(f"  {'Session':<18}  {'ATR':>5}  {'WR%':>6}  {'BE%':>5}  {'EV%':>8}  {'trades':>6}  {'passes':>6}  verdict")
        print(f"  {'-'*18}  {'-'*5}  {'-'*6}  {'-'*5}  {'-'*8}  {'-'*6}  {'-'*6}  -------")

        for sess_label, session in sessions:
            for atr_label, atr_thresh in [("off", 0.0), ("1.25x", strat["atr_mult"])]:
                wins, trades, passes = walk_forward(
                    strat, candles, atr_all, atr_mean_all, session, atr_thresh
                )
                if trades == 0:
                    print(f"  {sess_label:<18}  {atr_label:>5}   {'n/a':>5}   {be_pct:>5.1f}%  {'n/a':>8}  {trades:>6}  {passes}/{WINDOWS}  no trades")
                    continue

                wr  = wins / trades * 100
                ev  = (wr - be_pct) / 100 * payout * 100

                if wr < be_pct:
                    verdict = "LOSING (<BE)"
                elif sess_label == "EXCLUDED hours":
                    verdict = "+EV (profitable!)"
                else:
                    verdict = ""

                print(f"  {sess_label:<18}  {atr_label:>5}  {wr:>6.1f}%  {be_pct:>5.1f}%  {ev:>+8.3f}%  {trades:>6}  {passes}/{WINDOWS}  {verdict}")

    print()
    print(SEP)
    print("  LOSING  = WR < breakeven -- session filter was correct to exclude these hours")
    print("  +EV     = excluded hours were profitable -- potentially lost opportunity")
    print("  passes  = windows where WR >= BE (out of 4)")


if __name__ == "__main__":
    asyncio.run(main())
