"""
Multiplier strategy sweep — Phase 1 (Forex) + Phase 2 (Volatility indices)

Mechanics:
  - Entry: directional signal fires → open MULTUP or MULTDOWN
  - Stop-out: price moves 1/multiplier against you → stake fully lost
  - Take-profit: price moves tp_pct in your favour → close at profit
  - Commission: 0.03% of notional charged on entry

Phase 1 — Forex pairs with session filters:
  frxUSDJPY, frxAUDUSD, frxXAUUSD, frxEURUSD, frxGBPUSD
  Signals: RSI+EMA, RSI-Reversal, BB-Touch, MACD-Flip

Phase 2 — Volatility indices (24/7, no session filter):
  R_10, R_25, R_50, R_75, R_100
  Signals: RSI-Reversal, BB-Touch (mean-reversion)

Sweep axes:
  multiplier  : 100, 200, 300, 500
  tp_pct      : 0.3%, 0.5%, 1%, 2%, 3%

4-fold walk-forward on up to 5000 candles (5-min bars) per symbol.

Usage:
  python sweep_multiplier.py                    # all phases
  python sweep_multiplier.py --phase forex
  python sweep_multiplier.py --phase vol
  python sweep_multiplier.py --symbol frxUSDJPY
  python sweep_multiplier.py --fresh
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
import websockets

WS_URL    = "wss://ws.derivws.com/websockets/v3?app_id=1089"
CACHE_DIR = Path("data/scalp")
GRAN      = 300   # 5-min bars

WINDOWS    = 4
MIN_TRADES = 8    # minimum trades per window to score that window

COMMISSION_PCT = 0.0003   # 0.03% of notional

ATR_PERIOD = 14
ATR_WINDOW = 100

# ── Strategy definitions ──────────────────────────────────────────────────────

FOREX_STRATS = [
    {
        "label":      "USDJPY RSI+EMA",
        "symbol":     "frxUSDJPY",
        "sig":        "rsi_ema",
        "rsi_period": 10,
        "ema_period": 50,
        "slope_bars": 5,
        "rsi_entry":  40,
        "session":    set(range(15, 21)),
    },
    {
        "label":      "USDJPY MACD",
        "symbol":     "frxUSDJPY",
        "sig":        "macd",
        "fast":       12,
        "slow":       26,
        "sig_period": 9,
        "session":    set(range(15, 21)),
    },
    {
        "label":      "AUDUSD RSI-Rev",
        "symbol":     "frxAUDUSD",
        "sig":        "rsi_reversal",
        "rsi_period": 10,
        "rsi_os":     25,
        "session":    set(range(10, 16)),
    },
    {
        "label":      "XAUUSD BB-Touch",
        "symbol":     "frxXAUUSD",
        "sig":        "bb_touch",
        "bb_period":  10,
        "bb_std":     2.0,
        "session":    set(range(10, 16)),
    },
    {
        "label":      "EURUSD RSI-Rev",
        "symbol":     "frxEURUSD",
        "sig":        "rsi_reversal",
        "rsi_period": 10,
        "rsi_os":     25,
        "session":    set(range(10, 21)),
    },
    {
        "label":      "EURUSD MACD",
        "symbol":     "frxEURUSD",
        "sig":        "macd",
        "fast":       12,
        "slow":       26,
        "sig_period": 9,
        "session":    set(range(10, 21)),
    },
    {
        "label":      "GBPUSD RSI-Rev",
        "symbol":     "frxGBPUSD",
        "sig":        "rsi_reversal",
        "rsi_period": 10,
        "rsi_os":     25,
        "session":    set(range(10, 21)),
    },
    {
        "label":      "GBPUSD BB-Touch",
        "symbol":     "frxGBPUSD",
        "sig":        "bb_touch",
        "bb_period":  20,
        "bb_std":     2.0,
        "session":    set(range(10, 21)),
    },
]

VOL_STRATS = [
    {
        "label":      f"{sym} RSI-Rev",
        "symbol":     sym,
        "sig":        "rsi_reversal",
        "rsi_period": 10,
        "rsi_os":     20,
        "session":    None,
    }
    for sym in ["R_10", "R_25", "R_50", "R_75", "R_100"]
] + [
    {
        "label":      f"{sym} BB-Touch",
        "symbol":     sym,
        "sig":        "bb_touch",
        "bb_period":  20,
        "bb_std":     2.0,
        "session":    None,
    }
    for sym in ["R_10", "R_25", "R_50", "R_75", "R_100"]
]

MULTIPLIERS = [100, 200, 300, 500]
TP_PCTS     = [0.003, 0.005, 0.010, 0.020, 0.030]


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
        {
            "epoch": c.get("open_time", c.get("epoch", 0)),
            "open":  float(c["open"]),
            "high":  float(c["high"]),
            "low":   float(c["low"]),
            "close": float(c["close"]),
        }
        for c in msg["candles"]
    ]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache, "w") as f:
        json.dump(candles, f)
    return candles


# ── Indicators ────────────────────────────────────────────────────────────────

def rsi_series(closes, period):
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


def ema_series(closes, period):
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


def macd_series(closes, fast, slow, sig_period):
    ema_f  = ema_series(closes, fast)
    ema_s  = ema_series(closes, slow)
    macd   = [None] * len(closes)
    signal = [None] * len(closes)
    for i in range(slow - 1, len(closes)):
        if ema_f[i] is not None and ema_s[i] is not None:
            macd[i] = ema_f[i] - ema_s[i]
    macd_vals   = [v for v in macd if v is not None]
    sig_ema     = [None] * len(macd)
    start       = next(i for i, v in enumerate(macd) if v is not None)
    if len(macd_vals) >= sig_period:
        k   = 2 / (sig_period + 1)
        val = sum(macd_vals[:sig_period]) / sig_period
        si  = start + sig_period - 1
        sig_ema[si] = val
        for i in range(si + 1, len(macd)):
            if macd[i] is not None:
                val         = macd[i] * k + val * (1 - k)
                sig_ema[i]  = val
    hist = [None] * len(closes)
    for i in range(len(closes)):
        if macd[i] is not None and sig_ema[i] is not None:
            hist[i] = macd[i] - sig_ema[i]
    return hist


def bb_bands(window, n_std):
    n    = len(window)
    mean = sum(window) / n
    var  = sum((x - mean) ** 2 for x in window) / n
    std  = var ** 0.5
    return mean + n_std * std, mean - n_std * std


# ── Signal generators ─────────────────────────────────────────────────────────

def gen_rsi_ema(closes, rsi_period, rsi_entry, ema_period, slope_bars):
    rsi  = rsi_series(closes, rsi_period)
    ema  = ema_series(closes, ema_period)
    ob   = 100 - rsi_entry
    sigs = []
    start = max(rsi_period, ema_period) + slope_bars + 1
    for i in range(start, len(closes)):
        if None in (rsi[i], rsi[i - 1], ema[i], ema[i - slope_bars]):
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
        if None in (rsi[i], rsi[i - 1]):
            continue
        if rsi[i - 1] < rsi_os <= rsi[i]:
            out.append((i, +1))
        elif rsi[i - 1] > ob >= rsi[i]:
            out.append((i, -1))
    return out


def gen_bb_touch(closes, bb_period, bb_std):
    out = []
    for i in range(bb_period - 1, len(closes)):
        upper, lower = bb_bands(closes[i - bb_period + 1: i + 1], bb_std)
        c = closes[i]
        if c <= lower:
            out.append((i, +1))
        elif c >= upper:
            out.append((i, -1))
    return out


def gen_macd(closes, fast, slow, sig_period):
    hist = macd_series(closes, fast, slow, sig_period)
    out  = []
    for i in range(1, len(closes)):
        if None in (hist[i], hist[i - 1]):
            continue
        if hist[i - 1] <= 0 < hist[i]:
            out.append((i, +1))
        elif hist[i - 1] >= 0 > hist[i]:
            out.append((i, -1))
    return out


def gen_signals(strat, closes):
    s = strat["sig"]
    if s == "rsi_ema":
        return gen_rsi_ema(closes, strat["rsi_period"], strat["rsi_entry"],
                           strat["ema_period"], strat["slope_bars"])
    if s == "rsi_reversal":
        return gen_rsi_reversal(closes, strat["rsi_period"], strat["rsi_os"])
    if s == "bb_touch":
        return gen_bb_touch(closes, strat["bb_period"], strat["bb_std"])
    if s == "macd":
        return gen_macd(closes, strat["fast"], strat["slow"], strat["sig_period"])
    raise ValueError(f"Unknown sig: {s}")


# ── Multiplier simulator ──────────────────────────────────────────────────────

def sim_multiplier(candles, signals, session_hours, multiplier, tp_pct):
    """
    Returns (total_trades, wins, net_pnl_in_R)
    where R = 1 stake unit, win = +profit_R, loss = -1.0
    """
    stop_pct   = 1.0 / multiplier
    commission = multiplier * COMMISSION_PCT  # as fraction of stake

    wins = losses = total = 0
    net  = 0.0

    for sig_idx, direction in signals:
        # Session filter
        if session_hours is not None:
            h = (candles[sig_idx]["epoch"] // 3600 + 3) % 24
            if h not in session_hours:
                continue

        entry = candles[sig_idx]["close"]

        if direction == +1:  # LONG
            stop_price = entry * (1 - stop_pct)
            tp_price   = entry * (1 + tp_pct)
        else:                # SHORT
            stop_price = entry * (1 + stop_pct)
            tp_price   = entry * (1 - tp_pct)

        result = None
        for j in range(sig_idx + 1, len(candles)):
            lo = candles[j]["low"]
            hi = candles[j]["high"]

            if direction == +1:
                if lo <= stop_price:
                    result = -1.0
                    break
                if hi >= tp_price:
                    result = multiplier * tp_pct - commission
                    break
            else:
                if hi >= stop_price:
                    result = -1.0
                    break
                if lo <= tp_price:
                    result = multiplier * tp_pct - commission
                    break

        if result is None:
            continue  # position still open at end of segment — skip

        total += 1
        if result > 0:
            wins += 1
        else:
            losses += 1
        net += result

    return total, wins, net


# ── Walk-forward ──────────────────────────────────────────────────────────────

def walk_forward(strat, candles, multiplier, tp_pct):
    closes  = [c["close"] for c in candles]
    hours   = [(c["epoch"] // 3600 + 3) % 24 for c in candles]
    session = strat.get("session")
    n       = len(candles)
    ws      = n // WINDOWS

    total_trades = wins = net = passes = 0

    for w in range(WINDOWS):
        s   = w * ws
        e   = s + ws if w < WINDOWS - 1 else n
        seg = candles[s:e]

        # Hours in this segment
        seg_hours = [(c["epoch"] // 3600 + 3) % 24 for c in seg]

        sigs   = gen_signals(strat, [c["close"] for c in seg])
        t, w_, n_ = sim_multiplier(seg, sigs, session, multiplier, tp_pct)

        if t >= MIN_TRADES and n_ > 0:
            passes += 1

        total_trades += t
        wins         += w_
        net          += n_

    expectancy = net / total_trades if total_trades else 0.0
    wr         = wins / total_trades * 100 if total_trades else 0.0
    return total_trades, wr, expectancy, passes


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",  default="all", choices=["all", "forex", "vol"])
    parser.add_argument("--symbol", default="ALL")
    parser.add_argument("--fresh",  action="store_true")
    args = parser.parse_args()

    strats = []
    if args.phase in ("all", "forex"):
        strats += FOREX_STRATS
    if args.phase in ("all", "vol"):
        strats += VOL_STRATS

    if args.symbol.upper() != "ALL":
        strats = [s for s in strats if s["symbol"] == args.symbol.upper()]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Cache candle data per symbol
    symbol_data = {}
    for strat in strats:
        sym = strat["symbol"]
        if sym in symbol_data:
            continue
        print(f"Loading {sym}...", end=" ", flush=True)
        try:
            c = await fetch_ohlc(sym, args.fresh)
            symbol_data[sym] = c
            print(f"{len(c)} candles")
        except Exception as e:
            print(f"FAILED: {e}")
            symbol_data[sym] = None

    SEP  = "=" * 82
    THIN = "-" * 82

    # Per-strategy sweep
    processed = set()
    for strat in strats:
        sym     = strat["symbol"]
        candles = symbol_data.get(sym)
        if candles is None or len(candles) < 200:
            continue

        key = strat["label"]
        if key in processed:
            continue
        processed.add(key)

        print()
        print(SEP)
        print(f"  {strat['label']}  |  {sym}  |  {len(candles)} candles  |  {WINDOWS}-fold WF")
        sess = strat.get("session")
        if sess:
            sh = sorted(sess)
            print(f"  Session: {sh[0]}-{sh[-1]} EAT  |  commission={COMMISSION_PCT*100:.2f}%/notional")
        else:
            print(f"  Session: 24/7  |  commission={COMMISSION_PCT*100:.2f}%/notional")
        print(f"  {'Multiplier':>10}  {'TP%':>5}  {'Trades':>6}  {'WR%':>6}  {'Exp(R)':>8}  {'Passes':>6}")
        print(f"  {'-'*10}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*6}")

        results = []
        for mult in MULTIPLIERS:
            for tp in TP_PCTS:
                t, wr, exp, passes = walk_forward(strat, candles, mult, tp)
                if t < MIN_TRADES:
                    continue
                results.append((mult, tp, t, wr, exp, passes))

        results.sort(key=lambda r: (r[5], r[4]), reverse=True)

        for mult, tp, t, wr, exp, passes in results:
            flag = " ***" if passes == WINDOWS and exp > 0 else ""
            print(
                f"  {mult:>10}x  {tp*100:>4.1f}%  {t:>6}  {wr:>6.1f}%  {exp:>+8.3f}R  {passes}/{WINDOWS}{flag}"
            )

    print()
    print(SEP)
    print("  *** = 4/4 windows profitable (net_pnl > 0) AND minimum trades met")
    print("  Exp(R)  = net profit per trade in stake units (R)")
    print("  Stop-out= 1/multiplier adverse move (e.g. 100x -> 1.0%, 500x -> 0.2%)")


if __name__ == "__main__":
    asyncio.run(main())
