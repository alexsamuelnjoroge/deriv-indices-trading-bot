"""
XAU/USD binary strategy research — comprehensive signal sweep.

Previous RSI+EMA validation (research_scalp2.py, 5k bars) showed 3/3 STRONG
but proved overfitted: long-run WR is 50% vs BE=55.6% (sweep_binary_filters.py).

This sweep tests multiple signal families across 67k+ bars (3+ years)
with session and ATR filters to find any genuinely robust edge.

Signals tested:
  RSI reversal           — periods 7/10/14, OS 20/25/30/35
  EMA cross              — 5 standard pairs
  RSI+EMA (config style) — dip in trend
  BB bounce              — price touches / crosses band
  MACD histogram flip    — standard 12/26/9
  Stochastic %K          — %K exits OS/OB zone
  ATR momentum           — directional breakout when ATR expands
  Consecutive bars       — momentum after N bars same direction

Session windows (EAT = UTC+3):
  all, London (10-15), NY-overlap (15-20), London+NY (10-20)

ATR gate (relative to 100-bar mean ATR):
  0.0 (off), 1.0x, 1.25x

Hold durations: 1, 2, 3 bars (5, 10, 15 min)

Walk-forward: 4 windows. Printed: passes >= 3/4 only.
Min 15 trades per window.

Usage:
  python sweep_xau_binary.py            # use cached OHLC data
  python sweep_xau_binary.py --fresh    # re-fetch from Deriv
"""

import argparse, asyncio, json, sys
from pathlib import Path
import websockets

WS_URL    = "wss://ws.derivws.com/websockets/v3?app_id=1089"
CACHE_DIR = Path("data/scalp")

GRAN        = 300   # 5-min bars
WINDOWS     = 4
MIN_TRADES  = 15
PAYOUT      = 0.80
BE          = 1.0 / (1 + PAYOUT)   # 0.5556
ATR_PERIOD  = 14
ATR_WINDOW  = 100

HOLD_BARS   = [1, 2, 3]            # 5, 10, 15 min

SESSION_WINDOWS = {
    "all":        None,
    "London":     set(range(10, 16)),   # 07-12 UTC
    "NY-overlap": set(range(15, 21)),   # 12-17 UTC
    "London+NY":  set(range(10, 21)),   # 07-17 UTC
}

ATR_GATES = [0.0, 1.0, 1.25]


# ── Data ─────────────────────────────────────────────────────────────────────

async def fetch_ohlc(fresh: bool) -> list[dict]:
    cache = CACHE_DIR / "frxXAUUSD_300_ohlc.json"
    if cache.exists() and not fresh:
        with open(cache) as f:
            return json.load(f)
    async with websockets.connect(WS_URL, open_timeout=15) as ws:
        await ws.send(json.dumps({
            "ticks_history": "frxXAUUSD",
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
        {"epoch": c["open_time"], "open": float(c["open"]),
         "high": float(c["high"]), "low": float(c["low"]), "close": float(c["close"])}
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
    ch = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    g  = sum(c for c in ch[:period] if c > 0) / period
    l  = sum(-c for c in ch[:period] if c < 0) / period
    for i in range(period, len(closes)):
        d = ch[i-1]
        g = (g * (period-1) + max(d, 0)) / period
        l = (l * (period-1) + max(-d, 0)) / period
        out[i] = 100.0 if l == 0 else 100 - 100 / (1 + g/l)
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


def stoch_series(highs, lows, closes, period):
    """Stochastic %K."""
    out = [None] * len(closes)
    for i in range(period - 1, len(closes)):
        h  = max(highs[i - period + 1: i + 1])
        l  = min(lows[i - period + 1: i + 1])
        rng = h - l
        out[i] = 0.0 if rng == 0 else (closes[i] - l) / rng * 100
    return out


def atr_series(highs, lows, closes, period):
    trs = [None] + [
        max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        for i in range(1, len(closes))
    ]
    out = [None] * len(closes)
    for i in range(period, len(closes)):
        vals = [t for t in trs[i - period + 1: i + 1] if t is not None]
        if len(vals) == period:
            out[i] = sum(vals) / period
    return out


def atr_mean_series(atr, window):
    out = [None] * len(atr)
    for i in range(window, len(atr)):
        vals = [v for v in atr[i - window: i] if v is not None]
        if vals:
            out[i] = sum(vals) / len(vals)
    return out


def eat_hours(candles):
    return [(c["epoch"] // 3600 + 3) % 24 for c in candles]


# ── Signal generators ─────────────────────────────────────────────────────────

def sig_rsi_reversal(closes, period, os_lvl):
    ob  = 100 - os_lvl
    rsi = rsi_series(closes, period)
    out = []
    for i in range(period + 1, len(closes)):
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
        if rsi[i-1] is None:
            continue
        up = ema[i] > ema[i - slope_bars]
        if up and rsi[i-1] >= rsi_entry > rsi[i]:
            out.append((i, +1))
        elif not up and rsi[i-1] <= ob < rsi[i]:
            out.append((i, -1))
    return out


def sig_bb_bounce(closes, period, n_std):
    bb  = bb_series(closes, period, n_std)
    out = []
    for i in range(period, len(closes)):
        if bb[i] is None:
            continue
        upper, lower = bb[i]
        if closes[i] <= lower:
            out.append((i, +1))
        elif closes[i] >= upper:
            out.append((i, -1))
    return out


def sig_bb_reenter(closes, period, n_std):
    """Price was outside BB previous bar, returns inside this bar."""
    bb  = bb_series(closes, period, n_std)
    out = []
    for i in range(period + 1, len(closes)):
        if bb[i] is None or bb[i-1] is None:
            continue
        upper, lower = bb[i]
        pu, pl = bb[i-1]
        if closes[i-1] < pl and closes[i] >= pl:
            out.append((i, +1))
        elif closes[i-1] > pu and closes[i] <= pu:
            out.append((i, -1))
    return out


def sig_macd(closes, fast=12, slow=26, signal_period=9):
    ef  = ema_series(closes, fast)
    es  = ema_series(closes, slow)
    macd = [ef[i] - es[i] if ef[i] is not None and es[i] is not None else None
            for i in range(len(closes))]
    macd_c = [m for m in macd if m is not None]
    if len(macd_c) < signal_period:
        return []
    # compute signal line on the non-None part
    sig_line = [None] * len(closes)
    k  = 2 / (signal_period + 1)
    val = sum(macd_c[:signal_period]) / signal_period
    # find start index
    start = next(i for i in range(len(closes)) if macd[i] is not None)
    sig_line[start + signal_period - 1] = val
    j = signal_period
    for i in range(start + signal_period, len(closes)):
        if macd[i] is None:
            continue
        val = macd[i] * k + val * (1 - k)
        sig_line[i] = val
        j += 1
    hist = [macd[i] - sig_line[i]
            if macd[i] is not None and sig_line[i] is not None else None
            for i in range(len(closes))]
    out = []
    for i in range(slow + signal_period + 1, len(closes)):
        if hist[i] is None or hist[i-1] is None:
            continue
        if hist[i-1] < 0 <= hist[i]:
            out.append((i, +1))
        elif hist[i-1] > 0 >= hist[i]:
            out.append((i, -1))
    return out


def sig_stoch(closes, highs, lows, period, os_lvl):
    ob   = 100 - os_lvl
    stoch = stoch_series(highs, lows, closes, period)
    out  = []
    for i in range(period + 1, len(closes)):
        if stoch[i] is None or stoch[i-1] is None:
            continue
        if stoch[i-1] < os_lvl <= stoch[i]:
            out.append((i, +1))
        elif stoch[i-1] > ob >= stoch[i]:
            out.append((i, -1))
    return out


def sig_atr_momentum(candles, atr_thresh_mult):
    """Trade direction of bar when ATR expands beyond thresh_mult * mean ATR."""
    closes = [c["close"] for c in candles]
    highs  = [c["high"]  for c in candles]
    lows   = [c["low"]   for c in candles]
    atr    = atr_series(highs, lows, closes, ATR_PERIOD)
    am     = atr_mean_series(atr, ATR_WINDOW)
    out    = []
    for i in range(ATR_WINDOW + ATR_PERIOD + 1, len(closes)):
        if atr[i] is None or am[i] is None:
            continue
        if atr[i] < atr_thresh_mult * am[i]:
            continue
        direction = +1 if candles[i]["close"] > candles[i]["open"] else -1
        out.append((i, direction))
    return out


def sig_consec_bars(closes, n, reversal=False):
    """N consecutive bars same direction then trade that direction (momentum) or opposite (reversal)."""
    out = []
    for i in range(n, len(closes)):
        dirs = [1 if closes[j] > closes[j-1] else -1 for j in range(i-n+1, i+1)]
        if len(set(dirs)) == 1:  # all same direction
            d = dirs[0] if not reversal else -dirs[0]
            out.append((i, d))
    return out


# ── Filter + simulation ───────────────────────────────────────────────────────

def apply_filters(signals, atr_vals, atr_mean_vals, hours, atr_gate, session):
    out = []
    for i, d in signals:
        if atr_gate > 0 and i < len(atr_vals):
            a, m = atr_vals[i], atr_mean_vals[i]
            if a is None or m is None or a < atr_gate * m:
                continue
        if session is not None and i < len(hours) and hours[i] not in session:
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
    if total < MIN_TRADES:
        return None
    wr = wins / total
    return wins, total, wr


# ── Walk-forward evaluation ───────────────────────────────────────────────────

def evaluate(candles, atr_all, am_all, hours_all, sig_fn, hold_bars, atr_gate, session):
    closes = [c["close"] for c in candles]
    n, ws  = len(candles), len(candles) // WINDOWS
    results = []
    for w in range(WINDOWS):
        s = w * ws
        e = s + ws if w < WINDOWS - 1 else n
        seg_c  = closes[s:e]
        seg_a  = atr_all[s:e]
        seg_m  = am_all[s:e]
        seg_h  = hours_all[s:e]
        sigs   = sig_fn(seg_c)
        sigs   = apply_filters(sigs, seg_a, seg_m, seg_h, atr_gate, session)
        results.append(sim_binary(seg_c, sigs, hold_bars))
    return results


def evaluate_ohlc(candles, atr_all, am_all, hours_all, sig_fn, hold_bars, atr_gate, session):
    """For signals that need OHLC (stoch, atr_momentum)."""
    closes = [c["close"] for c in candles]
    n, ws  = len(candles), len(candles) // WINDOWS
    results = []
    for w in range(WINDOWS):
        s = w * ws
        e = s + ws if w < WINDOWS - 1 else n
        seg_c  = candles[s:e]
        seg_cl = closes[s:e]
        seg_a  = atr_all[s:e]
        seg_m  = am_all[s:e]
        seg_h  = hours_all[s:e]
        sigs   = sig_fn(seg_c)
        sigs   = apply_filters(sigs, seg_a, seg_m, seg_h, atr_gate, session)
        results.append(sim_binary(seg_cl, sigs, hold_bars))
    return results


def summarize(results):
    valid  = [r for r in results if r is not None]
    if not valid:
        return None
    passes = sum(1 for r in valid if r[2] >= BE)
    tw     = sum(r[0] for r in valid)
    tt     = sum(r[1] for r in valid)
    wr     = tw / tt if tt else 0
    ev     = (wr - BE) * PAYOUT
    return wr, tt, ev, passes, len(valid)


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading frxXAUUSD OHLC data...", end=" ", flush=True)
    candles = await fetch_ohlc(args.fresh)
    print(f"{len(candles)} candles ({len(candles)*GRAN//3600//24} days)")

    closes = [c["close"] for c in candles]
    highs  = [c["high"]  for c in candles]
    lows   = [c["low"]   for c in candles]
    atr_all = atr_series(highs, lows, closes, ATR_PERIOD)
    am_all  = atr_mean_series(atr_all, ATR_WINDOW)
    hours   = eat_hours(candles)

    SEP  = "=" * 100
    THIN = "-" * 100

    print(f"\nXAU/USD Binary Sweep | payout={PAYOUT*100:.0f}% BE={BE*100:.1f}%")
    print(f"Walk-forward: {WINDOWS} windows | Min {MIN_TRADES} trades/window | Show: passes >= 3/{WINDOWS}")
    print(SEP)

    results_all = []

    def run_signal(label, sig_fn, use_ohlc=False, skip_atr_gate=False):
        for hold_bars in HOLD_BARS:
            for sess_name, session in SESSION_WINDOWS.items():
                gates = [0.0] if skip_atr_gate else ATR_GATES
                for atr_gate in gates:
                    if use_ohlc:
                        results = evaluate_ohlc(
                            candles, atr_all, am_all, hours,
                            sig_fn, hold_bars, atr_gate, session,
                        )
                    else:
                        results = evaluate(
                            candles, atr_all, am_all, hours,
                            lambda c, f=sig_fn: f(c),
                            hold_bars, atr_gate, session,
                        )
                    summ = summarize(results)
                    if summ is None:
                        continue
                    wr, trades, ev, passes, valid = summ
                    results_all.append({
                        "label":    label,
                        "hold":     hold_bars,
                        "session":  sess_name,
                        "atr_gate": atr_gate,
                        "wr":       wr,
                        "trades":   trades,
                        "ev":       ev,
                        "passes":   passes,
                        "valid":    valid,
                    })

    # RSI reversal
    for period in [7, 10, 14]:
        for os_lvl in [20, 25, 30, 35]:
            run_signal(
                f"RSI({period:2d}) OS={os_lvl}",
                lambda c, p=period, o=os_lvl: sig_rsi_reversal(c, p, o),
            )

    # EMA cross
    for fast, slow in [(5, 20), (5, 30), (8, 21), (10, 30), (10, 50)]:
        run_signal(
            f"EMA({fast:2d}/{slow:2d}) cross",
            lambda c, f=fast, s=slow: sig_ema_cross(c, f, s),
        )

    # RSI+EMA
    for rp in [7, 10]:
        for re in [40, 45]:
            for ep in [20, 50]:
                for sb in [3, 5]:
                    run_signal(
                        f"RSI({rp})+EMA({ep:2d}) e={re} s={sb}",
                        lambda c, r=rp, e=re, p=ep, s=sb: sig_rsi_ema(c, r, e, p, s),
                    )

    # BB bounce
    for period in [10, 20]:
        for n_std in [1.5, 2.0, 2.5]:
            run_signal(
                f"BB({period:2d}/{n_std}) touch",
                lambda c, p=period, n=n_std: sig_bb_bounce(c, p, n),
            )
            run_signal(
                f"BB({period:2d}/{n_std}) reenter",
                lambda c, p=period, n=n_std: sig_bb_reenter(c, p, n),
            )

    # MACD histogram flip
    run_signal("MACD(12/26/9) flip", lambda c: sig_macd(c))

    # Stochastic
    def stoch_wrap(period, os_lvl):
        def fn(seg_c):
            n = len(seg_c)
            hi = highs[:n]
            lo = lows[:n]
            return sig_stoch(seg_c, hi, lo, period, os_lvl)
        return fn

    for period in [5, 14]:
        for os_lvl in [20, 25]:
            run_signal(
                f"Stoch({period:2d}) OS={os_lvl}",
                lambda c, p=period, o=os_lvl: sig_stoch(
                    c,
                    [candles[i]["high"] for i in range(len(c))],
                    [candles[i]["low"]  for i in range(len(c))],
                    p, o,
                ),
            )

    # ATR momentum (no ATR gate — it's already momentum-based)
    for mult in [1.25, 1.5, 1.75]:
        run_signal(
            f"ATR-momentum {mult}x",
            lambda segs, m=mult: sig_atr_momentum(candles[:len(segs)], m),
            use_ohlc=True,
            skip_atr_gate=True,
        )

    # Consecutive bars momentum / reversal
    for n in [2, 3]:
        run_signal(
            f"Consec-{n} momentum",
            lambda c, k=n: sig_consec_bars(c, k, reversal=False),
        )
        run_signal(
            f"Consec-{n} reversal",
            lambda c, k=n: sig_consec_bars(c, k, reversal=True),
        )

    # ── Report ────────────────────────────────────────────────────────────────

    strong = sorted(
        [r for r in results_all if r["passes"] >= 3],
        key=lambda x: (-x["passes"], -x["ev"]),
    )

    if not strong:
        print("\nNo combination reached 3/4 passes. XAU binary at 80% payout has no detectable edge")
        print("across these signal families over 3+ years of data.")
        print("Consider: different hold duration, different product (Multiplier), or different symbol.")
        return

    print(f"\nResults with >= 3/{WINDOWS} passes (sorted by passes then EV):")
    print(THIN)
    print(f"  {'Signal':38s} hold  session      atr   WR      EV%     Trades  Pass")
    print(THIN)
    prev_pass = None
    for r in strong[:40]:
        if r["passes"] != prev_pass:
            prev_pass = r["passes"]
            print(f"\n  --- {r['passes']}/{WINDOWS} passes ---")
        flag = " <<<" if r["passes"] == WINDOWS and r["ev"] > 0.02 else ""
        print(
            f"  {r['label']:38s} {r['hold']}bar  {r['session']:12s} {r['atr_gate']:.2f}  "
            f"{r['wr']*100:5.1f}%  {r['ev']*100:+6.2f}%  {r['trades']:7d}  "
            f"{r['passes']}/{r['valid']}{flag}"
        )

    print(f"\n{SEP}")
    print(f"BE={BE*100:.1f}% | payout={PAYOUT*100:.0f}% | {WINDOWS} x {len(candles)//WINDOWS:,} bars/window")
    print(f"<<< = 4/4 passes AND EV > 2%")


if __name__ == "__main__":
    asyncio.run(main())
