"""
15-minute bar strategy research sweep.
Standalone — does not modify config.yaml or any existing source files.

Tests RSI Mean Reversion, MACD, BB Squeeze, Donchian, EMA Cross
on frxXAUUSD, frxXAGUSD, frxGBPUSD using 15-min candles.

Walk-forward: 3 chronological folds, STRONG = 3/3 folds with EV > 0.05.
"""

import asyncio
import json
import math
from datetime import datetime
from pathlib import Path

import websockets

# ── Config ────────────────────────────────────────────────────────────────

SYMBOLS      = ["frxXAUUSD", "frxXAGUSD", "frxGBPUSD"]
GRANULARITY  = 900     # 15-minute bars
COUNT        = 5000    # ~52 days
MULTIPLIER   = 100
COMMISSION   = 0.02
MAX_BARS     = 96      # 24h timeout at 15-min resolution
LEGACY_WS    = "wss://ws.derivws.com/websockets/v3?app_id=1089"
CACHE_DIR    = Path("data/15min")

SL_OPTIONS   = [0.003, 0.005, 0.0075, 0.010, 0.015]
TP_OPTIONS   = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030]

# ── Data fetch ────────────────────────────────────────────────────────────

async def fetch_candles(symbol: str) -> list[dict]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{symbol}.json"
    if cache.exists():
        with open(cache) as f:
            data = json.load(f)
        print(f"  {symbol}: loaded {len(data)} bars from cache")
        return data

    print(f"  {symbol}: fetching from API...")
    async with websockets.connect(LEGACY_WS) as ws:
        await ws.send(json.dumps({
            "ticks_history": symbol,
            "style":         "candles",
            "granularity":   GRANULARITY,
            "count":         COUNT,
            "end":           "latest",
            "req_id":        1,
        }))
        raw = await asyncio.wait_for(ws.recv(), timeout=30)
        msg = json.loads(raw)
        if msg.get("error"):
            raise RuntimeError(f"{symbol}: {msg['error']['message']}")
        candles = [
            {
                "epoch": int(c["epoch"]),
                "open":  float(c["open"]),
                "high":  float(c["high"]),
                "low":   float(c["low"]),
                "close": float(c["close"]),
            }
            for c in msg["candles"]
        ]
    with open(cache, "w") as f:
        json.dump(candles, f)
    print(f"  {symbol}: fetched {len(candles)} bars")
    return candles

# ── Indicators (O(n) pre-computed series) ────────────────────────────────

def ema_series(values: list, period: int) -> list:
    out   = [None] * len(values)
    k     = 2 / (period + 1)
    count = 0
    s     = 0.0
    last  = None
    for i, v in enumerate(values):
        if v is None:
            continue
        if count < period:
            s += v
            count += 1
            if count == period:
                last   = s / period
                out[i] = last
        else:
            last   = v * k + last * (1 - k)
            out[i] = last
    return out

def rsi_series(closes: list, period: int) -> list:
    out = [None] * len(closes)
    if len(closes) < period + 1:
        return out
    gains  = [max(closes[i] - closes[i-1], 0) for i in range(1, len(closes))]
    losses = [max(closes[i-1] - closes[i], 0) for i in range(1, len(closes))]
    ag = sum(gains[:period])  / period
    al = sum(losses[:period]) / period
    def _rsi(ag, al):
        return 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    out[period] = _rsi(ag, al)
    for i in range(period + 1, len(closes)):
        ag = (ag * (period - 1) + gains[i-1])  / period
        al = (al * (period - 1) + losses[i-1]) / period
        out[i] = _rsi(ag, al)
    return out

def bb_stats(closes: list, period: int, std_mult: float):
    mids   = [None] * len(closes)
    widths = [None] * len(closes)
    for i in range(period - 1, len(closes)):
        w   = closes[i-period+1:i+1]
        mid = sum(w) / period
        std = math.sqrt(sum((x - mid)**2 for x in w) / period)
        mids[i]   = mid
        widths[i] = (2 * std_mult * std / mid) if mid else 0
    return mids, widths

def macd_series(closes: list, fast: int, slow: int, signal: int):
    ef   = ema_series(closes, fast)
    es   = ema_series(closes, slow)
    line = [
        (f - s) if (f is not None and s is not None) else None
        for f, s in zip(ef, es)
    ]
    sig  = ema_series(line, signal)
    hist = [
        (l - s) if (l is not None and s is not None) else None
        for l, s in zip(line, sig)
    ]
    return line, sig, hist

# ── Signal generators ─────────────────────────────────────────────────────

def sig_rsi(closes, period, os_lvl, ob_lvl):
    rsi  = rsi_series(closes, period)
    sigs = []
    for i in range(1, len(closes)):
        if rsi[i] is None or rsi[i-1] is None:
            continue
        if rsi[i-1] >= os_lvl > rsi[i]:
            sigs.append((i, +1))
        elif rsi[i-1] <= ob_lvl < rsi[i]:
            sigs.append((i, -1))
    return sigs

def sig_macd(closes, fast, slow, signal_p):
    line, _, hist = macd_series(closes, fast, slow, signal_p)
    sigs = []
    for i in range(1, len(closes)):
        if hist[i] is None or hist[i-1] is None:
            continue
        if hist[i-1] < 0 <= hist[i] and (line[i] or 0) > 0:
            sigs.append((i, +1))
        elif hist[i-1] > 0 >= hist[i] and (line[i] or 0) < 0:
            sigs.append((i, -1))
    return sigs

def sig_bb_squeeze(closes, period, std_mult, sq_pct):
    mids, widths = bb_stats(closes, period, std_mult)
    sigs         = []
    width_hist   = []
    prev_sq      = False
    for i in range(period, len(closes)):
        w = widths[i]
        if w is None:
            continue
        width_hist.append(w)
        if len(width_hist) < 20:
            continue
        threshold = sorted(width_hist)[int(len(width_hist) * sq_pct / 100)]
        in_sq     = w <= threshold
        if prev_sq and not in_sq:
            direction = +1 if closes[i] > (mids[i] or closes[i]) else -1
            sigs.append((i, direction))
        prev_sq = in_sq
    return sigs

def sig_donchian(candles, period):
    highs  = [c["high"]  for c in candles]
    lows   = [c["low"]   for c in candles]
    sigs   = []
    for i in range(period + 1, len(candles)):
        ch_high = max(highs[i-period-1:i-1])
        ch_low  = min(lows [i-period-1:i-1])
        if highs[i] > ch_high:
            sigs.append((i, +1))
        elif lows[i] < ch_low:
            sigs.append((i, -1))
    return sigs

def sig_ema_cross(closes, fast, slow):
    ef   = ema_series(closes, fast)
    es   = ema_series(closes, slow)
    sigs = []
    for i in range(1, len(closes)):
        if any(x is None for x in [ef[i], ef[i-1], es[i], es[i-1]]):
            continue
        if ef[i-1] <= es[i-1] and ef[i] > es[i]:
            sigs.append((i, +1))
        elif ef[i-1] >= es[i-1] and ef[i] < es[i]:
            sigs.append((i, -1))
    return sigs

# ── Simulator ─────────────────────────────────────────────────────────────

def simulate(candles, signals, sl_pct, tp_pct, max_bars=MAX_BARS):
    win_r  = MULTIPLIER * tp_pct - COMMISSION
    loss_r = -(MULTIPLIER * sl_pct + COMMISSION)
    sig_d  = {i: d for i, d in signals}
    results, open_pos = [], []

    for i, c in enumerate(candles):
        still = []
        for (entry, direction) in open_pos:
            if i <= entry:
                still.append((entry, direction))
                continue
            ep  = candles[entry]["close"]
            h, l = c["high"], c["low"]
            tp_p = ep * (1 + direction * tp_pct)
            sl_p = ep * (1 - direction * sl_pct)
            elapsed = i - entry
            if direction > 0:
                hit_sl = l <= sl_p
                hit_tp = h >= tp_p
            else:
                hit_sl = h >= sl_p
                hit_tp = l <= tp_p
            if hit_sl:
                results.append(loss_r)
            elif hit_tp:
                results.append(win_r)
            elif elapsed >= max_bars:
                results.append(loss_r)
            else:
                still.append((entry, direction))
        open_pos = still

        if i in sig_d and not open_pos:
            open_pos.append((i, sig_d[i]))

    n    = len(results)
    wins = sum(1 for r in results if r > 0)
    ev   = sum(results) / n if n else 0.0
    return n, wins, ev

def walk_forward(candles, sigs, sl_pct, tp_pct, n_folds=3):
    n         = len(candles)
    fold_size = n // n_folds
    folds     = []
    for fold in range(n_folds):
        start = fold * fold_size
        end   = start + fold_size if fold < n_folds - 1 else n
        f_sigs    = [(i - start, d) for i, d in sigs if start <= i < end]
        f_candles = candles[start:end]
        folds.append(simulate(f_candles, f_sigs, sl_pct, tp_pct))
    return folds

# ── Strategy sweep ────────────────────────────────────────────────────────

def be_wr(sl, tp):
    return 100 * (MULTIPLIER * sl + COMMISSION) / (
        MULTIPLIER * tp - COMMISSION + MULTIPLIER * sl + COMMISSION
    )

def sweep(candles, label, combos):
    """Find best (combo, sl, tp) by full-data EV, then walk-forward validate."""
    closes   = [c["close"] for c in candles]
    best_ev  = -999
    best     = None

    for name, sigs in combos:
        if len(sigs) < 10:
            continue
        for sl in SL_OPTIONS:
            for tp in TP_OPTIONS:
                if tp <= sl:
                    continue
                n, wins, ev = simulate(candles, sigs, sl, tp)
                if n < 10 or ev <= best_ev:
                    continue
                best_ev = ev
                best    = (name, sigs, sl, tp, n, wins)

    if best is None:
        return None

    name, sigs, sl, tp, n, wins = best
    folds   = walk_forward(candles, sigs, sl, tp)
    n_pass  = sum(1 for _, _, ev in folds if ev > 0.05)
    mean_ev = sum(ev for _, _, ev in folds) / len(folds)
    verdict = "STRONG" if n_pass == 3 else ("WEAK" if n_pass >= 2 else "FAIL")

    print(f"\n  [{label}]")
    print(f"  Best: {name}  SL={sl*100:.2f}%/TP={tp*100:.2f}%  "
          f"BE WR={be_wr(sl,tp):.0f}%")
    print(f"  Full: N={n}  WR={wins/n*100:.0f}%  EV={best_ev:+.4f}")
    print(f"  {'Fold':<5} {'N':>5} {'WR':>7} {'EV':>9}  Result")
    print(f"  {'─'*38}")
    for i, (fn, fw, fev) in enumerate(folds, 1):
        wr_s = f"{fw/fn*100:.0f}%" if fn > 0 else "N/A"
        res  = "PASS" if fev > 0.05 else "FAIL"
        print(f"  {i:<5} {fn:>5} {wr_s:>7} {fev:>+9.4f}  {res}")
    print(f"  → {n_pass}/3 folds  MeanEV={mean_ev:+.4f}  {verdict}")

    return (label, name, sl, tp, n, wins, folds, verdict, mean_ev)

# ── Per-symbol research ───────────────────────────────────────────────────

def research(symbol, candles):
    closes = [c["close"] for c in candles]
    t0 = datetime.utcfromtimestamp(candles[0]["epoch"]).strftime("%Y-%m-%d")
    t1 = datetime.utcfromtimestamp(candles[-1]["epoch"]).strftime("%Y-%m-%d")
    print(f"\n{'='*60}")
    print(f"{symbol}  ({len(candles)} × 15-min bars | {t0} → {t1})")
    print(f"{'='*60}")

    results = []

    # RSI Mean Reversion
    combos = [
        (f"RSI({p}) OS={os}/OB={100-os}",
         sig_rsi(closes, p, os, 100-os))
        for p  in [5, 7, 10, 14]
        for os in [20, 25, 30]
    ]
    r = sweep(candles, "RSI Mean Reversion", combos)
    if r: results.append(r)

    # MACD
    combos = [
        (f"MACD({f},{s},{sg})", sig_macd(closes, f, s, sg))
        for f, s, sg in [(3,8,5),(5,13,5),(8,17,9),(12,26,9)]
    ]
    r = sweep(candles, "MACD Momentum", combos)
    if r: results.append(r)

    # BB Squeeze
    combos = [
        (f"BB({p},std={st},sq={sq}%)", sig_bb_squeeze(closes, p, st, sq))
        for p  in [20, 30, 50]
        for st in [1.5, 2.0]
        for sq in [20, 30]
    ]
    r = sweep(candles, "BB Squeeze", combos)
    if r: results.append(r)

    # Donchian
    combos = [
        (f"Donchian({p})", sig_donchian(candles, p))
        for p in [10, 20, 30, 50]
    ]
    r = sweep(candles, "Donchian Breakout", combos)
    if r: results.append(r)

    # EMA Cross
    combos = [
        (f"EMA({f}/{s})", sig_ema_cross(closes, f, s))
        for f, s in [(3,15),(5,20),(10,30),(20,50)]
    ]
    r = sweep(candles, "EMA Cross", combos)
    if r: results.append(r)

    print(f"\n  {'─'*52}")
    print(f"  SUMMARY")
    print(f"  {'─'*52}")
    print(f"  {'Strategy':<22} {'WF':>4} {'MeanEV':>9}  Verdict")
    print(f"  {'─'*52}")
    for label, name, sl, tp, n, w, folds, verdict, mean_ev in \
            sorted(results, key=lambda x: -x[8]):
        np = sum(1 for _,_,ev in folds if ev > 0.05)
        print(f"  {label:<22} {np}/3  {mean_ev:>+9.4f}  {verdict}")

    return results

# ── Entry point ───────────────────────────────────────────────────────────

async def main():
    print("15-MINUTE BAR STRATEGY RESEARCH")
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print("Fetching candle data...\n")

    all_results = {}
    for sym in SYMBOLS:
        try:
            candles = await fetch_candles(sym)
            all_results[sym] = research(sym, candles)
        except Exception as e:
            print(f"ERROR [{sym}]: {e}")

    print("\n\n" + "="*60)
    print("STRONG STRATEGIES ACROSS ALL SYMBOLS (3/3 folds)")
    print("="*60)
    any_strong = False
    for sym, results in all_results.items():
        strong = [r for r in results if r[7] == "STRONG"]
        if strong:
            any_strong = True
            print(f"\n{sym}:")
            for label, name, sl, tp, n, w, folds, verdict, mean_ev in \
                    sorted(strong, key=lambda x: -x[8]):
                print(f"  {label}: {name}")
                print(f"    SL={sl*100:.2f}%  TP={tp*100:.2f}%  "
                      f"BE WR={be_wr(sl,tp):.0f}%  MeanEV={mean_ev:+.4f}")
    if not any_strong:
        print("\nNo STRONG strategies found at 15-min resolution.")
        print("Consider: 5-min bars, different symbols, or longer data window.")

if __name__ == "__main__":
    asyncio.run(main())
