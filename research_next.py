"""
research_next.py — Phase 2 research sweep.
Standalone — does not modify config.yaml or any existing source files.

Three research directions:
  1. 5-min bars on Gold and Silver (all 5 strategies)
  2. RSI + EMA trend filter at 15-min on Gold and Silver
     (gold_trend approach ported to 15-min — more signals than hourly)
  3. Brent Crude (frxXBRUSD) at 1-hour and 15-min (all 5 strategies)
"""

import asyncio
import json
import math
from datetime import datetime
from pathlib import Path

import websockets

# ── Constants ─────────────────────────────────────────────────────────────

MULTIPLIER  = 100
COMMISSION  = 0.02
LEGACY_WS   = "wss://ws.derivws.com/websockets/v3?app_id=1089"

SL_OPTIONS  = [0.003, 0.005, 0.0075, 0.010, 0.015]
TP_OPTIONS  = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030]

# ── Data fetch ────────────────────────────────────────────────────────────

async def fetch(symbol: str, granularity: int, count: int, cache_dir: Path) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{symbol}.json"
    if cache.exists():
        with open(cache) as f:
            data = json.load(f)
        mins = granularity // 60
        print(f"  {symbol} @{mins}min: {len(data)} bars (cached)")
        return data

    mins = granularity // 60
    print(f"  {symbol} @{mins}min: fetching from API...")
    async with websockets.connect(LEGACY_WS) as ws:
        await ws.send(json.dumps({
            "ticks_history": symbol,
            "style":         "candles",
            "granularity":   granularity,
            "count":         count,
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
    print(f"  {symbol} @{mins}min: {len(candles)} bars fetched")
    return candles

# ── Indicators ────────────────────────────────────────────────────────────

def ema_series(values: list, period: int) -> list:
    out = [None] * len(values)
    k, count, s, last = 2 / (period + 1), 0, 0.0, None
    for i, v in enumerate(values):
        if v is None:
            continue
        if count < period:
            s += v; count += 1
            if count == period:
                last = s / period; out[i] = last
        else:
            last = v * k + last * (1 - k); out[i] = last
    return out

def rsi_series(closes: list, period: int) -> list:
    out = [None] * len(closes)
    if len(closes) < period + 1:
        return out
    gains  = [max(closes[i] - closes[i-1], 0) for i in range(1, len(closes))]
    losses = [max(closes[i-1] - closes[i], 0) for i in range(1, len(closes))]
    ag = sum(gains[:period]) / period
    al = sum(losses[:period]) / period
    def _r(ag, al): return 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    out[period] = _r(ag, al)
    for i in range(period + 1, len(closes)):
        ag = (ag * (period - 1) + gains[i-1])  / period
        al = (al * (period - 1) + losses[i-1]) / period
        out[i] = _r(ag, al)
    return out

def bb_stats(closes: list, period: int, std_mult: float):
    mids, widths = [None] * len(closes), [None] * len(closes)
    for i in range(period - 1, len(closes)):
        w = closes[i-period+1:i+1]
        m = sum(w) / period
        s = math.sqrt(sum((x - m)**2 for x in w) / period)
        mids[i]   = m
        widths[i] = (2 * std_mult * s / m) if m else 0
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
    rsi = rsi_series(closes, period)
    out = []
    for i in range(1, len(closes)):
        if rsi[i] is None or rsi[i-1] is None: continue
        if rsi[i-1] >= os_lvl > rsi[i]: out.append((i, +1))
        elif rsi[i-1] <= ob_lvl < rsi[i]: out.append((i, -1))
    return out

def sig_macd(closes, fast, slow, signal_p):
    line, _, hist = macd_series(closes, fast, slow, signal_p)
    out = []
    for i in range(1, len(closes)):
        if hist[i] is None or hist[i-1] is None: continue
        if hist[i-1] < 0 <= hist[i] and (line[i] or 0) > 0: out.append((i, +1))
        elif hist[i-1] > 0 >= hist[i] and (line[i] or 0) < 0: out.append((i, -1))
    return out

def sig_bb_squeeze(closes, period, std_mult, sq_pct):
    mids, widths = bb_stats(closes, period, std_mult)
    out, wh, prev_sq = [], [], False
    for i in range(period, len(closes)):
        w = widths[i]
        if w is None: continue
        wh.append(w)
        if len(wh) < 20: continue
        thr = sorted(wh)[int(len(wh) * sq_pct / 100)]
        in_sq = w <= thr
        if prev_sq and not in_sq:
            d = +1 if closes[i] > (mids[i] or closes[i]) else -1
            out.append((i, d))
        prev_sq = in_sq
    return out

def sig_donchian(candles, period):
    hs  = [c["high"] for c in candles]
    ls  = [c["low"]  for c in candles]
    out = []
    for i in range(period + 1, len(candles)):
        if hs[i] > max(hs[i-period-1:i-1]): out.append((i, +1))
        elif ls[i] < min(ls[i-period-1:i-1]): out.append((i, -1))
    return out

def sig_ema_cross(closes, fast, slow):
    ef, es = ema_series(closes, fast), ema_series(closes, slow)
    out = []
    for i in range(1, len(closes)):
        if any(x is None for x in [ef[i], ef[i-1], es[i], es[i-1]]): continue
        if ef[i-1] <= es[i-1] and ef[i] > es[i]: out.append((i, +1))
        elif ef[i-1] >= es[i-1] and ef[i] < es[i]: out.append((i, -1))
    return out

def sig_rsi_trend(closes, rsi_period, entry_rsi, ema_period, slope_bars=3):
    """RSI pullback entry filtered by EMA trend direction (gold_trend style)."""
    rsi = rsi_series(closes, rsi_period)
    ema = ema_series(closes, ema_period)
    out = []
    for i in range(max(rsi_period, ema_period) + slope_bars, len(closes)):
        if rsi[i] is None or ema[i] is None or ema[i - slope_bars] is None:
            continue
        slope_up = ema[i] > ema[i - slope_bars]
        ob_lvl   = 100 - entry_rsi
        if slope_up and rsi[i-1] >= entry_rsi > rsi[i]:
            out.append((i, +1))   # uptrend pullback → buy rise
        elif not slope_up and rsi[i-1] <= ob_lvl < rsi[i]:
            out.append((i, -1))   # downtrend pullback → buy fall
    return out

# ── Simulator ─────────────────────────────────────────────────────────────

def simulate(candles, signals, sl_pct, tp_pct, max_bars):
    win_r  =   MULTIPLIER * tp_pct - COMMISSION
    loss_r = -(MULTIPLIER * sl_pct + COMMISSION)
    sig_d  = {i: d for i, d in signals}
    results, open_pos = [], []

    for i, c in enumerate(candles):
        still = []
        for entry, direction in open_pos:
            if i <= entry:
                still.append((entry, direction)); continue
            ep = candles[entry]["close"]
            h, l = c["high"], c["low"]
            tp_p = ep * (1 + direction * tp_pct)
            sl_p = ep * (1 - direction * sl_pct)
            hit_sl = (l <= sl_p) if direction > 0 else (h >= sl_p)
            hit_tp = (h >= tp_p) if direction > 0 else (l <= tp_p)
            if   hit_sl:                results.append(loss_r)
            elif hit_tp:                results.append(win_r)
            elif i - entry >= max_bars: results.append(loss_r)
            else:                       still.append((entry, direction))
        open_pos = still
        if i in sig_d and not open_pos:
            open_pos.append((i, sig_d[i]))

    n    = len(results)
    wins = sum(1 for r in results if r > 0)
    return n, wins, (sum(results) / n if n else 0.0)

def walk_forward(candles, sigs, sl, tp, max_bars, n_folds=3):
    n, fold_size = len(candles), len(candles) // n_folds
    folds = []
    for fold in range(n_folds):
        start = fold * fold_size
        end   = start + fold_size if fold < n_folds - 1 else n
        fs    = [(i - start, d) for i, d in sigs if start <= i < end]
        folds.append(simulate(candles[start:end], fs, sl, tp, max_bars))
    return folds

# ── Sweep ─────────────────────────────────────────────────────────────────

def be_wr(sl, tp):
    return 100 * (MULTIPLIER * sl + COMMISSION) / \
           (MULTIPLIER * tp - COMMISSION + MULTIPLIER * sl + COMMISSION)

def sweep(candles, label, combos, max_bars):
    closes   = [c["close"] for c in candles]
    best_ev, best = -999, None
    for name, sigs in combos:
        if len(sigs) < 8: continue
        for sl in SL_OPTIONS:
            for tp in TP_OPTIONS:
                if tp <= sl: continue
                n, wins, ev = simulate(candles, sigs, sl, tp, max_bars)
                if n < 8 or ev <= best_ev: continue
                best_ev, best = ev, (name, sigs, sl, tp, n, wins)

    if best is None:
        print(f"\n  [{label}]  insufficient signals")
        return None

    name, sigs, sl, tp, n, wins = best
    folds   = walk_forward(candles, sigs, sl, tp, max_bars)
    n_pass  = sum(1 for _, _, ev in folds if ev > 0.05)
    mean_ev = sum(ev for _, _, ev in folds) / len(folds)
    verdict = "STRONG" if n_pass == 3 else ("WEAK" if n_pass >= 2 else "FAIL")

    print(f"\n  [{label}]")
    print(f"  Best: {name}  SL={sl*100:.2f}%/TP={tp*100:.2f}%  BE WR={be_wr(sl,tp):.0f}%")
    print(f"  Full: N={n}  WR={wins/n*100:.0f}%  EV={best_ev:+.4f}")
    print(f"  {'Fold':<5} {'N':>5} {'WR':>7} {'EV':>9}  Result")
    print(f"  {'─'*38}")
    for i, (fn, fw, fev) in enumerate(folds, 1):
        wr_s = f"{fw/fn*100:.0f}%" if fn > 0 else "N/A"
        print(f"  {i:<5} {fn:>5} {wr_s:>7} {fev:>+9.4f}  {'PASS' if fev > 0.05 else 'FAIL'}")
    print(f"  → {n_pass}/3  MeanEV={mean_ev:+.4f}  {verdict}")
    return (label, name, sl, tp, n, wins, folds, verdict, mean_ev)

# ── Full strategy sweep for a dataset ────────────────────────────────────

def run_all_strategies(symbol, candles, max_bars, include_rsi_trend=True):
    """Run all strategies + optional RSI+EMA trend filter on given candle set."""
    closes  = [c["close"] for c in candles]
    results = []

    # RSI Mean Reversion
    combos = [(f"RSI({p}) OS={os}/{100-os}", sig_rsi(closes, p, os, 100-os))
              for p in [5, 7, 10, 14] for os in [20, 25, 30]]
    r = sweep(candles, "RSI Mean Reversion", combos, max_bars)
    if r: results.append(r)

    # MACD
    combos = [(f"MACD({f},{s},{sg})", sig_macd(closes, f, s, sg))
              for f, s, sg in [(3,8,5),(5,13,5),(8,17,9),(12,26,9)]]
    r = sweep(candles, "MACD Momentum", combos, max_bars)
    if r: results.append(r)

    # BB Squeeze
    combos = [(f"BB({p},std={st},sq={sq}%)", sig_bb_squeeze(closes, p, st, sq))
              for p in [20, 30, 50] for st in [1.5, 2.0] for sq in [20, 30]]
    r = sweep(candles, "BB Squeeze", combos, max_bars)
    if r: results.append(r)

    # Donchian
    combos = [(f"Donchian({p})", sig_donchian(candles, p))
              for p in [10, 20, 30, 50]]
    r = sweep(candles, "Donchian Breakout", combos, max_bars)
    if r: results.append(r)

    # EMA Cross
    combos = [(f"EMA({f}/{s})", sig_ema_cross(closes, f, s))
              for f, s in [(3,15),(5,20),(10,30),(20,50)]]
    r = sweep(candles, "EMA Cross", combos, max_bars)
    if r: results.append(r)

    # RSI + EMA Trend Filter (gold_trend style)
    if include_rsi_trend:
        combos = [
            (f"RSI({rp})<{er} + EMA({ep})",
             sig_rsi_trend(closes, rp, er, ep))
            for rp in [5, 7, 10, 14]
            for er in [35, 40, 45]
            for ep in [20, 50, 100]
        ]
        r = sweep(candles, "RSI + EMA Trend Filter", combos, max_bars)
        if r: results.append(r)

    # Summary
    print(f"\n  {'─'*52}")
    print(f"  SUMMARY")
    print(f"  {'─'*52}")
    print(f"  {'Strategy':<26} {'WF':>4} {'MeanEV':>9}  Verdict")
    print(f"  {'─'*52}")
    for label, name, sl, tp, n, w, folds, verdict, mean_ev in \
            sorted(results, key=lambda x: -x[8]):
        np = sum(1 for _,_,ev in folds if ev > 0.05)
        print(f"  {label:<26} {np}/3  {mean_ev:>+9.4f}  {verdict}")

    return results

def section_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def candle_range(candles):
    t0 = datetime.utcfromtimestamp(candles[0]["epoch"]).strftime("%Y-%m-%d")
    t1 = datetime.utcfromtimestamp(candles[-1]["epoch"]).strftime("%Y-%m-%d")
    return f"{len(candles)} bars | {t0} → {t1}"

# ── Main ──────────────────────────────────────────────────────────────────

async def main():
    print("PHASE 2 RESEARCH — 5-min / RSI+EMA Filter / Brent Crude")
    print("Fetching data...\n")

    # ── Fetch ──────────────────────────────────────────────────
    fin_5min = {}
    for sym in ["frxXAUUSD", "frxXAGUSD"]:
        fin_5min[sym] = await fetch(sym, 300, 5000, Path("data/5min"))

    fin_15min = {}
    for sym in ["frxXAUUSD", "frxXAGUSD"]:
        fin_15min[sym] = await fetch(sym, 900, 5000, Path("data/15min"))

    brent = {}
    for gran, label in [(3600, "1h"), (900, "15min")]:
        try:
            brent[label] = await fetch("frxXBRUSD", gran, 5000,
                                       Path(f"data/brent_{label}"))
        except Exception as e:
            print(f"  frxXBRUSD @{label}: {e}")
            brent[label] = None

    # ── Research 1: 5-min Gold & Silver ───────────────────────
    section_header("RESEARCH 1: 5-MIN BARS — Gold & Silver")
    all_strong = []

    for sym in ["frxXAUUSD", "frxXAGUSD"]:
        candles = fin_5min[sym]
        print(f"\n{'─'*60}")
        print(f"  {sym} @5-min  ({candle_range(candles)})")
        print(f"{'─'*60}")
        res = run_all_strategies(sym, candles, max_bars=48,
                                 include_rsi_trend=True)
        for r in res:
            if r[7] == "STRONG":
                all_strong.append((sym, "5-min", r))

    # ── Research 2: RSI + EMA trend filter @ 15-min ───────────
    section_header("RESEARCH 2: RSI + EMA TREND FILTER @ 15-MIN — Gold & Silver")

    for sym in ["frxXAUUSD", "frxXAGUSD"]:
        candles = fin_15min[sym]
        closes  = [c["close"] for c in candles]
        print(f"\n{'─'*60}")
        print(f"  {sym} @15-min  ({candle_range(candles)})")
        print(f"{'─'*60}")
        combos = [
            (f"RSI({rp})<{er} + EMA({ep})",
             sig_rsi_trend(closes, rp, er, ep))
            for rp in [5, 7, 10, 14]
            for er in [35, 40, 45]
            for ep in [20, 50, 100]
        ]
        r = sweep(candles, "RSI + EMA Trend Filter @15-min", combos, 96)
        if r and r[7] == "STRONG":
            all_strong.append((sym, "15-min RSI+EMA", r))

    # ── Research 3: Brent Crude ───────────────────────────────
    section_header("RESEARCH 3: BRENT CRUDE (frxXBRUSD)")

    for label, max_bars in [("1h", 48), ("15min", 96)]:
        candles = brent.get(label)
        if not candles:
            print(f"\n  frxXBRUSD @{label}: no data (symbol may be unavailable)")
            continue
        print(f"\n{'─'*60}")
        print(f"  frxXBRUSD @{label}  ({candle_range(candles)})")
        print(f"{'─'*60}")
        res = run_all_strategies("frxXBRUSD", candles, max_bars=max_bars,
                                 include_rsi_trend=True)
        for r in res:
            if r[7] == "STRONG":
                all_strong.append(("frxXBRUSD", label, r))

    # ── Final summary ─────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("ALL STRONG STRATEGIES (3/3 folds) — READY TO IMPLEMENT")
    print(f"{'='*60}")
    if not all_strong:
        print("\n  No STRONG strategies found across all research directions.")
    else:
        for sym, tf, (label, name, sl, tp, n, w, folds, verdict, mean_ev) in \
                sorted(all_strong, key=lambda x: -x[2][8]):
            print(f"\n  {sym} @{tf}  [{label}]")
            print(f"    {name}")
            print(f"    SL={sl*100:.2f}%  TP={tp*100:.2f}%  "
                  f"BE WR={be_wr(sl,tp):.0f}%  MeanEV={mean_ev:+.4f}")

if __name__ == "__main__":
    asyncio.run(main())
