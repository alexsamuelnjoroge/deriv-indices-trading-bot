"""
Pro strategy research — 5 institutional-grade strategies on real market pairs.

Strategies:
  1. Multi-TF Trend Pullback  — 1h trend (EMA200) + 5min RSI pullback entry
  2. Stochastic OB/OS Cross   — Stoch(5,3,3) cross in extreme zone + EMA slope filter
  3. Session Open Breakout    — Asian range breakout at London/NY open
  4. RSI Divergence           — Classic bearish/bullish divergence signal
  5. Pivot Point Bounce       — Daily pivot levels + RSI confirmation

Symbols + actual payouts (verified):
  frxXAUUSD  Gold    80%  min=5min
  frxUSDJPY  USDJPY  90%  min=15min
  frxEURUSD  EURUSD  85%  min=15min
  frxGBPUSD  GBPUSD  88%  min=15min
  frxAUDUSD  AUDUSD  85%  min=15min

Usage:
  python3 research_pro.py
  python3 research_pro.py --symbol frxXAUUSD   # single symbol
  python3 research_pro.py --strategy mtf        # single strategy
"""

import asyncio
import json
import sys
from pathlib import Path

import websockets

LEGACY_WS  = "wss://ws.derivws.com/websockets/v3?app_id=1089"
CACHE_DIR  = Path("data/scalp")   # reuse existing 5-min candle cache
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_1H   = Path("data/pro")
CACHE_1H.mkdir(parents=True, exist_ok=True)

FOLDS      = 3
MIN_TRADES = 8
GRAN_5M    = 300
GRAN_1H    = 3600

SYMBOL_CONFIG = {
    "frxXAUUSD": {"label": "Gold   ", "payout": 0.80, "min_hold_m": 5},
    "frxUSDJPY": {"label": "USD/JPY", "payout": 0.90, "min_hold_m": 15},
    "frxEURUSD": {"label": "EUR/USD", "payout": 0.85, "min_hold_m": 15},
    "frxGBPUSD": {"label": "GBP/USD", "payout": 0.88, "min_hold_m": 15},
    "frxAUDUSD": {"label": "AUD/USD", "payout": 0.85, "min_hold_m": 15},
}

# CLI filters
FILTER_SYM  = next((a.split("=")[1] for a in sys.argv if a.startswith("--symbol=")), None)
FILTER_STRAT = next((a.split("=")[1] for a in sys.argv if a.startswith("--strategy=")), None)


# ═══════════════════════════════════════════════════════════════
# Data fetching
# ═══════════════════════════════════════════════════════════════

async def fetch_closes(symbol: str, gran: int, count: int = 5000) -> list[float]:
    cache_dir = CACHE_DIR if gran == GRAN_5M else CACHE_1H
    cache = cache_dir / f"{symbol}_{gran}.json"
    if cache.exists():
        with open(cache) as f:
            return json.load(f)

    async with websockets.connect(LEGACY_WS, open_timeout=15) as ws:
        await ws.send(json.dumps({
            "ticks_history": symbol,
            "style":         "candles",
            "granularity":   gran,
            "count":         count,
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


async def fetch_ohlc(symbol: str, gran: int, count: int = 5000) -> list[dict]:
    """Returns list of {open, high, low, close} dicts."""
    cache = CACHE_1H / f"{symbol}_{gran}_ohlc.json"
    if cache.exists():
        with open(cache) as f:
            return json.load(f)

    async with websockets.connect(LEGACY_WS, open_timeout=15) as ws:
        await ws.send(json.dumps({
            "ticks_history": symbol,
            "style":         "candles",
            "granularity":   gran,
            "count":         count,
            "end":           "latest",
            "req_id":        1,
        }))
        raw = await asyncio.wait_for(ws.recv(), timeout=30)
        msg = json.loads(raw)

    if msg.get("error"):
        raise RuntimeError(msg["error"]["message"])

    bars = [{"open": float(c["open"]), "high": float(c["high"]),
             "low":  float(c["low"]),  "close": float(c["close"]),
             "epoch": int(c["epoch"])} for c in msg["candles"]]
    with open(cache, "w") as f:
        json.dump(bars, f)
    return bars


# ═══════════════════════════════════════════════════════════════
# Indicators
# ═══════════════════════════════════════════════════════════════

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


def stoch_series(highs, lows, closes, k_period=5, d_period=3):
    """Returns (%K, %D) series."""
    n   = len(closes)
    raw_k = [None] * n
    for i in range(k_period - 1, n):
        hi = max(highs[i - k_period + 1: i + 1])
        lo = min(lows[i - k_period + 1:  i + 1])
        raw_k[i] = 100 * (closes[i] - lo) / (hi - lo) if hi != lo else 50.0

    # Smooth %K with d_period SMA
    smooth_k = [None] * n
    for i in range(k_period + d_period - 2, n):
        vals = [raw_k[j] for j in range(i - d_period + 1, i + 1) if raw_k[j] is not None]
        if len(vals) == d_period:
            smooth_k[i] = sum(vals) / d_period

    # %D = SMA(smooth_k, d_period)
    d = [None] * n
    for i in range(k_period + 2 * d_period - 3, n):
        vals = [smooth_k[j] for j in range(i - d_period + 1, i + 1) if smooth_k[j] is not None]
        if len(vals) == d_period:
            d[i] = sum(vals) / d_period

    return smooth_k, d


def atr_series(highs, lows, closes, period=14):
    n  = len(closes)
    tr = [None] * n
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i]  - closes[i - 1]))
    out = [None] * n
    if n < period + 1:
        return out
    out[period] = sum(tr[1:period + 1]) / period
    for i in range(period + 1, n):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


# ═══════════════════════════════════════════════════════════════
# Strategy 1 — Multi-Timeframe Trend Pullback
# ═══════════════════════════════════════════════════════════════

def sig_mtf(closes_5m: list[float], closes_1h: list[float],
            ema_period=200, slope_bars=3,
            rsi_period=14, rsi_entry=40.0) -> list[tuple]:
    """
    1h EMA(200) defines trend direction.
    5min RSI pullback gives entry.
    Aligns 1h bars to 5min bar index (1h = 12 × 5min bars).
    """
    ratio   = GRAN_1H // GRAN_5M   # 12 five-min bars per 1h bar
    n5      = len(closes_5m)
    n1h     = len(closes_1h)
    ema_1h  = ema_series(closes_1h, ema_period)
    rsi_5m  = rsi_series(closes_5m, rsi_period)
    signals = []

    start = max(ema_period + slope_bars, rsi_period + 1) * ratio
    for i in range(start, n5):
        # Map 5min index to 1h index (approximate alignment)
        j = min(i // ratio, n1h - 1)
        if ema_1h[j] is None or ema_1h[j - slope_bars] is None:
            continue
        if rsi_5m[i] is None or rsi_5m[i - 1] is None:
            continue

        trend_up   = ema_1h[j] > ema_1h[j - slope_bars]
        trend_down = ema_1h[j] < ema_1h[j - slope_bars]
        ob = 100 - rsi_entry

        if trend_up and rsi_5m[i - 1] >= rsi_entry > rsi_5m[i]:
            signals.append((i, +1))
        elif trend_down and rsi_5m[i - 1] <= ob < rsi_5m[i]:
            signals.append((i, -1))

    return signals


# ═══════════════════════════════════════════════════════════════
# Strategy 2 — Stochastic OB/OS Cross + EMA Filter
# ═══════════════════════════════════════════════════════════════

def sig_stoch(ohlc: list[dict], k_period=5, d_period=3,
              ob=80.0, os_=20.0, ema_period=50, slope_bars=3) -> list[tuple]:
    highs  = [b["high"]  for b in ohlc]
    lows   = [b["low"]   for b in ohlc]
    closes = [b["close"] for b in ohlc]

    sk, sd    = stoch_series(highs, lows, closes, k_period, d_period)
    ema       = ema_series(closes, ema_period)
    signals   = []
    start     = max(k_period + 2 * d_period, ema_period + slope_bars) + 2

    for i in range(start, len(closes)):
        if any(x is None for x in [sk[i], sd[i], sk[i-1], sd[i-1], ema[i], ema[i-slope_bars]]):
            continue
        trend_up   = ema[i] > ema[i - slope_bars]
        trend_down = ema[i] < ema[i - slope_bars]

        # %K crosses above %D while below os_ threshold → BUY
        if sk[i-1] < sd[i-1] and sk[i] >= sd[i] and sk[i] < ob and trend_up:
            signals.append((i, +1))
        # %K crosses below %D while above ob threshold → SELL
        elif sk[i-1] > sd[i-1] and sk[i] <= sd[i] and sk[i] > os_ and trend_down:
            signals.append((i, -1))

    return signals


# ═══════════════════════════════════════════════════════════════
# Strategy 3 — Session Open Breakout
# ═══════════════════════════════════════════════════════════════

def sig_session_breakout(ohlc: list[dict], asian_start_h=0, asian_end_h=7,
                          london_open_h=7, london_window_h=2,
                          ny_open_h=13, ny_window_h=2,
                          lookback_bars=84) -> list[tuple]:
    """
    Asian session range = high/low over asian_start_h to asian_end_h UTC.
    Signal when price breaks above/below range during London or NY open window.
    Epochs are in UTC seconds.
    lookback_bars = bars used to build the Asian range (84 × 5min = 7h).
    """
    signals = []
    n = len(ohlc)

    for i in range(lookback_bars + 1, n):
        epoch = ohlc[i]["epoch"]
        h_utc = (epoch % 86400) // 3600

        # Only fire during London open window or NY open window
        in_london = london_open_h <= h_utc < london_open_h + london_window_h
        in_ny     = ny_open_h     <= h_utc < ny_open_h     + ny_window_h
        if not (in_london or in_ny):
            continue

        # Build Asian range from preceding bars (last ~84 bars = 7h of 5min)
        asian_bars = [
            b for b in ohlc[max(0, i - lookback_bars): i]
            if asian_start_h <= ((b["epoch"] % 86400) // 3600) < asian_end_h
        ]
        if len(asian_bars) < 10:
            continue

        range_high = max(b["high"] for b in asian_bars)
        range_low  = min(b["low"]  for b in asian_bars)
        span       = range_high - range_low
        if span <= 0:
            continue

        price = ohlc[i]["close"]
        # Break above range by at least 10% of span
        if price > range_high + 0.10 * span:
            signals.append((i, +1))
        elif price < range_low - 0.10 * span:
            signals.append((i, -1))

    return signals


# ═══════════════════════════════════════════════════════════════
# Strategy 4 — RSI Divergence
# ═══════════════════════════════════════════════════════════════

def sig_rsi_divergence(closes: list[float], rsi_period=14,
                        lookback=10, os_=40.0, ob=60.0) -> list[tuple]:
    """
    Bullish divergence: price lower low + RSI higher low (in OS zone).
    Bearish divergence: price higher high + RSI lower high (in OB zone).
    """
    rsi     = rsi_series(closes, rsi_period)
    signals = []
    start   = rsi_period + lookback + 2

    for i in range(start, len(closes)):
        if rsi[i] is None:
            continue

        # Look back for a swing low within lookback bars
        window_prices = closes[i - lookback: i]
        window_rsi    = [rsi[j] for j in range(i - lookback, i) if rsi[j] is not None]
        if not window_rsi:
            continue

        prev_price_low  = min(window_prices)
        prev_rsi_low    = min(window_rsi)
        prev_price_high = max(window_prices)
        prev_rsi_high   = max(window_rsi)

        # Bullish divergence: current price < prior low AND current RSI > prior RSI low
        if (closes[i] < prev_price_low and
                rsi[i] is not None and rsi[i] > prev_rsi_low and
                rsi[i] < os_):
            signals.append((i, +1))

        # Bearish divergence: current price > prior high AND current RSI < prior RSI high
        elif (closes[i] > prev_price_high and
              rsi[i] is not None and rsi[i] < prev_rsi_high and
              rsi[i] > ob):
            signals.append((i, -1))

    return signals


# ═══════════════════════════════════════════════════════════════
# Strategy 5 — Pivot Point Bounce
# ═══════════════════════════════════════════════════════════════

def sig_pivot_bounce(ohlc_5m: list[dict], ohlc_1d: list[dict],
                     rsi_period=14, rsi_os=40.0, tol_pct=0.001) -> list[tuple]:
    """
    Calculate daily pivot from prior day OHLC.
    Signal when 5min price is within tol_pct of a pivot level + RSI confirms.
    """
    closes_5m = [b["close"] for b in ohlc_5m]
    rsi       = rsi_series(closes_5m, rsi_period)
    signals   = []

    if len(ohlc_1d) < 2:
        return signals

    # Build a map: epoch_day_start → pivot levels
    pivots_by_day: dict[int, dict] = {}
    for bar in ohlc_1d:
        h, l, c = bar["high"], bar["low"], bar["close"]
        p  = (h + l + c) / 3
        r1 = 2 * p - l
        s1 = 2 * p - h
        r2 = p + (h - l)
        s2 = p - (h - l)
        day_key = (bar["epoch"] // 86400) * 86400
        pivots_by_day[day_key + 86400] = {"P": p, "R1": r1, "S1": s1, "R2": r2, "S2": s2}

    start = rsi_period + 2
    for i in range(start, len(ohlc_5m)):
        if rsi[i] is None:
            continue
        epoch    = ohlc_5m[i]["epoch"]
        day_key  = (epoch // 86400) * 86400
        pivots   = pivots_by_day.get(day_key)
        if not pivots:
            continue

        price = closes_5m[i]
        rsi_v = rsi[i]
        ob    = 100 - rsi_os

        for level_name, level in pivots.items():
            tol = level * tol_pct
            near = abs(price - level) <= tol
            if not near:
                continue
            # At support (S1, S2, P when below): RSI oversold → BUY
            if level_name in ("S1", "S2") and rsi_v < rsi_os:
                signals.append((i, +1))
                break
            # At resistance (R1, R2): RSI overbought → SELL
            elif level_name in ("R1", "R2") and rsi_v > ob:
                signals.append((i, -1))
                break
            # At pivot P: follow RSI direction
            elif level_name == "P":
                if rsi_v < rsi_os:
                    signals.append((i, +1))
                    break
                elif rsi_v > ob:
                    signals.append((i, -1))
                    break

    return signals


# ═══════════════════════════════════════════════════════════════
# Walk-forward engine
# ═══════════════════════════════════════════════════════════════

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


def walk_forward_closes(closes, signal_fn, hold_bars, payout):
    fs  = len(closes) // FOLDS
    out = []
    for f in range(FOLDS):
        s    = f * fs
        e    = s + fs if f < FOLDS - 1 else len(closes)
        fold = closes[s:e]
        out.append(sim_binary(fold, signal_fn(fold), hold_bars, payout))
    return out


def walk_forward_ohlc(ohlc, signal_fn, hold_bars, payout):
    closes = [b["close"] for b in ohlc]
    fs     = len(ohlc) // FOLDS
    out    = []
    for f in range(FOLDS):
        s    = f * fs
        e    = s + fs if f < FOLDS - 1 else len(ohlc)
        fold_ohlc   = ohlc[s:e]
        fold_closes = closes[s:e]
        sigs = signal_fn(fold_ohlc)
        out.append(sim_binary(fold_closes, sigs, hold_bars, payout))
    return out


def classify(folds):
    valid = [r for r in folds if r is not None]
    if len(valid) < FOLDS:
        return "SKIP"
    passes = sum(1 for r in valid if r[2] > 0.05)
    if passes == FOLDS:     return "STRONG"
    if passes >= FOLDS - 1: return "WEAK"
    return "FAIL"


def record_result(results, label, hold_m, folds, payout, trading_days):
    status = classify(folds)
    if status not in ("STRONG", "WEAK"):
        return
    valid = [r for r in folds if r]
    results.append({
        "signal":   label,
        "hold_min": hold_m,
        "folds":    folds,
        "mean_ev":  sum(r[2] for r in valid) / len(valid),
        "n_trades": sum(r[1] for r in valid),
        "per_day":  sum(r[1] for r in valid) / trading_days,
        "status":   status,
        "payout":   payout,
    })


# ═══════════════════════════════════════════════════════════════
# Per-symbol sweep
# ═══════════════════════════════════════════════════════════════

async def sweep_symbol(sym: str, cfg: dict) -> list[dict]:
    payout      = cfg["payout"]
    min_hold_m  = cfg["min_hold_m"]
    label       = cfg["label"]
    results     = []

    # Load 5-min candles
    try:
        closes_5m = await fetch_closes(sym, GRAN_5M)
        ohlc_5m   = await fetch_ohlc(sym, GRAN_5M)
    except Exception as e:
        print(f"  SKIP 5m data — {e}")
        return results

    # Load 1h candles (for MTF and pivot)
    try:
        closes_1h = await fetch_closes(sym, GRAN_1H)
        ohlc_1h   = await fetch_ohlc(sym, GRAN_1H)
    except Exception as e:
        print(f"  SKIP 1h data — {e}")
        closes_1h = []
        ohlc_1h   = []

    # Load daily candles (for pivot points)
    try:
        ohlc_1d = await fetch_ohlc(sym, 86400, count=500)
    except Exception as e:
        print(f"  SKIP 1d data — {e}")
        ohlc_1d = []

    # Trading days estimate
    days_5m = len(closes_5m) * GRAN_5M / (22 * 3600)

    # Hold bar options — try min and 2× min
    hold_options_5m = [(min_hold_m // 5, min_hold_m),
                       (min_hold_m // 5 * 2, min_hold_m * 2)]
    if min_hold_m == 5:
        hold_options_5m = [(1, 5), (2, 10), (3, 15)]

    def run(strat_name):
        return (FILTER_STRAT is None) or (strat_name in FILTER_STRAT)

    # ── Strategy 1: Multi-TF Trend Pullback ──────────────────────
    if run("mtf") and closes_1h:
        print(f"    [MTF] testing...")
        for ema_p in [100, 200]:
            for rsi_e in [35, 40, 45]:
                for slope_b in [2, 3, 5]:
                    for hb, hm in hold_options_5m:
                        folds = walk_forward_closes(
                            closes_5m,
                            lambda c, ep=ema_p, re=rsi_e, sb=slope_b:
                                sig_mtf(c, closes_1h, ema_period=ep,
                                        slope_bars=sb, rsi_entry=re),
                            hb, payout
                        )
                        record_result(results,
                            f"MTF EMA{ema_p} RSI<{rsi_e} s{slope_b}",
                            hm, folds, payout, days_5m)

    # ── Strategy 2: Stochastic + EMA ─────────────────────────────
    if run("stoch"):
        print(f"    [STOCH] testing...")
        for kp, dp in [(5, 3), (9, 3), (14, 3)]:
            for ob_lvl in [75, 80]:
                for ema_p in [20, 50]:
                    for slope_b in [3, 5]:
                        for hb, hm in hold_options_5m:
                            folds = walk_forward_ohlc(
                                ohlc_5m,
                                lambda o, k=kp, d=dp, ob=ob_lvl, ep=ema_p, sb=slope_b:
                                    sig_stoch(o, k_period=k, d_period=d,
                                              ob=ob, os_=100-ob,
                                              ema_period=ep, slope_bars=sb),
                                hb, payout
                            )
                            record_result(results,
                                f"STOCH({kp},{dp}) OB={ob_lvl} EMA{ema_p}/s{slope_b}",
                                hm, folds, payout, days_5m)

    # ── Strategy 3: Session Open Breakout ────────────────────────
    if run("session"):
        print(f"    [SESSION] testing...")
        for hb, hm in hold_options_5m:
            folds = walk_forward_ohlc(
                ohlc_5m,
                lambda o: sig_session_breakout(o),
                hb, payout
            )
            record_result(results, "SessionBreakout LDN+NY",
                          hm, folds, payout, days_5m)

    # ── Strategy 4: RSI Divergence ────────────────────────────────
    if run("div"):
        print(f"    [DIV] testing...")
        for rsi_p in [10, 14]:
            for lb in [8, 12, 16]:
                for os_lvl in [35, 40, 45]:
                    for hb, hm in hold_options_5m:
                        folds = walk_forward_closes(
                            closes_5m,
                            lambda c, rp=rsi_p, l=lb, os=os_lvl:
                                sig_rsi_divergence(c, rsi_period=rp,
                                                   lookback=l, os_=os, ob=100-os),
                            hb, payout
                        )
                        record_result(results,
                            f"RSIDivergence RSI{rsi_p} lb={lb} OS={os_lvl}",
                            hm, folds, payout, days_5m)

    # ── Strategy 5: Pivot Point Bounce ───────────────────────────
    if run("pivot") and ohlc_1d:
        print(f"    [PIVOT] testing...")
        for rsi_os in [35, 40, 45]:
            for tol in [0.0005, 0.001, 0.002]:
                for hb, hm in hold_options_5m:
                    folds = walk_forward_closes(
                        closes_5m,
                        lambda c, ros=rsi_os, t=tol:
                            sig_pivot_bounce(
                                [{"close": v, "epoch": i * GRAN_5M}
                                 for i, v in enumerate(c)],
                                ohlc_1d, rsi_os=ros, tol_pct=t
                            ),
                        hb, payout
                    )
                    record_result(results,
                        f"PivotBounce RSI_OS={rsi_os} tol={tol}",
                        hm, folds, payout, days_5m)

    return results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

async def main():
    print("=" * 80)
    print("PRO STRATEGY RESEARCH — Institutional-grade signals on real markets")
    print("=" * 80)
    print("Strategies: MTF Trend Pullback | Stochastic+EMA | Session Breakout |"
          " RSI Divergence | Pivot Bounce")
    print()

    all_results: list[dict] = []

    for sym, cfg in SYMBOL_CONFIG.items():
        if FILTER_SYM and sym != FILTER_SYM:
            continue

        be = round(100 / (1 + cfg["payout"]), 1)
        print(f"\n{'─'*70}")
        print(f"[{sym}] {cfg['label']}  payout={cfg['payout']*100:.0f}%  "
              f"BE WR={be}%  min_hold={cfg['min_hold_m']}min")
        print(f"{'─'*70}")

        results = await sweep_symbol(sym, cfg)
        for r in results:
            r["symbol"] = sym
            r["label"]  = cfg["label"]
        all_results.extend(results)

        strong = sum(1 for r in results if r["status"] == "STRONG")
        weak   = sum(1 for r in results if r["status"] == "WEAK")
        print(f"  → {strong} STRONG  {weak} WEAK")

    # ── Report ──────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("RESULTS — STRONG first, sorted by MeanEV")
    print(f"{'='*80}")

    strong_all = sorted(
        [r for r in all_results if r["status"] == "STRONG"],
        key=lambda x: -x["mean_ev"],
    )
    weak_top = sorted(
        [r for r in all_results if r["status"] == "WEAK"],
        key=lambda x: -x["mean_ev"],
    )[:20]

    for group, title in [(strong_all, "STRONG — 3/3 folds EV>0.05"),
                         (weak_top,   "WEAK   — 2/3 folds (top 20)")]:
        if not group:
            continue
        print(f"\n── {title} ──")
        prev = None
        for r in group:
            if r["symbol"] != prev:
                be = round(100 / (1 + r["payout"]), 1)
                print(f"\n  {r['label']} ({r['symbol']})  "
                      f"payout={r['payout']*100:.0f}%  BE={be}%")
                prev = r["symbol"]
            valid = [f for f in r["folds"] if f]
            wrs   = " / ".join(f"{f[0]*100:.1f}%" for f in valid)
            print(f"    {r['signal']:45s} hold={r['hold_min']:3d}min "
                  f"| [{wrs}] "
                  f"| MeanEV {r['mean_ev']:+.4f} "
                  f"| ~{r['per_day']:.1f}/day")

    total_s = len(strong_all)
    total_w = len([r for r in all_results if r["status"] == "WEAK"])
    print(f"\n✓ {total_s} STRONG  {total_w} WEAK across all symbols and strategies")

    if strong_all:
        top = strong_all[0]
        print(f"\nTop pick: {top['label']} — {top['signal']} "
              f"hold={top['hold_min']}min "
              f"| MeanEV {top['mean_ev']:+.4f} "
              f"| ~{top['per_day']:.1f}/day "
              f"| payout {top['payout']*100:.0f}%")


if __name__ == "__main__":
    asyncio.run(main())
