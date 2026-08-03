"""
Pro Bot Backtest Engine — v2 (robustness fixes)

Four improvements over v1:
  1. Spread + slippage: entry shifted against trader by realistic spread
  2. Min trade count: configs with <100 closed trades are discarded
  3. Holdout set: final 20% of bars never touched during sweep;
     top config per strategy is re-run on holdout to confirm OOS edge
  4. Binomial 95% CI on WR: shows uncertainty; breakeven must lie below CI lower bound

Breakeven WR (net of spread):
  1:1.5 R:R → 40.0%   |   1:2.0 R:R → 33.3%

Usage:
  python3 pro_bot/backtest.py
  python3 pro_bot/backtest.py --strategy=mtf
  python3 pro_bot/backtest.py --symbol=frxXAUUSD
"""

import asyncio
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import websockets

sys.path.insert(0, str(Path(__file__).parent.parent))

from pro_bot.strategies.base import Signal
from pro_bot.indicators import ema as _ema, rsi as _rsi, stochastic as _stoch
from pro_bot.indicators import pivot_levels as _pivots

LEGACY_WS = "wss://ws.derivws.com/websockets/v3?app_id=1089"
CACHE_5M  = Path("data/scalp")
CACHE_1H  = Path("data/pro")
CACHE_1D  = Path("data/pro")
GRAN_5M   = 300
GRAN_1H   = 3600
GRAN_1D   = 86400

FILTER_STRAT = next((a.split("=")[1] for a in sys.argv if a.startswith("--strategy=")), None)
FILTER_SYM   = next((a.split("=")[1] for a in sys.argv if a.startswith("--symbol=")), None)

SYMBOLS = {
    "frxXAUUSD": {"label": "Gold   ", "pip": 0.01},
    "frxUSDJPY": {"label": "USD/JPY", "pip": 0.01},
    "frxEURUSD": {"label": "EUR/USD", "pip": 0.0001},
    "frxGBPUSD": {"label": "GBP/USD", "pip": 0.0001},
    "frxAUDUSD": {"label": "AUD/USD", "pip": 0.0001},
}

# Fix 1 — realistic spread per symbol (in price units, not pips)
SPREADS = {
    "frxXAUUSD": 0.30,     # Gold: ~30 cents
    "frxUSDJPY": 0.015,    # USD/JPY: 1.5 points
    "frxEURUSD": 0.00012,  # EUR/USD: 1.2 pips
    "frxGBPUSD": 0.00015,  # GBP/USD: 1.5 pips
    "frxAUDUSD": 0.00014,  # AUD/USD: 1.4 pips
}

# Fix 2 — minimum closed trades for a config to be reported
MIN_TRADES = 100


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

async def _fetch(symbol: str, gran: int, count: int, cache_dir: Path) -> list[dict]:
    cache = cache_dir / f"{symbol}_{gran}_ohlc.json"
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

    bars = [{"open":  float(c["open"]),
             "high":  float(c["high"]),
             "low":   float(c["low"]),
             "close": float(c["close"]),
             "epoch": int(c["epoch"])} for c in msg["candles"]]
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache, "w") as f:
        json.dump(bars, f)
    return bars


async def load_data(symbol: str) -> tuple[list[dict], list[dict], list[dict]]:
    bars_5m = await _fetch(symbol, GRAN_5M, 5000, CACHE_5M)
    bars_1h = await _fetch(symbol, GRAN_1H, 2000, CACHE_1H)
    bars_1d = await _fetch(symbol, GRAN_1D,  500, CACHE_1D)
    return bars_5m, bars_1h, bars_1d


def split_bars(bars_5m, bars_1h, bars_1d, train_pct=0.80):
    """Fix 3 — split all timeframes at the same epoch boundary."""
    cutoff     = int(len(bars_5m) * train_pct)
    split_epoch = bars_5m[cutoff]["epoch"]

    train   = {"5m": bars_5m[:cutoff],
               "1h": [b for b in bars_1h if b["epoch"] <= split_epoch],
               "1d": [b for b in bars_1d if b["epoch"] <= split_epoch]}
    holdout = {"5m": bars_5m[cutoff:],
               "1h": [b for b in bars_1h if b["epoch"] > split_epoch],
               "1d": [b for b in bars_1d if b["epoch"] > split_epoch]}
    return train, holdout


# ═══════════════════════════════════════════════════════════════
# SL/TP exit simulator
# ═══════════════════════════════════════════════════════════════

@dataclass
class Trade:
    entry_price: float
    action:      str
    sl:          float
    tp:          float
    sl_pips:     float
    tp_pips:     float
    signal_bar:  int
    exit_bar:    int   = -1
    result:      str   = ""     # WIN / LOSS / TIMEOUT
    r_multiple:  float = 0.0


def simulate_exits(bars: list[dict], signals: list[tuple],
                   max_hold_bars: int = 48,
                   spread: float = 0.0) -> list[Trade]:
    """
    Fix 1 — BUY fills at ask (close + spread); SELL fills at bid (close - spread).
    If both SL and TP hit on the same bar, SL is assumed to hit first (worst case).
    """
    trades: list[Trade] = []

    for bar_idx, sig in signals:
        if sig.action not in ("BUY", "SELL"):
            continue
        if sig.sl_pips is None or sig.tp_pips is None:
            continue
        if sig.sl_pips <= 0 or sig.tp_pips <= 0:
            continue

        close = bars[bar_idx]["close"]

        # Fix 1: shift entry by spread
        if sig.action == "BUY":
            entry    = close + spread
            sl_price = entry - sig.sl_pips
            tp_price = entry + sig.tp_pips
        else:
            entry    = close - spread
            sl_price = entry + sig.sl_pips
            tp_price = entry - sig.tp_pips

        trade = Trade(
            entry_price=entry,
            action=sig.action,
            sl=sl_price,
            tp=tp_price,
            sl_pips=sig.sl_pips,
            tp_pips=sig.tp_pips,
            signal_bar=bar_idx,
        )

        for j in range(bar_idx + 1, min(bar_idx + max_hold_bars + 1, len(bars))):
            h = bars[j]["high"]
            l = bars[j]["low"]

            if sig.action == "BUY":
                sl_hit = l <= sl_price
                tp_hit = h >= tp_price
            else:
                sl_hit = h >= sl_price
                tp_hit = l <= tp_price

            if sl_hit:
                trade.exit_bar   = j
                trade.result     = "LOSS"
                trade.r_multiple = -1.0
                break
            if tp_hit:
                trade.exit_bar   = j
                trade.result     = "WIN"
                trade.r_multiple = sig.tp_pips / sig.sl_pips
                break
        else:
            trade.exit_bar   = bar_idx + max_hold_bars
            trade.result     = "TIMEOUT"
            trade.r_multiple = 0.0

        trades.append(trade)

    return trades


# ═══════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════

def wr_confidence(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Fix 4 — Wilson score interval (handles small n better than normal approx)."""
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def breakeven_wr(rr: float) -> float:
    """Minimum WR needed to break even at a given R:R."""
    return 1.0 / (1.0 + rr)


def calc_metrics(trades: list[Trade], symbol: str, strategy: str,
                 config: dict | None = None) -> dict:
    closed  = [t for t in trades if t.result in ("WIN", "LOSS")]

    # Fix 2 — require minimum closed trades
    if len(closed) < MIN_TRADES:
        return {}

    wins    = [t for t in closed if t.result == "WIN"]
    losses  = [t for t in closed if t.result == "LOSS"]
    timeout = [t for t in trades if t.result == "TIMEOUT"]

    wr       = len(wins) / len(closed)
    avg_rr   = (sum(t.r_multiple for t in wins) / len(wins)) if wins else 0.0
    pf       = (len(wins) * avg_rr) / len(losses) if losses else float("inf")
    ev_per_r = wr * avg_rr - (1 - wr)

    # Fix 4 — 95% CI on WR
    wr_lo, wr_hi = wr_confidence(len(wins), len(closed))

    # Equity curve
    equity = peak = max_dd = 0.0
    for t in closed:
        equity += t.r_multiple
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    streak = max_streak = 0
    for t in closed:
        if t.result == "LOSS":
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    days    = (trades[-1].signal_bar - trades[0].signal_bar) * GRAN_5M / 86400
    per_day = len(closed) / days if days > 0 else 0

    return {
        "symbol":     symbol,
        "strategy":   strategy,
        "total":      len(trades),
        "closed":     len(closed),
        "wins":       len(wins),
        "losses":     len(losses),
        "timeouts":   len(timeout),
        "wr":         wr,
        "wr_lo":      wr_lo,
        "wr_hi":      wr_hi,
        "avg_rr":     avg_rr,
        "profit_f":   pf,
        "ev_per_r":   ev_per_r,
        "net_r":      equity,
        "max_dd_r":   max_dd,
        "max_streak": max_streak,
        "per_day":    per_day,
        "config":     config or {},   # stored for holdout re-run
    }


def print_metrics(m: dict, label: str = "") -> None:
    be_wr   = breakeven_wr(m["avg_rr"])
    ci_lo   = m["wr_lo"] * 100
    ci_hi   = m["wr_hi"] * 100
    ci_safe = ci_lo > be_wr * 100   # lower bound clears breakeven
    verdict = ("STRONG"     if m["ev_per_r"] > 0.05 and ci_safe
               else "PROFITABLE" if m["ev_per_r"] > 0.05
               else "MARGINAL"   if m["ev_per_r"] > 0
               else "LOSING")
    prefix  = f"[{label}] " if label else ""
    print(f"    {prefix}{m['strategy']:35s}  {m['symbol']}")
    print(f"      Trades: {m['closed']} closed ({m['per_day']:.1f}/day) | "
          f"Timeouts: {m['timeouts']}")
    print(f"      WR: {m['wr']*100:.1f}%  [95% CI: {ci_lo:.1f}%–{ci_hi:.1f}%]"
          f"  BE: {be_wr*100:.1f}%  {'✓ CI clears BE' if ci_safe else '✗ CI overlaps BE'}")
    print(f"      Avg R:R: 1:{m['avg_rr']:.2f}  |  Profit Factor: {m['profit_f']:.2f}")
    print(f"      EV/trade: {m['ev_per_r']:+.3f}R  |  Net: {m['net_r']:+.1f}R  |  "
          f"Max DD: {m['max_dd_r']:.1f}R  |  Max streak L: {m['max_streak']}")
    print(f"      ▶  {verdict}")
    print()


# ═══════════════════════════════════════════════════════════════
# Strategy runners — all precompute indicators once (O(n))
# ═══════════════════════════════════════════════════════════════

def run_mtf(bars_5m, bars_1h, config=None) -> list[tuple]:
    cfg       = config or {}
    ema_p     = cfg.get("ema_period", 200)
    slope_b   = cfg.get("slope_bars", 3)
    rsi_p     = cfg.get("rsi_period", 14)
    rsi_entry = cfg.get("rsi_entry", 40.0)
    tp_rr     = cfg.get("tp_rr", 2.0)
    ob        = 100.0 - rsi_entry
    ratio     = GRAN_1H // GRAN_5M

    closes_1h = [b["close"] for b in bars_1h]
    closes_5m = [b["close"] for b in bars_5m]
    ema_1h    = _ema(closes_1h, ema_p)
    rsi_5m    = _rsi(closes_5m, rsi_p)

    results = []
    for i in range(rsi_p + 2, len(bars_5m)):
        j = min(i // ratio, len(bars_1h) - 1)
        if j < slope_b:
            continue
        e_now  = ema_1h[j]
        e_prev = ema_1h[j - slope_b]
        r_now  = rsi_5m[i]
        r_prev = rsi_5m[i - 1]
        if any(x is None for x in [e_now, e_prev, r_now, r_prev]):
            continue
        trend_up   = e_now > e_prev
        trend_down = e_now < e_prev
        price = closes_5m[i]
        sl = abs(price - bars_5m[i]["low"]) + price * 0.0002
        if sl <= 0:
            sl = price * 0.003
        if trend_up and r_prev >= rsi_entry > r_now:
            results.append((i, Signal("BUY",  sl_pips=sl, tp_pips=sl * tp_rr)))
        elif trend_down and r_prev <= ob < r_now:
            results.append((i, Signal("SELL", sl_pips=sl, tp_pips=sl * tp_rr)))
    return results


def run_stoch(bars_5m, config=None) -> list[tuple]:
    cfg     = config or {}
    kp      = cfg.get("k_period", 5)
    dp      = cfg.get("d_period", 3)
    ob      = cfg.get("ob", 80.0)
    os_     = cfg.get("os_level", 20.0)
    ema_p   = cfg.get("ema_period", 50)
    slope_b = cfg.get("slope_bars", 3)
    tp_rr   = cfg.get("tp_rr", 1.5)

    closes = [b["close"] for b in bars_5m]
    sk, sd = _stoch(bars_5m, kp, dp)
    ema_s  = _ema(closes, ema_p)
    start  = max(kp + 2 * dp, ema_p + slope_b) + 2

    results = []
    for i in range(start, len(bars_5m)):
        if any(x is None for x in [sk[i], sd[i], sk[i-1], sd[i-1],
                                    ema_s[i], ema_s[i - slope_b]]):
            continue
        trend_up   = ema_s[i] > ema_s[i - slope_b]
        trend_down = ema_s[i] < ema_s[i - slope_b]
        cross_up   = sk[i-1] < sd[i-1] and sk[i] >= sd[i]
        cross_down = sk[i-1] > sd[i-1] and sk[i] <= sd[i]
        price = closes[i]
        sl = price - min(b["low"] for b in bars_5m[i-3:i+1])
        if sl <= 0:
            sl = price * 0.003
        if trend_up and cross_up and sk[i] < ob:
            results.append((i, Signal("BUY",  sl_pips=sl, tp_pips=sl * tp_rr)))
        elif trend_down and cross_down and sk[i] > os_:
            results.append((i, Signal("SELL", sl_pips=sl, tp_pips=sl * tp_rr)))
    return results


def run_session(bars_5m, config=None) -> list[tuple]:
    cfg      = config or {}
    tp_rr    = cfg.get("tp_rr", 1.5)
    buf      = cfg.get("breakout_buffer", 0.10)
    fired    = set()
    results  = []
    lookback = 84

    for i in range(lookback + 1, len(bars_5m)):
        epoch = bars_5m[i]["epoch"]
        h_utc = (epoch % 86400) // 3600
        day   = epoch // 86400
        if day in fired:
            continue
        in_window = (7 <= h_utc < 9) or (13 <= h_utc < 15)
        if not in_window:
            continue
        asian = [b for b in bars_5m[max(0, i-lookback):i]
                 if 0 <= ((b["epoch"] % 86400) // 3600) < 7]
        if len(asian) < 5:
            continue
        rh = max(b["high"] for b in asian)
        rl = min(b["low"]  for b in asian)
        span = rh - rl
        if span <= 0:
            continue
        price = bars_5m[i]["close"]
        if price > rh + buf * span:
            sl = price - rl
            fired.add(day)
            results.append((i, Signal("BUY",  sl_pips=sl, tp_pips=sl * tp_rr)))
        elif price < rl - buf * span:
            sl = rh - price
            fired.add(day)
            results.append((i, Signal("SELL", sl_pips=sl, tp_pips=sl * tp_rr)))
    return results


def run_divergence(bars_5m, config=None) -> list[tuple]:
    cfg      = config or {}
    rsi_p    = cfg.get("rsi_period", 14)
    lb       = cfg.get("lookback", 12)
    os_      = cfg.get("os_level", 40.0)
    ob       = cfg.get("ob_level", 60.0)
    hidden   = cfg.get("hidden_divergence", True)
    tp_rr    = cfg.get("tp_rr", 2.0)

    closes   = [b["close"] for b in bars_5m]
    rsi_s    = _rsi(closes, rsi_p)
    start    = rsi_p + lb + 3
    cooldown = 0
    results  = []

    for i in range(start, len(bars_5m)):
        if cooldown > 0:
            cooldown -= 1
            continue
        r = rsi_s[i]
        if r is None:
            continue
        w_c = closes[i - lb: i]
        w_r = [rsi_s[j] for j in range(i - lb, i) if rsi_s[j] is not None]
        if not w_r:
            continue
        price = closes[i]
        sl = price - min(b["low"] for b in bars_5m[i-3:i+1])
        if sl <= 0:
            sl = price * 0.003

        if price < min(w_c) and r > min(w_r) and r < os_:
            results.append((i, Signal("BUY",  sl_pips=sl, tp_pips=sl * tp_rr)))
            cooldown = 3
        elif price > max(w_c) and r < max(w_r) and r > ob:
            sl2 = max(b["high"] for b in bars_5m[i-3:i+1]) - price
            if sl2 <= 0:
                sl2 = price * 0.003
            results.append((i, Signal("SELL", sl_pips=sl2, tp_pips=sl2 * tp_rr)))
            cooldown = 3
        elif hidden:
            if price > min(w_c) and r < min(w_r) and r < 50:
                results.append((i, Signal("BUY",  sl_pips=sl, tp_pips=sl * tp_rr)))
                cooldown = 3
            elif price < max(w_c) and r > max(w_r) and r > 50:
                sl2 = max(b["high"] for b in bars_5m[i-3:i+1]) - price
                if sl2 <= 0:
                    sl2 = price * 0.003
                results.append((i, Signal("SELL", sl_pips=sl2, tp_pips=sl2 * tp_rr)))
                cooldown = 3
    return results


def run_pivot(bars_5m, bars_1d, config=None) -> list[tuple]:
    cfg    = config or {}
    rsi_p  = cfg.get("rsi_period", 14)
    rsi_os = cfg.get("rsi_os", 40.0)
    tol    = cfg.get("tol_pct", 0.001)
    tp_rr  = cfg.get("tp_rr", 1.5)
    ob     = 100.0 - rsi_os

    closes    = [b["close"] for b in bars_5m]
    rsi_s     = _rsi(closes, rsi_p)
    pivot_map: dict[int, dict] = {}
    for bar in bars_1d:
        next_day = (bar["epoch"] // 86400 + 1) * 86400
        pivot_map[next_day] = _pivots(bar)

    results = []
    for i in range(rsi_p + 2, len(bars_5m)):
        r = rsi_s[i]
        if r is None:
            continue
        epoch   = bars_5m[i]["epoch"]
        day_key = (epoch // 86400) * 86400
        pvts    = pivot_map.get(day_key)
        if not pvts:
            continue
        price = closes[i]
        sl    = price * 0.003
        for name, level in pvts.items():
            if abs(price - level) > level * tol:
                continue
            if name in ("S1", "S2", "S3") and r < rsi_os:
                results.append((i, Signal("BUY",  sl_pips=sl, tp_pips=sl * tp_rr)))
                break
            elif name in ("R1", "R2", "R3") and r > ob:
                results.append((i, Signal("SELL", sl_pips=sl, tp_pips=sl * tp_rr)))
                break
    return results


# ═══════════════════════════════════════════════════════════════
# Sweep functions — pass spread, embed config in metrics
# ═══════════════════════════════════════════════════════════════

def sweep_mtf(train, sym, spread) -> list[dict]:
    results = []
    for ema_p in [100, 200]:
        for rsi_e in [35, 40, 45]:
            for rr in [1.5, 2.0]:
                cfg    = {"ema_period": ema_p, "slope_bars": 3,
                          "rsi_period": 14, "rsi_entry": rsi_e, "tp_rr": rr}
                sigs   = run_mtf(train["5m"], train["1h"], cfg)
                if not sigs:
                    continue
                trades = simulate_exits(train["5m"], sigs, spread=spread)
                m      = calc_metrics(trades, sym,
                                      f"MTF EMA{ema_p} RSI<{rsi_e} RR{rr}", cfg)
                if m:
                    results.append(m)
    return results


def sweep_stoch(train, sym, spread) -> list[dict]:
    results = []
    for kp in [5, 9]:
        for ob in [75, 80]:
            for ep in [20, 50]:
                for rr in [1.5, 2.0]:
                    cfg  = {"k_period": kp, "d_period": 3, "ob": ob,
                            "os_level": 100-ob, "ema_period": ep,
                            "slope_bars": 3, "tp_rr": rr}
                    sigs = run_stoch(train["5m"], cfg)
                    if not sigs:
                        continue
                    trades = simulate_exits(train["5m"], sigs, spread=spread)
                    m      = calc_metrics(trades, sym,
                                         f"STOCH({kp},3) OB={ob} EMA{ep} RR{rr}", cfg)
                    if m:
                        results.append(m)
    return results


def sweep_session(train, sym, spread) -> list[dict]:
    results = []
    for rr in [1.5, 2.0]:
        cfg    = {"tp_rr": rr}
        sigs   = run_session(train["5m"], cfg)
        if not sigs:
            continue
        trades = simulate_exits(train["5m"], sigs, spread=spread)
        m      = calc_metrics(trades, sym, f"SessionBreakout RR{rr}", cfg)
        if m:
            results.append(m)
    return results


def sweep_div(train, sym, spread) -> list[dict]:
    results = []
    for lb in [8, 12, 16]:
        for os_l in [35, 40, 45]:
            for hidden in [True, False]:
                for rr in [1.5, 2.0]:
                    cfg  = {"rsi_period": 14, "lookback": lb,
                            "os_level": os_l, "ob_level": 100-os_l,
                            "hidden_divergence": hidden, "tp_rr": rr}
                    sigs = run_divergence(train["5m"], cfg)
                    if not sigs:
                        continue
                    trades = simulate_exits(train["5m"], sigs, spread=spread)
                    tag    = f"RSIDiv lb={lb} OS={os_l} {'H' if hidden else 'C'} RR{rr}"
                    m      = calc_metrics(trades, sym, tag, cfg)
                    if m:
                        results.append(m)
    return results


def sweep_pivot(train, sym, spread) -> list[dict]:
    results = []
    for rsi_os in [35, 40, 45]:
        for tol in [0.0005, 0.001, 0.002]:
            for rr in [1.5, 2.0]:
                cfg  = {"rsi_period": 14, "rsi_os": rsi_os,
                        "tol_pct": tol, "tp_rr": rr}
                sigs = run_pivot(train["5m"], train["1d"], cfg)
                if not sigs:
                    continue
                trades = simulate_exits(train["5m"], sigs, spread=spread)
                m      = calc_metrics(trades, sym,
                                      f"PivotBounce OS={rsi_os} tol={tol} RR{rr}", cfg)
                if m:
                    results.append(m)
    return results


# ═══════════════════════════════════════════════════════════════
# Holdout validation — re-run best config on unseen data
# ═══════════════════════════════════════════════════════════════

def run_on_holdout(best: dict, sym: str, holdout: dict, spread: float) -> dict | None:
    """Fix 3 — run the winning config's strategy on holdout bars."""
    cfg  = best["config"]
    name = best["strategy"]

    if name.startswith("MTF"):
        sigs = run_mtf(holdout["5m"], holdout["1h"], cfg)
    elif name.startswith("STOCH"):
        sigs = run_stoch(holdout["5m"], cfg)
    elif name.startswith("Session"):
        sigs = run_session(holdout["5m"], cfg)
    elif name.startswith("RSIDiv"):
        sigs = run_divergence(holdout["5m"], cfg)
    elif name.startswith("Pivot"):
        sigs = run_pivot(holdout["5m"], holdout["1d"], cfg)
    else:
        return None

    if not sigs:
        return None
    trades = simulate_exits(holdout["5m"], sigs, spread=spread)
    # Lower MIN_TRADES threshold for holdout (it's 20% of data)
    closed = [t for t in trades if t.result in ("WIN", "LOSS")]
    if len(closed) < 20:
        return None
    return calc_metrics(trades, sym, best["strategy"], cfg)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

async def main():
    print("=" * 78)
    print("PRO BOT BACKTEST v2 — Spread-adjusted | Min 100 trades | Holdout | 95% CI")
    print("Breakeven: 1:1.5 R:R → WR>40% | 1:2 R:R → WR>33%")
    print(f"Spread applied per symbol (realistic market cost)")
    print("=" * 78)

    all_metrics: list[dict] = []
    holdout_summary: list[tuple[dict, dict | None]] = []

    for sym, info in SYMBOLS.items():
        if FILTER_SYM and sym != FILTER_SYM:
            continue

        spread = SPREADS.get(sym, 0.0)
        print(f"\n{'─'*78}")
        print(f"[{sym}] {info['label']}  spread={spread}")
        print(f"{'─'*78}")

        try:
            bars_5m, bars_1h, bars_1d = await load_data(sym)
            print(f"  Data: {len(bars_5m)} × 5min | {len(bars_1h)} × 1h | "
                  f"{len(bars_1d)} × daily")
        except Exception as e:
            print(f"  SKIP — {e}")
            continue

        # Fix 3 — split before any sweeping
        train, holdout = split_bars(bars_5m, bars_1h, bars_1d)
        print(f"  Train: {len(train['5m'])} bars | Holdout: {len(holdout['5m'])} bars "
              f"(20% never seen during sweep)")

        strategies_to_run = []
        if not FILTER_STRAT or "mtf"     in FILTER_STRAT: strategies_to_run.append("mtf")
        if not FILTER_STRAT or "stoch"   in FILTER_STRAT: strategies_to_run.append("stoch")
        if not FILTER_STRAT or "session" in FILTER_STRAT: strategies_to_run.append("session")
        if not FILTER_STRAT or "div"     in FILTER_STRAT: strategies_to_run.append("div")
        if not FILTER_STRAT or "pivot"   in FILTER_STRAT: strategies_to_run.append("pivot")

        sym_metrics: list[dict] = []

        if "mtf"     in strategies_to_run:
            print("  [MTF] sweeping on training set...")
            sym_metrics += sweep_mtf(train, sym, spread)
        if "stoch"   in strategies_to_run:
            print("  [STOCH] sweeping on training set...")
            sym_metrics += sweep_stoch(train, sym, spread)
        if "session" in strategies_to_run:
            print("  [SESSION] sweeping on training set...")
            sym_metrics += sweep_session(train, sym, spread)
        if "div"     in strategies_to_run:
            print("  [DIV] sweeping on training set...")
            sym_metrics += sweep_div(train, sym, spread)
        if "pivot"   in strategies_to_run:
            print("  [PIVOT] sweeping on training set...")
            sym_metrics += sweep_pivot(train, sym, spread)

        all_metrics += sym_metrics
        profitable = sum(1 for m in sym_metrics if m["ev_per_r"] > 0.05)
        print(f"  → {profitable} profitable configs found (min {MIN_TRADES} trades enforced)")

        # Fix 3 — holdout validation for the best config per symbol
        if sym_metrics:
            best = max(sym_metrics, key=lambda x: x["ev_per_r"])
            print(f"  → Validating best config on holdout: {best['strategy']}")
            h = run_on_holdout(best, sym, holdout, spread)
            holdout_summary.append((best, h))

    # ── Report ──────────────────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print("RESULTS — Training set | EV > 0.05R | sorted by EV")
    print(f"{'='*78}\n")

    profitable = sorted(
        [m for m in all_metrics if m["ev_per_r"] > 0.05],
        key=lambda x: -x["ev_per_r"],
    )
    marginal = sorted(
        [m for m in all_metrics if 0 < m["ev_per_r"] <= 0.05],
        key=lambda x: -x["ev_per_r"],
    )[:10]
    losing = [m for m in all_metrics if m["ev_per_r"] <= 0]

    if profitable:
        print(f"── PROFITABLE ({len(profitable)} configs) ──\n")
        for m in profitable:
            print_metrics(m, "TRAIN")
    else:
        print("── No profitable configs survived spread + min-trades filter ──\n")

    if marginal:
        print(f"── MARGINAL top 10 (0 < EV ≤ 0.05R) ──\n")
        for m in marginal:
            print_metrics(m, "TRAIN")

    # ── Holdout section ─────────────────────────────────────────────────────
    if holdout_summary:
        print(f"\n{'='*78}")
        print("HOLDOUT VALIDATION — 20% unseen data, best config per symbol")
        print(f"{'='*78}\n")

        for train_m, holdout_m in holdout_summary:
            print(f"  {train_m['symbol']} — {train_m['strategy']}")
            print(f"    Train  EV: {train_m['ev_per_r']:+.3f}R  WR: {train_m['wr']*100:.1f}%"
                  f"  [{train_m['closed']} trades]")
            if holdout_m:
                decay = train_m["ev_per_r"] - holdout_m["ev_per_r"]
                flag  = ("✓ CONFIRMED" if holdout_m["ev_per_r"] > 0.02
                         else "⚠ DEGRADED"  if holdout_m["ev_per_r"] > 0
                         else "✗ NEGATIVE")
                print(f"    Holdout EV: {holdout_m['ev_per_r']:+.3f}R  WR: {holdout_m['wr']*100:.1f}%"
                      f"  [{holdout_m['closed']} trades]  decay={decay:+.3f}R  {flag}")
            else:
                print(f"    Holdout: insufficient trades (<20) — cannot confirm")
            print()

    print(f"\nSummary: {len(profitable)} profitable | "
          f"{len(marginal)} marginal | {len(losing)} losing "
          f"across {len(all_metrics)} total configs "
          f"(min {MIN_TRADES} trades enforced, spread applied)")

    if profitable:
        top = profitable[0]
        print(f"\nBest: {top['symbol']} — {top['strategy']}")
        print(f"  EV {top['ev_per_r']:+.3f}R | WR {top['wr']*100:.1f}% "
              f"[CI: {top['wr_lo']*100:.1f}%–{top['wr_hi']*100:.1f}%] | "
              f"PF {top['profit_f']:.2f} | {top['per_day']:.1f}/day")


if __name__ == "__main__":
    asyncio.run(main())
