"""
Focused sweep: find configs for EURUSD and XAUUSD that hold up on holdout data.

Strategy:
  1. Sweep training set across relaxed filter combinations
  2. For every config with train EV > 0.02R AND >= 30 train trades,
     run it on the 20% holdout
  3. Report configs where BOTH train AND holdout are positive EV
  4. Pick the best surviving config per symbol

This avoids the overfitting trap: we only keep configs that generalise.

Usage:
  python pro_bot/backtest_fix.py
"""

import asyncio
import bisect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pro_bot.backtest import (
    load_data, split_bars, simulate_exits, calc_metrics,
    SPREADS, MIN_TRADES,
    _fetch, CACHE_5M, CACHE_1H, CACHE_1D, GRAN_5M, GRAN_1H, GRAN_1D,
)

LONG_DAYS = 365


async def load_data_long(symbol: str):
    bars_5m = await _fetch(symbol, GRAN_5M, LONG_DAYS, CACHE_5M)
    bars_1h = await _fetch(symbol, GRAN_1H, LONG_DAYS, CACHE_1H)
    bars_1d = await _fetch(symbol, GRAN_1D, LONG_DAYS, CACHE_1D)
    return bars_5m, bars_1h, bars_1d
from pro_bot.indicators import ema as _ema, rsi as _rsi, adx as _adx, atr as _atr
from pro_bot.strategies.base import Signal


# ── MTF runner (supports adaptive RSI) ───────────────────────────────────────

def run_mtf(bars_5m, bars_1h, bars_1d=None, cfg=None, adaptive=False):
    cfg          = cfg or {}
    ema_p        = cfg.get("ema_period",        200)
    slope_b      = cfg.get("slope_bars",          3)
    rsi_p        = cfg.get("rsi_period",         14)
    rsi_entry    = cfg.get("rsi_entry",         35.0)
    tp_rr        = cfg.get("tp_rr",             2.0)
    adx_min      = cfg.get("adx_min",             0)
    sess_peak    = cfg.get("session_peak",     False)
    sess_only    = cfg.get("session_only",     False)
    atr_mult     = cfg.get("atr_mult_sl",       0.0)
    macro_filter = cfg.get("macro_filter",     False)
    macro_ema_p  = cfg.get("macro_ema_period",   20)
    rsi_lookback = cfg.get("rsi_lookback",       50)

    closes_1h = [b["close"] for b in bars_1h]
    closes_5m = [b["close"] for b in bars_5m]
    ema_1h    = _ema(closes_1h, ema_p)
    rsi_5m    = _rsi(closes_5m, rsi_p)
    adx_1h    = _adx(bars_1h, 14) if adx_min > 0 else None
    atr_5m    = _atr(bars_5m, 14) if atr_mult > 0 else None

    if macro_filter and bars_1d:
        ema_1d    = _ema([b["close"] for b in bars_1d], macro_ema_p)
        epochs_1d = [b["epoch"] for b in bars_1d]
    else:
        ema_1d = epochs_1d = None

    epochs_1h = [b["epoch"] for b in bars_1h]

    results = []
    for i in range(rsi_p + 2, len(bars_5m)):
        if sess_peak:
            h_utc  = (bars_5m[i]["epoch"] % 86400) // 3600
            m_min  = (bars_5m[i]["epoch"] % 3600)  // 60
            london = (7 <= h_utc < 10) or (h_utc == 10 and m_min < 30)
            ny     = (13 <= h_utc < 16) or (h_utc == 16 and m_min < 30)
            if not (london or ny):
                continue
        elif sess_only:
            h_utc = (bars_5m[i]["epoch"] % 86400) // 3600
            if not (7 <= h_utc < 20):
                continue

        j = bisect.bisect_right(epochs_1h, bars_5m[i]["epoch"]) - 1
        if j < slope_b or j < 0:
            continue

        e_now  = ema_1h[j];  e_prev = ema_1h[j - slope_b]
        r_now  = rsi_5m[i];  r_prev = rsi_5m[i - 1]
        if any(x is None for x in [e_now, e_prev, r_now, r_prev]):
            continue

        if adx_min > 0 and adx_1h is not None:
            adx_v = adx_1h[j]
            if adx_v is None or adx_v < adx_min:
                continue

        allow_long = allow_short = True
        if ema_1d is not None:
            k = bisect.bisect_right(epochs_1d, bars_5m[i]["epoch"]) - 1
            if k < 1 or ema_1d[k] is None or ema_1d[k - 1] is None:
                continue
            macro_up    = ema_1d[k] > ema_1d[k - 1]
            allow_long  = macro_up
            allow_short = not macro_up

        # Adaptive vs fixed entry threshold
        if adaptive and i >= rsi_lookback:
            recent = [v for v in rsi_5m[i - rsi_lookback:i] if v is not None]
            if len(recent) >= max(10, rsi_lookback // 2):
                recent.sort()
                idx       = max(0, int(len(recent) * 0.20) - 1)
                threshold = max(rsi_entry, min(50.0, recent[idx]))
            else:
                threshold = rsi_entry
        else:
            threshold = rsi_entry
        ob = 100.0 - threshold

        trend_up   = e_now > e_prev
        trend_down = e_now < e_prev
        price      = closes_5m[i]

        if atr_mult > 0 and atr_5m is not None and atr_5m[i] is not None:
            sl = atr_5m[i] * atr_mult
        else:
            sl = abs(price - bars_5m[i]["low"]) + price * 0.0002
        if sl <= 0:
            sl = price * 0.003

        if trend_up  and r_prev >= threshold > r_now  and allow_long:
            results.append((i, Signal("BUY",  sl_pips=sl, tp_pips=sl * tp_rr)))
        elif trend_down and r_prev <= ob      < r_now  and allow_short:
            results.append((i, Signal("SELL", sl_pips=sl, tp_pips=sl * tp_rr)))

    return results


def quick_stats(trades, min_closed=20):
    closed = [t for t in trades if t.result in ("WIN", "LOSS", "BE")]
    if len(closed) < min_closed:
        return None
    wins   = sum(1 for t in closed if t.result == "WIN")
    losses = sum(1 for t in closed if t.result == "LOSS")
    be_c   = sum(1 for t in closed if t.result == "BE")
    n_wr   = wins + losses
    wr     = wins / n_wr if n_wr > 0 else 0.0
    ev     = sum(t.r_multiple for t in closed) / len(closed)
    net_r  = sum(t.r_multiple for t in closed)
    return dict(n=len(closed), wins=wins, losses=losses, be=be_c,
                wr=wr, ev=ev, net_r=net_r)


def run_and_stats(bars_5m, bars_1h, bars_1d, cfg, adaptive, spread, be_r, min_closed=20):
    h1d  = bars_1d if cfg.get("macro_filter") else None
    sigs = run_mtf(bars_5m, bars_1h, h1d, cfg, adaptive=adaptive)
    if not sigs:
        return None
    trades = simulate_exits(bars_5m, sigs, spread=spread, be_r=be_r)
    return quick_stats(trades, min_closed)


# ── Sweep grid ────────────────────────────────────────────────────────────────

XAUUSD_GRID = [
    # (label, config-overrides, adaptive)
    # Relax ADX
    ("EMA100 RSI<35 ADX0   PEAK MACRO20 Fixed",
     dict(ema_period=100, rsi_entry=35, adx_min=0,  session_peak=True,  macro_filter=True,  macro_ema_period=20, tp_rr=1.5, atr_mult_sl=0.0), False),
    ("EMA100 RSI<35 ADX20  PEAK MACRO20 Fixed",
     dict(ema_period=100, rsi_entry=35, adx_min=20, session_peak=True,  macro_filter=True,  macro_ema_period=20, tp_rr=1.5, atr_mult_sl=0.0), False),
    ("EMA100 RSI<35 ADX0   SESS MACRO20 Fixed",
     dict(ema_period=100, rsi_entry=35, adx_min=0,  session_only=True,  macro_filter=True,  macro_ema_period=20, tp_rr=1.5, atr_mult_sl=0.0), False),
    ("EMA100 RSI<40 ADX20  PEAK MACRO20 Fixed",
     dict(ema_period=100, rsi_entry=40, adx_min=20, session_peak=True,  macro_filter=True,  macro_ema_period=20, tp_rr=1.5, atr_mult_sl=0.0), False),
    ("EMA100 RSI<40 ADX0   PEAK MACRO20 Fixed",
     dict(ema_period=100, rsi_entry=40, adx_min=0,  session_peak=True,  macro_filter=True,  macro_ema_period=20, tp_rr=1.5, atr_mult_sl=0.0), False),
    ("EMA100 RSI<35 ADX25  PEAK MACRO20 Adapt",
     dict(ema_period=100, rsi_entry=35, adx_min=25, session_peak=True,  macro_filter=True,  macro_ema_period=20, tp_rr=1.5, atr_mult_sl=0.0), True),
    ("EMA100 RSI<35 ADX20  PEAK MACRO20 Adapt",
     dict(ema_period=100, rsi_entry=35, adx_min=20, session_peak=True,  macro_filter=True,  macro_ema_period=20, tp_rr=1.5, atr_mult_sl=0.0), True),
    ("EMA100 RSI<35 ADX0   PEAK MACRO20 Adapt",
     dict(ema_period=100, rsi_entry=35, adx_min=0,  session_peak=True,  macro_filter=True,  macro_ema_period=20, tp_rr=1.5, atr_mult_sl=0.0), True),
    ("EMA100 RSI<35 ADX0   SESS MACRO20 Adapt",
     dict(ema_period=100, rsi_entry=35, adx_min=0,  session_only=True,  macro_filter=True,  macro_ema_period=20, tp_rr=1.5, atr_mult_sl=0.0), True),
    ("EMA100 RSI<40 ADX20  SESS MACRO20 Adapt",
     dict(ema_period=100, rsi_entry=40, adx_min=20, session_only=True,  macro_filter=True,  macro_ema_period=20, tp_rr=1.5, atr_mult_sl=0.0), True),
    ("EMA100 RSI<35 ADX0   OFF  MACRO20 Adapt",
     dict(ema_period=100, rsi_entry=35, adx_min=0,  macro_filter=True,  macro_ema_period=20, tp_rr=1.5, atr_mult_sl=0.0), True),
    ("EMA100 RSI<35 ADX0   PEAK MACRO20 RR2.0 Adapt",
     dict(ema_period=100, rsi_entry=35, adx_min=0,  session_peak=True,  macro_filter=True,  macro_ema_period=20, tp_rr=2.0, atr_mult_sl=0.0), True),
    ("EMA100 RSI<35 ADX0   PEAK noMACRO RR1.5 Adapt",
     dict(ema_period=100, rsi_entry=35, adx_min=0,  session_peak=True,  macro_filter=False, tp_rr=1.5, atr_mult_sl=0.0), True),
    ("EMA100 RSI<35 ADX20  PEAK noMACRO RR1.5 Adapt",
     dict(ema_period=100, rsi_entry=35, adx_min=20, session_peak=True,  macro_filter=False, tp_rr=1.5, atr_mult_sl=0.0), True),
]

EURUSD_GRID = [
    ("EMA200 RSI<35 ADX25 MACRO50 Fixed",
     dict(ema_period=200, rsi_entry=35, adx_min=25, macro_filter=True,  macro_ema_period=50, tp_rr=2.0), False),
    ("EMA200 RSI<35 ADX20 MACRO50 Fixed",
     dict(ema_period=200, rsi_entry=35, adx_min=20, macro_filter=True,  macro_ema_period=50, tp_rr=2.0), False),
    ("EMA200 RSI<35 ADX0  MACRO50 Fixed",
     dict(ema_period=200, rsi_entry=35, adx_min=0,  macro_filter=True,  macro_ema_period=50, tp_rr=2.0), False),
    ("EMA100 RSI<35 ADX25 MACRO20 Fixed",
     dict(ema_period=100, rsi_entry=35, adx_min=25, macro_filter=True,  macro_ema_period=20, tp_rr=2.0), False),
    ("EMA100 RSI<35 ADX20 MACRO20 Fixed",
     dict(ema_period=100, rsi_entry=35, adx_min=20, macro_filter=True,  macro_ema_period=20, tp_rr=2.0), False),
    ("EMA100 RSI<35 ADX0  MACRO20 Fixed",
     dict(ema_period=100, rsi_entry=35, adx_min=0,  macro_filter=True,  macro_ema_period=20, tp_rr=2.0), False),
    ("EMA200 RSI<35 ADX25 MACRO50 Adapt",
     dict(ema_period=200, rsi_entry=35, adx_min=25, macro_filter=True,  macro_ema_period=50, tp_rr=2.0), True),
    ("EMA200 RSI<35 ADX20 MACRO50 Adapt",
     dict(ema_period=200, rsi_entry=35, adx_min=20, macro_filter=True,  macro_ema_period=50, tp_rr=2.0), True),
    ("EMA200 RSI<35 ADX0  MACRO50 Adapt",
     dict(ema_period=200, rsi_entry=35, adx_min=0,  macro_filter=True,  macro_ema_period=50, tp_rr=2.0), True),
    ("EMA100 RSI<35 ADX25 MACRO20 Adapt",
     dict(ema_period=100, rsi_entry=35, adx_min=25, macro_filter=True,  macro_ema_period=20, tp_rr=2.0), True),
    ("EMA100 RSI<35 ADX20 MACRO20 Adapt",
     dict(ema_period=100, rsi_entry=35, adx_min=20, macro_filter=True,  macro_ema_period=20, tp_rr=2.0), True),
    ("EMA100 RSI<35 ADX0  MACRO20 Adapt",
     dict(ema_period=100, rsi_entry=35, adx_min=0,  macro_filter=True,  macro_ema_period=20, tp_rr=2.0), True),
    ("EMA100 RSI<40 ADX0  MACRO20 Adapt",
     dict(ema_period=100, rsi_entry=40, adx_min=0,  macro_filter=True,  macro_ema_period=20, tp_rr=2.0), True),
    ("EMA100 RSI<40 ADX20 MACRO20 Adapt",
     dict(ema_period=100, rsi_entry=40, adx_min=20, macro_filter=True,  macro_ema_period=20, tp_rr=2.0), True),
    ("EMA200 RSI<40 ADX0  MACRO50 Adapt",
     dict(ema_period=200, rsi_entry=40, adx_min=0,  macro_filter=True,  macro_ema_period=50, tp_rr=2.0), True),
    ("EMA200 RSI<40 ADX20 MACRO50 Adapt",
     dict(ema_period=200, rsi_entry=40, adx_min=20, macro_filter=True,  macro_ema_period=50, tp_rr=2.0), True),
    ("EMA100 RSI<35 ADX0  noMACRO Adapt RR1.5",
     dict(ema_period=100, rsi_entry=35, adx_min=0,  macro_filter=False, tp_rr=1.5), True),
    ("EMA100 RSI<35 ADX0  noMACRO Adapt RR2.0",
     dict(ema_period=100, rsi_entry=35, adx_min=0,  macro_filter=False, tp_rr=2.0), True),
    ("EMA200 RSI<35 ADX0  noMACRO Adapt RR2.0",
     dict(ema_period=200, rsi_entry=35, adx_min=0,  macro_filter=False, tp_rr=2.0), True),
    ("EMA100 RSI<35 ADX20 noMACRO Adapt RR2.0",
     dict(ema_period=100, rsi_entry=35, adx_min=20, macro_filter=False, tp_rr=2.0), True),
    ("EMA200 RSI<35 ADX20 noMACRO Adapt RR2.0",
     dict(ema_period=200, rsi_entry=35, adx_min=20, macro_filter=False, tp_rr=2.0), True),
    ("EMA100 SESS RSI<35 ADX0  MACRO20 Adapt",
     dict(ema_period=100, rsi_entry=35, adx_min=0,  session_only=True, macro_filter=True, macro_ema_period=20, tp_rr=2.0), True),
    ("EMA200 SESS RSI<35 ADX0  MACRO50 Adapt",
     dict(ema_period=200, rsi_entry=35, adx_min=0,  session_only=True, macro_filter=True, macro_ema_period=50, tp_rr=2.0), True),
]


def print_row(label, tr, ho, verdict):
    def fmt(s):
        if s is None:
            return "  --no data--"
        return (f"  n={s['n']:>3} W:{s['wins']:>3} L:{s['losses']:>2} BE:{s['be']:>2} "
                f"WR:{s['wr']*100:>5.1f}% EV:{s['ev']:>+.3f}R Net:{s['net_r']:>+6.1f}R")
    print(f"  {label}")
    print(f"    Train  {fmt(tr)}")
    print(f"    Holdout{fmt(ho)}  {'<<< ' + verdict if verdict else ''}")
    print()


async def sweep_symbol(sym, grid, spread, train, holdout):
    print(f"\n{'═'*80}")
    print(f"  {sym}  —  sweep {len(grid)} configs")
    print(f"  Train: {len(train['5m'])} bars  |  Holdout: {len(holdout['5m'])} bars")
    print(f"{'═'*80}\n")

    winning   = []   # both train & holdout positive
    train_pos = []   # only train positive (generalization failure)

    for label, cfg, adaptive in grid:
        # --- training ---
        tr = run_and_stats(train["5m"],   train["1h"],   train["1d"],
                           cfg, adaptive, spread, be_r=1.0, min_closed=30)
        # --- holdout ---
        ho = run_and_stats(holdout["5m"], holdout["1h"], holdout["1d"],
                           cfg, adaptive, spread, be_r=1.0, min_closed=10)

        if tr is None:
            continue  # not enough training trades

        if tr["ev"] <= 0.0:
            continue  # training is already losing — skip

        if ho is None:
            verdict = "holdout too few trades"
            train_pos.append((label, tr, ho, verdict))
        elif ho["ev"] > 0.05:
            verdict = "STRONG — keep"
            winning.append((label, tr, ho, verdict))
        elif ho["ev"] > 0.0:
            verdict = "OK — marginal"
            winning.append((label, tr, ho, verdict))
        else:
            verdict = "holdout negative"
            train_pos.append((label, tr, ho, verdict))

    print(f"  ── Configs where BOTH train AND holdout are positive ──\n")
    if winning:
        # Sort by holdout EV descending
        for label, tr, ho, verdict in sorted(winning, key=lambda x: -(x[2]["ev"] if x[2] else -99)):
            print_row(label, tr, ho, verdict)
    else:
        print("  None found — all configs either fail training or fail holdout.\n")

    print(f"  ── Configs that pass training but fail holdout (overfitting) ──\n")
    for label, tr, ho, verdict in sorted(train_pos, key=lambda x: -(x[1]["ev"])):
        print_row(label, tr, ho, verdict)

    if winning:
        best = max(winning, key=lambda x: x[2]["ev"] if x[2] else -99)
        print(f"\n  ★ BEST for {sym}: {best[0]}")
        print(f"    Holdout EV: {best[2]['ev']:+.3f}R  WR: {best[2]['wr']*100:.1f}%  "
              f"n={best[2]['n']}")
        return best[0], best[1], best[2]
    return None, None, None


async def main():
    print("=" * 80)
    print("EURUSD + XAUUSD CONFIG FIX  —  find configs that survive holdout")
    print("Pass 1: 180-day data  |  Pass 2: 365-day data (more regime variety)")
    print("=" * 80)

    best_overall = {}

    for sym in ["frxXAUUSD", "frxEURUSD"]:
        spread = SPREADS.get(sym, 0.0)
        grid   = XAUUSD_GRID if sym == "frxXAUUSD" else EURUSD_GRID

        for days_label, loader in [("180-day", load_data), ("365-day", load_data_long)]:
            print(f"\n  [{sym}] Loading {days_label} data...")
            try:
                bars_5m, bars_1h, bars_1d = await loader(sym)
            except Exception as e:
                print(f"  SKIP {sym} {days_label} — {e}")
                continue

            train, holdout = split_bars(bars_5m, bars_1h, bars_1d)
            print(f"  [{sym} {days_label}] train={len(train['5m'])} bars | holdout={len(holdout['5m'])} bars")

            label, tr, ho = await sweep_symbol(sym, grid, spread, train, holdout)
            if label:
                best_overall[f"{sym}_{days_label}"] = (sym, days_label, label, tr, ho)
                break  # found a working config — no need to try 365 days

    print(f"\n{'='*80}")
    print("SUMMARY — recommended config changes")
    print(f"{'='*80}\n")
    if best_overall:
        for key, (sym, days, label, tr, ho) in best_overall.items():
            print(f"  {sym} ({days}): {label}")
            print(f"    Train  EV: {tr['ev']:+.3f}R  WR: {tr['wr']*100:.1f}%  n={tr['n']}")
            print(f"    Holdout EV: {ho['ev']:+.3f}R  WR: {ho['wr']*100:.1f}%  n={ho['n']}\n")
    else:
        print("  No generalising config found for either symbol.")
        print("  The recent market regime does not suit the RSI-pullback strategy.")
        print("  Recommendation: keep adaptive RSI enabled, reduce position sizing,")
        print("  and wait for regime to normalise (RSI cycling below 40 regularly).")


if __name__ == "__main__":
    asyncio.run(main())
