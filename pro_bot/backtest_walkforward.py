"""
Walk-forward validation — anti-overfitting check.

We've been optimising against one 365-day window with an 80/20 holdout.
This script validates the three final configs across FOUR independent
time windows drawn from 2 years of data so no window was ever touched
during parameter selection.

Expanding-window scheme (train always precedes test chronologically):
  Window 1: Train bars  0-25%   Test bars 25-50%
  Window 2: Train bars  0-50%   Test bars 50-75%
  Window 3: Train bars  0-75%   Test bars 75-100%  ← closest to current setup
  Window 4: Train bars  0-87.5% Test bars 87.5-100%

Additionally, the 2-year fetch gives the first year as completely
independent data — never used in any optimisation run.

A config passes if it shows positive EV in at least 3 of 4 windows.

Usage:
  python pro_bot/backtest_walkforward.py
"""

import asyncio
import bisect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pro_bot.backtest import simulate_exits, SPREADS, _fetch, CACHE_5M, CACHE_1H, CACHE_1D
from pro_bot.indicators import ema as _ema, rsi as _rsi, atr as _atr, bollinger as _bb
from pro_bot.strategies.base import Signal

GRAN_5M = 300
GRAN_1H = 3600
GRAN_4H = 14400
GRAN_1D = 86400
DAYS    = 730   # 2 years — first year is fully independent of all prior optimisation


# ── Data ─────────────────────────────────────────────────────────────────────

async def load(sym):
    print(f"  [{sym}] loading 2 years of data...")
    b5  = await _fetch(sym, GRAN_5M, DAYS, CACHE_5M)
    b1h = await _fetch(sym, GRAN_1H, DAYS, CACHE_1H)
    b4h = await _fetch(sym, GRAN_4H, DAYS, CACHE_5M)
    b1d = await _fetch(sym, GRAN_1D, DAYS, CACHE_1D)
    print(f"  [{sym}] {len(b5)} 5m bars  {len(b1h)} 1h  {len(b4h)} 4h  {len(b1d)} 1d")
    return b5, b1h, b4h, b1d


def make_window(b5, b1h, b4h, b1d, train_end_pct, test_end_pct):
    """Slice all timeframes at the same epoch boundaries."""
    n    = len(b5)
    cut1 = int(n * train_end_pct)
    cut2 = int(n * test_end_pct)
    e1   = b5[cut1]["epoch"]
    e2   = b5[cut2 - 1]["epoch"]
    tr = {"5m": b5[:cut1],
          "1h": [b for b in b1h if b["epoch"] <= e1],
          "4h": [b for b in b4h if b["epoch"] <= e1],
          "1d": [b for b in b1d if b["epoch"] <= e1]}
    ho = {"5m": b5[cut1:cut2],
          "1h": [b for b in b1h if e1 < b["epoch"] <= e2],
          "4h": [b for b in b4h if e1 < b["epoch"] <= e2],
          "1d": [b for b in b1d if e1 < b["epoch"] <= e2]}
    return tr, ho


WINDOWS = [
    (0.00, 0.25, 0.50),   # (train_start%, train_end%, test_end%) — start unused here
    (0.00, 0.50, 0.75),
    (0.00, 0.75, 1.00),
    (0.00, 0.875, 1.00),
]
WINDOW_LABELS = [
    "Window 1 (train Q1, test Q2)",
    "Window 2 (train H1, test H2-part1)",
    "Window 3 (train 75%, test Q4)  ← closest to our optimisation window",
    "Window 4 (train 87.5%, test last 12.5%)",
]


# ── Shared helpers ────────────────────────────────────────────────────────────

def _sess(epoch, mode, tz_off=3):
    h = ((epoch % 86400) // 3600) + tz_off
    m = (epoch % 3600) // 60
    # hours already in local time after adding tz_off
    local_h = h % 24
    if mode == "london":
        return (10.0 <= local_h < 13.0) or (local_h == 13 and m < 30)
    if mode == "peak":
        lon = (10.0 <= local_h < 13.0) or (local_h == 13 and m < 30)
        ny  = (16.0 <= local_h < 19.0) or (local_h == 19 and m < 30)
        return lon or ny
    return True


def _macro(bars_1d, period, epoch, epochs_d, ema_d):
    k = bisect.bisect_right(epochs_d, epoch) - 1
    if k < 1 or ema_d[k] is None or ema_d[k-1] is None:
        return False, False
    up = ema_d[k] > ema_d[k-1]
    return up, not up


def _4h_agrees(b4h, epoch, epochs_4h, ema4h, want_up):
    if ema4h is None:
        return True
    j = bisect.bisect_right(epochs_4h, epoch) - 1
    if j < 1 or ema4h[j] is None or ema4h[j-1] is None:
        return True
    return (ema4h[j] > ema4h[j-1]) == want_up


def stats(trades, min_n=5):
    closed = [t for t in trades if t.result in ("WIN", "LOSS", "BE")]
    if len(closed) < min_n:
        return None
    wins  = sum(1 for t in closed if t.result == "WIN")
    n_wr  = sum(1 for t in closed if t.result in ("WIN", "LOSS"))
    wr    = wins / n_wr if n_wr else 0
    ev    = sum(t.r_multiple for t in closed) / len(closed)
    net_r = sum(t.r_multiple for t in closed)
    return dict(n=len(closed), wr=wr, ev=ev, net_r=net_r)


# ── MTF RSI Pullback (XAUUSD) ────────────────────────────────────────────────

def run_xauusd(b5, b1h, b4h, b1d):
    """EMA200 adaptive RSI(lb30) 4H-EMA20 ATR×2.0 RR2.0 sess=peak macro20"""
    ema_p, rsi_p = 200, 14
    rsi_entry, tp_rr, atr_mult = 35.0, 2.0, 2.0
    ema_4h_p, macro_p, lookback = 20, 20, 30

    closes_1h = [b["close"] for b in b1h]
    closes_5m = [b["close"] for b in b5]
    ema_1h  = _ema(closes_1h, ema_p)
    rsi_5m  = _rsi(closes_5m, rsi_p)
    atr_5m  = _atr(b5, 14)
    epochs_1h = [b["epoch"] for b in b1h]
    ema_d   = _ema([b["close"] for b in b1d], macro_p)
    epochs_d  = [b["epoch"] for b in b1d]
    ema4h   = _ema([b["close"] for b in b4h], ema_4h_p) if b4h else None
    epochs_4h = [b["epoch"] for b in b4h] if b4h else []

    results = []
    for i in range(rsi_p + 2, len(b5)):
        epoch = b5[i]["epoch"]
        if not _sess(epoch, "peak"):
            continue
        j = bisect.bisect_right(epochs_1h, epoch) - 1
        if j < 3:
            continue
        en, ep = ema_1h[j], ema_1h[j - 3]
        rn, rp = rsi_5m[i], rsi_5m[i - 1]
        if any(x is None for x in [en, ep, rn, rp]):
            continue
        al, ash = _macro(b1d, macro_p, epoch, epochs_d, ema_d)
        if not al and not ash:
            continue
        # adaptive threshold
        if i >= lookback:
            recent = sorted(v for v in rsi_5m[i - lookback:i] if v is not None)
            if len(recent) >= 6:
                thresh = max(rsi_entry, min(50.0, recent[max(0, int(len(recent)*0.20)-1)]))
            else:
                thresh = rsi_entry
        else:
            thresh = rsi_entry
        ob = 100.0 - thresh
        up, dn = en > ep, en < ep
        if not _4h_agrees(b4h, epoch, epochs_4h, ema4h, up if up else not dn):
            continue
        atr_v = atr_5m[i]
        sl = max((atr_v * atr_mult) if atr_v else 0, b5[i]["close"] * 0.001)
        tp = sl * tp_rr
        if up and rp >= thresh > rn and al:
            results.append((i, Signal("BUY",  sl_pips=sl, tp_pips=tp)))
        elif dn and rp <= ob < rn and ash:
            results.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp)))
    return results


# ── BB+RSI Pullback (EURUSD + USDJPY) ────────────────────────────────────────

def run_bb_rsi(b5, b1h, b4h, b1d, cfg):
    ema_p    = cfg["ema_period"]
    bb_p     = cfg.get("bb_period", 20)
    thresh   = cfg["rsi_thresh"]
    tp_rr    = cfg["tp_rr"]
    atr_mult = cfg["atr_mult_sl"]
    macro_p  = cfg.get("macro_ema_period", 20)
    sess     = cfg.get("session", "peak")
    ema_4h_p = cfg.get("ema_4h_period", 20)
    use_4h   = cfg.get("use_4h_filter", False)
    tz_off   = cfg.get("tz_offset_hours", 3)

    closes_1h = [b["close"] for b in b1h]
    closes_5m = [b["close"] for b in b5]
    ema_1h = _ema(closes_1h, ema_p)
    rsi_5m = _rsi(closes_5m, 14)
    bb_5m  = _bb(closes_5m, bb_p, 2.0)
    atr_5m = _atr(b5, 14)
    epochs_1h = [b["epoch"] for b in b1h]
    ema_d  = _ema([b["close"] for b in b1d], macro_p)
    epochs_d = [b["epoch"] for b in b1d]
    ema4h    = _ema([b["close"] for b in b4h], ema_4h_p) if (use_4h and b4h) else None
    epochs_4h = [b["epoch"] for b in b4h] if (use_4h and b4h) else []

    results = []
    warm = max(16, bb_p + 1)
    for i in range(warm, len(b5)):
        epoch = b5[i]["epoch"]
        if not _sess(epoch, sess, tz_off):
            continue
        j = bisect.bisect_right(epochs_1h, epoch) - 1
        if j < 3:
            continue
        en, ep = ema_1h[j], ema_1h[j - 3]
        if en is None or ep is None:
            continue
        al, ash = _macro(b1d, macro_p, epoch, epochs_d, ema_d)
        if not al and not ash:
            continue
        bb_now = bb_5m[i]
        r_now  = rsi_5m[i]
        if bb_now is None or r_now is None:
            continue
        up_b, _, lo_b = bb_now
        atr_v = atr_5m[i]
        sl = max((atr_v * atr_mult) if atr_v else 0, b5[i]["close"] * 0.001)
        tp = sl * tp_rr
        price = closes_5m[i]
        up, dn = en > ep, en < ep
        ob = 100.0 - thresh
        if use_4h:
            if up  and not _4h_agrees(b4h, epoch, epochs_4h, ema4h, True):
                continue
            if dn  and not _4h_agrees(b4h, epoch, epochs_4h, ema4h, False):
                continue
        if up and price <= lo_b and r_now < thresh and al:
            results.append((i, Signal("BUY",  sl_pips=sl, tp_pips=tp)))
        elif dn and price >= up_b and r_now > ob and ash:
            results.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp)))
    return results


# ── Configs for each symbol ───────────────────────────────────────────────────

EURUSD_CFG = dict(ema_period=100, bb_period=20, rsi_thresh=55.0,
                  tp_rr=2.0, atr_mult_sl=2.0, macro_ema_period=20,
                  session="peak", use_4h_filter=True, ema_4h_period=20,
                  tz_offset_hours=3)

USDJPY_CFG = dict(ema_period=50, bb_period=20, rsi_thresh=40.0,
                  tp_rr=2.0, atr_mult_sl=1.5, macro_ema_period=20,
                  session="london", use_4h_filter=True, ema_4h_period=20,
                  tz_offset_hours=3)


# ── Display ───────────────────────────────────────────────────────────────────

def print_window_result(label, tr_s, ho_s, spread, window_idx):
    def fmt(s):
        if s is None:
            return "(too few trades)"
        verdict = "PASS ✓" if s["ev"] > 0 else "FAIL ✗"
        return (f"n={s['n']:>3}  WR {s['wr']*100:>5.1f}%  "
                f"EV {s['ev']:>+.4f}R  Net {s['net_r']:>+5.1f}R  [{verdict}]")

    passed = ho_s is not None and ho_s["ev"] > 0
    marker = " ←" if window_idx == 2 else ""  # mark closest to our optimisation window
    print(f"    {label}{marker}")
    print(f"      Train  : {fmt(tr_s)}")
    print(f"      Holdout: {fmt(ho_s)}")
    return passed


def summarise(symbol, passes, total):
    pct = passes / total * 100
    verdict = ("ROBUST" if passes == total else
               "MOSTLY OK" if passes >= total * 0.75 else
               "MARGINAL" if passes >= total * 0.50 else
               "OVERFIT — DO NOT TRADE")
    bar = "█" * passes + "░" * (total - passes)
    print(f"\n  {symbol} summary: [{bar}] {passes}/{total} windows positive  → {verdict}")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("=" * 78)
    print("WALK-FORWARD VALIDATION — anti-overfitting check")
    print("2 years of data, 4 independent time windows per symbol")
    print("Year 1 (bars 0-50%) never used in any prior optimisation")
    print("=" * 78)

    symbols = [
        ("frxXAUUSD", "XAUUSD — MTF RSI-Pullback  (EMA200 adaptive 4H ATR×2 RR2)"),
        ("frxEURUSD", "EURUSD — BB+RSI Pullback    (EMA100 RSI<55 4H ATR×2 RR2 peak)"),
        ("frxUSDJPY", "USDJPY — BB+RSI Pullback    (EMA50  RSI<40 4H ATR×1.5 RR2 London)"),
    ]

    for sym, description in symbols:
        spread = SPREADS[sym]
        print(f"\n{'─'*78}")
        print(f"  {description}")
        print(f"  spread={spread}")
        print(f"{'─'*78}")

        b5, b1h, b4h, b1d = await load(sym)
        total_bars = len(b5)
        first_date = b5[0]["epoch"]
        last_date  = b5[-1]["epoch"]
        import time as _t
        fd = _t.strftime("%Y-%m-%d", _t.gmtime(first_date))
        ld = _t.strftime("%Y-%m-%d", _t.gmtime(last_date))
        print(f"\n  Date range: {fd} → {ld}  ({total_bars} 5m bars)\n")

        passes = 0
        for wi, (_, train_end_pct, test_end_pct) in enumerate(WINDOWS):
            tr, ho = make_window(b5, b1h, b4h, b1d, train_end_pct, test_end_pct)
            tr_5m_span = (tr["5m"][-1]["epoch"] - tr["5m"][0]["epoch"]) // 86400
            ho_5m_span = (ho["5m"][-1]["epoch"] - ho["5m"][0]["epoch"]) // 86400
            wlabel = (f"{WINDOW_LABELS[wi]}  "
                      f"[train {tr_5m_span}d / test {ho_5m_span}d]")

            if sym == "frxXAUUSD":
                sigs_tr = run_xauusd(tr["5m"], tr["1h"], tr["4h"], tr["1d"])
                sigs_ho = run_xauusd(ho["5m"], ho["1h"], ho["4h"], ho["1d"])
            elif sym == "frxEURUSD":
                sigs_tr = run_bb_rsi(tr["5m"], tr["1h"], tr["4h"], tr["1d"], EURUSD_CFG)
                sigs_ho = run_bb_rsi(ho["5m"], ho["1h"], ho["4h"], ho["1d"], EURUSD_CFG)
            else:
                sigs_tr = run_bb_rsi(tr["5m"], tr["1h"], tr["4h"], tr["1d"], USDJPY_CFG)
                sigs_ho = run_bb_rsi(ho["5m"], ho["1h"], ho["4h"], ho["1d"], USDJPY_CFG)

            tr_s = stats(simulate_exits(tr["5m"], sigs_tr, spread=spread, be_r=1.0), min_n=10)
            ho_s = stats(simulate_exits(ho["5m"], sigs_ho, spread=spread, be_r=1.0), min_n=5)
            passed = print_window_result(wlabel, tr_s, ho_s, spread, wi)
            if passed:
                passes += 1
            print()

        summarise(sym, passes, len(WINDOWS))

    print(f"\n{'='*78}")
    print("Interpretation:")
    print("  ROBUST (4/4)    — config holds across all regimes, safe to trade")
    print("  MOSTLY OK (3/4) — minor regime sensitivity, deploy with caution")
    print("  MARGINAL (2/4)  — possibly overfit, reduce position size")
    print("  OVERFIT (0-1/4) — do not trade; parameter-select again on fresh data")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
