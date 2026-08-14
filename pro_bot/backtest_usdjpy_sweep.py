"""
USDJPY strategy search.

The RSI pullback is failing in the current holdout regime (21% WR, -0.43R EV).
This script tests four alternative approaches:
  A. BB+RSI pullback  — same strategy that works for EURUSD
  B. MACD histogram reversal  — catches momentum shifts, not RSI levels
  C. Donchian breakout  — trend-following, not counter-trend
  D. MTF RSI with looser params  — shorter EMA, session filter, adaptive RSI

For each approach we test key parameter variants and report Train + Holdout EV.
Only configs passing BOTH (train EV > 0, holdout EV > 0, min trades) are shown.

Usage:
  python pro_bot/backtest_usdjpy_sweep.py
"""

import asyncio
import bisect
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pro_bot.backtest import simulate_exits, SPREADS, _fetch, CACHE_5M, CACHE_1H, CACHE_1D
from pro_bot.indicators import ema as _ema, rsi as _rsi, atr as _atr, bollinger as _bb
from pro_bot.indicators import macd as _macd, donchian as _donch

GRAN_5M  = 300
GRAN_1H  = 3600
GRAN_4H  = 14400
GRAN_1D  = 86400
DAYS     = 365
SYM      = "frxUSDJPY"
SPREAD   = SPREADS[SYM]


# ── Data ─────────────────────────────────────────────────────────────────────

async def load():
    b5  = await _fetch(SYM, GRAN_5M, DAYS, CACHE_5M)
    b1h = await _fetch(SYM, GRAN_1H, DAYS, CACHE_1H)
    b4h = await _fetch(SYM, GRAN_4H, DAYS, CACHE_5M)   # reuse 5m cache dir
    b1d = await _fetch(SYM, GRAN_1D, DAYS, CACHE_1D)
    return b5, b1h, b4h, b1d


def split(b5, b1h, b4h, b1d, pct=0.80):
    cut   = int(len(b5) * pct)
    epoch = b5[cut]["epoch"]
    tr = {"5m": b5[:cut],
          "1h": [b for b in b1h if b["epoch"] <= epoch],
          "4h": [b for b in b4h if b["epoch"] <= epoch],
          "1d": [b for b in b1d if b["epoch"] <= epoch]}
    ho = {"5m": b5[cut:],
          "1h": [b for b in b1h if b["epoch"] > epoch],
          "4h": [b for b in b4h if b["epoch"] > epoch],
          "1d": [b for b in b1d if b["epoch"] > epoch]}
    return tr, ho


# ── Shared helpers ────────────────────────────────────────────────────────────

def _sess(epoch, mode):
    h = (epoch % 86400) // 3600
    m = (epoch % 3600)  // 60
    if mode == "peak":
        return ((7 <= h < 10) or (h == 10 and m < 30) or
                (13 <= h < 16) or (h == 16 and m < 30))
    if mode == "london":
        return (7 <= h < 10) or (h == 10 and m < 30)
    if mode == "ny":
        return (13 <= h < 16) or (h == 16 and m < 30)
    if mode == "asia":
        return 0 <= h < 7
    return True


def _macro_dirs(bars_1d, period):
    ema_d  = _ema([b["close"] for b in bars_1d], period)
    epochs = [b["epoch"] for b in bars_1d]
    return ema_d, epochs


def _macro_allow(ema_d, epochs_d, epoch):
    k = bisect.bisect_right(epochs_d, epoch) - 1
    if k < 1 or ema_d[k] is None or ema_d[k-1] is None:
        return False, False
    up = ema_d[k] > ema_d[k-1]
    return up, not up


def _4h_slope(bars_4h, ema_4h_p, epoch, epochs_4h, ema_4h_vals, want_up):
    if ema_4h_vals is None:
        return True
    j = bisect.bisect_right(epochs_4h, epoch) - 1
    if j < 1 or ema_4h_vals[j] is None or ema_4h_vals[j-1] is None:
        return True
    up4 = ema_4h_vals[j] > ema_4h_vals[j-1]
    return up4 if want_up else not up4


def stats(trades, min_n=8):
    closed = [t for t in trades if t.result in ("WIN", "LOSS", "BE")]
    if len(closed) < min_n:
        return None
    wins  = sum(1 for t in closed if t.result == "WIN")
    n_wr  = sum(1 for t in closed if t.result in ("WIN", "LOSS"))
    wr    = wins / n_wr if n_wr else 0
    ev    = sum(t.r_multiple for t in closed) / len(closed)
    net_r = sum(t.r_multiple for t in closed)
    return dict(n=len(closed), wr=wr, ev=ev, net_r=net_r)


def row(label, cfg_str, tr, ho, winners):
    if tr is None or tr["ev"] <= 0:
        return
    if ho is None or ho["ev"] <= 0:
        return
    winners.append((label, cfg_str, tr, ho))


def report(winners, title):
    if not winners:
        print(f"  {title}: (no config passed both train+holdout)\n")
        return
    print(f"\n  ── {title} ──")
    for label, cfg_str, tr, ho in sorted(winners, key=lambda x: -x[3]["ev"]):
        print(f"  {label}")
        print(f"    Config  : {cfg_str}")
        print(f"    Train   : EV {tr['ev']:+.4f}R  WR {tr['wr']*100:.1f}%  n={tr['n']}")
        print(f"    Holdout : EV {ho['ev']:+.4f}R  WR {ho['wr']*100:.1f}%  n={ho['n']}  Net {ho['net_r']:+.1f}R")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Strategy A — BB+RSI pullback (same as EURUSD)
# ══════════════════════════════════════════════════════════════════════════════

def run_bb_rsi(b5, b1h, b4h, b1d, cfg):
    ema_p    = cfg["ema_period"]
    bb_p     = cfg["bb_period"]
    bb_std   = cfg["bb_std"]
    rsi_p    = cfg["rsi_period"]
    thresh   = cfg["rsi_thresh"]
    tp_rr    = cfg["tp_rr"]
    atr_mult = cfg["atr_mult_sl"]
    macro_p  = cfg["macro_ema_period"]
    sess     = cfg.get("session", "peak")
    use_4h   = cfg.get("use_4h_filter", False)
    ema_4h_p = cfg.get("ema_4h_period", 20)
    slope_b  = cfg.get("slope_bars", 3)

    closes_1h = [b["close"] for b in b1h]
    closes_5m = [b["close"] for b in b5]
    ema_1h = _ema(closes_1h, ema_p)
    rsi_5m = _rsi(closes_5m, rsi_p)
    bb_5m  = _bb(closes_5m, bb_p, bb_std)
    atr_5m = _atr(b5, 14)
    epochs_1h = [b["epoch"] for b in b1h]
    ema_d, epochs_d = _macro_dirs(b1d, macro_p)

    ema4h_vals = epochs_4h = None
    if use_4h and b4h:
        ema4h_vals = _ema([b["close"] for b in b4h], ema_4h_p)
        epochs_4h  = [b["epoch"] for b in b4h]

    results = []
    warm = max(rsi_p + 2, bb_p + 1)
    for i in range(warm, len(b5)):
        epoch = b5[i]["epoch"]
        if not _sess(epoch, sess):
            continue
        j = bisect.bisect_right(epochs_1h, epoch) - 1
        if j < slope_b:
            continue
        en, ep = ema_1h[j], ema_1h[j - slope_b]
        if en is None or ep is None:
            continue
        al, ash = _macro_allow(ema_d, epochs_d, epoch)
        if not al and not ash:
            continue
        bb_now = bb_5m[i]
        r_now  = rsi_5m[i]
        if bb_now is None or r_now is None:
            continue
        up_b, _, lo_b = bb_now
        atr_v = atr_5m[i]
        sl = max((atr_v * atr_mult) if atr_v else 0, b5[i]["close"] * 0.0005)
        tp = sl * tp_rr
        price = closes_5m[i]
        up, dn = en > ep, en < ep
        ob = 100.0 - thresh
        if use_4h:
            if up  and not _4h_slope(b4h, ema_4h_p, epoch, epochs_4h, ema4h_vals, True):
                continue
            if dn  and not _4h_slope(b4h, ema_4h_p, epoch, epochs_4h, ema4h_vals, False):
                continue
        from pro_bot.strategies.base import Signal
        if up and price <= lo_b and r_now < thresh and al:
            results.append((i, Signal("BUY",  sl_pips=sl, tp_pips=tp)))
        elif dn and price >= up_b and r_now > ob and ash:
            results.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp)))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Strategy B — MACD histogram reversal + 1H trend
# ══════════════════════════════════════════════════════════════════════════════

def run_macd(b5, b1h, b4h, b1d, cfg):
    ema_p    = cfg["ema_period"]
    fast     = cfg.get("macd_fast",   12)
    slow     = cfg.get("macd_slow",   26)
    sig_p    = cfg.get("macd_signal",  9)
    tp_rr    = cfg["tp_rr"]
    atr_mult = cfg["atr_mult_sl"]
    macro_p  = cfg["macro_ema_period"]
    sess     = cfg.get("session", "peak")
    slope_b  = cfg.get("slope_bars", 3)
    use_4h   = cfg.get("use_4h_filter", False)
    ema_4h_p = cfg.get("ema_4h_period", 20)
    min_hist = cfg.get("min_hist", 0.0)  # minimum MACD histogram magnitude to qualify

    closes_1h = [b["close"] for b in b1h]
    closes_5m = [b["close"] for b in b5]
    ema_1h  = _ema(closes_1h, ema_p)
    atr_5m  = _atr(b5, 14)
    epochs_1h = [b["epoch"] for b in b1h]
    ema_d, epochs_d = _macro_dirs(b1d, macro_p)

    _, _, hist_5m = _macd(closes_5m, fast, slow, sig_p)

    ema4h_vals = epochs_4h = None
    if use_4h and b4h:
        ema4h_vals = _ema([b["close"] for b in b4h], ema_4h_p)
        epochs_4h  = [b["epoch"] for b in b4h]

    from pro_bot.strategies.base import Signal
    results = []
    warm = slow + sig_p + 2
    for i in range(warm, len(b5)):
        epoch = b5[i]["epoch"]
        if not _sess(epoch, sess):
            continue
        j = bisect.bisect_right(epochs_1h, epoch) - 1
        if j < slope_b:
            continue
        en, ep = ema_1h[j], ema_1h[j - slope_b]
        if en is None or ep is None:
            continue
        al, ash = _macro_allow(ema_d, epochs_d, epoch)
        if not al and not ash:
            continue
        h_now  = hist_5m[i]
        h_prev = hist_5m[i - 1]
        if h_now is None or h_prev is None:
            continue
        atr_v = atr_5m[i]
        sl = max((atr_v * atr_mult) if atr_v else 0, b5[i]["close"] * 0.001)
        tp = sl * tp_rr
        up, dn = en > ep, en < ep
        if use_4h:
            if up  and not _4h_slope(b4h, ema_4h_p, epoch, epochs_4h, ema4h_vals, True):
                continue
            if dn  and not _4h_slope(b4h, ema_4h_p, epoch, epochs_4h, ema4h_vals, False):
                continue
        # BUY: trend up, MACD hist crosses from negative to positive (momentum reversal up)
        if up and h_prev < 0 and h_now >= 0 and al and abs(h_prev) >= min_hist:
            results.append((i, Signal("BUY",  sl_pips=sl, tp_pips=tp)))
        # SELL: trend down, MACD hist crosses from positive to negative
        elif dn and h_prev > 0 and h_now <= 0 and ash and abs(h_prev) >= min_hist:
            results.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp)))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Strategy C — Donchian channel breakout + 1H trend
# ══════════════════════════════════════════════════════════════════════════════

def run_donchian(b5, b1h, b4h, b1d, cfg):
    ema_p    = cfg["ema_period"]
    don_p    = cfg.get("donchian_period", 20)
    tp_rr    = cfg["tp_rr"]
    atr_mult = cfg["atr_mult_sl"]
    macro_p  = cfg["macro_ema_period"]
    sess     = cfg.get("session", "peak")
    slope_b  = cfg.get("slope_bars", 3)
    use_4h   = cfg.get("use_4h_filter", False)
    ema_4h_p = cfg.get("ema_4h_period", 20)

    closes_1h = [b["close"] for b in b1h]
    closes_5m = [b["close"] for b in b5]
    ema_1h  = _ema(closes_1h, ema_p)
    atr_5m  = _atr(b5, 14)
    epochs_1h = [b["epoch"] for b in b1h]
    ema_d, epochs_d = _macro_dirs(b1d, macro_p)
    don_hi, don_lo  = _donch(b5, don_p)

    ema4h_vals = epochs_4h = None
    if use_4h and b4h:
        ema4h_vals = _ema([b["close"] for b in b4h], ema_4h_p)
        epochs_4h  = [b["epoch"] for b in b4h]

    from pro_bot.strategies.base import Signal
    results = []
    for i in range(don_p + 1, len(b5)):
        epoch = b5[i]["epoch"]
        if not _sess(epoch, sess):
            continue
        j = bisect.bisect_right(epochs_1h, epoch) - 1
        if j < slope_b:
            continue
        en, ep = ema_1h[j], ema_1h[j - slope_b]
        if en is None or ep is None:
            continue
        al, ash = _macro_allow(ema_d, epochs_d, epoch)
        if not al and not ash:
            continue
        hi = don_hi[i - 1]  # use prior bar's channel (avoid look-ahead)
        lo = don_lo[i - 1]
        if hi is None or lo is None:
            continue
        atr_v = atr_5m[i]
        sl = max((atr_v * atr_mult) if atr_v else 0, b5[i]["close"] * 0.001)
        tp = sl * tp_rr
        price = closes_5m[i]
        up, dn = en > ep, en < ep
        if use_4h:
            if up  and not _4h_slope(b4h, ema_4h_p, epoch, epochs_4h, ema4h_vals, True):
                continue
            if dn  and not _4h_slope(b4h, ema_4h_p, epoch, epochs_4h, ema4h_vals, False):
                continue
        if up and price > hi and al:
            results.append((i, Signal("BUY",  sl_pips=sl, tp_pips=tp)))
        elif dn and price < lo and ash:
            results.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp)))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Strategy D — MTF RSI pullback (wider parameter grid)
# ══════════════════════════════════════════════════════════════════════════════

def run_rsi(b5, b1h, b4h, b1d, cfg):
    ema_p    = cfg["ema_period"]
    rsi_p    = cfg.get("rsi_period", 14)
    rsi_ent  = cfg.get("rsi_entry", 35.0)
    tp_rr    = cfg["tp_rr"]
    atr_mult = cfg["atr_mult_sl"]
    macro_p  = cfg["macro_ema_period"]
    adx_min  = cfg.get("adx_min", 0)
    adaptive = cfg.get("adaptive", False)
    lookback = cfg.get("rsi_lookback", 50)
    sess     = cfg.get("session", "off")
    slope_b  = cfg.get("slope_bars", 3)
    use_4h   = cfg.get("use_4h_filter", False)
    ema_4h_p = cfg.get("ema_4h_period", 20)

    closes_1h = [b["close"] for b in b1h]
    closes_5m = [b["close"] for b in b5]
    ema_1h  = _ema(closes_1h, ema_p)
    rsi_5m  = _rsi(closes_5m, rsi_p)
    atr_5m  = _atr(b5, 14)
    epochs_1h = [b["epoch"] for b in b1h]
    ema_d, epochs_d = _macro_dirs(b1d, macro_p)

    ema4h_vals = epochs_4h = None
    if use_4h and b4h:
        ema4h_vals = _ema([b["close"] for b in b4h], ema_4h_p)
        epochs_4h  = [b["epoch"] for b in b4h]

    from pro_bot.strategies.base import Signal
    results = []
    for i in range(rsi_p + 2, len(b5)):
        epoch = b5[i]["epoch"]
        if not _sess(epoch, sess):
            continue
        j = bisect.bisect_right(epochs_1h, epoch) - 1
        if j < slope_b:
            continue
        en, ep = ema_1h[j], ema_1h[j - slope_b]
        rn, rp = rsi_5m[i], rsi_5m[i - 1]
        if any(x is None for x in [en, ep, rn, rp]):
            continue
        al, ash = _macro_allow(ema_d, epochs_d, epoch)
        if not al and not ash:
            continue
        if adaptive and i >= lookback:
            recent = sorted(v for v in rsi_5m[i - lookback:i] if v is not None)
            if len(recent) >= max(10, lookback // 2):
                thresh = max(rsi_ent, min(50.0, recent[max(0, int(len(recent)*0.20)-1)]))
            else:
                thresh = rsi_ent
        else:
            thresh = rsi_ent
        ob = 100.0 - thresh
        atr_v = atr_5m[i]
        sl = max((atr_v * atr_mult) if atr_v else 0, b5[i]["close"] * 0.001)
        tp = sl * tp_rr
        up, dn = en > ep, en < ep
        if use_4h:
            if up  and not _4h_slope(b4h, ema_4h_p, epoch, epochs_4h, ema4h_vals, True):
                continue
            if dn  and not _4h_slope(b4h, ema_4h_p, epoch, epochs_4h, ema4h_vals, False):
                continue
        if up and rp >= thresh > rn and al:
            results.append((i, Signal("BUY",  sl_pips=sl, tp_pips=tp)))
        elif dn and rp <= ob < rn and ash:
            results.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp)))
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("=" * 78)
    print(f"USDJPY — Strategy search (365-day, 80/20 holdout, spread={SPREAD})")
    print("=" * 78)

    b5, b1h, b4h, b1d = await load()
    train, hold = split(b5, b1h, b4h, b1d)
    print(f"  Train {len(train['5m'])} 5m bars | Hold {len(hold['5m'])} 5m bars\n")

    # ── A: BB+RSI pullback ────────────────────────────────────────────────────
    print("── A: BB+RSI Pullback ─────────────────────────────────────────────────")
    bb_w = []
    for ema_p, bb_p, thresh, tp_rr, atr_m, use4h, sess in product(
        [50, 100, 200],         # 1H EMA period
        [20, 30],               # BB period
        [45.0, 50.0, 55.0],    # RSI threshold
        [1.5, 2.0],             # RR
        [1.5, 2.0],             # ATR multiplier
        [False, True],          # 4H filter
        ["off", "peak"],        # session
    ):
        cfg = dict(ema_period=ema_p, bb_period=bb_p, bb_std=2.0, rsi_period=14,
                   rsi_thresh=thresh, tp_rr=tp_rr, atr_mult_sl=atr_m,
                   macro_ema_period=20, session=sess,
                   use_4h_filter=use4h, ema_4h_period=20, slope_bars=3)
        sigs_tr = run_bb_rsi(train["5m"], train["1h"], train["4h"], train["1d"], cfg)
        sigs_ho = run_bb_rsi(hold["5m"],  hold["1h"],  hold["4h"],  hold["1d"],  cfg)
        tr = stats(simulate_exits(train["5m"], sigs_tr, spread=SPREAD, be_r=1.0), min_n=30)
        ho = stats(simulate_exits(hold["5m"],  sigs_ho, spread=SPREAD, be_r=1.0), min_n=8)
        tag = f"EMA{ema_p} BB({bb_p}) RSI<{thresh} RR{tp_rr} ATR×{atr_m} 4H={use4h} sess={sess}"
        row(tag, "", tr, ho, bb_w)

    report(bb_w[:10], "BB+RSI — Top configs (train+holdout positive)")

    # ── B: MACD reversal ──────────────────────────────────────────────────────
    print("── B: MACD Histogram Reversal ─────────────────────────────────────────")
    macd_w = []
    for ema_p, fast, slow, sig_p, tp_rr, atr_m, use4h, sess in product(
        [50, 100, 200],
        [8, 12],
        [21, 26],
        [7, 9],
        [1.5, 2.0],
        [1.5, 2.0],
        [False, True],
        ["off", "peak"],
    ):
        if fast >= slow:
            continue
        cfg = dict(ema_period=ema_p, macd_fast=fast, macd_slow=slow, macd_signal=sig_p,
                   tp_rr=tp_rr, atr_mult_sl=atr_m, macro_ema_period=20, session=sess,
                   use_4h_filter=use4h, ema_4h_period=20, slope_bars=3)
        sigs_tr = run_macd(train["5m"], train["1h"], train["4h"], train["1d"], cfg)
        sigs_ho = run_macd(hold["5m"],  hold["1h"],  hold["4h"],  hold["1d"],  cfg)
        tr = stats(simulate_exits(train["5m"], sigs_tr, spread=SPREAD, be_r=1.0), min_n=30)
        ho = stats(simulate_exits(hold["5m"],  sigs_ho, spread=SPREAD, be_r=1.0), min_n=8)
        tag = f"EMA{ema_p} MACD({fast},{slow},{sig_p}) RR{tp_rr} ATR×{atr_m} 4H={use4h} sess={sess}"
        row(tag, "", tr, ho, macd_w)

    report(macd_w[:10], "MACD Reversal — Top configs")

    # ── C: Donchian breakout ──────────────────────────────────────────────────
    print("── C: Donchian Breakout ────────────────────────────────────────────────")
    don_w = []
    for ema_p, don_p, tp_rr, atr_m, use4h, sess in product(
        [50, 100, 200],
        [10, 20, 40],
        [1.5, 2.0, 2.5],
        [1.5, 2.0],
        [False, True],
        ["off", "peak"],
    ):
        cfg = dict(ema_period=ema_p, donchian_period=don_p, tp_rr=tp_rr,
                   atr_mult_sl=atr_m, macro_ema_period=20, session=sess,
                   use_4h_filter=use4h, ema_4h_period=20, slope_bars=3)
        sigs_tr = run_donchian(train["5m"], train["1h"], train["4h"], train["1d"], cfg)
        sigs_ho = run_donchian(hold["5m"],  hold["1h"],  hold["4h"],  hold["1d"],  cfg)
        tr = stats(simulate_exits(train["5m"], sigs_tr, spread=SPREAD, be_r=1.0), min_n=30)
        ho = stats(simulate_exits(hold["5m"],  sigs_ho, spread=SPREAD, be_r=1.0), min_n=8)
        tag = f"EMA{ema_p} Don({don_p}) RR{tp_rr} ATR×{atr_m} 4H={use4h} sess={sess}"
        row(tag, "", tr, ho, don_w)

    report(don_w[:10], "Donchian Breakout — Top configs")

    # ── D: RSI pullback, wider grid ───────────────────────────────────────────
    print("── D: RSI Pullback (wider grid) ────────────────────────────────────────")
    rsi_w = []
    for ema_p, rsi_entry, tp_rr, atr_m, adaptive, lb, use4h, sess in product(
        [50, 100, 200],
        [30.0, 35.0, 40.0],
        [1.5, 2.0],
        [1.5, 2.0],
        [False, True],
        [30, 50],
        [False, True],
        ["off", "peak"],
    ):
        cfg = dict(ema_period=ema_p, rsi_period=14, rsi_entry=rsi_entry,
                   tp_rr=tp_rr, atr_mult_sl=atr_m, macro_ema_period=20,
                   adaptive=adaptive, rsi_lookback=lb, session=sess,
                   use_4h_filter=use4h, ema_4h_period=20, slope_bars=3)
        sigs_tr = run_rsi(train["5m"], train["1h"], train["4h"], train["1d"], cfg)
        sigs_ho = run_rsi(hold["5m"],  hold["1h"],  hold["4h"],  hold["1d"],  cfg)
        tr = stats(simulate_exits(train["5m"], sigs_tr, spread=SPREAD, be_r=1.0), min_n=30)
        ho = stats(simulate_exits(hold["5m"],  sigs_ho, spread=SPREAD, be_r=1.0), min_n=8)
        tag = f"EMA{ema_p} RSI<{rsi_entry} RR{tp_rr} ATR×{atr_m} ada={adaptive} lb={lb} 4H={use4h} sess={sess}"
        row(tag, "", tr, ho, rsi_w)

    report(rsi_w[:10], "RSI Pullback wider grid — Top configs")

    # ── Overall best ─────────────────────────────────────────────────────────
    all_winners = (
        [("A-BB+RSI", *x[1:]) for x in bb_w] +
        [("B-MACD",   *x[1:]) for x in macd_w] +
        [("C-Don",    *x[1:]) for x in don_w] +
        [("D-RSI",    *x[1:]) for x in rsi_w]
    )
    print("=" * 78)
    print("OVERALL BEST — sorted by holdout EV\n")
    if not all_winners:
        print("  No config passed both train and holdout with sufficient trades.")
        print("  JPY regime may be unfavourable — consider disabling temporarily.\n")
    else:
        for kind, cfg_str, tr, ho in sorted(all_winners, key=lambda x: -x[3]["ev"])[:10]:
            print(f"  [{kind}] {cfg_str}")
            print(f"    Train   : EV {tr['ev']:+.4f}R  WR {tr['wr']*100:.1f}%  n={tr['n']}")
            print(f"    Holdout : EV {ho['ev']:+.4f}R  WR {ho['wr']*100:.1f}%  n={ho['n']}  Net {ho['net_r']:+.1f}R")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
