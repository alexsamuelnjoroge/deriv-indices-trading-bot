"""
Test whether the XAUUSD stacked improvements (4H EMA filter, ATR×2.0, lookback 30)
also improve EURUSD and USDJPY.

EURUSD  — BB+RSI strategy, test: ATR×2.0 + 4H EMA filter
USDJPY  — MTF RSI-Pullback, test: 4H EMA filter + ATR×2.0 + adaptive lookback 30

Usage:
  python pro_bot/backtest_stack_all.py
"""

import asyncio
import bisect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pro_bot.backtest import simulate_exits, SPREADS
from pro_bot.backtest import _fetch, CACHE_5M, CACHE_1H, CACHE_1D, GRAN_1H, GRAN_1D

GRAN_5M = 300
from pro_bot.indicators import ema as _ema, rsi as _rsi, atr as _atr, bollinger as _bb
from pro_bot.strategies.base import Signal

GRAN_4H  = 14400
CACHE_4H = CACHE_1H
DAYS     = 365


# ── Data ─────────────────────────────────────────────────────────────────────

async def load(sym):
    b5  = await _fetch(sym, GRAN_5M, DAYS, CACHE_5M)
    b1h = await _fetch(sym, GRAN_1H, DAYS, CACHE_1H)
    b4h = await _fetch(sym, GRAN_4H, DAYS, CACHE_4H)
    b1d = await _fetch(sym, GRAN_1D, DAYS, CACHE_1D)
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

def _in_sess(epoch, mode):
    h = (epoch % 86400) // 3600
    m = (epoch % 3600)  // 60
    if mode == "peak":
        return ((7 <= h < 10) or (h == 10 and m < 30) or
                (13 <= h < 16) or (h == 16 and m < 30))
    if mode == "full":
        return 7 <= h < 20
    return True


def _macro(bars_1d, period, epoch, epochs_1d, ema_1d):
    k = bisect.bisect_right(epochs_1d, epoch) - 1
    if k < 1 or ema_1d[k] is None or ema_1d[k-1] is None:
        return False, False
    up = ema_1d[k] > ema_1d[k-1]
    return up, not up


def _4h_ok(bars_4h, epoch, epochs_4h, ema_4h, trend_up):
    """Returns True if 4H EMA slope agrees with the 1H trend direction."""
    if ema_4h is None:
        return True
    j = bisect.bisect_right(epochs_4h, epoch) - 1
    if j < 1 or ema_4h[j] is None or ema_4h[j-1] is None:
        return True
    ema4_up = ema_4h[j] > ema_4h[j-1]
    return ema4_up if trend_up else not ema4_up


# ── MTF RSI-Pullback runner (for USDJPY) ─────────────────────────────────────

def run_rsi(bars_5m, bars_1h, bars_4h, bars_1d, cfg):
    ema_p       = cfg.get("ema_period",       200)
    slope_b     = cfg.get("slope_bars",         3)
    rsi_p       = cfg.get("rsi_period",        14)
    rsi_entry   = cfg.get("rsi_entry",        35.0)
    tp_rr       = cfg.get("tp_rr",            1.5)
    atr_mult    = cfg.get("atr_mult_sl",      1.5)
    macro_ema_p = cfg.get("macro_ema_period",  20)
    adx_min     = cfg.get("adx_min",           25)
    adaptive    = cfg.get("adaptive",        False)
    lookback    = cfg.get("rsi_lookback",      50)
    sess        = cfg.get("session",         "off")
    use_4h      = cfg.get("use_4h_filter",  False)
    ema_4h_p    = cfg.get("ema_4h_period",    20)

    closes_1h = [b["close"] for b in bars_1h]
    closes_5m = [b["close"] for b in bars_5m]
    ema_1h    = _ema(closes_1h, ema_p)
    rsi_5m    = _rsi(closes_5m, rsi_p)
    atr_5m    = _atr(bars_5m, 14)
    epochs_1h = [b["epoch"] for b in bars_1h]
    ema_1d    = _ema([b["close"] for b in bars_1d], macro_ema_p)
    epochs_1d = [b["epoch"] for b in bars_1d]

    ema_4h = epochs_4h = None
    if use_4h and bars_4h:
        ema_4h    = _ema([b["close"] for b in bars_4h], ema_4h_p)
        epochs_4h = [b["epoch"] for b in bars_4h]

    results = []
    for i in range(rsi_p + 2, len(bars_5m)):
        epoch = bars_5m[i]["epoch"]
        if not _in_sess(epoch, sess):
            continue
        j = bisect.bisect_right(epochs_1h, epoch) - 1
        if j < slope_b or j < 0:
            continue
        e_now, e_prev = ema_1h[j], ema_1h[j - slope_b]
        r_now, r_prev = rsi_5m[i], rsi_5m[i - 1]
        if any(x is None for x in [e_now, e_prev, r_now, r_prev]):
            continue

        al, ash = _macro(bars_1d, macro_ema_p, epoch, epochs_1d, ema_1d)
        if not al and not ash:
            continue

        if adaptive and i >= lookback:
            recent = sorted(v for v in rsi_5m[i - lookback:i] if v is not None)
            if len(recent) >= max(10, lookback // 2):
                thresh = max(rsi_entry, min(50.0, recent[max(0, int(len(recent)*0.20)-1)]))
            else:
                thresh = rsi_entry
        else:
            thresh = rsi_entry
        ob = 100.0 - thresh

        up, dn = e_now > e_prev, e_now < e_prev

        if use_4h and not _4h_ok(bars_4h, epoch, epochs_4h, ema_4h, up if up else not dn):
            continue

        atr_v = atr_5m[i]
        sl = max((atr_v * atr_mult) if atr_v else 0, bars_5m[i]["close"] * 0.001)
        tp = sl * tp_rr

        if up and r_prev >= thresh > r_now and al:
            results.append((i, Signal("BUY",  sl_pips=sl, tp_pips=tp)))
        elif dn and r_prev <= ob < r_now and ash:
            results.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp)))
    return results


# ── BB+RSI runner (for EURUSD) ───────────────────────────────────────────────

def run_bb_rsi(bars_5m, bars_1h, bars_4h, bars_1d, cfg):
    ema_p       = cfg.get("ema_period",       100)
    slope_b     = cfg.get("slope_bars",         3)
    bb_p        = cfg.get("bb_period",         20)
    bb_std      = cfg.get("bb_std",           2.0)
    rsi_p       = cfg.get("rsi_period",        14)
    rsi_thresh  = cfg.get("rsi_thresh",       55.0)
    tp_rr       = cfg.get("tp_rr",            1.5)
    atr_mult    = cfg.get("atr_mult_sl",      1.5)
    macro_ema_p = cfg.get("macro_ema_period",  20)
    sess        = cfg.get("session",         "peak")
    use_4h      = cfg.get("use_4h_filter",  False)
    ema_4h_p    = cfg.get("ema_4h_period",    20)

    closes_1h = [b["close"] for b in bars_1h]
    closes_5m = [b["close"] for b in bars_5m]
    ema_1h    = _ema(closes_1h, ema_p)
    rsi_5m    = _rsi(closes_5m, rsi_p)
    bb_5m     = _bb(closes_5m, bb_p, bb_std)
    atr_5m    = _atr(bars_5m, 14)
    epochs_1h = [b["epoch"] for b in bars_1h]
    ema_1d    = _ema([b["close"] for b in bars_1d], macro_ema_p)
    epochs_1d = [b["epoch"] for b in bars_1d]

    ema_4h = epochs_4h = None
    if use_4h and bars_4h:
        ema_4h    = _ema([b["close"] for b in bars_4h], ema_4h_p)
        epochs_4h = [b["epoch"] for b in bars_4h]

    results = []
    warm = max(rsi_p + 2, bb_p + 1)
    for i in range(warm, len(bars_5m)):
        epoch = bars_5m[i]["epoch"]
        if not _in_sess(epoch, sess):
            continue
        j = bisect.bisect_right(epochs_1h, epoch) - 1
        if j < slope_b or j < 0:
            continue
        e_now, e_prev = ema_1h[j], ema_1h[j - slope_b]
        if e_now is None or e_prev is None:
            continue
        if bb_5m[i] is None or rsi_5m[i] is None:
            continue

        al, ash = _macro(bars_1d, macro_ema_p, epoch, epochs_1d, ema_1d)
        if not al and not ash:
            continue

        up, dn = e_now > e_prev, e_now < e_prev

        if use_4h:
            if up  and not _4h_ok(bars_4h, epoch, epochs_4h, ema_4h, True):
                continue
            if dn  and not _4h_ok(bars_4h, epoch, epochs_4h, ema_4h, False):
                continue

        up_b, _mid, lo_b = bb_5m[i]
        r = rsi_5m[i]
        atr_v = atr_5m[i]
        sl = max((atr_v * atr_mult) if atr_v else 0, bars_5m[i]["close"] * 0.0005)
        tp = sl * tp_rr
        price = closes_5m[i]
        ob = 100.0 - rsi_thresh

        if up and price <= lo_b and r < rsi_thresh and al:
            results.append((i, Signal("BUY",  sl_pips=sl, tp_pips=tp)))
        elif dn and price >= up_b and r > ob and ash:
            results.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp)))
    return results


# ── Stats / display ───────────────────────────────────────────────────────────

def stats(trades, min_n=10):
    closed = [t for t in trades if t.result in ("WIN", "LOSS", "BE")]
    if len(closed) < min_n:
        return None
    wins  = [t for t in closed if t.result == "WIN"]
    losses= [t for t in closed if t.result == "LOSS"]
    be_c  = [t for t in closed if t.result == "BE"]
    n_wr  = len(wins) + len(losses)
    wr    = len(wins) / n_wr if n_wr else 0
    ev    = sum(t.r_multiple for t in closed) / len(closed)
    net_r = sum(t.r_multiple for t in closed)
    buys  = [t for t in closed if t.action == "BUY"]
    sells = [t for t in closed if t.action == "SELL"]
    bwr   = sum(1 for t in buys  if t.result=="WIN") / max(1, sum(1 for t in buys  if t.result in ("WIN","LOSS")))
    swr   = sum(1 for t in sells if t.result=="WIN") / max(1, sum(1 for t in sells if t.result in ("WIN","LOSS")))
    return dict(n=len(closed), wins=len(wins), losses=len(losses), be=len(be_c),
                wr=wr, ev=ev, net_r=net_r,
                buys=len(buys), sells=len(sells), buy_wr=bwr, sel_wr=swr)


def show(label, tr, ho, baseline_ho_ev=None):
    def f(s):
        if s is None:
            return "(too few trades)"
        tag = "STRONG" if s["ev"] > 0.05 else "OK" if s["ev"] > 0 else "NEG"
        return (f"n={s['n']:>3}  WR:{s['wr']*100:>5.1f}%  "
                f"EV:{s['ev']:>+.4f}R  Net:{s['net_r']:>+5.1f}R  [{tag}]")

    if tr is None or tr["ev"] <= 0:
        return False

    improvement = ""
    if baseline_ho_ev and ho and ho["ev"] > baseline_ho_ev:
        improvement = f"  ↑ +{ho['ev'] - baseline_ho_ev:+.4f}R vs baseline"

    print(f"  {label}")
    print(f"    Train  : {f(tr)}")
    print(f"    Holdout: {f(ho)}{improvement}")
    print()
    return ho is not None and ho["ev"] > 0


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("=" * 78)
    print("STACKED IMPROVEMENTS — EURUSD + USDJPY")
    print("Testing: 4H EMA filter, ATR×2.0, adaptive lookback 30")
    print("=" * 78)

    # ══════════════════════════════════════════════════════════════════════════
    # USDJPY
    # ══════════════════════════════════════════════════════════════════════════
    sym = "frxUSDJPY"
    spread = SPREADS[sym]
    print(f"\n{'─'*78}")
    print(f"  {sym}  |  spread={spread}")
    print(f"  Current config: EMA200 ADX25 RR1.5 ATR×1.5 adaptive=false session=off")
    print(f"{'─'*78}\n")

    b5, b1h, b4h, b1d = await load(sym)
    train, hold = split(b5, b1h, b4h, b1d)
    print(f"  Train {len(train['5m'])} bars | Hold {len(hold['5m'])} bars | 4H {len(train['4h'])+len(hold['4h'])} bars\n")

    BASE_JPY = dict(ema_period=200, slope_bars=3, rsi_period=14, rsi_entry=35.0,
                    tp_rr=1.5, atr_mult_sl=1.5, macro_ema_period=20,
                    adx_min=25, adaptive=False, rsi_lookback=50,
                    session="off", use_4h_filter=False)

    def t_jpy(label, cfg):
        sigs_tr = run_rsi(train["5m"], train["1h"], train["4h"], train["1d"], cfg)
        sigs_ho = run_rsi(hold["5m"],  hold["1h"],  hold["4h"],  hold["1d"],  cfg)
        tr = stats(simulate_exits(train["5m"], sigs_tr, spread=spread, be_r=1.0), min_n=20)
        ho = stats(simulate_exits(hold["5m"],  sigs_ho, spread=spread, be_r=1.0), min_n=8)
        return show(label, tr, ho, baseline_jpy_ev)

    # Get baseline first
    sigs = run_rsi(hold["5m"], hold["1h"], hold["4h"], hold["1d"], BASE_JPY)
    ho_base = stats(simulate_exits(hold["5m"], sigs, spread=spread, be_r=1.0), min_n=8)
    baseline_jpy_ev = ho_base["ev"] if ho_base else 0

    show("Baseline (current live config)",
         stats(simulate_exits(train["5m"],
               run_rsi(train["5m"], train["1h"], train["4h"], train["1d"], BASE_JPY),
               spread=spread, be_r=1.0), min_n=20),
         ho_base, None)

    print("  ── Single improvements ──\n")
    t_jpy("+ 4H EMA20 filter",        {**BASE_JPY, "use_4h_filter": True,  "ema_4h_period": 20})
    t_jpy("+ ATR×2.0",                {**BASE_JPY, "atr_mult_sl": 2.0})
    t_jpy("+ adaptive RSI lb=30",     {**BASE_JPY, "adaptive": True, "rsi_lookback": 30})
    t_jpy("+ adaptive RSI lb=50",     {**BASE_JPY, "adaptive": True, "rsi_lookback": 50})

    print("  ── Combinations ──\n")
    t_jpy("+ 4H + ATR×2.0",
          {**BASE_JPY, "use_4h_filter": True, "ema_4h_period": 20, "atr_mult_sl": 2.0})
    t_jpy("+ 4H + adaptive lb=30",
          {**BASE_JPY, "use_4h_filter": True, "ema_4h_period": 20,
           "adaptive": True, "rsi_lookback": 30})
    t_jpy("+ ATR×2.0 + adaptive lb=30",
          {**BASE_JPY, "atr_mult_sl": 2.0, "adaptive": True, "rsi_lookback": 30})
    t_jpy("+ 4H + ATR×2.0 + adaptive lb=30",
          {**BASE_JPY, "use_4h_filter": True, "ema_4h_period": 20,
           "atr_mult_sl": 2.0, "adaptive": True, "rsi_lookback": 30})

    # ══════════════════════════════════════════════════════════════════════════
    # EURUSD
    # ══════════════════════════════════════════════════════════════════════════
    sym = "frxEURUSD"
    spread = SPREADS[sym]
    print(f"\n{'─'*78}")
    print(f"  {sym}  |  spread={spread}")
    print(f"  Current config: BB(20,2.0) RSI<55 EMA100 RR1.5 ATR×1.5 session=peak macro20")
    print(f"{'─'*78}\n")

    b5, b1h, b4h, b1d = await load(sym)
    train, hold = split(b5, b1h, b4h, b1d)
    print(f"  Train {len(train['5m'])} bars | Hold {len(hold['5m'])} bars | 4H {len(train['4h'])+len(hold['4h'])} bars\n")

    BASE_EUR = dict(ema_period=100, slope_bars=3, bb_period=20, bb_std=2.0,
                    rsi_period=14, rsi_thresh=55.0, tp_rr=1.5, atr_mult_sl=1.5,
                    macro_ema_period=20, session="peak", use_4h_filter=False)

    def t_eur(label, cfg):
        sigs_tr = run_bb_rsi(train["5m"], train["1h"], train["4h"], train["1d"], cfg)
        sigs_ho = run_bb_rsi(hold["5m"],  hold["1h"],  hold["4h"],  hold["1d"],  cfg)
        tr = stats(simulate_exits(train["5m"], sigs_tr, spread=spread, be_r=1.0), min_n=20)
        ho = stats(simulate_exits(hold["5m"],  sigs_ho, spread=spread, be_r=1.0), min_n=8)
        return show(label, tr, ho, baseline_eur_ev)

    sigs = run_bb_rsi(hold["5m"], hold["1h"], hold["4h"], hold["1d"], BASE_EUR)
    ho_base = stats(simulate_exits(hold["5m"], sigs, spread=spread, be_r=1.0), min_n=8)
    baseline_eur_ev = ho_base["ev"] if ho_base else 0

    show("Baseline (current live config)",
         stats(simulate_exits(train["5m"],
               run_bb_rsi(train["5m"], train["1h"], train["4h"], train["1d"], BASE_EUR),
               spread=spread, be_r=1.0), min_n=20),
         ho_base, None)

    print("  ── Single improvements ──\n")
    t_eur("+ ATR×2.0",                {**BASE_EUR, "atr_mult_sl": 2.0})
    t_eur("+ 4H EMA20 filter",        {**BASE_EUR, "use_4h_filter": True, "ema_4h_period": 20})
    t_eur("+ 4H EMA50 filter",        {**BASE_EUR, "use_4h_filter": True, "ema_4h_period": 50})
    t_eur("+ RR 2.0",                 {**BASE_EUR, "tp_rr": 2.0})
    t_eur("+ BB std 1.5",             {**BASE_EUR, "bb_std": 1.5})

    print("  ── Combinations ──\n")
    t_eur("+ 4H EMA20 + ATR×2.0",
          {**BASE_EUR, "use_4h_filter": True, "ema_4h_period": 20, "atr_mult_sl": 2.0})
    t_eur("+ 4H EMA20 + RR 2.0",
          {**BASE_EUR, "use_4h_filter": True, "ema_4h_period": 20, "tp_rr": 2.0})
    t_eur("+ ATR×2.0 + RR 2.0",
          {**BASE_EUR, "atr_mult_sl": 2.0, "tp_rr": 2.0})
    t_eur("+ 4H EMA20 + ATR×2.0 + RR 2.0",
          {**BASE_EUR, "use_4h_filter": True, "ema_4h_period": 20,
           "atr_mult_sl": 2.0, "tp_rr": 2.0})

    print("=" * 78)
    print("Done — apply any improvements that show BOTH train + holdout positive.")


if __name__ == "__main__":
    asyncio.run(main())
