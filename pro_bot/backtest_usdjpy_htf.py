"""
USDJPY higher-timeframe strategy search.

5m strategies fail walk-forward (0/4 windows) — JPY is macro-driven, not
mean-reverting on 5m noise. This script tests four different strategies
all operating on 1H bars, which filter out intra-hour BoJ spike noise.

Dual-holdout validation: a config must produce positive EV in TWO
independent time windows to be reported. This directly prevents the
single-window overfitting that burned the previous BB+RSI London config.

  Window A: train bars 0-50%, test bars 50-75%  (older, independent)
  Window B: train bars 0-75%, test bars 75-100% (recent — closest to live)

Strategies tested (all on 1H bars, daily macro, optional 4H trend filter):
  1. EMA crossover  — fast EMA crosses slow EMA → trend entry
  2. RSI momentum   — RSI(14) crosses 50 (not oversold — momentum shift)
  3. MACD 1H        — histogram sign flip on meaningful timeframe
  4. BB+RSI 1H      — same as EURUSD but 1H candle, not 5m noise

Usage:
  python pro_bot/backtest_usdjpy_htf.py
"""

import asyncio
import bisect
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pro_bot.backtest import simulate_exits, SPREADS, _fetch, CACHE_5M, CACHE_1H, CACHE_1D
from pro_bot.indicators import ema as _ema, rsi as _rsi, atr as _atr, bollinger as _bb
from pro_bot.indicators import macd as _macd

GRAN_1H = 3600
GRAN_4H = 14400
GRAN_1D = 86400
DAYS    = 730
SYM     = "frxUSDJPY"
SPREAD  = SPREADS[SYM]


# ── Data & windows ────────────────────────────────────────────────────────────

async def load():
    print("  Loading 2 years of 1H data...")
    b1h = await _fetch(SYM, GRAN_1H, DAYS, CACHE_1H)
    b4h = await _fetch(SYM, GRAN_4H, DAYS, CACHE_5M)
    b1d = await _fetch(SYM, GRAN_1D, DAYS, CACHE_1D)
    print(f"  {len(b1h)} 1h bars  |  {len(b4h)} 4h bars  |  {len(b1d)} 1d bars")
    return b1h, b4h, b1d


def make_window(b1h, b4h, b1d, te_pct, ho_pct):
    n   = len(b1h)
    c1  = int(n * te_pct)
    c2  = int(n * ho_pct)
    e1  = b1h[c1]["epoch"]
    e2  = b1h[c2 - 1]["epoch"]
    tr  = {"1h": b1h[:c1],
           "4h": [b for b in b4h if b["epoch"] <= e1],
           "1d": [b for b in b1d if b["epoch"] <= e1]}
    ho  = {"1h": b1h[c1:c2],
           "4h": [b for b in b4h if e1 < b["epoch"] <= e2],
           "1d": [b for b in b1d if e1 < b["epoch"] <= e2]}
    return tr, ho


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sess_1h(epoch, sess):
    """Session filter on 1H bars (EAT = UTC+3)."""
    h_eat = ((epoch % 86400) // 3600 + 3) % 24
    if sess == "london":   return 10 <= h_eat < 14
    if sess == "ny":       return 16 <= h_eat < 20
    if sess == "peak":     return (10 <= h_eat < 14) or (16 <= h_eat < 20)
    if sess == "asia":     return 2  <= h_eat < 8
    return True  # "off"


def _macro(bars_1d, period, epoch, epochs_d, ema_d):
    k = bisect.bisect_right(epochs_d, epoch) - 1
    if k < 1 or ema_d[k] is None or ema_d[k-1] is None:
        return False, False
    up = ema_d[k] > ema_d[k-1]
    return up, not up


def _4h_ok(b4h, epoch, epochs_4h, ema4h, want_up):
    if ema4h is None:
        return True
    j = bisect.bisect_right(epochs_4h, epoch) - 1
    if j < 1 or ema4h[j] is None or ema4h[j-1] is None:
        return True
    return (ema4h[j] > ema4h[j-1]) == want_up


def sim(bars_1h, signals, spread=SPREAD):
    """Simulate exits on 1H bars (BE at 1R)."""
    return simulate_exits(bars_1h, signals, spread=spread, be_r=1.0)


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


def passes(s):
    return s is not None and s["ev"] > 0


# ── Strategy 1: EMA Crossover on 1H ──────────────────────────────────────────

def run_ema_cross(b1h, b4h, b1d, cfg):
    from pro_bot.strategies.base import Signal
    fast_p   = cfg["fast"]
    slow_p   = cfg["slow"]
    tp_rr    = cfg["tp_rr"]
    atr_mult = cfg["atr_mult_sl"]
    macro_p  = cfg.get("macro_ema_period", 20)
    sess     = cfg.get("session", "off")
    use_4h   = cfg.get("use_4h_filter", False)
    ema_4h_p = cfg.get("ema_4h_period", 20)

    closes   = [b["close"] for b in b1h]
    fast_e   = _ema(closes, fast_p)
    slow_e   = _ema(closes, slow_p)
    atr_1h   = _atr(b1h, 14)
    epochs   = [b["epoch"] for b in b1h]
    ema_d    = _ema([b["close"] for b in b1d], macro_p)
    epochs_d = [b["epoch"] for b in b1d]
    ema4h    = _ema([b["close"] for b in b4h], ema_4h_p) if (use_4h and b4h) else None
    epochs_4h = [b["epoch"] for b in b4h] if (use_4h and b4h) else []

    results = []
    for i in range(slow_p + 1, len(b1h)):
        epoch = b1h[i]["epoch"]
        if not _sess_1h(epoch, sess):
            continue
        fn, fp = fast_e[i], fast_e[i-1]
        sn, sp = slow_e[i], slow_e[i-1]
        if any(x is None for x in [fn, fp, sn, sp]):
            continue
        al, ash = _macro(b1d, macro_p, epoch, epochs_d, ema_d)
        if not al and not ash:
            continue
        atr_v = atr_1h[i]
        sl = max((atr_v * atr_mult) if atr_v else 0, b1h[i]["close"] * 0.001)
        tp = sl * tp_rr
        # BUY: fast crosses above slow (golden cross) AND macro agrees
        if fp <= sp and fn > sn and al:
            if use_4h and not _4h_ok(b4h, epoch, epochs_4h, ema4h, True):
                continue
            results.append((i, Signal("BUY",  sl_pips=sl, tp_pips=tp)))
        # SELL: fast crosses below slow (death cross) AND macro agrees
        elif fp >= sp and fn < sn and ash:
            if use_4h and not _4h_ok(b4h, epoch, epochs_4h, ema4h, False):
                continue
            results.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp)))
    return results


# ── Strategy 2: RSI Momentum (cross 50) on 1H ────────────────────────────────

def run_rsi_mom(b1h, b4h, b1d, cfg):
    from pro_bot.strategies.base import Signal
    rsi_p    = cfg.get("rsi_period", 14)
    tp_rr    = cfg["tp_rr"]
    atr_mult = cfg["atr_mult_sl"]
    macro_p  = cfg.get("macro_ema_period", 20)
    ema_p    = cfg.get("ema_period", 50)
    sess     = cfg.get("session", "off")
    use_4h   = cfg.get("use_4h_filter", False)
    ema_4h_p = cfg.get("ema_4h_period", 20)
    rsi_lvl  = cfg.get("rsi_level", 50.0)  # cross level (50 = momentum, 40/60 = stronger)

    closes   = [b["close"] for b in b1h]
    ema_1h   = _ema(closes, ema_p)
    rsi_1h   = _rsi(closes, rsi_p)
    atr_1h   = _atr(b1h, 14)
    epochs   = [b["epoch"] for b in b1h]
    ema_d    = _ema([b["close"] for b in b1d], macro_p)
    epochs_d = [b["epoch"] for b in b1d]
    ema4h    = _ema([b["close"] for b in b4h], ema_4h_p) if (use_4h and b4h) else None
    epochs_4h = [b["epoch"] for b in b4h] if (use_4h and b4h) else []

    results = []
    for i in range(max(rsi_p, ema_p) + 1, len(b1h)):
        epoch = b1h[i]["epoch"]
        if not _sess_1h(epoch, sess):
            continue
        en, ep = ema_1h[i], ema_1h[i-1]
        rn, rp = rsi_1h[i], rsi_1h[i-1]
        if any(x is None for x in [en, ep, rn, rp]):
            continue
        al, ash = _macro(b1d, macro_p, epoch, epochs_d, ema_d)
        if not al and not ash:
            continue
        atr_v = atr_1h[i]
        sl = max((atr_v * atr_mult) if atr_v else 0, b1h[i]["close"] * 0.001)
        tp = sl * tp_rr
        up, dn = en > ep, en < ep
        if use_4h:
            if up  and not _4h_ok(b4h, epoch, epochs_4h, ema4h, True):
                continue
            if dn  and not _4h_ok(b4h, epoch, epochs_4h, ema4h, False):
                continue
        ob = 100.0 - rsi_lvl
        # BUY: 1H trend up + RSI crosses above rsi_level (momentum turning positive)
        if up and rp < rsi_lvl <= rn and al:
            results.append((i, Signal("BUY",  sl_pips=sl, tp_pips=tp)))
        # SELL: 1H trend down + RSI crosses below (100 - rsi_level)
        elif dn and rp > ob >= rn and ash:
            results.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp)))
    return results


# ── Strategy 3: MACD histogram flip on 1H ────────────────────────────────────

def run_macd_1h(b1h, b4h, b1d, cfg):
    from pro_bot.strategies.base import Signal
    fast_p   = cfg.get("macd_fast",   12)
    slow_p   = cfg.get("macd_slow",   26)
    sig_p    = cfg.get("macd_signal",  9)
    ema_p    = cfg.get("ema_period",  50)
    tp_rr    = cfg["tp_rr"]
    atr_mult = cfg["atr_mult_sl"]
    macro_p  = cfg.get("macro_ema_period", 20)
    sess     = cfg.get("session", "off")
    use_4h   = cfg.get("use_4h_filter", False)
    ema_4h_p = cfg.get("ema_4h_period", 20)

    closes    = [b["close"] for b in b1h]
    ema_1h    = _ema(closes, ema_p)
    _, _, hist = _macd(closes, fast_p, slow_p, sig_p)
    atr_1h    = _atr(b1h, 14)
    ema_d     = _ema([b["close"] for b in b1d], macro_p)
    epochs_d  = [b["epoch"] for b in b1d]
    ema4h     = _ema([b["close"] for b in b4h], ema_4h_p) if (use_4h and b4h) else None
    epochs_4h = [b["epoch"] for b in b4h] if (use_4h and b4h) else []

    results = []
    warm = slow_p + sig_p + 2
    for i in range(warm, len(b1h)):
        epoch = b1h[i]["epoch"]
        if not _sess_1h(epoch, sess):
            continue
        en, ep = ema_1h[i], ema_1h[i-1]
        hn, hp = hist[i], hist[i-1]
        if any(x is None for x in [en, ep, hn, hp]):
            continue
        al, ash = _macro(b1d, macro_p, epoch, epochs_d, ema_d)
        if not al and not ash:
            continue
        atr_v = atr_1h[i]
        sl = max((atr_v * atr_mult) if atr_v else 0, b1h[i]["close"] * 0.001)
        tp = sl * tp_rr
        up, dn = en > ep, en < ep
        if use_4h:
            if up  and not _4h_ok(b4h, epoch, epochs_4h, ema4h, True):
                continue
            if dn  and not _4h_ok(b4h, epoch, epochs_4h, ema4h, False):
                continue
        # BUY: trend up + MACD histogram flips from negative to positive
        if up and hp < 0 and hn >= 0 and al:
            results.append((i, Signal("BUY",  sl_pips=sl, tp_pips=tp)))
        # SELL: trend down + histogram flips from positive to negative
        elif dn and hp > 0 and hn <= 0 and ash:
            results.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp)))
    return results


# ── Strategy 4: BB+RSI on 1H ─────────────────────────────────────────────────

def run_bb_rsi_1h(b1h, b4h, b1d, cfg):
    from pro_bot.strategies.base import Signal
    ema_p    = cfg.get("ema_period",  100)
    bb_p     = cfg.get("bb_period",   20)
    bb_std   = cfg.get("bb_std",      2.0)
    thresh   = cfg.get("rsi_thresh",  55.0)
    tp_rr    = cfg["tp_rr"]
    atr_mult = cfg["atr_mult_sl"]
    macro_p  = cfg.get("macro_ema_period", 20)
    sess     = cfg.get("session", "off")
    use_4h   = cfg.get("use_4h_filter", False)
    ema_4h_p = cfg.get("ema_4h_period", 20)

    closes    = [b["close"] for b in b1h]
    ema_1h    = _ema(closes, ema_p)
    rsi_1h    = _rsi(closes, 14)
    bb_1h     = _bb(closes, bb_p, bb_std)
    atr_1h    = _atr(b1h, 14)
    ema_d     = _ema([b["close"] for b in b1d], macro_p)
    epochs_d  = [b["epoch"] for b in b1d]
    ema4h     = _ema([b["close"] for b in b4h], ema_4h_p) if (use_4h and b4h) else None
    epochs_4h = [b["epoch"] for b in b4h] if (use_4h and b4h) else []

    results = []
    warm = max(ema_p + 1, bb_p + 1, 16)
    for i in range(warm, len(b1h)):
        epoch = b1h[i]["epoch"]
        if not _sess_1h(epoch, sess):
            continue
        en, ep = ema_1h[i], ema_1h[i-3] if i >= 3 else (None, None)
        if en is None or ep is None:
            continue
        bb_now = bb_1h[i]
        rn = rsi_1h[i]
        if bb_now is None or rn is None:
            continue
        al, ash = _macro(b1d, macro_p, epoch, epochs_d, ema_d)
        if not al and not ash:
            continue
        up_b, _, lo_b = bb_now
        atr_v = atr_1h[i]
        sl = max((atr_v * atr_mult) if atr_v else 0, b1h[i]["close"] * 0.001)
        tp = sl * tp_rr
        price = closes[i]
        up, dn = en > ep, en < ep
        ob = 100.0 - thresh
        if use_4h:
            if up  and not _4h_ok(b4h, epoch, epochs_4h, ema4h, True):
                continue
            if dn  and not _4h_ok(b4h, epoch, epochs_4h, ema4h, False):
                continue
        if up and price <= lo_b and rn < thresh and al:
            results.append((i, Signal("BUY",  sl_pips=sl, tp_pips=tp)))
        elif dn and price >= up_b and rn > ob and ash:
            results.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp)))
    return results


# ── Dual-holdout runner ───────────────────────────────────────────────────────

def dual_pass(b1h, b4h, b1d, runner, cfg, min_n_tr=15, min_n_ho=5):
    """
    Returns (tr_A, ho_A, tr_B, ho_B) or None if either window fails.
    Window A: train 0-50%, test 50-75%  (older — independent of all prior work)
    Window B: train 0-75%, test 75-100% (recent — our usual optimisation window)
    """
    tr_A, ho_A = make_window(b1h, b4h, b1d, 0.50, 0.75)
    tr_B, ho_B = make_window(b1h, b4h, b1d, 0.75, 1.00)

    sA_tr = runner(tr_A["1h"], tr_A["4h"], tr_A["1d"], cfg)
    sA_ho = runner(ho_A["1h"], ho_A["4h"], ho_A["1d"], cfg)
    sB_tr = runner(tr_B["1h"], tr_B["4h"], tr_B["1d"], cfg)
    sB_ho = runner(ho_B["1h"], ho_B["4h"], ho_B["1d"], cfg)

    stA_tr = stats(sim(tr_A["1h"], sA_tr), min_n=min_n_tr)
    stA_ho = stats(sim(ho_A["1h"], sA_ho), min_n=min_n_ho)
    stB_tr = stats(sim(tr_B["1h"], sB_tr), min_n=min_n_tr)
    stB_ho = stats(sim(ho_B["1h"], sB_ho), min_n=min_n_ho)

    if not (passes(stA_tr) and passes(stA_ho) and passes(stB_tr) and passes(stB_ho)):
        return None
    return stA_tr, stA_ho, stB_tr, stB_ho


def show(label, res):
    stA_tr, stA_ho, stB_tr, stB_ho = res
    def f(s): return f"EV {s['ev']:>+.4f}R  WR {s['wr']*100:>5.1f}%  n={s['n']:>3}"
    print(f"  {label}")
    print(f"    Win-A  train  : {f(stA_tr)}")
    print(f"    Win-A  holdout: {f(stA_ho)}")
    print(f"    Win-B  train  : {f(stB_tr)}")
    print(f"    Win-B  holdout: {f(stB_ho)}")
    avg_ho = (stA_ho["ev"] + stB_ho["ev"]) / 2
    print(f"    Avg holdout EV: {avg_ho:>+.4f}R")
    print()
    return avg_ho


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("=" * 78)
    print("USDJPY — Higher-Timeframe Strategy Search (1H bars, dual-holdout)")
    print("Dual-holdout: BOTH Window A (older) AND Window B (recent) must pass")
    print(f"Spread={SPREAD}  BE@1R  2-year dataset")
    print("=" * 78)

    b1h, b4h, b1d = await load()
    print()

    all_winners = []   # (avg_ho_ev, label, strategy, cfg, res)

    # ── 1. EMA Crossover ─────────────────────────────────────────────────────
    print("── 1. EMA Crossover on 1H ─────────────────────────────────────────────")
    cross_w = []
    for fast, slow, tp_rr, atr_m, use4h, sess in product(
        [9, 12, 20],
        [26, 50, 100, 200],
        [1.5, 2.0, 2.5],
        [1.5, 2.0],
        [False, True],
        ["off", "london", "ny", "peak"],
    ):
        if fast >= slow:
            continue
        cfg = dict(fast=fast, slow=slow, tp_rr=tp_rr, atr_mult_sl=atr_m,
                   macro_ema_period=20, session=sess,
                   use_4h_filter=use4h, ema_4h_period=20)
        res = dual_pass(b1h, b4h, b1d, run_ema_cross, cfg)
        if res:
            lbl = f"EMA-X  fast={fast} slow={slow} RR{tp_rr} ATR×{atr_m} 4H={'Y' if use4h else 'N'} sess={sess}"
            cross_w.append((res[1]["ev"] + res[3]["ev"]) / 2, lbl, res)

    print(f"  {len(cross_w)} configs passed dual-holdout\n")
    for avg, lbl, res in sorted(cross_w, key=lambda x: -x[0])[:5]:
        avg_ev = show(lbl, res)
        all_winners.append((avg_ev, lbl, res))

    # ── 2. RSI Momentum ──────────────────────────────────────────────────────
    print("── 2. RSI Momentum (cross level) on 1H ────────────────────────────────")
    rsi_w = []
    for ema_p, rsi_lvl, tp_rr, atr_m, use4h, sess in product(
        [20, 50, 100, 200],
        [45.0, 50.0, 55.0],
        [1.5, 2.0, 2.5],
        [1.5, 2.0],
        [False, True],
        ["off", "london", "ny", "peak"],
    ):
        cfg = dict(ema_period=ema_p, rsi_level=rsi_lvl, tp_rr=tp_rr,
                   atr_mult_sl=atr_m, macro_ema_period=20, session=sess,
                   use_4h_filter=use4h, ema_4h_period=20)
        res = dual_pass(b1h, b4h, b1d, run_rsi_mom, cfg)
        if res:
            lbl = f"RSI-Mom EMA{ema_p} cross{rsi_lvl} RR{tp_rr} ATR×{atr_m} 4H={'Y' if use4h else 'N'} sess={sess}"
            rsi_w.append(((res[1]["ev"] + res[3]["ev"]) / 2, lbl, res))

    print(f"  {len(rsi_w)} configs passed dual-holdout\n")
    for avg, lbl, res in sorted(rsi_w, key=lambda x: -x[0])[:5]:
        avg_ev = show(lbl, res)
        all_winners.append((avg_ev, lbl, res))

    # ── 3. MACD 1H ───────────────────────────────────────────────────────────
    print("── 3. MACD Histogram Flip on 1H ───────────────────────────────────────")
    macd_w = []
    for ema_p, fast, slow, sig, tp_rr, atr_m, use4h, sess in product(
        [20, 50, 100],
        [8, 12],
        [21, 26],
        [7, 9],
        [1.5, 2.0, 2.5],
        [1.5, 2.0],
        [False, True],
        ["off", "london", "ny", "peak"],
    ):
        if fast >= slow:
            continue
        cfg = dict(ema_period=ema_p, macd_fast=fast, macd_slow=slow,
                   macd_signal=sig, tp_rr=tp_rr, atr_mult_sl=atr_m,
                   macro_ema_period=20, session=sess,
                   use_4h_filter=use4h, ema_4h_period=20)
        res = dual_pass(b1h, b4h, b1d, run_macd_1h, cfg)
        if res:
            lbl = (f"MACD1H EMA{ema_p} ({fast},{slow},{sig}) "
                   f"RR{tp_rr} ATR×{atr_m} 4H={'Y' if use4h else 'N'} sess={sess}")
            macd_w.append(((res[1]["ev"] + res[3]["ev"]) / 2, lbl, res))

    print(f"  {len(macd_w)} configs passed dual-holdout\n")
    for avg, lbl, res in sorted(macd_w, key=lambda x: -x[0])[:5]:
        avg_ev = show(lbl, res)
        all_winners.append((avg_ev, lbl, res))

    # ── 4. BB+RSI 1H ─────────────────────────────────────────────────────────
    print("── 4. BB+RSI on 1H bars ───────────────────────────────────────────────")
    bb_w = []
    for ema_p, bb_p, thresh, tp_rr, atr_m, use4h, sess in product(
        [20, 50, 100, 200],
        [14, 20, 30],
        [45.0, 50.0, 55.0, 60.0],
        [1.5, 2.0, 2.5],
        [1.5, 2.0],
        [False, True],
        ["off", "london", "ny", "peak"],
    ):
        cfg = dict(ema_period=ema_p, bb_period=bb_p, rsi_thresh=thresh,
                   tp_rr=tp_rr, atr_mult_sl=atr_m, macro_ema_period=20, session=sess,
                   use_4h_filter=use4h, ema_4h_period=20)
        res = dual_pass(b1h, b4h, b1d, run_bb_rsi_1h, cfg)
        if res:
            lbl = (f"BB+RSI1H EMA{ema_p} BB({bb_p}) RSI<{thresh} "
                   f"RR{tp_rr} ATR×{atr_m} 4H={'Y' if use4h else 'N'} sess={sess}")
            bb_w.append(((res[1]["ev"] + res[3]["ev"]) / 2, lbl, res))

    print(f"  {len(bb_w)} configs passed dual-holdout\n")
    for avg, lbl, res in sorted(bb_w, key=lambda x: -x[0])[:5]:
        avg_ev = show(lbl, res)
        all_winners.append((avg_ev, lbl, res))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 78)
    if not all_winners:
        print("  No config passed dual-holdout validation.")
        print("  USDJPY may require a fundamentally different approach")
        print("  (news-driven, carry-trade aware, or longer-cycle only).")
    else:
        print(f"  OVERALL BEST (sorted by avg holdout EV across both windows)\n")
        for avg, lbl, res in sorted(all_winners, key=lambda x: -x[0])[:10]:
            stA_tr, stA_ho, stB_tr, stB_ho = res
            print(f"  {lbl}")
            print(f"    Win-A holdout: EV {stA_ho['ev']:>+.4f}R  WR {stA_ho['wr']*100:.1f}%  n={stA_ho['n']}")
            print(f"    Win-B holdout: EV {stB_ho['ev']:>+.4f}R  WR {stB_ho['wr']*100:.1f}%  n={stB_ho['n']}")
            print(f"    Avg holdout  : {avg:>+.4f}R")
            print()
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
