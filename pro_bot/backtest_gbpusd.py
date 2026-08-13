"""
GBPUSD strategy search — dual-holdout validation.

Three strategy families tested on 2 years of data:
  1. BB+RSI Pullback     (5m entry, 1H trend, 4H slope, daily macro)
  2. MTF RSI-Pullback    (5m entry, 1H trend, 4H slope, daily macro)
  3. MACD Histogram Flip (1H entry, 4H slope, daily macro)

Dual-holdout: BOTH training AND holdout must pass in two independent windows.
  Window A: train 0-50%,  test 50-75%  (older, fully independent)
  Window B: train 0-75%,  test 75-100% (recent)

Usage:
  python pro_bot/backtest_gbpusd.py
"""

import asyncio
import bisect
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pro_bot.backtest import simulate_exits, SPREADS, _fetch, CACHE_5M, CACHE_1H, CACHE_1D
from pro_bot.indicators import ema as _ema, rsi as _rsi, atr as _atr, bollinger as _bb, macd as _macd
from pro_bot.strategies.base import Signal

SYM    = "frxGBPUSD"
SPREAD = SPREADS[SYM]
DAYS   = 730

GRAN_5M = 300
GRAN_1H = 3600
GRAN_4H = 14400
GRAN_1D = 86400


# ── Data ─────────────────────────────────────────────────────────────────────

async def load():
    print(f"  Loading {DAYS}-day dataset...")
    b5  = await _fetch(SYM, GRAN_5M, DAYS, CACHE_5M)
    b1h = await _fetch(SYM, GRAN_1H, DAYS, CACHE_1H)
    b4h = await _fetch(SYM, GRAN_4H, DAYS, CACHE_1H)
    b1d = await _fetch(SYM, GRAN_1D, DAYS, CACHE_1D)
    print(f"  {len(b5)} 5m | {len(b1h)} 1H | {len(b4h)} 4H | {len(b1d)} 1D\n")
    return b5, b1h, b4h, b1d


# ── Windows ───────────────────────────────────────────────────────────────────

def make_window_5m(b5, b1h, b4h, b1d, te_pct, ho_pct):
    """Slice all timeframes aligned to 5m epoch boundaries."""
    n  = len(b5)
    c1 = int(n * te_pct)
    c2 = int(n * ho_pct)
    e1 = b5[c1]["epoch"]
    e2 = b5[c2 - 1]["epoch"]
    tr = {"5m": b5[:c1],
          "1h": [b for b in b1h if b["epoch"] <= e1],
          "4h": [b for b in b4h if b["epoch"] <= e1],
          "1d": [b for b in b1d if b["epoch"] <= e1]}
    ho = {"5m": b5[c1:c2],
          "1h": [b for b in b1h if e1 < b["epoch"] <= e2],
          "4h": [b for b in b4h if e1 < b["epoch"] <= e2],
          "1d": [b for b in b1d if e1 < b["epoch"] <= e2]}
    return tr, ho


def make_window_1h(b1h, b4h, b1d, te_pct, ho_pct):
    """Slice aligned to 1H epoch boundaries (for MACD 1H strategy)."""
    n  = len(b1h)
    c1 = int(n * te_pct)
    c2 = int(n * ho_pct)
    e1 = b1h[c1]["epoch"]
    e2 = b1h[c2 - 1]["epoch"]
    tr = {"1h": b1h[:c1],
          "4h": [b for b in b4h if b["epoch"] <= e1],
          "1d": [b for b in b1d if b["epoch"] <= e1]}
    ho = {"1h": b1h[c1:c2],
          "4h": [b for b in b4h if e1 < b["epoch"] <= e2],
          "1d": [b for b in b1d if e1 < b["epoch"] <= e2]}
    return tr, ho


# ── Shared helpers ────────────────────────────────────────────────────────────

def _sess(epoch, sess, tz_off=3):
    h = ((epoch % 86400) // 3600 + tz_off) % 24
    if sess == "london":  return 10 <= h < 14
    if sess == "ny":      return 16 <= h < 20
    if sess == "peak":    return (10 <= h < 14) or (16 <= h < 20)
    return True


def _macro_gate(epoch, epochs_d, ema_d):
    k = bisect.bisect_right(epochs_d, epoch) - 1
    if k < 1 or ema_d[k] is None or ema_d[k-1] is None:
        return False, False
    up = ema_d[k] > ema_d[k-1]
    return up, not up


def _4h_ok(epoch, epochs_4h, ema4h, want_up):
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
    return dict(n=len(closed), wr=wr, ev=ev)


def sim5(bars, signals):
    return simulate_exits(bars, signals, spread=SPREAD, be_r=1.0)


def sim1h(bars, signals):
    return simulate_exits(bars, signals, spread=SPREAD, be_r=1.0)


# ── Strategy 1: BB+RSI Pullback (5m) ─────────────────────────────────────────

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

    closes_5m = [b["close"] for b in b5]
    closes_1h = [b["close"] for b in b1h]
    ema_1h    = _ema(closes_1h, ema_p)
    rsi_5m    = _rsi(closes_5m, 14)
    bb_5m     = _bb(closes_5m, bb_p, 2.0)
    atr_5m    = _atr(b5, 14)
    epochs_1h = [b["epoch"] for b in b1h]
    ema_d     = _ema([b["close"] for b in b1d], macro_p)
    epochs_d  = [b["epoch"] for b in b1d]
    ema4h     = _ema([b["close"] for b in b4h], ema_4h_p) if (use_4h and b4h) else None
    epochs_4h = [b["epoch"] for b in b4h] if (use_4h and b4h) else []

    results = []
    warm = max(bb_p + 1, 16)
    for i in range(warm, len(b5)):
        epoch = b5[i]["epoch"]
        if not _sess(epoch, sess):
            continue
        j = bisect.bisect_right(epochs_1h, epoch) - 1
        if j < 3:
            continue
        en, ep = ema_1h[j], ema_1h[j - 3]
        if en is None or ep is None:
            continue
        al, ash = _macro_gate(epoch, epochs_d, ema_d)
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
            if up  and not _4h_ok(epoch, epochs_4h, ema4h, True):  continue
            if dn  and not _4h_ok(epoch, epochs_4h, ema4h, False): continue
        if up and price <= lo_b and r_now < thresh and al:
            results.append((i, Signal("BUY",  sl_pips=sl, tp_pips=tp)))
        elif dn and price >= up_b and r_now > ob and ash:
            results.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp)))
    return results


# ── Strategy 2: MTF RSI-Pullback (5m) ────────────────────────────────────────

def run_mtf_rsi(b5, b1h, b4h, b1d, cfg):
    ema_p    = cfg["ema_period"]
    rsi_e    = cfg.get("rsi_entry", 35.0)
    lookback = cfg.get("rsi_lookback", 30)
    adaptive = cfg.get("rsi_adaptive", True)
    tp_rr    = cfg["tp_rr"]
    atr_mult = cfg["atr_mult_sl"]
    macro_p  = cfg.get("macro_ema_period", 20)
    sess     = cfg.get("session", "peak")
    ema_4h_p = cfg.get("ema_4h_period", 20)
    use_4h   = cfg.get("use_4h_filter", False)

    closes_5m = [b["close"] for b in b5]
    closes_1h = [b["close"] for b in b1h]
    ema_1h    = _ema(closes_1h, ema_p)
    rsi_5m    = _rsi(closes_5m, 14)
    atr_5m    = _atr(b5, 14)
    epochs_1h = [b["epoch"] for b in b1h]
    ema_d     = _ema([b["close"] for b in b1d], macro_p)
    epochs_d  = [b["epoch"] for b in b1d]
    ema4h     = _ema([b["close"] for b in b4h], ema_4h_p) if (use_4h and b4h) else None
    epochs_4h = [b["epoch"] for b in b4h] if (use_4h and b4h) else []

    results = []
    for i in range(max(15, lookback), len(b5)):
        epoch = b5[i]["epoch"]
        if not _sess(epoch, sess):
            continue
        j = bisect.bisect_right(epochs_1h, epoch) - 1
        if j < 3:
            continue
        en, ep = ema_1h[j], ema_1h[j - 3]
        rn, rp = rsi_5m[i], rsi_5m[i - 1]
        if any(x is None for x in [en, ep, rn, rp]):
            continue
        al, ash = _macro_gate(epoch, epochs_d, ema_d)
        if not al and not ash:
            continue
        # adaptive RSI threshold
        if adaptive and i >= lookback:
            recent = sorted(v for v in rsi_5m[i - lookback:i] if v is not None)
            thresh = (max(rsi_e, min(50.0, recent[max(0, int(len(recent) * 0.20) - 1)]))
                      if len(recent) >= 6 else rsi_e)
        else:
            thresh = rsi_e
        ob = 100.0 - thresh
        atr_v = atr_5m[i]
        sl = max((atr_v * atr_mult) if atr_v else 0, b5[i]["close"] * 0.001)
        tp = sl * tp_rr
        up, dn = en > ep, en < ep
        if use_4h:
            if up  and not _4h_ok(epoch, epochs_4h, ema4h, True):  continue
            if dn  and not _4h_ok(epoch, epochs_4h, ema4h, False): continue
        if up and rp >= thresh > rn and al:
            results.append((i, Signal("BUY",  sl_pips=sl, tp_pips=tp)))
        elif dn and rp <= ob < rn and ash:
            results.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp)))
    return results


# ── Strategy 3: MACD 1H ───────────────────────────────────────────────────────

def run_macd_1h(b1h, b4h, b1d, cfg):
    fast_p  = cfg["macd_fast"]
    slow_p  = cfg["macd_slow"]
    sig_p   = cfg["macd_signal"]
    ema_p   = cfg["ema_period"]
    tp_rr   = cfg["tp_rr"]
    atr_m   = cfg["atr_mult_sl"]
    macro_p = cfg["macro_ema_period"]
    sess    = cfg["session"]
    ema4h_p = cfg["ema_4h_period"]

    closes     = [b["close"] for b in b1h]
    ema_1h     = _ema(closes, ema_p)
    _, _, hist = _macd(closes, fast_p, slow_p, sig_p)
    atr_1h     = _atr(b1h, 14)
    ema_d      = _ema([b["close"] for b in b1d], macro_p)
    epochs_d   = [b["epoch"] for b in b1d]
    ema4h      = _ema([b["close"] for b in b4h], ema4h_p) if b4h else None
    epochs_4h  = [b["epoch"] for b in b4h] if b4h else []

    results = []
    warm = slow_p + sig_p + 2
    for i in range(warm, len(b1h)):
        epoch = b1h[i]["epoch"]
        if not _sess(epoch, sess):
            continue
        en, ep = ema_1h[i], ema_1h[i-1]
        hn, hp = hist[i],   hist[i-1]
        if any(x is None for x in [en, ep, hn, hp]):
            continue
        al, ash = _macro_gate(epoch, epochs_d, ema_d)
        if not al and not ash:
            continue
        up, dn = en > ep, en < ep
        if b4h:
            if up  and not _4h_ok(epoch, epochs_4h, ema4h, True):  continue
            if dn  and not _4h_ok(epoch, epochs_4h, ema4h, False): continue
        atr_v = atr_1h[i]
        sl = max((atr_v * atr_m) if atr_v else 0, b1h[i]["close"] * 0.001)
        tp = sl * tp_rr
        if up and hp < 0 and hn >= 0 and al:
            results.append((i, Signal("BUY",  sl_pips=sl, tp_pips=tp)))
        elif dn and hp > 0 and hn <= 0 and ash:
            results.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp)))
    return results


# ── Dual-holdout runner ───────────────────────────────────────────────────────
# Compute signals ONCE on full dataset, split by bar index.
# Avoids recomputing large indicator arrays (RSI/BB/EMA on 52k bars) × 4 windows.

def _split_sigs(all_sigs, cut1, cut2, n):
    """Partition signals into (train_A, holdout_A, train_B, holdout_B)."""
    tr_A = [(i,      s) for i, s in all_sigs if i <  cut1]
    ho_A = [(i-cut1, s) for i, s in all_sigs if cut1 <= i < cut2]
    tr_B = [(i,      s) for i, s in all_sigs if i <  cut2]
    ho_B = [(i-cut2, s) for i, s in all_sigs if i >= cut2]
    return tr_A, ho_A, tr_B, ho_B


def dual_pass_5m(b5, b1h, b4h, b1d, runner, cfg, min_n_tr=15, min_n_ho=5):
    # Run once on the full dataset — indicators warm up naturally
    all_sigs = runner(b5, b1h, b4h, b1d, cfg)

    n    = len(b5)
    cut1 = int(n * 0.50)   # 50%
    cut2 = int(n * 0.75)   # 75%

    tr_A, ho_A, tr_B, ho_B = _split_sigs(all_sigs, cut1, cut2, n)

    stA_tr = stats(sim5(b5[:cut1],        tr_A), min_n=min_n_tr)
    stA_ho = stats(sim5(b5[cut1:cut2],    ho_A), min_n=min_n_ho)
    stB_tr = stats(sim5(b5[:cut2],        tr_B), min_n=min_n_tr)
    stB_ho = stats(sim5(b5[cut2:],        ho_B), min_n=min_n_ho)

    def ok(s): return s is not None and s["ev"] > 0
    if not (ok(stA_tr) and ok(stA_ho) and ok(stB_tr) and ok(stB_ho)):
        return None
    return stA_ho, stB_ho


def dual_pass_1h(b1h, b4h, b1d, cfg, min_n_tr=15, min_n_ho=5):
    all_sigs = run_macd_1h(b1h, b4h, b1d, cfg)

    n    = len(b1h)
    cut1 = int(n * 0.50)
    cut2 = int(n * 0.75)

    tr_A, ho_A, tr_B, ho_B = _split_sigs(all_sigs, cut1, cut2, n)

    stA_tr = stats(sim1h(b1h[:cut1],     tr_A), min_n=min_n_tr)
    stA_ho = stats(sim1h(b1h[cut1:cut2], ho_A), min_n=min_n_ho)
    stB_tr = stats(sim1h(b1h[:cut2],     tr_B), min_n=min_n_tr)
    stB_ho = stats(sim1h(b1h[cut2:],     ho_B), min_n=min_n_ho)

    def ok(s): return s is not None and s["ev"] > 0
    if not (ok(stA_tr) and ok(stA_ho) and ok(stB_tr) and ok(stB_ho)):
        return None
    return stA_ho, stB_ho


# ── Reporting ─────────────────────────────────────────────────────────────────

def show_top(winners, title, n=5):
    if not winners:
        print(f"  {title}: (none)")
        return
    print(f"\n  {title}  ({len(winners)} configs passed)")
    for avg_ev, total_n, lbl, hoA, hoB in sorted(winners, key=lambda x: -x[0])[:n]:
        print(f"    {lbl}")
        print(f"      Win-A: EV {hoA['ev']:+.3f}R  WR {hoA['wr']*100:.1f}%  n={hoA['n']}")
        print(f"      Win-B: EV {hoB['ev']:+.3f}R  WR {hoB['wr']*100:.1f}%  n={hoB['n']}")
        print(f"      avg EV {avg_ev:+.3f}R  total_ho_n={total_n}")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("=" * 72)
    print(f"GBPUSD strategy search — dual-holdout validation")
    print(f"Spread={SPREAD}  BE@1R  {DAYS}-day dataset")
    print("=" * 72)

    b5, b1h, b4h, b1d = await load()

    import time as _t
    fd = _t.strftime("%Y-%m-%d", _t.gmtime(b5[0]["epoch"]))
    ld = _t.strftime("%Y-%m-%d", _t.gmtime(b5[-1]["epoch"]))
    print(f"  Date range: {fd} → {ld}\n")

    # ── 1. BB+RSI Pullback (5m) ───────────────────────────────────────────────
    print("── 1. BB+RSI Pullback (5m) ─────────────────────────────────────────")
    bb_winners = []
    total = 0
    for ema_p, thresh, tp_rr, atr_m, use4h, sess in product(
        [50, 100, 200],
        [40.0, 45.0, 50.0, 55.0],
        [1.5, 2.0, 2.5],
        [1.5, 2.0],
        [False, True],
        ["off", "london", "peak"],
    ):
        total += 1
        cfg = dict(ema_period=ema_p, rsi_thresh=thresh, bb_period=20,
                   tp_rr=tp_rr, atr_mult_sl=atr_m, macro_ema_period=20,
                   session=sess, use_4h_filter=use4h, ema_4h_period=20)
        res = dual_pass_5m(b5, b1h, b4h, b1d, run_bb_rsi, cfg)
        if res:
            hoA, hoB = res
            avg_ev  = (hoA["ev"] + hoB["ev"]) / 2
            total_n = hoA["n"] + hoB["n"]
            lbl = (f"BB+RSI EMA{ema_p} RSI<{thresh} RR{tp_rr} "
                   f"ATR×{atr_m} 4H={'Y' if use4h else 'N'} sess={sess}")
            bb_winners.append((avg_ev, total_n, lbl, hoA, hoB))

    show_top(bb_winners, f"BB+RSI ({total} configs tested)", n=5)

    # ── 2. MTF RSI-Pullback (5m) ──────────────────────────────────────────────
    print("\n── 2. MTF RSI-Pullback (5m) ────────────────────────────────────────")
    rsi_winners = []
    total = 0
    for ema_p, rsi_e, lb, tp_rr, atr_m, use4h, sess in product(
        [100, 200],
        [30.0, 35.0, 40.0],
        [20, 30],
        [1.5, 2.0, 2.5],
        [1.5, 2.0],
        [False, True],
        ["off", "peak"],
    ):
        total += 1
        cfg = dict(ema_period=ema_p, rsi_entry=rsi_e, rsi_lookback=lb,
                   rsi_adaptive=True, tp_rr=tp_rr, atr_mult_sl=atr_m,
                   macro_ema_period=20, session=sess,
                   use_4h_filter=use4h, ema_4h_period=20)
        res = dual_pass_5m(b5, b1h, b4h, b1d, run_mtf_rsi, cfg)
        if res:
            hoA, hoB = res
            avg_ev  = (hoA["ev"] + hoB["ev"]) / 2
            total_n = hoA["n"] + hoB["n"]
            lbl = (f"MTF-RSI EMA{ema_p} entry<{rsi_e} lb{lb} RR{tp_rr} "
                   f"ATR×{atr_m} 4H={'Y' if use4h else 'N'} sess={sess}")
            rsi_winners.append((avg_ev, total_n, lbl, hoA, hoB))

    show_top(rsi_winners, f"MTF RSI-Pullback ({total} configs tested)", n=5)

    # ── 3. MACD 1H ────────────────────────────────────────────────────────────
    print("\n── 3. MACD Histogram Flip (1H) ─────────────────────────────────────")
    macd_winners = []
    total = 0
    for fast, slow, sig_p, ema_p, sess, tp_rr in product(
        [8, 12],
        [21, 26],
        [7, 9],
        [20, 50, 100, 200],
        ["off", "london", "ny", "peak"],
        [1.5, 2.0, 2.5],
    ):
        total += 1
        cfg = dict(macd_fast=fast, macd_slow=slow, macd_signal=sig_p,
                   ema_period=ema_p, tp_rr=tp_rr,
                   atr_mult_sl=2.0, macro_ema_period=20, ema_4h_period=20,
                   session=sess)
        res = dual_pass_1h(b1h, b4h, b1d, cfg)
        if res:
            hoA, hoB = res
            avg_ev  = (hoA["ev"] + hoB["ev"]) / 2
            total_n = hoA["n"] + hoB["n"]
            lbl = (f"MACD({fast},{slow},{sig_p}) EMA{ema_p} "
                   f"RR{tp_rr} sess={sess}")
            macd_winners.append((avg_ev, total_n, lbl, hoA, hoB))

    show_top(macd_winners, f"MACD 1H ({total} configs tested)", n=5)

    # ── Overall best ──────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  OVERALL BEST (top 5 by avg holdout EV across all strategies)")
    print("=" * 72)
    all_w = bb_winners + rsi_winners + macd_winners
    if not all_w:
        print("  No configs passed dual-holdout for any strategy.")
    else:
        for avg_ev, total_n, lbl, hoA, hoB in sorted(all_w, key=lambda x: -x[0])[:5]:
            print(f"\n  {lbl}")
            print(f"    Win-A holdout: EV {hoA['ev']:+.3f}R  WR {hoA['wr']*100:.1f}%  n={hoA['n']}")
            print(f"    Win-B holdout: EV {hoB['ev']:+.3f}R  WR {hoB['wr']*100:.1f}%  n={hoB['n']}")
            print(f"    Avg EV {avg_ev:+.3f}R  total_ho_n={total_n}")

    print("\n" + "=" * 72)
    total_pass = len(bb_winners) + len(rsi_winners) + len(macd_winners)
    print(f"  Passed: BB+RSI {len(bb_winners)} | MTF-RSI {len(rsi_winners)} | "
          f"MACD-1H {len(macd_winners)} | Total {total_pass}")
    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(main())
