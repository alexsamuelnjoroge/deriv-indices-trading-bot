"""
USDJPY MACD 1H — volume sweep.

Core config fixed: MACD(8,21,9), ATR×2.0, RR 2.0, 4H-EMA20=Y, macro20
Sweep: session × ema_period

Goal: find the relaxation that maximises total holdout trade count
      while both dual-holdout windows remain positive EV.

Output sorted by total_n descending so the highest-frequency
passing configs appear first.

Usage:
  python pro_bot/backtest_usdjpy_volume.py
"""

import asyncio
import bisect
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pro_bot.backtest import simulate_exits, SPREADS, _fetch, CACHE_5M, CACHE_1H, CACHE_1D
from pro_bot.indicators import ema as _ema, atr as _atr, macd as _macd
from pro_bot.strategies.base import Signal

GRAN_1H = 3600
GRAN_4H = 14400
GRAN_1D = 86400
DAYS    = 730
SYM     = "frxUSDJPY"
SPREAD  = SPREADS[SYM]


async def load():
    print("  Loading 2 years of 1H data...")
    b1h = await _fetch(SYM, GRAN_1H, DAYS, CACHE_1H)
    b4h = await _fetch(SYM, GRAN_4H, DAYS, CACHE_1H)
    b1d = await _fetch(SYM, GRAN_1D, DAYS, CACHE_1D)
    print(f"  {len(b1h)} 1h bars  |  {len(b4h)} 4h bars  |  {len(b1d)} 1d bars\n")
    return b1h, b4h, b1d


def make_window(b1h, b4h, b1d, te_pct, ho_pct):
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


def _sess(epoch, sess):
    h = ((epoch % 86400) // 3600 + 3) % 24
    if sess == "london":  return 10 <= h < 14
    if sess == "ny":      return 16 <= h < 20
    if sess == "peak":    return (10 <= h < 14) or (16 <= h < 20)
    if sess == "asia":    return 2  <= h < 8
    return True  # "off" — trade all hours


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


def run_macd(b1h, b4h, b1d, cfg):
    fast_p   = cfg["macd_fast"]
    slow_p   = cfg["macd_slow"]
    sig_p    = cfg["macd_signal"]
    ema_p    = cfg["ema_period"]
    tp_rr    = cfg["tp_rr"]
    atr_mult = cfg["atr_mult_sl"]
    macro_p  = cfg["macro_ema_period"]
    sess     = cfg["session"]
    ema4h_p  = cfg["ema_4h_period"]

    closes    = [b["close"] for b in b1h]
    ema_1h    = _ema(closes, ema_p)
    _, _, hist = _macd(closes, fast_p, slow_p, sig_p)
    atr_1h    = _atr(b1h, 14)
    ema_d     = _ema([b["close"] for b in b1d], macro_p)
    epochs_d  = [b["epoch"] for b in b1d]
    ema4h     = _ema([b["close"] for b in b4h], ema4h_p) if b4h else None
    epochs_4h = [b["epoch"] for b in b4h] if b4h else []

    results = []
    warm = max(slow_p + sig_p + 2, ema_p + 2)
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
            if up  and not _4h_ok(epoch, epochs_4h, ema4h, True):
                continue
            if dn  and not _4h_ok(epoch, epochs_4h, ema4h, False):
                continue
        atr_v = atr_1h[i]
        sl = max((atr_v * atr_mult) if atr_v else 0, b1h[i]["close"] * 0.001)
        tp = sl * tp_rr
        if up and hp < 0 and hn >= 0 and al:
            results.append((i, Signal("BUY",  sl_pips=sl, tp_pips=tp)))
        elif dn and hp > 0 and hn <= 0 and ash:
            results.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp)))
    return results


def st(trades, min_n=5):
    closed = [t for t in trades if t.result in ("WIN", "LOSS", "BE")]
    if len(closed) < min_n:
        return None
    wins  = sum(1 for t in closed if t.result == "WIN")
    n_wr  = sum(1 for t in closed if t.result in ("WIN", "LOSS"))
    wr    = wins / n_wr if n_wr else 0
    ev    = sum(t.r_multiple for t in closed) / len(closed)
    return dict(n=len(closed), wr=wr, ev=ev)


def sim(bars, signals):
    """Use default max_hold_bars=48 to match the original HTF sweep."""
    return simulate_exits(bars, signals, spread=SPREAD, be_r=1.0)


def dual_pass(b1h, b4h, b1d, cfg, min_n_tr=15, min_n_ho=5):
    tr_A, ho_A = make_window(b1h, b4h, b1d, 0.50, 0.75)
    tr_B, ho_B = make_window(b1h, b4h, b1d, 0.75, 1.00)

    sigs_trA = run_macd(tr_A["1h"], tr_A["4h"], tr_A["1d"], cfg)
    sigs_hoA = run_macd(ho_A["1h"], ho_A["4h"], ho_A["1d"], cfg)
    sigs_trB = run_macd(tr_B["1h"], tr_B["4h"], tr_B["1d"], cfg)
    sigs_hoB = run_macd(ho_B["1h"], ho_B["4h"], ho_B["1d"], cfg)

    stA_tr = st(sim(tr_A["1h"], sigs_trA), min_n=min_n_tr)
    stA_ho = st(sim(ho_A["1h"], sigs_hoA), min_n=min_n_ho)
    stB_tr = st(sim(tr_B["1h"], sigs_trB), min_n=min_n_tr)
    stB_ho = st(sim(ho_B["1h"], sigs_hoB), min_n=min_n_ho)

    def ok(s): return s is not None and s["ev"] > 0
    if not (ok(stA_tr) and ok(stA_ho) and ok(stB_tr) and ok(stB_ho)):
        return None
    return stA_ho, stB_ho


async def main():
    print("=" * 70)
    print("USDJPY MACD 1H — Session × EMA-period volume sweep")
    print("Fixed: MACD(8,21,9)  ATR×2.0  RR 2.0  4H-EMA20=ON  macro20")
    print("Goal : maximise total holdout n while both windows stay positive EV")
    print("=" * 70)

    b1h, b4h, b1d = await load()

    SESSIONS   = ["off", "london", "ny", "peak"]
    EMA_PERIODS = [20, 50, 100]

    BASE = dict(
        macd_fast=8, macd_slow=21, macd_signal=9,
        tp_rr=2.0, atr_mult_sl=2.0,
        macro_ema_period=20, ema_4h_period=20,
    )

    winners = []
    all_results = []

    for sess, ema_p in product(SESSIONS, EMA_PERIODS):
        cfg = {**BASE, "session": sess, "ema_period": ema_p}
        label = f"sess={sess:<6}  EMA{ema_p:<3}"

        # Count raw signals across full dataset for reference
        all_sigs = run_macd(b1h, b4h, b1d, cfg)
        total_sigs = len(all_sigs)

        result = dual_pass(b1h, b4h, b1d, cfg)
        all_results.append((label, total_sigs, result))

    # Sort all by total signal count descending
    all_results.sort(key=lambda x: x[1], reverse=True)

    print(f"{'Config':<28}  {'Raw sigs':>8}  {'Win-A (50-75%)':>28}  {'Win-B (75-100%)':>28}")
    print("-" * 100)
    for label, total_sigs, res in all_results:
        if res is None:
            ho_A_str = ho_B_str = "(FAIL)"
        else:
            ho_A_s, ho_B_s = res
            ho_A_str = f"EV {ho_A_s['ev']:+.3f}R WR {ho_A_s['wr']*100:.0f}% n={ho_A_s['n']}"
            ho_B_str = f"EV {ho_B_s['ev']:+.3f}R WR {ho_B_s['wr']*100:.0f}% n={ho_B_s['n']}"
            winners.append((label, total_sigs, ho_A_s, ho_B_s))
        print(f"  {label:<26}  {total_sigs:>8}  {ho_A_str:>28}  {ho_B_str:>28}")

    print()
    if not winners:
        print("No configs passed dual-holdout.")
        return

    print("=" * 70)
    print(f"PASSING CONFIGS sorted by total holdout n  ({len(winners)} / {len(all_results)})")
    print("=" * 70)
    winners.sort(key=lambda x: x[2]["n"] + x[3]["n"], reverse=True)
    for label, total_sigs, hoA, hoB in winners:
        total_ho_n = hoA["n"] + hoB["n"]
        avg_ev = (hoA["ev"] + hoB["ev"]) / 2
        print(f"\n  {label}   total_ho_n={total_ho_n}  avg_ev={avg_ev:+.3f}R")
        print(f"    Win-A: EV {hoA['ev']:+.3f}R  WR {hoA['wr']*100:.1f}%  n={hoA['n']}")
        print(f"    Win-B: EV {hoB['ev']:+.3f}R  WR {hoB['wr']*100:.1f}%  n={hoB['n']}")

    print()
    best = winners[0]
    print(f"  Highest n: {best[0]}  →  {best[2]['n'] + best[3]['n']} total holdout trades")
    print(f"  (current deployed config = sess=peak EMA100 → "
          f"peak_ema100 entry in table above)")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
