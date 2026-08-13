"""
MACD 1H dual-holdout sweep across all three symbols.

Tests whether the MACD histogram flip on 1H bars generalises to
XAUUSD and EURUSD, not just USDJPY.

Fixed core: ATR×2.0, 4H-EMA20=ON, macro20, BE@1R
Sweep: MACD(fast, slow, signal) × ema_period × session × tp_rr

Dual-holdout: BOTH Windows A (50-75%) and B (75-100%) must show
positive EV in training AND holdout.

Usage:
  python pro_bot/backtest_macd1h_all.py
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


async def load(sym):
    b1h = await _fetch(sym, GRAN_1H, DAYS, CACHE_1H)
    b4h = await _fetch(sym, GRAN_4H, DAYS, CACHE_1H)
    b1d = await _fetch(sym, GRAN_1D, DAYS, CACHE_1D)
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


def _sess(epoch, sess, tz_off=3):
    h = ((epoch % 86400) // 3600 + tz_off) % 24
    if sess == "london":  return 10 <= h < 14
    if sess == "ny":      return 16 <= h < 20
    if sess == "peak":    return (10 <= h < 14) or (16 <= h < 20)
    return True  # "off"


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


def sim(bars, signals, spread):
    return simulate_exits(bars, signals, spread=spread, be_r=1.0)


def dual_pass(b1h, b4h, b1d, cfg, spread, min_n_tr=15, min_n_ho=5):
    tr_A, ho_A = make_window(b1h, b4h, b1d, 0.50, 0.75)
    tr_B, ho_B = make_window(b1h, b4h, b1d, 0.75, 1.00)

    def run_and_sim(w):
        sigs = run_macd(w["1h"], w["4h"], w["1d"], cfg)
        return sim(w["1h"], sigs, spread)

    stA_tr = st(run_and_sim(tr_A), min_n=min_n_tr)
    stA_ho = st(run_and_sim(ho_A), min_n=min_n_ho)
    stB_tr = st(run_and_sim(tr_B), min_n=min_n_tr)
    stB_ho = st(run_and_sim(ho_B), min_n=min_n_ho)

    def ok(s): return s is not None and s["ev"] > 0
    if not (ok(stA_tr) and ok(stA_ho) and ok(stB_tr) and ok(stB_ho)):
        return None
    return stA_ho, stB_ho


async def sweep_symbol(sym, label):
    spread = SPREADS[sym]
    print(f"\n{'─'*70}")
    print(f"  {label}  spread={spread}")
    print(f"{'─'*70}")
    print(f"  Loading data...", end=" ", flush=True)
    b1h, b4h, b1d = await load(sym)
    print(f"{len(b1h)} 1H bars | {len(b4h)} 4H bars | {len(b1d)} 1D bars")

    winners = []

    for fast, slow, sig_p, ema_p, sess, tp_rr in product(
        [8, 12],
        [21, 26],
        [7, 9],
        [20, 50, 100, 200],
        ["off", "london", "ny", "peak"],
        [1.5, 2.0, 2.5],
    ):
        cfg = dict(
            macd_fast=fast, macd_slow=slow, macd_signal=sig_p,
            ema_period=ema_p, tp_rr=tp_rr,
            atr_mult_sl=2.0, macro_ema_period=20, ema_4h_period=20,
            session=sess,
        )
        res = dual_pass(b1h, b4h, b1d, cfg, spread)
        if res:
            hoA, hoB = res
            avg_ev = (hoA["ev"] + hoB["ev"]) / 2
            total_n = hoA["n"] + hoB["n"]
            label_cfg = (f"MACD({fast},{slow},{sig_p}) EMA{ema_p} "
                         f"RR{tp_rr} sess={sess}")
            winners.append((avg_ev, total_n, label_cfg, hoA, hoB))

    total_configs = 2 * 2 * 2 * 4 * 4 * 3  # = 384
    print(f"\n  {len(winners)} / {total_configs} configs passed dual-holdout\n")

    if not winners:
        print("  No configs passed.")
        return

    # Show top 5 by avg holdout EV, then top 5 by total n
    print(f"  TOP 5 by avg holdout EV:")
    for avg_ev, total_n, lbl, hoA, hoB in sorted(winners, key=lambda x: -x[0])[:5]:
        print(f"    {lbl}")
        print(f"      Win-A holdout: EV {hoA['ev']:+.3f}R  WR {hoA['wr']*100:.1f}%  n={hoA['n']}")
        print(f"      Win-B holdout: EV {hoB['ev']:+.3f}R  WR {hoB['wr']*100:.1f}%  n={hoB['n']}")
        print(f"      Avg EV {avg_ev:+.3f}R  total_ho_n={total_n}")

    print(f"\n  TOP 5 by total holdout n (most trades):")
    for avg_ev, total_n, lbl, hoA, hoB in sorted(winners, key=lambda x: -x[1])[:5]:
        print(f"    {lbl}")
        print(f"      Win-A holdout: EV {hoA['ev']:+.3f}R  WR {hoA['wr']*100:.1f}%  n={hoA['n']}")
        print(f"      Win-B holdout: EV {hoB['ev']:+.3f}R  WR {hoB['wr']*100:.1f}%  n={hoB['n']}")
        print(f"      Avg EV {avg_ev:+.3f}R  total_ho_n={total_n}")


async def main():
    print("=" * 70)
    print("MACD 1H dual-holdout sweep — XAUUSD, EURUSD, USDJPY")
    print("MACD(fast,slow,sig) × EMA × session × RR  |  384 configs per symbol")
    print("Fixed: ATR×2.0  4H-EMA20=ON  macro20  BE@1R")
    print("=" * 70)

    await sweep_symbol("frxXAUUSD", "XAUUSD (Gold)")
    await sweep_symbol("frxEURUSD", "EURUSD")
    await sweep_symbol("frxUSDJPY", "USDJPY")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
