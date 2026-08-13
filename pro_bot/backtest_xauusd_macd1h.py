"""
XAUUSD MACD 1H — 4-window walk-forward validation.

Tests two candidate configs from the dual-holdout sweep:
  A) Best EV   : MACD(12,26,7) EMA100 RR2.5 sess=peak  → avg +0.734R
  B) Best vol  : MACD(8,26,7)  EMA100 RR2.5 sess=off   → n=60, avg +0.405R

Expanding-window scheme (identical to main backtest_walkforward.py):
  Window 1: train 0-25%   test 25-50%
  Window 2: train 0-50%   test 50-75%
  Window 3: train 0-75%   test 75-100%  ← closest to optimisation window
  Window 4: train 0-87.5% test 87.5-100%

A config must pass all 4 windows (positive holdout EV) to be ROBUST.
"""

import asyncio
import bisect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pro_bot.backtest import simulate_exits, SPREADS, _fetch, CACHE_5M, CACHE_1H, CACHE_1D
from pro_bot.indicators import ema as _ema, atr as _atr, macd as _macd
from pro_bot.strategies.base import Signal

SYM    = "frxXAUUSD"
SPREAD = SPREADS[SYM]
DAYS   = 730

GRAN_1H = 3600
GRAN_4H = 14400
GRAN_1D = 86400

WINDOWS = [
    (0.00, 0.25, 0.50),
    (0.00, 0.50, 0.75),
    (0.00, 0.75, 1.00),
    (0.00, 0.875, 1.00),
]
WINDOW_LABELS = [
    "Window 1 (train Q1,    test Q2)   ",
    "Window 2 (train H1,    test H2p1) ",
    "Window 3 (train 75%,   test Q4)   ← closest to optimisation",
    "Window 4 (train 87.5%, test 12.5%)",
]


async def load():
    print(f"  Loading {DAYS}-day dataset for {SYM}...")
    b1h = await _fetch(SYM, GRAN_1H, DAYS, CACHE_1H)
    b4h = await _fetch(SYM, GRAN_4H, DAYS, CACHE_1H)
    b1d = await _fetch(SYM, GRAN_1D, DAYS, CACHE_1D)
    print(f"  {len(b1h)} 1H bars | {len(b4h)} 4H bars | {len(b1d)} 1D bars\n")
    return b1h, b4h, b1d


def make_window(b1h, b4h, b1d, train_end_pct, test_end_pct):
    """Slice by 1H-bar percentages (consistent with the HTF sweep)."""
    n  = len(b1h)
    c1 = int(n * train_end_pct)
    c2 = int(n * test_end_pct)
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


def sim(bars, signals):
    return simulate_exits(bars, signals, spread=SPREAD, be_r=1.0)


def run_wf(b1h, b4h, b1d, cfg, label):
    print(f"  Config: {label}")

    passes = 0
    for wi, (_, train_end_pct, test_end_pct) in enumerate(WINDOWS):
        tr, ho = make_window(b1h, b4h, b1d, train_end_pct, test_end_pct)

        sigs_tr = run_macd(tr["1h"], tr["4h"], tr["1d"], cfg)
        sigs_ho = run_macd(ho["1h"], ho["4h"], ho["1d"], cfg)

        tr_s = stats(sim(tr["1h"], sigs_tr), min_n=10)
        ho_s = stats(sim(ho["1h"], sigs_ho), min_n=5)

        def fmt(s):
            if s is None:
                return "(too few trades)"
            v = "PASS ✓" if s["ev"] > 0 else "FAIL ✗"
            return (f"n={s['n']:>3}  WR {s['wr']*100:>5.1f}%  "
                    f"EV {s['ev']:>+.4f}R  Net {s['net_r']:>+5.1f}R  [{v}]")

        passed = ho_s is not None and ho_s["ev"] > 0
        if passed:
            passes += 1

        tr_days = (tr["1h"][-1]["epoch"] - tr["1h"][0]["epoch"]) // 86400 if tr["1h"] else 0
        ho_days = (ho["1h"][-1]["epoch"] - ho["1h"][0]["epoch"]) // 86400 if ho["1h"] else 0
        marker  = " ←" if wi == 2 else ""
        print(f"    {WINDOW_LABELS[wi]}  [train {tr_days}d / test {ho_days}d]{marker}")
        print(f"      Train  : {fmt(tr_s)}")
        print(f"      Holdout: {fmt(ho_s)}")

    bar     = "█" * passes + "░" * (len(WINDOWS) - passes)
    verdict = ("ROBUST"    if passes == 4 else
               "MOSTLY OK" if passes >= 3 else
               "MARGINAL"  if passes >= 2 else
               "OVERFIT — DO NOT TRADE")
    print(f"\n    [{bar}] {passes}/{len(WINDOWS)} windows positive  → {verdict}\n")
    return passes


async def main():
    print("=" * 78)
    print("XAUUSD MACD 1H — 4-window walk-forward validation")
    print(f"Symbol: {SYM}  Spread: {SPREAD}  BE@1R  {DAYS}-day dataset")
    print("=" * 78 + "\n")

    b1h, b4h, b1d = await load()

    import time as _t
    fd = _t.strftime("%Y-%m-%d", _t.gmtime(b1h[0]["epoch"]))
    ld = _t.strftime("%Y-%m-%d", _t.gmtime(b1h[-1]["epoch"]))
    print(f"  Date range: {fd} → {ld}  ({len(b1h)} 1H bars)\n")
    print("─" * 78)

    configs = [
        (
            "MACD(12,26,7) EMA100 RR2.5 sess=peak  ← best holdout EV (+0.734R avg)",
            dict(macd_fast=12, macd_slow=26, macd_signal=7, ema_period=100,
                 tp_rr=2.5, atr_mult_sl=2.0, macro_ema_period=20, ema_4h_period=20,
                 session="peak"),
        ),
        (
            "MACD(8,26,7)  EMA100 RR2.5 sess=off   ← most trades (n=60 per holdout pair)",
            dict(macd_fast=8, macd_slow=26, macd_signal=7, ema_period=100,
                 tp_rr=2.5, atr_mult_sl=2.0, macro_ema_period=20, ema_4h_period=20,
                 session="off"),
        ),
        (
            "MACD(8,26,7)  EMA100 RR2.0 sess=off   ← high-vol lower-RR variant",
            dict(macd_fast=8, macd_slow=26, macd_signal=7, ema_period=100,
                 tp_rr=2.0, atr_mult_sl=2.0, macro_ema_period=20, ema_4h_period=20,
                 session="off"),
        ),
    ]

    results = []
    for label, cfg in configs:
        passes = run_wf(b1h, b4h, b1d, cfg, label)
        results.append((passes, label))
        print("─" * 78)

    print("\n  SUMMARY")
    print("  " + "─" * 60)
    for passes, label in results:
        bar     = "█" * passes + "░" * (4 - passes)
        verdict = ("ROBUST" if passes == 4 else
                   "MOSTLY OK" if passes >= 3 else
                   "MARGINAL" if passes >= 2 else "OVERFIT")
        print(f"  [{bar}] {passes}/4  {label[:55]}")
        print(f"         → {verdict}")

    print("\n" + "=" * 78)
    print("Interpretation:")
    print("  ROBUST (4/4)    — safe to add as a concurrent strategy on XAUUSD")
    print("  MOSTLY OK (3/4) — deploy with caution / reduced position size")
    print("  MARGINAL (2/4)  — do not deploy")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
