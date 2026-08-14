"""
Focused USDJPY BB+RSI sweep — narrow around candidates from the full sweep.

The wide sweep found BB+RSI is the only viable strategy.
Best holdout EV seen: +0.157R n=86, +0.154R n=267.
This script prints full config details for every passing config.

Usage:
  python pro_bot/backtest_usdjpy_focus.py
"""

import asyncio
import bisect
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pro_bot.backtest import simulate_exits, SPREADS, _fetch, CACHE_5M, CACHE_1H, CACHE_1D
from pro_bot.indicators import ema as _ema, rsi as _rsi, atr as _atr, bollinger as _bb

GRAN_5M = 300
GRAN_1H = 3600
GRAN_4H = 14400
GRAN_1D = 86400
DAYS    = 365
SYM     = "frxUSDJPY"
SPREAD  = SPREADS[SYM]


async def load():
    b5  = await _fetch(SYM, GRAN_5M, DAYS, CACHE_5M)
    b1h = await _fetch(SYM, GRAN_1H, DAYS, CACHE_1H)
    b4h = await _fetch(SYM, GRAN_4H, DAYS, CACHE_5M)
    b1d = await _fetch(SYM, GRAN_1D, DAYS, CACHE_1D)
    return b5, b1h, b4h, b1d


def split(b5, b1h, b4h, b1d, pct=0.80):
    cut   = int(len(b5) * pct)
    epoch = b5[cut]["epoch"]
    tr = {"5m": b5[:cut],  "1h": [b for b in b1h if b["epoch"] <= epoch],
          "4h": [b for b in b4h if b["epoch"] <= epoch],
          "1d": [b for b in b1d if b["epoch"] <= epoch]}
    ho = {"5m": b5[cut:],  "1h": [b for b in b1h if b["epoch"] > epoch],
          "4h": [b for b in b4h if b["epoch"] > epoch],
          "1d": [b for b in b1d if b["epoch"] > epoch]}
    return tr, ho


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
    return True


def _macro(bars_1d, period, epoch, epochs_d, ema_d):
    k = bisect.bisect_right(epochs_d, epoch) - 1
    if k < 1 or ema_d[k] is None or ema_d[k-1] is None:
        return False, False
    up = ema_d[k] > ema_d[k-1]
    return up, not up


def _4h_ok(bars_4h, epoch, epochs_4h, ema4h, want_up):
    if ema4h is None:
        return True
    j = bisect.bisect_right(epochs_4h, epoch) - 1
    if j < 1 or ema4h[j] is None or ema4h[j-1] is None:
        return True
    up4 = ema4h[j] > ema4h[j-1]
    return up4 if want_up else not up4


def run_bb_rsi(b5, b1h, b4h, b1d, cfg):
    from pro_bot.strategies.base import Signal
    ema_p    = cfg["ema_period"]
    bb_p     = cfg["bb_period"]
    bb_std   = cfg.get("bb_std", 2.0)
    rsi_p    = cfg.get("rsi_period", 14)
    thresh   = cfg["rsi_thresh"]
    tp_rr    = cfg["tp_rr"]
    atr_mult = cfg["atr_mult_sl"]
    macro_p  = cfg.get("macro_ema_period", 20)
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
    ema_d  = _ema([b["close"] for b in b1d], macro_p)
    epochs_d = [b["epoch"] for b in b1d]

    ema4h = epochs_4h = None
    if use_4h and b4h:
        ema4h    = _ema([b["close"] for b in b4h], ema_4h_p)
        epochs_4h = [b["epoch"] for b in b4h]

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
        al, ash = _macro(b1d, macro_p, epoch, epochs_d, ema_d)
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
            if up  and not _4h_ok(b4h, epoch, epochs_4h, ema4h, True):
                continue
            if dn  and not _4h_ok(b4h, epoch, epochs_4h, ema4h, False):
                continue
        if up and price <= lo_b and r_now < thresh and al:
            results.append((i, Signal("BUY",  sl_pips=sl, tp_pips=tp)))
        elif dn and price >= up_b and r_now > ob and ash:
            results.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp)))
    return results


def stats(trades, min_n=8):
    closed = [t for t in trades if t.result in ("WIN", "LOSS", "BE")]
    if len(closed) < min_n:
        return None
    wins  = sum(1 for t in closed if t.result == "WIN")
    n_wr  = sum(1 for t in closed if t.result in ("WIN", "LOSS"))
    wr    = wins / n_wr if n_wr else 0
    ev    = sum(t.r_multiple for t in closed) / len(closed)
    net_r = sum(t.r_multiple for t in closed)
    buys  = [t for t in closed if t.action == "BUY"]
    sells = [t for t in closed if t.action == "SELL"]
    bwr   = sum(1 for t in buys  if t.result=="WIN") / max(1, sum(1 for t in buys  if t.result in ("WIN","LOSS")))
    swr   = sum(1 for t in sells if t.result=="WIN") / max(1, sum(1 for t in sells if t.result in ("WIN","LOSS")))
    return dict(n=len(closed), wr=wr, ev=ev, net_r=net_r,
                buys=len(buys), sells=len(sells), buy_wr=bwr, sel_wr=swr)


async def main():
    print("=" * 78)
    print("USDJPY — Focused BB+RSI sweep (full config labels)")
    print(f"Spread={SPREAD}  BE@1R  80/20 holdout  365 days")
    print("=" * 78)

    b5, b1h, b4h, b1d = await load()
    train, hold = split(b5, b1h, b4h, b1d)
    print(f"  Train {len(train['5m'])} bars | Hold {len(hold['5m'])} bars\n")

    winners = []
    total = 0
    for ema_p, bb_p, thresh, tp_rr, atr_m, use4h, ema_4h_p, sess in product(
        [50, 100, 200],
        [20, 30],
        [40.0, 45.0, 50.0, 55.0, 60.0],
        [1.5, 2.0, 2.5],
        [1.5, 2.0],
        [False, True],
        [20, 50],
        ["off", "london", "ny", "peak"],
    ):
        if not use4h and ema_4h_p != 20:  # deduplicate — ema_4h_p irrelevant when no filter
            continue
        total += 1
        cfg = dict(ema_period=ema_p, bb_period=bb_p, bb_std=2.0, rsi_period=14,
                   rsi_thresh=thresh, tp_rr=tp_rr, atr_mult_sl=atr_m,
                   macro_ema_period=20, session=sess,
                   use_4h_filter=use4h, ema_4h_period=ema_4h_p, slope_bars=3)
        sigs_tr = run_bb_rsi(train["5m"], train["1h"], train["4h"], train["1d"], cfg)
        sigs_ho = run_bb_rsi(hold["5m"],  hold["1h"],  hold["4h"],  hold["1d"],  cfg)
        tr = stats(simulate_exits(train["5m"], sigs_tr, spread=SPREAD, be_r=1.0), min_n=30)
        ho = stats(simulate_exits(hold["5m"],  sigs_ho, spread=SPREAD, be_r=1.0), min_n=8)
        if tr and tr["ev"] > 0 and ho and ho["ev"] > 0:
            label = (f"EMA{ema_p} BB({bb_p}) RSI<{thresh} RR{tp_rr} "
                     f"ATR×{atr_m} 4H={'EMA'+str(ema_4h_p) if use4h else 'off'} sess={sess}")
            winners.append((label, tr, ho))

    print(f"  Tested {total} configs | {len(winners)} passed train+holdout\n")
    print("── All passing configs, sorted by holdout EV ──────────────────────────\n")
    for label, tr, ho in sorted(winners, key=lambda x: -x[2]["ev"]):
        print(f"  {label}")
        print(f"    Train  : EV {tr['ev']:>+.4f}R  WR {tr['wr']*100:>5.1f}%  n={tr['n']}")
        print(f"    Holdout: EV {ho['ev']:>+.4f}R  WR {ho['wr']*100:>5.1f}%  n={ho['n']}"
              f"  Net {ho['net_r']:>+.1f}R"
              f"  B:{ho['buys']}({ho['buy_wr']*100:.0f}%) S:{ho['sells']}({ho['sel_wr']*100:.0f}%)")
        print()

    if not winners:
        print("  No config passed both train and holdout.")
        print("  Recommendation: disable USDJPY until regime normalises.\n")
    else:
        best = max(winners, key=lambda x: x[2]["ev"])
        print("=" * 78)
        print(f"  BEST: {best[0]}")
        print(f"    Holdout EV {best[2]['ev']:+.4f}R  WR {best[2]['wr']*100:.1f}%  n={best[2]['n']}")


if __name__ == "__main__":
    asyncio.run(main())
