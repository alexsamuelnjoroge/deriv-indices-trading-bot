"""
GBPUSD BB+RSI Pullback (5m) — 4-window walk-forward validation.

Best config from dual-holdout sweep:
  A) Best EV  : EMA200 RSI<40 RR1.5 ATR×2.0 4H=Y sess=london  → avg +0.418R n=133
  B) Best vol : EMA200 RSI<50 RR1.5 ATR×2.0 4H=Y sess=london  → avg +0.407R n=154

Expanding-window scheme (identical to all other walk-forwards):
  Window 1: train 0-25%   test 25-50%
  Window 2: train 0-50%   test 50-75%
  Window 3: train 0-75%   test 75-100%  <- closest to optimisation
  Window 4: train 0-87.5% test 87.5-100%

A config must pass all 4 windows (positive holdout EV) to be ROBUST.
"""

import asyncio
import bisect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pro_bot.backtest import simulate_exits, SPREADS, _fetch, CACHE_5M, CACHE_1H, CACHE_1D
from pro_bot.indicators import ema as _ema, rsi as _rsi, atr as _atr, bollinger as _bb
from pro_bot.strategies.base import Signal

SYM    = "frxGBPUSD"
SPREAD = SPREADS[SYM]
DAYS   = 730

WINDOWS = [
    (0.00, 0.25, 0.50),
    (0.00, 0.50, 0.75),
    (0.00, 0.75, 1.00),
    (0.00, 0.875, 1.00),
]
WINDOW_LABELS = [
    "Window 1 (train Q1,    test Q2)   ",
    "Window 2 (train H1,    test H2p1) ",
    "Window 3 (train 75%,   test Q4)   <- closest to optimisation",
    "Window 4 (train 87.5%, test 12.5%)",
]


def _sess(epoch, tz_off=3):
    """London only: 10-14 EAT."""
    h = ((epoch % 86400) // 3600 + tz_off) % 24
    return 10 <= h < 14


def _macro_gate(epoch, epochs_d, ema_d):
    k = bisect.bisect_right(epochs_d, epoch) - 1
    if k < 1 or ema_d[k] is None or ema_d[k - 1] is None:
        return False, False
    up = ema_d[k] > ema_d[k - 1]
    return up, not up


def _4h_ok(epoch, epochs_4h, ema4h, want_up):
    j = bisect.bisect_right(epochs_4h, epoch) - 1
    if j < 1 or ema4h[j] is None or ema4h[j - 1] is None:
        return True
    return (ema4h[j] > ema4h[j - 1]) == want_up


def run_bb_rsi(b5, b1h, b4h, b1d, cfg):
    ema_p    = cfg["ema_period"]
    rsi_t    = cfg["rsi_thresh"]
    tp_rr    = cfg["tp_rr"]
    atr_mult = cfg["atr_mult_sl"]
    macro_p  = cfg["macro_ema_period"]
    ema4h_p  = cfg["ema_4h_period"]

    closes5  = [b["close"] for b in b5]
    closes1h = [b["close"] for b in b1h]

    ema1h  = _ema(closes1h, ema_p)
    rsi5m  = _rsi(closes5, 14)
    bb5m   = _bb(closes5, 20, 2.0)
    atr5m  = _atr(b5, 14)

    ep1h   = [b["epoch"] for b in b1h]
    ema_d  = _ema([b["close"] for b in b1d], macro_p)
    epd    = [b["epoch"] for b in b1d]
    ema4h  = _ema([b["close"] for b in b4h], ema4h_p) if b4h else None
    ep4h   = [b["epoch"] for b in b4h] if b4h else []

    results = []
    warm = max(21, 16)
    for i in range(warm, len(b5)):
        ep = b5[i]["epoch"]
        if not _sess(ep):
            continue

        j = bisect.bisect_right(ep1h, ep) - 1
        if j < 3:
            continue
        en, epv = ema1h[j], ema1h[j - 3]
        if en is None or epv is None:
            continue

        al, ash = _macro_gate(ep, epd, ema_d)
        if not al and not ash:
            continue

        bb   = bb5m[i]
        rn   = rsi5m[i]
        atr_v = atr5m[i]
        if bb is None or rn is None or atr_v is None:
            continue

        up_b, _, lo_b = bb
        sl = max(atr_v * atr_mult, b5[i]["close"] * 0.001)
        tp = sl * tp_rr
        pr = closes5[i]

        up = en > epv
        dn = en < epv
        ob = 100 - rsi_t  # overbought mirror

        if cfg["use_4h_filter"]:
            if up and not _4h_ok(ep, ep4h, ema4h, True):
                continue
            if dn and not _4h_ok(ep, ep4h, ema4h, False):
                continue

        if up and pr <= lo_b and rn < rsi_t and al:
            results.append((i, Signal("BUY",  sl_pips=sl, tp_pips=tp)))
        elif dn and pr >= up_b and rn > ob and ash:
            results.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp)))

    return results


def _split_sigs(all_sigs, c1, c2):
    tr_sigs = [(i,      s) for i, s in all_sigs if i < c1]
    ho_sigs = [(i - c1, s) for i, s in all_sigs if c1 <= i < c2]
    return tr_sigs, ho_sigs


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


def run_wf(b5, b1h, b4h, b1d, cfg, label):
    print(f"  Config: {label}")

    # Compute signals once on full 5m dataset
    all_sigs = run_bb_rsi(b5, b1h, b4h, b1d, cfg)
    n = len(b5)

    passes = 0
    for wi, (_, te_pct, ho_pct) in enumerate(WINDOWS):
        c1 = int(n * te_pct)
        c2 = int(n * ho_pct)

        tr_sigs, ho_sigs = _split_sigs(all_sigs, c1, c2)
        # Also need train from 0 to c1 (use signals where i < c1 — already done above)

        tr_s = stats(sim(b5[:c1], tr_sigs), min_n=10)
        ho_s = stats(sim(b5[c1:c2], ho_sigs), min_n=5)

        def fmt(s):
            if s is None:
                return "(too few trades)"
            v = "PASS v" if s["ev"] > 0 else "FAIL x"
            return (f"n={s['n']:>3}  WR {s['wr']*100:>5.1f}%  "
                    f"EV {s['ev']:>+.4f}R  Net {s['net_r']:>+5.1f}R  [{v}]")

        passed = ho_s is not None and ho_s["ev"] > 0
        if passed:
            passes += 1

        tr_days = (b5[c1 - 1]["epoch"] - b5[0]["epoch"]) // 86400 if c1 > 0 else 0
        ho_days = (b5[c2 - 1]["epoch"] - b5[c1]["epoch"]) // 86400 if c2 > c1 else 0
        marker  = " <-" if wi == 2 else ""
        print(f"    {WINDOW_LABELS[wi]}  [train {tr_days}d / test {ho_days}d]{marker}")
        print(f"      Train  : {fmt(tr_s)}")
        print(f"      Holdout: {fmt(ho_s)}")

    bar     = "#" * passes + "." * (len(WINDOWS) - passes)
    verdict = ("ROBUST"    if passes == 4 else
               "MOSTLY OK" if passes >= 3 else
               "MARGINAL"  if passes >= 2 else
               "OVERFIT -- DO NOT TRADE")
    print(f"\n    [{bar}] {passes}/{len(WINDOWS)} windows positive  -> {verdict}\n")
    return passes


async def main():
    import time as _t
    print("=" * 78)
    print("GBPUSD BB+RSI Pullback (5m) -- 4-window walk-forward validation")
    print(f"Symbol: {SYM}  Spread: {SPREAD}  BE@1R  {DAYS}-day dataset")
    print("=" * 78 + "\n")

    print("  Loading data...", end=" ", flush=True)
    b5  = await _fetch(SYM, 300,   DAYS, CACHE_5M)
    b1h = await _fetch(SYM, 3600,  DAYS, CACHE_1H)
    b4h = await _fetch(SYM, 14400, DAYS, CACHE_1H)
    b1d = await _fetch(SYM, 86400, DAYS, CACHE_1D)
    print(f"{len(b5)} 5m | {len(b1h)} 1H | {len(b4h)} 4H | {len(b1d)} 1D")

    fd = _t.strftime("%Y-%m-%d", _t.gmtime(b5[0]["epoch"]))
    ld = _t.strftime("%Y-%m-%d", _t.gmtime(b5[-1]["epoch"]))
    print(f"  Date range: {fd} -> {ld}\n")
    print("-" * 78)

    configs = [
        (
            "EMA200 RSI<40 RR1.5 ATR×2.0 4H=Y sess=london  <- best EV (+0.418R avg)",
            dict(ema_period=200, rsi_thresh=40.0, tp_rr=1.5, atr_mult_sl=2.0,
                 macro_ema_period=20, ema_4h_period=20, use_4h_filter=True),
        ),
        (
            "EMA200 RSI<50 RR1.5 ATR×2.0 4H=Y sess=london  <- most trades (n=154)",
            dict(ema_period=200, rsi_thresh=50.0, tp_rr=1.5, atr_mult_sl=2.0,
                 macro_ema_period=20, ema_4h_period=20, use_4h_filter=True),
        ),
    ]

    results = []
    for label, cfg in configs:
        passes = run_wf(b5, b1h, b4h, b1d, cfg, label)
        results.append((passes, label))
        print("-" * 78)

    print("\n  SUMMARY")
    print("  " + "-" * 60)
    for passes, label in results:
        bar     = "#" * passes + "." * (4 - passes)
        verdict = ("ROBUST"    if passes == 4 else
                   "MOSTLY OK" if passes >= 3 else
                   "MARGINAL"  if passes >= 2 else "OVERFIT")
        print(f"  [{bar}] {passes}/4  {label[:60]}")
        print(f"         -> {verdict}")

    print("\n" + "=" * 78)
    print("Interpretation:")
    print("  ROBUST (4/4)    -- safe to add as a concurrent strategy on GBPUSD")
    print("  MOSTLY OK (3/4) -- deploy with caution / reduced position size")
    print("  MARGINAL (2/4)  -- do not deploy")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
