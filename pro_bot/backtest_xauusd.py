"""
Focused XAUUSD improvement sweep.

Tests on top of the current best config (EMA200, ADX0, RR2.0, adaptive RSI, PEAK, MACRO20):
  1. Direction split — long-only vs both (gold uptrend = shorts may be dragging EV down)
  2. Macro period — 20 vs 50 vs 100 (longer = more stable direction gate)
  3. RR ratio — 2.0, 2.5, 3.0 (wider TP may improve EV at the cost of WR)
  4. ATR SL multiplier — 1.0, 1.5, 2.0 (tighter SL = higher RR at same TP price)
  5. 4H EMA intermediate filter — adds a quality gate between 1h and daily
  6. RSI adaptive lookback — 30, 50, 100 bars

For each variant: reports train EV + holdout EV side-by-side.
Best configs are those where BOTH are positive.

Usage:
  python pro_bot/backtest_xauusd.py
"""

import asyncio
import bisect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pro_bot.backtest import (
    split_bars, simulate_exits,
    SPREADS,
    _fetch, CACHE_5M, CACHE_1H, CACHE_1D, GRAN_5M, GRAN_1H, GRAN_1D,
)
from pro_bot.indicators import ema as _ema, rsi as _rsi, atr as _atr
from pro_bot.strategies.base import Signal

SYM      = "frxXAUUSD"
SPREAD   = SPREADS[SYM]
DAYS     = 365
GRAN_4H  = 14400
CACHE_4H = CACHE_1H   # reuse same cache dir


# ── Data loading ──────────────────────────────────────────────────────────────

async def load():
    b5  = await _fetch(SYM, GRAN_5M, DAYS, CACHE_5M)
    b1h = await _fetch(SYM, GRAN_1H, DAYS, CACHE_1H)
    b4h = await _fetch(SYM, GRAN_4H, DAYS, CACHE_4H)
    b1d = await _fetch(SYM, GRAN_1D, DAYS, CACHE_1D)
    return b5, b1h, b4h, b1d


def split_all(b5, b1h, b4h, b1d, pct=0.80):
    cut        = int(len(b5) * pct)
    split_epoch = b5[cut]["epoch"]
    train = {"5m": b5[:cut],
             "1h": [b for b in b1h if b["epoch"] <= split_epoch],
             "4h": [b for b in b4h if b["epoch"] <= split_epoch],
             "1d": [b for b in b1d if b["epoch"] <= split_epoch]}
    hold  = {"5m": b5[cut:],
             "1h": [b for b in b1h if b["epoch"] > split_epoch],
             "4h": [b for b in b4h if b["epoch"] > split_epoch],
             "1d": [b for b in b1d if b["epoch"] > split_epoch]}
    return train, hold


# ── Strategy runner ───────────────────────────────────────────────────────────

def run(bars_5m, bars_1h, bars_4h, bars_1d, cfg):
    ema_p        = cfg.get("ema_period",       200)
    slope_b      = cfg.get("slope_bars",         3)
    rsi_p        = cfg.get("rsi_period",        14)
    rsi_entry    = cfg.get("rsi_entry",        35.0)
    tp_rr        = cfg.get("tp_rr",            2.0)
    atr_mult     = cfg.get("atr_mult_sl",      1.5)
    macro_ema_p  = cfg.get("macro_ema_period",  20)
    adaptive     = cfg.get("adaptive",         True)
    lookback     = cfg.get("rsi_lookback",      50)
    direction    = cfg.get("direction",      "both")   # both | long | short
    use_4h       = cfg.get("use_4h_filter",  False)
    ema_4h_p     = cfg.get("ema_4h_period",    50)

    closes_1h = [b["close"] for b in bars_1h]
    closes_5m = [b["close"] for b in bars_5m]
    ema_1h    = _ema(closes_1h, ema_p)
    rsi_5m    = _rsi(closes_5m, rsi_p)
    atr_5m    = _atr(bars_5m, 14)
    epochs_1h = [b["epoch"] for b in bars_1h]

    # Macro: daily EMA slope
    ema_1d    = _ema([b["close"] for b in bars_1d], macro_ema_p)
    epochs_1d = [b["epoch"] for b in bars_1d]

    # Optional 4H EMA filter
    ema_4h    = epochs_4h = None
    if use_4h and bars_4h:
        ema_4h    = _ema([b["close"] for b in bars_4h], ema_4h_p)
        epochs_4h = [b["epoch"] for b in bars_4h]

    results = []
    for i in range(rsi_p + 2, len(bars_5m)):
        epoch = bars_5m[i]["epoch"]

        # Session peak: London 07:00-10:30 UTC + NY 13:00-16:30 UTC
        h_utc = (epoch % 86400) // 3600
        m_min = (epoch % 3600)  // 60
        london = (7 <= h_utc < 10) or (h_utc == 10 and m_min < 30)
        ny     = (13 <= h_utc < 16) or (h_utc == 16 and m_min < 30)
        if not (london or ny):
            continue

        j = bisect.bisect_right(epochs_1h, epoch) - 1
        if j < slope_b or j < 0:
            continue
        e_now, e_prev = ema_1h[j], ema_1h[j - slope_b]
        r_now, r_prev = rsi_5m[i], rsi_5m[i - 1]
        if any(x is None for x in [e_now, e_prev, r_now, r_prev]):
            continue

        # Macro direction
        k = bisect.bisect_right(epochs_1d, epoch) - 1
        if k < 1 or ema_1d[k] is None or ema_1d[k - 1] is None:
            continue
        macro_up    = ema_1d[k] > ema_1d[k - 1]
        allow_long  = macro_up
        allow_short = not macro_up

        # Direction override
        if direction == "long":
            allow_short = False
        elif direction == "short":
            allow_long = False

        # 4H EMA filter: price must be on correct side of 4H EMA
        if ema_4h is not None and epochs_4h is not None:
            j4 = bisect.bisect_right(epochs_4h, epoch) - 1
            if j4 >= 1 and ema_4h[j4] is not None and ema_4h[j4 - 1] is not None:
                ema4_up = ema_4h[j4] > ema_4h[j4 - 1]
                # Only allow long if 4H EMA also trending up; short if trending down
                allow_long  = allow_long  and ema4_up
                allow_short = allow_short and (not ema4_up)

        # Adaptive entry threshold
        if adaptive and i >= lookback:
            recent = sorted(v for v in rsi_5m[i - lookback:i] if v is not None)
            if len(recent) >= max(10, lookback // 2):
                thresh = max(rsi_entry, min(50.0, recent[max(0, int(len(recent) * 0.20) - 1)]))
            else:
                thresh = rsi_entry
        else:
            thresh = rsi_entry
        ob = 100.0 - thresh

        # SL / TP
        atr_v = atr_5m[i]
        sl = (atr_v * atr_mult) if atr_v else bars_5m[i]["close"] * 0.001
        sl = max(sl, bars_5m[i]["close"] * 0.001)
        tp = sl * tp_rr

        up, dn = e_now > e_prev, e_now < e_prev

        if up and r_prev >= thresh > r_now and allow_long:
            results.append((i, Signal("BUY",  sl_pips=sl, tp_pips=tp)))
        elif dn and r_prev <= ob < r_now and allow_short:
            results.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp)))

    return results


# ── Stats helpers ─────────────────────────────────────────────────────────────

def stats(trades, min_n=10):
    closed = [t for t in trades if t.result in ("WIN", "LOSS", "BE")]
    if len(closed) < min_n:
        return None
    wins   = [t for t in closed if t.result == "WIN"]
    losses = [t for t in closed if t.result == "LOSS"]
    be_c   = [t for t in closed if t.result == "BE"]
    buys   = [t for t in closed if t.action == "BUY"]
    sells  = [t for t in closed if t.action == "SELL"]
    n_wr   = len(wins) + len(losses)
    wr     = len(wins) / n_wr if n_wr > 0 else 0
    ev     = sum(t.r_multiple for t in closed) / len(closed)
    net_r  = sum(t.r_multiple for t in closed)
    buy_wr = (sum(1 for t in buys  if t.result=="WIN") /
              max(1, sum(1 for t in buys  if t.result in ("WIN","LOSS"))))
    sel_wr = (sum(1 for t in sells if t.result=="WIN") /
              max(1, sum(1 for t in sells if t.result in ("WIN","LOSS"))))
    return dict(n=len(closed), wins=len(wins), losses=len(losses), be=len(be_c),
                wr=wr, ev=ev, net_r=net_r,
                buys=len(buys), sells=len(sells),
                buy_wr=buy_wr, sel_wr=sel_wr)


def row(label, tr, ho):
    def f(s, highlight=False):
        if s is None:
            return "  (too few)"
        tag = "STRONG" if s["ev"] > 0.05 else "OK" if s["ev"] > 0 else "NEG"
        mark = " <<<" if highlight and s["ev"] > 0.05 else ""
        return (f"n={s['n']:>3} WR:{s['wr']*100:>5.1f}% "
                f"EV:{s['ev']:>+.3f}R Net:{s['net_r']:>+5.1f}R "
                f"B:{s['buys']}({s['buy_wr']*100:.0f}%) "
                f"S:{s['sells']}({s['sel_wr']*100:.0f}%) [{tag}]{mark}")
    ho_good = ho is not None and ho["ev"] > 0
    if tr is None or tr["ev"] <= 0:
        return   # skip losing training configs silently
    print(f"  {label:<52}  train {f(tr)}")
    print(f"  {'':<52}  hold  {f(ho, ho_good)}")
    print()


# ── Sweep ─────────────────────────────────────────────────────────────────────

async def main():
    print("=" * 90)
    print(f"XAUUSD IMPROVEMENT SWEEP — 365 days, 80/20 holdout, spread={SPREAD}")
    print("Base config: EMA200 ADX0 RSI-adaptive PEAK MACRO20 ATR1.5 RR2.0")
    print("=" * 90)

    print("\n  Fetching data...")
    b5, b1h, b4h, b1d = await load()
    train, hold = split_all(b5, b1h, b4h, b1d)
    print(f"  Train {len(train['5m'])} bars | Hold {len(hold['5m'])} bars "
          f"| 4H train {len(train['4h'])} | 4H hold {len(hold['4h'])}\n")

    winners = []

    def test(label, cfg):
        sigs_tr = run(train["5m"], train["1h"], train["4h"], train["1d"], cfg)
        sigs_ho = run(hold["5m"],  hold["1h"],  hold["4h"],  hold["1d"],  cfg)
        tr = stats(simulate_exits(train["5m"], sigs_tr, spread=SPREAD, be_r=1.0), min_n=30)
        ho = stats(simulate_exits(hold["5m"],  sigs_ho, spread=SPREAD, be_r=1.0), min_n=10)
        row(label, tr, ho)
        if tr and ho and tr["ev"] > 0 and ho["ev"] > 0:
            winners.append((label, cfg, tr, ho))

    BASE = dict(ema_period=200, slope_bars=3, rsi_period=14, rsi_entry=35.0,
                tp_rr=2.0, atr_mult_sl=1.5, macro_ema_period=20,
                adaptive=True, rsi_lookback=50, direction="both",
                use_4h_filter=False)

    # ── 1. Direction split ────────────────────────────────────────────────────
    print("── 1. Direction split ──────────────────────────────────────────────\n")
    test("both directions (baseline)",     {**BASE})
    test("LONG only",                      {**BASE, "direction": "long"})
    test("SHORT only",                     {**BASE, "direction": "short"})

    # ── 2. Macro period (more stable direction gate) ──────────────────────────
    print("── 2. Macro EMA period ─────────────────────────────────────────────\n")
    for mp in [10, 20, 50, 100]:
        test(f"macro EMA{mp}",             {**BASE, "macro_ema_period": mp})
    # Long-only + macro variants
    for mp in [50, 100]:
        test(f"LONG only + macro EMA{mp}", {**BASE, "direction": "long",
                                            "macro_ema_period": mp})

    # ── 3. RR ratio ───────────────────────────────────────────────────────────
    print("── 3. RR ratio ─────────────────────────────────────────────────────\n")
    for rr in [1.5, 2.0, 2.5, 3.0]:
        test(f"RR {rr}",                   {**BASE, "tp_rr": rr})
    for rr in [2.5, 3.0]:
        test(f"LONG only + RR {rr}",       {**BASE, "tp_rr": rr, "direction": "long"})

    # ── 4. ATR SL multiplier ──────────────────────────────────────────────────
    print("── 4. ATR SL multiplier ────────────────────────────────────────────\n")
    for am in [0.8, 1.0, 1.2, 1.5, 2.0]:
        test(f"ATR×{am}",                  {**BASE, "atr_mult_sl": am})

    # ── 5. 4H EMA intermediate filter ────────────────────────────────────────
    print("── 5. 4H EMA intermediate filter ───────────────────────────────────\n")
    for p4h in [20, 50, 100]:
        test(f"4H EMA{p4h} filter",        {**BASE, "use_4h_filter": True,
                                            "ema_4h_period": p4h})
        test(f"LONG + 4H EMA{p4h}",        {**BASE, "use_4h_filter": True,
                                            "ema_4h_period": p4h,
                                            "direction": "long"})

    # ── 6. RSI adaptive lookback ──────────────────────────────────────────────
    print("── 6. RSI adaptive lookback ────────────────────────────────────────\n")
    for lb in [20, 30, 50, 100]:
        test(f"adaptive lookback {lb}",    {**BASE, "rsi_lookback": lb})

    # ── 7. Combined best guesses ──────────────────────────────────────────────
    print("── 7. Combined variants ────────────────────────────────────────────\n")
    test("LONG + macro50 + RR2.5",
         {**BASE, "direction": "long", "macro_ema_period": 50, "tp_rr": 2.5})
    test("LONG + macro50 + RR3.0",
         {**BASE, "direction": "long", "macro_ema_period": 50, "tp_rr": 3.0})
    test("LONG + macro50 + RR2.0 + ATR1.0",
         {**BASE, "direction": "long", "macro_ema_period": 50, "tp_rr": 2.0,
          "atr_mult_sl": 1.0})
    test("LONG + 4H50 + macro50 + RR2.5",
         {**BASE, "direction": "long", "use_4h_filter": True, "ema_4h_period": 50,
          "macro_ema_period": 50, "tp_rr": 2.5})
    test("LONG + 4H50 + macro50 + RR3.0",
         {**BASE, "direction": "long", "use_4h_filter": True, "ema_4h_period": 50,
          "macro_ema_period": 50, "tp_rr": 3.0})
    test("LONG + lb30 + macro50 + RR2.5",
         {**BASE, "direction": "long", "rsi_lookback": 30, "macro_ema_period": 50,
          "tp_rr": 2.5})
    test("LONG + lb20 + macro50 + RR2.0",
         {**BASE, "direction": "long", "rsi_lookback": 20, "macro_ema_period": 50})

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 90)
    print(f"WINNERS — {len(winners)} configs where BOTH train + holdout are positive\n")
    if winners:
        best = max(winners, key=lambda x: x[3]["ev"])
        sorted_w = sorted(winners, key=lambda x: -x[3]["ev"])
        for label, cfg, tr, ho in sorted_w[:10]:
            mark = " ★ BEST" if (label, cfg, tr, ho) == best else ""
            print(f"  [{label}]{mark}")
            print(f"    Train  : EV {tr['ev']:+.3f}R  WR {tr['wr']*100:.1f}%  "
                  f"B:{tr['buys']}({tr['buy_wr']*100:.0f}%) S:{tr['sells']}({tr['sel_wr']*100:.0f}%)")
            print(f"    Holdout: EV {ho['ev']:+.3f}R  WR {ho['wr']*100:.1f}%  "
                  f"n={ho['n']}  Net {ho['net_r']:+.1f}R")
            print()

        print(f"  ★ BEST CONFIG:")
        _, cfg, tr, ho = best
        for k, v in cfg.items():
            if v != BASE.get(k):
                print(f"    {k}: {BASE.get(k)} → {v}  (changed)")
            else:
                print(f"    {k}: {v}")
    else:
        print("  None — current config is already at the local optimum.")


if __name__ == "__main__":
    asyncio.run(main())
