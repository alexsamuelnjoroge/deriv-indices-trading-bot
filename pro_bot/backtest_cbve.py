"""
Candle Body Velocity Exhaustion (CBVE) -- proprietary strategy.

Core thesis:
  A trend doesn't reverse instantly -- it decelerates first.
  Healthy trend = consistent or growing candle bodies (momentum accelerating).
  Exhaustion = body size suddenly collapses after a sequence of expanding bodies.
  This body deceleration precedes the reversal candle by 1-2 bars, giving early entry.

Signal (on 1H bars):
  1. Compute rolling N-bar average body size
  2. Detect accel_bars consecutive same-direction candles with GROWING bodies
     (each body strictly larger than the previous)
  3. On the bar AFTER the acceleration sequence, check if body < exhaust_ratio x avg_body
  4. Enter COUNTER-TREND to the acceleration
  5. SL: ATR(14) x atr_mult
  6. TP: RR x SL

Why this edge exists:
  The expanding-body acceleration draws in late momentum traders (FOMO buyers/sellers).
  When the exhaustion candle appears, those late traders are trapped.
  Smart money absorbs their orders and reverses. We enter at the same time as smart money.

This is structurally different from all existing strategies:
  - No RSI, EMA, MACD, or Bollinger Band
  - Purely body-size dynamics -- the derivative of momentum, not momentum itself
  - Counter-trend timing is precise (enters before reversal candle forms)
"""

import asyncio
import bisect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pro_bot.backtest import simulate_exits, SPREADS, _fetch, CACHE_1H, CACHE_1D
from pro_bot.indicators import atr as _atr, ema as _ema
from pro_bot.strategies.base import Signal

DAYS = 730

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


# ── Signal generator ─────────────────────────────────────────────────────────

def run_cbve(b1h, b1d, cfg):
    accel_bars    = cfg["accel_bars"]     # consecutive growing same-direction bodies needed
    exhaust_ratio = cfg["exhaust_ratio"]  # exhaustion body must be < this × avg body
    avg_bars      = cfg["avg_bars"]       # rolling average body period
    tp_rr         = cfg["tp_rr"]
    atr_mult      = cfg["atr_mult_sl"]
    use_macro     = cfg.get("macro_filter", False)
    macro_p       = cfg.get("macro_ema_period", 20)
    min_accel_mult = cfg.get("min_accel_body_mult", 1.0)  # each body must be > this x avg to qualify

    atr1h = _atr(b1h, 14)

    if use_macro and b1d:
        ema_d = _ema([b["close"] for b in b1d], macro_p)
        ep_d  = [b["epoch"] for b in b1d]
    else:
        ema_d = ep_d = None

    signals = []
    start   = max(accel_bars + avg_bars + 5, 30)

    for i in range(start, len(b1h)):
        bar   = b1h[i]
        epoch = bar["epoch"]

        atr_val = atr1h[i]
        if atr_val is None or atr_val <= 0:
            continue

        # Macro gate
        allow_long = allow_short = True
        if ema_d is not None:
            k = bisect.bisect_right(ep_d, epoch) - 1
            if k >= 1 and ema_d[k] is not None and ema_d[k - 1] is not None:
                macro_up    = ema_d[k] > ema_d[k - 1]
                allow_long  = macro_up
                allow_short = not macro_up

        # Rolling average body size over previous avg_bars (not including current bar)
        recent_bodies = [abs(b1h[j]["close"] - b1h[j]["open"])
                         for j in range(i - avg_bars, i)]
        avg_body = sum(recent_bodies) / len(recent_bodies)
        if avg_body <= 0:
            continue

        # Current bar body -- must be the exhaustion bar (very small)
        cur_body = abs(bar["close"] - bar["open"])
        if cur_body >= exhaust_ratio * avg_body:
            continue  # not small enough -- no exhaustion

        # Examine the previous accel_bars bars for a directional acceleration
        prev = b1h[i - accel_bars: i]
        prev_bodies = [abs(b["close"] - b["open"]) for b in prev]

        # All bars must be above the minimum size (not tiny dojis forming the sequence)
        if any(pb < min_accel_mult * avg_body * 0.5 for pb in prev_bodies):
            continue

        # Check strict body expansion: each successive body strictly larger than previous
        expanding = all(prev_bodies[j] > prev_bodies[j - 1]
                        for j in range(1, len(prev_bodies)))
        if not expanding:
            continue

        # Check directional consistency (all bars same direction)
        bull_count = sum(1 for b in prev if b["close"] > b["open"])
        bear_count = sum(1 for b in prev if b["close"] < b["open"])

        sl = atr_val * atr_mult
        tp = sl * tp_rr

        if bull_count == accel_bars and allow_short:
            # Upward acceleration just exhausted → SELL
            signals.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp,
                                      reason="CBVE_bull_exhaust")))
        elif bear_count == accel_bars and allow_long:
            # Downward acceleration just exhausted → BUY
            signals.append((i, Signal("BUY", sl_pips=sl, tp_pips=tp,
                                      reason="CBVE_bear_exhaust")))

    return signals


# ── Simulation / stats ───────────────────────────────────────────────────────

def sim(bars, sigs, spread):
    return simulate_exits(bars, sigs, spread=spread, be_r=1.0, max_hold_bars=48)


def stats(trades, min_n=5):
    closed = [t for t in trades if t.result in ("WIN", "LOSS", "BE")]
    if len(closed) < min_n:
        return None
    wins = sum(1 for t in closed if t.result == "WIN")
    n_wr = sum(1 for t in closed if t.result in ("WIN", "LOSS"))
    ev   = sum(t.r_multiple for t in closed) / len(closed)
    return dict(n=len(closed), wr=wins / n_wr if n_wr else 0, ev=ev,
                net_r=sum(t.r_multiple for t in closed))


# ── Walk-forward ─────────────────────────────────────────────────────────────

def _split_sigs(all_sigs, c1, c2):
    return ([(i, s) for i, s in all_sigs if i < c1],
            [(i - c1, s) for i, s in all_sigs if c1 <= i < c2])


def run_wf(b1h, b1d, cfg, label, spread, verbose=True):
    all_sigs = run_cbve(b1h, b1d, cfg)
    n        = len(b1h)
    passes   = 0

    if verbose:
        print(f"  Config: {label}  [{len(all_sigs)} total signals]")

    for wi, (_, te_pct, ho_pct) in enumerate(WINDOWS):
        c1 = int(n * te_pct)
        c2 = int(n * ho_pct)

        tr_sigs, ho_sigs = _split_sigs(all_sigs, c1, c2)
        tr_s = stats(sim(b1h[:c1],    tr_sigs, spread), min_n=5)
        ho_s = stats(sim(b1h[c1:c2], ho_sigs, spread), min_n=3)

        def fmt(s):
            if s is None:
                return "(too few trades)"
            v = "PASS v" if s["ev"] > 0 else "FAIL x"
            return (f"n={s['n']:>3}  WR {s['wr']*100:>5.1f}%  "
                    f"EV {s['ev']:>+.4f}R  Net {s['net_r']:>+5.1f}R  [{v}]")

        passed = ho_s is not None and ho_s["ev"] > 0
        if passed:
            passes += 1

        if verbose:
            tr_d = (b1h[c1 - 1]["epoch"] - b1h[0]["epoch"]) // 86400 if c1 > 0 else 0
            ho_d = (b1h[c2 - 1]["epoch"] - b1h[c1]["epoch"]) // 86400 if c2 > c1 else 0
            mk   = " <-" if wi == 2 else ""
            print(f"    {WINDOW_LABELS[wi]}  [train {tr_d}d / test {ho_d}d]{mk}")
            print(f"      Train  : {fmt(tr_s)}")
            print(f"      Holdout: {fmt(ho_s)}")

    if verbose:
        bar     = "#" * passes + "." * (len(WINDOWS) - passes)
        verdict = ("ROBUST"    if passes == 4 else
                   "MOSTLY OK" if passes >= 3 else
                   "MARGINAL"  if passes >= 2 else
                   "OVERFIT -- DO NOT TRADE")
        print(f"\n    [{bar}] {passes}/{len(WINDOWS)} windows positive  -> {verdict}\n")

    return passes


# ── Parameter sweep ──────────────────────────────────────────────────────────

def sweep(b1h, b1d, train_end_idx, spread):
    train_b1h = b1h[:train_end_idx]
    results   = []

    for accel_bars in [2, 3]:
        for exhaust_ratio in [0.15, 0.25, 0.35]:
            for avg_bars in [5, 8]:
                for tp_rr in [1.5, 2.0, 3.0]:
                    for atr_mult in [1.0, 1.5, 2.0]:
                        for macro in [False, True]:
                            cfg = dict(
                                accel_bars=accel_bars,
                                exhaust_ratio=exhaust_ratio,
                                avg_bars=avg_bars,
                                tp_rr=tp_rr,
                                atr_mult_sl=atr_mult,
                                macro_filter=macro,
                                macro_ema_period=20,
                                min_accel_body_mult=1.0,
                            )
                            label = (
                                f"CBVE accel{accel_bars} "
                                f"exhaust{exhaust_ratio} "
                                f"avg{avg_bars} "
                                f"RR{tp_rr} ATR×{atr_mult} "
                                f"{'MACRO' if macro else 'free'}"
                            )
                            sigs = run_cbve(train_b1h, b1d, cfg)
                            if not sigs:
                                continue
                            trades = sim(train_b1h, sigs, spread)
                            s = stats(trades, min_n=10)
                            if s and s["ev"] > 0:
                                results.append((s["ev"], s["n"], label, cfg))

    results.sort(key=lambda x: -x[0])
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

SYMBOLS_TO_TEST = [
    ("frxXAUUSD", SPREADS["frxXAUUSD"]),
    ("frxEURUSD", SPREADS["frxEURUSD"]),
    ("frxGBPUSD", SPREADS["frxGBPUSD"]),
    ("frxUSDJPY", SPREADS["frxUSDJPY"]),
]


async def main():
    import time as _t

    print("=" * 78)
    print("Candle Body Velocity Exhaustion (CBVE) -- 1H -- 4-window walk-forward")
    print(f"BE@1R  max_hold=48H  {DAYS}-day dataset")
    print("Thesis: momentum deceleration (body shrink) precedes reversal by 1-2 bars")
    print("=" * 78 + "\n")

    all_robust = []

    for sym, spread in SYMBOLS_TO_TEST:
        print(f"\n{'═' * 78}")
        print(f"  {sym}  spread={spread}")
        print(f"{'═' * 78}")

        print("  Loading data...", end=" ", flush=True)
        b1h = await _fetch(sym, 3600,  DAYS, CACHE_1H)
        b1d = await _fetch(sym, 86400, DAYS, CACHE_1D)
        print(f"{len(b1h)} 1H bars | {len(b1d)} daily bars")

        fd = _t.strftime("%Y-%m-%d", _t.gmtime(b1h[0]["epoch"]))
        ld = _t.strftime("%Y-%m-%d", _t.gmtime(b1h[-1]["epoch"]))
        print(f"  Date range: {fd} -> {ld}")

        train_end = int(len(b1h) * 0.60)
        print(f"\n  Phase 1 -- sweep on training set ({train_end} 1H bars / 60%)...")
        ranked = sweep(b1h, b1d, train_end, spread)

        print(f"  {len(ranked)} configs with positive EV on training data.")
        if not ranked:
            print(f"  No profitable CBVE configs found for {sym}.\n")
            continue

        print("  Top 5 training configs:")
        for ev, n, label, _ in ranked[:5]:
            print(f"    EV {ev:>+.4f}R  n={n:>3}  {label}")

        top5 = ranked[:5]
        print(f"\n  Phase 2 -- 4-window walk-forward on top 5 configs")
        print(f"  {'-' * 60}")

        wf_results = []
        for ev, n, label, cfg in top5:
            print(f"\n  {'─' * 70}")
            passes = run_wf(b1h, b1d, cfg, label, spread, verbose=True)
            wf_results.append((passes, ev, label, cfg))

        robust = [(p, ev, l, c) for p, ev, l, c in wf_results if p >= 3]
        robust.sort(key=lambda x: (-x[0], -x[1]))

        print(f"\n  {sym} SUMMARY:")
        if robust:
            for passes, ev, label, cfg in robust:
                bar     = "#" * passes + "." * (4 - passes)
                verdict = "ROBUST" if passes == 4 else "MOSTLY OK"
                print(f"  [{bar}] {passes}/4  {label}")
                print(f"         -> {verdict}  (train EV {ev:+.4f}R)")
                if passes == 4:
                    print(f"  BEST CONFIG: {cfg}")
                all_robust.append((sym, passes, ev, label, cfg))
        else:
            print(f"  No CBVE config passed 3+ windows for {sym}.")

    print("\n" + "=" * 78)
    print("OVERALL ROBUST / MOSTLY OK:")
    print("=" * 78)
    if all_robust:
        for sym, passes, ev, label, cfg in sorted(all_robust, key=lambda x: (-x[1], -x[2])):
            bar     = "#" * passes + "." * (4 - passes)
            verdict = "ROBUST" if passes == 4 else "MOSTLY OK"
            print(f"  [{bar}] {passes}/4  {sym}  {label}")
            print(f"         -> {verdict}  train EV {ev:+.4f}R")
    else:
        print("  No CBVE strategy passed 3+ windows on any symbol.")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
