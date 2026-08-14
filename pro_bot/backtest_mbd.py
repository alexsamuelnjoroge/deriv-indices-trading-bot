"""
Multi-Bar Divergence (MBD) -- proprietary strategy.

Core thesis:
  Price action over multiple time windows reveals hidden momentum divergences
  that are invisible when looking at single bars or simple oscillators.

  The insight: institutional distribution or accumulation rarely appears in
  one bar. It happens across many bars as smart money quietly unloads into
  strength or buys into weakness. When this happens, price keeps moving in
  one direction (short-window trend) but the FORCE behind each move diminishes
  (long-window context shows exhaustion).

  Two divergence types we detect:

  1. BEARISH MBD (SELL):
     - Long-window displacement >= min_long_atr  (price is UP significantly)
     - Short-window displacement <= max_short_atr (recent bars: weak or reversing)
     → Price extended over long term but is losing momentum → SELL

  2. BULLISH MBD (BUY):
     - Long-window displacement <= -min_long_atr  (price is DOWN significantly)
     - Short-window displacement >= -max_short_atr (recent bars: bouncing)
     → Price dropped far but downside momentum is fading → BUY

  Metrics (ATR-normalized):
    long_ret  = (close[i] - close[i - long_w])  / atr[i]
    short_ret = (close[i] - close[i - short_w]) / atr[i]

  Different from RSI divergence because:
  - No indicator calculation — pure price/ATR ratio
  - We control both windows directly
  - We measure ACTUAL displacement, not a scaled oscillator
  - The short vs long comparison captures multi-scale momentum shifts

  Different from SME: SME uses session open as anchor.
  MBD uses a rolling N-bar lookback — it fires at any time of day.
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


# ── Signal generator ──────────────────────────────────────────────────────────

def run_mbd(b1h, b1d, cfg):
    long_w        = cfg["long_window"]      # bars to look back for "context" move
    short_w       = cfg["short_window"]     # bars to look back for "recent" move
    min_long_atr  = cfg["min_long_atr"]     # long displacement must exceed this
    max_short_atr = cfg["max_short_atr"]    # recent displacement within ±this (divergence)
    cooldown_bars = cfg.get("cooldown_bars", 12)
    tp_rr         = cfg["tp_rr"]
    atr_mult      = cfg["atr_mult_sl"]
    use_macro     = cfg.get("macro_filter", False)
    macro_p       = cfg.get("macro_ema_period", 20)

    atr1h = _atr(b1h, 14)

    if use_macro and b1d:
        ema_d = _ema([b["close"] for b in b1d], macro_p)
        ep_d  = [b["epoch"] for b in b1d]
    else:
        ema_d = ep_d = None

    signals    = []
    last_sig_i = -999

    for i in range(long_w + 5, len(b1h)):
        bar   = b1h[i]
        epoch = bar["epoch"]

        if i - last_sig_i < cooldown_bars:
            continue

        atr_val = atr1h[i]
        if atr_val is None or atr_val <= 0:
            continue

        # ATR-normalized displacements
        long_ret  = (bar["close"] - b1h[i - long_w]["close"])  / atr_val
        short_ret = (bar["close"] - b1h[i - short_w]["close"]) / atr_val

        # Macro filter
        allow_long = allow_short = True
        if ema_d is not None:
            k = bisect.bisect_right(ep_d, epoch) - 1
            if k >= 1 and ema_d[k] is not None and ema_d[k - 1] is not None:
                macro_up    = ema_d[k] > ema_d[k - 1]
                allow_long  = macro_up
                allow_short = not macro_up

        sl = atr_val * atr_mult
        tp = sl * tp_rr

        # Bearish MBD: price up long-term but short-term momentum fading
        if (long_ret >= min_long_atr and
                abs(short_ret) <= max_short_atr and
                allow_short):
            signals.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp,
                                      reason=f"MBD_sell_L{long_ret:.1f}S{short_ret:.1f}")))
            last_sig_i = i

        # Bullish MBD: price down long-term but short-term momentum fading
        elif (long_ret <= -min_long_atr and
                  abs(short_ret) <= max_short_atr and
                  allow_long):
            signals.append((i, Signal("BUY", sl_pips=sl, tp_pips=tp,
                                      reason=f"MBD_buy_L{long_ret:.1f}S{short_ret:.1f}")))
            last_sig_i = i

    return signals


# ── Simulation / stats ────────────────────────────────────────────────────────

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


# ── Walk-forward ──────────────────────────────────────────────────────────────

def _split_sigs(all_sigs, c1, c2):
    return ([(i, s) for i, s in all_sigs if i < c1],
            [(i - c1, s) for i, s in all_sigs if c1 <= i < c2])


def run_wf(b1h, b1d, cfg, label, spread, verbose=True):
    all_sigs = run_mbd(b1h, b1d, cfg)
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
            tr_d = (b1h[c1-1]["epoch"] - b1h[0]["epoch"]) // 86400 if c1 > 0 else 0
            ho_d = (b1h[c2-1]["epoch"] - b1h[c1]["epoch"]) // 86400 if c2 > c1 else 0
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


# ── Parameter sweep ───────────────────────────────────────────────────────────

def sweep(b1h, b1d, train_end_idx, spread):
    train_b1h = b1h[:train_end_idx]
    results   = []

    for long_w in [6, 12, 24]:
        for short_w in [1, 2, 3]:
            if short_w >= long_w:
                continue
            for min_long_atr in [1.0, 1.5, 2.0, 2.5]:
                for max_short_atr in [0.2, 0.4, 0.6]:
                    for cooldown in [6, 12, 24]:
                        for tp_rr in [1.5, 2.0, 3.0]:
                            for atr_mult in [1.0, 1.5, 2.0]:
                                for macro in [False, True]:
                                    cfg = dict(
                                        long_window=long_w,
                                        short_window=short_w,
                                        min_long_atr=min_long_atr,
                                        max_short_atr=max_short_atr,
                                        cooldown_bars=cooldown,
                                        tp_rr=tp_rr,
                                        atr_mult_sl=atr_mult,
                                        macro_filter=macro,
                                        macro_ema_period=20,
                                    )
                                    label = (
                                        f"MBD L{long_w}/S{short_w} "
                                        f"longMove>={min_long_atr}ATR shortFade<={max_short_atr}ATR "
                                        f"cool{cooldown} RR{tp_rr} ATRx{atr_mult} "
                                        f"{'MACRO' if macro else 'free'}"
                                    )
                                    sigs = run_mbd(train_b1h, b1d, cfg)
                                    if not sigs:
                                        continue
                                    trades = sim(train_b1h, sigs, spread)
                                    s = stats(trades, min_n=8)
                                    if s and s["ev"] > 0:
                                        results.append((s["ev"], s["n"], label, cfg))

    results.sort(key=lambda x: -x[0])
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

SYMBOLS_TO_TEST = [
    ("frxXAUUSD", SPREADS["frxXAUUSD"]),
    ("frxEURUSD", SPREADS["frxEURUSD"]),
    ("frxGBPUSD", SPREADS["frxGBPUSD"]),
    ("frxUSDJPY", SPREADS["frxUSDJPY"]),
]


async def main():
    import time as _t

    print("=" * 78)
    print("Multi-Bar Divergence (MBD) -- 1H -- 4-window walk-forward")
    print(f"BE@1R  max_hold=48H  {DAYS}-day dataset")
    print("Thesis: price extended (long-window) but momentum fading (short-window)")
    print("Bullish/Bearish divergence = price vs ATR-normalized multi-bar velocity")
    print("=" * 78 + "\n")

    all_robust = []

    for sym, spread in SYMBOLS_TO_TEST:
        print(f"\n{'=' * 78}")
        print(f"  {sym}  spread={spread}")
        print(f"{'=' * 78}")

        print("  Loading data...", end=" ", flush=True)
        b1h = await _fetch(sym, 3600,  DAYS, CACHE_1H)
        b1d = await _fetch(sym, 86400, DAYS, CACHE_1D)
        print(f"{len(b1h)} 1H bars | {len(b1d)} daily bars")

        fd = _t.strftime("%Y-%m-%d", _t.gmtime(b1h[0]["epoch"]))
        ld = _t.strftime("%Y-%m-%d", _t.gmtime(b1h[-1]["epoch"]))
        print(f"  Date range: {fd} -> {ld}")

        train_end = int(len(b1h) * 0.60)
        print(f"\n  Phase 1 -- sweep ({train_end} bars / 60%)...")
        ranked = sweep(b1h, b1d, train_end, spread)

        print(f"  {len(ranked)} configs with positive EV on training data.")
        if not ranked:
            print(f"  No profitable MBD configs found for {sym}.\n")
            continue

        print("  Top 5:")
        for ev, n, label, _ in ranked[:5]:
            print(f"    EV {ev:>+.4f}R  n={n:>3}  {label}")

        wf_results = []
        for ev, n, label, cfg in ranked[:5]:
            print(f"\n  {'-' * 70}")
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
            print(f"  No MBD config passed 3+ windows for {sym}.")

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
        print("  No MBD strategy passed 3+ windows on any symbol.")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
