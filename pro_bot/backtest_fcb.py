"""
Fractal Compression Breakout (FCB) -- proprietary strategy.

Core thesis:
  Single-timeframe Bollinger squeeze is over-known and arbitraged.
  Multi-scale compression -- when volatility is simultaneously compressed at H1
  relative to its own medium-term average -- signals a coiling spring.
  When price breaks out of this multi-scale silence, the move is structural.

  Implementation uses ATR ratio as the compression measure:
    Compressed = ATR(fast) / ATR(slow) < compression_ratio
    where slow ATR covers ~4x the period of fast ATR

  Edge vs standard squeeze:
  - Standard squeeze fires dozens of times per month (most are noise)
  - Multi-scale compression fires rarely (2-4× per month per symbol)
  - When it fires, the energy release is proportional to the depth of compression

Signal (on 1H bars):
  1. fast_atr = ATR(14) on 1H (14-hour short-term volatility)
  2. slow_atr = ATR(56) on 1H (56-hour = ~4H equivalent volatility)
  3. Compression when fast_atr / slow_atr < compression_ratio (e.g. 0.70)
  4. Breakout when 1H close exceeds N-bar highest high → BUY
     OR 1H close falls below N-bar lowest low → SELL
  5. Both conditions required simultaneously: compressed AND breaking out
  6. SL: slow_atr x atr_mult (uses the slower volatility for SL sizing)
  7. TP: RR x SL
  8. One trade per compression event (reset after entry)
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

def run_fcb(b1h, b1d, cfg):
    comp_ratio  = cfg["compression_ratio"]  # fast/slow ATR < this = compressed
    breakout_n  = cfg["breakout_n"]         # N-bar highest/lowest for breakout detection
    fast_period = cfg.get("fast_atr", 14)   # short ATR period
    slow_period = cfg.get("slow_atr", 56)   # long ATR period (~4H equivalent)
    tp_rr       = cfg["tp_rr"]
    atr_mult    = cfg["atr_mult_sl"]        # SL = slow_atr x this
    use_macro   = cfg.get("macro_filter", False)
    macro_p     = cfg.get("macro_ema_period", 20)
    min_comp_bars = cfg.get("min_comp_bars", 3)  # must be compressed for N bars before breakout

    fast_atr = _atr(b1h, fast_period)
    slow_atr = _atr(b1h, slow_period)

    if use_macro and b1d:
        ema_d = _ema([b["close"] for b in b1d], macro_p)
        ep_d  = [b["epoch"] for b in b1d]
    else:
        ema_d = ep_d = None

    signals      = []
    in_trade     = False
    last_sig_i   = -999   # cooldown: no overlapping signals

    start = max(slow_period + breakout_n + min_comp_bars + 5, 80)

    for i in range(start, len(b1h)):
        bar   = b1h[i]
        epoch = bar["epoch"]

        # Cooldown: skip until previous signal has had time to resolve (48 bars)
        if i - last_sig_i < 48:
            continue

        fa = fast_atr[i]
        sa = slow_atr[i]
        if fa is None or sa is None or sa <= 0:
            continue

        # Macro gate
        allow_long = allow_short = True
        if ema_d is not None:
            k = bisect.bisect_right(ep_d, epoch) - 1
            if k >= 1 and ema_d[k] is not None and ema_d[k - 1] is not None:
                macro_up    = ema_d[k] > ema_d[k - 1]
                allow_long  = macro_up
                allow_short = not macro_up

        # Check compression ratio at current bar and previous min_comp_bars bars
        currently_compressed = all(
            (fast_atr[j] is not None and slow_atr[j] is not None and slow_atr[j] > 0
             and fast_atr[j] / slow_atr[j] < comp_ratio)
            for j in range(i - min_comp_bars, i + 1)
        )
        if not currently_compressed:
            continue

        # Breakout detection: N-bar highest high / lowest low (excluding current bar)
        window = b1h[i - breakout_n: i]
        if len(window) < breakout_n:
            continue
        n_high = max(b["high"] for b in window)
        n_low  = min(b["low"]  for b in window)

        close = bar["close"]

        # SL sized off the slow (structural) ATR
        sl = sa * atr_mult
        tp = sl * tp_rr

        if close > n_high and allow_long:
            signals.append((i, Signal("BUY", sl_pips=sl, tp_pips=tp,
                                      reason="FCB_comp_breakout_up")))
            last_sig_i = i
        elif close < n_low and allow_short:
            signals.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp,
                                      reason="FCB_comp_breakout_dn")))
            last_sig_i = i

    return signals


# ── Simulation / stats ───────────────────────────────────────────────────────

def sim(bars, sigs, spread):
    return simulate_exits(bars, sigs, spread=spread, be_r=1.0, max_hold_bars=72)


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
    all_sigs = run_fcb(b1h, b1d, cfg)
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

    for comp_ratio in [0.60, 0.70, 0.80]:
        for breakout_n in [12, 20, 30]:
            for min_comp_bars in [2, 4]:
                for tp_rr in [2.0, 3.0]:
                    for atr_mult in [1.5, 2.0]:
                        for macro in [False, True]:
                            cfg = dict(
                                compression_ratio=comp_ratio,
                                breakout_n=breakout_n,
                                min_comp_bars=min_comp_bars,
                                fast_atr=14,
                                slow_atr=56,
                                tp_rr=tp_rr,
                                atr_mult_sl=atr_mult,
                                macro_filter=macro,
                                macro_ema_period=20,
                            )
                            label = (
                                f"FCB comp{comp_ratio} "
                                f"N{breakout_n} "
                                f"minC{min_comp_bars} "
                                f"RR{tp_rr} ATR×{atr_mult} "
                                f"{'MACRO' if macro else 'free'}"
                            )
                            sigs = run_fcb(train_b1h, b1d, cfg)
                            if not sigs:
                                continue
                            trades = sim(train_b1h, sigs, spread)
                            s = stats(trades, min_n=8)
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
    print("Fractal Compression Breakout (FCB) -- 1H -- 4-window walk-forward")
    print(f"BE@1R  max_hold=72H  {DAYS}-day dataset")
    print("Thesis: multi-scale ATR compression precedes explosive structural breakout")
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
            print(f"  No profitable FCB configs found for {sym}.\n")
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
            print(f"  No FCB config passed 3+ windows for {sym}.")

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
        print("  No FCB strategy passed 3+ windows on any symbol.")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
