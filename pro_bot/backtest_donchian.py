"""
Donchian Channel Breakout (H4 timeframe) -- multi-symbol walk-forward.

Strategy (Turtle Trader methodology adapted to modern market):
  1. On each H4 bar close, compute the N-bar Donchian channel
     (highest high and lowest low of the past N bars, excluding current)
  2. BUY  when H4 close breaks above the N-bar high (new regime high)
  3. SELL when H4 close breaks below the N-bar low  (new regime low)
  4. SL:  ATR(14) × atr_mult below entry
  5. TP:  RR × SL distance
  6. Macro gate: D1 EMA slope must agree with breakout direction
  7. One position per symbol at a time (no pyramid for simplicity)

Edge: When price breaks a multi-day high/low, it signals a genuine regime shift
  beyond the noise of the past N bars -- professional momentum traders and CTAs
  trigger buy orders here, adding fuel to the move.

Timeframe: 4H (16 bars/day) so N=30 ≈ 7 trading days, N=60 ≈ 15 days, N=90 ≈ 22 days.

This is structurally different from all existing strategies:
  - No RSI, no EMA cross, no Bollinger Band
  - Pure price discovery -- breakout above all recent highs is the signal
  - Longer timeframe = fewer trades but larger moves captured
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

def run_donchian(b4h, b1d, cfg):
    n_bars   = cfg["n_bars"]       # Donchian lookback in 4H bars
    tp_rr    = cfg["tp_rr"]
    atr_mult = cfg["atr_mult_sl"]
    use_macro = cfg.get("macro_filter", False)
    macro_p   = cfg.get("macro_ema_period", 20)
    long_only = cfg.get("long_only", False)
    short_only = cfg.get("short_only", False)

    atr4h = _atr(b4h, 14)

    if use_macro and b1d:
        ema_d = _ema([b["close"] for b in b1d], macro_p)
        ep_d  = [b["epoch"] for b in b1d]
    else:
        ema_d = ep_d = None

    signals    = []
    in_trade   = False   # one position at a time (track direction)
    trade_dir  = None

    for i in range(n_bars + 1, len(b4h)):
        bar   = b4h[i]
        epoch = bar["epoch"]

        # Donchian channel: highest high and lowest low of the PREVIOUS n_bars bars
        window = b4h[i - n_bars: i]
        ch_high = max(b["high"] for b in window)
        ch_low  = min(b["low"]  for b in window)

        atr_val = atr4h[i]
        if atr_val is None or atr_val <= 0:
            continue

        # Macro gate
        allow_long = allow_short = True
        if ema_d is not None:
            k = bisect.bisect_right(ep_d, epoch) - 1
            if k < 1 or ema_d[k] is None or ema_d[k - 1] is None:
                continue
            macro_up    = ema_d[k] > ema_d[k - 1]
            allow_long  = macro_up
            allow_short = not macro_up

        if long_only:
            allow_short = False
        if short_only:
            allow_long = False

        close = bar["close"]
        sl    = atr_val * atr_mult
        tp    = sl * tp_rr

        if close > ch_high and allow_long:
            signals.append((i, Signal("BUY",  sl_pips=sl, tp_pips=tp)))
        elif close < ch_low and allow_short:
            signals.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp)))

    return signals


# ── Simulation / stats ───────────────────────────────────────────────────────

def sim(bars, sigs, spread):
    return simulate_exits(bars, sigs, spread=spread, be_r=1.0, max_hold_bars=240)


def stats(trades, min_n=5):
    closed = [t for t in trades if t.result in ("WIN", "LOSS", "BE")]
    if len(closed) < min_n:
        return None
    wins  = sum(1 for t in closed if t.result == "WIN")
    n_wr  = sum(1 for t in closed if t.result in ("WIN", "LOSS"))
    ev    = sum(t.r_multiple for t in closed) / len(closed)
    return dict(n=len(closed), wr=wins / n_wr if n_wr else 0, ev=ev,
                net_r=sum(t.r_multiple for t in closed))


# ── Walk-forward ─────────────────────────────────────────────────────────────

def _split_sigs(all_sigs, c1, c2):
    return ([(i, s) for i, s in all_sigs if i < c1],
            [(i - c1, s) for i, s in all_sigs if c1 <= i < c2])


def run_wf(b4h, b1d, cfg, label, spread, verbose=True):
    all_sigs = run_donchian(b4h, b1d, cfg)
    n        = len(b4h)
    passes   = 0

    if verbose:
        print(f"  Config: {label}  [{len(all_sigs)} total signals]")

    for wi, (_, te_pct, ho_pct) in enumerate(WINDOWS):
        c1 = int(n * te_pct)
        c2 = int(n * ho_pct)

        tr_sigs, ho_sigs = _split_sigs(all_sigs, c1, c2)
        tr_s = stats(sim(b4h[:c1],    tr_sigs, spread), min_n=5)
        ho_s = stats(sim(b4h[c1:c2], ho_sigs, spread), min_n=3)

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
            tr_d = (b4h[c1 - 1]["epoch"] - b4h[0]["epoch"]) // 86400 if c1 > 0 else 0
            ho_d = (b4h[c2 - 1]["epoch"] - b4h[c1]["epoch"]) // 86400 if c2 > c1 else 0
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

def sweep(b4h, b1d, train_end_idx, spread):
    train_b4h = b4h[:train_end_idx]
    results   = []

    for n_bars in [20, 30, 50, 80]:
        for tp_rr in [2.0, 3.0]:
            for atr_mult in [1.5, 2.0]:
                for macro in [False, True]:
                    for direction in ["both", "long", "short"]:
                        cfg = dict(
                            n_bars=n_bars,
                            tp_rr=tp_rr,
                            atr_mult_sl=atr_mult,
                            macro_filter=macro,
                            macro_ema_period=20,
                            long_only=(direction == "long"),
                            short_only=(direction == "short"),
                        )
                        label = (
                            f"DC{n_bars} RR{tp_rr} ATR×{atr_mult} "
                            f"{'MACRO' if macro else 'free'} {direction}"
                        )
                        sigs = run_donchian(train_b4h, b1d, cfg)
                        if not sigs:
                            continue
                        trades = sim(train_b4h, sigs, spread)
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
    print("Donchian Channel Breakout (4H) -- multi-symbol walk-forward validation")
    print(f"BE@1R  max_hold=240 bars (60 days)  {DAYS}-day dataset")
    print("=" * 78 + "\n")

    all_robust = []

    for sym, spread in SYMBOLS_TO_TEST:
        print(f"\n{'═' * 78}")
        print(f"  {sym}  spread={spread}")
        print(f"{'═' * 78}")

        print("  Loading data...", end=" ", flush=True)
        # 4H bars stored in CACHE_1H (same Deriv endpoint, different granularity)
        from pro_bot.backtest import CACHE_1H
        b4h = await _fetch(sym, 14400, DAYS, CACHE_1H)
        b1d = await _fetch(sym, 86400, DAYS, CACHE_1D)
        print(f"{len(b4h)} 4H bars | {len(b1d)} daily bars")

        fd = _t.strftime("%Y-%m-%d", _t.gmtime(b4h[0]["epoch"]))
        ld = _t.strftime("%Y-%m-%d", _t.gmtime(b4h[-1]["epoch"]))
        print(f"  Date range: {fd} -> {ld}")

        train_end = int(len(b4h) * 0.60)
        print(f"\n  Phase 1 -- sweep on training set ({train_end} 4H bars / 60%)...")
        ranked = sweep(b4h, b1d, train_end, spread)

        print(f"  {len(ranked)} configs with positive EV on training data.")
        if not ranked:
            print(f"  No profitable Donchian configs found for {sym}.\n")
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
            passes = run_wf(b4h, b1d, cfg, label, spread, verbose=True)
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
                all_robust.append((sym, passes, ev, label, cfg))
        else:
            print(f"  No Donchian config passed 3+ windows for {sym}.")

    print("\n" + "=" * 78)
    print("OVERALL ROBUST / MOSTLY OK CONFIGS:")
    print("=" * 78)
    if all_robust:
        for sym, passes, ev, label, cfg in sorted(all_robust, key=lambda x: (-x[1], -x[2])):
            bar     = "#" * passes + "." * (4 - passes)
            verdict = "ROBUST" if passes == 4 else "MOSTLY OK"
            print(f"  [{bar}] {passes}/4  {sym}  {label}")
            print(f"         -> {verdict}  train EV {ev:+.4f}R")
            if passes == 4:
                print(f"         Config: {cfg}")
    else:
        print("  No strategy passed 3+ windows on any symbol.")
        print("  Donchian breakout is not viable with this dataset.")

    print("\n" + "=" * 78)
    print("Interpretation:")
    print("  ROBUST (4/4)    -- safe to wire into live bot")
    print("  MOSTLY OK (3/4) -- deploy with caution / half size")
    print("  MARGINAL (2/4)  -- do not deploy")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
