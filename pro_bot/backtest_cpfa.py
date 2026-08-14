"""
CPF Acceleration (CPFA) -- proprietary strategy.

Core thesis:
  CPF (Close Position Factor) = (close - low) / (high - low)
  It measures WHERE in the bar's range price closed:
    CPF = 1.0  → closed at the high (maximum bullish energy)
    CPF = 0.5  → closed at the midpoint (neutral)
    CPF = 0.0  → closed at the low (maximum bearish energy)

  When N consecutive 1H bars ALL close in extreme territory in the same direction,
  it signals that one side has been dominant for an extended period and is
  approaching exhaustion. Professional traders call this "buying/selling into
  strength that's running out of fuel."

  This is structurally different from RSI or momentum oscillators:
  - RSI uses price-to-price changes (close-to-close)
  - CPF uses INTRA-BAR positioning — where price ended up WITHIN the bar
  - A bar can have a small net movement but high CPF (bought all the way up,
    closed at top) — that's a different signal than a bar that barely moved

  Signal types:
    1. STREAK: after N consecutive bars with CPF > high_thresh → SELL exhaustion
               after N consecutive bars with CPF < low_thresh → BUY exhaustion
    2. REVERSAL: after N streak, first bar that breaks the pattern → entry signal

  Entry:
    - Signal fires on bar N+1 after the streak
    - SL: ATR mult beyond the streak's extreme (high for sell, low for buy)
    - TP: RR x SL
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


# ── CPF computation ───────────────────────────────────────────────────────────

def _cpf_series(b1h):
    """Compute Close Position Factor for each bar."""
    cpf = []
    for bar in b1h:
        rng = bar["high"] - bar["low"]
        if rng > 1e-10:
            cpf.append((bar["close"] - bar["low"]) / rng)
        else:
            cpf.append(0.5)
    return cpf


# ── Signal generator ──────────────────────────────────────────────────────────

def run_cpfa(b1h, b1d, cfg):
    min_streak    = cfg["min_streak"]              # consecutive extreme CPF bars needed
    cpf_high      = cfg.get("cpf_high", 0.7)       # threshold for bullish extreme
    cpf_low       = cfg.get("cpf_low",  0.3)       # threshold for bearish extreme
    signal_type   = cfg.get("signal_type", "streak") # "streak" or "reversal"
    cooldown_bars = cfg.get("cooldown_bars", 12)
    tp_rr         = cfg["tp_rr"]
    atr_mult      = cfg["atr_mult_sl"]
    use_macro     = cfg.get("macro_filter", False)
    macro_p       = cfg.get("macro_ema_period", 20)

    cpf    = _cpf_series(b1h)
    atr1h  = _atr(b1h, 14)

    if use_macro and b1d:
        ema_d = _ema([b["close"] for b in b1d], macro_p)
        ep_d  = [b["epoch"] for b in b1d]
    else:
        ema_d = ep_d = None

    signals    = []
    last_sig_i = -999

    bull_streak = 0  # consecutive bars with CPF > cpf_high
    bear_streak = 0  # consecutive bars with CPF < cpf_low

    for i in range(min_streak + 5, len(b1h)):
        bar   = b1h[i]
        epoch = bar["epoch"]

        # Update streaks based on PREVIOUS bar (i-1)
        prev_cpf = cpf[i - 1]
        if prev_cpf > cpf_high:
            bull_streak += 1
            bear_streak  = 0
        elif prev_cpf < cpf_low:
            bear_streak += 1
            bull_streak  = 0
        else:
            bull_streak = 0
            bear_streak = 0

        if i - last_sig_i < cooldown_bars:
            continue

        atr_val = atr1h[i]
        if atr_val is None or atr_val <= 0:
            continue

        allow_long = allow_short = True
        if ema_d is not None:
            k = bisect.bisect_right(ep_d, epoch) - 1
            if k >= 1 and ema_d[k] is not None and ema_d[k - 1] is not None:
                macro_up    = ema_d[k] > ema_d[k - 1]
                allow_long  = macro_up
                allow_short = not macro_up

        sl = atr_val * atr_mult
        tp = sl * tp_rr

        if signal_type == "streak":
            # Fire immediately when streak threshold is reached
            if bull_streak >= min_streak and allow_short:
                signals.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp,
                                          reason=f"CPFA_sell_streak{bull_streak}")))
                last_sig_i  = i
                bull_streak = 0
            elif bear_streak >= min_streak and allow_long:
                signals.append((i, Signal("BUY", sl_pips=sl, tp_pips=tp,
                                          reason=f"CPFA_buy_streak{bear_streak}")))
                last_sig_i  = i
                bear_streak = 0

        elif signal_type == "reversal":
            # Fire on first bar that BREAKS the streak (CPF crosses to opposite side)
            # After bull streak: fire SELL when current bar's CPF < cpf_low (first reversal bar)
            # After bear streak: fire BUY when current bar's CPF > cpf_high
            if bull_streak >= min_streak and cpf[i] < cpf_low and allow_short:
                signals.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp,
                                          reason=f"CPFA_rev_sell_after{bull_streak}")))
                last_sig_i  = i
                bull_streak = 0
            elif bear_streak >= min_streak and cpf[i] > cpf_high and allow_long:
                signals.append((i, Signal("BUY", sl_pips=sl, tp_pips=tp,
                                          reason=f"CPFA_rev_buy_after{bear_streak}")))
                last_sig_i  = i
                bear_streak = 0

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
    all_sigs = run_cpfa(b1h, b1d, cfg)
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

    for signal_type in ["streak", "reversal"]:
        for min_streak in [2, 3, 4, 5]:
            for cpf_h, cpf_l in [(0.7, 0.3), (0.75, 0.25), (0.8, 0.2)]:
                for cooldown in [6, 12, 24]:
                    for tp_rr in [1.5, 2.0, 3.0]:
                        for atr_mult in [1.0, 1.5, 2.0]:
                            for macro in [False, True]:
                                cfg = dict(
                                    min_streak=min_streak,
                                    cpf_high=cpf_h,
                                    cpf_low=cpf_l,
                                    signal_type=signal_type,
                                    cooldown_bars=cooldown,
                                    tp_rr=tp_rr,
                                    atr_mult_sl=atr_mult,
                                    macro_filter=macro,
                                    macro_ema_period=20,
                                )
                                label = (
                                    f"CPFA {signal_type} streak>={min_streak} "
                                    f"thresh[{cpf_l},{cpf_h}] cool{cooldown} "
                                    f"RR{tp_rr} ATRx{atr_mult} "
                                    f"{'MACRO' if macro else 'free'}"
                                )
                                sigs = run_cpfa(train_b1h, b1d, cfg)
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
    print("CPF Acceleration (CPFA) -- 1H -- 4-window walk-forward")
    print(f"BE@1R  max_hold=48H  {DAYS}-day dataset")
    print("Thesis: N consecutive extreme CPF bars = directional exhaustion = fade")
    print("CPF = (close-low)/(high-low) measures intra-bar positioning, not price change")
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
            print(f"  No profitable CPFA configs found for {sym}.\n")
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
            print(f"  No CPFA config passed 3+ windows for {sym}.")

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
        print("  No CPFA strategy passed 3+ windows on any symbol.")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
