"""
London Close Reversal (LCR) -- proprietary strategy.

Core thesis:
  The London session (7:00-16:00 UTC) sets the dominant intraday trend.
  As London traders exit at session close (16:00-17:00 UTC), they take
  profit en masse, which creates a mechanical counter-trend move.
  The 16:00-17:00 window has elevated institutional position-squaring
  activity that drives mean-reversion against the London trend.

Signal (on 1H bars):
  1. Compute net price displacement during the London session (7:00-16:00 UTC)
     using bars in that window.
  2. At 16:00 bar (London close), if London displacement > min_displacement × ATR:
     - Bullish London day → SELL (profit-taking reversal)
     - Bearish London day → BUY
  3. Cooldown: skip if already traded today (one signal per day).
  4. Session gate: only fire on bars at 16:00 UTC (±1h tolerance).
  5. ATR-based SL/TP.

Why this edge exists:
  London close profit-taking is a real, predictable liquidity event.
  Systematic position squaring creates temporary supply/demand imbalance
  that fades within 4-8 hours as NY session prices the real move.
"""

import asyncio
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from pro_bot.backtest import simulate_exits, SPREADS, _fetch, CACHE_1H, CACHE_1D
from pro_bot.strategies.base import Signal

DAYS = 350

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

LONDON_OPEN_UTC  = 7   # 07:00 UTC
LONDON_CLOSE_UTC = 16  # 16:00 UTC signal bar


# ── ATR helper ─────────────────────────────────────────────────────────────────

def _atr(bars, period=14):
    result = [None] * len(bars)
    for i in range(1, len(bars)):
        trs = []
        for j in range(max(1, i - period + 1), i + 1):
            h, l, cp = bars[j]["high"], bars[j]["low"], bars[j - 1]["close"]
            trs.append(max(h - l, abs(h - cp), abs(l - cp)))
        if len(trs) >= period // 2:
            result[i] = sum(trs) / len(trs)
    return result


# ── Signal generator ──────────────────────────────────────────────────────────

def run_lcr(b1h, b1d, cfg):
    min_disp_atr = cfg["min_displacement_atr"]  # London displacement must exceed this × ATR
    tp_rr        = cfg["tp_rr"]
    atr_mult     = cfg["atr_mult_sl"]
    signal_hour  = cfg.get("signal_hour", 16)   # which UTC hour fires the trade
    session_bars = cfg.get("session_bars", 8)    # bars to look back for London displacement

    atr14 = _atr(b1h, 14)
    signals = []

    last_signal_day = -1  # track last day we signalled (to avoid duplicate daily signals)

    for i in range(session_bars + 20, len(b1h)):
        bar   = b1h[i]
        epoch = bar["epoch"]
        hour  = (epoch // 3600) % 24
        day   = epoch // 86400

        if hour != signal_hour:
            continue
        if day == last_signal_day:
            continue

        atr_val = atr14[i]
        if atr_val is None or atr_val <= 0:
            continue

        # Gather bars from the London session up to (not including) the current bar
        london_bars = []
        for j in range(i - 1, max(i - session_bars - 1, 0), -1):
            bj = b1h[j]
            bj_hour = (bj["epoch"] // 3600) % 24
            if bj_hour >= LONDON_OPEN_UTC:
                london_bars.insert(0, bj)
            else:
                break

        if len(london_bars) < 3:
            continue  # not enough London bars

        london_open_price  = london_bars[0]["open"]
        london_close_price = london_bars[-1]["close"]
        displacement       = london_close_price - london_open_price

        if abs(displacement) < min_disp_atr * atr_val:
            continue  # London session didn't move enough

        sl = atr_val * atr_mult
        tp = sl * tp_rr

        if displacement > 0:
            # London was bullish → fade at close → SELL
            signals.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp,
                                      reason="LCR_bull_fade")))
        else:
            # London was bearish → fade at close → BUY
            signals.append((i, Signal("BUY", sl_pips=sl, tp_pips=tp,
                                      reason="LCR_bear_fade")))

        last_signal_day = day

    return signals


# ── Simulation / stats ────────────────────────────────────────────────────────

def sim(bars, sigs, spread):
    return simulate_exits(bars, sigs, spread=spread, be_r=1.0, max_hold_bars=12)


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
    all_sigs = run_lcr(b1h, b1d, cfg)
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


# ── Parameter sweep ───────────────────────────────────────────────────────────

def sweep(b1h, b1d, train_end_idx, spread):
    train_b1h = b1h[:train_end_idx]
    results   = []

    for min_disp in [0.5, 1.0, 1.5, 2.0]:
        for tp_rr in [1.5, 2.0, 3.0]:
            for atr_mult in [1.0, 1.5, 2.0]:
                for session_bars in [6, 8, 10]:
                    for signal_hour in [15, 16]:
                        cfg = dict(
                            min_displacement_atr=min_disp,
                            tp_rr=tp_rr,
                            atr_mult_sl=atr_mult,
                            session_bars=session_bars,
                            signal_hour=signal_hour,
                        )
                        label = (f"LCR disp{min_disp}xATR sess{session_bars}b "
                                 f"hour{signal_hour}UTC RR{tp_rr} ATR×{atr_mult}")
                        sigs = run_lcr(train_b1h, b1d, cfg)
                        if not sigs:
                            continue
                        trades = sim(train_b1h, sigs, spread)
                        s = stats(trades, min_n=10)
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
    print("London Close Reversal (LCR) -- 1H -- 4-window walk-forward")
    print(f"BE@1R  max_hold=12H  {DAYS}-day dataset")
    print("Thesis: London close profit-taking creates mechanical counter-trend at 16:00 UTC")
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
            print(f"  No profitable LCR configs found for {sym}.\n")
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
            print(f"  No LCR config passed 3+ windows for {sym}.")

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
        print("  No LCR strategy passed 3+ windows on any symbol.")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
