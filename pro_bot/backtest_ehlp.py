"""
Equal Highs/Lows Liquidity Hunt (EHLP) -- proprietary strategy.

Core thesis:
  Retail traders place stop-losses just above obvious equal highs or below
  equal lows. Smart money engineers a "sweep" of those stops -- briefly
  poking above/below the equal level -- then reverses sharply once the
  liquidity has been captured. The sweep bar itself is the entry signal.

Signal (on 1H bars):
  1. Scan the previous N bars for "equal highs": two or more bar highs
     within `equality_tol × ATR` of each other.
  2. Current bar's HIGH exceeds the equal-high level (the sweep).
  3. Current bar CLOSES back below the equal-high level (reversal close).
  4. → SELL signal (liquidity grabbed above equal highs, reversal follows).
  5. Mirror logic for equal lows → BUY.
  6. SL: above the sweep high + small buffer. TP: RR × SL.

Why this edge exists:
  Equal highs/lows are visible on every retail charting platform.
  Stops cluster there by construction. Institutional order flow
  sweeps them to fill large orders in the opposite direction.
  The sweep-and-reverse is mechanically predictable because the
  stop cluster is a known liquidity pool.
"""

import asyncio
import sys
from pathlib import Path

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


def _find_equal_level(prices, tol, min_count=2):
    """
    Find a cluster of prices within `tol` of each other.
    Returns the average level if >= min_count prices cluster, else None.
    """
    if not prices:
        return None
    # Sort and find the densest cluster
    sorted_p = sorted(prices)
    best_cluster = []
    for i in range(len(sorted_p)):
        cluster = [sorted_p[i]]
        for j in range(i + 1, len(sorted_p)):
            if sorted_p[j] - sorted_p[i] <= tol:
                cluster.append(sorted_p[j])
            else:
                break
        if len(cluster) > len(best_cluster):
            best_cluster = cluster
    if len(best_cluster) >= min_count:
        return sum(best_cluster) / len(best_cluster)
    return None


# ── Signal generator ──────────────────────────────────────────────────────────

def run_ehlp(b1h, b1d, cfg):
    lookback      = cfg["lookback"]           # bars to scan for equal levels
    equality_tol  = cfg["equality_tol_atr"]   # tolerance for "equal" in ATR multiples
    min_count     = cfg.get("min_count", 2)   # minimum bars needed to form equal level
    sl_buffer     = cfg.get("sl_buffer", 0.3) # additional ATR buffer on top of sweep for SL
    tp_rr         = cfg["tp_rr"]
    atr_mult      = cfg["atr_mult_sl"]

    atr14 = _atr(b1h, 14)
    signals = []
    start = lookback + 20

    for i in range(start, len(b1h)):
        bar   = b1h[i]
        atr_val = atr14[i]
        if atr_val is None or atr_val <= 0:
            continue

        tol = equality_tol * atr_val
        prior = b1h[i - lookback: i]

        prior_highs = [b["high"] for b in prior]
        prior_lows  = [b["low"]  for b in prior]

        # Check for equal highs sweep
        eq_high = _find_equal_level(prior_highs, tol, min_count)
        if eq_high is not None:
            if bar["high"] > eq_high and bar["close"] < eq_high:
                # Swept above equal highs and closed back below → SELL
                sl = (bar["high"] - bar["close"]) + sl_buffer * atr_val
                sl = max(sl, atr_val * atr_mult)
                tp = sl * tp_rr
                signals.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp,
                                          reason="EHLP_high_sweep")))
                continue  # one signal per bar

        # Check for equal lows sweep
        eq_low = _find_equal_level(prior_lows, tol, min_count)
        if eq_low is not None:
            if bar["low"] < eq_low and bar["close"] > eq_low:
                # Swept below equal lows and closed back above → BUY
                sl = (bar["close"] - bar["low"]) + sl_buffer * atr_val
                sl = max(sl, atr_val * atr_mult)
                tp = sl * tp_rr
                signals.append((i, Signal("BUY", sl_pips=sl, tp_pips=tp,
                                          reason="EHLP_low_sweep")))

    return signals


# ── Simulation / stats ────────────────────────────────────────────────────────

def sim(bars, sigs, spread):
    return simulate_exits(bars, sigs, spread=spread, be_r=1.0, max_hold_bars=24)


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
    all_sigs = run_ehlp(b1h, b1d, cfg)
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

    for lookback in [10, 20, 30]:
        for eq_tol in [0.1, 0.2, 0.3]:
            for min_count in [2, 3]:
                for tp_rr in [1.5, 2.0, 3.0]:
                    for atr_mult in [1.0, 1.5]:
                        for sl_buf in [0.1, 0.3]:
                            cfg = dict(
                                lookback=lookback,
                                equality_tol_atr=eq_tol,
                                min_count=min_count,
                                sl_buffer=sl_buf,
                                tp_rr=tp_rr,
                                atr_mult_sl=atr_mult,
                            )
                            label = (f"EHLP lb{lookback} tol{eq_tol}ATR "
                                     f"cnt{min_count} buf{sl_buf} "
                                     f"RR{tp_rr} ATR×{atr_mult}")
                            sigs = run_ehlp(train_b1h, b1d, cfg)
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
    print("Equal Highs/Lows Liquidity Hunt (EHLP) -- 1H -- 4-window walk-forward")
    print(f"BE@1R  max_hold=24H  {DAYS}-day dataset")
    print("Thesis: stop cluster sweep at equal H/L levels → institutional reversal")
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
            print(f"  No profitable EHLP configs found for {sym}.\n")
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
            print(f"  No EHLP config passed 3+ windows for {sym}.")

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
        print("  No EHLP strategy passed 3+ windows on any symbol.")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
