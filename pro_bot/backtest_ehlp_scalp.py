"""
Equal Highs/Lows Liquidity Hunt (EHLP) -- M5 and M15 timeframes.

Why test scalp timeframes?
  On H1, equal-level clusters are rare and the sweep condition (high exceeds
  level, closes back below within ONE bar) is too strict for a 60-min bar.
  On M5/M15 the same structure forms over 1-4 hours (12-48 M5 bars) and the
  sweep bar is just 5-15 minutes wide -- a much tighter and more realistic
  signal that institutional flow actually creates.

M15 bars are derived by aggregating M5 bars in-script (no separate cache).

Signal:
  1. Scan prior `lookback` bars for a cluster of "equal highs" within
     `equality_tol_atr × ATR(14)` of each other (min_count bars in cluster).
  2. Current bar sweeps above the cluster level (bar.high > eq_high).
  3. Current bar CLOSES back below eq_high → trapped breakout buyers → SELL.
  4. Mirror for equal lows → BUY.
  5. SL = sweep distance + buffer. TP = RR × SL.

Parameters differ from H1 version:
  - Wider ATR tolerance (0.3-1.5) since M5 ATR is smaller → 1-pip clusters
    need 0.5-1.5 × ATR tolerance to capture the liquidity zone.
  - Shorter max_hold (24-72 M5 bars = 2-6h; 12-36 M15 bars = 3-9h).
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pro_bot.backtest import simulate_exits, SPREADS, _fetch, CACHE_5M, CACHE_1D
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

SYMBOLS_TO_TEST = [
    ("frxXAUUSD", SPREADS["frxXAUUSD"]),
    ("frxEURUSD", SPREADS["frxEURUSD"]),
    ("frxGBPUSD", SPREADS["frxGBPUSD"]),
    ("frxUSDJPY", SPREADS["frxUSDJPY"]),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def aggregate_to_m15(m5_bars: list) -> list:
    """Aggregate M5 bars to M15 by bucketing on 15-minute epoch boundaries."""
    buckets: dict = {}
    for b in m5_bars:
        key = (b["epoch"] // 900) * 900
        if key not in buckets:
            buckets[key] = {"open": b["open"], "high": b["high"],
                            "low": b["low"], "close": b["close"], "epoch": key}
        else:
            buckets[key]["high"]  = max(buckets[key]["high"],  b["high"])
            buckets[key]["low"]   = min(buckets[key]["low"],   b["low"])
            buckets[key]["close"] = b["close"]
    return [buckets[k] for k in sorted(buckets)]


def _find_equal_level(prices, tol, min_count=2):
    """
    Return average price of the densest cluster within tol, or None.
    O(n log n) two-pointer after sorting — fast enough for M5 bar counts.
    """
    if not prices:
        return None
    sorted_p = sorted(prices)
    best_count = 0
    best_lo = best_hi = 0
    lo = 0
    for hi in range(len(sorted_p)):
        while sorted_p[hi] - sorted_p[lo] > tol:
            lo += 1
        count = hi - lo + 1
        if count > best_count:
            best_count = count
            best_lo, best_hi = lo, hi
    if best_count < min_count:
        return None
    return sum(sorted_p[best_lo:best_hi + 1]) / best_count


# ── Signal generator ──────────────────────────────────────────────────────────

def run_ehlp(bars, cfg):
    lookback      = cfg["lookback"]
    equality_tol  = cfg["equality_tol_atr"]
    min_count     = cfg.get("min_count", 2)
    sl_buffer     = cfg.get("sl_buffer_atr", 0.3)
    tp_rr         = cfg["tp_rr"]
    atr_mult      = cfg["atr_mult_sl"]

    atr14 = _atr(bars, 14)
    signals = []
    start = lookback + 20

    for i in range(start, len(bars)):
        bar     = bars[i]
        atr_val = atr14[i]
        if atr_val is None or atr_val <= 0:
            continue

        tol   = equality_tol * atr_val
        prior = bars[i - lookback: i]

        prior_highs = [b["high"] for b in prior]
        prior_lows  = [b["low"]  for b in prior]

        # ── Equal highs sweep ──────────────────────────────────────────────
        eq_high = _find_equal_level(prior_highs, tol, min_count)
        if eq_high is not None and bar["high"] > eq_high and bar["close"] < eq_high:
            sl = (bar["high"] - bar["close"]) + sl_buffer * atr_val
            sl = max(sl, atr_val * atr_mult)
            tp = sl * tp_rr
            signals.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp,
                                      reason="EHLP_high_sweep")))
            continue

        # ── Equal lows sweep ───────────────────────────────────────────────
        eq_low = _find_equal_level(prior_lows, tol, min_count)
        if eq_low is not None and bar["low"] < eq_low and bar["close"] > eq_low:
            sl = (bar["close"] - bar["low"]) + sl_buffer * atr_val
            sl = max(sl, atr_val * atr_mult)
            tp = sl * tp_rr
            signals.append((i, Signal("BUY", sl_pips=sl, tp_pips=tp,
                                      reason="EHLP_low_sweep")))

    return signals


# ── Simulation / stats ────────────────────────────────────────────────────────

def sim(bars, sigs, spread, max_hold):
    return simulate_exits(bars, sigs, spread=spread, be_r=1.0,
                          max_hold_bars=max_hold)


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


def run_wf(bars, cfg, label, spread, max_hold, verbose=True):
    all_sigs = run_ehlp(bars, cfg)
    n        = len(bars)
    passes   = 0

    if verbose:
        print(f"  Config: {label}  [{len(all_sigs)} signals]")

    for wi, (_, te_pct, ho_pct) in enumerate(WINDOWS):
        c1 = int(n * te_pct)
        c2 = int(n * ho_pct)
        tr_sigs, ho_sigs = _split_sigs(all_sigs, c1, c2)
        tr_s = stats(sim(bars[:c1],    tr_sigs, spread, max_hold), min_n=5)
        ho_s = stats(sim(bars[c1:c2], ho_sigs, spread, max_hold), min_n=3)

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
            tr_d = (bars[c1 - 1]["epoch"] - bars[0]["epoch"]) // 86400 if c1 > 0 else 0
            ho_d = (bars[c2 - 1]["epoch"] - bars[c1]["epoch"]) // 86400 if c2 > c1 else 0
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

def sweep(bars, train_end_idx, spread, max_hold):
    # Tight sweep — M5 has 67k bars so keep configs small.
    # 3 × 3 × 2 × 2 × 2 = 72 configs per symbol/timeframe.
    train_bars = bars[:train_end_idx]
    results    = []

    for lookback in [16, 32, 48]:
        for eq_tol in [0.5, 1.0, 1.5]:
            for min_count in [2, 3]:
                for tp_rr in [2.0, 3.0]:
                    for atr_mult in [1.5, 2.0]:
                        sl_buf = 0.3
                        cfg = dict(
                            lookback=lookback,
                            equality_tol_atr=eq_tol,
                            min_count=min_count,
                            sl_buffer_atr=sl_buf,
                            tp_rr=tp_rr,
                            atr_mult_sl=atr_mult,
                        )
                        label = (f"EHLP lb{lookback} tol{eq_tol}ATR "
                                 f"cnt{min_count} "
                                 f"RR{tp_rr} ATR×{atr_mult}")
                        sigs = run_ehlp(train_bars, cfg)
                        if not sigs:
                            continue
                        trades = sim(train_bars, sigs, spread, max_hold)
                        s = stats(trades, min_n=10)
                        if s and s["ev"] > 0:
                            results.append((s["ev"], s["n"], label, cfg))

    results.sort(key=lambda x: -x[0])
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    import time as _t

    print("=" * 78)
    print("Equal Highs/Lows Liquidity Hunt (EHLP) -- M5 + M15 -- 4-window walk-forward")
    print(f"BE@1R  {DAYS}-day dataset  (M15 aggregated from M5 cache)")
    print("Thesis: stop cluster sweep at equal H/L levels → institutional reversal")
    print("=" * 78)

    all_robust = []

    for sym, spread in SYMBOLS_TO_TEST:
        print(f"\n{'═' * 78}")
        print(f"  {sym}  spread={spread}")
        print(f"{'═' * 78}")

        print("  Loading M5 data...", end=" ", flush=True)
        m5 = await _fetch(sym, 300, DAYS, CACHE_5M)
        print(f"{len(m5)} bars")

        m15 = aggregate_to_m15(m5)
        print(f"  Aggregated to M15: {len(m15)} bars")

        fd = _t.strftime("%Y-%m-%d", _t.gmtime(m5[0]["epoch"]))
        ld = _t.strftime("%Y-%m-%d", _t.gmtime(m5[-1]["epoch"]))
        print(f"  Date range: {fd} -> {ld}")

        for tf_label, bars, max_hold in [
            ("M5",  m5,  72),   # max hold 6h on M5
            ("M15", m15, 36),   # max hold 9h on M15
        ]:
            print(f"\n  ── {tf_label} ({len(bars)} bars, max_hold={max_hold} bars) ──────────────")

            train_end = int(len(bars) * 0.60)
            print(f"  Phase 1 -- sweep on training set ({train_end} bars / 60%)...")
            ranked = sweep(bars, train_end, spread, max_hold)

            print(f"  {len(ranked)} configs with positive training EV.")
            if not ranked:
                print(f"  No profitable EHLP-{tf_label} configs found for {sym}.\n")
                continue

            print("  Top 5 training configs:")
            for ev, n, label, _ in ranked[:5]:
                print(f"    EV {ev:>+.4f}R  n={n:>4}  {label}")

            top5 = ranked[:5]
            print(f"\n  Phase 2 -- 4-window walk-forward on top 5 {tf_label} configs")
            print(f"  {'-' * 60}")

            wf_results = []
            for ev, n, label, cfg in top5:
                print(f"\n  {'─' * 70}")
                passes = run_wf(bars, cfg, label, spread, max_hold, verbose=True)
                wf_results.append((passes, ev, label, cfg, tf_label))

            robust = [(p, ev, l, c, tf) for p, ev, l, c, tf in wf_results if p >= 3]
            robust.sort(key=lambda x: (-x[0], -x[1]))

            print(f"\n  {sym} {tf_label} SUMMARY:")
            if robust:
                for passes, ev, label, cfg, tf in robust:
                    bar     = "#" * passes + "." * (4 - passes)
                    verdict = "ROBUST" if passes == 4 else "MOSTLY OK"
                    print(f"  [{bar}] {passes}/4  {label}")
                    print(f"         -> {verdict}  (train EV {ev:+.4f}R)")
                    if passes == 4:
                        print(f"  BEST CONFIG: {cfg}")
                    all_robust.append((sym, tf_label, passes, ev, label, cfg))
            else:
                print(f"  No EHLP-{tf_label} config passed 3+ windows for {sym}.")

    print("\n" + "=" * 78)
    print("OVERALL ROBUST / MOSTLY OK (M5 + M15):")
    print("=" * 78)
    if all_robust:
        for sym, tf, passes, ev, label, cfg in sorted(
                all_robust, key=lambda x: (-x[2], -x[3])):
            bar     = "#" * passes + "." * (4 - passes)
            verdict = "ROBUST" if passes == 4 else "MOSTLY OK"
            print(f"  [{bar}] {passes}/4  {sym} {tf}  {label}")
            print(f"         -> {verdict}  train EV {ev:+.4f}R")
    else:
        print("  No EHLP config passed 3+ windows on M5 or M15 on any symbol.")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
