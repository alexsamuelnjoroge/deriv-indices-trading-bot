"""
Daily ATR Contraction → H1 Expansion (DACE) -- proprietary strategy.

Core thesis:
  Markets breathe: periods of low daily volatility (ATR contraction across
  multiple daily bars) represent energy accumulation. When that compression
  releases, the first H1 bar that breaks out of the recent compressed range
  catches the expansion move in its early stages.

  Unlike standard breakout strategies, DACE requires MULTI-DAY ATR compression
  as a prerequisite -- this filters out normal consolidation and targets the
  relatively rare, high-conviction expansion events.

Signal (on H1 bars, using daily ATR for compression detection):
  1. On each H1 bar, check the last N daily bars for ATR contraction:
     Each daily bar's true range must be <= prev_bar's true range (monotone).
     OR: last N daily ATR values are all below their own N-day SMA (softer).
  2. At the current H1 bar, if range > expansion_mult × recent low daily ATR:
     - H1 bar closes bullish → BUY (breakout expanding upward)
     - H1 bar closes bearish → SELL (breakout expanding downward)
  3. ATR-based SL/TP.

Why this edge exists:
  Multi-day ATR compression creates stop clusters near the compressed range
  boundaries. When these stop clusters are hit, the resulting stop-run
  provides the fuel for an extended expansion move. The H1 expansion bar
  is the trigger that confirms the stop-run is underway.
"""

import asyncio
import bisect
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


# ── ATR helpers ────────────────────────────────────────────────────────────────

def _atr_series(bars, period=14):
    """Return list of ATR values (same length as bars, None until warm)."""
    result = [None] * len(bars)
    for i in range(1, len(bars)):
        trs = []
        for j in range(max(1, i - period + 1), i + 1):
            h, l, cp = bars[j]["high"], bars[j]["low"], bars[j - 1]["close"]
            trs.append(max(h - l, abs(h - cp), abs(l - cp)))
        if len(trs) >= period // 2:
            result[i] = sum(trs) / len(trs)
    return result


def _true_range_series(bars):
    """Return simple true range per bar (None for first bar)."""
    result = [None] * len(bars)
    for i in range(1, len(bars)):
        h, l, cp = bars[i]["high"], bars[i]["low"], bars[i - 1]["close"]
        result[i] = max(h - l, abs(h - cp), abs(l - cp))
    return result


# ── Signal generator ──────────────────────────────────────────────────────────

def run_dace(b1h, b1d, cfg):
    compress_days   = cfg["compress_days"]        # days of daily ATR contraction required
    expansion_mult  = cfg["expansion_mult"]       # H1 range > this × avg daily ATR to qualify
    method          = cfg.get("method", "below_sma")  # "monotone" or "below_sma"
    sma_period      = cfg.get("sma_period", 10)   # for below_sma method
    tp_rr           = cfg["tp_rr"]
    atr_mult        = cfg["atr_mult_sl"]

    atr1h  = _atr_series(b1h, 14)
    atr1d  = _atr_series(b1d, 14)
    tr1d   = _true_range_series(b1d)

    ep_d   = [b["epoch"] for b in b1d]
    signals = []
    start  = 50  # allow daily data to warm up

    for i in range(start, len(b1h)):
        bar   = b1h[i]
        epoch = bar["epoch"]

        atr_val = atr1h[i]
        if atr_val is None or atr_val <= 0:
            continue

        # Find corresponding daily bar index
        k = bisect.bisect_right(ep_d, epoch) - 1
        if k < compress_days + sma_period:
            continue

        # Check for daily ATR contraction
        compressed = False

        if method == "monotone":
            # Each of the last compress_days bars must have TR <= previous bar's TR
            trs = [tr1d[k - compress_days + j + 1] for j in range(compress_days)]
            if all(t is not None for t in trs):
                compressed = all(trs[j] <= trs[j - 1] for j in range(1, len(trs)))

        elif method == "below_sma":
            # Last compress_days daily ATRs must all be below their SMA
            recent_atrs = [atr1d[k - compress_days + j + 1]
                           for j in range(compress_days)]
            sma_atrs    = [atr1d[k - sma_period + j + 1] for j in range(sma_period)]
            if (all(a is not None for a in recent_atrs) and
                    all(a is not None for a in sma_atrs)):
                sma_val    = sum(sma_atrs) / len(sma_atrs)
                compressed = all(a < sma_val for a in recent_atrs)

        if not compressed:
            continue

        # Reference low daily ATR as the compression baseline
        recent_d_atrs = [atr1d[k - compress_days + j + 1] for j in range(compress_days)]
        valid_d_atrs  = [a for a in recent_d_atrs if a is not None]
        if not valid_d_atrs:
            continue
        avg_compressed_atr = sum(valid_d_atrs) / len(valid_d_atrs)

        # H1 expansion bar: range must exceed the expansion threshold
        h1_range = bar["high"] - bar["low"]
        if h1_range < expansion_mult * avg_compressed_atr:
            continue

        sl = atr_val * atr_mult
        tp = sl * tp_rr

        if bar["close"] > bar["open"]:
            # Bullish expansion → BUY
            signals.append((i, Signal("BUY", sl_pips=sl, tp_pips=tp,
                                      reason="DACE_bull_expansion")))
        elif bar["close"] < bar["open"]:
            # Bearish expansion → SELL
            signals.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp,
                                      reason="DACE_bear_expansion")))

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
    all_sigs = run_dace(b1h, b1d, cfg)
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

    for compress_days in [3, 5, 7]:
        for expansion_mult in [0.3, 0.5, 0.7]:
            for method in ["monotone", "below_sma"]:
                for tp_rr in [2.0, 3.0]:
                    for atr_mult in [1.0, 1.5, 2.0]:
                        cfg = dict(
                            compress_days=compress_days,
                            expansion_mult=expansion_mult,
                            method=method,
                            sma_period=10,
                            tp_rr=tp_rr,
                            atr_mult_sl=atr_mult,
                        )
                        label = (f"DACE cmp{compress_days}d exp{expansion_mult}x "
                                 f"{method} RR{tp_rr} ATR×{atr_mult}")
                        sigs = run_dace(train_b1h, b1d, cfg)
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
    print("Daily ATR Contraction -> H1 Expansion (DACE) -- 1H -- 4-window walk-forward")
    print(f"BE@1R  max_hold=48H  {DAYS}-day dataset")
    print("Thesis: multi-day daily ATR compression releases into H1 expansion breakout")
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
            print(f"  No profitable DACE configs found for {sym}.\n")
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
            print(f"  No DACE config passed 3+ windows for {sym}.")

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
        print("  No DACE strategy passed 3+ windows on any symbol.")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
