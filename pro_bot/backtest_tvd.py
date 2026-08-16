"""
Range-Body Participation Divergence (TVD) -- proprietary strategy.
(Originally conceived as "Tick Volume Divergence"; redesigned for OHLC-only data.)

Core thesis:
  When price makes a new N-bar extreme (high or low), but the bar that makes
  that extreme closes very weakly (body/range ratio is tiny), it signals that
  the participation was thin -- price was pushed to the extreme but reversed
  before the close.

  This is the OHLC proxy for tick volume divergence:
    - High tick volume at extreme → strong breakout (don't fade)
    - Low tick volume at extreme → trapped breakout, reversal likely (fade)
  A tiny body/range ratio on an extreme bar is equivalent to "range was large
  but most participants ended up near the open" = weak participation.

Signal (on 1H bars):
  1. Check if current bar's high is the N-bar highest high → new extreme
  2. Compute body_ratio = |close - open| / (high - low)
  3. If body_ratio < threshold → participation was weak → SELL (fading new high)
  4. Similarly for new N-bar low with weak body → BUY
  5. Optional: require bar to close in the lower half of its range for SELL
     (close < midpoint of high-low) as extra confirmation of rejection.
  6. ATR-based SL/TP.

Why this edge exists:
  Breakout traders get trapped when price briefly exceeds prior highs/lows
  but closes back inside the range. Their stops cluster just above the sweep
  high / below the sweep low. Once those stops are hit by the reversal, the
  momentum accelerates in the counter direction.
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


# ── Signal generator ──────────────────────────────────────────────────────────

def run_tvd(b1h, b1d, cfg):
    lookback      = cfg["lookback"]            # N-bar window for defining the extreme
    body_ratio_th = cfg["body_ratio_thresh"]   # body/range must be < this to signal
    close_confirm = cfg.get("close_confirm", True)  # bar must close in direction of rejection
    min_range_atr = cfg.get("min_range_atr", 0.5)   # bar range must be > this × ATR
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

        rng = bar["high"] - bar["low"]
        if rng < min_range_atr * atr_val:
            continue  # bar too small to be a meaningful participation signal

        body = abs(bar["close"] - bar["open"])
        if rng == 0:
            continue
        body_ratio = body / rng

        if body_ratio >= body_ratio_th:
            continue  # body too large → strong participation → not exhaustion

        midpoint = (bar["high"] + bar["low"]) / 2

        # Prior N bars excluding current
        prior = b1h[i - lookback: i]
        prior_highs = [b["high"] for b in prior]
        prior_lows  = [b["low"]  for b in prior]

        sl = atr_val * atr_mult
        tp = sl * tp_rr

        # New extreme high + weak close → SELL
        if bar["high"] >= max(prior_highs):
            if not close_confirm or bar["close"] < midpoint:
                signals.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp,
                                          reason="TVD_extreme_high_weak")))

        # New extreme low + weak close → BUY
        elif bar["low"] <= min(prior_lows):
            if not close_confirm or bar["close"] > midpoint:
                signals.append((i, Signal("BUY", sl_pips=sl, tp_pips=tp,
                                          reason="TVD_extreme_low_weak")))

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
    all_sigs = run_tvd(b1h, b1d, cfg)
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

    for lookback in [8, 12, 20]:
        for body_th in [0.20, 0.30, 0.40]:
            for close_confirm in [True, False]:
                for min_range in [0.3, 0.6]:
                    for tp_rr in [1.5, 2.0, 3.0]:
                        for atr_mult in [1.0, 1.5, 2.0]:
                            cfg = dict(
                                lookback=lookback,
                                body_ratio_thresh=body_th,
                                close_confirm=close_confirm,
                                min_range_atr=min_range,
                                tp_rr=tp_rr,
                                atr_mult_sl=atr_mult,
                            )
                            label = (f"TVD lb{lookback} body<{body_th} "
                                     f"{'cc' if close_confirm else 'nocc'} "
                                     f"minR{min_range} RR{tp_rr} ATR×{atr_mult}")
                            sigs = run_tvd(train_b1h, b1d, cfg)
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
    print("Range-Body Participation Divergence (TVD) -- 1H -- 4-window walk-forward")
    print(f"BE@1R  max_hold=24H  {DAYS}-day dataset")
    print("Thesis: new N-bar extreme with tiny body/range = thin participation = fade")
    print("Note: OHLC-only proxy for tick volume; no volume data available in feed")
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
            print(f"  No profitable TVD configs found for {sym}.\n")
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
            print(f"  No TVD config passed 3+ windows for {sym}.")

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
        print("  No TVD strategy passed 3+ windows on any symbol.")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
