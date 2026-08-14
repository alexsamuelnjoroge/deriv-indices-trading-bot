"""
Inside Bar Sequence Breakout (IBSB) -- proprietary strategy.

Core thesis:
  An "inside bar" is a bar whose ENTIRE range (high to low) is contained within
  the previous bar's range. It represents a market in perfect equilibrium —
  neither buyers nor sellers could extend beyond what the previous bar established.

  When 2+ CONSECUTIVE inside bars form, the market is coiling like a compressed
  spring. Each additional inside bar increases the compression — the range is
  getting tighter, order flow is balancing, energy is accumulating.

  The FIRST BAR that breaks out of the inside bar cluster — that extends beyond
  any previous high or low in the sequence — is the release of that compressed
  energy. The breakout direction reveals which side dominated the silent battle.

  This is completely unlike CRD (anomalous single bar SIZE relative to history).
  IBSB specifically requires a sequence of QUIET bars, then measures the CLUSTER
  BREAKOUT — not just any large bar, but specifically the resolution of an
  identifiable compression sequence.

Signal (1H bars):
  1. Track consecutive "inside bars": bar[i].high <= bar[i-1].high
                                       AND bar[i].low >= bar[i-1].low
  2. When a streak of >= min_inside_bars ends with a bar that:
       - bar.high > max(high of all bars in the cluster)  => BUY breakout
       - bar.low < min(low of all bars in the cluster)    => SELL breakout
  3. Require the breakout to be meaningful: breakout distance > min_break × ATR
  4. Enter in breakout direction
  5. SL: ATR × mult, TP: RR × SL
  6. Cooldown, macro filter

Why consecutive inside bars?
  A single inside bar is common (every trending market has them).
  Two or more consecutive inside bars is unusual — it means price was unable
  to extend in ANY direction for multiple hours. That accumulated pressure
  is what makes the eventual breakout reliable.

Sequence definition:
  We track the CUMULATIVE cluster range (max high and min low of all inside bars).
  The breakout bar must exceed the CLUSTER range, not just the immediately prior bar.
  This makes it a true cluster resolution, not just a comparison to the previous bar.
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

def run_ibsb(b1h, b1d, cfg):
    min_inside     = cfg["min_inside_bars"]  # minimum consecutive inside bars in cluster
    min_break      = cfg["min_break"]        # breakout must exceed cluster by > min_break × ATR
    cooldown_bars  = cfg.get("cooldown_bars", 24)
    tp_rr          = cfg["tp_rr"]
    atr_mult       = cfg["atr_mult_sl"]
    use_macro      = cfg.get("macro_filter", False)
    macro_p        = cfg.get("macro_ema_period", 20)

    atr1h = _atr(b1h, 14)

    if use_macro and b1d:
        ema_d = _ema([b["close"] for b in b1d], macro_p)
        ep_d  = [b["epoch"] for b in b1d]
    else:
        ema_d = ep_d = None

    signals       = []
    last_sig_i    = -999
    inside_streak = 0      # consecutive inside bars
    cluster_high  = None   # max high of the inside bar cluster
    cluster_low   = None   # min low of the inside bar cluster

    start = 15

    for i in range(start, len(b1h)):
        bar      = b1h[i]
        prev_bar = b1h[i - 1]

        atr_val = atr1h[i]
        if atr_val is None or atr_val <= 0:
            inside_streak = 0
            cluster_high = cluster_low = None
            continue

        # Check if current bar is an inside bar relative to previous bar
        is_inside = (bar["high"] <= prev_bar["high"] and
                     bar["low"]  >= prev_bar["low"])

        if is_inside:
            if inside_streak == 0:
                # Start of a new cluster: anchor bar is prev_bar, first inside is bar
                cluster_high = prev_bar["high"]
                cluster_low  = prev_bar["low"]
            inside_streak += 1
            # Expand cluster bounds (shouldn't need to since it's inside, but be safe)
            cluster_high = max(cluster_high, bar["high"])
            cluster_low  = min(cluster_low,  bar["low"])
            continue

        # Current bar is NOT an inside bar
        if inside_streak >= min_inside and cluster_high is not None:
            # This bar broke out of the cluster
            epoch = bar["epoch"]

            if i - last_sig_i >= cooldown_bars:
                allow_long = allow_short = True
                if ema_d is not None:
                    k = bisect.bisect_right(ep_d, epoch) - 1
                    if k >= 1 and ema_d[k] is not None and ema_d[k - 1] is not None:
                        macro_up    = ema_d[k] > ema_d[k - 1]
                        allow_long  = macro_up
                        allow_short = not macro_up

                sl = atr_val * atr_mult
                tp = sl * tp_rr

                # BUY breakout: bar's high exceeded cluster high
                if (bar["high"] > cluster_high and
                        bar["high"] - cluster_high >= min_break * atr_val and
                        bar["close"] > bar["open"] and
                        allow_long):
                    signals.append((i, Signal("BUY", sl_pips=sl, tp_pips=tp,
                                              reason=f"IBSB_buy_str{inside_streak}")))
                    last_sig_i = i

                # SELL breakout: bar's low pierced cluster low
                elif (bar["low"] < cluster_low and
                        cluster_low - bar["low"] >= min_break * atr_val and
                        bar["close"] < bar["open"] and
                        allow_short):
                    signals.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp,
                                              reason=f"IBSB_sell_str{inside_streak}")))
                    last_sig_i = i

        # Reset cluster
        inside_streak = 0
        cluster_high  = None
        cluster_low   = None

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
    all_sigs = run_ibsb(b1h, b1d, cfg)
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

    for min_inside in [2, 3, 4]:
        for min_break in [0.0, 0.1, 0.3]:
            for cooldown in [12, 24]:
                for tp_rr in [1.5, 2.0, 3.0]:
                    for atr_mult in [1.0, 1.5, 2.0]:
                        for macro in [False, True]:
                            cfg = dict(
                                min_inside_bars=min_inside,
                                min_break=min_break,
                                cooldown_bars=cooldown,
                                tp_rr=tp_rr,
                                atr_mult_sl=atr_mult,
                                macro_filter=macro,
                                macro_ema_period=20,
                            )
                            label = (
                                f"IBSB inside>={min_inside} break>{min_break}ATR "
                                f"cool{cooldown} RR{tp_rr} ATRx{atr_mult} "
                                f"{'MACRO' if macro else 'free'}"
                            )
                            sigs = run_ibsb(train_b1h, b1d, cfg)
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
    print("Inside Bar Sequence Breakout (IBSB) -- 1H -- 4-window walk-forward")
    print(f"BE@1R  max_hold=48H  {DAYS}-day dataset")
    print("Thesis: 2+ consecutive inside bars = compressed spring => breakout direction = signal")
    print("Multi-bar cluster: the longer the inside sequence, the more reliable the breakout")
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
            print(f"  No profitable IBSB configs found for {sym}.\n")
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
            print(f"  No IBSB config passed 3+ windows for {sym}.")

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
        print("  No IBSB strategy passed 3+ windows on any symbol.")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
