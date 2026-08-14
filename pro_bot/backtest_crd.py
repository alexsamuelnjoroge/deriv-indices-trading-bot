"""
Candle Rhythm Disruption (CRD) -- proprietary strategy.

Core thesis:
  Markets breathe in a rhythm — candles alternate large/small in a roughly
  predictable cycle. When a SINGLE bar breaks this rhythm by being dramatically
  larger than recent bars, a large institutional order hit the market.

  The direction that institutional order moved price IS the direction to trade:
    - Anomalously large BULL candle → institutional buying → BUY (trend mode)
    - Anomalously large BEAR candle → institutional selling → SELL (trend mode)

  This is the opposite of CBVE (which fades exhaustion).
  CRD follows the institutional order, not fades it.

  The disruption candle itself has already made a big move, but institutional
  orders rarely complete in one bar. The position gets built across multiple
  bars → follow-through expected in the next 24-48H.

Signal (1H bars):
  1. Compute each bar's range ratio: range[i] / range[i-1]
  2. Compute rolling average of range ratios (avg_period bars)
  3. Trigger when: range_ratio[i] > disruption_mult × rolling_avg
  4. Require minimum body: body > min_body_ratio × full_range (no doji disruptions)
  5. Enter in direction of disruption bar body (or fade in fade_mode)
  6. SL: ATR × mult, TP: RR × SL
  7. Cooldown: block re-entry for cooldown_bars after each signal

  Parameters also test FADE mode: entering AGAINST the disruption direction
  (disruption may be a news spike trap designed to draw in momentum traders).
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

def run_crd(b1h, b1d, cfg):
    avg_period     = cfg["avg_period"]       # rolling average for range ratios
    disrupt_mult   = cfg["disruption_mult"]  # bar must be > this × avg to trigger
    min_body_ratio = cfg["min_body_ratio"]   # body / full_range must exceed this (no dojis)
    fade_mode      = cfg.get("fade_mode", False)  # True = enter against disruption
    cooldown_bars  = cfg.get("cooldown_bars", 24) # bars to wait after a signal
    tp_rr          = cfg["tp_rr"]
    atr_mult       = cfg["atr_mult_sl"]
    use_macro      = cfg.get("macro_filter", False)
    macro_p        = cfg.get("macro_ema_period", 20)
    min_prev_range_ratio = 0.10  # prev bar range must be > this × ATR (avoid near-zero)

    atr1h = _atr(b1h, 14)

    if use_macro and b1d:
        ema_d = _ema([b["close"] for b in b1d], macro_p)
        ep_d  = [b["epoch"] for b in b1d]
    else:
        ema_d = ep_d = None

    signals    = []
    last_sig_i = -999
    start      = avg_period + 5

    for i in range(start, len(b1h)):
        bar   = b1h[i]
        epoch = bar["epoch"]

        # Cooldown
        if i - last_sig_i < cooldown_bars:
            continue

        atr_val = atr1h[i]
        if atr_val is None or atr_val <= 0:
            continue

        prev_bar = b1h[i - 1]
        prev_range = prev_bar["high"] - prev_bar["low"]
        if prev_range < min_prev_range_ratio * atr_val:
            continue  # previous bar range too small — ratio would be artificially inflated

        cur_range = bar["high"] - bar["low"]
        if cur_range <= 0:
            continue

        # Body size check — filter out dojis
        body = abs(bar["close"] - bar["open"])
        if body < min_body_ratio * cur_range:
            continue

        # Rolling average of range ratios over previous avg_period bars
        ratios = []
        for j in range(i - avg_period, i):
            if j < 1:
                continue
            r_j   = b1h[j]["high"] - b1h[j]["low"]
            r_j_1 = b1h[j - 1]["high"] - b1h[j - 1]["low"]
            if r_j_1 > min_prev_range_ratio * atr_val:
                ratios.append(r_j / r_j_1)

        if len(ratios) < avg_period // 2:
            continue
        avg_ratio = sum(ratios) / len(ratios)

        cur_ratio = cur_range / prev_range
        if cur_ratio <= disrupt_mult * avg_ratio:
            continue  # not anomalous enough

        # Macro gate
        allow_long = allow_short = True
        if ema_d is not None:
            k = bisect.bisect_right(ep_d, epoch) - 1
            if k >= 1 and ema_d[k] is not None and ema_d[k - 1] is not None:
                macro_up    = ema_d[k] > ema_d[k - 1]
                allow_long  = macro_up
                allow_short = not macro_up

        sl = atr_val * atr_mult
        tp = sl * tp_rr

        # Direction of disruption bar
        bar_bull = bar["close"] > bar["open"]

        # In trend mode: follow the disruption; in fade mode: fade it
        go_long = bar_bull if not fade_mode else not bar_bull

        if go_long and allow_long:
            signals.append((i, Signal("BUY", sl_pips=sl, tp_pips=tp,
                                      reason=f"CRD_ratio{cur_ratio:.1f}x")))
            last_sig_i = i
        elif not go_long and allow_short:
            signals.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp,
                                      reason=f"CRD_ratio{cur_ratio:.1f}x")))
            last_sig_i = i

    return signals


# ── Simulation / stats ───────────────────────────────────────────────────────

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


# ── Walk-forward ─────────────────────────────────────────────────────────────

def _split_sigs(all_sigs, c1, c2):
    return ([(i, s) for i, s in all_sigs if i < c1],
            [(i - c1, s) for i, s in all_sigs if c1 <= i < c2])


def run_wf(b1h, b1d, cfg, label, spread, verbose=True):
    all_sigs = run_crd(b1h, b1d, cfg)
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


# ── Parameter sweep ──────────────────────────────────────────────────────────

def sweep(b1h, b1d, train_end_idx, spread):
    train_b1h = b1h[:train_end_idx]
    results   = []

    for avg_period in [5, 10]:
        for disrupt_mult in [1.5, 2.0, 2.5]:
            for min_body in [0.30, 0.50]:
                for fade in [False, True]:
                    for cooldown in [12, 24]:
                        for tp_rr in [1.5, 2.0, 3.0]:
                            for atr_mult in [1.0, 1.5]:
                                for macro in [False, True]:
                                    cfg = dict(
                                        avg_period=avg_period,
                                        disruption_mult=disrupt_mult,
                                        min_body_ratio=min_body,
                                        fade_mode=fade,
                                        cooldown_bars=cooldown,
                                        tp_rr=tp_rr,
                                        atr_mult_sl=atr_mult,
                                        macro_filter=macro,
                                        macro_ema_period=20,
                                    )
                                    label = (
                                        f"CRD avg{avg_period} "
                                        f"x{disrupt_mult} "
                                        f"body{min_body} "
                                        f"{'FADE' if fade else 'TREND'} "
                                        f"cool{cooldown} "
                                        f"RR{tp_rr} ATR×{atr_mult} "
                                        f"{'MACRO' if macro else 'free'}"
                                    )
                                    sigs = run_crd(train_b1h, b1d, cfg)
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
    print("Candle Rhythm Disruption (CRD) -- 1H -- 4-window walk-forward")
    print(f"BE@1R  max_hold=48H  {DAYS}-day dataset")
    print("Thesis: anomalously large candle = institutional order → trade its direction")
    print("Tests both TREND (follow) and FADE (counter) modes")
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
        print(f"\n  Phase 1 -- sweep ({train_end} bars / 60%)...")
        ranked = sweep(b1h, b1d, train_end, spread)

        print(f"  {len(ranked)} configs with positive EV on training data.")
        if not ranked:
            print(f"  No profitable CRD configs found for {sym}.\n")
            continue

        print("  Top 5:")
        for ev, n, label, _ in ranked[:5]:
            print(f"    EV {ev:>+.4f}R  n={n:>3}  {label}")

        wf_results = []
        for ev, n, label, cfg in ranked[:5]:
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
            print(f"  No CRD config passed 3+ windows for {sym}.")

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
        print("  No CRD strategy passed 3+ windows on any symbol.")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
