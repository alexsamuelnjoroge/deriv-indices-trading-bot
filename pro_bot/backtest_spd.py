"""
Spread Pair Divergence (SPD) -- EURUSD vs GBPUSD -- 1H -- 4-window walk-forward

Core thesis:
  EURUSD and GBPUSD are the world's most correlated major currency pairs (both
  are EUR/GBP vs USD). When both move in the same direction but by different
  magnitudes, the underperformer tends to catch up to the leader.

  This is structurally DIFFERENT from CSD (Cross-Symbol Divergence):
    CSD: the LEAD symbol moves, the LAG symbol barely moves (0-correlation moment)
    SPD: BOTH symbols move the same direction, but by different amounts
         (correlated movement, magnitude divergence within the move)

  SPD signal (example -- BUY GBPUSD):
    Over the last N bars:
      eu_ret  = (EURUSD[i] - EURUSD[i-N]) / EURUSD_ATR  >= min_lead_atr  (EUR moved up)
      gb_ret  = (GBPUSD[i] - GBPUSD[i-N]) / GBPUSD_ATR  > 0              (GBP also up)
      eu_ret - gb_ret >= min_divergence  (EUR outperformed GBP by at least this much)
    --> BUY GBPUSD: GBP catching up to EUR's move

  SPD signal (example -- SELL GBPUSD):
    eu_ret <= -min_lead_atr  (EUR down)
    gb_ret < 0               (GBP also down)
    gb_ret - eu_ret >= min_divergence  (EUR fell MORE than GBP)
    --> SELL GBPUSD: GBP catching up to EUR's larger downward move

  The strategy is tested in both directions:
    "gbpusd" -- signal on GBPUSD, EUR is the leader
    "eurusd" -- signal on EURUSD, GBP is the leader
  Based on our CSD research, EURUSD consistently leads, so GBPUSD signals
  are expected to dominate.
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


def _align(b1h_eu, b1h_gb):
    """Match bars by epoch, return aligned (eu_idx, gb_idx) pairs."""
    ep_eu = {b["epoch"]: i for i, b in enumerate(b1h_eu)}
    ep_gb = {b["epoch"]: i for i, b in enumerate(b1h_gb)}
    common = sorted(set(ep_eu) & set(ep_gb))
    return [(ep_eu[e], ep_gb[e]) for e in common]


def run_spd(b1h_eu, b1h_gb, b1d_eu, b1d_gb, cfg, signal_on="gbpusd"):
    """
    signal_on: "gbpusd" = EUR leads, signal on GBP
               "eurusd" = GBP leads, signal on EUR
    """
    window        = cfg["window"]
    min_lead_atr  = cfg["min_lead_atr"]
    min_div       = cfg["min_divergence"]
    cooldown_bars = cfg.get("cooldown_bars", 12)
    tp_rr         = cfg["tp_rr"]
    atr_mult      = cfg["atr_mult_sl"]
    use_macro     = cfg.get("macro_filter", False)
    macro_p       = cfg.get("macro_ema_period", 20)

    atr_eu = _atr(b1h_eu, 14)
    atr_gb = _atr(b1h_gb, 14)

    aligned = _align(b1h_eu, b1h_gb)

    if signal_on == "gbpusd":
        b1d_sig = b1d_gb
        b1h_sig = b1h_gb
        spread  = SPREADS["frxGBPUSD"]
    else:
        b1d_sig = b1d_eu
        b1h_sig = b1h_eu
        spread  = SPREADS["frxEURUSD"]

    if use_macro and b1d_sig:
        ema_d = _ema([b["close"] for b in b1d_sig], macro_p)
        ep_d  = [b["epoch"] for b in b1d_sig]
    else:
        ema_d = ep_d = None

    signals    = []
    last_sig_i = -999

    for pair_i, (ei, gi) in enumerate(aligned):
        if ei < window + 5 or gi < window + 5:
            continue

        if signal_on == "gbpusd":
            sig_i   = gi
            lag_i   = gi
        else:
            sig_i   = ei
            lag_i   = ei

        if sig_i - last_sig_i < cooldown_bars:
            continue

        atr_eu_val = atr_eu[ei]
        atr_gb_val = atr_gb[gi]
        if atr_eu_val is None or atr_eu_val <= 0:
            continue
        if atr_gb_val is None or atr_gb_val <= 0:
            continue

        # Find aligned bar from window bars ago
        if pair_i < window:
            continue
        ei_w, gi_w = aligned[pair_i - window]

        eu_ret = (b1h_eu[ei]["close"] - b1h_eu[ei_w]["close"]) / atr_eu_val
        gb_ret = (b1h_gb[gi]["close"] - b1h_gb[gi_w]["close"]) / atr_gb_val

        epoch = b1h_sig[sig_i]["epoch"]

        allow_long = allow_short = True
        if ema_d is not None:
            k = bisect.bisect_right(ep_d, epoch) - 1
            if k >= 1 and ema_d[k] is not None and ema_d[k - 1] is not None:
                macro_up    = ema_d[k] > ema_d[k - 1]
                allow_long  = macro_up
                allow_short = not macro_up

        if signal_on == "gbpusd":
            atr_sig = atr_gb_val
            leader_ret = eu_ret
            lagger_ret = gb_ret
        else:
            atr_sig = atr_eu_val
            leader_ret = gb_ret
            lagger_ret = eu_ret

        sl = atr_sig * atr_mult
        tp = sl * tp_rr

        # BUY lag: leader up, lagger also up but less
        if (leader_ret >= min_lead_atr and
                lagger_ret > 0 and
                leader_ret - lagger_ret >= min_div and
                allow_long):
            signals.append((sig_i, Signal("BUY", sl_pips=sl, tp_pips=tp,
                                          reason=f"SPD_buy_eu{eu_ret:.1f}_gb{gb_ret:.1f}")))
            last_sig_i = sig_i

        # SELL lag: leader down, lagger also down but less
        elif (leader_ret <= -min_lead_atr and
                  lagger_ret < 0 and
                  lagger_ret - leader_ret >= min_div and
                  allow_short):
            signals.append((sig_i, Signal("SELL", sl_pips=sl, tp_pips=tp,
                                          reason=f"SPD_sell_eu{eu_ret:.1f}_gb{gb_ret:.1f}")))
            last_sig_i = sig_i

    return signals


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


def _split_sigs(all_sigs, c1, c2):
    return ([(i, s) for i, s in all_sigs if i < c1],
            [(i - c1, s) for i, s in all_sigs if c1 <= i < c2])


def run_wf(b1h_eu, b1h_gb, b1d_eu, b1d_gb, cfg, label, signal_on, verbose=True):
    all_sigs = run_spd(b1h_eu, b1h_gb, b1d_eu, b1d_gb, cfg, signal_on)
    if signal_on == "gbpusd":
        b1h_sig = b1h_gb
        spread  = SPREADS["frxGBPUSD"]
    else:
        b1h_sig = b1h_eu
        spread  = SPREADS["frxEURUSD"]

    n      = len(b1h_sig)
    passes = 0

    if verbose:
        print(f"  Config [{signal_on}]: {label}  [{len(all_sigs)} total signals]")

    for wi, (_, te_pct, ho_pct) in enumerate(WINDOWS):
        c1 = int(n * te_pct)
        c2 = int(n * ho_pct)

        tr_sigs, ho_sigs = _split_sigs(all_sigs, c1, c2)
        tr_s = stats(sim(b1h_sig[:c1],   tr_sigs, spread), min_n=5)
        ho_s = stats(sim(b1h_sig[c1:c2], ho_sigs, spread), min_n=3)

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
            tr_d = (b1h_sig[c1-1]["epoch"] - b1h_sig[0]["epoch"]) // 86400 if c1 > 0 else 0
            ho_d = (b1h_sig[c2-1]["epoch"] - b1h_sig[c1]["epoch"]) // 86400 if c2 > c1 else 0
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


def sweep(b1h_eu, b1h_gb, b1d_eu, b1d_gb, train_end_eu, train_end_gb):
    results = []

    for signal_on in ["gbpusd", "eurusd"]:
        for window in [6, 12, 24]:
            for min_lead in [0.5, 1.0, 1.5, 2.0]:
                for min_div in [0.3, 0.5, 1.0, 1.5]:
                    for cooldown in [6, 12, 24]:
                        for tp_rr in [1.5, 2.0, 3.0]:
                            for atr_mult in [1.0, 1.5, 2.0]:
                                for macro in [False, True]:
                                    cfg = dict(
                                        window=window,
                                        min_lead_atr=min_lead,
                                        min_divergence=min_div,
                                        cooldown_bars=cooldown,
                                        tp_rr=tp_rr,
                                        atr_mult_sl=atr_mult,
                                        macro_filter=macro,
                                        macro_ema_period=20,
                                    )
                                    label = (
                                        f"SPD win{window} lead>={min_lead}ATR "
                                        f"div>={min_div}ATR cool{cooldown} "
                                        f"RR{tp_rr} ATRx{atr_mult} "
                                        f"{'MACRO' if macro else 'free'}"
                                    )
                                    if signal_on == "gbpusd":
                                        sigs = run_spd(b1h_eu[:train_end_eu],
                                                       b1h_gb[:train_end_gb],
                                                       b1d_eu, b1d_gb, cfg, signal_on)
                                        if not sigs:
                                            continue
                                        trades = sim(b1h_gb[:train_end_gb], sigs,
                                                     SPREADS["frxGBPUSD"])
                                    else:
                                        sigs = run_spd(b1h_eu[:train_end_eu],
                                                       b1h_gb[:train_end_gb],
                                                       b1d_eu, b1d_gb, cfg, signal_on)
                                        if not sigs:
                                            continue
                                        trades = sim(b1h_eu[:train_end_eu], sigs,
                                                     SPREADS["frxEURUSD"])

                                    s = stats(trades, min_n=8)
                                    if s and s["ev"] > 0:
                                        results.append((s["ev"], s["n"], label, cfg, signal_on))

    results.sort(key=lambda x: -x[0])
    return results


async def main():
    import time as _t

    print("=" * 78)
    print("Spread Pair Divergence (SPD) -- EURUSD vs GBPUSD -- 1H -- 4-window walk-forward")
    print(f"BE@1R  max_hold=48H  {DAYS}-day dataset")
    print("Thesis: both pairs move same direction, underperformer catches up")
    print("Distinct from CSD: SPD requires both symbols to have already moved")
    print("=" * 78 + "\n")

    print("  Loading data...", end=" ", flush=True)
    b1h_eu = await _fetch("frxEURUSD", 3600,  DAYS, CACHE_1H)
    b1h_gb = await _fetch("frxGBPUSD", 3600,  DAYS, CACHE_1H)
    b1d_eu = await _fetch("frxEURUSD", 86400, DAYS, CACHE_1D)
    b1d_gb = await _fetch("frxGBPUSD", 86400, DAYS, CACHE_1D)
    print(f"EU {len(b1h_eu)} 1H | GB {len(b1h_gb)} 1H")

    fd = _t.strftime("%Y-%m-%d", _t.gmtime(b1h_eu[0]["epoch"]))
    ld = _t.strftime("%Y-%m-%d", _t.gmtime(b1h_eu[-1]["epoch"]))
    print(f"  Date range: {fd} -> {ld}")

    train_end_eu = int(len(b1h_eu) * 0.60)
    train_end_gb = int(len(b1h_gb) * 0.60)
    print(f"\n  Phase 1 -- sweep (60% training window)...")
    ranked = sweep(b1h_eu, b1h_gb, b1d_eu, b1d_gb, train_end_eu, train_end_gb)

    print(f"  {len(ranked)} configs with positive EV on training data.")
    if not ranked:
        print("  No profitable SPD configs found.\n")
        return

    print("  Top 5:")
    for ev, n, label, _, signal_on in ranked[:5]:
        print(f"    EV {ev:>+.4f}R  n={n:>3}  [{signal_on}]  {label}")

    all_robust = []
    wf_results = []
    for ev, n, label, cfg, signal_on in ranked[:5]:
        print(f"\n  {'-' * 70}")
        passes = run_wf(b1h_eu, b1h_gb, b1d_eu, b1d_gb, cfg, label, signal_on, verbose=True)
        wf_results.append((passes, ev, label, cfg, signal_on))

    robust = [(p, ev, l, c, s) for p, ev, l, c, s in wf_results if p >= 3]
    robust.sort(key=lambda x: (-x[0], -x[1]))

    print("\n  SUMMARY:")
    if robust:
        for passes, ev, label, cfg, signal_on in robust:
            bar     = "#" * passes + "." * (4 - passes)
            verdict = "ROBUST" if passes == 4 else "MOSTLY OK"
            print(f"  [{bar}] {passes}/4  [{signal_on}]  {label}")
            print(f"         -> {verdict}  (train EV {ev:+.4f}R)")
            if passes == 4:
                print(f"  BEST CONFIG [{signal_on}]: {cfg}")
            all_robust.append((signal_on, passes, ev, label, cfg))
    else:
        print("  No SPD config passed 3+ windows.")

    print("\n" + "=" * 78)
    print("OVERALL ROBUST / MOSTLY OK:")
    print("=" * 78)
    if all_robust:
        for signal_on, passes, ev, label, cfg in sorted(all_robust, key=lambda x: (-x[1], -x[2])):
            bar     = "#" * passes + "." * (4 - passes)
            verdict = "ROBUST" if passes == 4 else "MOSTLY OK"
            print(f"  [{bar}] {passes}/4  [{signal_on}]  {label}")
            print(f"         -> {verdict}  train EV {ev:+.4f}R")
    else:
        print("  No SPD strategy passed 3+ windows.")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
