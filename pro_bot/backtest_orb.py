"""
Opening Range Breakout (ORB) -- proprietary strategy.

Core thesis:
  The first full trading hour of the London session (07:00-07:59 UTC) defines
  the "opening range" -- the high and low established as the dominant market
  participants arrive. This range captures the initial battle between buyers
  and sellers with full institutional participation.

  When price subsequently BREAKS and CLOSES beyond the opening range boundary,
  it signals that one side has definitively won the opening battle. The
  breakout direction carries strong momentum follow-through.

  BUY: next bar after opening range CLOSES above ORB high
  SELL: next bar after opening range CLOSES below ORB low

  Filters:
  - ORB must be at least min_orb_atr x ATR wide (too narrow = indecision, no edge)
  - ORB must be at most max_orb_atr x ATR wide (too wide = already volatile, edge fades)
  - Breakout bar must close beyond ORB by at least min_break x ATR
  - Macro direction filter (daily EMA)

  This is fundamentally different from all prior strategies:
  - SHD/ARS/PDL: fade a level that was briefly exceeded
  - ORB: follow a level that was cleanly broken and held
  - The opening range is the equilibrium zone; a sustained break = momentum
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

def run_orb(b1h, b1d, cfg):
    min_orb_atr   = cfg["min_orb_atr"]    # ORB must be at least this wide (ATR mult)
    max_orb_atr   = cfg["max_orb_atr"]    # ORB must not be wider than this (ATR mult)
    min_break     = cfg["min_break"]      # close must exceed ORB by at least this (ATR mult)
    max_entry_h   = cfg.get("max_entry_h", 12)  # last London hour to enter (inclusive)
    cooldown_bars = cfg.get("cooldown_bars", 24)
    tp_rr         = cfg["tp_rr"]
    atr_mult      = cfg["atr_mult_sl"]
    use_macro     = cfg.get("macro_filter", False)
    macro_p       = cfg.get("macro_ema_period", 20)

    atr1h = _atr(b1h, 14)

    if use_macro and b1d:
        ema_d = _ema([b["close"] for b in b1d], macro_p)
        ep_d  = [b["epoch"] for b in b1d]
    else:
        ema_d = ep_d = None

    signals    = []
    last_sig_i = -999

    # Pre-index: for each bar, find if it's the London opening bar (h==7)
    # Build a map: day_midnight -> (orb_high, orb_low, orb_atr) for that day
    orb_map = {}
    for i, bar in enumerate(b1h):
        h = (bar["epoch"] % 86400) // 3600
        if h == 7:
            day_midnight = (bar["epoch"] // 86400) * 86400
            atr_val = atr1h[i]
            if atr_val and atr_val > 0:
                orb_map[day_midnight] = (bar["high"], bar["low"], atr_val, i)

    for i in range(2, len(b1h)):
        bar   = b1h[i]
        epoch = bar["epoch"]
        h     = (epoch % 86400) // 3600

        # Only fire during London/NY overlap (08:00-max_entry_h UTC)
        if not (8 <= h <= max_entry_h):
            continue

        if i - last_sig_i < cooldown_bars:
            continue

        atr_val = atr1h[i]
        if atr_val is None or atr_val <= 0:
            continue

        day_midnight = (epoch // 86400) * 86400
        if day_midnight not in orb_map:
            continue

        orb_high, orb_low, orb_atr, orb_bar_i = orb_map[day_midnight]
        orb_size = orb_high - orb_low

        # Skip if ORB is too narrow or too wide
        if orb_size < min_orb_atr * orb_atr:
            continue
        if orb_size > max_orb_atr * orb_atr:
            continue

        # Macro filter
        allow_long = allow_short = True
        if ema_d is not None:
            k = bisect.bisect_right(ep_d, epoch) - 1
            if k >= 1 and ema_d[k] is not None and ema_d[k - 1] is not None:
                macro_up    = ema_d[k] > ema_d[k - 1]
                allow_long  = macro_up
                allow_short = not macro_up

        sl = atr_val * atr_mult
        tp = sl * tp_rr

        # BUY breakout: bar closes above ORB high by at least min_break x ATR
        if (bar["close"] > orb_high and
                bar["close"] - orb_high >= min_break * atr_val and
                bar["close"] > bar["open"] and
                allow_long):
            signals.append((i, Signal("BUY", sl_pips=sl, tp_pips=tp,
                                      reason=f"ORB_buy")))
            last_sig_i = i

        # SELL breakout: bar closes below ORB low
        elif (bar["close"] < orb_low and
                  orb_low - bar["close"] >= min_break * atr_val and
                  bar["close"] < bar["open"] and
                  allow_short):
            signals.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp,
                                      reason=f"ORB_sell")))
            last_sig_i = i

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
    all_sigs = run_orb(b1h, b1d, cfg)
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

    for min_orb in [0.3, 0.5, 0.7, 1.0]:
        for max_orb in [2.0, 3.0, 4.0]:
            for min_break in [0.0, 0.1, 0.2]:
                for max_entry_h in [10, 12, 15]:
                    for tp_rr in [1.5, 2.0, 3.0]:
                        for atr_mult in [1.0, 1.5, 2.0]:
                            for macro in [False, True]:
                                if min_orb >= max_orb:
                                    continue
                                cfg = dict(
                                    min_orb_atr=min_orb,
                                    max_orb_atr=max_orb,
                                    min_break=min_break,
                                    max_entry_h=max_entry_h,
                                    cooldown_bars=24,
                                    tp_rr=tp_rr,
                                    atr_mult_sl=atr_mult,
                                    macro_filter=macro,
                                    macro_ema_period=20,
                                )
                                label = (
                                    f"ORB orb[{min_orb}-{max_orb}]ATR "
                                    f"brk>{min_break}ATR entry<=h{max_entry_h} "
                                    f"RR{tp_rr} ATRx{atr_mult} "
                                    f"{'MACRO' if macro else 'free'}"
                                )
                                sigs = run_orb(train_b1h, b1d, cfg)
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
    print("Opening Range Breakout (ORB) -- 1H -- 4-window walk-forward")
    print(f"BE@1R  max_hold=48H  {DAYS}-day dataset")
    print("Thesis: London 07:00 bar defines equilibrium; sustained break = momentum")
    print("ORB is a FOLLOW signal (opposite of ARS/PDL which are FADE signals)")
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
            print(f"  No profitable ORB configs found for {sym}.\n")
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
            print(f"  No ORB config passed 3+ windows for {sym}.")

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
        print("  No ORB strategy passed 3+ windows on any symbol.")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
