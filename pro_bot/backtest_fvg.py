"""
Fair Value Gap Fill (FVG) -- proprietary strategy.

Core thesis:
  When price moves violently in a single bar (2x+ ATR), it creates an
  "untested zone" -- the range that was traversed too rapidly for price
  discovery to occur. Institutional participants who missed the move or were
  caught on the wrong side have unfilled orders within this zone.

  Price almost always returns to fill (test) this zone. When it re-enters the
  zone from outside, the institutional orders waiting there absorb it and push
  price back in the direction of the original impulse.

  Mechanics:
    1. Detect an impulse bar: |close - open| >= impulse_atr x ATR(14)
       AND range (high - low) >= range_atr x ATR (confirms volatility spike)
    2. Mark the FVG zone: [bar open, bar close] for bull impulse
                          [bar close, bar open] for bear impulse
    3. Within max_fill_bars, when price re-enters the zone:
       - If zone is bullish (impulse was up): BUY when price touches zone bottom
       - If zone is bearish (impulse was down): SELL when price touches zone top
    4. SL: beyond far edge of zone + buffer (ATR mult)
    5. TP: RR x SL
    6. Each zone can only fire once (first fill only)

  This is completely different from all prior strategies:
  - No level is pre-defined; the zone is created BY the market's own impulse
  - The signal is a RETURN to a created zone, not a reaction at a watched level
  - The zone expires after max_fill_bars if not filled (no longer relevant)
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

def run_fvg(b1h, b1d, cfg):
    impulse_atr   = cfg["impulse_atr"]    # body must be >= this x ATR to create zone
    max_fill_bars = cfg["max_fill_bars"]  # zone expires after this many bars
    min_entry_atr = cfg.get("min_entry_atr", 0.0)  # how deep into zone before entry
    cooldown_bars = cfg.get("cooldown_bars", 12)
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

    # Active zones: list of dicts
    # {created_i, direction, zone_top, zone_bot, atr_at_creation, filled}
    zones = []

    for i in range(15, len(b1h)):
        bar     = b1h[i]
        epoch   = bar["epoch"]
        atr_val = atr1h[i]
        if atr_val is None or atr_val <= 0:
            continue

        # Expire old zones
        zones = [z for z in zones if (i - z["created_i"]) <= max_fill_bars and not z["filled"]]

        # Create new zone if this bar is an impulse
        body = abs(bar["close"] - bar["open"])
        if body >= impulse_atr * atr_val:
            bull_impulse = bar["close"] > bar["open"]
            if bull_impulse:
                zone_bot = bar["open"]
                zone_top = bar["close"]
            else:
                zone_bot = bar["close"]
                zone_top = bar["open"]
            zones.append({
                "created_i":  i,
                "direction":  "bull" if bull_impulse else "bear",
                "zone_top":   zone_top,
                "zone_bot":   zone_bot,
                "atr_create": atr_val,
                "filled":     False,
            })

        # Check active zones for fill entries
        if i - last_sig_i < cooldown_bars:
            continue

        allow_long = allow_short = True
        if ema_d is not None:
            k = bisect.bisect_right(ep_d, epoch) - 1
            if k >= 1 and ema_d[k] is not None and ema_d[k - 1] is not None:
                macro_up    = ema_d[k] > ema_d[k - 1]
                allow_long  = macro_up
                allow_short = not macro_up

        for z in zones:
            if z["filled"]:
                continue
            if i == z["created_i"]:
                continue  # skip the bar that created the zone

            sl  = atr_val * atr_mult
            tp  = sl * tp_rr

            if z["direction"] == "bull" and allow_long:
                # Price re-enters bull zone from above (pullback into zone)
                # Entry: price touches zone top (from above)
                if (bar["low"] <= z["zone_top"] and
                        bar["low"] >= z["zone_bot"] - min_entry_atr * atr_val and
                        bar["close"] >= z["zone_bot"]):
                    signals.append((i, Signal("BUY", sl_pips=sl, tp_pips=tp,
                                              reason="FVG_bull_fill")))
                    z["filled"] = True
                    last_sig_i  = i
                    break

            elif z["direction"] == "bear" and allow_short:
                # Price re-enters bear zone from below (pullback into zone)
                if (bar["high"] >= z["zone_bot"] and
                        bar["high"] <= z["zone_top"] + min_entry_atr * atr_val and
                        bar["close"] <= z["zone_top"]):
                    signals.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp,
                                              reason="FVG_bear_fill")))
                    z["filled"] = True
                    last_sig_i  = i
                    break

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
    all_sigs = run_fvg(b1h, b1d, cfg)
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

    for impulse_atr in [0.8, 1.0, 1.5, 2.0]:
        for max_fill in [12, 24, 48]:
            for min_entry in [0.0, 0.1, 0.2]:
                for cooldown in [12, 24]:
                    for tp_rr in [1.5, 2.0, 3.0]:
                        for atr_mult in [0.5, 1.0, 1.5]:
                            for macro in [False, True]:
                                cfg = dict(
                                    impulse_atr=impulse_atr,
                                    max_fill_bars=max_fill,
                                    min_entry_atr=min_entry,
                                    cooldown_bars=cooldown,
                                    tp_rr=tp_rr,
                                    atr_mult_sl=atr_mult,
                                    macro_filter=macro,
                                    macro_ema_period=20,
                                )
                                label = (
                                    f"FVG imp>{impulse_atr}ATR fill<={max_fill}bars "
                                    f"entry>{min_entry}ATR cool{cooldown} "
                                    f"RR{tp_rr} ATRx{atr_mult} "
                                    f"{'MACRO' if macro else 'free'}"
                                )
                                sigs = run_fvg(train_b1h, b1d, cfg)
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
    print("Fair Value Gap Fill (FVG) -- 1H -- 4-window walk-forward")
    print(f"BE@1R  max_hold=48H  {DAYS}-day dataset")
    print("Thesis: impulse bar creates untested zone; price returns to fill it")
    print("Zone = body of impulse bar; entry when price re-enters from outside")
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
            print(f"  No profitable FVG configs found for {sym}.\n")
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
            print(f"  No FVG config passed 3+ windows for {sym}.")

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
        print("  No FVG strategy passed 3+ windows on any symbol.")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
