"""
Progressive Level Testing (PLT) -- proprietary strategy.

Core thesis:
  When price makes SUCCESSIVE ATTEMPTS at a resistance or support level, each
  attempt reaching a LOWER HIGH (at resistance) or HIGHER LOW (at support) than
  the previous, institutions are DISTRIBUTING into the attempts, not accumulating.

  This is the fingerprint of smart money building a position against retail:
    - Retail sees "price keeps testing this level" and expects a breakout
    - Institutions use each test to sell into retail buying pressure
    - Each successive test hits less high (sellers absorbing more each time)
    - The declining reach IS the signal — the level is growing stronger, not weaker

  This is fundamentally different from SHD (single-bar level breach).
  PLT requires 3+ SEPARATE ATTEMPTS over multiple bars — a multi-bar institutional
  accumulation/distribution pattern that no single-bar strategy can see.

Signal (1H bars):
  At RESISTANCE (for SELL signal):
    1. Find the N-bar structural high (the "resistance level")
    2. Track successive bars that approach within probe_zone × ATR of the resistance
    3. Count: if each successive approach hits a LOWER high than the previous approach
       AND there are >= min_tests such approaches
    4. After min_tests confirmed declining approaches: SELL
    5. Reset: if any approach EXCEEDS the previous approach's high, reset the counter

  At SUPPORT (mirror, for BUY signal):
    - Successive approaches where each reaches a HIGHER low than the previous
    - Institutions absorbing selling pressure at support
    - After min_tests confirmed rising approach lows: BUY

  SL: ATR × mult above the resistance (or below the support)
  TP: RR × SL
  Cooldown: block re-entry for cooldown_bars
  Macro filter: daily EMA direction must agree

Why declining reach matters:
  In a genuine breakout setup, price should reach HIGHER on each test (accumulating).
  Declining reach means each rally is being sold off SOONER — the level is defended
  by increasing institutional selling. The declining reach is the institution's
  fingerprint, revealing distribution before the move.
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

def run_plt(b1h, b1d, cfg):
    level_lookback = cfg["level_lookback"]  # bars to look back to define the structural level
    probe_zone     = cfg["probe_zone"]      # ATR multiple — how close to level counts as a test
    min_tests      = cfg["min_tests"]       # minimum number of declining tests required
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

    signals    = []
    last_sig_i = -999
    start      = level_lookback + 10

    # Resistance test tracker: list of successive highs that approached resistance
    # Support test tracker: list of successive lows that approached support
    res_tests = []   # list of (bar_idx, bar_high) for successive resistance tests
    sup_tests = []   # list of (bar_idx, bar_low) for successive support tests
    res_level = None
    sup_level = None

    for i in range(start, len(b1h)):
        if i - last_sig_i < cooldown_bars:
            continue

        bar   = b1h[i]
        epoch = bar["epoch"]

        atr_val = atr1h[i]
        if atr_val is None or atr_val <= 0:
            continue

        # Define structural levels from the lookback window (excluding current bar)
        lookback = b1h[i - level_lookback: i]
        new_res = max(b["high"] for b in lookback)
        new_sup = min(b["low"]  for b in lookback)

        # If the structural level shifted significantly, reset trackers
        if res_level is None or abs(new_res - res_level) > 2 * atr_val:
            res_level = new_res
            res_tests = []
        if sup_level is None or abs(new_sup - sup_level) > 2 * atr_val:
            sup_level = new_sup
            sup_tests = []

        res_level = new_res
        sup_level = new_sup

        # Check if this bar APPROACHED the resistance
        approach_res = bar["high"] >= res_level - probe_zone * atr_val

        if approach_res:
            if not res_tests:
                res_tests.append((i, bar["high"]))
            else:
                prev_high = res_tests[-1][1]
                if bar["high"] < prev_high:
                    # Declining reach — this test didn't go as high as the last
                    res_tests.append((i, bar["high"]))
                elif bar["high"] >= prev_high:
                    # This test exceeded the previous — reset, not declining
                    res_tests = [(i, bar["high"])]

        # Check if this bar APPROACHED the support
        approach_sup = bar["low"] <= sup_level + probe_zone * atr_val

        if approach_sup:
            if not sup_tests:
                sup_tests.append((i, bar["low"]))
            else:
                prev_low = sup_tests[-1][1]
                if bar["low"] > prev_low:
                    # Rising reach — this test didn't go as low as the last
                    sup_tests.append((i, bar["low"]))
                elif bar["low"] <= prev_low:
                    # This test exceeded (went lower) — reset
                    sup_tests = [(i, bar["low"])]

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

        # SELL signal: enough declining resistance tests
        if len(res_tests) >= min_tests and allow_short and not approach_sup:
            # Only fire once per cluster — require at least 1 bar gap from last test
            last_test_i = res_tests[-1][0]
            if i > last_test_i and i - last_sig_i >= cooldown_bars:
                signals.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp,
                                          reason=f"PLT_sell_{len(res_tests)}tests")))
                last_sig_i = i
                res_tests  = []  # reset after firing

        # BUY signal: enough rising support tests
        elif len(sup_tests) >= min_tests and allow_long and not approach_res:
            last_test_i = sup_tests[-1][0]
            if i > last_test_i and i - last_sig_i >= cooldown_bars:
                signals.append((i, Signal("BUY", sl_pips=sl, tp_pips=tp,
                                          reason=f"PLT_buy_{len(sup_tests)}tests")))
                last_sig_i = i
                sup_tests  = []

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
    all_sigs = run_plt(b1h, b1d, cfg)
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

    for level_lb in [20, 30, 50]:
        for probe in [0.5, 1.0, 2.0]:
            for min_t in [2, 3]:
                for cooldown in [12, 24]:
                    for tp_rr in [1.5, 2.0, 3.0]:
                        for atr_mult in [1.0, 1.5, 2.0]:
                            for macro in [False, True]:
                                cfg = dict(
                                    level_lookback=level_lb,
                                    probe_zone=probe,
                                    min_tests=min_t,
                                    cooldown_bars=cooldown,
                                    tp_rr=tp_rr,
                                    atr_mult_sl=atr_mult,
                                    macro_filter=macro,
                                    macro_ema_period=20,
                                )
                                label = (
                                    f"PLT lb{level_lb} probe{probe}ATR "
                                    f"tests>={min_t} cool{cooldown} "
                                    f"RR{tp_rr} ATRx{atr_mult} "
                                    f"{'MACRO' if macro else 'free'}"
                                )
                                sigs = run_plt(train_b1h, b1d, cfg)
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
    print("Progressive Level Testing (PLT) -- 1H -- 4-window walk-forward")
    print(f"BE@1R  max_hold=48H  {DAYS}-day dataset")
    print("Thesis: successive lower highs at resistance = institutional distribution")
    print("Multi-bar pattern: 2+ declining approaches to a structural level => fade")
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
            print(f"  No profitable PLT configs found for {sym}.\n")
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
            print(f"  No PLT config passed 3+ windows for {sym}.")

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
        print("  No PLT strategy passed 3+ windows on any symbol.")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
