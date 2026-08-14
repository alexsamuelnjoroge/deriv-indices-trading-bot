"""
Order Flow Balance Shift (OFBS) -- proprietary strategy.

Core thesis:
  Every bar tells us not just WHERE price went but WHERE WITHIN THE BAR price
  ended up. A bar that opens at the bottom and closes at the top = buyers
  dominated the entire bar. A bar that opened, went far up, then collapsed back
  to the middle = sellers fought back and won the second half.

  We capture this with the "Close Position Factor" (CPF):
    CPF = (close - low) / (high - low)
    CPF = 1.0 → close at the absolute high = pure buyer dominance
    CPF = 0.0 → close at the absolute low = pure seller dominance
    CPF = 0.5 → close exactly in the middle = equilibrium

  Track the CUMULATIVE CPF average over N bars. This gives us a "net order
  flow balance" — how strongly buyers have been dominating the tape.

  The SIGNAL is not the current CPF level, but the SHIFT:
    When CPF average drops from BULL ZONE (>0.65) to NEUTRAL or BEAR ZONE (<0.45)
    within M bars → institutional selling has entered to absorb buying
    → SELL signal

    When CPF average rises from BEAR ZONE (<0.35) to NEUTRAL or BULL ZONE (>0.55)
    → institutional buying has absorbed selling → BUY signal

  This is structurally different from:
    WDR: measures WICK BALANCE (upper vs lower wick ratio — where rejection sits)
    CBVE: measures BODY SIZE (how much of the bar is directional)
    RVD: measures CLOSE-TO-CLOSE velocity (ground covered per bar)
    OFBS: measures CLOSE POSITION within the bar (buyer/seller intensity per bar)

  OFBS detects the ACCUMULATED SHIFT in intensity, not individual bar events.

Signal (1H bars):
  1. Compute CPF for each bar: (close - low) / (high - low)
  2. Track rolling average CPF over avg_period bars
  3. Track rolling average CPF over a RECENT shorter window (shift_period bars)
  4. SELL: recent CPF avg dropped from bull_zone to bear_zone within shift_period
           AND the longer-term avg was in bull_zone (confirming the prior state)
  5. BUY: mirror
  6. SL: ATR × mult, TP: RR × SL
  7. Cooldown, macro filter

Alternative formulation (tested as a variant):
  Direct trigger: when CPF[i] spikes below bear_thresh after avg_period bars
  of CPF > bull_thresh → single-bar intensity shift (faster signal, more noise)
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


# ── CPF helper ────────────────────────────────────────────────────────────────

def _cpf(bar):
    """Close Position Factor: where in the bar's range did price close?"""
    rng = bar["high"] - bar["low"]
    if rng <= 0:
        return 0.5  # flat bar = neutral
    return (bar["close"] - bar["low"]) / rng


# ── Signal generator ──────────────────────────────────────────────────────────

def run_ofbs(b1h, b1d, cfg):
    avg_period     = cfg["avg_period"]     # rolling CPF window for prior state
    bull_thresh    = cfg["bull_thresh"]    # CPF avg must be above this to be in bull state
    bear_thresh    = cfg["bear_thresh"]    # CPF current/recent must fall below this to trigger
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

    # Precompute CPF for all bars
    cpf_vals = [_cpf(b) for b in b1h]

    signals    = []
    last_sig_i = -999
    start      = avg_period + 5

    for i in range(start, len(b1h)):
        if i - last_sig_i < cooldown_bars:
            continue

        atr_val = atr1h[i]
        if atr_val is None or atr_val <= 0:
            continue

        # Rolling CPF average over the prior avg_period bars (excluding current)
        prior_cpf = [cpf_vals[j] for j in range(i - avg_period, i)]
        avg_prior  = sum(prior_cpf) / len(prior_cpf)

        cur_cpf = cpf_vals[i]

        bar   = b1h[i]
        epoch = bar["epoch"]

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

        # SELL: prior state was bull (buyers dominating), current bar = seller spike
        if (avg_prior > bull_thresh and
                cur_cpf < bear_thresh and
                allow_short):
            signals.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp,
                                      reason=f"OFBS_sell_avg{avg_prior:.2f}->cpf{cur_cpf:.2f}")))
            last_sig_i = i

        # BUY: prior state was bear (sellers dominating), current bar = buyer spike
        elif (avg_prior < (1 - bull_thresh) and
                cur_cpf > (1 - bear_thresh) and
                allow_long):
            signals.append((i, Signal("BUY", sl_pips=sl, tp_pips=tp,
                                      reason=f"OFBS_buy_avg{avg_prior:.2f}->cpf{cur_cpf:.2f}")))
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
    all_sigs = run_ofbs(b1h, b1d, cfg)
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

    for avg_period in [5, 10, 15]:
        for bull_t in [0.55, 0.60, 0.65]:
            for bear_t in [0.30, 0.35, 0.40]:
                for cooldown in [12, 24]:
                    for tp_rr in [1.5, 2.0, 3.0]:
                        for atr_mult in [1.0, 1.5, 2.0]:
                            for macro in [False, True]:
                                cfg = dict(
                                    avg_period=avg_period,
                                    bull_thresh=bull_t,
                                    bear_thresh=bear_t,
                                    cooldown_bars=cooldown,
                                    tp_rr=tp_rr,
                                    atr_mult_sl=atr_mult,
                                    macro_filter=macro,
                                    macro_ema_period=20,
                                )
                                label = (
                                    f"OFBS avg{avg_period} "
                                    f"bull>{bull_t} bear<{bear_t} "
                                    f"cool{cooldown} RR{tp_rr} ATRx{atr_mult} "
                                    f"{'MACRO' if macro else 'free'}"
                                )
                                sigs = run_ofbs(train_b1h, b1d, cfg)
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
    print("Order Flow Balance Shift (OFBS) -- 1H -- 4-window walk-forward")
    print(f"BE@1R  max_hold=48H  {DAYS}-day dataset")
    print("Thesis: close position within bar = order flow intensity per bar")
    print("CPF avg drops from bull zone to bear = institutional selling absorbed retail buying")
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
            print(f"  No profitable OFBS configs found for {sym}.\n")
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
            print(f"  No OFBS config passed 3+ windows for {sym}.")

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
        print("  No OFBS strategy passed 3+ windows on any symbol.")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
