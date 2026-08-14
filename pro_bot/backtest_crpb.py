"""
Compression Range Position Bias (CRPB) -- proprietary strategy.

Core thesis:
  This extends VRT (Volatility Regime Transition). VRT detects WHEN a volatility
  expansion begins but is agnostic about direction — it follows the first bar.

  CRPB adds a critical insight: WHERE within the compression range is price sitting
  DURING the quiet period predicts which direction the expansion will take BEFORE
  the first expansion bar even forms.

  Think of a coiled spring: if you push a spring down from the TOP, it will release
  upward. If you push from the BOTTOM, it releases downward. The position of the
  compression WITHIN the quiet range is the equivalent — it reveals which side has
  been winning the quiet battle.

  During compression:
    - If price consistently closes in the UPPER zone (>60% of compression range):
      buyers are in control during the quiet phase → they will release upward → BUY
    - If price consistently closes in the LOWER zone (<40% of compression range):
      sellers are in control during the quiet phase → release downward → SELL

  Combined signal: VRT trigger (compression ends) + CRPB bias (direction predicted
  from position during compression) → only take signals where BOTH agree.

  This can OVERRIDE the body direction filter in VRT: even if the first expansion
  bar is bearish, if CRPB says the range position strongly predicted a bull move,
  we ignore the bearish first bar (or wait for confirmation).

  In standalone mode, CRPB fires at the FIRST expansion bar and overrides the
  body-direction signal when the range position is strong enough.

Signal (1H bars):
  1. Detect compression: same as VRT — ATR below ATR_SMA for min_quiet_bars
  2. During compression, compute:
       position[j] = (close[j] - min_low_of_compression) / (max_high_of_compression - min_low)
  3. Compute average position over all compression bars
  4. At VRT trigger (first bar with ATR > ATR_SMA):
       If avg_position > upper_bias_thresh → BUY (regardless of first bar direction)
       If avg_position < lower_bias_thresh → SELL (regardless of first bar direction)
       Otherwise: use first bar's body direction (standard VRT behavior)
  5. SL: ATR × mult, TP: RR × SL
  6. Cooldown, macro filter

Difference from VRT:
  VRT always follows the FIRST EXPANSION BAR direction.
  CRPB uses the COMPRESSION PHASE position to predict direction — which means it can
  be correct even when the first expansion bar is a false start in the wrong direction.

  This makes CRPB potentially HIGHER ACCURACY but potentially LOWER FREQUENCY
  if the position filter is strict.
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


def _atr_sma(atr_series, period):
    n   = len(atr_series)
    out = [None] * n
    for i in range(period - 1, n):
        vals = [atr_series[j] for j in range(i - period + 1, i + 1)
                if atr_series[j] is not None]
        if len(vals) == period:
            out[i] = sum(vals) / period
    return out


# ── Signal generator ──────────────────────────────────────────────────────────

def run_crpb(b1h, b1d, cfg):
    atr_sma_period   = cfg["atr_sma_period"]
    min_quiet_bars   = cfg["min_quiet_bars"]
    upper_bias       = cfg["upper_bias"]      # compression avg position must be > this → BUY
    lower_bias       = cfg["lower_bias"]      # compression avg position must be < this → SELL
    min_body_ratio   = cfg.get("min_body_ratio", 0.2)
    cooldown_bars    = cfg.get("cooldown_bars", 24)
    tp_rr            = cfg["tp_rr"]
    atr_mult         = cfg["atr_mult_sl"]
    use_macro        = cfg.get("macro_filter", False)
    macro_p          = cfg.get("macro_ema_period", 20)
    bias_overrides   = cfg.get("bias_overrides_body", True)  # bias direction overrides first bar

    atr1h     = _atr(b1h, 14)
    atr_sma_v = _atr_sma(atr1h, atr_sma_period)

    if use_macro and b1d:
        ema_d = _ema([b["close"] for b in b1d], macro_p)
        ep_d  = [b["epoch"] for b in b1d]
    else:
        ema_d = ep_d = None

    signals    = []
    last_sig_i = -999

    quiet_count   = 0
    comp_highs    = []   # highs of bars during compression phase
    comp_lows     = []   # lows
    comp_closes   = []   # closes

    start = atr_sma_period + 5

    for i in range(start, len(b1h)):
        bar   = b1h[i]
        epoch = bar["epoch"]

        fa = atr1h[i]
        sm = atr_sma_v[i]
        fa_prev = atr1h[i - 1] if i > 0 else None
        sm_prev = atr_sma_v[i - 1] if i > 0 else None

        if fa is None or sm is None or fa_prev is None or sm_prev is None:
            quiet_count = 0
            comp_highs = comp_lows = comp_closes = []
            continue

        # Track quiet state and accumulate compression data
        if fa_prev <= sm_prev:
            quiet_count += 1
            comp_highs.append(b1h[i - 1]["high"])
            comp_lows.append(b1h[i - 1]["low"])
            comp_closes.append(b1h[i - 1]["close"])
        else:
            quiet_count = 0
            comp_highs = []
            comp_lows  = []
            comp_closes = []

        # Transition trigger: previous bar was quiet, current is NOT
        if fa <= sm:
            continue
        if fa_prev > sm_prev:
            continue
        if quiet_count < min_quiet_bars:
            continue

        if i - last_sig_i < cooldown_bars:
            quiet_count = 0
            comp_highs = comp_lows = comp_closes = []
            continue

        # Compute compression range position bias
        if not comp_highs:
            quiet_count = 0
            continue

        comp_high = max(comp_highs)
        comp_low  = min(comp_lows)
        comp_rng  = comp_high - comp_low

        if comp_rng <= 0:
            quiet_count = 0
            comp_highs = comp_lows = comp_closes = []
            continue

        # Average position of closes within the compression range
        positions = [(c - comp_low) / comp_rng for c in comp_closes]
        avg_pos   = sum(positions) / len(positions)

        # Body check on the trigger bar
        body = abs(bar["close"] - bar["open"])
        if body < min_body_ratio * fa:
            quiet_count = 0
            comp_highs = comp_lows = comp_closes = []
            continue

        # Macro gate
        allow_long = allow_short = True
        if ema_d is not None:
            k = bisect.bisect_right(ep_d, epoch) - 1
            if k >= 1 and ema_d[k] is not None and ema_d[k - 1] is not None:
                macro_up    = ema_d[k] > ema_d[k - 1]
                allow_long  = macro_up
                allow_short = not macro_up

        sl = fa * atr_mult
        tp = sl * tp_rr

        # Determine direction
        bar_bull = bar["close"] > bar["open"]

        if avg_pos > upper_bias and bias_overrides:
            # Strong upper bias → BUY regardless of first bar direction
            go_long = True
        elif avg_pos < lower_bias and bias_overrides:
            # Strong lower bias → SELL regardless of first bar direction
            go_long = False
        else:
            # Moderate or no bias → use first bar's body direction (standard VRT)
            go_long = bar_bull

        if go_long and allow_long:
            signals.append((i, Signal("BUY", sl_pips=sl, tp_pips=tp,
                                      reason=f"CRPB_buy_pos{avg_pos:.2f}_q{quiet_count}")))
            last_sig_i = i
        elif not go_long and allow_short:
            signals.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp,
                                      reason=f"CRPB_sell_pos{avg_pos:.2f}_q{quiet_count}")))
            last_sig_i = i

        quiet_count = 0
        comp_highs = comp_lows = comp_closes = []

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
    all_sigs = run_crpb(b1h, b1d, cfg)
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

    for sma_p in [20, 30, 50]:
        for quiet_bars in [3, 5, 8]:
            for upper_b in [0.55, 0.60, 0.65]:
                for lower_b in [0.35, 0.40, 0.45]:
                    if lower_b >= upper_b:
                        continue
                    for bias_override in [True, False]:
                        for cooldown in [12, 24]:
                            for tp_rr in [1.5, 2.0, 3.0]:
                                for atr_mult in [1.0, 1.5, 2.0]:
                                    for macro in [False, True]:
                                        cfg = dict(
                                            atr_sma_period=sma_p,
                                            min_quiet_bars=quiet_bars,
                                            upper_bias=upper_b,
                                            lower_bias=lower_b,
                                            min_body_ratio=0.2,
                                            cooldown_bars=cooldown,
                                            tp_rr=tp_rr,
                                            atr_mult_sl=atr_mult,
                                            macro_filter=macro,
                                            macro_ema_period=20,
                                            bias_overrides_body=bias_override,
                                        )
                                        ov = "OVR" if bias_override else "BODY"
                                        label = (
                                            f"CRPB sma{sma_p} q{quiet_bars} "
                                            f"up>{upper_b} lo<{lower_b} {ov} "
                                            f"cool{cooldown} RR{tp_rr} ATRx{atr_mult} "
                                            f"{'MACRO' if macro else 'free'}"
                                        )
                                        sigs = run_crpb(train_b1h, b1d, cfg)
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
    print("Compression Range Position Bias (CRPB) -- 1H -- 4-window walk-forward")
    print(f"BE@1R  max_hold=48H  {DAYS}-day dataset")
    print("Thesis: WHERE price sits during compression predicts expansion direction")
    print("Extension of VRT: position within the quiet range = directional spring tension")
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
            print(f"  No profitable CRPB configs found for {sym}.\n")
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
            print(f"  No CRPB config passed 3+ windows for {sym}.")

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
        print("  No CRPB strategy passed 3+ windows on any symbol.")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
