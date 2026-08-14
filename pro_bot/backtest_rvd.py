"""
Return Velocity Decay (RVD) -- proprietary strategy.

Core thesis:
  Genuine institutional accumulation produces SUSTAINED close-to-close momentum:
  each bar carries the same or greater directional velocity. The amount of GROUND
  COVERED per bar stays constant or grows — because institutions keep buying.

  When close-to-close returns start SHRINKING over consecutive bars in the same
  direction, smart money is withdrawing. The trend is running on retail FOMO —
  momentum that gets smaller with each push. The final, smallest push is the
  precise moment to enter counter-trend.

  Mathematically: return[i] < return[i-1] < return[i-2] (magnitude strictly
  decreasing) over a streak of same-direction closes.

  This is structurally different from all prior strategies:
    CBVE: measures body SIZE (directional body as fraction of full bar range)
    WDR:  measures wick BALANCE (upper vs lower wick ratio)
    CRD:  measures range SIZE (current bar vs recent average)
    RVD:  measures close-to-close VELOCITY (actual ground covered per bar, ATR-normalized)

Signal (1H bars):
  1. Compute ATR(14)-normalized close-to-close return per bar:
       ret[i] = (close[i] - close[i-1]) / atr[i]
  2. Build a consecutive same-direction streak ending at the current bar
  3. Trigger when:
       a. Streak length >= min_streak bars (sustained move — not a 2-bar blip)
       b. The last decay_bars returns are EACH SMALLER in magnitude than the previous
          (strictly decelerating: |ret[i]| < |ret[i-1]| < ... < |ret[i-decay_bars+1]|)
       c. Each return magnitude >= min_ret (minimum threshold — avoids noise bars)
  4. Enter COUNTER to the streak direction on trigger bar
  5. SL: ATR * mult, TP: RR * SL
  6. Cooldown + macro filter

Intuition:
  A ball rolling up a hill decelerates as gravity takes over. Each step is
  shorter than the last. The moment just before it stops is the optimal entry
  for the downward journey. RVD finds the market's equivalent.

  Critical: macro filter prevents fading a macro trend. In a bull regime,
  this strategy only fades exhausted bull runs that are against the macro —
  or takes longs when bearish streaks decelerate.
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

def run_rvd(b1h, b1d, cfg):
    min_streak    = cfg["min_streak"]    # minimum consecutive same-direction bars
    decay_bars    = cfg["decay_bars"]    # consecutive decelerating bars required (<=min_streak)
    min_ret       = cfg["min_ret"]       # minimum normalized return to count (avoids noise)
    cooldown_bars = cfg.get("cooldown_bars", 24)
    tp_rr         = cfg["tp_rr"]
    atr_mult      = cfg["atr_mult_sl"]
    use_macro     = cfg.get("macro_filter", False)
    macro_p       = cfg.get("macro_ema_period", 20)

    atr1h  = _atr(b1h, 14)
    closes = [b["close"] for b in b1h]

    if use_macro and b1d:
        ema_d = _ema([b["close"] for b in b1d], macro_p)
        ep_d  = [b["epoch"] for b in b1d]
    else:
        ema_d = ep_d = None

    # Precompute ATR-normalized returns
    norm_ret = [None] * len(b1h)
    for i in range(1, len(b1h)):
        a = atr1h[i]
        if a and a > 0:
            norm_ret[i] = (closes[i] - closes[i - 1]) / a

    signals    = []
    last_sig_i = -999
    start      = max(min_streak + 5, 20)

    for i in range(start, len(b1h)):
        if i - last_sig_i < cooldown_bars:
            continue

        r_i = norm_ret[i]
        if r_i is None or abs(r_i) < min_ret:
            continue

        atr_val = atr1h[i]
        if atr_val is None or atr_val <= 0:
            continue

        # Build the streak: look back from i, same direction, magnitude >= min_ret
        streak_rets = [r_i]
        for j in range(i - 1, max(i - min_streak - 2, 0), -1):
            rj = norm_ret[j]
            if rj is None:
                break
            # same sign and large enough
            if rj * r_i <= 0:  # different direction
                break
            if abs(rj) < min_ret:
                break
            streak_rets.append(rj)

        if len(streak_rets) < min_streak:
            continue

        # streak_rets[0] = ret[i] (most recent), streak_rets[1] = ret[i-1], etc.
        # Check decay in the most recent decay_bars returns (indices 0..decay_bars-1)
        if len(streak_rets) < decay_bars:
            continue

        # Magnitudes must be strictly decreasing (streak_rets[0] < streak_rets[1] < ...)
        magnitudes = [abs(streak_rets[k]) for k in range(decay_bars)]
        is_decaying = all(magnitudes[k] < magnitudes[k + 1]
                         for k in range(decay_bars - 1))

        if not is_decaying:
            continue

        epoch = b1h[i]["epoch"]

        allow_long = allow_short = True
        if ema_d is not None:
            k = bisect.bisect_right(ep_d, epoch) - 1
            if k >= 1 and ema_d[k] is not None and ema_d[k - 1] is not None:
                macro_up    = ema_d[k] > ema_d[k - 1]
                allow_long  = macro_up
                allow_short = not macro_up

        sl = atr_val * atr_mult
        tp = sl * tp_rr

        streak_bull = r_i > 0  # streak was going up — fade means SELL

        if streak_bull and allow_short:
            signals.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp,
                                      reason=f"RVD_sell_str{len(streak_rets)}_d{decay_bars}")))
            last_sig_i = i
        elif not streak_bull and allow_long:
            signals.append((i, Signal("BUY", sl_pips=sl, tp_pips=tp,
                                      reason=f"RVD_buy_str{len(streak_rets)}_d{decay_bars}")))
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
    all_sigs = run_rvd(b1h, b1d, cfg)
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

    for min_streak in [3, 4, 5]:
        for decay_bars in [2, 3]:
            if decay_bars > min_streak:
                continue
            for min_ret in [0.05, 0.10, 0.20]:
                for cooldown in [12, 24]:
                    for tp_rr in [1.5, 2.0, 3.0]:
                        for atr_mult in [1.0, 1.5, 2.0]:
                            for macro in [False, True]:
                                cfg = dict(
                                    min_streak=min_streak,
                                    decay_bars=decay_bars,
                                    min_ret=min_ret,
                                    cooldown_bars=cooldown,
                                    tp_rr=tp_rr,
                                    atr_mult_sl=atr_mult,
                                    macro_filter=macro,
                                    macro_ema_period=20,
                                )
                                label = (
                                    f"RVD str{min_streak} decay{decay_bars} "
                                    f"minR{min_ret} cool{cooldown} "
                                    f"RR{tp_rr} ATRx{atr_mult} "
                                    f"{'MACRO' if macro else 'free'}"
                                )
                                sigs = run_rvd(train_b1h, b1d, cfg)
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
    print("Return Velocity Decay (RVD) -- 1H -- 4-window walk-forward")
    print(f"BE@1R  max_hold=48H  {DAYS}-day dataset")
    print("Thesis: close-to-close velocity shrinks bar-by-bar = institutional fuel depleted")
    print("Signal: N consecutive same-direction closes with each smaller than last => counter-trend")
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
            print(f"  No profitable RVD configs found for {sym}.\n")
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
            print(f"  No RVD config passed 3+ windows for {sym}.")

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
        print("  No RVD strategy passed 3+ windows on any symbol.")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
