"""
Daily Bar Streak Filter (DBSF) -- 1H -- 4-window walk-forward

Core thesis:
  Professional traders know that after N consecutive daily closes in the same
  direction, the NEXT trading day carries elevated mean-reversion probability.
  The daily close-to-close streak is a DIFFERENT lens from CPF, RSI, or MBD:
  it counts CALENDAR DAYS of sustained directional commitment, not intrabar
  positioning or ATR-normalized momentum.

  After a bull streak of N days (each day closed higher than the prior):
    The next London open is biased SHORT -- fade the streak.
  After a bear streak of N days:
    The next London open is biased LONG.

  Entry: the FIRST 1H bar at or after entry_hour UTC on the day following
  the streak completion. One signal per streak completion maximum.
  The daily bar data used is always the PRIOR day's completed bar, so there
  is no lookahead.

  The macro filter is optionally inverted (streak_invert=True) so that it
  aligns streaks with trend continuation rather than mean reversion, testing
  both hypotheses.
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


def _build_daily_streak_map(b1d):
    """
    For each daily bar, compute the streak length and direction at bar close.
    Returns list of (midnight_epoch, streak, direction) sorted by epoch.
    direction: +1 = bull streak, -1 = bear streak, 0 = flat
    """
    result = []
    for i in range(1, len(b1d)):
        if b1d[i]["close"] > b1d[i - 1]["close"]:
            direction = 1
        elif b1d[i]["close"] < b1d[i - 1]["close"]:
            direction = -1
        else:
            direction = 0

        if not result or direction == 0:
            streak = 1 if direction != 0 else 0
        else:
            prev_streak, prev_dir = result[-1][1], result[-1][2]
            streak = prev_streak + 1 if direction == prev_dir else 1

        result.append((b1d[i]["epoch"], streak, direction))

    return result


def run_dbsf(b1h, b1d, cfg):
    min_streak   = cfg["min_streak"]         # consecutive same-direction daily closes
    entry_hour   = cfg.get("entry_hour", 7)  # earliest 1H bar to fire signal (UTC)
    cooldown_bars = cfg.get("cooldown_bars", 24)
    tp_rr        = cfg["tp_rr"]
    atr_mult     = cfg["atr_mult_sl"]
    use_macro    = cfg.get("macro_filter", False)
    macro_p      = cfg.get("macro_ema_period", 20)

    atr1h       = _atr(b1h, 14)
    streak_map  = _build_daily_streak_map(b1d)
    streak_eps  = [row[0] for row in streak_map]  # daily epochs

    if use_macro and b1d:
        ema_d = _ema([b["close"] for b in b1d], macro_p)
        ep_d  = [b["epoch"] for b in b1d]
    else:
        ema_d = ep_d = None

    signals    = []
    last_sig_i = -999
    fired_days = set()

    for i in range(20, len(b1h)):
        if i - last_sig_i < cooldown_bars:
            continue

        bar   = b1h[i]
        epoch = bar["epoch"]
        h     = (epoch % 86400) // 3600
        day   = (epoch // 86400) * 86400

        # Only fire during / after entry_hour on each day
        if h < entry_hour:
            continue
        if day in fired_days:
            continue

        atr_val = atr1h[i]
        if atr_val is None or atr_val <= 0:
            continue

        # Find yesterday's completed daily bar streak
        yesterday_midnight = day - 86400
        k = bisect.bisect_right(streak_eps, yesterday_midnight) - 1
        if k < 0:
            continue
        _, streak, direction = streak_map[k]
        if streak < min_streak or direction == 0:
            continue

        # The streak is complete as of yesterday -- fade it today
        allow_long = allow_short = True
        if ema_d is not None:
            m = bisect.bisect_right(ep_d, epoch) - 1
            if m >= 1 and ema_d[m] is not None and ema_d[m - 1] is not None:
                macro_up    = ema_d[m] > ema_d[m - 1]
                allow_long  = macro_up
                allow_short = not macro_up

        sl = atr_val * atr_mult
        tp = sl * tp_rr

        if direction == 1 and allow_short:
            # Bull streak ended yesterday -- fade: SELL
            signals.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp,
                                      reason=f"DBSF_sell_streak{streak}")))
            fired_days.add(day)
            last_sig_i = i

        elif direction == -1 and allow_long:
            # Bear streak ended yesterday -- fade: BUY
            signals.append((i, Signal("BUY", sl_pips=sl, tp_pips=tp,
                                      reason=f"DBSF_buy_streak{streak}")))
            fired_days.add(day)
            last_sig_i = i

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


def run_wf(b1h, b1d, cfg, label, spread, verbose=True):
    all_sigs = run_dbsf(b1h, b1d, cfg)
    n        = len(b1h)
    passes   = 0

    if verbose:
        print(f"  Config: {label}  [{len(all_sigs)} total signals]")

    for wi, (_, te_pct, ho_pct) in enumerate(WINDOWS):
        c1 = int(n * te_pct)
        c2 = int(n * ho_pct)

        tr_sigs, ho_sigs = _split_sigs(all_sigs, c1, c2)
        tr_s = stats(sim(b1h[:c1],      tr_sigs, spread), min_n=5)
        ho_s = stats(sim(b1h[c1:c2],    ho_sigs, spread), min_n=3)

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


def sweep(b1h, b1d, train_end_idx, spread):
    train_b1h = b1h[:train_end_idx]
    results   = []

    for min_streak in [2, 3, 4, 5]:
        for entry_hour in [7, 9, 13]:
            for cooldown in [12, 24]:
                for tp_rr in [1.5, 2.0, 3.0]:
                    for atr_mult in [1.0, 1.5, 2.0]:
                        for macro in [False, True]:
                            cfg = dict(
                                min_streak=min_streak,
                                entry_hour=entry_hour,
                                cooldown_bars=cooldown,
                                tp_rr=tp_rr,
                                atr_mult_sl=atr_mult,
                                macro_filter=macro,
                                macro_ema_period=20,
                            )
                            label = (
                                f"DBSF streak>={min_streak} entry@{entry_hour}h "
                                f"cool{cooldown} RR{tp_rr} ATRx{atr_mult} "
                                f"{'MACRO' if macro else 'free'}"
                            )
                            sigs = run_dbsf(train_b1h, b1d, cfg)
                            if not sigs:
                                continue
                            trades = sim(train_b1h, sigs, spread)
                            s = stats(trades, min_n=8)
                            if s and s["ev"] > 0:
                                results.append((s["ev"], s["n"], label, cfg))

    results.sort(key=lambda x: -x[0])
    return results


SYMBOLS_TO_TEST = [
    ("frxXAUUSD", SPREADS["frxXAUUSD"]),
    ("frxEURUSD", SPREADS["frxEURUSD"]),
    ("frxGBPUSD", SPREADS["frxGBPUSD"]),
    ("frxUSDJPY", SPREADS["frxUSDJPY"]),
]


async def main():
    import time as _t

    print("=" * 78)
    print("Daily Bar Streak Filter (DBSF) -- 1H -- 4-window walk-forward")
    print(f"BE@1R  max_hold=48H  {DAYS}-day dataset")
    print("Thesis: N consecutive daily closes same direction -> next-day intraday fade")
    print("Streak measured on close-to-close basis; entry at London/NY open 1H bar")
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
            print(f"  No profitable DBSF configs found for {sym}.\n")
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
            print(f"  No DBSF config passed 3+ windows for {sym}.")

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
        print("  No DBSF strategy passed 3+ windows on any symbol.")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
