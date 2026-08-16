"""
Session Handoff Pullback (SHP) -- proprietary strategy.

Core thesis:
  The London session (7:00-12:00 UTC) sets the dominant intraday bias.
  When New York opens (13:00-14:00 UTC), there is frequently an initial
  counter-move (pullback) against the London trend as NY players
  re-price and test the London move. If this NY pullback is shallow
  (not invalidating the London trend), it is a high-probability entry
  point to join the London trend as it resumes.

Signal (on 1H bars):
  1. Compute the London session displacement: net move from 7:00 open
     to the 12:00 close bar (5 bars).
  2. London displacement must exceed min_disp × ATR to be "trending".
  3. At the NY open window (13:00-16:00 UTC), monitor for a pullback:
     - If London was bullish: price retraces at least pullback_min_pct
       of the London move, but not more than pullback_max_pct.
  4. On the first H1 bar in the NY window where price shows rejection
     of the pullback (bar closes in London trend direction): enter.
  5. SL: ATR-based below/above the NY pullback low/high.
     TP: RR × SL.
  6. One signal per day maximum.

Why this edge exists:
  London trend is the primary session-level move. NY open is a known
  liquidity event where retail stops from the London push are hunted.
  Once the NY "shakeout" finds its floor, institutional flow resumes
  the London direction. The setup has a defined invalidation level
  (if NY retraces more than pullback_max_pct, the London trend is done).
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pro_bot.backtest import simulate_exits, SPREADS, _fetch, CACHE_1H, CACHE_1D
from pro_bot.strategies.base import Signal

DAYS = 350

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

LONDON_START_UTC  = 7
LONDON_END_UTC    = 12  # last bar of the London reference session
NY_WINDOW_START   = 13
NY_WINDOW_END     = 16  # signal must fire by this hour


# ── ATR helper ─────────────────────────────────────────────────────────────────

def _atr(bars, period=14):
    result = [None] * len(bars)
    for i in range(1, len(bars)):
        trs = []
        for j in range(max(1, i - period + 1), i + 1):
            h, l, cp = bars[j]["high"], bars[j]["low"], bars[j - 1]["close"]
            trs.append(max(h - l, abs(h - cp), abs(l - cp)))
        if len(trs) >= period // 2:
            result[i] = sum(trs) / len(trs)
    return result


# ── Signal generator ──────────────────────────────────────────────────────────

def run_shp(b1h, b1d, cfg):
    min_disp_atr    = cfg["min_displacement_atr"]   # London move must exceed this × ATR
    pullback_min    = cfg["pullback_min_pct"]        # pullback must retrace at least this %
    pullback_max    = cfg["pullback_max_pct"]        # but not more than this % (trend invalid)
    tp_rr           = cfg["tp_rr"]
    atr_mult        = cfg["atr_mult_sl"]

    atr14 = _atr(b1h, 14)
    signals = []

    # Track state per trading day
    last_signal_day   = -1
    london_high       = {}  # day -> session high
    london_low        = {}  # day -> session low
    london_disp       = {}  # day -> net displacement (+ = bullish)
    london_open_price = {}  # day -> open price of London session
    ny_pullback_lo    = {}  # day -> lowest low seen in NY window (for bull pullback)
    ny_pullback_hi    = {}  # day -> highest high seen in NY window (for bear pullback)
    ny_bar_count      = {}  # day -> number of NY window bars processed

    for i in range(30, len(b1h)):
        bar   = b1h[i]
        epoch = bar["epoch"]
        hour  = (epoch // 3600) % 24
        day   = epoch // 86400

        atr_val = atr14[i]
        if atr_val is None or atr_val <= 0:
            continue

        # Accumulate London session data
        if LONDON_START_UTC <= hour <= LONDON_END_UTC:
            if day not in london_high:
                london_high[day]       = bar["high"]
                london_low[day]        = bar["low"]
                london_open_price[day] = bar["open"]
                ny_bar_count[day]      = 0
            else:
                london_high[day] = max(london_high[day], bar["high"])
                london_low[day]  = min(london_low[day],  bar["low"])

            if hour == LONDON_END_UTC:
                london_disp[day] = bar["close"] - london_open_price[day]

        # NY window: look for pullback + rejection
        elif NY_WINDOW_START <= hour <= NY_WINDOW_END:
            if day == last_signal_day:
                continue
            if day not in london_disp:
                continue

            disp = london_disp[day]
            if abs(disp) < min_disp_atr * atr_val:
                continue  # London move too small

            london_move_size = abs(disp)
            lon_open = london_open_price[day]
            lon_ref  = lon_open + disp  # London close price (approximate)

            ny_bar_count[day] = ny_bar_count.get(day, 0) + 1

            if disp > 0:
                # Bullish London day: track NY pullback low
                ny_pullback_lo[day] = min(
                    ny_pullback_lo.get(day, bar["low"]), bar["low"])
                retrace = (lon_ref - ny_pullback_lo[day]) / london_move_size

                if pullback_min <= retrace <= pullback_max:
                    # Pullback is in valid range -- check for rejection (bullish close)
                    if bar["close"] > bar["open"]:
                        sl = (bar["close"] - ny_pullback_lo[day]) + atr_mult * atr_val * 0.5
                        sl = max(sl, atr_val * atr_mult)
                        tp = sl * tp_rr
                        signals.append((i, Signal("BUY", sl_pips=sl, tp_pips=tp,
                                                  reason="SHP_bull_resume")))
                        last_signal_day = day

            elif disp < 0:
                # Bearish London day: track NY pullback high
                ny_pullback_hi[day] = max(
                    ny_pullback_hi.get(day, bar["high"]), bar["high"])
                retrace = (ny_pullback_hi[day] - lon_ref) / london_move_size

                if pullback_min <= retrace <= pullback_max:
                    if bar["close"] < bar["open"]:
                        sl = (ny_pullback_hi[day] - bar["close"]) + atr_mult * atr_val * 0.5
                        sl = max(sl, atr_val * atr_mult)
                        tp = sl * tp_rr
                        signals.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp,
                                                  reason="SHP_bear_resume")))
                        last_signal_day = day

    return signals


# ── Simulation / stats ────────────────────────────────────────────────────────

def sim(bars, sigs, spread):
    return simulate_exits(bars, sigs, spread=spread, be_r=1.0, max_hold_bars=12)


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
    all_sigs = run_shp(b1h, b1d, cfg)
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
            tr_d = (b1h[c1 - 1]["epoch"] - b1h[0]["epoch"]) // 86400 if c1 > 0 else 0
            ho_d = (b1h[c2 - 1]["epoch"] - b1h[c1]["epoch"]) // 86400 if c2 > c1 else 0
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

    for min_disp in [0.5, 1.0, 1.5]:
        for pb_min in [0.20, 0.30, 0.40]:
            for pb_max in [0.60, 0.75]:
                if pb_min >= pb_max:
                    continue
                for tp_rr in [1.5, 2.0, 3.0]:
                    for atr_mult in [1.0, 1.5, 2.0]:
                        cfg = dict(
                            min_displacement_atr=min_disp,
                            pullback_min_pct=pb_min,
                            pullback_max_pct=pb_max,
                            tp_rr=tp_rr,
                            atr_mult_sl=atr_mult,
                        )
                        label = (f"SHP disp{min_disp}xATR "
                                 f"pb{int(pb_min*100)}-{int(pb_max*100)}% "
                                 f"RR{tp_rr} ATR×{atr_mult}")
                        sigs = run_shp(train_b1h, b1d, cfg)
                        if not sigs:
                            continue
                        trades = sim(train_b1h, sigs, spread)
                        s = stats(trades, min_n=10)
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
    print("Session Handoff Pullback (SHP) -- 1H -- 4-window walk-forward")
    print(f"BE@1R  max_hold=12H  {DAYS}-day dataset")
    print("Thesis: London trend + shallow NY pullback + rejection = London trend resumes")
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
        print(f"\n  Phase 1 -- sweep on training set ({train_end} 1H bars / 60%)...")
        ranked = sweep(b1h, b1d, train_end, spread)

        print(f"  {len(ranked)} configs with positive EV on training data.")
        if not ranked:
            print(f"  No profitable SHP configs found for {sym}.\n")
            continue

        print("  Top 5 training configs:")
        for ev, n, label, _ in ranked[:5]:
            print(f"    EV {ev:>+.4f}R  n={n:>3}  {label}")

        top5 = ranked[:5]
        print(f"\n  Phase 2 -- 4-window walk-forward on top 5 configs")
        print(f"  {'-' * 60}")

        wf_results = []
        for ev, n, label, cfg in top5:
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
            print(f"  No SHP config passed 3+ windows for {sym}.")

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
        print("  No SHP strategy passed 3+ windows on any symbol.")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
