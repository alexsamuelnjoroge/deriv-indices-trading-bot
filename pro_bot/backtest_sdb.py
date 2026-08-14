"""
Session Displacement Bias (SDB) -- proprietary strategy.

Core thesis:
  Within the London morning session (07:00–12:59 UTC), the order in which
  price makes its HIGH and LOW reveals the institutional directional bias.

  "Ascending session":  LOW forms early, HIGH forms late
    → buyers systematically pushed price up through the session
    → institutional accumulation → expect NY continuation → BUY at 13:00 UTC

  "Descending session": HIGH forms early, LOW forms late
    → sellers systematically pushed price down through the session
    → institutional distribution → expect NY continuation → SELL at 13:00 UTC

  This is fundamentally different from SMB (which uses Asian session close POSITION).
  SDB uses the TEMPORAL STRUCTURE of the London session — WHEN extremes formed,
  not where price ended up.

Signal (on 1H bars):
  1. For each day, observe London morning: bars at 07:00, 08:00, 09:00, 10:00, 11:00, 12:00 UTC
  2. Find hour_of_high and hour_of_low within this window
  3. Compute temporal separation: |hour_of_high - hour_of_low| / 5 → [0, 1]
  4. If separation < min_strength: session was chaotic → no trade
  5. At the 13:00 UTC bar (NY open), enter in the London session direction
  6. One trade per day, expires max 8H later
  7. SL: ATR × mult, TP: RR × SL

Why 13:00 UTC (NY open)?
  13:00 UTC = London/NY overlap. This is the highest liquidity hour of the day.
  Institutional London positions continue being managed; NY institutions join.
  The London morning bias is most likely to play out during this overlap.

Parameters also test FADE mode: entering AGAINST London morning direction
  (some markets exhibit London reversal at NY open instead of continuation).
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

LONDON_OBS_START = 7   # 07:00 UTC — London open
LONDON_OBS_END   = 12  # 12:00 UTC — last observation bar (6 bars: 07-12)
ENTRY_HOUR_UTC   = 13  # enter at NY open

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


# ── Session shape builder ─────────────────────────────────────────────────────

def _day(epoch):   return epoch // 86400
def _utc_h(epoch): return (epoch % 86400) // 3600


def _build_london_shapes(b1h):
    """
    For each trading day, find the temporal positions of HIGH and LOW
    within the London morning observation window (07:00–12:00 UTC).
    Returns: day_key → {'ascending': bool, 'strength': float}
    """
    daily = {}
    for bar in b1h:
        h = _utc_h(bar["epoch"])
        if LONDON_OBS_START <= h <= LONDON_OBS_END:
            d = _day(bar["epoch"])
            daily.setdefault(d, []).append(bar)

    shapes = {}
    for d, bars in daily.items():
        if len(bars) < 4:
            continue
        bars_s = sorted(bars, key=lambda b: b["epoch"])
        n_bars  = len(bars_s)

        hi_idx = max(range(n_bars), key=lambda k: bars_s[k]["high"])
        lo_idx = min(range(n_bars), key=lambda k: bars_s[k]["low"])

        if n_bars == 1:
            continue

        hi_pos = hi_idx / (n_bars - 1)  # 0=first bar, 1=last bar
        lo_pos = lo_idx / (n_bars - 1)

        ascending = lo_pos < hi_pos  # low came before high → price trended up
        strength  = abs(hi_pos - lo_pos)  # temporal separation: higher = clearer shape

        shapes[d] = {
            "ascending": ascending,
            "strength":  strength,
            "session_hi": max(b["high"] for b in bars_s),
            "session_lo": min(b["low"]  for b in bars_s),
        }
    return shapes


# ── Signal generator ─────────────────────────────────────────────────────────

def run_sdb(b1h, b1d, cfg):
    min_strength = cfg["min_strength"]   # temporal separation must be > this
    fade         = cfg.get("fade", False) # True = trade AGAINST London direction
    entry_h      = cfg.get("entry_hour_utc", ENTRY_HOUR_UTC)
    tp_rr        = cfg["tp_rr"]
    atr_mult     = cfg["atr_mult_sl"]
    use_macro    = cfg.get("macro_filter", False)
    macro_p      = cfg.get("macro_ema_period", 20)

    atr1h  = _atr(b1h, 14)
    shapes = _build_london_shapes(b1h)

    if use_macro and b1d:
        ema_d = _ema([b["close"] for b in b1d], macro_p)
        ep_d  = [b["epoch"] for b in b1d]
    else:
        ema_d = ep_d = None

    signals    = []
    day_traded = set()

    for i in range(20, len(b1h)):
        bar   = b1h[i]
        epoch = bar["epoch"]
        d     = _day(epoch)
        h     = _utc_h(epoch)

        if h != entry_h:
            continue
        if d in day_traded:
            continue

        shape = shapes.get(d)
        if shape is None:
            continue
        if shape["strength"] < min_strength:
            continue

        atr_val = atr1h[i]
        if atr_val is None or atr_val <= 0:
            continue

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

        london_bull = shape["ascending"]
        if fade:
            london_bull = not london_bull  # reverse the signal

        if london_bull and allow_long:
            signals.append((i, Signal("BUY", sl_pips=sl, tp_pips=tp,
                                      reason=f"SDB_asc_str{shape['strength']:.2f}")))
            day_traded.add(d)
        elif not london_bull and allow_short:
            signals.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp,
                                      reason=f"SDB_desc_str{shape['strength']:.2f}")))
            day_traded.add(d)

    return signals


# ── Simulation / stats ───────────────────────────────────────────────────────

def sim(bars, sigs, spread):
    return simulate_exits(bars, sigs, spread=spread, be_r=1.0, max_hold_bars=8)


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
    all_sigs = run_sdb(b1h, b1d, cfg)
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

    for min_strength in [0.30, 0.50, 0.70]:
        for fade in [False, True]:
            for entry_h in [13, 14]:
                for tp_rr in [1.5, 2.0, 3.0]:
                    for atr_mult in [1.0, 1.5]:
                        for macro in [False, True]:
                            cfg = dict(
                                min_strength=min_strength,
                                fade=fade,
                                entry_hour_utc=entry_h,
                                tp_rr=tp_rr,
                                atr_mult_sl=atr_mult,
                                macro_filter=macro,
                                macro_ema_period=20,
                            )
                            label = (
                                f"SDB str>{min_strength} "
                                f"{'FADE' if fade else 'CONT'} "
                                f"entry{entry_h}UTC "
                                f"RR{tp_rr} ATR×{atr_mult} "
                                f"{'MACRO' if macro else 'free'}"
                            )
                            sigs = run_sdb(train_b1h, b1d, cfg)
                            if not sigs:
                                continue
                            trades = sim(train_b1h, sigs, spread)
                            s = stats(trades, min_n=10)
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
    print("Session Displacement Bias (SDB) -- 1H -- 4-window walk-forward")
    print(f"BE@1R  max_hold=8H  {DAYS}-day dataset")
    print("Thesis: WHEN London extremes form (not where) reveals institutional direction")
    print("Entry: 13:00 UTC (NY open) | Tests both CONTINUATION and FADE modes")
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
            print(f"  No profitable SDB configs found for {sym}.\n")
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
            print(f"  No SDB config passed 3+ windows for {sym}.")

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
        print("  No SDB strategy passed 3+ windows on any symbol.")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
