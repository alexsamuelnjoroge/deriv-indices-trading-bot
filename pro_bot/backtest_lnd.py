"""
London-NY Divergence (LND) -- 1H -- 4-window walk-forward

Core thesis:
  London and New York sessions regularly disagree. When the London session
  builds a significant directional move (07:00-12:59 UTC), the NY open
  (13:00-14:59 UTC) frequently fades that move as US institutions take the
  opposite view, or as London traders take profit.

  London displacement = (close of last London bar) - (close of first London bar)
  Normalized by 14-period ATR on the hour that the NY signal fires.

  If London_disp >= min_london_atr --> London went UP strongly --> SELL at NY open
  If London_disp <= -min_london_atr --> London went DOWN strongly --> BUY at NY open

  One signal per day maximum (the NY open window is limited).
  Entry on the first NY bar (13:xx UTC) that satisfies the London condition.

  Structural novelty vs ARS/SME:
    ARS uses the ASIAN session as reference.
    SME measures cumulative displacement from the session open, all sessions.
    LND specifically targets the HANDOFF between London and NY: two large
    institutional sessions with conflicting interests.
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


def _build_london_displacement(b1h, atr1h):
    """
    For each trading day, compute the normalized London session displacement.
    London open: first bar at 07:xx UTC.
    London close reference: last bar with open < 13:00 UTC (i.e., 12:xx UTC bar).
    Returns dict: day_midnight_epoch -> (displacement_atr, last_london_bar_idx)
    """
    out = {}
    n   = len(b1h)
    i   = 0
    while i < n:
        h   = (b1h[i]["epoch"] % 86400) // 3600
        day = (b1h[i]["epoch"] // 86400) * 86400

        if h == 7 and day not in out:
            # Find London open close price
            lon_open_close = b1h[i]["close"]
            lon_open_atr   = atr1h[i]

            # Walk forward to find last bar before 13:00
            j = i
            while j + 1 < n:
                next_h = (b1h[j + 1]["epoch"] % 86400) // 3600
                next_day = (b1h[j + 1]["epoch"] // 86400) * 86400
                if next_day != day or next_h >= 13:
                    break
                j += 1

            lon_close_close = b1h[j]["close"]
            ref_atr = atr1h[j]
            if ref_atr and ref_atr > 0:
                disp = (lon_close_close - lon_open_close) / ref_atr
                out[day] = (disp, j)

        i += 1
    return out


def run_lnd(b1h, b1d, cfg):
    min_london_atr = cfg["min_london_atr"]
    ny_window      = cfg.get("ny_window", 2)      # hours into NY to accept signals (1 or 2)
    cooldown_bars  = cfg.get("cooldown_bars", 24)
    tp_rr          = cfg["tp_rr"]
    atr_mult       = cfg["atr_mult_sl"]
    use_macro      = cfg.get("macro_filter", False)
    macro_p        = cfg.get("macro_ema_period", 20)

    atr1h   = _atr(b1h, 14)
    lon_map = _build_london_displacement(b1h, atr1h)

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

        # Only fire during NY open window
        if not (13 <= h < 13 + ny_window):
            continue
        if day in fired_days:
            continue

        atr_val = atr1h[i]
        if atr_val is None or atr_val <= 0:
            continue

        lnd = lon_map.get(day)
        if lnd is None:
            continue
        lon_disp, _ = lnd

        allow_long = allow_short = True
        if ema_d is not None:
            k = bisect.bisect_right(ep_d, epoch) - 1
            if k >= 1 and ema_d[k] is not None and ema_d[k - 1] is not None:
                macro_up    = ema_d[k] > ema_d[k - 1]
                allow_long  = macro_up
                allow_short = not macro_up

        sl = atr_val * atr_mult
        tp = sl * tp_rr

        if lon_disp >= min_london_atr and allow_short:
            signals.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp,
                                      reason=f"LND_sell_lon{lon_disp:.2f}ATR")))
            fired_days.add(day)
            last_sig_i = i

        elif lon_disp <= -min_london_atr and allow_long:
            signals.append((i, Signal("BUY", sl_pips=sl, tp_pips=tp,
                                      reason=f"LND_buy_lon{lon_disp:.2f}ATR")))
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
    all_sigs = run_lnd(b1h, b1d, cfg)
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

    for min_lon in [1.0, 1.5, 2.0, 2.5, 3.0]:
        for ny_window in [1, 2]:
            for cooldown in [12, 24]:
                for tp_rr in [1.5, 2.0, 3.0]:
                    for atr_mult in [1.0, 1.5, 2.0]:
                        for macro in [False, True]:
                            cfg = dict(
                                min_london_atr=min_lon,
                                ny_window=ny_window,
                                cooldown_bars=cooldown,
                                tp_rr=tp_rr,
                                atr_mult_sl=atr_mult,
                                macro_filter=macro,
                                macro_ema_period=20,
                            )
                            label = (
                                f"LND lon>={min_lon}ATR ny{ny_window}h "
                                f"cool{cooldown} RR{tp_rr} ATRx{atr_mult} "
                                f"{'MACRO' if macro else 'free'}"
                            )
                            sigs = run_lnd(train_b1h, b1d, cfg)
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
    print("London-NY Divergence (LND) -- 1H -- 4-window walk-forward")
    print(f"BE@1R  max_hold=48H  {DAYS}-day dataset")
    print("Thesis: large London session move gets faded by NY open institutions")
    print("London: 07:00-12:59 UTC | NY entry window: 13:00-14:59 UTC")
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
            print(f"  No profitable LND configs found for {sym}.\n")
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
            print(f"  No LND config passed 3+ windows for {sym}.")

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
        print("  No LND strategy passed 3+ windows on any symbol.")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
