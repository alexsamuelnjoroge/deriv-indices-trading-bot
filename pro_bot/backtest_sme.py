"""
Session Move Exhaustion (SME) -- proprietary strategy.

Core thesis:
  Every trading session has a natural "energy budget" -- the total directional
  distance price can sustain before exhaustion. Once price has moved X multiples
  of ATR from the session open, the probability of continuation drops sharply
  and mean-reversion becomes the high-probability bet.

  This is different from RVD (which measures bar-by-bar velocity decay).
  SME measures the CUMULATIVE SESSION DISPLACEMENT -- total ground covered
  from the session open to the current bar, regardless of how it got there.

  A professional trader "feels" this: "Gold is up $45 today -- it's tired, I'm
  looking for a fade." SME quantifies exactly when that fatigue threshold is hit.

  Mechanics:
    1. At each session open (London: 07:00 UTC), record the opening price
    2. For each subsequent bar in the session, compute:
       displacement = |current_close - session_open| / ATR(14)
    3. When displacement crosses exhaustion_thresh:
       - If session moved UP by that much  → SELL (fade the up-move)
       - If session moved DOWN by that much → BUY (fade the down-move)
    4. Only one signal per session (first crossing of threshold)
    5. SL: ATR x mult (placed against the move direction)
    6. TP: RR x SL
    7. Macro filter optional (but note: extreme moves often exhaust even with macro)

  Sessions tested:
    London: 07:00-16:59 UTC
    NY:     13:00-21:59 UTC
    Both: fire in either session
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


# ── Session open tracker ───────────────────────────────────────────────────────

def _session_opens(b1h, session):
    """
    Build a dict: day_midnight -> {london_open_price, ny_open_price}
    London open = 07:00 UTC bar close
    NY open     = 13:00 UTC bar close
    """
    opens = {}
    for bar in b1h:
        h            = (bar["epoch"] % 86400) // 3600
        day_midnight = (bar["epoch"] // 86400) * 86400
        if day_midnight not in opens:
            opens[day_midnight] = {"london": None, "ny": None}
        if h == 7 and opens[day_midnight]["london"] is None:
            opens[day_midnight]["london"] = bar["open"]
        if h == 13 and opens[day_midnight]["ny"] is None:
            opens[day_midnight]["ny"] = bar["open"]
    return opens


# ── Signal generator ──────────────────────────────────────────────────────────

def run_sme(b1h, b1d, cfg):
    exhaustion    = cfg["exhaustion_thresh"]   # displacement in ATR multiples
    session       = cfg.get("session", "both") # "london" | "ny" | "both"
    cooldown_bars = cfg.get("cooldown_bars", 12)
    tp_rr         = cfg["tp_rr"]
    atr_mult      = cfg["atr_mult_sl"]
    use_macro     = cfg.get("macro_filter", False)
    macro_p       = cfg.get("macro_ema_period", 20)

    atr1h  = _atr(b1h, 14)
    s_opens = _session_opens(b1h, session)

    if use_macro and b1d:
        ema_d = _ema([b["close"] for b in b1d], macro_p)
        ep_d  = [b["epoch"] for b in b1d]
    else:
        ema_d = ep_d = None

    signals    = []
    last_sig_i = -999

    # Track whether the threshold has already fired this session-day
    fired_today = {}  # day_midnight -> {"london": False, "ny": False}

    for i in range(24, len(b1h)):
        bar   = b1h[i]
        epoch = bar["epoch"]
        h     = (epoch % 86400) // 3600

        # Determine which session this bar belongs to
        in_london = (7 <= h <= 16)
        in_ny     = (13 <= h <= 21)

        if session == "london" and not in_london:
            continue
        if session == "ny" and not in_ny:
            continue
        if session == "both" and not (in_london or in_ny):
            continue

        if i - last_sig_i < cooldown_bars:
            continue

        atr_val = atr1h[i]
        if atr_val is None or atr_val <= 0:
            continue

        day_midnight = (epoch // 86400) * 86400
        if day_midnight not in s_opens:
            continue

        if day_midnight not in fired_today:
            fired_today[day_midnight] = {"london": False, "ny": False}

        sess_key = None
        sess_open = None
        if in_london and (session in ("london", "both")):
            if not fired_today[day_midnight]["london"]:
                sess_key  = "london"
                sess_open = s_opens[day_midnight].get("london")
        if in_ny and (session in ("ny", "both")) and sess_key is None:
            if not fired_today[day_midnight]["ny"]:
                sess_key  = "ny"
                sess_open = s_opens[day_midnight].get("ny")

        if sess_key is None or sess_open is None:
            continue

        displacement = (bar["close"] - sess_open) / atr_val

        if abs(displacement) < exhaustion:
            continue

        # Threshold crossed — fade the session move
        allow_long = allow_short = True
        if ema_d is not None:
            k = bisect.bisect_right(ep_d, epoch) - 1
            if k >= 1 and ema_d[k] is not None and ema_d[k - 1] is not None:
                macro_up    = ema_d[k] > ema_d[k - 1]
                allow_long  = macro_up
                allow_short = not macro_up

        sl = atr_val * atr_mult
        tp = sl * tp_rr

        if displacement > 0 and allow_short:
            signals.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp,
                                      reason=f"SME_sell_d{displacement:.2f}ATR")))
            fired_today[day_midnight][sess_key] = True
            last_sig_i = i

        elif displacement < 0 and allow_long:
            signals.append((i, Signal("BUY", sl_pips=sl, tp_pips=tp,
                                      reason=f"SME_buy_d{displacement:.2f}ATR")))
            fired_today[day_midnight][sess_key] = True
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
    all_sigs = run_sme(b1h, b1d, cfg)
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

    for exhaustion in [1.5, 2.0, 2.5, 3.0, 4.0]:
        for session in ["london", "ny", "both"]:
            for cooldown in [6, 12]:
                for tp_rr in [1.5, 2.0, 3.0]:
                    for atr_mult in [1.0, 1.5, 2.0]:
                        for macro in [False, True]:
                            cfg = dict(
                                exhaustion_thresh=exhaustion,
                                session=session,
                                cooldown_bars=cooldown,
                                tp_rr=tp_rr,
                                atr_mult_sl=atr_mult,
                                macro_filter=macro,
                                macro_ema_period=20,
                            )
                            label = (
                                f"SME exh>{exhaustion}ATR sess={session} "
                                f"cool{cooldown} RR{tp_rr} ATRx{atr_mult} "
                                f"{'MACRO' if macro else 'free'}"
                            )
                            sigs = run_sme(train_b1h, b1d, cfg)
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
    print("Session Move Exhaustion (SME) -- 1H -- 4-window walk-forward")
    print(f"BE@1R  max_hold=48H  {DAYS}-day dataset")
    print("Thesis: total session displacement beyond X-ATR threshold = exhaustion = fade")
    print("Different from RVD (bar-by-bar): measures cumulative session ground covered")
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
            print(f"  No profitable SME configs found for {sym}.\n")
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
            print(f"  No SME config passed 3+ windows for {sym}.")

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
        print("  No SME strategy passed 3+ windows on any symbol.")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
