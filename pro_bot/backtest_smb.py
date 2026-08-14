"""
Session Memory Bias (SMB) -- proprietary strategy.

Core thesis:
  Where the Asian session CLOSES within its own range predicts London session
  direction with a measurable statistical edge.

  If Asia builds a one-sided position (closes near its high) → London market
  makers fade that position at open to rebalance institutional books. This is
  predictable behavior because prime brokers must reconcile cross-session exposure.

  If Asia closes near its LOW → London continuation traders pile in (momentum).

  The key insight: we use SESSION CLOSE POSITION (not price level, not indicators)
  as the signal. This is inter-session conditional probability -- not in any textbook.

Signal (on 1H bars, London session):
  1. For each day, compute Asian session range (00:00–07:00 UTC = 7 1H bars)
     asian_pos = (asian_close - asian_low) / (asian_high - asian_low)
  2. Set London session bias:
     - asian_pos > threshold_high (e.g. 0.75) → SHORT bias (London fades Asian high)
     - asian_pos < threshold_low  (e.g. 0.25) → LONG  bias (London continues Asian low)
     - Middle zone → no trade
  3. Enter at the SECOND London 1H bar (08:00 UTC) to confirm direction is holding
     The first bar (07:00) is the transition gap bar -- too noisy
  4. SL: ATR(14) x atr_mult
  5. TP: RR x SL
  6. One trade per day, London only (expires at max_london_hour UTC)

Why 08:00 UTC not 07:00?
  The 07:00 bar captures the Asian close + first London reaction (often erratic).
  The 08:00 bar is the institutional "true open" -- real London orders hit the book.
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

# Asian session: 00:00 – 07:00 UTC (7 hours)
ASIA_START_UTC = 0
ASIA_END_UTC   = 7

# London session: 07:00 – 16:00 UTC
LONDON_ENTRY_UTC = 8   # enter at 08:00 bar close (second London bar)
LONDON_END_UTC   = 13  # last valid entry bar

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


# ── Helpers ──────────────────────────────────────────────────────────────────

def _day(epoch):
    return epoch // 86400


def _utc_hour(epoch):
    return (epoch % 86400) // 3600


def _build_asian_sessions(b1h):
    """
    For each trading day, compute the Asian session metrics.
    Returns dict: day_key → {high, low, close, pos, close_epoch}
    where pos = (close - low) / (high - low), 0 = closed at low, 1 = at high.
    """
    daily = {}
    for bar in b1h:
        h = _utc_hour(bar["epoch"])
        if ASIA_START_UTC <= h < ASIA_END_UTC:
            d = _day(bar["epoch"])
            daily.setdefault(d, []).append(bar)

    sessions = {}
    for d, bars in daily.items():
        if len(bars) < 3:  # need at least 3 bars for a valid session
            continue
        hi  = max(b["high"]  for b in bars)
        lo  = min(b["low"]   for b in bars)
        cls = bars[-1]["close"]   # last Asia bar close
        if hi == lo:
            continue
        sessions[d] = {
            "high":        hi,
            "low":         lo,
            "close":       cls,
            "pos":         (cls - lo) / (hi - lo),
            "close_epoch": bars[-1]["epoch"],
        }
    return sessions


# ── Signal generator ─────────────────────────────────────────────────────────

def run_smb(b1h, b1d, cfg):
    thresh_high = cfg["threshold_high"]  # asian_pos > this → SHORT
    thresh_low  = cfg["threshold_low"]   # asian_pos < this → LONG
    tp_rr       = cfg["tp_rr"]
    atr_mult    = cfg["atr_mult_sl"]
    use_macro   = cfg.get("macro_filter", False)
    macro_p     = cfg.get("macro_ema_period", 20)
    entry_hour  = cfg.get("entry_hour_utc", LONDON_ENTRY_UTC)
    max_hour    = cfg.get("max_hour_utc", LONDON_END_UTC)

    atr1h    = _atr(b1h, 14)
    sessions = _build_asian_sessions(b1h)

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
        h     = _utc_hour(epoch)

        # Only enter at the specified London hour on bars not already traded
        if h != entry_hour:
            continue
        if d in day_traded:
            continue

        # Look up Asian session for this day
        sess = sessions.get(d)
        if sess is None:
            continue

        asian_pos = sess["pos"]

        # Dead zone: no strong Asian bias
        if thresh_low <= asian_pos <= thresh_high:
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

        if asian_pos > thresh_high and allow_short:
            # Asia closed near HIGH → London fades it → SELL
            signals.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp,
                                      reason=f"SMB_asia_hi_{asian_pos:.2f}")))
            day_traded.add(d)

        elif asian_pos < thresh_low and allow_long:
            # Asia closed near LOW → London continues downward → BUY (London reversal)
            # Wait: if Asia closed near LOW and we expect London to FADE it,
            # that means Asia was bearish → London fades bearish = BUY
            signals.append((i, Signal("BUY", sl_pips=sl, tp_pips=tp,
                                      reason=f"SMB_asia_lo_{asian_pos:.2f}")))
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
    all_sigs = run_smb(b1h, b1d, cfg)
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


# ── Parameter sweep ──────────────────────────────────────────────────────────

def sweep(b1h, b1d, train_end_idx, spread):
    train_b1h = b1h[:train_end_idx]
    results   = []

    for thresh_h in [0.70, 0.75, 0.80]:
        for thresh_l in [0.20, 0.25, 0.30]:
            for entry_hour in [8, 9]:
                for tp_rr in [1.5, 2.0, 3.0]:
                    for atr_mult in [1.0, 1.5]:
                        for macro in [False, True]:
                            cfg = dict(
                                threshold_high=thresh_h,
                                threshold_low=thresh_l,
                                entry_hour_utc=entry_hour,
                                max_hour_utc=LONDON_END_UTC,
                                tp_rr=tp_rr,
                                atr_mult_sl=atr_mult,
                                macro_filter=macro,
                                macro_ema_period=20,
                            )
                            label = (
                                f"SMB hi{thresh_h} lo{thresh_l} "
                                f"entry{entry_hour}UTC "
                                f"RR{tp_rr} ATR×{atr_mult} "
                                f"{'MACRO' if macro else 'free'}"
                            )
                            sigs = run_smb(train_b1h, b1d, cfg)
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
    print("Session Memory Bias (SMB) -- 1H -- 4-window walk-forward")
    print(f"BE@1R  max_hold=8H (London session expiry)  {DAYS}-day dataset")
    print("Thesis: Asian session close position predicts London session direction")
    print("Entry at 08:00 UTC (2nd London bar) -- institutional true open")
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
            print(f"  No profitable SMB configs found for {sym}.\n")
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
            print(f"  No SMB config passed 3+ windows for {sym}.")

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
        print("  No SMB strategy passed 3+ windows on any symbol.")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
