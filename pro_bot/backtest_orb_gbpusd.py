"""
London Opening Range Breakout (ORB) -- GBPUSD walk-forward validation.

Strategy (genuinely different from BB+RSI pullback):
  1. Define the opening range: first range_mins of London open (07:00 UTC / 10:00 EAT)
  2. After range closes, watch for a 5m bar that closes outside the range
  3. Entry in breakout direction, ATR-based SL, fixed RR take-profit
  4. One trade per day only; must trigger before max_trade_hour_utc
  5. Expansion filter: range must be > min_range_atr_ratio x ATR(14)
  6. Macro gate: daily EMA(20) slope filters direction on trending days

Edge source: Institutional flow at the London open creates directional bias.
This is structurally different from the existing strategies (no RSI, no EMA trend,
no Bollinger Bands) -- purely price-structure based.

Methodology:
  Phase 1 -- Parameter sweep on first 60% of data (training)
  Phase 2 -- 4-window walk-forward on top configs
  Config must pass all 4 windows (positive holdout EV) to be ROBUST.
"""

import asyncio
import bisect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pro_bot.backtest import simulate_exits, SPREADS, _fetch, CACHE_5M, CACHE_1D
from pro_bot.indicators import atr as _atr, ema as _ema
from pro_bot.strategies.base import Signal

SYM    = "frxGBPUSD"
SPREAD = SPREADS[SYM]
DAYS   = 730

LONDON_OPEN_UTC = 7   # 07:00 UTC = 10:00 EAT

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


def _utc_hm(epoch):
    sod = epoch % 86400
    return sod // 3600, (sod % 3600) // 60


def _build_ranges(b5, range_mins):
    """
    Build daily opening ranges from 5m bars.
    Range = first range_mins of London open (starting 07:00 UTC).
    Returns: day_key -> {high, low, complete_epoch, width}
    """
    needed   = range_mins // 5
    daily    = {}
    for bar in b5:
        h, m = _utc_hm(bar["epoch"])
        mins_from_open = (h - LONDON_OPEN_UTC) * 60 + m
        if 0 <= mins_from_open < range_mins:
            d = _day(bar["epoch"])
            daily.setdefault(d, []).append(bar)

    ranges = {}
    for d, bars in daily.items():
        if len(bars) < needed:
            continue
        rh = max(b["high"] for b in bars)
        rl = min(b["low"]  for b in bars)
        ranges[d] = {
            "high":           rh,
            "low":            rl,
            "complete_epoch": bars[-1]["epoch"] + 300,
            "width":          rh - rl,
        }
    return ranges


# ── Signal generator ─────────────────────────────────────────────────────────

def run_orb(b5, b1d, cfg):
    range_mins  = cfg["range_mins"]
    tp_rr       = cfg["tp_rr"]
    atr_mult    = cfg["atr_mult_sl"]
    min_rng_atr = cfg["min_range_atr_ratio"]
    max_h_utc   = cfg["max_trade_hour_utc"]
    use_macro   = cfg.get("macro_filter", False)
    macro_p     = cfg.get("macro_ema_period", 20)

    ranges = _build_ranges(b5, range_mins)
    atr5m  = _atr(b5, 14)

    if use_macro and b1d:
        ema_d = _ema([b["close"] for b in b1d], macro_p)
        ep_d  = [b["epoch"] for b in b1d]
    else:
        ema_d = ep_d = None

    signals    = []
    day_traded = set()

    for i in range(20, len(b5)):
        epoch = b5[i]["epoch"]
        d     = _day(epoch)
        if d in day_traded:
            continue

        h_utc, _ = _utc_hm(epoch)
        if h_utc >= max_h_utc:
            continue

        rng = ranges.get(d)
        if rng is None or epoch < rng["complete_epoch"]:
            continue

        atr_val = atr5m[i]
        if atr_val is None or atr_val <= 0:
            continue
        if rng["width"] < min_rng_atr * atr_val:
            continue

        allow_long = allow_short = True
        if ema_d is not None:
            k = bisect.bisect_right(ep_d, epoch) - 1
            if k < 1 or ema_d[k] is None or ema_d[k - 1] is None:
                continue
            macro_up    = ema_d[k] > ema_d[k - 1]
            allow_long  = macro_up
            allow_short = not macro_up

        close = b5[i]["close"]
        sl    = atr_val * atr_mult
        tp    = sl * tp_rr

        if close > rng["high"] and allow_long:
            signals.append((i, Signal("BUY",  sl_pips=sl, tp_pips=tp)))
            day_traded.add(d)
        elif close < rng["low"] and allow_short:
            signals.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp)))
            day_traded.add(d)

    return signals


# ── Simulation / stats ───────────────────────────────────────────────────────

def sim(bars, sigs):
    return simulate_exits(bars, sigs, spread=SPREAD, be_r=1.0)


def stats(trades, min_n=5):
    closed = [t for t in trades if t.result in ("WIN", "LOSS", "BE")]
    if len(closed) < min_n:
        return None
    wins  = sum(1 for t in closed if t.result == "WIN")
    n_wr  = sum(1 for t in closed if t.result in ("WIN", "LOSS"))
    ev    = sum(t.r_multiple for t in closed) / len(closed)
    return dict(n=len(closed), wr=wins / n_wr if n_wr else 0, ev=ev,
                net_r=sum(t.r_multiple for t in closed))


# ── Walk-forward ─────────────────────────────────────────────────────────────

def _split_sigs(all_sigs, c1, c2):
    return ([(i, s) for i, s in all_sigs if i < c1],
            [(i - c1, s) for i, s in all_sigs if c1 <= i < c2])


def run_wf(b5, b1d, cfg, label, verbose=True):
    all_sigs = run_orb(b5, b1d, cfg)
    n        = len(b5)
    passes   = 0

    if verbose:
        print(f"  Config: {label}  [{len(all_sigs)} total signals]")

    for wi, (_, te_pct, ho_pct) in enumerate(WINDOWS):
        c1 = int(n * te_pct)
        c2 = int(n * ho_pct)

        tr_sigs, ho_sigs = _split_sigs(all_sigs, c1, c2)
        tr_s = stats(sim(b5[:c1],    tr_sigs), min_n=10)
        ho_s = stats(sim(b5[c1:c2], ho_sigs), min_n=5)

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
            tr_d = (b5[c1 - 1]["epoch"] - b5[0]["epoch"]) // 86400 if c1 > 0 else 0
            ho_d = (b5[c2 - 1]["epoch"] - b5[c1]["epoch"]) // 86400 if c2 > c1 else 0
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

def sweep(b5, b1d, train_end_idx):
    """Sweep configs on training data; return sorted (ev, cfg, label) list."""
    train_b5 = b5[:train_end_idx]

    results = []
    for range_mins in [15, 30]:
        for tp_rr in [1.5, 2.0]:
            for atr_mult in [1.5, 2.0]:
                for min_rng_atr in [0.3, 0.5, 0.7]:
                    for max_h_utc in [9, 10, 11]:
                        for macro in [False, True]:
                            cfg = dict(
                                range_mins=range_mins,
                                tp_rr=tp_rr,
                                atr_mult_sl=atr_mult,
                                min_range_atr_ratio=min_rng_atr,
                                max_trade_hour_utc=max_h_utc,
                                macro_filter=macro,
                                macro_ema_period=20,
                            )
                            label = (
                                f"ORB{range_mins:2d}m "
                                f"RR{tp_rr} "
                                f"ATR×{atr_mult} "
                                f"minRng{min_rng_atr}×ATR "
                                f"cut{max_h_utc:02d}UTC "
                                f"{'MACRO' if macro else 'free'}"
                            )
                            sigs = run_orb(train_b5, b1d, cfg)
                            if not sigs:
                                continue
                            trades = sim(train_b5, sigs)
                            s = stats(trades, min_n=20)
                            if s and s["ev"] > 0:
                                results.append((s["ev"], s["n"], label, cfg))

    results.sort(key=lambda x: -x[0])
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

async def main():
    import time as _t

    print("=" * 78)
    print("GBPUSD London Opening Range Breakout (ORB) -- 4-window walk-forward")
    print(f"Symbol: {SYM}  Spread: {SPREAD}  BE@1R  {DAYS}-day dataset")
    print("=" * 78 + "\n")

    print("  Loading data...", end=" ", flush=True)
    b5  = await _fetch(SYM, 300,   DAYS, CACHE_5M)
    b1d = await _fetch(SYM, 86400, DAYS, CACHE_1D)
    print(f"{len(b5)} 5m bars | {len(b1d)} daily bars")

    fd = _t.strftime("%Y-%m-%d", _t.gmtime(b5[0]["epoch"]))
    ld = _t.strftime("%Y-%m-%d", _t.gmtime(b5[-1]["epoch"]))
    print(f"  Date range: {fd} -> {ld}")

    # Phase 1: sweep on first 60% of data
    train_end = int(len(b5) * 0.60)
    print(f"\n  Phase 1 -- parameter sweep on training set ({train_end} bars / 60%)...")
    ranked = sweep(b5, b1d, train_end)

    print(f"  {len(ranked)} configs with positive EV on training data.\n")
    if not ranked:
        print("  No profitable configs found. ORB does not appear viable on GBPUSD.")
        return

    print("  Top 10 training configs:")
    for ev, n, label, _ in ranked[:10]:
        print(f"    EV {ev:>+.4f}R  n={n:>3}  {label}")

    # Phase 2: walk-forward validation on top 5 configs
    top5 = ranked[:5]
    print(f"\n{'=' * 78}")
    print("  Phase 2 -- 4-window walk-forward on top 5 configs")
    print(f"{'=' * 78}\n")

    wf_results = []
    for ev, n, label, cfg in top5:
        print(f"{'─' * 78}")
        passes = run_wf(b5, b1d, cfg, label, verbose=True)
        wf_results.append((passes, ev, label, cfg))
        print(f"{'─' * 78}")

    print("\n  FINAL SUMMARY")
    print("  " + "-" * 60)
    robust = [(p, ev, l, c) for p, ev, l, c in wf_results if p >= 3]
    robust.sort(key=lambda x: (-x[0], -x[1]))

    if robust:
        for passes, ev, label, cfg in robust:
            bar     = "#" * passes + "." * (4 - passes)
            verdict = "ROBUST" if passes == 4 else "MOSTLY OK"
            print(f"  [{bar}] {passes}/4  {label}")
            print(f"         -> {verdict}  (train EV {ev:+.4f}R)")
            if passes == 4:
                print(f"\n  BEST CONFIG TO WIRE IN:")
                for k, v in cfg.items():
                    print(f"    {k}: {v}")
    else:
        print("  No config passed 3+ windows.")
        print("  ORB on GBPUSD does not have a robust quantified edge with these parameters.")
        print("  Recommendation: do NOT trade this setup live.")

    print("\n" + "=" * 78)
    print("Interpretation:")
    print("  ROBUST (4/4)    -- safe to add as a live strategy")
    print("  MOSTLY OK (3/4) -- deploy with caution / half position size")
    print("  MARGINAL (2/4)  -- do not deploy")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
