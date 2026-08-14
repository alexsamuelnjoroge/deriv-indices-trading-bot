"""
Bar Pattern Sequences (BPS) -- 1H -- 4-window walk-forward

Core thesis:
  Specific 1H candlestick patterns at key price levels carry higher predictive
  value than the pattern alone. The level provides the context (WHY price might
  reverse there); the pattern provides the timing (WHEN the reversal is starting).

  Two pattern types tested:

  1. ENGULFING at key levels:
     Bearish: bar[i-1] is bullish AND bar[i] opens >= bar[i-1].close
              AND bar[i] closes <= bar[i-1].open (body fully engulfs prev bar)
              AT resistance (within zone_atr of PDH or Asian High) --> SELL
     Bullish: mirror at support --> BUY

  2. PIN BAR (wick rejection) at key levels:
     Bearish pin: upper shadow >= body_mult * body AND close in bottom 40% of range
                  AT resistance --> SELL
     Bullish pin: lower shadow >= body_mult * body AND close in top 40% of range
                  AT support --> BUY

  Key levels (same as LSH):
    PDH / PDL -- previous day high/low
    Asian H/L -- 22:00-06:59 UTC session boundary

  The level proximity requirement is the critical filter: without it these
  patterns have no demonstrated edge (random bars at random prices).
"""

import asyncio
import bisect
import sys
from collections import defaultdict
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


def _build_pdh_pdl(b1d):
    out = {}
    for i in range(1, len(b1d)):
        today = b1d[i]["epoch"]
        out[today] = {"pdh": b1d[i - 1]["high"], "pdl": b1d[i - 1]["low"]}
    return out


def _build_asian_ranges(b1h):
    buckets = defaultdict(lambda: {"highs": [], "lows": []})
    for bar in b1h:
        h   = (bar["epoch"] % 86400) // 3600
        day = (bar["epoch"] // 86400) * 86400
        if 22 <= h <= 23:
            buckets[day + 86400]["highs"].append(bar["high"])
            buckets[day + 86400]["lows"].append(bar["low"])
        elif 0 <= h <= 6:
            buckets[day]["highs"].append(bar["high"])
            buckets[day]["lows"].append(bar["low"])
    return {
        d: (max(v["highs"]), min(v["lows"]))
        for d, v in buckets.items()
        if v["highs"]
    }


def _near_level(price, levels, zone_atr, atr_val, side):
    """Returns True if price is within zone_atr of any level in the appropriate direction."""
    for lvl in levels:
        dist = abs(price - lvl) / atr_val
        if dist <= zone_atr:
            if side == "resist" and price <= lvl:
                return True
            if side == "support" and price >= lvl:
                return True
    return False


def run_bps(b1h, b1d, cfg):
    pattern_type  = cfg["pattern_type"]       # "engulfing" | "pin_bar" | "both"
    zone_atr      = cfg.get("zone_atr", 0.5)  # proximity to level in ATR
    body_mult     = cfg.get("body_mult", 2.0)  # for pin bar: shadow >= body_mult * body
    cooldown_bars = cfg.get("cooldown_bars", 12)
    tp_rr         = cfg["tp_rr"]
    atr_mult      = cfg["atr_mult_sl"]
    use_macro     = cfg.get("macro_filter", False)
    macro_p       = cfg.get("macro_ema_period", 20)

    atr1h   = _atr(b1h, 14)
    pdh_pdl = _build_pdh_pdl(b1d)
    asian   = _build_asian_ranges(b1h)

    if use_macro and b1d:
        ema_d = _ema([b["close"] for b in b1d], macro_p)
        ep_d  = [b["epoch"] for b in b1d]
    else:
        ema_d = ep_d = None

    signals    = []
    last_sig_i = -999

    for i in range(5, len(b1h)):
        if i - last_sig_i < cooldown_bars:
            continue

        bar     = b1h[i]
        bar_p   = b1h[i - 1]
        epoch   = bar["epoch"]
        atr_val = atr1h[i]
        if atr_val is None or atr_val <= 0:
            continue

        day = (epoch // 86400) * 86400
        h   = (epoch % 86400) // 3600

        resist  = []
        support = []

        lv = pdh_pdl.get(day)
        if lv:
            resist.append(lv["pdh"])
            support.append(lv["pdl"])

        if h >= 7:
            ar = asian.get(day)
            if ar:
                resist.append(ar[0])
                support.append(ar[1])

        allow_long = allow_short = True
        if ema_d is not None:
            k = bisect.bisect_right(ep_d, epoch) - 1
            if k >= 1 and ema_d[k] is not None and ema_d[k - 1] is not None:
                macro_up    = ema_d[k] > ema_d[k - 1]
                allow_long  = macro_up
                allow_short = not macro_up

        sl = atr_val * atr_mult
        tp = sl * tp_rr
        fired = False

        bar_range  = bar["high"] - bar["low"]
        bar_body   = abs(bar["close"] - bar["open"])
        bar_p_range = bar_p["high"] - bar_p["low"]
        bar_p_body  = abs(bar_p["close"] - bar_p["open"])

        if pattern_type in ("engulfing", "both"):
            is_prev_bull = bar_p["close"] > bar_p["open"]
            is_prev_bear = bar_p["close"] < bar_p["open"]
            is_curr_bear = bar["close"] < bar["open"]
            is_curr_bull = bar["close"] > bar["open"]

            # Bearish engulfing at resistance
            if (not fired and allow_short and is_prev_bull and is_curr_bear and
                    bar["open"] >= bar_p["close"] and
                    bar["close"] <= bar_p["open"] and
                    _near_level(bar["high"], resist, zone_atr, atr_val, "resist")):
                signals.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp,
                                          reason="BPS_bear_engulf")))
                last_sig_i = i
                fired = True

            # Bullish engulfing at support
            if (not fired and allow_long and is_prev_bear and is_curr_bull and
                    bar["open"] <= bar_p["close"] and
                    bar["close"] >= bar_p["open"] and
                    _near_level(bar["low"], support, zone_atr, atr_val, "support")):
                signals.append((i, Signal("BUY", sl_pips=sl, tp_pips=tp,
                                          reason="BPS_bull_engulf")))
                last_sig_i = i
                fired = True

        if not fired and pattern_type in ("pin_bar", "both") and bar_range > 1e-10:
            upper_shadow = bar["high"] - max(bar["open"], bar["close"])
            lower_shadow = min(bar["open"], bar["close"]) - bar["low"]
            close_pos    = (bar["close"] - bar["low"]) / bar_range  # 0=at low, 1=at high

            # Bearish pin (shooting star): long upper shadow at resistance
            if (not fired and allow_short and
                    bar_body > 1e-10 and
                    upper_shadow >= body_mult * bar_body and
                    close_pos <= 0.40 and
                    _near_level(bar["high"], resist, zone_atr, atr_val, "resist")):
                signals.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp,
                                          reason="BPS_bear_pin")))
                last_sig_i = i
                fired = True

            # Bullish pin (hammer): long lower shadow at support
            if (not fired and allow_long and
                    bar_body > 1e-10 and
                    lower_shadow >= body_mult * bar_body and
                    close_pos >= 0.60 and
                    _near_level(bar["low"], support, zone_atr, atr_val, "support")):
                signals.append((i, Signal("BUY", sl_pips=sl, tp_pips=tp,
                                          reason="BPS_bull_pin")))
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
    all_sigs = run_bps(b1h, b1d, cfg)
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

    for pattern_type in ["engulfing", "pin_bar", "both"]:
        for zone_atr in [0.3, 0.5, 1.0]:
            for body_mult in [1.5, 2.0, 3.0]:
                for cooldown in [6, 12, 24]:
                    for tp_rr in [1.5, 2.0, 3.0]:
                        for atr_mult in [1.0, 1.5, 2.0]:
                            for macro in [False, True]:
                                cfg = dict(
                                    pattern_type=pattern_type,
                                    zone_atr=zone_atr,
                                    body_mult=body_mult,
                                    cooldown_bars=cooldown,
                                    tp_rr=tp_rr,
                                    atr_mult_sl=atr_mult,
                                    macro_filter=macro,
                                    macro_ema_period=20,
                                )
                                label = (
                                    f"BPS {pattern_type} zone<={zone_atr}ATR "
                                    f"shadow>={body_mult}xbody "
                                    f"cool{cooldown} RR{tp_rr} ATRx{atr_mult} "
                                    f"{'MACRO' if macro else 'free'}"
                                )
                                sigs = run_bps(train_b1h, b1d, cfg)
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
    print("Bar Pattern Sequences (BPS) -- 1H -- 4-window walk-forward")
    print(f"BE@1R  max_hold=48H  {DAYS}-day dataset")
    print("Thesis: engulfing and pin bar patterns AT key levels (PDH/PDL, Asian H/L)")
    print("Patterns tested: engulfing | pin_bar | both")
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
            print(f"  No profitable BPS configs found for {sym}.\n")
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
            print(f"  No BPS config passed 3+ windows for {sym}.")

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
        print("  No BPS strategy passed 3+ windows on any symbol.")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
