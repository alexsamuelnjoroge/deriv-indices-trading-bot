"""
Round Number Magnetism (RNM) -- proprietary strategy.

Core thesis:
  Human traders, algorithms, and institutions cluster their orders at round
  numbers. On XAUUSD: $3000, $3050, $3100. On EURUSD: 1.0800, 1.0850.
  These levels accumulate massive pools of stop-losses, take-profits, and
  new-position limit orders.

  When price approaches a round number:
    - Approaching from below: a wall of sell limit orders (resistance)
    - Approaching from above: a wall of buy limit orders (support)

  The signal: bar's WICK touched the round number zone but the CLOSE moved
  AWAY. This indicates the limit orders absorbed the momentum — the bar
  reached for the round number, found the wall, and bounced.

  This is the ONLY strategy in our suite using ABSOLUTE price levels.
  All other strategies use relative patterns (candle size, wick ratio, velocity,
  session shape). RNM exploits the one market property that is invariant to
  trend regime: human psychology attaches importance to round numbers regardless
  of market direction.

  Why NOT just buy/sell at the round number?
  Because the bounce must be CONFIRMED by price action: the bar must have
  touched (via wick) and already closed away. We're entering on confirmation,
  not anticipation, which avoids being run over when the round number BREAKS.

Signal (1H bars):
  Round number grids (tested by symbol):
    XAUUSD: $50 increments  (3000, 3050, 3100, ...)
    EURUSD: 0.0050 increments (1.0800, 1.0850, ...)
    GBPUSD: 0.0050 increments (1.2500, 1.2550, ...)
    USDJPY: 1.0 increments    (145, 146, ...)

  For each bar:
    1. Find nearest round number to the bar's midpoint (high+low)/2
    2. BUY trigger (round number below):
         bar.low <= round_num + touch_zone * ATR  (wick reached it from above)
         bar.close >= round_num + min_bounce * ATR (closed meaningfully above)
         bar.close > bar.open  (bullish close = confirmation)
    3. SELL trigger (round number above):
         bar.high >= round_num - touch_zone * ATR  (wick reached it from below)
         bar.close <= round_num - min_bounce * ATR (closed meaningfully below)
         bar.close < bar.open  (bearish close = confirmation)
    4. SL: ATR * mult, TP: RR * SL
    5. Cooldown per level: block re-entry within cooldown_bars of same level touch
    6. Macro filter

Level freshness:
  Each round number level is tracked independently. After a signal fires at a
  level, it needs cooldown_bars before firing again at the SAME level. Different
  levels are independent — price can bounce off $3100 immediately after $3050.
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

# Natural round number spacing per symbol
ROUND_GRIDS = {
    "frxXAUUSD": 50.0,    # $50 increments
    "frxEURUSD": 0.0050,  # 50-pip increments
    "frxGBPUSD": 0.0050,  # 50-pip increments
    "frxUSDJPY": 1.0,     # 1-yen increments
}

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


# ── Round number helpers ──────────────────────────────────────────────────────

def nearest_round(price, grid):
    """Nearest round number level to price."""
    return round(round(price / grid) * grid, 10)


def find_relevant_level(bar, grid, atr_val, touch_zone):
    """
    Returns (level, direction) if bar's wick reached a round number within
    touch_zone * ATR.
      direction='buy'  → round number is below (bar.low touched it)
      direction='sell' → round number is above (bar.high touched it)
    Returns None if no relevant level found.
    """
    mid = (bar["high"] + bar["low"]) / 2

    # Check level below (potential buy)
    level_below = round(round(bar["low"] / grid) * grid, 10)
    dist_below  = bar["low"] - level_below
    if 0 <= dist_below <= touch_zone * atr_val:
        return level_below, "buy"

    # Check level above (potential sell)
    level_above = level_below + grid
    dist_above  = level_above - bar["high"]
    if 0 <= dist_above <= touch_zone * atr_val:
        return level_above, "sell"

    return None


# ── Signal generator ──────────────────────────────────────────────────────────

def run_rnm(b1h, b1d, cfg, symbol):
    touch_zone    = cfg["touch_zone"]    # ATR multiples — how close to level counts as touch
    min_bounce    = cfg["min_bounce"]    # ATR multiples — close must be this far from level
    cooldown_bars = cfg.get("cooldown_bars", 24)
    tp_rr         = cfg["tp_rr"]
    atr_mult      = cfg["atr_mult_sl"]
    use_macro     = cfg.get("macro_filter", False)
    macro_p       = cfg.get("macro_ema_period", 20)
    grid          = ROUND_GRIDS.get(symbol, 0.0050)

    atr1h = _atr(b1h, 14)

    if use_macro and b1d:
        ema_d = _ema([b["close"] for b in b1d], macro_p)
        ep_d  = [b["epoch"] for b in b1d]
    else:
        ema_d = ep_d = None

    signals    = []
    last_sig_i = -999
    level_last = {}  # level → last bar index that fired

    start = 20

    for i in range(start, len(b1h)):
        if i - last_sig_i < cooldown_bars:
            continue

        bar   = b1h[i]
        epoch = bar["epoch"]

        atr_val = atr1h[i]
        if atr_val is None or atr_val <= 0:
            continue

        result = find_relevant_level(bar, grid, atr_val, touch_zone)
        if result is None:
            continue

        level, direction = result

        # Per-level cooldown
        if level in level_last and i - level_last[level] < cooldown_bars:
            continue

        sl = atr_val * atr_mult
        tp = sl * tp_rr

        allow_long = allow_short = True
        if ema_d is not None:
            k = bisect.bisect_right(ep_d, epoch) - 1
            if k >= 1 and ema_d[k] is not None and ema_d[k - 1] is not None:
                macro_up    = ema_d[k] > ema_d[k - 1]
                allow_long  = macro_up
                allow_short = not macro_up

        if direction == "buy":
            bounce_dist = bar["close"] - level
            if (bounce_dist >= min_bounce * atr_val
                    and bar["close"] > bar["open"]
                    and allow_long):
                signals.append((i, Signal("BUY", sl_pips=sl, tp_pips=tp,
                                          reason=f"RNM_buy_lvl{level:.4f}")))
                last_sig_i = i
                level_last[level] = i

        elif direction == "sell":
            bounce_dist = level - bar["close"]
            if (bounce_dist >= min_bounce * atr_val
                    and bar["close"] < bar["open"]
                    and allow_short):
                signals.append((i, Signal("SELL", sl_pips=sl, tp_pips=tp,
                                          reason=f"RNM_sell_lvl{level:.4f}")))
                last_sig_i = i
                level_last[level] = i

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


def run_wf(b1h, b1d, cfg, label, spread, symbol, verbose=True):
    all_sigs = run_rnm(b1h, b1d, cfg, symbol)
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

def sweep(b1h, b1d, train_end_idx, spread, symbol):
    train_b1h = b1h[:train_end_idx]
    results   = []

    for touch_zone in [0.5, 1.0, 2.0]:
        for min_bounce in [0.2, 0.4, 0.6]:
            for cooldown in [12, 24]:
                for tp_rr in [1.5, 2.0, 3.0]:
                    for atr_mult in [1.0, 1.5, 2.0]:
                        for macro in [False, True]:
                            cfg = dict(
                                touch_zone=touch_zone,
                                min_bounce=min_bounce,
                                cooldown_bars=cooldown,
                                tp_rr=tp_rr,
                                atr_mult_sl=atr_mult,
                                macro_filter=macro,
                                macro_ema_period=20,
                            )
                            label = (
                                f"RNM touch{touch_zone}ATR bounce{min_bounce}ATR "
                                f"cool{cooldown} RR{tp_rr} ATRx{atr_mult} "
                                f"{'MACRO' if macro else 'free'}"
                            )
                            sigs = run_rnm(train_b1h, b1d, cfg, symbol)
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
    print("Round Number Magnetism (RNM) -- 1H -- 4-window walk-forward")
    print(f"BE@1R  max_hold=48H  {DAYS}-day dataset")
    print("Thesis: limit orders cluster at round numbers => wick-touch + close-away = bounce entry")
    print("The ONLY strategy in the suite using absolute price levels (not relative patterns)")
    print("=" * 78 + "\n")

    all_robust = []

    for sym, spread in SYMBOLS_TO_TEST:
        print(f"\n{'=' * 78}")
        print(f"  {sym}  spread={spread}  grid={ROUND_GRIDS.get(sym)}")
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
        ranked = sweep(b1h, b1d, train_end, spread, sym)

        print(f"  {len(ranked)} configs with positive EV on training data.")
        if not ranked:
            print(f"  No profitable RNM configs found for {sym}.\n")
            continue

        print("  Top 5:")
        for ev, n, label, _ in ranked[:5]:
            print(f"    EV {ev:>+.4f}R  n={n:>3}  {label}")

        wf_results = []
        for ev, n, label, cfg in ranked[:5]:
            print(f"\n  {'-' * 70}")
            passes = run_wf(b1h, b1d, cfg, label, spread, sym, verbose=True)
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
            print(f"  No RNM config passed 3+ windows for {sym}.")

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
        print("  No RNM strategy passed 3+ windows on any symbol.")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
