#!/usr/bin/env python3
"""
sweep_boom_crash.py  --  counter-spike drift analysis for Boom/Crash indices

Multiplier strategy:
    After a BOOM spike  → MULTDOWN (ride the inter-spike downward drift)
    After a CRASH spike → MULTUP   (ride the inter-spike upward drift)

Accumulator analysis:
    Uses live ticks_stayed_in data from the Deriv API to compute E[NR]
    at each growth rate for each symbol.  Spike timing is memoryless so
    the API's empirical survival distribution IS the ground truth.

Usage:
    python sweep_boom_crash.py                     # both modes, all symbols
    python sweep_boom_crash.py --mode mult
    python sweep_boom_crash.py --mode accu
    python sweep_boom_crash.py --symbol BOOM500
"""

import asyncio, sys, os, argparse, statistics, math
from collections import defaultdict

sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()
from src.api.client import DerivClient

TOKEN  = os.getenv("DERIV_API_TOKEN", "")
APP_ID = os.getenv("DERIV_APP_ID", "1089")

# ── symbols ──────────────────────────────────────────────────────────────────
BOOM_SYMS  = ["BOOM300N",  "BOOM500",  "BOOM600",  "BOOM900",  "BOOM1000"]
CRASH_SYMS = ["CRASH300N", "CRASH500", "CRASH600", "CRASH900", "CRASH1000"]
ALL_SYMS   = BOOM_SYMS + CRASH_SYMS

# Available multipliers by symbol group
MULT_MAP = {
    "BOOM300N":  [20, 40, 60, 80, 100],
    "BOOM500":   [100, 150, 200, 300, 400],
    "BOOM600":   [100, 150, 200, 300, 400],
    "BOOM900":   [100, 150, 200, 300, 400],
    "BOOM1000":  [100, 200, 300, 400, 500],
    "CRASH300N": [20, 40, 60, 80, 100],
    "CRASH500":  [100, 150, 200, 300, 400],
    "CRASH600":  [100, 150, 200, 300, 400],
    "CRASH900":  [100, 150, 200, 300, 400],
    "CRASH1000": [100, 200, 300, 400, 500],
}

ACCU_GROWTH_RATES = [0.01, 0.02, 0.03, 0.04, 0.05]

# ── simulation constants ─────────────────────────────────────────────────────
BAR_SECS     = 60          # 1-min bars
N_BARS       = 5000        # ~3.5 days per symbol
COMMISSION   = 0.0003      # 0.03% of notional per side

# Spike body multiplier: body > SPIKE_MULT × rolling-median body → spike detected
SPIKE_MULTS = [3.0, 5.0, 8.0, 12.0]

# TP levels to sweep (fraction of price)
TP_PCTS = [0.0003, 0.0005, 0.001, 0.002, 0.003, 0.005]

# Settle bars after spike before entering (0 = same bar close as entry)
SETTLE_BARS_LIST = [0, 1]

MIN_TRADES = 8
N_FOLDS    = 4


# ── OHLC fetching ─────────────────────────────────────────────────────────────
async def fetch_ohlc(client, symbol, count=N_BARS, granularity=BAR_SECS):
    req = {
        "ticks_history": symbol,
        "style": "candles",
        "granularity": granularity,
        "count": count,
        "end": "latest",
    }
    r = await client._send(req)
    raw = r.get("candles", [])
    candles = []
    for c in raw:
        candles.append({
            "epoch": c.get("epoch", c.get("open_time", 0)),
            "open":  float(c["open"]),
            "high":  float(c["high"]),
            "low":   float(c["low"]),
            "close": float(c["close"]),
        })
    return candles


# ── spike detection ──────────────────────────────────────────────────────────
def detect_spikes(candles, symbol, spike_mult, lookback=50):
    """
    Spikes on Boom/Crash appear as large BAR BODIES (open→close direction),
    not wicks, because the price jumps and stays at the new level within the bar.

    BOOM  spike: close >> open (large upward body)
    CRASH spike: open >> close (large downward body)

    A bar is a spike when its body > spike_mult × rolling-median body size.
    """
    is_boom = "BOOM" in symbol
    bodies  = [abs(c["close"] - c["open"]) / max(c["open"], 1e-8)
               for c in candles]
    spikes  = []
    for i in range(lookback, len(candles)):
        med = statistics.median(bodies[i - lookback: i])
        if med <= 1e-10:
            continue
        c    = candles[i]
        body = (c["close"] - c["open"]) / max(c["open"], 1e-8)
        if is_boom  and body >  spike_mult * med:
            spikes.append((i, +1))
        elif not is_boom and body < -spike_mult * med:
            spikes.append((i, -1))
    return spikes


# ── rolling ATR (body-based, ignoring wicks) ─────────────────────────────────
def rolling_atr(candles, period=14):
    atrs = [None] * len(candles)
    for i in range(period, len(candles)):
        bodies = [abs(candles[j]["close"] - candles[j]["open"]) / candles[j]["close"]
                  for j in range(i - period, i)]
        atrs[i] = statistics.mean(bodies)
    return atrs


# ── multiplier simulation ─────────────────────────────────────────────────────
def sim_mult(candles, signals, symbol, multiplier, tp_pct, settle_bars):
    """
    After each signal bar:
      - Wait settle_bars, then enter at next bar's open
      - BOOM spike → MULTDOWN (short): profit when price falls tp_pct
      - CRASH spike → MULTUP  (long):  profit when price rises tp_pct

    Returns list of normalised returns (NR):
      win  : multiplier × tp_pct − commission
      loss : −1.0  (natural stop-out at 1/multiplier adverse move)
    """
    stop_pct   = 1.0 / multiplier
    commission = multiplier * COMMISSION
    is_boom    = "BOOM" in symbol

    nrs = []
    last_entry = -1

    for sig_idx, direction in signals:
        entry_idx = sig_idx + settle_bars + 1
        if entry_idx >= len(candles):
            continue
        if entry_idx <= last_entry:
            continue

        entry = candles[entry_idx]["open"]

        # direction of the expected drift (opposite to spike)
        # BOOM spike (+1) → drift is DOWN → short → adverse = up
        # CRASH spike (−1) → drift is UP → long → adverse = down
        if is_boom:          # MULTDOWN: short
            stop_price = entry * (1 + stop_pct)
            tp_price   = entry * (1 - tp_pct)
        else:                # MULTUP: long
            stop_price = entry * (1 - stop_pct)
            tp_price   = entry * (1 + tp_pct)

        result = None
        for j in range(entry_idx + 1, len(candles)):
            hi = candles[j]["high"]
            lo = candles[j]["low"]

            if is_boom:      # short: loss if hi hits stop, win if lo hits tp
                if hi >= stop_price:
                    result = -1.0
                    break
                if lo <= tp_price:
                    result = multiplier * tp_pct - commission
                    break
            else:            # long: loss if lo hits stop, win if hi hits tp
                if lo <= stop_price:
                    result = -1.0
                    break
                if hi >= tp_price:
                    result = multiplier * tp_pct - commission
                    break

        if result is None:
            continue        # trade still open at end of data — skip

        nrs.append(result)
        last_entry = j

    return nrs


# ── walk-forward validation ──────────────────────────────────────────────────
def walk_forward(candles, symbol, multiplier, tp_pct, spike_mult,
                 settle_bars, n_folds=N_FOLDS, min_trades=MIN_TRADES):
    n = len(candles)
    fold_size = n // n_folds
    passes = 0
    fold_nrs = []

    for fold in range(n_folds):
        start = fold * fold_size
        end   = start + fold_size
        chunk = candles[start:end]
        sigs  = detect_spikes(chunk, symbol, spike_mult)
        nrs   = sim_mult(chunk, sigs, symbol, multiplier, tp_pct, settle_bars)
        if len(nrs) < min_trades:
            continue
        e_nr = sum(nrs) / len(nrs)
        fold_nrs.append(e_nr)
        if e_nr > 0:
            passes += 1

    return passes, fold_nrs


# ── accumulator live analysis ─────────────────────────────────────────────────
async def analyze_accu(client, symbol):
    """
    Query the live API for each growth rate and compute E[NR] from the
    ticks_stayed_in empirical survival distribution.

    Assumption:
        k >= max_ticks  →  survived to expiry  →  NR = (1+g)^max_ticks − 1
        k <  max_ticks  →  knocked out          →  NR = −1.0
    """
    results = []
    for gr in ACCU_GROWTH_RATES:
        req = {
            "proposal": 1, "amount": 1, "basis": "stake",
            "contract_type": "ACCU", "currency": "USD",
            "underlying_symbol": symbol, "growth_rate": gr,
        }
        try:
            r = await client._send(req)
            prop    = r.get("proposal", {})
            details = prop.get("contract_details", {})
            hist    = details.get("ticks_stayed_in", [])
            mt      = details.get("maximum_ticks", 90)
            if not hist:
                continue

            nrs = []
            for k in hist:
                if k >= mt:
                    nrs.append((1 + gr) ** mt - 1)
                else:
                    nrs.append(-1.0)

            p_surv    = sum(1 for k in hist if k >= mt) / len(hist)
            e_nr      = sum(nrs) / len(nrs)
            mean_ticks = sum(hist) / len(hist)

            results.append({
                "growth_rate": gr,
                "max_ticks":   mt,
                "mean_ticks":  round(mean_ticks, 1),
                "p_survive":   round(p_surv, 3),
                "e_nr":        round(e_nr, 4),
                "surv_payout": round((1 + gr) ** mt - 1, 2),
            })
        except Exception as e:
            print(f"  [{symbol}] accu gr={gr} error: {e}")

    return results


# ── multiplier sweep for one symbol ──────────────────────────────────────────
def sweep_mult_symbol(candles, symbol):
    best = []   # (e_nr, passes, config)

    for spike_mult in SPIKE_MULTS:
        sigs = detect_spikes(candles, symbol, spike_mult)
        n_spikes = len(sigs)
        if n_spikes < MIN_TRADES * N_FOLDS:
            continue

        for mult in MULT_MAP[symbol]:
            for tp_pct in TP_PCTS:
                for sb in SETTLE_BARS_LIST:
                    passes, fold_nrs = walk_forward(
                        candles, symbol, mult, tp_pct, spike_mult, sb
                    )
                    if not fold_nrs:
                        continue
                    e_nr = sum(fold_nrs) / len(fold_nrs)
                    best.append({
                        "sym":      symbol,
                        "spike":    spike_mult,
                        "mult":     mult,
                        "tp":       tp_pct,
                        "settle":   sb,
                        "passes":   passes,
                        "folds":    len(fold_nrs),
                        "e_nr":     round(e_nr, 4),
                        "n_spikes": n_spikes,
                    })

    # Sort: prefer 4/4 passes, then highest E[NR]
    best.sort(key=lambda x: (-x["passes"], -x["e_nr"]))
    return best


# ── printing helpers ──────────────────────────────────────────────────────────
def print_mult_results(all_results, top_n=5):
    print()
    print("=" * 72)
    print("  MULTIPLIER RESULTS  (counter-spike drift strategy)")
    print("=" * 72)
    print(f"{'Symbol':<12} {'Mult':>5} {'TP%':>7} {'SpikeX':>7} {'Stl':>4} "
          f"{'WF':>5} {'E[NR]':>8}")
    print("-" * 72)

    for sym, rows in all_results.items():
        if not rows:
            print(f"{sym:<12}  no qualifying configurations")
            continue
        top = rows[:top_n]
        for r in top:
            wf_str = f"{r['passes']}/{r['folds']}"
            flag   = " <--" if r["passes"] == r["folds"] else ""
            print(f"{r['sym']:<12} {r['mult']:>5}x {r['tp']*100:>6.3f}% "
                  f"{r['spike']:>7.0f}x {r['settle']:>4} "
                  f"{wf_str:>5} {r['e_nr']:>+8.4f}{flag}")
        print()


def print_accu_results(all_accu):
    print()
    print("=" * 72)
    print("  ACCUMULATOR RESULTS  (empirical survival from live API)")
    print("=" * 72)
    print(f"{'Symbol':<12} {'Growth':>7} {'MaxTk':>6} {'MeanTk':>7} "
          f"{'P(surv)':>8} {'SurvPay':>9} {'E[NR]':>8}")
    print("-" * 72)

    for sym, rows in all_accu.items():
        if not rows:
            print(f"{sym:<12}  no data")
            continue
        for r in rows:
            flag = " <--" if r["e_nr"] > 0 else ""
            print(f"{sym:<12} {r['growth_rate']*100:>6.0f}% "
                  f"{r['max_ticks']:>6} {r['mean_ticks']:>7.1f} "
                  f"{r['p_survive']:>8.3f} {r['surv_payout']:>8.2f}x "
                  f"{r['e_nr']:>+8.4f}{flag}")
        print()


# ── main ──────────────────────────────────────────────────────────────────────
async def main(args):
    client = DerivClient(api_token=TOKEN, app_id=APP_ID)
    await client.connect()

    symbols = [args.symbol] if args.symbol else ALL_SYMS
    mode    = args.mode

    mult_results = {}
    accu_results = {}

    for sym in symbols:
        print(f"\n[{sym}] fetching {N_BARS} bars @ {BAR_SECS}s ...", end=" ", flush=True)
        try:
            candles = await fetch_ohlc(client, sym)
        except Exception as e:
            print(f"ERROR: {e}")
            continue
        print(f"{len(candles)} bars")

        # ── multiplier sweep ──
        if mode in ("mult", "both"):
            rows = sweep_mult_symbol(candles, sym)
            mult_results[sym] = rows
            top = [r for r in rows if r["passes"] == r["folds"]]
            if top:
                r = top[0]
                print(f"  MULT best: {r['mult']}x TP={r['tp']*100:.3f}% "
                      f"spike={r['spike']:.0f}x => "
                      f"E[NR]={r['e_nr']:+.4f} ({r['passes']}/{r['folds']})")
            else:
                print("  MULT: no 4/4 configurations found")

        # ── accumulator analysis ──
        if mode in ("accu", "both"):
            print(f"  [{sym}] querying accumulator proposals ...", end=" ", flush=True)
            rows = await analyze_accu(client, sym)
            accu_results[sym] = rows
            best_accu = max(rows, key=lambda x: x["e_nr"]) if rows else None
            if best_accu:
                print(f"best gr={best_accu['growth_rate']*100:.0f}% "
                      f"E[NR]={best_accu['e_nr']:+.4f} "
                      f"p_surv={best_accu['p_survive']:.3f}")
            else:
                print("no data")

    await client.disconnect()

    if mode in ("mult", "both") and mult_results:
        print_mult_results(mult_results)

    if mode in ("accu", "both") and accu_results:
        print_accu_results(accu_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",   default="both",
                        choices=["mult", "accu", "both"])
    parser.add_argument("--symbol", default=None,
                        help="Single symbol, e.g. BOOM500")
    args = parser.parse_args()
    asyncio.run(main(args))
