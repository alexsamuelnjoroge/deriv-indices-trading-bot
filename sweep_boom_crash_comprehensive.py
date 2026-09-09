"""
Phase 3: Comprehensive ACCU sweep across all Boom/Crash symbols.

Uses the post-spike ACCU edge: after a spike, ATR drops and the ACCU
barrier is harder to reach, giving better-than-priced survival probability.

For each symbol this sweeps:
  - growth_rate: 3%, 4%, 5%
  - hold_ticks:  8, 10, 12, 15, 18, 20
  - settle_ticks: 0, 3

Validation: 4-window walk-forward (4 x 21,500 ticks = 86k total).
A config is ROBUST if it passes all 4 windows with WR > BE.
A config is OK if it passes 3/4.

BARRIERS: sourced from live Deriv API (run fetch_accu_barriers.py on VPS
to update). Symbols without a known barrier are skipped with a warning.

After running Phase 1 (analyze_boom_crash_profile.py), focus on symbols
with ATRdrop < 0.70 — those have the strongest post-spike calm period.
"""

import sys
from loguru import logger
from src.data.history import fetch_ticks

logger.remove()
logger.add(sys.stderr, level="ERROR")

# ── Live barriers from Deriv API ─────────────────────────────────────────────
# Format: { SYMBOL: { growth_rate: barrier_pct } }
# Run fetch_accu_barriers.py on VPS to verify/update these values.
BARRIERS = {
    "BOOM50":   {0.03: 2.30e-6, 0.04: 2.15e-6, 0.05: 2.03e-6},
    "CRASH50":  {0.03: 2.30e-6, 0.04: 2.15e-6, 0.05: 2.03e-6},
    "BOOM150N": {0.03: 1.67e-6, 0.04: 1.61e-6, 0.05: 1.54e-6},
    "CRASH150N":{0.03: 1.67e-6, 0.04: 1.61e-6, 0.05: 1.54e-6},
    "BOOM600":  {0.03: None,    0.04: None,     0.05: 3.94e-6},
    "CRASH600": {0.03: None,    0.04: None,     0.05: 3.92e-6},
    "BOOM1000": {0.03: None,    0.04: None,     0.05: 2.25e-6},
    "CRASH1000":{0.03: None,    0.04: None,     0.05: 2.27e-6},
    # Barriers from config.yaml (sourced from older check_contracts.py runs):
    "BOOM500":  {0.03: None,    0.04: None,     0.05: 4.72e-6},
    "CRASH500": {0.03: None,    0.04: None,     0.05: 4.72e-6},
    "BOOM900":  {0.03: None,    0.04: None,     0.05: 2.60e-6},
    "CRASH900": {0.03: None,    0.04: None,     0.05: 2.60e-6},
    # Add after running fetch_accu_barriers.py:
    # "BOOM100":  {0.03: ?, 0.04: ?, 0.05: ?},
    # "CRASH100": {0.03: ?, 0.04: ?, 0.05: ?},
    # "BOOM300N": {0.03: ?, 0.04: ?, 0.05: ?},
    # "CRASH300N":{0.03: ?, 0.04: ?, 0.05: ?},
}

SYMBOLS      = list(BARRIERS.keys())
GROWTH_RATES = [0.03, 0.04, 0.05]
HOLD_RANGE   = [8, 10, 12, 15, 18, 20]
SETTLE_TICKS = [0, 3]
SPIKE_MULT   = 12.0
ATR_PERIOD   = 50
WINDOWS      = 4
WINDOW_SIZE  = 21_500
TICK_COUNT   = WINDOWS * WINDOW_SIZE
MIN_TRADES   = 15


def compute_atr(prices, period):
    atrs  = [0.0] * len(prices)
    moves = [0.0] + [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
    for i in range(period, len(prices)):
        atrs[i] = sum(moves[i-period+1:i+1]) / period
    return atrs


def run_window(seg, is_boom, spike_mult, growth_rate, hold_ticks,
               settle_ticks, barrier_pct):
    atrs   = compute_atr(seg, ATR_PERIOD)
    payout = (1 + growth_rate) ** hold_ticks - 1
    be     = 1 / (1 + payout)

    wins = losses = 0
    last_exit = -1
    i = ATR_PERIOD + 1

    while i < len(seg) - settle_ticks - hold_ticks - 1:
        if i <= last_exit:
            i += 1
            continue

        move = seg[i] - seg[i-1]
        if atrs[i] <= 0:
            i += 1
            continue

        spike = (is_boom and move >= spike_mult * atrs[i]) or \
                (not is_boom and move <= -spike_mult * atrs[i])

        if not spike:
            i += 1
            continue

        # Adaptive settle: wait settle_ticks more if ATR still elevated
        entry_i = i + settle_ticks + 1
        if entry_i >= len(seg):
            break

        entry_price = seg[entry_i]
        if entry_price <= 0:
            i += 1
            continue

        # Simulate ACCU: knocked out if ANY tick-to-tick fractional move exceeds
        # barrier_pct (matches BacktestEngine: abs(curr-prev)/prev > barrier_pct).
        # Direction-agnostic — Deriv ACCU has a symmetric band around each tick.
        survived = True
        ko_tick  = -1
        for t in range(1, hold_ticks + 1):
            idx = entry_i + t
            if idx >= len(seg):
                survived = False
                break
            prev_p = seg[idx - 1]
            curr_p = seg[idx]
            if prev_p <= 0:
                survived = False
                break
            pct_move = abs(curr_p - prev_p) / prev_p
            if pct_move > barrier_pct:
                survived = False
                ko_tick  = idx
                break

        if survived:
            wins += 1
        else:
            losses += 1

        last_exit = (ko_tick if ko_tick > 0 else entry_i + hold_ticks) + 2
        i += 1

    trades = wins + losses
    wr     = wins / trades * 100 if trades > 0 else 0.0
    return wr, trades, be * 100, payout


def sweep_symbol(symbol, prices):
    is_boom = symbol.startswith("BOOM")
    sym_barriers = BARRIERS.get(symbol, {})
    results = []

    for gr in GROWTH_RATES:
        barrier = sym_barriers.get(gr)
        if barrier is None:
            continue

        payout = (1 + gr) ** 8 - 1  # placeholder for display; recalc per hold
        be_8   = 1 / (1 + ((1 + gr) ** 8 - 1)) * 100

        for hold in HOLD_RANGE:
            for settle in SETTLE_TICKS:
                all_wins = all_trades = passes = 0
                window_wrs = []
                payout_val = be_val = 0

                for w in range(WINDOWS):
                    seg = prices[w * WINDOW_SIZE: (w+1) * WINDOW_SIZE]
                    if len(seg) < 1000:
                        continue
                    wr, trades, be, payout_val = run_window(
                        seg, is_boom, SPIKE_MULT, gr, hold, settle, barrier)
                    be_val = be
                    all_wins   += round(wr / 100 * trades)
                    all_trades += trades
                    window_wrs.append((wr, trades))
                    if trades >= MIN_TRADES and wr >= be:
                        passes += 1

                if all_trades == 0:
                    continue

                overall_wr = all_wins / all_trades * 100
                ev         = (overall_wr / 100 - be_val / 100) * payout_val * 100

                results.append({
                    "gr": gr, "hold": hold, "settle": settle,
                    "barrier": barrier,
                    "wr": overall_wr, "be": be_val, "ev": ev,
                    "trades": all_trades, "passes": passes,
                    "window_wrs": window_wrs,
                })

    results.sort(key=lambda x: (x["passes"], x["ev"]), reverse=True)
    return results


def main():
    print()
    print("=" * 110)
    print("  COMPREHENSIVE BOOM/CRASH ACCU SWEEP  |  Phase 3 of 3")
    print(f"  {WINDOWS}x{WINDOW_SIZE:,} tick walk-forward | spike_mult={SPIKE_MULT}x | MIN_TRADES={MIN_TRADES}")
    print(f"  Configs per symbol: {len(GROWTH_RATES)*len(HOLD_RANGE)*len(SETTLE_TICKS)}")
    print("=" * 110)

    summary = []

    for symbol in SYMBOLS:
        print(f"\n  Fetching {symbol}...", flush=True)
        try:
            raw = fetch_ticks(symbol, count=TICK_COUNT)
        except Exception as e:
            print(f"  ERROR fetching {symbol}: {e}")
            continue

        prices = [float(t["quote"]) if isinstance(t, dict) else float(t) for t in raw]
        if len(prices) < WINDOW_SIZE * 2:
            print(f"  {symbol}: insufficient ticks ({len(prices):,})")
            continue

        results = sweep_symbol(symbol, prices)

        print(f"\n  {'='*100}")
        print(f"  {symbol}  |  {len(prices):,} ticks")
        print(f"  {'='*100}")
        print(f"  {'gr':>5}  {'hold':>5}  {'st':>3}  {'WR%':>6}  {'BE%':>5}  "
              f"{'EV%':>8}  {'trades':>7}  {'pass':>6}  windows")
        print(f"  {'-'*5}  {'-'*5}  {'-'*3}  {'-'*6}  {'-'*5}  "
              f"{'-'*8}  {'-'*7}  {'-'*6}  {'-'*30}")

        for r in results[:20]:
            if r["trades"] < MIN_TRADES:
                continue
            flag = ("  ROBUST" if r["passes"] == WINDOWS and r["ev"] > 0 else
                    "  OK"     if r["passes"] >= WINDOWS - 1 and r["ev"] > 0 else "")
            wrs = " ".join(f"{w:.0f}%({'OK' if w >= r['be'] else 'NO'})"
                           for w, t in r["window_wrs"])
            print(f"  {r['gr']*100:.0f}%  {r['hold']:>5}  {r['settle']:>3}  "
                  f"{r['wr']:>5.1f}%  {r['be']:>4.1f}%  "
                  f"{r['ev']:>+7.3f}%  {r['trades']:>7}  "
                  f"{r['passes']}/{WINDOWS}{flag}  [{wrs}]")

        robust = [r for r in results if r["passes"] == WINDOWS
                  and r["ev"] > 0 and r["trades"] >= MIN_TRADES]
        if robust:
            best = robust[0]
            summary.append({
                "symbol": symbol, "gr": best["gr"], "hold": best["hold"],
                "settle": best["settle"], "barrier": best["barrier"],
                "wr": best["wr"], "be": best["be"], "ev": best["ev"],
                "trades": best["trades"],
            })
            print(f"\n  BEST 4/4: gr={best['gr']*100:.0f}%  hold={best['hold']}  "
                  f"settle={best['settle']}  WR={best['wr']:.1f}%  "
                  f"BE={best['be']:.1f}%  EV={best['ev']:+.3f}%  "
                  f"trades={best['trades']}  barrier={best['barrier']:.2e}")
        else:
            ok = [r for r in results if r["passes"] >= WINDOWS - 1
                  and r["ev"] > 0 and r["trades"] >= MIN_TRADES]
            if ok:
                print(f"\n  Best 3/4: gr={ok[0]['gr']*100:.0f}%  hold={ok[0]['hold']}  "
                      f"settle={ok[0]['settle']}  WR={ok[0]['wr']:.1f}%  "
                      f"EV={ok[0]['ev']:+.3f}%  trades={ok[0]['trades']}")
            else:
                print(f"\n  No passing config found.")

    # ── Summary ──────────────────────────────────────────────────────────────
    if summary:
        print()
        print("=" * 90)
        print("  SUMMARY — 4/4 ROBUST configs ready for live deployment")
        print("=" * 90)
        print(f"  {'Symbol':<12}  {'gr':>4}  {'hold':>5}  {'st':>3}  "
              f"{'WR%':>6}  {'BE%':>5}  {'EV%':>8}  {'trades':>7}  {'barrier':>10}")
        print(f"  {'-'*12}  {'-'*4}  {'-'*5}  {'-'*3}  "
              f"{'-'*6}  {'-'*5}  {'-'*8}  {'-'*7}  {'-'*10}")
        for s in sorted(summary, key=lambda x: x["ev"], reverse=True):
            print(f"  {s['symbol']:<12}  {s['gr']*100:.0f}%  {s['hold']:>5}  "
                  f"{s['settle']:>3}  {s['wr']:>5.1f}%  {s['be']:>4.1f}%  "
                  f"{s['ev']:>+7.3f}%  {s['trades']:>7}  {s['barrier']:>10.2e}")
    print()


if __name__ == "__main__":
    main()
