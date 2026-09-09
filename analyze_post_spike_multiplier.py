"""
Phase 2: Post-spike Multiplier strategy analysis.

After a BOOM spike, price reverts ~98.5% of the time (confirmed in recoil analysis).
Deriv allows Multiplier contracts on all Boom/Crash symbols.

Strategy:
  BOOM spike UP  -> buy DOWN multiplier (short the reversion)
  CRASH spike DOWN -> buy UP multiplier (long the reversion)

This script:
  1. Shows the reversion magnitude profile at T+1..T+30 per symbol
  2. Simulates multiplier trades with all TP/SL combinations
  3. Finds the highest-EV config accounting for Deriv commission

Multiplier math (Deriv):
  stake = $1, mult = 100
  notional = $100
  TP at price move of +tp_pct: profit = tp_pct * mult * stake
  SL at price move of -sl_pct: loss   = sl_pct * mult * stake
  Commission = commission_pct * notional (charged at entry)

  Net EV per trade = WR * TP_profit - (1-WR) * SL_loss - commission
"""

import sys
from loguru import logger
from src.data.history import fetch_ticks

logger.remove()
logger.add(sys.stderr, level="ERROR")

SYMBOLS       = ["BOOM50", "CRASH50", "BOOM600", "CRASH600"]
TICK_COUNT    = 86_000
ATR_PERIOD    = 50
SPIKE_MULT    = 12.0     # same threshold as live trading config
MAX_HOLD      = 50       # ticks to hold before forced exit

MULTIPLIERS   = [100, 200, 500]
# TP/SL as fraction of entry price (0.002 = 0.2% move)
TP_PCTS       = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.010, 0.020]
SL_PCTS       = [0.001,  0.002, 0.003, 0.005, 0.010, 0.020]
COMMISSION    = 0.0003   # 0.03% of notional — conservative Deriv estimate


def compute_atr(prices, period):
    atrs  = [0.0] * len(prices)
    moves = [0.0] + [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
    for i in range(period, len(prices)):
        atrs[i] = sum(moves[i-period+1:i+1]) / period
    return atrs


def detect_spikes(prices, atrs, is_boom):
    spikes = []
    for i in range(ATR_PERIOD + 1, len(prices) - MAX_HOLD - 1):
        if atrs[i] <= 0:
            continue
        move = prices[i] - prices[i-1]
        ratio = move / atrs[i]
        if (is_boom and ratio >= SPIKE_MULT) or (not is_boom and ratio <= -SPIKE_MULT):
            spikes.append(i)
    return spikes


def reversion_profile(prices, spikes, is_boom):
    """Print reversion magnitude at each time step."""
    profile = {}
    for t in [1, 2, 3, 5, 7, 10, 15, 20, 30]:
        wins = total = 0
        rev_pcts = []
        for idx in spikes:
            if idx + t >= len(prices):
                continue
            entry      = prices[idx]
            spike_size = abs(prices[idx] - prices[idx-1])
            if spike_size <= 0 or entry <= 0:
                continue
            if is_boom:
                move = entry - prices[idx + t]        # positive = price fell (recoil)
            else:
                move = prices[idx + t] - entry        # positive = price rose (recoil)
            if move > 0:
                wins += 1
            total += 1
            rev_pcts.append(move / entry * 100)
        if total:
            profile[t] = {
                "wr":      wins / total * 100,
                "avg_rev": sum(rev_pcts) / len(rev_pcts),
                "n":       total,
            }
    return profile


def simulate_multiplier(prices, spikes, is_boom, tp_pct, sl_pct, mult):
    wins = losses = forced = 0
    ev_sum = 0.0

    for idx in spikes:
        entry = prices[idx]
        if entry <= 0:
            continue

        hit = None
        for t in range(1, MAX_HOLD + 1):
            if idx + t >= len(prices):
                break
            curr = prices[idx + t]
            if is_boom:
                move_pct = (entry - curr) / entry   # positive = price fell (we're short)
            else:
                move_pct = (curr - entry) / entry   # positive = price rose (we're long)

            if move_pct >= tp_pct:
                hit = tp_pct * mult - COMMISSION * mult
                wins += 1
                break
            elif move_pct <= -sl_pct:
                hit = -sl_pct * mult - COMMISSION * mult
                losses += 1
                break

        if hit is None:
            # Force exit at max hold
            if idx + MAX_HOLD < len(prices):
                curr = prices[idx + MAX_HOLD]
                if is_boom:
                    move_pct = (entry - curr) / entry
                else:
                    move_pct = (curr - entry) / entry
                hit = move_pct * mult - COMMISSION * mult
                forced += 1
                if move_pct > 0:
                    wins += 1
                else:
                    losses += 1
            else:
                continue

        ev_sum += hit

    total = wins + losses  # forced counted in wins/losses
    if total == 0:
        return None
    return {
        "tp": tp_pct, "sl": sl_pct, "mult": mult,
        "wr":       wins / total * 100,
        "avg_ev":   ev_sum / total,
        "total_ev": ev_sum,
        "n":        total,
        "forced":   forced,
    }


def main():
    print()
    print("=" * 110)
    print("  POST-SPIKE MULTIPLIER ANALYSIS  |  Phase 2 of 3")
    print(f"  spike_mult={SPIKE_MULT}x | ATR={ATR_PERIOD} | max_hold={MAX_HOLD}t")
    print(f"  BOOM: short after spike | CRASH: long after spike")
    print(f"  Commission={COMMISSION*100:.3f}% of notional")
    print("=" * 110)

    for symbol in SYMBOLS:
        print(f"\n  Fetching {symbol}...", flush=True)
        try:
            raw = fetch_ticks(symbol, count=TICK_COUNT)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        prices  = [float(t["quote"]) if isinstance(t, dict) else float(t) for t in raw]
        atrs    = compute_atr(prices, ATR_PERIOD)
        is_boom = symbol.startswith("BOOM")
        spikes  = detect_spikes(prices, atrs, is_boom)

        print(f"\n  {'='*90}")
        print(f"  {symbol}  |  {len(prices):,} ticks  |  {len(spikes)} spikes  |  "
              f"avg freq ~{len(prices)/max(1,len(spikes)):.0f} ticks/spike")
        print(f"  {'='*90}")

        if len(spikes) < 5:
            print("  Insufficient spikes — skip")
            continue

        # --- Reversion profile ---
        profile = reversion_profile(prices, spikes, is_boom)
        print(f"\n  Reversion profile (direction: {'DOWN' if is_boom else 'UP'} after spike)")
        print(f"  {'T':>5}  {'WR%':>7}  {'AvgMove%':>10}  {'N':>5}")
        print(f"  {'-'*5}  {'-'*7}  {'-'*10}  {'-'*5}")
        for t, d in profile.items():
            be_flag = " <-- above 50%" if d["wr"] > 50 else ""
            print(f"  {t:>5}  {d['wr']:>6.1f}%  {d['avg_rev']:>+9.4f}%  {d['n']:>5}{be_flag}")

        # --- Multiplier simulation ---
        all_results = []
        for mult in MULTIPLIERS:
            for tp in TP_PCTS:
                for sl in SL_PCTS:
                    res = simulate_multiplier(prices, spikes, is_boom, tp, sl, mult)
                    if res and res["n"] >= 5:
                        all_results.append(res)

        all_results.sort(key=lambda x: x["total_ev"], reverse=True)

        print(f"\n  Top 20 multiplier configs by total EV (stake = $1 per trade)")
        print(f"  {'Mult':>6}  {'TP%':>6}  {'SL%':>6}  {'WR%':>6}  {'AvgEV':>8}  "
              f"{'Trades':>7}  {'TotalEV':>9}  {'R:R':>5}")
        print(f"  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*7}  {'-'*9}  {'-'*5}")

        for r in all_results[:20]:
            rr = r["tp"] / r["sl"] if r["sl"] > 0 else 0
            flag = "  <-- BEST" if r == all_results[0] else ""
            print(f"  {r['mult']:>6}x  {r['tp']*100:>5.3f}%  {r['sl']*100:>5.3f}%  "
                  f"{r['wr']:>5.1f}%  {r['avg_ev']:>+7.4f}x  "
                  f"{r['n']:>7}  {r['total_ev']:>+8.3f}x  {rr:>4.1f}:1{flag}")

        if all_results:
            best = all_results[0]
            print(f"\n  Best config: mult={best['mult']}x  "
                  f"TP={best['tp']*100:.3f}%  SL={best['sl']*100:.3f}%  "
                  f"WR={best['wr']:.1f}%  AvgEV={best['avg_ev']:+.4f}x/trade  "
                  f"TotalEV={best['total_ev']:+.3f}x")
        else:
            print("\n  No positive-EV multiplier config found.")

    print()


if __name__ == "__main__":
    main()
