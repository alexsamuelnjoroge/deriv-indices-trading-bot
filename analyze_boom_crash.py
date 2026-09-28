"""
Boom/Crash edge analysis — collects live ticks and mines for exploitable patterns.

Analyses run:
  1. Spike detection & frequency  — how often do spikes occur, are they clustered?
  2. Post-spike WR at settle=0,1,2,3 × dur=1,2,3,5,10,20t — optimize recoil params
  3. Inter-spike interval histogram — is there a predictable cool-off period?
  4. Pre-spike drift  — does price trend up/down in the N ticks before a spike?
  5. Second-spike rate — how often does a second spike follow within K ticks?
  6. Digit distribution — chi-squared uniformity on last digit

Usage:
  source venv/bin/activate
  python3 analyze_boom_crash.py BOOM500 600
  python3 analyze_boom_crash.py BOOM500 CRASH500 900
  python3 analyze_boom_crash.py BOOM500 BOOM1000 CRASH500 CRASH1000 900
"""

import asyncio
import math
import os
import sys
from collections import defaultdict
from dotenv import load_dotenv
from src.api.client import DerivClient

SYMBOLS  = [a for a in sys.argv[1:] if not a.isdigit()] or ["BOOM500"]
DURATION = int(next((a for a in sys.argv[1:] if a.isdigit()), 600))

# Spike detection
ATR_PERIOD   = 50
SPIKE_MULT   = 2.5    # move >= SPIKE_MULT × ATR = spike
# Post-spike analysis grid
SETTLE_RANGE = [0, 1, 2, 3]
DUR_RANGE    = [1, 2, 3, 5, 10, 20]
# Pre-spike drift window
PRE_WINDOW   = 10   # ticks before spike to measure drift
# Second-spike check window
SECOND_K     = 30   # ticks after spike to look for a second spike


def compute_atr(prices: list, period: int) -> list:
    """ATR as |close - prev_close| for tick data."""
    if len(prices) < 2:
        return [0.0]
    trs = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
    atrs = []
    avg = sum(trs[:period]) / period if len(trs) >= period else sum(trs) / max(len(trs), 1)
    for i, tr in enumerate(trs):
        if i < period:
            atrs.append(avg)
        else:
            avg = (avg * (period - 1) + tr) / period
            atrs.append(avg)
    return atrs


def analyze(symbol: str, prices: list, epochs: list):
    n = len(prices)
    if n < ATR_PERIOD + 20:
        print(f"\n[{symbol}] Only {n} ticks — need {ATR_PERIOD + 20}+ for analysis")
        return

    print(f"\n{'='*66}")
    print(f"[{symbol}]  {n} ticks over {DURATION}s")
    print(f"{'='*66}")

    atrs = compute_atr(prices, ATR_PERIOD)
    moves = [prices[i] - prices[i-1] for i in range(1, n)]

    # ── 1. Spike detection ──────────────────────────────────────────
    spikes = []  # list of (index_in_prices, direction: 'up'/'down', size_in_atrs)
    for i, (move, atr) in enumerate(zip(moves, atrs)):
        if atr == 0:
            continue
        ratio = abs(move) / atr
        if ratio >= SPIKE_MULT:
            direction = "up" if move > 0 else "down"
            price_idx = i + 1  # spike lands on prices[price_idx]
            spikes.append((price_idx, direction, ratio))

    print(f"\n1. SPIKE DETECTION  (threshold = {SPIKE_MULT}× ATR)")
    print(f"   Total spikes : {len(spikes)}")
    if not spikes:
        print("   No spikes found — try a lower SPIKE_MULT or longer duration")
    else:
        up_spikes   = sum(1 for _, d, _ in spikes if d == "up")
        down_spikes = len(spikes) - up_spikes
        avg_ratio   = sum(r for _, _, r in spikes) / len(spikes)
        rate        = len(spikes) / (DURATION / 60)
        print(f"   Up spikes    : {up_spikes}  |  Down spikes: {down_spikes}")
        print(f"   Rate         : {rate:.1f} spikes/min")
        print(f"   Avg size     : {avg_ratio:.1f}× ATR")

    # ── 2. Post-spike WR grid ───────────────────────────────────────
    print(f"\n2. POST-SPIKE WIN RATE  (settle ticks × contract duration)")
    if not spikes:
        print("   No spikes to analyze")
    else:
        # Build outcome table: [settle][dur] = (wins, total)
        results = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for idx, direction, _ in spikes:
            # Expected direction: Boom up-spike → price should fall (PUT wins if next < current)
            #                     Crash down-spike → price should rise (CALL wins if next > current)
            # We check the price change over each (settle+dur) window
            for settle in SETTLE_RANGE:
                entry_idx = idx + settle
                if entry_idx >= n:
                    continue
                entry_price = prices[entry_idx]
                for dur in DUR_RANGE:
                    exit_idx = entry_idx + dur
                    if exit_idx >= n:
                        continue
                    exit_price = prices[exit_idx]
                    delta = exit_price - entry_price
                    if direction == "up":
                        won = delta < 0  # PUT wins
                    else:
                        won = delta > 0  # CALL wins
                    results[settle][dur][0] += int(won)
                    results[settle][dur][1] += 1

        # Print table
        print(f"   settle\\dur  " + "".join(f"  {d:>4}t" for d in DUR_RANGE))
        best_wr = 0
        best_cell = None
        for settle in SETTLE_RANGE:
            row = f"   settle={settle}    "
            for dur in DUR_RANGE:
                w, t = results[settle][dur]
                if t >= 5:
                    wr = w / t * 100
                    if wr > best_wr:
                        best_wr = wr
                        best_cell = (settle, dur, w, t)
                    row += f"  {wr:4.0f}%"
                else:
                    row += "    -- "
            print(row)
        if best_cell:
            s, d, w, t = best_cell
            print(f"\n   Best: settle={s}t dur={d}t → WR={w/t*100:.1f}% ({w}/{t} spikes)")

    # ── 3. Inter-spike interval ─────────────────────────────────────
    print(f"\n3. INTER-SPIKE INTERVAL  (ticks between consecutive spikes)")
    if len(spikes) < 2:
        print("   Need ≥2 spikes")
    else:
        intervals = [spikes[i+1][0] - spikes[i][0] for i in range(len(spikes)-1)]
        avg_iv   = sum(intervals) / len(intervals)
        min_iv   = min(intervals)
        max_iv   = max(intervals)
        # Histogram buckets
        buckets  = [0]*6  # <10, 10-30, 30-60, 60-100, 100-200, 200+
        thresholds = [10, 30, 60, 100, 200]
        for iv in intervals:
            b = sum(1 for t in thresholds if iv >= t)
            buckets[b] += 1
        labels = ["<10", "10-30", "30-60", "60-100", "100-200", "200+"]
        print(f"   Count: {len(intervals)}  |  Avg: {avg_iv:.1f}t  |  Min: {min_iv}t  |  Max: {max_iv}t")
        print("   " + "  ".join(f"{l}:{buckets[i]}" for i, l in enumerate(labels)))

        # Cool-off check: if we skip K ticks after a spike, do we avoid catching next spike?
        for skip in [5, 10, 20, 30]:
            caught = sum(1 for iv in intervals if iv <= skip)
            print(f"   Cooldown={skip}t: would catch {caught}/{len(intervals)} second-spikes "
                  f"({caught/len(intervals)*100:.0f}% exposure)")

    # ── 4. Pre-spike drift ──────────────────────────────────────────
    print(f"\n4. PRE-SPIKE DRIFT  (avg price change in {PRE_WINDOW} ticks before spike)")
    if not spikes:
        print("   No spikes")
    else:
        pre_drifts = []
        for idx, direction, _ in spikes:
            start = idx - PRE_WINDOW
            if start < 0:
                continue
            drift = prices[idx - 1] - prices[start]  # price move in window before spike
            pre_drifts.append((direction, drift))

        if pre_drifts:
            up_pre   = [d for dir_, d in pre_drifts if dir_ == "up"]
            down_pre = [d for dir_, d in pre_drifts if dir_ == "down"]
            if up_pre:
                avg_up_pre = sum(up_pre) / len(up_pre)
                pos_pre_up = sum(1 for d in up_pre if d > 0) / len(up_pre) * 100
                print(f"   Before UP spikes  : avg drift = {avg_up_pre:+.4f}  "
                      f"({pos_pre_up:.0f}% were rising pre-spike)")
            if down_pre:
                avg_dn_pre = sum(down_pre) / len(down_pre)
                pos_pre_dn = sum(1 for d in down_pre if d < 0) / len(down_pre) * 100
                print(f"   Before DOWN spikes: avg drift = {avg_dn_pre:+.4f}  "
                      f"({pos_pre_dn:.0f}% were falling pre-spike)")

    # ── 5. Second-spike rate ────────────────────────────────────────
    print(f"\n5. SECOND-SPIKE RATE  (within {SECOND_K} ticks of first spike)")
    if len(spikes) < 2:
        print("   Need ≥2 spikes")
    else:
        spike_idxs = set(idx for idx, _, _ in spikes)
        second = 0
        for idx, _, _ in spikes:
            for k in range(1, SECOND_K + 1):
                if idx + k in spike_idxs:
                    second += 1
                    break
        print(f"   {second}/{len(spikes)} spikes followed by another within {SECOND_K}t "
              f"({second/len(spikes)*100:.0f}%)")

    # ── 6. Digit distribution ───────────────────────────────────────
    pip_size = 2  # Boom/Crash price has 2dp
    digits = [int(f"{p:.{pip_size}f}"[-1]) for p in prices]
    counts = [digits.count(d) for d in range(10)]
    nn = len(digits)
    expected = nn / 10
    chi2 = sum((c - expected)**2 / expected for c in counts)
    verdict = ("p<0.001 *** NON-UNIFORM" if chi2 > 27.9 else
               "p<0.01  ** NON-UNIFORM"  if chi2 > 21.7 else
               "p<0.05  *  possibly non-uniform" if chi2 > 16.9 else
               "p>0.05     uniform")
    print(f"\n6. DIGIT DISTRIBUTION  (pip_size={pip_size})")
    print("   " + "  ".join(f"{d}:{counts[d]/nn*100:.1f}%" for d in range(10)))
    print(f"   Chi²={chi2:.2f}  {verdict}")

    # Autocorrelation at lag 1-3
    wins = [1 if d > 4 else 0 for d in digits]
    mean_w = sum(wins) / nn
    var_w  = sum((w - mean_w)**2 for w in wins) / nn
    sig_t  = 2 / math.sqrt(nn)
    if var_w > 0:
        print("   Autocorrelation:")
        for lag in range(1, 4):
            cov = sum((wins[i] - mean_w)*(wins[i+lag] - mean_w)
                      for i in range(nn - lag)) / (nn - lag)
            r = cov / var_w
            sig = " *** SIGNIFICANT" if abs(r) > sig_t else ""
            print(f"     lag {lag}: r={r:+.4f}{sig}")


async def collect(client, symbol: str, duration: int):
    prices, epochs = [], []

    async def on_tick(tick: dict):
        q = tick.get("quote")
        if q is not None:
            prices.append(float(q))
            epochs.append(tick.get("epoch", 0))

    client.on_tick(symbol, on_tick)
    await client.subscribe_ticks(symbol)
    print(f"[{symbol}] subscribed, collecting {duration}s…", flush=True)
    await asyncio.sleep(duration)
    return symbol, prices, epochs


async def main():
    load_dotenv()
    token = os.getenv("DERIV_API_TOKEN") or os.getenv("DERIV_TOKEN")
    if not token:
        print("ERROR: DERIV_API_TOKEN not set")
        return

    client = DerivClient(token)
    await client.connect()

    print(f"Collecting {SYMBOLS} in parallel for {DURATION}s…")
    results = await asyncio.gather(*[collect(client, s, DURATION) for s in SYMBOLS])

    for symbol, prices, epochs in results:
        analyze(symbol, prices, epochs)

    await client.disconnect()


asyncio.run(main())
