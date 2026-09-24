"""
Deep statistical analysis of live tick digit patterns.
Collects all symbols in PARALLEL for DURATION seconds, then runs:

  1. Digit distribution + chi-squared uniformity test
  2. Digit transition matrix  P(next | current)
  3. Win/loss autocorrelation at lags 1-5
  4. Streak recovery  P(WIN | after N consecutive losses)
  5. Price-level digit bias

Usage:
  python analyze_digit_patterns.py R_50 600
  python analyze_digit_patterns.py R_25 R_50 R_75 300
  python analyze_digit_patterns.py R_10 R_25 R_50 R_75 R_100 600
"""
import asyncio
import math
import os
import sys
from collections import defaultdict
from dotenv import load_dotenv
from src.api.client import DerivClient

SYMBOLS  = [a for a in sys.argv[1:] if not a.isdigit()] or ["R_50"]
DURATION = int(next((a for a in sys.argv[1:] if a.isdigit()), 600))
PIP_SIZE = {"R_10": 3, "R_25": 3, "R_50": 4, "R_75": 4, "R_100": 2}
BARRIER  = 4  # DIGITOVER(4): wins on digits 5-9


def chi2_uniform(counts: list) -> tuple:
    n = sum(counts)
    k = len(counts)
    expected = n / k
    stat = sum((c - expected) ** 2 / expected for c in counts)
    # df = k-1 = 9 critical values
    if stat > 27.9:
        return stat, "p<0.001 *** NON-UNIFORM"
    if stat > 21.7:
        return stat, "p<0.01  **  NON-UNIFORM"
    if stat > 16.9:
        return stat, "p<0.05  *   possibly non-uniform"
    return stat, "p>0.05      uniform (no exploitable edge)"


def analyze(symbol: str, digits: list, prices: list, pip: int):
    n = len(digits)
    if n < 100:
        print(f"\n[{symbol}] Only {n} ticks collected — need 100+ for meaningful analysis")
        return

    print(f"\n{'='*62}")
    print(f"[{symbol}]  {n} ticks  pip_size={pip}")
    print(f"{'='*62}")

    # 1. Overall distribution
    counts = [digits.count(d) for d in range(10)]
    over4  = sum(counts[d] for d in range(5, 10))
    stat, verdict = chi2_uniform(counts)
    print("\n1. DIGIT DISTRIBUTION")
    print("   " + "  ".join(f"{d}:{counts[d]/n*100:.1f}%" for d in range(10)))
    print(f"   OVER({BARRIER}) WR : {over4/n*100:.2f}%  (break-even at payout=95% is 51.3%)")
    print(f"   Chi-squared   : {stat:.2f}  {verdict}")

    # 2. Transition matrix
    print("\n2. DIGIT TRANSITION MATRIX  P(next | current)  [%]")
    trans = defaultdict(lambda: defaultdict(int))
    for i in range(n - 1):
        trans[digits[i]][digits[i + 1]] += 1

    print("        " + "".join(f"  {d:2d}" for d in range(10)))
    notable = []
    for row in range(10):
        total_row = sum(trans[row].values())
        if total_row == 0:
            continue
        cells = []
        for col in range(10):
            pct = trans[row][col] / total_row * 100
            cells.append(f"{pct:4.0f}")
            if abs(pct - 10) >= 10 and total_row >= 20:
                notable.append((row, col, pct, total_row))
        over4_row = sum(trans[row][d] for d in range(5, 10)) / total_row * 100
        print(f"   row {row} | " + " ".join(cells) + f"   [OVER4={over4_row:.0f}%]")

    if notable:
        print(f"\n   *** NOTABLE TRANSITIONS (>=10pp from expected 10%):")
        for row, col, pct, total in notable:
            tag = "HIGH" if pct > 10 else "LOW "
            over4_row = sum(trans[row][d] for d in range(5, 10)) / total * 100
            print(f"   After {row} → digit {col}: {pct:.1f}% ({tag})  |  OVER4 after {row}: {over4_row:.1f}%")
    else:
        print(f"\n   No notable transitions (all within ±10pp of 10%)")

    # 3. Win/loss autocorrelation
    print(f"\n3. WIN/LOSS AUTOCORRELATION  (WIN = digit > {BARRIER})")
    wins = [1 if d > BARRIER else 0 for d in digits]
    mean_w = sum(wins) / n
    var_w  = sum((w - mean_w) ** 2 for w in wins) / n
    sig_threshold = 2 / math.sqrt(n)
    if var_w > 0:
        for lag in range(1, 6):
            cov = sum((wins[i] - mean_w) * (wins[i + lag] - mean_w)
                      for i in range(n - lag)) / (n - lag)
            r = cov / var_w
            sig = " *** SIGNIFICANT" if abs(r) > sig_threshold else ""
            print(f"   lag {lag}: r={r:+.4f}{sig}")
    else:
        print("   All results identical — variance = 0")

    # 4. Streak recovery
    print(f"\n4. STREAK RECOVERY  P(WIN | after N consecutive losses)")
    for streak_len in range(1, 6):
        opps = wins_after = 0
        i = 0
        while i < n - streak_len:
            if all(wins[i + j] == 0 for j in range(streak_len)):
                opps += 1
                wins_after += wins[i + streak_len]
                i += streak_len
            else:
                i += 1
        if opps >= 10:
            wr = wins_after / opps * 100
            base = over4 / n * 100
            diff = wr - base
            sig = " **" if abs(diff) > 5 else ""
            print(f"   After {streak_len} loss{'es' if streak_len>1 else ''}: "
                  f"{wins_after}/{opps} = {wr:.1f}%  (base={base:.1f}%, diff={diff:+.1f}%{sig})")

    # 5. Price-level bias
    if prices:
        print(f"\n5. PRICE-LEVEL DIGIT BIAS  (4 buckets by price quartile)")
        lo, hi = min(prices), max(prices)
        if hi > lo:
            bucket_size = (hi - lo) / 4
            buckets = defaultdict(list)
            for p, d in zip(prices, digits):
                b = min(3, int((p - lo) / bucket_size))
                buckets[b].append(d)
            for b in range(4):
                bd = buckets[b]
                if len(bd) < 20:
                    continue
                over = sum(1 for d in bd if d > BARRIER)
                lo_b = lo + b * bucket_size
                hi_b = lo_b + bucket_size
                wr_b = over / len(bd) * 100
                print(f"   {lo_b:.4f}–{hi_b:.4f}: {over}/{len(bd)} = {wr_b:.1f}% OVER{BARRIER}")
        else:
            print("   Price range too narrow to bucket")


async def collect_symbol(client, symbol: str, duration: int):
    pip = PIP_SIZE.get(symbol, 4)
    digits: list = []
    prices: list = []

    # Factory avoids closure-capture problem in parallel collection
    def make_callback(d_list, p_list, dp):
        async def on_tick(tick: dict):
            q = tick.get("quote")
            if q is not None:
                p_list.append(float(q))
                d_list.append(int(f"{float(q):.{dp}f}"[-1]))
        return on_tick

    client.on_tick(symbol, make_callback(digits, prices, pip))
    await client.subscribe_ticks(symbol)
    print(f"[{symbol}] subscribed, collecting {duration}s …", flush=True)
    await asyncio.sleep(duration)
    return symbol, digits, prices, pip


async def main():
    load_dotenv()
    token = os.getenv("DERIV_API_TOKEN") or os.getenv("DERIV_TOKEN")
    if not token:
        print("ERROR: DERIV_API_TOKEN not set")
        return

    client = DerivClient(token)
    await client.connect()

    print(f"Collecting {len(SYMBOLS)} symbol(s) in parallel for {DURATION}s: {SYMBOLS}")
    results = await asyncio.gather(
        *[collect_symbol(client, sym, DURATION) for sym in SYMBOLS]
    )

    for symbol, digits, prices, pip in results:
        analyze(symbol, digits, prices, pip)

    await client.disconnect()


asyncio.run(main())
