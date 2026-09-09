"""
Phase 1: Profile ALL Boom/Crash synthetic indices.

For each symbol measures:
  - Spike frequency (avg ticks between spikes)
  - Spike size (avg ATR multiples)
  - ATR drop ratio: ATR in 5 ticks after spike / ATR in 5 ticks before spike
    < 1.0 means calmer post-spike = better ACCU survival conditions
  - Directional WR: % of post-spike ticks that move in recoil direction (T+1, T+5, T+10)
  - Reversion magnitude: avg % of spike recovered at T+5, T+10, T+20
    (relevant for multiplier sizing)

Output ranks symbols by ATR drop ratio to identify best ACCU candidates.
"""

import sys
from loguru import logger
from src.data.history import fetch_ticks

logger.remove()
logger.add(sys.stderr, level="ERROR")

SYMBOLS = [
    "BOOM50",   "CRASH50",
    "BOOM100",  "CRASH100",
    "BOOM150N", "CRASH150N",
    "BOOM300N", "CRASH300N",
    "BOOM500",  "CRASH500",
    "BOOM600",  "CRASH600",
    "BOOM900",  "CRASH900",
    "BOOM1000", "CRASH1000",
]

TICK_COUNT = 50_000
ATR_PERIOD = 50
SPIKE_MULT = 5.0    # low threshold to catch all spikes for profiling
MAX_T      = 30     # measure reversion up to T+30


def compute_atr(prices, period):
    atrs = [0.0] * len(prices)
    moves = [0.0] + [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
    for i in range(period, len(prices)):
        atrs[i] = sum(moves[i-period+1:i+1]) / period
    return atrs


def analyze(symbol):
    try:
        raw = fetch_ticks(symbol, count=TICK_COUNT)
    except Exception as e:
        return None, f"fetch error: {e}"

    prices = [float(t["quote"]) if isinstance(t, dict) else float(t) for t in raw]
    if len(prices) < 2000:
        return None, f"only {len(prices)} ticks"

    atrs   = compute_atr(prices, ATR_PERIOD)
    is_boom = symbol.startswith("BOOM")

    spikes      = []
    gaps        = []
    last_spike  = ATR_PERIOD

    for i in range(ATR_PERIOD + 1, len(prices) - MAX_T - 1):
        if atrs[i] <= 0:
            continue
        move  = prices[i] - prices[i-1]
        ratio = move / atrs[i]
        if (is_boom and ratio >= SPIKE_MULT) or (not is_boom and ratio <= -SPIKE_MULT):
            spikes.append(i)
            gaps.append(i - last_spike)
            last_spike = i

    if len(spikes) < 5:
        return None, f"only {len(spikes)} spikes"

    spike_atrs  = []
    atr_ratios  = []
    wr_counts   = {t: [0, 0] for t in [1, 5, 10, 20]}   # [wins, total]
    rev_sums    = {t: []     for t in [1, 5, 10, 20]}

    for idx in spikes:
        if idx < ATR_PERIOD + 6 or idx + MAX_T >= len(prices):
            continue

        spike_move = abs(prices[idx] - prices[idx-1])
        atr_now    = atrs[idx]
        if atr_now <= 0 or spike_move <= 0:
            continue

        atr_pre  = sum(abs(prices[idx-k] - prices[idx-k-1]) for k in range(1, 6)) / 5
        atr_post = sum(abs(prices[idx+k] - prices[idx+k-1]) for k in range(1, 6)) / 5

        spike_atrs.append(spike_move / atr_now)
        if atr_pre > 0:
            atr_ratios.append(atr_post / atr_pre)

        direction = 1 if is_boom else -1   # expected recoil direction

        for t in [1, 5, 10, 20]:
            if idx + t >= len(prices):
                break
            recoil = (prices[idx] - prices[idx+t]) * direction  # positive = recoil happened
            wr_counts[t][1] += 1
            if recoil > 0:
                wr_counts[t][0] += 1
            rev_pct = recoil / spike_move * 100
            rev_sums[t].append(rev_pct)

    n = len(spike_atrs)
    if n == 0:
        return None, "no valid spikes"

    return {
        "symbol":      symbol,
        "n_spikes":    n,
        "avg_freq":    sum(gaps[1:]) / max(1, len(gaps) - 1),
        "spike_size":  sum(spike_atrs) / n,
        "atr_drop":    sum(atr_ratios) / len(atr_ratios) if atr_ratios else 1.0,
        "wr_t1":       wr_counts[1][0]  / max(1, wr_counts[1][1])  * 100,
        "wr_t5":       wr_counts[5][0]  / max(1, wr_counts[5][1])  * 100,
        "wr_t10":      wr_counts[10][0] / max(1, wr_counts[10][1]) * 100,
        "rev_t5":      sum(rev_sums[5])  / max(1, len(rev_sums[5])),
        "rev_t10":     sum(rev_sums[10]) / max(1, len(rev_sums[10])),
        "rev_t20":     sum(rev_sums[20]) / max(1, len(rev_sums[20])),
    }, None


def main():
    print()
    print("=" * 110)
    print("  BOOM/CRASH SYMBOL PROFILE  |  Phase 1 of 3")
    print(f"  {TICK_COUNT//1000}k ticks/symbol | spike_mult={SPIKE_MULT}x | ATR={ATR_PERIOD}")
    print("=" * 110)

    results, errors = [], []

    for sym in SYMBOLS:
        print(f"  {sym:<12}...", end="", flush=True)
        r, err = analyze(sym)
        if r:
            results.append(r)
            print(f" {r['n_spikes']} spikes | freq={r['avg_freq']:.0f}t | "
                  f"spike={r['spike_size']:.1f}x ATR | ATRdrop={r['atr_drop']:.3f}")
        else:
            errors.append((sym, err))
            print(f" SKIP: {err}")

    if not results:
        print("\n  No results.")
        return

    results.sort(key=lambda x: x["atr_drop"])

    print()
    print(f"  {'Symbol':<12}  {'N':>5}  {'Freq':>6}  {'SpkSz':>6}  {'ATRdrop':>8}  "
          f"{'WR_T1':>7}  {'WR_T5':>7}  {'WR_T10':>7}  {'Rev_T5':>8}  {'Rev_T20':>8}")
    print(f"  {'-'*12}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*8}  "
          f"{'-'*7}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*8}")

    for r in results:
        drop_flag = " <-- calm" if r["atr_drop"] < 0.60 else ""
        print(f"  {r['symbol']:<12}  {r['n_spikes']:>5}  {r['avg_freq']:>5.0f}t  "
              f"{r['spike_size']:>5.1f}x  {r['atr_drop']:>8.3f}  "
              f"{r['wr_t1']:>6.1f}%  {r['wr_t5']:>6.1f}%  {r['wr_t10']:>6.1f}%  "
              f"{r['rev_t5']:>+7.2f}%  {r['rev_t20']:>+7.2f}%{drop_flag}")

    print()
    print("  KEY:")
    print("  Freq      = avg ticks between spikes")
    print("  SpkSz     = avg spike size in ATR multiples at moment of spike")
    print("  ATRdrop   = (ATR in 5t after spike) / (ATR in 5t before spike)")
    print("              < 0.60 = significantly calmer post-spike -> strong ACCU edge candidate")
    print("  WR_T1/5/10= % of post-spike ticks that move in recoil direction")
    print("  Rev_T5/20 = avg % of spike magnitude recovered (positive = recoil confirmed)")
    print()
    print("  RANKING: sorted by ATRdrop ascending (most calm post-spike first)")

    if errors:
        print(f"\n  Unavailable: {', '.join(s for s, _ in errors)}")
    print()


if __name__ == "__main__":
    main()
