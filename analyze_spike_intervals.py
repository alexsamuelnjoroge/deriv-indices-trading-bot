"""
Spike interval distribution analysis.

For calm-ACCU symbols (CRASH500, BOOM150N) and post-spike symbols (CRASH1000, BOOM1000):
  - Detects all spikes using the live strategy's spike_mult x long_ATR threshold
  - Measures inter-spike intervals (ticks between consecutive spikes)
  - Reports min / mean / std / percentiles
  - Recommends a max_ticks_since_spike value for CalmAccuStrategy:
    refuse new entries when ticks_since_spike > threshold (spike "overdue")

Usage:
  python analyze_spike_intervals.py
"""

import json, math
from pathlib import Path

CACHE_DIR  = Path("data")
TICK_COUNT = 60_000

SYMBOLS = [
    {"symbol": "CRASH500",  "spike_mult": 15.0, "atr_period": 50, "nominal_freq": 500},
    {"symbol": "BOOM150N",  "spike_mult": 15.0, "atr_period": 50, "nominal_freq": 150},
    {"symbol": "CRASH1000", "spike_mult": 15.0, "atr_period": 50, "nominal_freq": 1000},
    {"symbol": "BOOM1000",  "spike_mult": 15.0, "atr_period": 50, "nominal_freq": 1000},
    {"symbol": "JD50",      "spike_mult": 10.0, "atr_period": 30, "nominal_freq": 50},
    {"symbol": "JD75",      "spike_mult": 10.0, "atr_period": 30, "nominal_freq": 75},
    {"symbol": "JD100",     "spike_mult": 10.0, "atr_period": 30, "nominal_freq": 100},
]


def simple_atr(prices: list[float], period: int) -> float | None:
    hist = prices[-(period + 1):]
    if len(hist) < period + 1:
        return None
    ranges = [abs(hist[i] - hist[i - 1]) for i in range(1, len(hist))]
    avg = sum(ranges) / len(ranges)
    return avg if avg > 0 else None


def find_spikes(ticks: list[dict], spike_mult: float, atr_period: int) -> list[int]:
    """Return list of tick indices where a spike was detected."""
    prices  = []
    spikes  = []
    for i, t in enumerate(ticks):
        prices.append(float(t["quote"]))
        if len(prices) < atr_period + 2:
            continue
        atr = simple_atr(prices, atr_period)
        if atr is None:
            continue
        move = abs(prices[-1] - prices[-2])
        if move > spike_mult * atr:
            spikes.append(i)
    return spikes


def percentile(data: list[float], p: float) -> float:
    if not data:
        return float("nan")
    s = sorted(data)
    idx = (len(s) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def analyze(cfg: dict) -> None:
    sym  = cfg["symbol"]
    path = CACHE_DIR / f"{sym}_{TICK_COUNT}.json"
    if not path.exists():
        print(f"  {sym}: no cache")
        return

    with open(path) as f:
        ticks = json.load(f)

    spikes = find_spikes(ticks, cfg["spike_mult"], cfg["atr_period"])

    if len(spikes) < 5:
        print(f"  {sym}: too few spikes detected ({len(spikes)}) — adjust spike_mult?")
        return

    # Inter-spike intervals
    intervals = [spikes[i] - spikes[i - 1] for i in range(1, len(spikes))]

    n        = len(intervals)
    mean_iv  = sum(intervals) / n
    variance = sum((x - mean_iv) ** 2 for x in intervals) / n
    std_iv   = math.sqrt(variance)

    p10  = percentile(intervals, 10)
    p25  = percentile(intervals, 25)
    p50  = percentile(intervals, 50)
    p75  = percentile(intervals, 75)
    p90  = percentile(intervals, 90)
    p95  = percentile(intervals, 95)

    # Recommendation for max_ticks_since_spike:
    # Refuse entry when spike appears overdue — i.e. ticks_since_spike > p75
    # This blocks the riskiest 25% of intervals (where spike is very late)
    rec_max = int(p75)

    # Also: what fraction of 20-tick holds would contain a spike, given ticks_since_spike=rec_max?
    # If intervals are exponential with mean=mean_iv:
    # P(spike in next 20t | already past threshold) is hard to calculate without distributional form
    # Instead: among intervals longer than rec_max, what fraction have a spike within the next 20 ticks?
    long_ivs = [iv for iv in intervals if iv > rec_max]
    risky_holds = sum(1 for iv in long_ivs if iv < rec_max + 20)
    risky_pct   = risky_holds / len(long_ivs) * 100 if long_ivs else 0

    SEP = "=" * 65
    print()
    print(SEP)
    print(f"  {sym}  |  {len(spikes)} spikes in {TICK_COUNT} ticks  |  nominal_freq={cfg['nominal_freq']}")
    print(SEP)
    print(f"  Inter-spike intervals (n={n}):")
    print(f"    min={min(intervals)}  mean={mean_iv:.0f}  std={std_iv:.0f}  max={max(intervals)}")
    print(f"    p10={p10:.0f}  p25={p25:.0f}  p50={p50:.0f}  p75={p75:.0f}  p90={p90:.0f}  p95={p95:.0f}")
    print()
    print(f"  Nominal (expected) spike every {cfg['nominal_freq']} ticks")
    print(f"  Measured mean:  {mean_iv:.0f} ticks between spikes")
    print(f"  CV (std/mean):  {std_iv/mean_iv:.2f}  (1.0 = pure Poisson, <1.0 = more regular)")
    print()
    print(f"  Recommended max_ticks_since_spike: {rec_max} (= p75 of intervals)")
    print(f"  -> Blocks entry when spike is later than 75% of observed intervals")
    if long_ivs:
        print(f"  -> Among {len(long_ivs)} intervals > {rec_max}: {risky_pct:.0f}% had a spike within next 20 ticks")
    print()

    # Is the distribution more regular than Poisson (CV < 1)?
    cv = std_iv / mean_iv
    if cv < 0.7:
        print(f"  NOTE: CV={cv:.2f} — significantly more regular than Poisson.")
        print(f"        Spike countdown IS meaningful — overdue ticks predict imminent spike.")
    elif cv < 1.0:
        print(f"  NOTE: CV={cv:.2f} — slightly more regular than Poisson.")
        print(f"        Spike countdown has weak but real predictive value.")
    else:
        print(f"  NOTE: CV={cv:.2f} — as random or more random than Poisson.")
        print(f"        Spike countdown has little predictive value (memoryless).")


if __name__ == "__main__":
    print("Spike interval distribution analysis")
    print("spike_mult and atr_period match live config for each symbol\n")
    for cfg in SYMBOLS:
        analyze(cfg)
    print()
