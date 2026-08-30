"""
analyze_index.py — Algorithmic property research for Deriv synthetic indices.

Investigates structural properties of Deriv's price-generation engine to design
strategies from first principles rather than historical pattern mining.

Key questions answered:
  CRASH/BOOM: Are crashes periodic or random? ("due effect")
              How fast does price recover post-spike?
              Is inter-spike drift consistent enough to trade?
  R_ VOLATILITY: Is digit distribution truly uniform? Are ticks autocorrelated?
  JD: Is spike direction balanced? How consistent is post-spike recoil?

Usage:
  python analyze_index.py --symbol CRASH150N
  python analyze_index.py --symbol R_50
  python analyze_index.py --family crash
  python analyze_index.py --family boom
  python analyze_index.py --family vol
  python analyze_index.py --family jd
  python analyze_index.py --symbol ALL
"""

import argparse
import math
import statistics
import sys
from collections import Counter, defaultdict

from loguru import logger

from src.data.history import fetch_ticks

logger.remove()
logger.add(sys.stderr, level="ERROR")

TICK_COUNT = 100_000
SEP    = "=" * 100

SYMBOL_META = {
    "CRASH150N": {"direction": "crash", "family": "crash_boom", "rate": 150},
    "CRASH300N": {"direction": "crash", "family": "crash_boom", "rate": 300},
    "CRASH500":  {"direction": "crash", "family": "crash_boom", "rate": 500},
    "CRASH1000": {"direction": "crash", "family": "crash_boom", "rate": 1000},
    "BOOM150N":  {"direction": "boom",  "family": "crash_boom", "rate": 150},
    "BOOM600":   {"direction": "boom",  "family": "crash_boom", "rate": 600},
    "BOOM1000":  {"direction": "boom",  "family": "crash_boom", "rate": 1000},
    "R_10":  {"direction": "vol", "family": "volatility"},
    "R_25":  {"direction": "vol", "family": "volatility"},
    "R_50":  {"direction": "vol", "family": "volatility"},
    "R_75":  {"direction": "vol", "family": "volatility"},
    "R_100": {"direction": "vol", "family": "volatility"},
    "JD50":  {"direction": "jd", "family": "jd"},
    "JD75":  {"direction": "jd", "family": "jd"},
}

FAMILIES = {
    "crash": ["CRASH150N", "CRASH300N", "CRASH500", "CRASH1000"],
    "boom":  ["BOOM150N", "BOOM600", "BOOM1000"],
    "vol":   ["R_10", "R_25", "R_50", "R_75", "R_100"],
    "jd":    ["JD50", "JD75"],
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def pearson(x: list, y: list) -> float:
    if len(x) < 3:
        return 0.0
    mx, my = statistics.mean(x), statistics.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx  = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy  = math.sqrt(sum((b - my) ** 2 for b in y))
    return num / (dx * dy) if dx * dy > 0 else 0.0


def infer_decimal_places(prices: list) -> int:
    """Detect the most common decimal-place count from the first 500 prices."""
    counter = Counter()
    for p in prices[:500]:
        s = f"{p:.8f}".rstrip("0")
        dp = len(s.split(".")[1]) if "." in s else 0
        counter[dp] += 1
    return counter.most_common(1)[0][0]


def last_digit(price: float, dp: int) -> int:
    """Last digit of price at the given decimal precision."""
    return int(round(price * (10 ** dp))) % 10


def detect_spikes(prices: list, direction: str, rate: int) -> list:
    """
    Detect spike ticks calibrated to the expected rate.

    Uses the (1 - 1/rate) percentile of all absolute moves as the threshold,
    then filters by direction. Returns a list of indices into prices.
    """
    moves    = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    abs_mv   = sorted(abs(m) for m in moves)
    idx      = min(int((1.0 - 1.0 / rate) * len(abs_mv)), len(abs_mv) - 1)
    threshold = abs_mv[idx]

    spikes = []
    for i, m in enumerate(moves):
        if direction == "crash" and m < -threshold:
            spikes.append(i + 1)
        elif direction == "boom" and m > threshold:
            spikes.append(i + 1)
        elif direction == "jd" and abs(m) >= threshold:
            spikes.append(i + 1)
    return spikes


# ─── CRASH/BOOM analyses ──────────────────────────────────────────────────────

def spike_timing(prices: list, spikes: list, meta: dict) -> None:
    """
    1. Inter-spike interval distribution + coefficient of variation.
       CV ≈ 1.0 → geometric / memoryless (no timing edge).
       CV < 0.7 → approximately periodic (timing edge likely).

    2. Conditional crash probability — the "due effect" test.
       If P(crash | T ticks since last) increases with T → crashes become more
       predictable the longer it has been since the last one.
    """
    n    = len(prices)
    rate = meta["rate"]

    print(f"\n  ┌─ SPIKE TIMING  [{len(spikes)} spikes detected in {n:,} ticks]")

    if len(spikes) < 20:
        print(f"  │  Too few spikes ({len(spikes)}) for reliable analysis — need 20+")
        print(f"  └─")
        return

    # ── Inter-spike intervals ─────────────────────────────────────────────────
    intervals = [spikes[i] - spikes[i - 1] for i in range(1, len(spikes))]
    mean_iv  = statistics.mean(intervals)
    med_iv   = statistics.median(intervals)
    std_iv   = statistics.stdev(intervals) if len(intervals) > 1 else 0.0
    cv       = std_iv / mean_iv if mean_iv > 0 else 0.0

    print(f"  │")
    print(f"  │  Inter-spike interval statistics:")
    print(f"  │    Advertised rate : 1 per {rate} ticks")
    print(f"  │    Observed mean   : 1 per {mean_iv:.1f} ticks")
    print(f"  │    Median / StdDev : {med_iv:.0f} / {std_iv:.1f} ticks")
    print(f"  │    Min / Max       : {min(intervals)} / {max(intervals)} ticks")
    print(f"  │    CV (std/mean)   : {cv:.3f}")

    if cv < 0.60:
        cv_verdict = "STRONGLY PERIODIC  — crashes arrive on a near-fixed schedule"
    elif cv < 0.80:
        cv_verdict = "MILDLY PERIODIC    — some schedule regularity, timing edge possible"
    elif cv < 1.15:
        cv_verdict = "NEAR-GEOMETRIC     — close to memoryless, minimal timing edge"
    else:
        cv_verdict = "OVER-DISPERSED     — crashes cluster; long safe windows, sudden bunches"
    print(f"  │    Verdict         : {cv_verdict}")

    # ── Interval histogram ────────────────────────────────────────────────────
    bkt = max(10, rate // 5)
    hist: dict[int, int] = defaultdict(int)
    for iv in intervals:
        hist[(iv // bkt) * bkt] += 1
    max_h = max(hist.values())

    print(f"  │")
    print(f"  │  Interval histogram (bucket = {bkt} ticks):")
    for b in sorted(hist):
        bar = "█" * int(24 * hist[b] / max_h)
        print(f"  │    {b:>5}–{b+bkt-1:<5} : {hist[b]:>4}  {bar}")

    # ── Conditional crash probability ─────────────────────────────────────────
    spike_set = set(spikes)
    bkt_cp    = max(10, rate // 6)

    tss  = 0          # ticks since last spike
    cond: dict[int, list] = defaultdict(list)  # bucket → [0/1]

    for i in range(1, n):
        b = (tss // bkt_cp) * bkt_cp
        cond[b].append(1 if i in spike_set else 0)
        if i in spike_set:
            tss = 0
        else:
            tss += 1

    base_pct = 1.0 / rate * 100
    print(f"  │")
    print(f"  │  Conditional crash probability  (memoryless baseline = {base_pct:.3f}%/tick):")
    print(f"  │  {'T since last':>12}  {'ticks':>6}  {'crashes':>7}  {'P%/tick':>8}  {'vs base':>8}  Signal")

    probs_by_bucket = []
    for b in sorted(cond):
        vals   = cond[b]
        total  = len(vals)
        n_spk  = sum(vals)
        if total < 30:
            continue
        p      = n_spk / total * 100
        ratio  = p / base_pct if base_pct > 0 else 1.0
        probs_by_bucket.append((b, p, ratio))
        if ratio < 0.40:
            sig = "SUPPRESSED  ← safe window to enter ACCU"
        elif ratio < 0.75:
            sig = "Below avg"
        elif ratio < 1.35:
            sig = "Normal"
        elif ratio < 2.5:
            sig = "Elevated  ← caution"
        else:
            sig = "HIGH RISK ← avoid / sell ACCU"
        print(f"  │  {b:>6}–{b+bkt_cp-1:<5}  {total:>6}  {n_spk:>7}  {p:>8.3f}%  {ratio:>7.2f}x  {sig}")

    if len(probs_by_bucket) >= 3:
        early = statistics.mean(p for _, p, _ in probs_by_bucket[:2])
        late  = statistics.mean(p for _, p, _ in probs_by_bucket[-2:])
        tr    = late / early if early > 0 else 1.0
        if tr > 2.5:
            due = f"STRONG DUE EFFECT (×{tr:.1f}) — crash probability rises sharply → TIMING EDGE"
        elif tr > 1.4:
            due = f"MILD DUE EFFECT (×{tr:.1f}) — some predictability at long intervals"
        elif tr < 0.5:
            due = f"GRACE-PERIOD EFFECT (×{tr:.1f}) — probability suppressed early after spike"
        else:
            due = f"NO TIMING EDGE (×{tr:.1f}) — probability stable across all intervals"
        print(f"  │")
        print(f"  │  Due-effect verdict: {due}")

    print(f"  └─")


def post_spike_trajectory(prices: list, spikes: list, direction: str,
                          lookforward: int = 60, min_gap: int = 20) -> None:
    """
    How does price move after each spike?

    Normalised recovery = (price[s+k] - price[s]) / |spike_size| × 100
      0%   = price still at crash/boom level
      100% = fully returned to pre-spike level
      >100% = overshot in recoil direction
    """
    n = len(prices)

    # Drop clustered spikes and those too close to the end
    clean = []
    for i, s in enumerate(spikes):
        if i > 0 and (s - spikes[i - 1]) < min_gap:
            continue
        if s + lookforward >= n:
            continue
        clean.append(s)

    print(f"\n  ┌─ POST-SPIKE TRAJECTORY  [{len(clean)} isolated spikes, lookforward={lookforward}t]")

    if len(clean) < 10:
        print(f"  │  Too few isolated spikes for trajectory analysis.")
        print(f"  └─")
        return

    print(f"  │  Normalised: 0% = at crash price | 100% = fully recovered | >100% = overshot")
    print(f"  │")
    print(f"  │  {'Tick':>5}  {'Median':>8}  {'Mean':>8}  {'% >0%':>8}  {'% >50%':>9}  {'% >100%':>9}  Interpretation")

    half_rec_tick = full_rec_tick = None

    for k in [1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 60]:
        if k > lookforward:
            break

        recoveries = []
        for s in clean:
            spike_size = abs(prices[s] - prices[s - 1])
            if spike_size == 0:
                continue
            if direction == "crash":
                rec = (prices[s + k] - prices[s]) / spike_size * 100
            else:  # boom
                rec = (prices[s] - prices[s + k]) / spike_size * 100
            recoveries.append(rec)

        if not recoveries:
            continue

        med  = statistics.median(recoveries)
        mean = statistics.mean(recoveries)
        p0   = sum(1 for r in recoveries if r > 0)   / len(recoveries) * 100
        p50  = sum(1 for r in recoveries if r > 50)  / len(recoveries) * 100
        p100 = sum(1 for r in recoveries if r > 100) / len(recoveries) * 100

        if half_rec_tick is None and med >= 50:
            half_rec_tick = k
        if full_rec_tick is None and med >= 100:
            full_rec_tick = k

        bar_n = max(0, min(20, int(med / 5)))
        bar   = "█" * bar_n

        if p100 >= 70:
            interp = "Strong full recoil"
        elif p50 >= 70:
            interp = "Consistent half-recoil"
        elif p0 >= 60:
            interp = "Directionally correct"
        elif med < 0:
            interp = "Continues in spike direction"
        else:
            interp = "Mixed"

        print(f"  │  {k:>5}  {med:>+8.1f}%  {mean:>+8.1f}%  {p0:>8.1f}%  {p50:>9.1f}%  {p100:>9.1f}%  {interp}  {bar}")

    print(f"  │")
    if half_rec_tick:
        print(f"  │  Median 50% recovery  → tick +{half_rec_tick}")
    if full_rec_tick:
        print(f"  │  Median full recovery → tick +{full_rec_tick}  ← optimal ACCU hold_ticks")
    else:
        print(f"  │  Median full recovery → not within +{lookforward} ticks (extend hold_ticks)")
    print(f"  └─")


def inter_spike_drift(prices: list, spikes: list, direction: str) -> None:
    """
    Is the price movement between spikes directional and consistent enough to trade?
    CRASH indices should drift upward between crashes; BOOM indices downward.
    """
    if len(spikes) < 5:
        return

    print(f"\n  ┌─ INTER-SPIKE DRIFT")

    segs = []
    for i in range(len(spikes) - 1):
        s0, s1 = spikes[i], spikes[i + 1] - 1
        if s1 <= s0 + 1:
            continue
        length      = s1 - s0
        start_price = prices[s0]
        end_price   = prices[s1]
        change_pct  = (end_price - start_price) / start_price * 100 if start_price else 0
        drift_pt    = (end_price - start_price) / length
        segs.append({"len": length, "chg": change_pct, "dpt": drift_pt})

    if not segs:
        print(f"  │  No segments found.")
        print(f"  └─")
        return

    avg_len  = statistics.mean(s["len"] for s in segs)
    avg_chg  = statistics.mean(s["chg"] for s in segs)
    avg_dpt  = statistics.mean(s["dpt"] for s in segs)
    med_chg  = statistics.median(s["chg"] for s in segs)

    if direction == "crash":
        pct_correct = sum(1 for s in segs if s["chg"] > 0) / len(segs) * 100
        dir_label   = "upward (CRASH expects upward inter-spike drift)"
    else:
        pct_correct = sum(1 for s in segs if s["chg"] < 0) / len(segs) * 100
        dir_label   = "downward (BOOM expects downward inter-spike drift)"

    print(f"  │  Segments: {len(segs)}  |  Avg length: {avg_len:.0f} ticks")
    print(f"  │  Avg price change  : {avg_chg:+.4f}%  (median {med_chg:+.4f}%)")
    print(f"  │  Avg drift/tick    : {avg_dpt:+.8f}")
    print(f"  │  % {dir_label}: {pct_correct:.1f}%")

    if pct_correct >= 75 and abs(avg_chg) > 0.05:
        verdict = "STRONG DIRECTIONAL DRIFT — inter-spike trend is mechanically reliable"
    elif pct_correct >= 60:
        verdict = "MODERATE DRIFT — present but noisy; weaker signal"
    else:
        verdict = "WEAK / NO DRIFT — inter-spike movement is near-random"
    print(f"  │  Verdict: {verdict}")
    print(f"  └─")


# ─── Volatility analyses ──────────────────────────────────────────────────────

def digit_distribution(prices: list) -> None:
    """
    Full digit distribution 0–9 and win rates for multiple DIGITOVER/UNDER contracts.
    """
    dp     = infer_decimal_places(prices)
    counts = Counter(last_digit(p, dp) for p in prices)
    total  = sum(counts.values())

    print(f"\n  ┌─ DIGIT DISTRIBUTION  [{total:,} ticks  |  detected precision: {dp} dp]")
    print(f"  │  (DIGITOVER/UNDER payout 95%  BE = 51.28%)")
    print(f"  │")
    print(f"  │  {'Digit':>6}  {'Count':>8}  {'Actual%':>8}  {'Dev from 10%':>13}  Bar")

    for d in range(10):
        cnt = counts[d]
        pct = cnt / total * 100
        dev = pct - 10.0
        bar = "█" * int(pct * 1.5)
        sign = "+" if dev >= 0 else ""
        print(f"  │  {d:>6}  {cnt:>8}  {pct:>8.3f}%  {sign}{dev:>+12.3f}%  {bar}")

    print(f"  │")
    print(f"  │  Contract win rates:")
    be = 51.28

    for label, digits in [
        ("DIGITOVER(4)  last > 4",    list(range(5, 10))),
        ("DIGITOVER(5)  last > 5",    list(range(6, 10))),
        ("DIGITOVER(6)  last > 6",    list(range(7, 10))),
        ("DIGITUNDER(5) last < 5",    list(range(0, 5))),
        ("DIGITUNDER(4) last < 4",    list(range(0, 4))),
        ("DIGITUNDER(3) last < 3",    list(range(0, 3))),
        ("DIGITEVEN     last is even", [0, 2, 4, 6, 8]),
        ("DIGITODD      last is odd",  [1, 3, 5, 7, 9]),
    ]:
        wr  = sum(counts[d] for d in digits) / total * 100
        ev  = wr - be
        tag = f"  ✓ EV={ev:+.3f}%" if wr > be else f"  ✗ EV={ev:+.3f}%"
        print(f"  │    {label:<28}: {wr:>7.3f}%{tag}")

    print(f"  └─")


def autocorrelation(prices: list, lags: list = None) -> None:
    """
    Tick return autocorrelation (momentum vs mean-reversion) and volatility clustering.
    """
    if lags is None:
        lags = [1, 2, 3, 5, 10, 20, 50, 100]

    rets  = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    arets = [abs(r) for r in rets]
    n     = len(rets)

    print(f"\n  ┌─ AUTOCORRELATION  [{n:,} returns]")

    # ── Return autocorrelation ────────────────────────────────────────────────
    print(f"  │")
    print(f"  │  Return r[t] autocorrelation (positive = momentum, negative = mean-reversion):")
    print(f"  │  {'Lag':>5}  {'Corr':>8}  Interpretation")
    for lag in lags:
        if lag >= n:
            break
        c = pearson(rets[lag:], rets[:-lag])
        if c > 0.05:
            interp = "Momentum"
        elif c < -0.05:
            interp = "Mean-reverting"
        else:
            interp = "Noise / random"
        sign = "+" if c >= 0 else ""
        bar  = ("+" if c >= 0 else "-") + "█" * int(abs(c) * 40)
        print(f"  │  {lag:>5}  {sign}{c:>+7.4f}  {interp}  {bar}")

    # ── Volatility clustering ─────────────────────────────────────────────────
    print(f"  │")
    print(f"  │  Volatility clustering |r[t]| autocorrelation (positive = calm/volatile regimes persist):")
    print(f"  │  {'Lag':>5}  {'Corr':>8}  Interpretation")
    for lag in lags[:5]:
        if lag >= n:
            break
        c = pearson(arets[lag:], arets[:-lag])
        if c > 0.10:
            interp = "Volatility clusters — regime-aware strategies viable"
        elif c > 0.03:
            interp = "Mild clustering"
        else:
            interp = "No persistence"
        sign = "+" if c >= 0 else ""
        print(f"  │  {lag:>5}  {sign}{c:>+7.4f}  {interp}")

    # ── Tick size distribution ────────────────────────────────────────────────
    srt = sorted(arets)
    p50  = srt[int(0.50 * len(srt))]
    p90  = srt[int(0.90 * len(srt))]
    p95  = srt[int(0.95 * len(srt))]
    p99  = srt[int(0.99 * len(srt))]
    p999 = srt[int(0.999 * len(srt))]
    fat_ratio = p99 / p50 if p50 > 0 else 0

    print(f"  │")
    print(f"  │  Tick size distribution:")
    print(f"  │    Mean: {statistics.mean(arets):.6f}   Std: {statistics.stdev(arets):.6f}")
    print(f"  │    P50: {p50:.6f}  P90: {p90:.6f}  P95: {p95:.6f}")
    print(f"  │    P99: {p99:.6f}  P99.9: {p999:.6f}  P99/P50: {fat_ratio:.1f}x")
    if fat_ratio > 10:
        print(f"  │    → Very fat tails — large moves are common; ACCU survival is hard")
    elif fat_ratio > 4:
        print(f"  │    → Fat tails — non-Gaussian; larger outliers than expected")
    else:
        print(f"  │    → Near-normal tail behaviour")
    print(f"  └─")


# ─── JD analyses ─────────────────────────────────────────────────────────────

def jd_analysis(prices: list, spikes: list) -> None:
    """
    JD-specific: spike direction balance + post-spike recoil for each direction separately.
    """
    n     = len(prices)
    moves = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

    print(f"\n  ┌─ JD SPIKE ANALYSIS  [{len(spikes)} spikes]")

    if len(spikes) < 10:
        print(f"  │  Too few spikes for analysis.")
        print(f"  └─")
        return

    up_spikes   = [s for s in spikes if moves[s - 1] > 0]
    down_spikes = [s for s in spikes if moves[s - 1] < 0]

    print(f"  │")
    print(f"  │  Direction balance:")
    print(f"  │    Up spikes   : {len(up_spikes):>4}  ({len(up_spikes)/len(spikes)*100:.1f}%)")
    print(f"  │    Down spikes : {len(down_spikes):>4}  ({len(down_spikes)/len(spikes)*100:.1f}%)")

    exp = len(spikes) / 2
    chi2 = ((len(up_spikes) - exp) ** 2 + (len(down_spikes) - exp) ** 2) / exp
    if chi2 > 3.84:
        bias_dir = "UP" if len(up_spikes) > len(down_spikes) else "DOWN"
        print(f"  │    χ²={chi2:.2f} > 3.84 → SIGNIFICANT {bias_dir} BIAS (p<0.05)")
    else:
        print(f"  │    χ²={chi2:.2f} ≤ 3.84 → no significant directional bias")

    lookforward = 20
    min_gap     = 10

    for label, subset, rec_dir in [
        ("DOWN-spike → expect BUY_RISE recoil", down_spikes, "up"),
        ("UP-spike   → expect BUY_FALL recoil", up_spikes,   "down"),
    ]:
        usable = [s for i, s in enumerate(subset)
                  if (i == 0 or s - subset[i - 1] >= min_gap) and s + lookforward < n]
        print(f"  │")
        print(f"  │  {label}  ({len(usable)} usable spikes)")
        if len(usable) < 5:
            print(f"  │    (insufficient)")
            continue
        print(f"  │  {'Tick':>5}  {'Median%':>8}  {'% correct dir':>14}  Interpretation")
        for k in [1, 2, 3, 5, 8, 10, 15, 20]:
            recoils = []
            for s in usable:
                spike_move = moves[s - 1]
                future     = prices[s + k] - prices[s]
                # Normalised recoil: positive = moving in expected recoil direction
                if rec_dir == "up":
                    recoils.append(future / abs(spike_move) * 100)
                else:
                    recoils.append(-future / abs(spike_move) * 100)
            med   = statistics.median(recoils)
            p_pos = sum(1 for r in recoils if r > 0) / len(recoils) * 100
            if p_pos > 65:
                interp = "Strong recoil signal"
            elif p_pos > 55:
                interp = "Mild recoil signal"
            else:
                interp = "Weak / unreliable"
            print(f"  │  {k:>5}  {med:>+8.1f}%  {p_pos:>14.1f}%  {interp}")

    print(f"  └─")


# ─── Main dispatcher ──────────────────────────────────────────────────────────

def run_symbol(symbol: str) -> None:
    meta = SYMBOL_META.get(symbol.upper())
    if not meta:
        print(f"\n  Unknown symbol: {symbol}")
        return

    print()
    print(SEP)
    print(f"  {symbol.upper()}  |  family={meta['family']}  direction={meta['direction']}"
          + (f"  rate=1/{meta['rate']}" if "rate" in meta else ""))
    print(SEP)

    raw    = fetch_ticks(symbol.upper(), count=TICK_COUNT + 2000)
    prices = [float(t["quote"]) for t in raw][-TICK_COUNT:]
    print(f"  Using {len(prices):,} ticks  |  price range: {min(prices):.4f} – {max(prices):.4f}")

    family    = meta["family"]
    direction = meta["direction"]

    if family == "crash_boom":
        spikes = detect_spikes(prices, direction, meta["rate"])
        spike_timing(prices, spikes, meta)
        post_spike_trajectory(prices, spikes, direction)
        inter_spike_drift(prices, spikes, direction)

    elif family == "volatility":
        digit_distribution(prices)
        autocorrelation(prices)

    elif family == "jd":
        # JD: use top 1% as spikes (both directions)
        spikes = detect_spikes(prices, "jd", rate=100)
        jd_analysis(prices, spikes)
        digit_distribution(prices)
        autocorrelation(prices)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deriv index algorithmic property analysis")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--symbol", help="Symbol name or ALL")
    group.add_argument("--family", choices=["crash", "boom", "vol", "jd"])
    args = parser.parse_args()

    if args.symbol:
        symbols = list(SYMBOL_META.keys()) if args.symbol.upper() == "ALL" else [args.symbol.upper()]
    else:
        symbols = FAMILIES[args.family]

    for sym in symbols:
        run_symbol(sym)

    print()
    print(SEP)
    print("  LEGEND")
    print("  CV < 0.7   → spikes are approximately periodic → timing edge exists")
    print("  Due effect → P(crash) rises with T since last → can predict crash windows")
    print("  Suppressed → P(crash) very low right after spike → safe ACCU entry window")
    print("  Full recovery tick → optimal hold_ticks for ACCU / recoil binary")
    print("  Digit EV   → how far above/below 51.28% BE the true win rate sits")
    print("  Autocorr   → +ve = momentum, −ve = mean-reversion, ≈0 = random")
    print()


if __name__ == "__main__":
    main()
