"""
Statistical analysis of Deriv synthetic index tick data.

Tests whether the series has any exploitable structure, or is a pure random walk.
Results tell us which strategy direction (if any) can beat the house edge.

Tests:
  1. Hurst exponent (R/S analysis)
       H < 0.5  -> mean-reverting  -> RSI reversal-type strategies have a basis
       H = 0.5  -> pure random walk -> no price-action strategy can win
       H > 0.5  -> trending        -> momentum strategies have a basis

  2. Lag-1..20 autocorrelation of tick directions
       Negative ACF at lag 1 -> mean-reversion signal
       Positive ACF at lag 1 -> momentum signal
       All near zero          -> no exploitable serial structure

  3. Runs test (Wald-Wolfowitz)
       Significantly fewer runs than expected  -> trending (positive autocorr)
       Significantly more runs than expected   -> mean-reverting (negative autocorr)
       Z near 0                                -> random

  4. Ljung-Box Q statistic
       Q > critical value -> serial correlation exists somewhere in lags 1-20
       Q < critical value -> no serial correlation detected

  5. Autocorrelation of SQUARED returns (volatility clustering)
       High ACF of |returns| -> volatility clusters -> trade during calm periods

  6. House-edge breakeven calculation
       Shows what autocorrelation magnitude is needed to overcome the payout gap.

Usage:
  python stat_analysis.py
  python stat_analysis.py --symbol R_50
  python stat_analysis.py --no-fresh    # reuse cached tick file
"""

import argparse
import math
import os

import yaml
from dotenv import load_dotenv
from loguru import logger
logger.remove()

from src.data.history import fetch_ticks

load_dotenv()

# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--symbol",   default=None)
parser.add_argument("--no-fresh", action="store_true", help="Reuse cached ticks")
args = parser.parse_args()

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

PAYOUT = cfg["risk"].get("payout_pct", 0.9232)
BE     = 1 / (1 + PAYOUT)          # breakeven win probability

if args.symbol:
    SYMBOL = args.symbol.upper()
else:
    enabled = [s for s in cfg.get("symbols", []) if s.get("enabled", True)]
    SYMBOL  = enabled[0]["symbol"] if enabled else "R_25"

# ── Fetch ticks ────────────────────────────────────────────────────────────────
print(f"Fetching ticks for {SYMBOL}...")
ticks = fetch_ticks(
    SYMBOL,
    count=150_000,
    fresh=not args.no_fresh,
    app_id=os.getenv("DERIV_APP_ID", "1089"),
    api_token=os.getenv("DERIV_API_TOKEN"),
)
prices = [float(t["quote"]) for t in ticks]
N      = len(prices)
print(f"  Got {N:,} ticks\n")

# ── Derived series ─────────────────────────────────────────────────────────────
returns    = [prices[i] - prices[i-1] for i in range(1, N)]   # raw tick differences
directions = [1 if r > 0 else (-1 if r < 0 else 0) for r in returns]
dirs_clean = [d for d in directions if d != 0]                # exclude flat ticks
abs_rets   = [abs(r) for r in returns]

n_up   = sum(1 for d in dirs_clean if d > 0)
n_down = sum(1 for d in dirs_clean if d < 0)
n_flat = sum(1 for d in directions  if d == 0)

# ── Statistical helpers ────────────────────────────────────────────────────────

def mean(xs):
    return sum(xs) / len(xs)

def variance(xs):
    m = mean(xs)
    return sum((x - m)**2 for x in xs) / len(xs)

def std(xs):
    return math.sqrt(variance(xs))

def autocorr(series, lag):
    """Pearson autocorrelation at a given lag."""
    n   = len(series)
    m   = mean(series)
    var = variance(series)
    if var == 0 or n <= lag:
        return 0.0
    cov = sum((series[i] - m) * (series[i - lag] - m) for i in range(lag, n)) / n
    return cov / var

def hurst_rs(prices, n_scales=20):
    """
    Hurst exponent via R/S (rescaled range) analysis.
    Uses logarithmically-spaced chunk sizes from 8 to len/8.
    Returns (H, chunk_sizes, rs_means).
    """
    n       = len(prices)
    min_sz  = 8
    max_sz  = n // 8
    # build ~n_scales log-spaced sizes
    sizes, rs_means = [], []
    ratio = (max_sz / min_sz) ** (1 / (n_scales - 1))
    sz    = float(min_sz)
    seen  = set()
    for _ in range(n_scales):
        s = max(min_sz, int(round(sz)))
        if s not in seen and s <= max_sz:
            seen.add(s)
            sizes.append(s)
        sz *= ratio

    for s in sizes:
        rs_list = []
        for start in range(0, n - s + 1, s):
            chunk = prices[start:start + s]
            m     = mean(chunk)
            devs  = [p - m for p in chunk]
            cumul = []
            acc   = 0.0
            for d in devs:
                acc += d
                cumul.append(acc)
            R = max(cumul) - min(cumul)
            S = std(chunk)
            if S > 0:
                rs_list.append(R / S)
        if rs_list:
            rs_means.append(mean(rs_list))
        else:
            rs_means.append(None)

    # Filter out None values
    valid = [(s, rs) for s, rs in zip(sizes, rs_means) if rs is not None]
    if len(valid) < 3:
        return 0.5, [], []
    sizes_v, rs_v = zip(*valid)

    # OLS on log-log
    lx  = [math.log(s) for s in sizes_v]
    ly  = [math.log(r) for r in rs_v]
    mx  = mean(lx); my = mean(ly)
    num = sum((lx[i] - mx) * (ly[i] - my) for i in range(len(lx)))
    den = sum((lx[i] - mx)**2               for i in range(len(lx)))
    H   = num / den if den else 0.5
    return H, list(sizes_v), list(rs_v)

def runs_test(dirs):
    """Wald-Wolfowitz runs test. Returns (runs, expected, z_score)."""
    n    = len(dirs)
    n_u  = sum(1 for d in dirs if d > 0)
    n_d  = n - n_u
    if n_u == 0 or n_d == 0:
        return 0, 0, 0.0
    runs = 1 + sum(1 for i in range(1, n) if dirs[i] != dirs[i-1])
    exp  = 2 * n_u * n_d / n + 1
    var  = 2 * n_u * n_d * (2*n_u*n_d - n) / (n**2 * (n-1))
    z    = (runs - exp) / math.sqrt(var)
    return runs, exp, z

def ljung_box(series, max_lag=20):
    """Ljung-Box Q statistic. Q > chi2_critical(max_lag, 0.05) -> serial correlation."""
    n    = len(series)
    acfs = [autocorr(series, k) for k in range(1, max_lag + 1)]
    Q    = n * (n + 2) * sum(acfs[k]**2 / (n - k - 1) for k in range(max_lag))
    return Q, acfs

def bar(val, lo=-0.25, hi=0.25, width=40):
    """ASCII bar centered at zero, for ACF plot."""
    center = width // 2
    norm   = (val - lo) / (hi - lo)
    pos    = int(round(norm * width))
    pos    = max(0, min(width - 1, pos))
    b      = ["."] * width
    b[center] = "|"
    if pos != center:
        lo_p = min(pos, center)
        hi_p = max(pos, center)
        for i in range(lo_p, hi_p + 1):
            b[i] = "#"
    return "".join(b)

def interpret_z(z):
    if abs(z) > 2.576:
        return "p<0.01 (highly significant)"
    elif abs(z) > 1.960:
        return "p<0.05 (significant)"
    elif abs(z) > 1.282:
        return "p<0.20 (marginal)"
    else:
        return "p>0.20 (not significant)"

# ── Run tests ─────────────────────────────────────────────────────────────────
W = 62
print("=" * W)
print(f"  STATISTICAL ANALYSIS  |  {SYMBOL}  |  {N:,} ticks")
print("=" * W)

# ── 1. Basic tick stats ────────────────────────────────────────────────────────
print()
print("TICK DIRECTION BREAKDOWN")
print(f"  Up ticks  : {n_up:>7,}  ({n_up/len(dirs_clean)*100:.1f}%)")
print(f"  Down ticks: {n_down:>7,}  ({n_down/len(dirs_clean)*100:.1f}%)")
print(f"  Flat ticks: {n_flat:>7,}  ({n_flat/len(directions)*100:.1f}%)")
print(f"  Mean return    : {mean(returns):+.6f}")
print(f"  Std of returns : {std(returns):.6f}")

skew_num  = mean([(r - mean(returns))**3 for r in returns])
skew_den  = std(returns)**3
skewness  = skew_num / skew_den if skew_den else 0
kurt_num  = mean([(r - mean(returns))**4 for r in returns])
kurt_den  = std(returns)**4
kurtosis  = (kurt_num / kurt_den) - 3 if kurt_den else 0
print(f"  Skewness       : {skewness:+.4f}  (0 = symmetric)")
print(f"  Excess kurtosis: {kurtosis:+.4f}  (0 = normal tails, >0 = fat tails)")

# ── 2. Hurst exponent ─────────────────────────────────────────────────────────
print()
print("HURST EXPONENT  (R/S analysis)")
H, sizes, rs_vals = hurst_rs(prices)
if H < 0.45:
    h_label = "MEAN-REVERTING  -> reversal strategies have a structural basis"
elif H > 0.55:
    h_label = "TRENDING        -> momentum strategies have a structural basis"
else:
    h_label = "RANDOM WALK     -> no price-action strategy has a structural basis"
print(f"  H = {H:.4f}  ->  {h_label}")
print(f"  (H = 0.5 is pure random walk; 95% range for RW at {N:,} ticks: ~[0.49, 0.51])")

# ── 3. Runs test ──────────────────────────────────────────────────────────────
print()
print("RUNS TEST  (Wald-Wolfowitz)")
runs, exp_runs, z_runs = runs_test(dirs_clean)
print(f"  Observed runs : {runs:,}")
print(f"  Expected runs : {exp_runs:,.1f}  (under pure randomness)")
diff_pct = (runs - exp_runs) / exp_runs * 100
direction = "more" if runs > exp_runs else "fewer"
print(f"  Difference    : {abs(diff_pct):.1f}% {direction} runs than expected")
print(f"  Z-score       : {z_runs:+.3f}  ({interpret_z(z_runs)})")
if z_runs > 1.96:
    runs_interp = "FEWER runs -> ticks TREND (up follows up, down follows down)"
elif z_runs < -1.96:
    runs_interp = "MORE runs -> ticks ALTERNATE (up follows down, mean-reverting)"
else:
    runs_interp = "runs count consistent with pure randomness"
print(f"  Interpretation: {runs_interp}")

# ── 4. Autocorrelation of directions ─────────────────────────────────────────
print()
print("AUTOCORRELATION OF TICK DIRECTIONS  (lags 1-20)")
sig_thresh = 1.96 / math.sqrt(len(dirs_clean))
print(f"  Significance threshold: ±{sig_thresh:.4f}  (95% CI for N={len(dirs_clean):,})")
print(f"  Minimum autocorr to break even vs {PAYOUT*100:.2f}% payout: "
      f"±{2*(BE - 0.5):.4f}")
print()
print(f"  {'Lag':>3}  {'ACF':>8}  {'Significant?':12}  Chart (center=0, range ±0.25)")
print(f"  {'-'*3}  {'-'*8}  {'-'*12}  {'-'*40}")
acfs_dir = []
for lag in range(1, 21):
    ac = autocorr(dirs_clean, lag)
    acfs_dir.append(ac)
    sig  = "YES ***" if abs(ac) > sig_thresh else "no"
    b    = bar(ac)
    print(f"  {lag:>3}  {ac:>+8.5f}  {sig:<12}  {b}")

# ── 5. Ljung-Box test ─────────────────────────────────────────────────────────
print()
print("LJUNG-BOX TEST  (joint test for serial correlation, lags 1-20)")
Q_dir, _ = ljung_box(dirs_clean, max_lag=20)
# Chi-squared 5% critical: df=20 -> 31.41
chi2_crit = 31.41
print(f"  Q statistic : {Q_dir:.2f}")
print(f"  Critical val: {chi2_crit:.2f}  (chi-squared, df=20, p=0.05)")
if Q_dir > chi2_crit:
    print(f"  Result      : Q > critical -> SERIAL CORRELATION DETECTED in directions")
else:
    print(f"  Result      : Q < critical -> no significant serial correlation")

# ── 6. Autocorrelation of absolute returns (volatility clustering) ───────────
print()
print("VOLATILITY CLUSTERING  (autocorrelation of |returns|, lags 1-10)")
print(f"  If high: volatility clusters -> trade during calm periods for better odds")
print()
print(f"  {'Lag':>3}  {'ACF(|ret|)':>10}  {'Significant?':12}")
print(f"  {'-'*3}  {'-'*10}  {'-'*12}")
sig_thresh_r = 1.96 / math.sqrt(len(abs_rets))
for lag in range(1, 11):
    ac  = autocorr(abs_rets, lag)
    sig = "YES ***" if abs(ac) > sig_thresh_r else "no"
    print(f"  {lag:>3}  {ac:>+10.5f}  {sig}")

# ── 7. House edge breakeven ────────────────────────────────────────────────────
print()
print("HOUSE EDGE CONTEXT")
print(f"  Payout          : {PAYOUT*100:.2f}%")
print(f"  Breakeven WR    : {BE*100:.2f}%")
house_edge_pct = (0.5 - BE) * 100
print(f"  Edge against you: {house_edge_pct:.2f}% per trade on a 50/50 signal")
print()
print("  What autocorrelation (lag-1) is needed to overcome the house edge?")
needed_ac = 2 * (BE - 0.5)
print(f"  Required lag-1 ACF: >= {needed_ac:.4f}  (in the exploitable direction)")
actual_ac1 = acfs_dir[0] if acfs_dir else 0
if abs(actual_ac1) >= needed_ac:
    print(f"  Observed lag-1 ACF: {actual_ac1:+.5f}  -> SUFFICIENT to overcome edge (if signal captures it)")
else:
    print(f"  Observed lag-1 ACF: {actual_ac1:+.5f}  -> INSUFFICIENT to overcome edge alone")
    print(f"  Gap: {needed_ac - abs(actual_ac1):.5f} more autocorrelation needed")

# ── 8. Summary verdict ────────────────────────────────────────────────────────
print()
print("=" * W)
print("  VERDICT")
print("=" * W)
verdicts = []
if H < 0.45:
    verdicts.append(f"  Hurst H={H:.3f}: mean-reverting structure present")
elif H > 0.55:
    verdicts.append(f"  Hurst H={H:.3f}: trending structure present")
else:
    verdicts.append(f"  Hurst H={H:.3f}: consistent with random walk")

if abs(z_runs) > 1.96:
    if z_runs < 0:
        verdicts.append(f"  Runs test z={z_runs:.2f}: significant alternation (mean-reverting ticks)")
    else:
        verdicts.append(f"  Runs test z={z_runs:.2f}: significant trending (momentum ticks)")
else:
    verdicts.append(f"  Runs test z={z_runs:.2f}: consistent with random tick sequence")

if Q_dir > chi2_crit:
    verdicts.append(f"  Ljung-Box Q={Q_dir:.1f}: serial correlation detected across lags 1-20")
else:
    verdicts.append(f"  Ljung-Box Q={Q_dir:.1f}: no significant serial correlation")

if abs(actual_ac1) >= needed_ac:
    verdicts.append(f"  Lag-1 ACF={actual_ac1:+.5f}: large enough to theoretically beat house edge")
else:
    verdicts.append(f"  Lag-1 ACF={actual_ac1:+.5f}: too small to beat house edge alone")

for v in verdicts:
    print(v)

print()
exploitable = (abs(actual_ac1) >= needed_ac) or (Q_dir > chi2_crit and abs(acfs_dir[0]) > sig_thresh)
if exploitable:
    print("  CONCLUSION: Some statistical structure detected.")
    if acfs_dir[0] < -needed_ac:
        print("  Direction: mean-reverting -> RSI/streak reversal strategies have a basis.")
        print("  Recommended next step: tune RSI parameters specifically to the detected")
        print("  autocorrelation lag and test with tighter thresholds.")
    elif acfs_dir[0] > needed_ac:
        print("  Direction: trending -> momentum strategies have a basis.")
        print("  Recommended next step: build and test RSI momentum (trade WITH the extreme)")
    else:
        print("  Structure is present but lag-1 alone may not be enough.")
        print("  Look at which lags showed significance and build a multi-lag signal.")
else:
    print("  CONCLUSION: No exploitable structure found in this dataset.")
    print("  The series behaves as a random walk. No price-action strategy can")
    print("  consistently overcome the 7.68% house edge on this instrument.")
    print("  Recommended: stop trading R_25 Rise/Fall and consider either:")
    print("    1. A different Deriv product type (accumulators, multipliers)")
    print("    2. A real (non-synthetic) market where genuine edges exist")

print()
print("=" * W)
