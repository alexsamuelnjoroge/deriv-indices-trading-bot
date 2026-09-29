"""
Master synthetic index edge scanner.

Subscribes to every Deriv synthetic index in parallel, collects tick data,
then runs a full battery of statistical tests to rank exploitable edges.

Tests per symbol:
  1. Digit distribution — chi-squared uniformity; per-digit frequency
  2. Digit autocorrelation — lag 1-5 (detects JD100-style serial correlation)
  3. DIGITOVER base rate per barrier — estimates Deriv payout vs actual probability
  4. Post-spike direction — continuation vs recoil WR grid (for Crash/Boom/Jump)
  5. Price-move autocorrelation — direction persistence (for binary CALL/PUT)
  6. ATR volatility distribution — low-percentile calm periods (for ACCU timing)

Usage:
  source venv/bin/activate
  python3 analyze_all_synthetic.py            # 900s (15 min)
  python3 analyze_all_synthetic.py 1800       # 30 min — more reliable stats
"""

import asyncio
import math
import os
import sys
from collections import defaultdict
from dotenv import load_dotenv
from src.api.client import DerivClient

DURATION = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 900

# All synthetic indices to probe — skips any that fail to subscribe
SYMBOLS = [
    # Volatility (1s tick)
    "R_10", "R_25", "R_50", "R_75", "R_100",
    # Volatility (1Hz — faster tick)
    "R_10_1Hz", "R_25_1Hz", "R_50_1Hz", "R_75_1Hz", "R_100_1Hz",
    # Crash
    "CRASH300N", "CRASH500", "CRASH1000",
    # Boom
    "BOOM300N", "BOOM500", "BOOM1000",
    # Jump
    "JD10", "JD25", "JD50", "JD75", "JD100",
    # Step
    "STPIDX",
    # Range Break
    "RNGBR100", "RNGBR200",
]

ATR_PERIOD  = 50
SPIKE_MULT  = 2.5
DUR_RANGE   = [1, 2, 3, 5, 10]
SETTLE_RANGE = [0, 1, 2]


# ── helpers ────────────────────────────────────────────────────────────────

def auto_pip(prices: list) -> int:
    """Detect decimal places from the first 20 prices."""
    for p in prices[:20]:
        s = f"{p:.10f}".rstrip("0")
        if "." in s:
            return len(s.split(".")[1])
    return 2


def last_digit(price: float, pip: int) -> int:
    return int(f"{price:.{pip}f}"[-1])


def compute_atr(prices: list, period: int) -> list:
    if len(prices) < 2:
        return [0.0]
    trs = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
    avg = sum(trs[:period]) / period if len(trs) >= period else sum(trs) / max(len(trs), 1)
    atrs = []
    for i, tr in enumerate(trs):
        if i < period:
            atrs.append(avg)
        else:
            avg = (avg * (period - 1) + tr) / period
            atrs.append(avg)
    return atrs


def autocorr(series: list, lag: int) -> float:
    n = len(series)
    if n <= lag:
        return 0.0
    mean = sum(series) / n
    var = sum((x - mean) ** 2 for x in series) / n
    if var == 0:
        return 0.0
    cov = sum((series[i] - mean) * (series[i + lag] - mean) for i in range(n - lag)) / (n - lag)
    return cov / var


def sig_threshold(n: int) -> float:
    return 2.0 / math.sqrt(n)


# ── per-symbol analysis ─────────────────────────────────────────────────────

def analyze(symbol: str, prices: list) -> dict:
    """Run all tests, return a findings dict."""
    n = len(prices)
    findings = {"symbol": symbol, "n": n, "edges": []}

    if n < 80:
        findings["skip"] = f"only {n} ticks"
        return findings

    pip = auto_pip(prices)
    findings["pip"] = pip

    # ── 1 & 2 & 3: digit tests ─────────────────────────────────────────────
    digits = [last_digit(p, pip) for p in prices]
    counts = [digits.count(d) for d in range(10)]
    expected = n / 10
    chi2 = sum((c - expected) ** 2 / expected for c in counts)
    digit_pct = [c / n * 100 for c in counts]
    findings["digit_chi2"] = chi2
    findings["digit_pct"] = digit_pct

    # Autocorrelation on DIGITOVER(4) signal: 1 if digit>4 else 0
    over4 = [1 if d > 4 else 0 for d in digits]
    over4_rate = sum(over4) / n
    findings["over4_rate"] = over4_rate

    sig_t = sig_threshold(n)
    acorrs = []
    for lag in range(1, 6):
        r = autocorr(over4, lag)
        acorrs.append(r)
    findings["autocorr"] = acorrs

    # Strongest digit autocorrelation
    best_lag = max(range(5), key=lambda i: abs(acorrs[i]))
    best_r   = acorrs[best_lag]
    if abs(best_r) > sig_t:
        findings["edges"].append({
            "type": "digit_autocorr",
            "detail": f"lag-{best_lag+1} r={best_r:+.3f} (sig>{sig_t:.3f})",
            "strength": abs(best_r),
        })

    # Per-trigger conditional WR for each digit → DIGITOVER(4)
    cond_wr = {}
    for trigger_d in range(10):
        idxs = [i for i, d in enumerate(digits[:-1]) if d == trigger_d]
        if len(idxs) < 10:
            continue
        wins = sum(1 for i in idxs if digits[i + 1] > 4)
        cond_wr[trigger_d] = (wins / len(idxs), len(idxs))

    findings["cond_wr_over4"] = cond_wr

    # Implied BE from base DIGITOVER(4) rate (Deriv prices to base rate)
    implied_payout = (1 - over4_rate) / over4_rate if over4_rate > 0 else 0
    implied_be     = 1 / (1 + implied_payout) if implied_payout > 0 else 0
    findings["implied_payout"] = implied_payout
    findings["implied_be"]     = implied_be

    # Flag triggers where conditional WR meaningfully beats implied BE
    for d, (wr, cnt) in cond_wr.items():
        margin = wr - implied_be
        if margin > 0.04 and cnt >= 15:  # >4% above BE, decent sample
            findings["edges"].append({
                "type": "digit_trigger",
                "detail": f"digit {d} → DIGITOVER(4): WR={wr*100:.1f}% vs BE={implied_be*100:.1f}% (+{margin*100:.1f}%) n={cnt}",
                "strength": margin,
            })

    # Same analysis for DIGITUNDER(4): win if digit < 4
    under4_rate = 1 - over4_rate
    implied_payout_u = over4_rate / under4_rate if under4_rate > 0 else 0
    implied_be_u     = 1 / (1 + implied_payout_u) if implied_payout_u > 0 else 0
    for trigger_d in range(10):
        idxs = [i for i, td in enumerate(digits[:-1]) if td == trigger_d]
        if len(idxs) < 10:
            continue
        wins_u = sum(1 for i in idxs if digits[i + 1] < 4)
        wr_u   = wins_u / len(idxs)
        margin_u = wr_u - implied_be_u
        if margin_u > 0.04 and len(idxs) >= 15:
            findings["edges"].append({
                "type": "digit_trigger_under",
                "detail": f"digit {trigger_d} → DIGITUNDER(4): WR={wr_u*100:.1f}% vs BE={implied_be_u*100:.1f}% (+{margin_u*100:.1f}%) n={len(idxs)}",
                "strength": margin_u,
            })

    # ── 4: price-move autocorrelation (direction persistence) ──────────────
    moves    = [prices[i] - prices[i - 1] for i in range(1, n)]
    up_flags = [1 if m > 0 else 0 for m in moves]
    m_acorrs = [autocorr(up_flags, lag) for lag in range(1, 4)]
    findings["move_autocorr"] = m_acorrs

    for lag_i, r in enumerate(m_acorrs):
        if abs(r) > sig_t and abs(r) > 0.05:
            dir_ = "continuation" if r > 0 else "recoil"
            findings["edges"].append({
                "type": "move_autocorr",
                "detail": f"move lag-{lag_i+1} r={r:+.3f} → price {dir_} ({abs(r)*100:.1f}% predictive)",
                "strength": abs(r),
            })

    # ── 5: spike detection & post-spike direction ──────────────────────────
    if n >= ATR_PERIOD + 20:
        atrs   = compute_atr(prices, ATR_PERIOD)
        spikes = []
        for i, (move, atr) in enumerate(zip(moves, atrs)):
            if atr > 0 and abs(move) / atr >= SPIKE_MULT:
                spikes.append((i + 1, "up" if move > 0 else "down"))
        findings["spike_count"] = len(spikes)

        if len(spikes) >= 5:
            results = defaultdict(lambda: defaultdict(lambda: [0, 0]))
            for idx, direction in spikes:
                for settle in SETTLE_RANGE:
                    ei = idx + settle
                    if ei >= n:
                        continue
                    ep = prices[ei]
                    for dur in DUR_RANGE:
                        xi = ei + dur
                        if xi >= n:
                            continue
                        xp  = prices[xi]
                        # recoil: up spike → expect fall
                        won_recoil = (xp < ep) if direction == "up" else (xp > ep)
                        results[settle][dur][0] += int(won_recoil)
                        results[settle][dur][1] += 1
            findings["spike_results"] = {
                f"s{s}d{d}": results[s][d] for s in SETTLE_RANGE for d in DUR_RANGE
            }

            # Find best recoil cell
            best_wr, best_cell = 0.0, None
            for s in SETTLE_RANGE:
                for d in DUR_RANGE:
                    w, t = results[s][d]
                    if t >= 5:
                        wr = w / t
                        if wr > best_wr:
                            best_wr, best_cell = wr, (s, d, w, t)

            if best_cell and best_wr > 0.60:
                s, d, w, t = best_cell
                findings["edges"].append({
                    "type": "spike_recoil",
                    "detail": f"spike recoil settle={s}t dur={d}t: WR={best_wr*100:.1f}% ({w}/{t} spikes)",
                    "strength": best_wr - 0.5,
                })

            # Continuation check (inverse of recoil)
            best_cont, best_cont_cell = 0.0, None
            for s in SETTLE_RANGE:
                for d in DUR_RANGE:
                    w, t = results[s][d]
                    if t >= 5:
                        cont_wr = (t - w) / t  # continuation wins if recoil loses
                        if cont_wr > best_cont:
                            best_cont, best_cont_cell = cont_wr, (s, d, t - w, t)

            if best_cont_cell and best_cont > 0.60:
                s, d, w, t = best_cont_cell
                findings["edges"].append({
                    "type": "spike_continuation",
                    "detail": f"spike continuation settle={s}t dur={d}t: WR={best_cont*100:.1f}% ({w}/{t} spikes)",
                    "strength": best_cont - 0.5,
                })

    # ── 6: ATR calm regime (for ACCU timing) ──────────────────────────────
    if n >= ATR_PERIOD + 20:
        atr_vals = compute_atr(prices, ATR_PERIOD)[ATR_PERIOD:]
        if atr_vals:
            sorted_atrs  = sorted(atr_vals)
            atr_p10      = sorted_atrs[int(len(sorted_atrs) * 0.10)]
            atr_p50      = sorted_atrs[int(len(sorted_atrs) * 0.50)]
            calm_fraction = sum(1 for a in atr_vals if a <= atr_p10) / len(atr_vals)
            findings["atr_p10"]       = atr_p10
            findings["atr_p50"]       = atr_p50
            findings["calm_fraction"] = calm_fraction

    # Rank edges by strength
    findings["edges"].sort(key=lambda e: e["strength"], reverse=True)
    return findings


# ── collection ─────────────────────────────────────────────────────────────

async def collect(client: DerivClient, symbol: str, duration: int):
    prices = []

    async def on_tick(tick: dict):
        q = tick.get("quote")
        if q is not None:
            prices.append(float(q))

    try:
        client.on_tick(symbol, on_tick)
        await client.subscribe_ticks(symbol)
        return symbol, prices, None
    except Exception as e:
        return symbol, prices, str(e)


# ── report ─────────────────────────────────────────────────────────────────

def report(findings_list: list):
    # Sort: symbols with edges first, by best edge strength
    def sort_key(f):
        if f.get("skip"):
            return -1
        return max((e["strength"] for e in f.get("edges", [])), default=0)

    findings_list.sort(key=sort_key, reverse=True)

    print("\n" + "=" * 70)
    print(f"  SYNTHETIC INDEX EDGE SCAN — {DURATION}s collection")
    print("=" * 70)

    for f in findings_list:
        sym = f["symbol"]
        n   = f.get("n", 0)

        if f.get("skip"):
            print(f"\n[{sym}]  SKIP — {f['skip']}")
            continue

        pip   = f.get("pip", "?")
        edges = f.get("edges", [])
        flag  = " *** EDGE FOUND" if edges else ""
        print(f"\n{'─'*70}")
        print(f"[{sym}]  {n} ticks  pip={pip}{flag}")

        # Digit summary
        over4 = f.get("over4_rate", 0)
        chi2  = f.get("digit_chi2", 0)
        ipay  = f.get("implied_payout", 0)
        ibe   = f.get("implied_be", 0)
        chi_tag = ("NON-UNIFORM ***" if chi2 > 27.9 else
                   "non-uniform *"   if chi2 > 16.9 else "uniform")
        print(f"  Digits: chi²={chi2:.1f} {chi_tag}")
        pct_str = "  ".join(f"{d}:{f['digit_pct'][d]:.1f}%" for d in range(10))
        print(f"  Dist:   {pct_str}")
        print(f"  OVER4 base={over4*100:.1f}%  implied payout={ipay*100:.0f}%  BE={ibe*100:.1f}%")

        # Autocorrelation
        ac = f.get("autocorr", [])
        sig_t = sig_threshold(n)
        ac_str = "  ".join(
            f"lag{i+1}:{r:+.3f}{'*' if abs(r)>sig_t else ''}"
            for i, r in enumerate(ac)
        )
        print(f"  AutoCorr(digit):  {ac_str}")

        mac = f.get("move_autocorr", [])
        mac_str = "  ".join(
            f"lag{i+1}:{r:+.3f}{'*' if abs(r)>sig_t else ''}"
            for i, r in enumerate(mac)
        )
        print(f"  AutoCorr(move):   {mac_str}")

        # Spikes
        sc = f.get("spike_count")
        if sc is not None:
            rate = sc / (DURATION / 60)
            print(f"  Spikes: {sc} ({rate:.1f}/min)")

        # ATR calm
        atr50 = f.get("atr_p50")
        atr10 = f.get("atr_p10")
        if atr50 is not None:
            print(f"  ATR p50={atr50:.5f}  p10={atr10:.5f}  (calm 10% of time)")

        # Edges
        if edges:
            print(f"  EDGES:")
            for e in edges:
                print(f"    [{e['type']}] {e['detail']}")
        else:
            print(f"  No edges above threshold")

    # Summary table
    with_edges = [f for f in findings_list if f.get("edges")]
    print("\n" + "=" * 70)
    print(f"  SUMMARY — {len(with_edges)}/{len(findings_list)} symbols with edges")
    print("=" * 70)
    if with_edges:
        for f in with_edges:
            best = f["edges"][0]
            print(f"  {f['symbol']:15s}  [{best['type']}]  {best['detail']}")
    else:
        print("  No edges found above threshold — try a longer collection window")


# ── main ───────────────────────────────────────────────────────────────────

async def main():
    load_dotenv()
    token = os.getenv("DERIV_API_TOKEN") or os.getenv("DERIV_TOKEN")
    if not token:
        print("ERROR: DERIV_API_TOKEN not set in .env")
        return

    client = DerivClient(token)
    await client.connect()

    print(f"Subscribing to {len(SYMBOLS)} symbols, collecting {DURATION}s…")
    print("(symbols that fail to subscribe are silently skipped)\n")

    # Subscribe all in parallel — ignore failures
    price_stores: dict[str, list] = {}
    failed = []

    async def try_subscribe(sym):
        prices = []
        try:
            async def on_tick(tick: dict):
                q = tick.get("quote")
                if q is not None:
                    prices.append(float(q))
            client.on_tick(sym, on_tick)
            await client.subscribe_ticks(sym)
            price_stores[sym] = prices
            print(f"  ✓ {sym}")
        except Exception as e:
            failed.append(sym)
            print(f"  ✗ {sym}: {e}")

    await asyncio.gather(*[try_subscribe(s) for s in SYMBOLS])

    if failed:
        print(f"\nSkipped {len(failed)}: {', '.join(failed)}")

    print(f"\nCollecting for {DURATION}s…")
    await asyncio.sleep(DURATION)

    await client.disconnect()

    # Analyse
    findings_list = [analyze(sym, prices) for sym, prices in price_stores.items()]
    # Add placeholder for failed symbols
    for sym in failed:
        findings_list.append({"symbol": sym, "n": 0, "edges": [], "skip": "subscription failed"})

    report(findings_list)


asyncio.run(main())
