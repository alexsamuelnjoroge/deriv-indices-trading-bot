"""
JD50 Noise Reduction Investigation

Live losses prompted this deeper analysis. Investigates four angles:

  A  Walk-forward stability (5×10k) — verify the 3/3 edge holds on more data
  B  Time-of-day (UTC 4-hour buckets) — identify losing hours to block
  C  Spike isolation — isolated spikes (>N ticks quiet) vs clustered ones
  D  Parameter re-sweep focused on noise filters:
       spike_mult       [2, 3, 4, 5, 6]
       spike_cooldown   [3, 5, 8, 10, 15]
       contract_dur     [3, 5, 7]
       require_recoil   [True, False]

Key context:
  - JD50 spikes are algorithmic: down-spike → BUY_RISE, up-spike → BUY_FALL
  - ATR_PERIOD=30 absorbs multiple spikes, so spike_mult=3 is the effective
    threshold (not 8-15 as used for Crash/Boom)
  - cooldown=5 means entry fires at tick+5 after the spike; this is the
    implicit settle window
  - Clustered spikes (multiple spikes within the cooldown window) may
    produce noisier recoils than isolated ones

Usage:
  python sweep_jd50_deep.py
  python sweep_jd50_deep.py --section a    # just walk-forward
  python sweep_jd50_deep.py --section b    # just time-of-day
  python sweep_jd50_deep.py --section c    # just isolation analysis
  python sweep_jd50_deep.py --section d    # just parameter sweep
"""

import argparse
import sys
from collections import defaultdict

from loguru import logger

from src.data.history import fetch_ticks

logger.remove()
logger.add(sys.stderr, level="ERROR", format="{time:HH:mm:ss} | {level} | {message}")

SYMBOL      = "JD50"
ATR_PERIOD  = 30
PAYOUT_PCT  = 0.92
BE_PCT      = 100 / (1 + PAYOUT_PCT)  # 52.08%

TOTAL_TICKS  = 50_000
WINDOW_SIZE  = 10_000
WINDOWS      = TOTAL_TICKS // WINDOW_SIZE
MIN_TRADES   = 10

SEP = "=" * 100

# Current live config (baseline)
LIVE_SPIKE_MULT  = 3.0
LIVE_COOLDOWN    = 5
LIVE_DUR         = 5
LIVE_RECOIL      = False


# ── Core simulator ────────────────────────────────────────────────────────────

def _atr(prices: list[float], end: int) -> float | None:
    hist = prices[max(0, end - ATR_PERIOD): end]
    if len(hist) < ATR_PERIOD:
        return None
    ranges = [abs(hist[i] - hist[i - 1]) for i in range(1, len(hist))]
    avg = sum(ranges) / len(ranges)
    return avg if avg > 0 else None


def simulate(
    ticks: list[dict],
    spike_mult: float,
    cooldown: int,
    dur: int,
    recoil_confirm: bool,
) -> list[dict]:
    """
    Returns list of trade dicts:
      won, epoch, ticks_since_last_spike, spike_size_atr, direction
    """
    prices = [float(t["quote"]) for t in ticks]
    epochs = [int(t["epoch"]) for t in ticks]
    n = len(prices)
    trades = []

    pending_dir          = None
    settle_left          = 0       # not used here — cooldown handles the wait
    cooldown_left        = 0
    ticks_since_spike    = 9999
    last_spike_idx       = -9999

    for i in range(ATR_PERIOD + 1, n - dur):
        if cooldown_left > 0:
            cooldown_left -= 1
        ticks_since_spike += 1

        atr = _atr(prices, i)
        if atr is None or atr <= 0:
            continue

        move = prices[i] - prices[i - 1]

        # ── Spike detection ────────────────────────────────────────────
        if abs(move) > spike_mult * atr:
            pending_dir       = 1 if move > 0 else -1
            cooldown_left     = cooldown
            ticks_since_spike = 0
            last_spike_idx    = i
            continue

        # ── Fire entry when cooldown expires ──────────────────────────
        if pending_dir is not None and cooldown_left == 0:
            direction = pending_dir

            if recoil_confirm:
                recoil = prices[i] - prices[i - 1]
                confirmed = (
                    (direction == -1 and recoil > 0) or
                    (direction ==  1 and recoil < 0)
                )
                if not confirmed:
                    pending_dir = None
                    continue

            entry   = prices[i]
            exit_p  = prices[i + dur]
            won     = (exit_p > entry) if direction == -1 else (exit_p < entry)

            trades.append({
                "won":               won,
                "epoch":             epochs[i],
                "ticks_since_spike": i - last_spike_idx,
                "spike_size_atr":    abs(prices[last_spike_idx] - prices[last_spike_idx - 1]) / (_atr(prices, last_spike_idx) or 1),
                "direction":         direction,
            })
            pending_dir = None

    return trades


def wf_stats(trades: list[dict]) -> dict:
    if not trades:
        return {"wr": 0.0, "ev": 0.0, "n": 0}
    wins = sum(t["won"] for t in trades)
    wr   = wins / len(trades) * 100
    ev   = (wr / 100 - BE_PCT / 100) * PAYOUT_PCT * 100
    return {"wr": wr, "ev": ev, "n": len(trades)}


# ── Section A: Walk-forward stability ─────────────────────────────────────────

def section_a(ticks: list[dict]) -> None:
    print()
    print(SEP)
    print("  A — WALK-FORWARD STABILITY  (5×10k, current live params)")
    print(f"  spike_mult={LIVE_SPIKE_MULT}×  cooldown={LIVE_COOLDOWN}t  dur={LIVE_DUR}t  "
          f"recoil={LIVE_RECOIL}  payout={PAYOUT_PCT*100:.0f}%  BE={BE_PCT:.1f}%")
    print(SEP)

    splits = [ticks[i * WINDOW_SIZE: (i + 1) * WINDOW_SIZE] for i in range(WINDOWS)]
    passes = 0
    all_trades = []

    for w, seg in enumerate(splits):
        trades = simulate(seg, LIVE_SPIKE_MULT, LIVE_COOLDOWN, LIVE_DUR, LIVE_RECOIL)
        s = wf_stats(trades)
        flag = "PASS" if s["n"] >= MIN_TRADES and s["wr"] >= BE_PCT else "FAIL"
        if flag == "PASS":
            passes += 1
        all_trades.extend(trades)
        print(f"  W{w+1}: WR={s['wr']:5.1f}%  EV={s['ev']:+7.3f}%  n={s['n']:4}  [{flag}]")

    s = wf_stats(all_trades)
    flag = "***" if passes == WINDOWS and s["ev"] > 0 else (
           "**"  if passes >= WINDOWS - 1 and s["ev"] > 0 else "")
    print(f"  ALL: WR={s['wr']:5.1f}%  EV={s['ev']:+7.3f}%  n={s['n']:4}  "
          f"{passes}/{WINDOWS} passes {flag}")


# ── Section B: Time-of-day analysis ───────────────────────────────────────────

def section_b(ticks: list[dict]) -> None:
    print()
    print(SEP)
    print("  B — TIME-OF-DAY  (UTC 4-hour buckets, current live params)")
    print(SEP)

    trades = simulate(ticks, LIVE_SPIKE_MULT, LIVE_COOLDOWN, LIVE_DUR, LIVE_RECOIL)

    buckets: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        hour_utc  = (t["epoch"] % 86400) // 3600
        bucket    = f"{(hour_utc // 4) * 4:02d}-{(hour_utc // 4) * 4 + 4:02d}"
        buckets[bucket].append(t)

    print(f"  {'UTC':>5}  {'WR%':>6}  {'EV%':>8}  {'n':>5}  bar")
    print(f"  {'-'*5}  {'-'*6}  {'-'*8}  {'-'*5}  ---")

    for label in sorted(buckets):
        s = wf_stats(buckets[label])
        if s["n"] < 5:
            continue
        bar   = "#" * int(max(0, s["wr"] - 40) / 2)
        flag  = " <<" if s["wr"] < BE_PCT else ""
        eat_start = (int(label[:2]) + 3) % 24
        eat_end   = (int(label[:2]) + 3 + 4) % 24
        eat_label = f"EAT {eat_start:02d}-{eat_end:02d}"
        print(f"  {label}  {s['wr']:>6.1f}%  {s['ev']:>+8.3f}%  {s['n']:>5}  "
              f"{bar:<20} {eat_label}{flag}")

    print()
    print(f"  BE = {BE_PCT:.1f}%  |  << = losing bucket  |  EAT = UTC+3")


# ── Section C: Spike isolation ────────────────────────────────────────────────

def section_c(ticks: list[dict]) -> None:
    print()
    print(SEP)
    print("  C — SPIKE ISOLATION  (isolated vs clustered, current live params)")
    print(SEP)

    trades = simulate(ticks, LIVE_SPIKE_MULT, LIVE_COOLDOWN, LIVE_DUR, LIVE_RECOIL)

    thresholds = [5, 10, 15, 20, 30]

    print(f"  {'isolation':>12}  {'WR%':>6}  {'EV%':>8}  {'n':>5}")
    print(f"  {'-'*12}  {'-'*6}  {'-'*8}  {'-'*5}")

    for thresh in thresholds:
        isolated   = [t for t in trades if t["ticks_since_spike"] >= thresh]
        clustered  = [t for t in trades if t["ticks_since_spike"] <  thresh]
        si = wf_stats(isolated)
        sc = wf_stats(clustered)
        print(f"  isolated>={thresh:>2}t  {si['wr']:>6.1f}%  {si['ev']:>+8.3f}%  {si['n']:>5}")
        print(f"  cluster <{thresh:>2}t   {sc['wr']:>6.1f}%  {sc['ev']:>+8.3f}%  {sc['n']:>5}")
        print()

    # Also split by spike direction
    print(f"  {'direction':>10}  {'WR%':>6}  {'EV%':>8}  {'n':>5}")
    print(f"  {'-'*10}  {'-'*6}  {'-'*8}  {'-'*5}")
    for direction, label in [(-1, "down->RISE"), (1, "up->FALL")]:
        sub = [t for t in trades if t["direction"] == direction]
        s   = wf_stats(sub)
        print(f"  {label:>10}  {s['wr']:>6.1f}%  {s['ev']:>+8.3f}%  {s['n']:>5}")


# ── Section D: Parameter sweep ────────────────────────────────────────────────

def section_d(ticks: list[dict]) -> None:
    print()
    print(SEP)
    print("  D — PARAMETER SWEEP  (5×10k walk-forward, noise filter focus)")
    print(SEP)

    splits = [ticks[i * WINDOW_SIZE: (i + 1) * WINDOW_SIZE] for i in range(WINDOWS)]

    SPIKE_MULTS = [2.0, 3.0, 4.0, 5.0, 6.0]
    COOLDOWNS   = [3, 5, 8, 10, 15]
    DURATIONS   = [3, 5, 7]
    RECOILS     = [False, True]

    results = []
    for spike in SPIKE_MULTS:
        for cd in COOLDOWNS:
            for dur in DURATIONS:
                for recoil in RECOILS:
                    passes = 0
                    all_t  = []
                    for seg in splits:
                        t = simulate(seg, spike, cd, dur, recoil)
                        s = wf_stats(t)
                        if s["n"] >= MIN_TRADES and s["wr"] >= BE_PCT:
                            passes += 1
                        all_t.extend(t)
                    s = wf_stats(all_t)
                    results.append({
                        "spike": spike, "cd": cd, "dur": dur, "recoil": recoil,
                        "wr": s["wr"], "ev": s["ev"], "n": s["n"],
                        "passes": passes,
                    })

    results.sort(key=lambda x: (x["passes"], x["ev"]), reverse=True)

    print(f"  {'spk':>4}  {'cd':>3}  {'dur':>3}  {'rcl':>4}  "
          f"{'WR%':>6}  {'EV%':>8}  {'n':>5}  pass")
    print(f"  {'-'*4}  {'-'*3}  {'-'*3}  {'-'*4}  "
          f"{'-'*6}  {'-'*8}  {'-'*5}  ----")

    shown = 0
    for r in results:
        if r["n"] < MIN_TRADES:
            continue
        if shown >= 30 and r["ev"] <= 0:
            break
        live = " <-- LIVE" if (
            r["spike"] == LIVE_SPIKE_MULT and r["cd"] == LIVE_COOLDOWN and
            r["dur"] == LIVE_DUR and r["recoil"] == LIVE_RECOIL
        ) else ""
        flag = " ***" if r["passes"] >= WINDOWS and r["ev"] > 0 else (
               " *"   if r["passes"] >= WINDOWS - 1 and r["ev"] > 0 else "")
        print(f"  {r['spike']:>4.0f}  {r['cd']:>3}  {r['dur']:>3}  "
              f"{'Y' if r['recoil'] else 'N':>4}  "
              f"{r['wr']:>6.1f}%  {r['ev']:>+8.3f}%  {r['n']:>5}  "
              f"{r['passes']}/{WINDOWS}{flag}{live}")
        shown += 1


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="JD50 noise reduction investigation")
    parser.add_argument("--section", default="all",
                        choices=["a", "b", "c", "d", "all"],
                        help="Which section to run (default: all)")
    args = parser.parse_args()

    print()
    print(f"  JD50 NOISE INVESTIGATION  |  {TOTAL_TICKS:,} ticks  |  {WINDOWS}×{WINDOW_SIZE:,}")
    print(f"  Live config: spike={LIVE_SPIKE_MULT}×  cooldown={LIVE_COOLDOWN}t  "
          f"dur={LIVE_DUR}t  recoil={LIVE_RECOIL}  payout={PAYOUT_PCT*100:.0f}%")
    print()

    ticks = fetch_ticks(SYMBOL, count=TOTAL_TICKS)
    ticks = ticks[-TOTAL_TICKS:]
    print(f"  Loaded {len(ticks):,} ticks for {SYMBOL}")

    sec = args.section
    if sec in ("a", "all"):
        section_a(ticks)
    if sec in ("b", "all"):
        section_b(ticks)
    if sec in ("c", "all"):
        section_c(ticks)
    if sec in ("d", "all"):
        section_d(ticks)

    print()
    print(SEP)
    print(f"  BE={BE_PCT:.1f}%  payout={PAYOUT_PCT*100:.0f}%  *** all windows pass AND EV>0")


if __name__ == "__main__":
    main()
