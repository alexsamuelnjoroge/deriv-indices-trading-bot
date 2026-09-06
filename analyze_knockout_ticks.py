"""
Estimate knockout tick positions from existing ACCU logs.

Matches ACCU open/close pairs by contract_id, computes elapsed seconds
(≈ ticks, since Boom/Crash tick ~1/s), and shows distribution per symbol.

Usage (on VPS):
    python analyze_knockout_ticks.py /var/log/trading-bot/*.log
    python analyze_knockout_ticks.py /root/trading-bot/logs/bot.log
"""

import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta

# ── Log line format ──────────────────────────────────────────────────────────
# 2024-09-04 00:00:42 | INFO | ...
# or
# 00:00:42 | INFO | ...

TIME_FULL  = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
TIME_SHORT = re.compile(r"^(\d{2}:\d{2}:\d{2})")

OPEN_PAT  = re.compile(r"\[([A-Z0-9_]+)\] ACCU opened \| ID: (\d+) \|.*hold=(\d+)t")
CLOSE_PAT = re.compile(r"\[([A-Z0-9_]+)\] ACCU (\d+) (WIN|LOSS \(knockout\))")


def parse_time(line: str):
    m = TIME_FULL.search(line)
    if m:
        return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    m = TIME_SHORT.match(line)
    if m:
        return datetime.strptime(m.group(1), "%H:%M:%S")
    return None


def main(paths):
    opens: dict[str, dict] = {}          # contract_id -> {symbol, time, hold_ticks}
    knockouts: dict[str, list] = defaultdict(list)  # symbol -> [ticks_survived]
    wins_by_sym: dict[str, int] = defaultdict(int)

    ref_date = datetime(2000, 1, 1)  # fallback date for short timestamps

    for path in paths:
        try:
            lines = open(path, encoding="utf-8", errors="replace").readlines()
        except FileNotFoundError:
            print(f"[skip] {path} not found")
            continue

        current_date = ref_date
        for line in lines:
            t = parse_time(line)
            if t and t.year > 2000:
                current_date = t.replace(hour=0, minute=0, second=0)
            if t and t.year == 1900:
                # short timestamp — attach to current day
                t = current_date.replace(hour=t.hour, minute=t.minute, second=t.second)

            m = OPEN_PAT.search(line)
            if m:
                sym, cid, hold = m.group(1), m.group(2), int(m.group(3))
                opens[cid] = {"symbol": sym, "time": t, "hold_ticks": hold}
                continue

            m = CLOSE_PAT.search(line)
            if m:
                sym, cid, outcome = m.group(1), m.group(2), m.group(3)
                if outcome == "WIN":
                    wins_by_sym[sym] += 1
                else:  # LOSS (knockout)
                    if cid in opens and opens[cid]["time"] and t:
                        delta = t - opens[cid]["time"]
                        secs = delta.total_seconds()
                        # handle midnight rollover
                        if secs < 0:
                            secs += 86400
                        hold_ticks = opens[cid]["hold_ticks"]
                        ticks = min(round(secs), hold_ticks)  # cap at hold_ticks
                        knockouts[sym].append((ticks, hold_ticks))
                    del opens.get(cid, {}) or {}
                    opens.pop(cid, None)

    # ── Report ───────────────────────────────────────────────────────────────
    all_syms = sorted(set(list(knockouts.keys()) + list(wins_by_sym.keys())))

    if not all_syms:
        print("No ACCU trades found in provided log files.")
        return

    for sym in all_syms:
        ko_list = knockouts[sym]
        wins    = wins_by_sym[sym]
        total   = len(ko_list) + wins
        if total == 0:
            continue

        print(f"\n{'='*60}")
        print(f"  {sym}   wins={wins}  knockouts={len(ko_list)}  total={total}")
        print(f"{'='*60}")

        if not ko_list:
            print("  No knockouts recorded.")
            continue

        hold_ticks = ko_list[0][1]  # use first observed hold_ticks
        ticks_arr  = [t for t, _ in ko_list]
        avg  = sum(ticks_arr) / len(ticks_arr)
        mn   = min(ticks_arr)
        mx   = max(ticks_arr)

        print(f"  hold_ticks={hold_ticks}  avg_ko_tick={avg:.1f}  min={mn}  max={mx}")

        # Bucket distribution
        buckets = defaultdict(int)
        for tick in ticks_arr:
            buckets[tick] += 1

        print(f"  Knockout distribution (tick → count):")
        for tick in sorted(buckets):
            bar = "█" * buckets[tick]
            pct = buckets[tick] / len(ticks_arr) * 100
            print(f"    t{tick:>3}/{hold_ticks}  {bar:<20} {buckets[tick]:3d} ({pct:4.1f}%)")

        # Early (<=3), mid (4..hold-3), late (hold-2..hold)
        early = sum(1 for t in ticks_arr if t <= 3)
        late  = sum(1 for t in ticks_arr if t >= hold_ticks - 1)
        print(f"\n  Early (t≤3): {early}/{len(ticks_arr)} ({early/len(ticks_arr)*100:.0f}%)"
              f"   Late (t≥{hold_ticks-1}): {late}/{len(ticks_arr)} ({late/len(ticks_arr)*100:.0f}%)")

    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_knockout_ticks.py <logfile> [logfile ...]")
        sys.exit(1)
    main(sys.argv[1:])
