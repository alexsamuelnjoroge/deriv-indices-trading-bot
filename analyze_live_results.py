"""Parse bot log files and report per-symbol win/loss stats.

Usage:
  python analyze_live_results.py                  # all logs in logs/
  python analyze_live_results.py --since 2026-08-25  # only from this date forward
  python analyze_live_results.py --today          # today's log only
"""

import argparse
import glob
import os
import re
from collections import defaultdict
from datetime import datetime, date

LOG_DIR = "logs"

WIN_PAT  = re.compile(r"\[([A-Z0-9_]+)\]\s+WIN\s+\|\s+Profit:\s+\+?([\d.]+)")
LOSS_PAT = re.compile(r"\[([A-Z0-9_]+)\]\s+LOSS\s+\|\s+Loss:\s+-([\d.]+)")
DATE_PAT = re.compile(r"^(\d{4}-\d{2}-\d{2})")   # loguru format
DATE_ALT = re.compile(r"^(\d{2}:\d{2}:\d{2})")   # old HH:MM:SS format


def log_date_from_name(path):
    """Extract date from log filename like bot_2026-08-25.log"""
    m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(path))
    if m:
        return datetime.strptime(m.group(1), "%Y-%m-%d").date()
    return None


def parse_logs(since: date = None):
    stats = defaultdict(lambda: {"wins": 0, "losses": 0, "profit": 0.0, "loss_amt": 0.0})

    log_files = sorted(glob.glob(os.path.join(LOG_DIR, "bot_*.log")))
    if not log_files:
        print(f"No bot_*.log files found in {LOG_DIR}/")
        return stats

    for path in log_files:
        file_date = log_date_from_name(path)
        if since and file_date and file_date < since:
            continue

        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = WIN_PAT.search(line)
                if m:
                    sym, amt = m.group(1), float(m.group(2))
                    stats[sym]["wins"] += 1
                    stats[sym]["profit"] += amt
                    continue
                m = LOSS_PAT.search(line)
                if m:
                    sym, amt = m.group(1), float(m.group(2))
                    stats[sym]["losses"] += 1
                    stats[sym]["loss_amt"] += amt

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", help="Only include logs from this date (YYYY-MM-DD)")
    parser.add_argument("--today", action="store_true", help="Only today's log")
    args = parser.parse_args()

    since = None
    if args.today:
        since = date.today()
    elif args.since:
        since = datetime.strptime(args.since, "%Y-%m-%d").date()

    if since:
        print(f"\nAnalyzing logs from {since} onwards...\n")
    else:
        print("\nAnalyzing all available logs...\n")

    stats = parse_logs(since)

    if not stats:
        print("No trades found.")
        return

    # sort by total trades descending
    rows = sorted(stats.items(), key=lambda x: x[1]["wins"] + x[1]["losses"], reverse=True)

    print(f"  {'Symbol':<12} {'Wins':>6} {'Losses':>7} {'Total':>6}  {'WR%':>6}  {'Net P&L':>10}  {'Avg Win':>8}  {'Avg Loss':>9}")
    print(f"  {'-'*12}  {'-'*5}  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*9}")

    grand_wins = grand_losses = 0
    grand_profit = grand_loss = 0.0

    for sym, s in rows:
        w, l = s["wins"], s["losses"]
        total = w + l
        if total == 0:
            continue
        wr = w / total * 100
        net = s["profit"] - s["loss_amt"]
        avg_win  = s["profit"] / w if w else 0
        avg_loss = s["loss_amt"] / l if l else 0
        net_str  = f"${net:+.2f}"
        flag = ""
        if total >= 10:
            if wr >= 70:   flag = " +"
            elif wr < 45:  flag = " -"
        print(f"  {sym:<12}  {w:>5}  {l:>6}  {total:>5}  {wr:>5.1f}%  {net_str:>10}  ${avg_win:>7.2f}  ${avg_loss:>8.2f}{flag}")
        grand_wins   += w
        grand_losses += l
        grand_profit += s["profit"]
        grand_loss   += s["loss_amt"]

    total_all = grand_wins + grand_losses
    if total_all:
        wr_all = grand_wins / total_all * 100
        net_all = grand_profit - grand_loss
        print(f"\n  {'TOTAL':<12}  {grand_wins:>5}  {grand_losses:>6}  {total_all:>5}  {wr_all:>5.1f}%  ${net_all:>+9.2f}")

    print()


if __name__ == "__main__":
    main()
