"""
Live performance tracker.

Reads bot log files and reports win rate statistics broken down by hour,
day, rolling window, and trade sequence.

Usage:
  python track.py                   # read local logs/
  python track.py --days 7          # last 7 days only
  python track.py --symbol R_25     # one symbol only
  python track.py --pull-vps        # SCP logs from VPS first, then analyse

On VPS directly:
  ssh root@204.48.29.156 "cd ~/trading-bot && python track.py"
"""

import argparse
import glob
import math
import os
import re
import subprocess
from collections import defaultdict
from datetime import date, datetime, timedelta

# ── Regex patterns (flexible whitespace to handle log format variations) ──────
WIN_RE  = re.compile(
    r"^(\d{2}:\d{2}:\d{2}) \| INFO\s+\| \[(\w+)\] WIN\s+\| Profit: \+([\d.]+) \| Balance: ([\d.]+)"
)
LOSS_RE = re.compile(
    r"^(\d{2}:\d{2}:\d{2}) \| INFO\s+\| \[(\w+)\] LOSS \| Loss:\s+-?([\d.]+) \| Balance: ([\d.]+)"
)

BREAKEVEN = 52.0
VPS_HOST  = "root@204.48.29.156"
VPS_LOGDIR = "~/trading-bot/logs/"


# ── Data loading ───────────────────────────────────────────────────────────────

def parse_logs(logdir: str, days: int = None, symbol: str = None) -> list:
    """Return sorted list of (datetime, symbol, won: bool, profit: float, balance: float)."""
    trades = []
    cutoff = (date.today() - timedelta(days=days)) if days else None

    for log_file in sorted(glob.glob(os.path.join(logdir, "bot_*.log"))):
        basename = os.path.basename(log_file)
        try:
            log_date = datetime.strptime(basename, "bot_%Y-%m-%d.log").date()
        except ValueError:
            continue
        if cutoff and log_date < cutoff:
            continue

        with open(log_file, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                m = WIN_RE.match(line)
                if m:
                    ts, sym, profit, balance = m.groups()
                    if symbol and sym != symbol:
                        continue
                    dt = datetime.combine(log_date,
                                          datetime.strptime(ts, "%H:%M:%S").time())
                    trades.append((dt, sym, True, float(profit), float(balance)))
                    continue
                m = LOSS_RE.match(line)
                if m:
                    ts, sym, loss, balance = m.groups()
                    if symbol and sym != symbol:
                        continue
                    dt = datetime.combine(log_date,
                                          datetime.strptime(ts, "%H:%M:%S").time())
                    trades.append((dt, sym, False, -float(loss), float(balance)))

    trades.sort(key=lambda x: x[0])
    return trades


def pull_vps_logs(local_dir: str = "logs_vps"):
    """SCP log files from VPS into a local directory."""
    os.makedirs(local_dir, exist_ok=True)
    print(f"Pulling logs from {VPS_HOST}:{VPS_LOGDIR} ...")
    result = subprocess.run(
        ["scp", f"{VPS_HOST}:{VPS_LOGDIR}bot_*.log", local_dir + "/"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"SCP error: {result.stderr.strip()}")
    else:
        count = len(glob.glob(os.path.join(local_dir, "bot_*.log")))
        print(f"  Pulled {count} log file(s) into {local_dir}/\n")
    return local_dir


# ── Helpers ────────────────────────────────────────────────────────────────────

def wr(wins: int, total: int) -> float:
    return round(wins / total * 100, 1) if total > 0 else 0.0


def ci95(wins: int, total: int) -> tuple:
    """Wilson score 95% CI (more accurate than normal approx for small samples)."""
    if total == 0:
        return (0.0, 100.0)
    p = wins / total
    z = 1.96
    n = total
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0, centre - margin) * 100, 1),
            round(min(1, centre + margin) * 100, 1))


def bar(pct: float, width: int = 20) -> str:
    filled = round(pct / 100 * width)
    return "#" * filled + "." * (width - filled)


def wr_label(pct: float) -> str:
    if pct >= BREAKEVEN + 5:
        return f"{pct:.1f}% (+)"
    elif pct >= BREAKEVEN:
        return f"{pct:.1f}% (~)"
    else:
        return f"{pct:.1f}% (-)"


def streak_lengths(trades: list) -> tuple:
    best_win = best_loss = cur_win = cur_loss = 0
    for _, _, won, _, _ in trades:
        if won:
            cur_win += 1; cur_loss = 0
            best_win = max(best_win, cur_win)
        else:
            cur_loss += 1; cur_win = 0
            best_loss = max(best_loss, cur_loss)
    return best_win, best_loss


# ── Report ─────────────────────────────────────────────────────────────────────

def report(trades: list, symbol_filter: str = None):
    if not trades:
        print("No trades found in the specified log directory and date range.")
        return

    total   = len(trades)
    wins    = sum(1 for _, _, w, _, _ in trades if w)
    losses  = total - wins
    net_pnl = round(sum(p for _, _, _, p, _ in trades), 2)
    bal     = trades[-1][4]
    overall = wr(wins, total)
    lo, hi  = ci95(wins, total)
    latest  = trades[-1][0].strftime("%Y-%m-%d %H:%M")

    W = 62
    sym_label = symbol_filter or "ALL SYMBOLS"
    print("=" * W)
    print(f"  LIVE PERFORMANCE TRACKER  |  {sym_label}  |  {latest} EAT")
    print("=" * W)

    # ── Summary ────────────────────────────────────────────────────────────
    print()
    print("SUMMARY")
    print(f"  Total trades  : {total}")
    print(f"  Wins / Losses : {wins} W / {losses} L")
    print(f"  Win rate      : {overall:.1f}%   (breakeven: {BREAKEVEN:.1f}%)")
    print(f"  95% CI        : [{lo:.1f}%, {hi:.1f}%]")
    print(f"  Net P&L       : {net_pnl:+.2f} USD")
    print(f"  Balance       : {bal:.2f} USD")

    if lo > BREAKEVEN:
        verdict = "CONFIRMED EDGE - lower bound of CI is above breakeven"
    elif overall > BREAKEVEN:
        verdict = "LIKELY EDGE    - avg above breakeven, needs more trades to confirm"
    else:
        verdict = "NO EDGE        - average below breakeven"
    print(f"  Verdict       : {verdict}")

    # ── Rolling windows ─────────────────────────────────────────────────────
    print()
    print("ROLLING WINDOWS  (most recent N trades)")
    for n in [10, 20, 30, 50, 100]:
        if total >= n:
            chunk  = trades[-n:]
            w_n    = sum(1 for _, _, w, _, _ in chunk if w)
            rwr    = wr(w_n, n)
            lo_n, hi_n = ci95(w_n, n)
            arrow  = "^" if rwr >= BREAKEVEN else "v"
            print(f"  Last {n:>3}: {rwr:>5.1f}%  CI [{lo_n:.0f}%-{hi_n:.0f}%]  "
                  f"{w_n}W/{n-w_n}L  {arrow}")

    # ── Trade sequence ───────────────────────────────────────────────────────
    print()
    print("TRADE SEQUENCE  (W=win  L=loss,  each row = 20 trades)")
    chunk_size = 20
    for i in range(0, total, chunk_size):
        chunk = trades[i: i + chunk_size]
        seq   = " ".join("W" if w else "L" for _, _, w, _, _ in chunk)
        w_c   = sum(1 for _, _, w, _, _ in chunk if w)
        pct   = wr(w_c, len(chunk))
        flag  = "^" if pct >= BREAKEVEN else " "
        print(f"  [{i+1:>3}-{i+len(chunk):<3}] {seq:<59}  {pct:>5.1f}% {flag}")

    # ── Win rate by hour of day ──────────────────────────────────────────────
    print()
    print("WIN RATE BY HOUR (EAT)")
    print(f"  {'Hour':<6} {'Trd':>4}  {'W/L':>7}  {'WR':>6}  {'95% CI':>14}  {'Bar (each # = 5%)':20}")
    hour_data = defaultdict(lambda: [0, 0])
    for dt, _, won, _, _ in trades:
        hour_data[dt.hour][1] += 1
        if won:
            hour_data[dt.hour][0] += 1

    for h in sorted(hour_data):
        w_h, t_h = hour_data[h]
        rwr = wr(w_h, t_h)
        lo_h, hi_h = ci95(w_h, t_h)
        b = bar(rwr, 20)
        flag = " ^" if rwr >= BREAKEVEN else "  "
        print(f"  {h:02d}:00  {t_h:>4}  {w_h}/{t_h-w_h:<5}  "
              f"{rwr:>5.1f}%  [{lo_h:>4.0f}%-{hi_h:>4.0f}%]  {b}{flag}")

    # ── Win rate by day ──────────────────────────────────────────────────────
    print()
    print("WIN RATE BY DAY  (running = cumulative WR from day 1)")
    print(f"  {'Date':<12} {'Trd':>4}  {'W/L':>7}  {'Daily WR':>9}  {'Running WR':>11}")
    day_data = defaultdict(lambda: [0, 0])
    for dt, _, won, _, _ in trades:
        day_data[dt.date()][1] += 1
        if won:
            day_data[dt.date()][0] += 1

    run_w = run_t = 0
    for d in sorted(day_data):
        w_d, t_d = day_data[d]
        run_w += w_d; run_t += t_d
        dwr  = wr(w_d, t_d)
        rwr  = wr(run_w, run_t)
        flag = " ^" if dwr >= BREAKEVEN else "  "
        print(f"  {d}  {t_d:>4}  {w_d}/{t_d-w_d:<5}  "
              f"{dwr:>7.1f}%{flag}  {rwr:>9.1f}%")

    # ── Streaks ──────────────────────────────────────────────────────────────
    best_win, best_loss = streak_lengths(trades)
    # Current streak
    cur_streak = 0
    cur_dir    = None
    for _, _, won, _, _ in reversed(trades):
        direction = "W" if won else "L"
        if cur_dir is None:
            cur_dir = direction
        if direction == cur_dir:
            cur_streak += 1
        else:
            break

    print()
    print("STREAKS")
    print(f"  Best win streak   : {best_win} consecutive wins")
    print(f"  Worst loss streak : {best_loss} consecutive losses")
    print(f"  Current streak    : {cur_streak} {cur_dir}")

    # ── Confidence milestone ─────────────────────────────────────────────────
    print()
    print("CONFIDENCE MILESTONES")
    milestones = [50, 100, 200, 500]
    for m in milestones:
        if total >= m:
            lo_m, hi_m = ci95(
                round(wins * m / total), m
            )
            note = "REACHED" if lo_m > BREAKEVEN else "reached (edge not confirmed)"
            print(f"  {m:>4} trades: 95% CI [{lo_m:.1f}%, {hi_m:.1f}%]  {note}")
        else:
            needed = m - total
            # Estimate CI at milestone assuming current WR holds
            est_wins = round(wins + overall / 100 * needed)
            lo_m, hi_m = ci95(est_wins, m)
            note = "edge confirmed" if lo_m > BREAKEVEN else "edge still uncertain"
            print(f"  {m:>4} trades: need {needed:>3} more  ->  "
                  f"CI est [{lo_m:.1f}%, {hi_m:.1f}%]  ({note})")

    print()
    print("=" * W)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live trading performance tracker")
    parser.add_argument("--logdir",   default="logs",  help="Log directory (default: logs/)")
    parser.add_argument("--days",     type=int,        help="Only include last N days")
    parser.add_argument("--symbol",   type=str,        help="Filter to one symbol e.g. R_25")
    parser.add_argument("--pull-vps", action="store_true",
                        help=f"SCP logs from {VPS_HOST} first")
    args = parser.parse_args()

    logdir = args.logdir
    if args.pull_vps:
        logdir = pull_vps_logs()

    trades = parse_logs(logdir, days=args.days, symbol=args.symbol)
    report(trades, symbol_filter=args.symbol)
