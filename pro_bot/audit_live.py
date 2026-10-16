"""
Live performance audit of pro_bot from MT5 deal history (read-only).

Groups deals into positions (magic 20260101), maps each to its strategy via
the order comment, attributes pyramid adds to their parent position, and
reports per strategy/symbol: trades, win rate, net $, EV in R.

Usage:
    python pro_bot/audit_live.py [--since 2026-08-01]
"""

import argparse
import re
from collections import defaultdict
from datetime import datetime, timezone

import MetaTrader5 as mt5

MAGIC = 20260101

PREFIX = [
    (r"^EMA (up|dn) \| price@", "bb_rsi_pullback"),
    (r"^EMA (up|down|dn) \| RSI", "mtf_pullback"),
    (r"^MACD hist flip", "macd_momentum"),
    (r"^PDL first touch", "pdl"),
    (r"^CRW_", "crw"),
    (r"^SHP_", "shp"),
    (r"^(Bullish|Bearish|Hidden)", "rsi_divergence"),
    (r"^([A-Z]{2,5})\b", None),          # e.g. "FVG bull ...", "CBVE buy ..."
]


def strategy_of(comment: str) -> str:
    reason = comment.split("|", 1)[1] if "|" in comment else comment
    for pat, name in PREFIX:
        m = re.match(pat, reason)
        if m:
            return name or m.group(1).lower()
    return f"? {reason[:20]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-01-01")
    args = ap.parse_args()
    since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)

    assert mt5.initialize(), mt5.last_error()
    ai = mt5.account_info()
    deals = [d for d in (mt5.history_deals_get(since, datetime.now(timezone.utc)) or []) if d.magic == MAGIC]
    orders = {o.ticket: o for o in (mt5.history_orders_get(since, datetime.now(timezone.utc)) or [])}

    pos = defaultdict(lambda: {"in": None, "out": []})
    for d in deals:
        if d.entry == mt5.DEAL_ENTRY_IN:
            pos[d.position_id]["in"] = d
        else:
            pos[d.position_id]["out"].append(d)

    info = {}
    rows = []
    for pid, p in pos.items():
        d_in = p["in"]
        if d_in is None or not p["out"]:
            continue
        sym = d_in.symbol
        if sym not in info:
            info[sym] = mt5.symbol_info(sym)
        si = info[sym]
        o = orders.get(d_in.order)
        sl = o.sl if o else 0.0
        net = sum(x.profit + x.commission + x.swap for x in p["out"]) + d_in.commission
        risk = abs(d_in.price - sl) / si.trade_tick_size * si.trade_tick_value * d_in.volume if sl else 0
        rows.append(dict(pid=pid, sym=sym, t_in=d_in.time, t_out=max(x.time for x in p["out"]),
                         comment=d_in.comment, strat=strategy_of(d_in.comment), net=net,
                         r=net / risk if risk else None, pyramid="pyramid" in d_in.comment))

    # pyramid adds -> parent: latest non-pyramid position on same symbol open at that time
    rows.sort(key=lambda r: r["t_in"])
    for r in rows:
        if r["pyramid"]:
            parents = [q for q in rows if not q["pyramid"] and q["sym"] == r["sym"]
                       and q["t_in"] <= r["t_in"] <= q["t_out"]]
            r["strat"] = parents[-1]["strat"] if parents else "pyramid (orphan)"

    first = datetime.fromtimestamp(rows[0]["t_in"], timezone.utc).date() if rows else "-"
    last = datetime.fromtimestamp(rows[-1]["t_out"], timezone.utc).date() if rows else "-"
    print(f"\nAccount {ai.login} {ai.server}  balance {ai.balance:.2f} {ai.currency}  "
          f"| closed positions {len(rows)}  {first} -> {last}\n")

    def table(key_fn, title, base_only):
        g = defaultdict(list)
        for r in rows:
            if base_only and r["pyramid"]:
                continue
            g[key_fn(r)].append(r)
        print(f"  {title}")
        print(f"  {'group':<28}{'n':>4}{'WR%':>7}{'net $':>10}{'EV R':>8}{'net R':>8}")
        tot = []
        for k, rs in sorted(g.items(), key=lambda kv: sum(x["net"] for x in kv[1])):
            rr = [x["r"] for x in rs if x["r"] is not None]
            wr = sum(x["net"] > 0 for x in rs) / len(rs) * 100
            ev = sum(rr) / len(rr) if rr else float("nan")
            print(f"  {k:<28}{len(rs):>4}{wr:>7.1f}{sum(x['net'] for x in rs):>10.2f}{ev:>8.2f}{sum(rr):>8.1f}")
            tot += rs
        rr = [x["r"] for x in tot if x["r"] is not None]
        print(f"  {'TOTAL':<28}{len(tot):>4}{sum(x['net'] > 0 for x in tot) / len(tot) * 100:>7.1f}"
              f"{sum(x['net'] for x in tot):>10.2f}{sum(rr) / len(rr):>8.2f}{sum(rr):>8.1f}\n")

    table(lambda r: f"{r['strat']} {r['sym']}", "By strategy + symbol (base entries only, R from original SL)", True)
    table(lambda r: r["strat"] + (" +pyr" if r["pyramid"] else ""), "Base vs pyramid adds (all symbols)", False)
    table(lambda r: r["sym"], "By symbol (incl. pyramids)", False)
    table(lambda r: datetime.fromtimestamp(r["t_in"], timezone.utc).strftime("%Y-%m %b"), "By month (incl. pyramids)", False)

    xau = [r for r in rows if r["sym"].startswith("XAU") and not r["pyramid"]]
    unk = sorted({r["comment"] for r in rows if r["strat"].startswith("?")})
    if unk:
        print("  Unmapped comments:", unk)
    print(f"  XAUUSD base trades: {len(xau)}")
    mt5.shutdown()


if __name__ == "__main__":
    main()
