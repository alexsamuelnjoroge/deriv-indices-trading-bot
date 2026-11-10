"""
XAUUSD incubator — trades the frozen watch-list rules on the MT5 demo so they can be
judged on data that did not exist when they were chosen (started 2026-11-10).

Rules and their research numbers live in pro_bot/incubator_signals.py.
Separate process from pro_bot.main; its positions use their own magic numbers.

    python -m pro_bot.incubator              # trade on demo
    python -m pro_bot.incubator --dry-run    # log signals only

Each rule holds at most one position. Entry: market order right after the signal bar
closes. Exit: market close H bars later (time exit, as researched). A wide disaster
stop (6 sigma of the holding-period move) is the only stop. Trades are written to
logs/incubator_trades.csv; review after ~6 months (2027-05).
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

import MetaTrader5 as mt5

from pro_bot.execution.mt5_client import MT5Client
from pro_bot.incubator_signals import GOLD, RULES, align, compute_signals, session_id, symbols_needed

DEFAULT_RULES = ("A_asia_dip", "B_usd_resid")
MAGIC_BASE = 20261110
BARS = 8000                      # ~28 sessions: 20 for betas + 288-bar warm-ups
MAX_SIGNAL_AGE = 120             # seconds after bar close; older signals are skipped
DISASTER_SIGMA = 6.0
STATE_FILE = Path("data/incubator_state.json")
TRADES_FILE = Path("logs/incubator_trades.csv")
TRADE_FIELDS = ["rule", "side", "signal_bar_utc", "z", "entry_time_utc", "entry_price",
                "exit_time_utc", "exit_price", "lots", "gross_bp", "profit_usd",
                "commission_usd", "swap_usd", "net_usd", "exit_reason", "ticket"]
NY = ZoneInfo("America/New_York")


def server_offset(utc_ts: float) -> int:
    """IC Markets server time = New York time + 7h (UTC+2 winter, UTC+3 summer)."""
    ny = datetime.fromtimestamp(utc_ts, timezone.utc).astimezone(NY)
    return int(ny.utcoffset().total_seconds()) + 7 * 3600


def magic_for(rule: str) -> int:
    return MAGIC_BASE + list(RULES).index(rule)


# ── state ────────────────────────────────────────────────────────────────────
def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"open": {}, "last_bar": 0}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE_FILE)


def record_trade(row: dict) -> None:
    TRADES_FILE.parent.mkdir(parents=True, exist_ok=True)
    new = not TRADES_FILE.exists()
    with open(TRADES_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRADE_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def utc_str(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# ── market data ──────────────────────────────────────────────────────────────
def fetch_closes(symbols, forming_bar_utc: int):
    """Closed M5 bars for every symbol, indexed by UTC bar open time."""
    out = {}
    for sym in symbols:
        r = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 0, BARS)
        if r is None or len(r) == 0:
            logger.warning(f"no bars for {sym}: {mt5.last_error()}")
            return None, None
        t = np.array([x["time"] for x in r], dtype=np.int64)
        t = t - np.array([server_offset(x) for x in t])
        df = pd.DataFrame({"open": r["open"], "close": r["close"]},
                          index=pd.to_datetime(t, unit="s", utc=True))
        df = df[df.index.asi8 // 10**9 < forming_bar_utc]
        out[sym] = df
    gold = out.pop(GOLD)
    return gold, {s: d["close"] for s, d in out.items()}


# ── trade lifecycle ──────────────────────────────────────────────────────────
def close_and_record(client: MT5Client, rule: str, tr: dict, reason: str) -> None:
    ticket = tr["ticket"]
    reason = tr.get("close_reason", reason)
    if reason != "stop_or_external" and mt5.positions_get(ticket=ticket):
        if not client.close_position(ticket, magic=magic_for(rule)):
            logger.error(f"[{rule}] close failed for ticket {ticket}; will retry")
            return
    tr["close_reason"] = reason
    time.sleep(1.0)
    deals = mt5.history_deals_get(position=ticket) or []
    if not deals:
        deals = mt5.history_deals_get(datetime.fromtimestamp(tr["entry_ts"] - 86400),
                                      datetime.fromtimestamp(time.time() + 86400 * 2)) or []
        deals = [d for d in deals if d.position_id == ticket]
    entry = [d for d in deals if d.entry == mt5.DEAL_ENTRY_IN]
    exits = [d for d in deals if d.entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_OUT_BY)]
    if not exits:
        logger.warning(f"[{rule}] no exit deal found yet for ticket {ticket}; will retry")
        return
    entry_px = entry[0].price if entry else tr["entry_price"]
    exit_px = exits[-1].price
    exit_t = exits[-1].time - server_offset(exits[-1].time)
    profit = sum(d.profit for d in deals)
    comm = sum(d.commission for d in deals)
    swap = sum(d.swap for d in deals)
    gross_bp = tr["side"] * np.log(exit_px / entry_px) * 1e4
    record_trade({
        "rule": rule, "side": tr["side"], "signal_bar_utc": utc_str(tr["signal_bar"]),
        "z": round(tr["z"], 3), "entry_time_utc": utc_str(tr["entry_ts"]),
        "entry_price": entry_px, "exit_time_utc": utc_str(exit_t), "exit_price": exit_px,
        "lots": tr["lots"], "gross_bp": round(gross_bp, 2), "profit_usd": round(profit, 2),
        "commission_usd": round(comm, 2), "swap_usd": round(swap, 2),
        "net_usd": round(profit + comm + swap, 2), "exit_reason": reason, "ticket": ticket,
    })
    logger.info(f"[{rule}] CLOSED {reason} | {'BUY' if tr['side'] > 0 else 'SELL'} "
                f"{entry_px} -> {exit_px} | gross {gross_bp:+.1f}bp | net ${profit + comm + swap:+.2f}")
    tr["closed"] = True


def manage_open(client: MT5Client, state: dict) -> None:
    if not state["open"]:
        return
    positions = mt5.positions_get(symbol=GOLD)
    if positions is None:
        logger.warning(f"positions_get failed: {mt5.last_error()}")
        return
    live = {p.ticket for p in positions}
    now = time.time()
    for rule, tr in list(state["open"].items()):
        if tr["ticket"] not in live:
            close_and_record(client, rule, tr, "stop_or_external")
        elif now >= tr["exit_ts"]:
            close_and_record(client, rule, tr, "time")
        if tr.get("closed"):
            del state["open"][rule]
            save_state(state)


def adopt_orphans(client: MT5Client, state: dict, rules) -> None:
    """Close incubator positions MT5 holds but state doesn't know about (e.g. lost state file)."""
    positions = mt5.positions_get(symbol=GOLD) or []
    known = {tr["ticket"] for tr in state["open"].values()}
    magics = {magic_for(r): r for r in RULES}
    for p in positions:
        if p.magic in magics and p.ticket not in known:
            logger.warning(f"orphan incubator position {p.ticket} ({magics[p.magic]}) — closing")
            client.close_position(p.ticket, magic=p.magic)


def process_bar(client: MT5Client, state: dict, rules, lots: float, dry_run: bool,
                forming_bar_utc: int) -> None:
    gold, others = fetch_closes(symbols_needed({r: RULES[r] for r in rules}), forming_bar_utc)
    if gold is None or len(gold) < 2000:
        return
    df = align(gold["close"], others)
    sig = compute_signals(df, {r: RULES[r] for r in rules})
    i_ts = int(df.index[-1].timestamp())
    bar_close = i_ts + 300
    age = time.time() - bar_close
    for rule in rules:
        side = int(sig[f"{rule}_side"].iloc[-1])
        z = float(sig[f"{rule}_z"].iloc[-1])
        if side == 0:
            continue
        H = RULES[rule]["H"]
        exit_bar = i_ts + H * 300
        tag = f"[{rule}] signal {'BUY' if side > 0 else 'SELL'} z={z:+.2f} bar={utc_str(i_ts)}"
        if rule in state["open"]:
            logger.info(f"{tag} — skipped, position already open")
            continue
        if not (session_id(i_ts) == session_id(bar_close) == session_id(exit_bar)):
            logger.info(f"{tag} — skipped, would cross the 20:30 UTC session boundary")
            continue
        if age > MAX_SIGNAL_AGE:
            logger.info(f"{tag} — skipped, signal is {age:.0f}s old")
            continue
        if dry_run:
            logger.info(f"{tag} — DRY RUN, no order")
            continue
        tick = mt5.symbol_info_tick(GOLD)
        px = tick.ask if side > 0 else tick.bid
        sig_bp = float(pd.Series(np.diff(np.log(df[GOLD].to_numpy()[-289:])) * 1e4).std())
        stop_dist = px * DISASTER_SIGMA * sig_bp * np.sqrt(H) / 1e4
        sl = px - stop_dist if side > 0 else px + stop_dist
        res = client.place_order(GOLD, "BUY" if side > 0 else "SELL", lots, sl, 0.0,
                                 comment=f"incub_{rule[0]}", magic=magic_for(rule))
        if not res.success:
            logger.error(f"{tag} — order failed: {res.error}")
            continue
        pos = mt5.positions_get(symbol=GOLD) or []
        ticket = next((p.ticket for p in pos if p.magic == magic_for(rule)), res.ticket)
        state["open"][rule] = {
            "ticket": ticket, "side": side, "z": z, "lots": lots, "signal_bar": i_ts,
            "entry_ts": time.time(), "entry_price": res.price,
            "exit_ts": bar_close + H * 300, "sl": sl,
        }
        save_state(state)
        logger.info(f"{tag} — OPENED ticket {ticket} @ {res.price} SL {sl:.2f} "
                    f"exit at {utc_str(bar_close + H * 300)} UTC")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", default=",".join(DEFAULT_RULES))
    ap.add_argument("--lots", type=float, default=0.01)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-real", action="store_true")
    args = ap.parse_args()
    rules = [r.strip() for r in args.rules.split(",") if r.strip()]
    unknown = [r for r in rules if r not in RULES]
    if unknown:
        sys.exit(f"unknown rules: {unknown}; choose from {list(RULES)}")

    load_dotenv()
    Path("logs").mkdir(exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {level:<7} | {message}")
    logger.add("logs/incubator_{time:YYYYMMDD}.log", level="DEBUG", rotation="1 day", retention="60 days")

    client = MT5Client()
    if not client.connect():
        sys.exit("MT5 connection failed")
    if mt5.account_info().trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO and not args.allow_real:
        sys.exit("Refusing to run on a non-demo account without --allow-real")
    for s in symbols_needed({r: RULES[r] for r in rules}):
        mt5.symbol_select(s, True)

    state = load_state()
    adopt_orphans(client, state, rules)
    logger.info(f"incubator up | rules={rules} lots={args.lots} dry_run={args.dry_run} "
                f"open={list(state['open'])}")

    while True:
        try:
            manage_open(client, state)
            r = mt5.copy_rates_from_pos(GOLD, mt5.TIMEFRAME_M5, 0, 1)
            if r is not None and len(r):
                forming = int(r[0]["time"]) - server_offset(int(r[0]["time"]))
                closed_bar = forming - 300
                if closed_bar > state["last_bar"]:
                    process_bar(client, state, rules, args.lots, args.dry_run, forming)
                    state["last_bar"] = closed_bar
                    save_state(state)
            time.sleep(5)
        except KeyboardInterrupt:
            logger.info("stopped by user")
            break
        except Exception as e:
            logger.exception(f"loop error: {e}")
            time.sleep(30)
            if mt5.terminal_info() is None:
                client.connect()


if __name__ == "__main__":
    main()
