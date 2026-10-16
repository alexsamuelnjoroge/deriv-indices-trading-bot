"""
Gold daily-break drift bot on Deriv multipliers.

Buys XAUUSD x100 shortly before Deriv's daily gold break and sells a fixed
time after the reopen. The break time comes from Deriv's trading_times API,
so US daylight-saving shifts, Fridays and holidays are handled by the schedule.

  python break_bot.py              # run forever
  python break_bot.py --once       # one pass then exit
  python break_bot.py --dry-run    # log the schedule, place no orders
"""

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger

from src.api.client import DerivClient
from src.monitoring.telegram_alerts import build_alerter


def utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M")


class BreakBot:
    def __init__(self, cfg: dict, client: DerivClient, alerter, dry_run: bool):
        self.cfg = cfg
        self.client = client
        self.alerter = alerter
        self.dry_run = dry_run
        self.state_path = Path(cfg["state_file"])
        self.state = (json.loads(self.state_path.read_text()) if self.state_path.exists()
                      else {"realized_pnl": 0.0, "position": None, "last_trade_date": None})
        self._sessions: dict[str, dict] = {}

    def _save(self):
        if self.dry_run:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        tmp.replace(self.state_path)

    async def alert(self, text: str):
        if self.alerter:
            await self.alerter.send_message(f"🌙 *Break bot*\n{text}")

    # ── schedule ─────────────────────────────────────────────────────────

    async def session(self, date: str) -> dict:
        """Deriv trading times for the symbol on `date` (YYYY-MM-DD), cached."""
        if date not in self._sessions:
            resp = await self.client._send({"trading_times": date})
            times = None
            for m in resp["trading_times"]["markets"]:
                for sm in m["submarkets"]:
                    for s in sm["symbols"]:
                        if s.get("underlying_symbol", s.get("symbol")) == self.cfg["symbol"]:
                            times = s["times"]
            self._sessions[date] = times or {"open": ["--"], "close": ["--"]}
        return self._sessions[date]

    async def plan_for(self, day: datetime) -> dict | None:
        """Entry/exit epochs for the break on `day`, or None if there is no break."""
        date = day.strftime("%Y-%m-%d")
        t = await self.session(date)
        if len(t["close"]) < 2 or len(t["open"]) < 2:
            return None                     # Friday, holiday or no intraday break
        base = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        hms = lambda s: timedelta(hours=int(s[:2]), minutes=int(s[3:5]), seconds=int(s[6:8]))
        brk = (base + hms(t["close"][0])).timestamp()
        reopen = (base + hms(t["open"][1])).timestamp()
        exit_at = reopen + self.cfg["exit_minutes_after_reopen"] * 60
        nxt = await self.session((base + timedelta(days=1)).strftime("%Y-%m-%d"))
        if nxt["open"][0] != "00:00:00":    # next day closed: get out before midnight
            exit_at = min(exit_at, (base + timedelta(hours=23, minutes=55)).timestamp())
        return {"date": date, "entry_at": brk - self.cfg["entry_minutes_before_break"] * 60,
                "break_at": brk, "reopen_at": reopen, "exit_at": exit_at}

    # ── orders ───────────────────────────────────────────────────────────

    async def contract(self, cid: str) -> dict:
        resp = await self.client._send({"proposal_open_contract": 1, "contract_id": int(cid)})
        return resp["proposal_open_contract"]

    def record(self, pos: dict, st: dict, reason: str) -> str:
        pnl = float(st.get("profit", 0))
        exit_spot = float(st.get("exit_tick") or st.get("current_spot") or 0)
        move_bp = (exit_spot / pos["entry_spot"] - 1) * 1e4 if pos["entry_spot"] and exit_spot else 0
        self.state["realized_pnl"] += pnl
        path = Path(self.cfg["trades_csv"])
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["break_date", "opened_utc", "closed_utc", "stake", "entry_spot",
                            "exit_spot", "move_bp", "pnl", "reason"])
            w.writerow([pos["date"], utc(pos["opened"]), utc(time.time()), f"{pos['stake']:.2f}",
                        pos["entry_spot"], exit_spot, f"{move_bp:.1f}", f"{pnl:.2f}", reason])
        msg = (f"CLOSED {pos['date']} | {reason} | move {move_bp:+.1f}bp | P&L {pnl:+.2f} | "
               f"bot P&L {self.state['realized_pnl']:+.2f}")
        logger.info(msg)
        return msg

    async def step(self):
        now = time.time()
        pos = self.state["position"]

        if pos:
            st = await self.contract(pos["contract_id"])
            if st.get("is_sold"):
                msg = self.record(pos, st, "stopped/settled by Deriv")
                self.state["position"] = None
                await self.alert(msg)
            elif now >= pos["exit_at"]:
                try:
                    await self.client.sell_contract(pos["contract_id"])
                    msg = self.record(pos, await self.contract(pos["contract_id"]), "scheduled exit")
                    self.state["position"] = None
                    await self.alert(msg)
                except RuntimeError as e:
                    logger.warning(f"exit not filled yet ({e}) — will retry")
            return

        today = datetime.now(timezone.utc)
        plan = await self.plan_for(today)
        if not plan or self.state["last_trade_date"] == plan["date"]:
            return
        if not (plan["entry_at"] <= now < plan["break_at"] - 60):
            return
        if self.dry_run:
            logger.info(f"[dry-run] would buy now; exit {utc(plan['exit_at'])}")
            self.state["last_trade_date"] = plan["date"]
            return

        capital = self.cfg["capital"] + self.state["realized_pnl"]
        stake = max(round(capital * self.cfg["stake_pct"], 2), 1.0)
        self.state["last_trade_date"] = plan["date"]
        try:
            res = await self.client.buy_multiplier(
                symbol=self.cfg["symbol"], contract_type="MULTUP", stake=stake,
                multiplier=self.cfg["multiplier"], sl_amount=stake * self.cfg["sl_frac_of_stake"],
                tp_amount=None)
        except RuntimeError as e:
            logger.warning(f"entry failed for {plan['date']}: {e}")
            return
        cid = str(res["contract_id"])
        st = await self.contract(cid)
        self.state["position"] = {"contract_id": cid, "date": plan["date"], "stake": stake,
                                  "opened": now, "exit_at": plan["exit_at"],
                                  "entry_spot": float(st.get("entry_spot") or st.get("current_spot") or 0)}
        msg = (f"OPENED long gold {plan['date']} | stake ${stake:.2f} x{self.cfg['multiplier']} | "
               f"spot {self.state['position']['entry_spot']} | break {utc(plan['break_at'])} | "
               f"exit {utc(plan['exit_at'])} UTC")
        logger.info(msg)
        await self.alert(msg)

    async def next_plan_text(self) -> str:
        day = datetime.now(timezone.utc)
        for k in range(8):
            plan = await self.plan_for(day + timedelta(days=k))
            if plan and plan["break_at"] > time.time() and self.state["last_trade_date"] != plan["date"]:
                return (f"next trade: buy {utc(plan['entry_at'])} | break {utc(plan['break_at'])} | "
                        f"reopen {utc(plan['reopen_at'])} | exit {utc(plan['exit_at'])} UTC")
        return "no break found in the next 7 days"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-real", action="store_true")
    ap.add_argument("--config", default="break_bot.yaml")
    args = ap.parse_args()

    load_dotenv()
    os.makedirs("logs", exist_ok=True)
    logger.remove()
    logger.add("logs/break_bot_{time:YYYY-MM-DD}.log", rotation="00:00", retention="30 days",
               level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}",
               filter=lambda r: r["name"] != "src.api.client" or r["level"].no >= 30)

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    main_cfg = yaml.safe_load(open("config.yaml", encoding="utf-8")) if Path("config.yaml").exists() else {}
    client = DerivClient(os.getenv("DERIV_API_TOKEN"), os.getenv("DERIV_APP_ID", "1089"))
    await client.connect()
    if not client.is_virtual and not args.allow_real:
        logger.error("Connected account is NOT a demo account — refusing to trade. Use --allow-real to override.")
        await client.disconnect()
        return

    bot = BreakBot(cfg, client, build_alerter(main_cfg), args.dry_run)
    mode = "DRY RUN" if args.dry_run else ("DEMO" if client.is_virtual else "REAL")
    logger.info(f"Break bot started [{mode}] | {cfg['symbol']} | {await bot.next_plan_text()}")
    last_status = 0.0
    try:
        while True:
            try:
                await bot.step()
            except Exception as e:
                logger.exception(f"step failed: {e}")
            finally:
                bot._save()
            if time.time() - last_status > 3600 or args.once:
                pos = bot.state["position"]
                logger.info(f"status | bot P&L {bot.state['realized_pnl']:+.2f} | "
                            + (f"LONG since {utc(pos['opened'])}, exit {utc(pos['exit_at'])}" if pos
                               else await bot.next_plan_text()))
                last_status = time.time()
            if args.once:
                break
            await asyncio.sleep(cfg["poll_seconds"])
    finally:
        await client.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
