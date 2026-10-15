"""
Daily Donchian trend bot on Deriv multipliers (XAU, XAG, BTC, ETH).

Separate process from main.py. Runs on a demo account unless --allow-real.

  python trend_bot.py              # run forever
  python trend_bot.py --once       # one pass then exit
  python trend_bot.py --dry-run    # log signals, place no orders, don't save state
"""

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger

from src.api.client import DerivClient
from src.monitoring.telegram_alerts import build_alerter
from src.strategies.donchian_trend import entry_signal, exit_signal

DAY = 86400


def fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M")


class TrendBot:
    def __init__(self, cfg: dict, client: DerivClient, alerter, dry_run: bool):
        self.cfg = cfg
        self.client = client
        self.alerter = alerter
        self.dry_run = dry_run
        self.state_path = Path(cfg["state_file"])
        self.state = self._load_state()

    # ── state ────────────────────────────────────────────────────────────

    def _load_state(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return {"realized_pnl": 0.0, "symbols": {}}

    def _save_state(self):
        if self.dry_run:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        tmp.replace(self.state_path)

    def _sym(self, sym: str) -> dict:
        return self.state["symbols"].setdefault(sym, {"last_bar": 0, "position": None, "pending": None})

    # ── market data ──────────────────────────────────────────────────────

    async def completed_daily_bars(self, sym: str) -> list[dict]:
        resp = await self.client._send({"ticks_history": sym, "style": "candles",
                                        "granularity": DAY, "count": 150, "end": "latest"})
        now = time.time()
        bars = [{"epoch": int(c["epoch"]), "high": float(c["high"]),
                 "low": float(c["low"]), "close": float(c["close"])}
                for c in resp.get("candles", [])]
        return [b for b in bars if b["epoch"] + DAY <= now]

    async def contract_status(self, cid: str) -> dict:
        resp = await self.client._send({"proposal_open_contract": 1, "contract_id": int(cid)})
        return resp["proposal_open_contract"]

    # ── sizing ───────────────────────────────────────────────────────────

    def stake_for_new_trade(self) -> float:
        capital = self.cfg["capital"] + self.state["realized_pnl"]
        risk = capital * self.cfg["risk_pct"]
        return max(round(risk / self.cfg["sl_frac_of_stake"], 2), 1.0)

    # ── orders ───────────────────────────────────────────────────────────

    async def open_contract(self, sym: str, direction: int, stake: float) -> dict:
        ct = "MULTUP" if direction == 1 else "MULTDOWN"
        res = await self.client.buy_multiplier(
            symbol=sym, contract_type=ct, stake=stake,
            multiplier=self.cfg["multiplier"],
            sl_amount=stake * self.cfg["sl_frac_of_stake"], tp_amount=None,
        )
        cid = str(res["contract_id"])
        st = await self.contract_status(cid)
        return {"contract_id": cid, "stake": stake, "expiry": int(st.get("date_expiry") or 0),
                "entry_spot": float(st.get("entry_spot") or st.get("current_spot") or 0)}

    async def close_contract(self, cid: str) -> float:
        await self.client.sell_contract(cid)
        st = await self.contract_status(cid)
        return float(st.get("profit", 0))

    # ── trade bookkeeping ────────────────────────────────────────────────

    def record_close(self, sym: str, pos: dict, leg_pnl: float, reason: str):
        total = pos.get("roll_pnl", 0.0) + leg_pnl
        risk = pos["stake"] * self.cfg["sl_frac_of_stake"]
        self.state["realized_pnl"] += total
        path = Path(self.cfg["trades_csv"])
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["symbol", "direction", "opened_utc", "closed_utc", "stake",
                            "pnl", "r_multiple", "rolls", "reason"])
            w.writerow([sym, "LONG" if pos["dir"] == 1 else "SHORT", fmt_ts(pos["opened"]),
                        fmt_ts(time.time()), f"{pos['stake']:.2f}", f"{total:.2f}",
                        f"{total / risk:.2f}", pos.get("rolls", 0), reason])
        msg = (f"[{sym}] CLOSED {'LONG' if pos['dir'] == 1 else 'SHORT'} | {reason} | "
               f"P&L {total:+.2f} ({total / risk:+.2f}R) | rolls {pos.get('rolls', 0)} | "
               f"bot P&L {self.state['realized_pnl']:+.2f}")
        logger.info(msg)
        return msg

    async def alert(self, text: str):
        if self.alerter:
            await self.alerter.send_message(f"📈 *Trend bot*\n{text}")

    # ── per-symbol step ──────────────────────────────────────────────────

    async def step(self, sym: str):
        s = self._sym(sym)
        pos = s["position"]

        # 1. was the contract closed by Deriv (stop-loss, stake-out, expiry)?
        if pos and not self.dry_run:
            st = await self.contract_status(pos["contract_id"])
            if st.get("is_sold"):
                msg = self.record_close(sym, pos, float(st.get("profit", 0)), "stop/settled by Deriv")
                s["position"] = pos = None
                await self.alert(msg)

        # 2. new completed daily bar -> evaluate signals
        bars = await self.completed_daily_bars(sym)
        if bars and bars[-1]["epoch"] > s["last_bar"]:
            s["last_bar"] = bars[-1]["epoch"]
            if pos:
                if exit_signal(bars, self.cfg["exit_bars"], pos["dir"]):
                    s["pending"] = {"action": "exit"}
                    logger.info(f"[{sym}] exit signal on {fmt_ts(bars[-1]['epoch'])} close {bars[-1]['close']}")
            else:
                sig = entry_signal(bars, self.cfg["entry_bars"])
                s["pending"] = {"action": "enter", "dir": sig} if sig else None
                if sig:
                    logger.info(f"[{sym}] {'LONG' if sig == 1 else 'SHORT'} breakout on "
                                f"{fmt_ts(bars[-1]['epoch'])} close {bars[-1]['close']}")

        # 3. execute pending order (retried each pass while the market is closed)
        pend = s["pending"]
        if pend and not self.dry_run:
            try:
                if pend["action"] == "exit" and pos:
                    pnl = await self.close_contract(pos["contract_id"])
                    msg = self.record_close(sym, pos, pnl, f"{self.cfg['exit_bars']}-day exit channel")
                    s["position"] = pos = None
                    await self.alert(msg)
                elif pend["action"] == "enter" and not pos:
                    stake = self.stake_for_new_trade()
                    leg = await self.open_contract(sym, pend["dir"], stake)
                    s["position"] = pos = {"dir": pend["dir"], "opened": time.time(),
                                           "rolls": 0, "roll_pnl": 0.0, **leg}
                    msg = (f"[{sym}] OPENED {'LONG' if pend['dir'] == 1 else 'SHORT'} | stake ${stake:.2f} "
                           f"x{self.cfg['multiplier']} | risk ${stake * self.cfg['sl_frac_of_stake']:.2f} | "
                           f"spot {leg['entry_spot']} | expires {fmt_ts(leg['expiry'])}")
                    logger.info(msg)
                    await self.alert(msg)
                s["pending"] = None
            except RuntimeError as e:
                logger.warning(f"[{sym}] {pend['action']} not filled yet ({e}) — will retry")

        # 4. roll before expiry (crypto multipliers expire ~7.6 days after purchase)
        roll_at = (pos or {}).get("expiry", 0) - self.cfg["roll_hours_before_expiry"] * 3600
        if pos and pos["expiry"] and time.time() >= roll_at and not self.dry_run:
            leg_pnl = await self.close_contract(pos["contract_id"])
            pos["roll_pnl"] = pos.get("roll_pnl", 0.0) + leg_pnl
            try:
                leg = await self.open_contract(sym, pos["dir"], pos["stake"])
                pos.update(leg)
                pos["rolls"] = pos.get("rolls", 0) + 1
                logger.info(f"[{sym}] rolled (leg P&L {leg_pnl:+.2f}) new expiry {fmt_ts(leg['expiry'])}")
            except RuntimeError as e:
                msg = self.record_close(sym, pos, 0.0, f"roll re-open failed: {e}")
                s["position"] = None
                await self.alert(msg)

    async def run_pass(self):
        for sym in self.cfg["symbols"]:
            try:
                await self.step(sym)
            except Exception as e:
                logger.exception(f"[{sym}] step failed: {e}")
            finally:
                self._save_state()

    def status_lines(self) -> list[str]:
        out = [f"bot P&L {self.state['realized_pnl']:+.2f} | capital "
               f"{self.cfg['capital'] + self.state['realized_pnl']:.2f} | next stake "
               f"${self.stake_for_new_trade():.2f}"]
        for sym in self.cfg["symbols"]:
            s = self._sym(sym)
            p = s["position"]
            pos_txt = (f"{'LONG' if p['dir'] == 1 else 'SHORT'} since {fmt_ts(p['opened'])}" if p else "flat")
            out.append(f"  {sym:<10} {pos_txt:<30} last bar {fmt_ts(s['last_bar']) if s['last_bar'] else '-'}"
                       f"  pending {s['pending']}")
        return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-real", action="store_true")
    ap.add_argument("--config", default="trend_bot.yaml")
    args = ap.parse_args()

    load_dotenv()
    os.makedirs("logs", exist_ok=True)
    logger.remove()
    logger.add("logs/trend_bot_{time:YYYY-MM-DD}.log", rotation="00:00", retention="30 days",
               level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}",
               filter=lambda r: r["name"] != "src.api.client" or r["level"].no >= 30)

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    main_cfg = yaml.safe_load(open("config.yaml", encoding="utf-8")) if Path("config.yaml").exists() else {}
    alerter = build_alerter(main_cfg)

    client = DerivClient(os.getenv("DERIV_API_TOKEN"), os.getenv("DERIV_APP_ID", "1089"))
    await client.connect()
    if not client.is_virtual and not args.allow_real:
        logger.error("Connected account is NOT a demo account — refusing to trade. Use --allow-real to override.")
        await client.disconnect()
        return

    bot = TrendBot(cfg, client, alerter, args.dry_run)
    mode = "DRY RUN" if args.dry_run else ("DEMO" if client.is_virtual else "REAL")
    logger.info(f"Trend bot started [{mode}] | {', '.join(cfg['symbols'])} | "
                f"Donchian {cfg['entry_bars']}/{cfg['exit_bars']} | risk {cfg['risk_pct']*100:.2f}%")
    await bot.alert(f"started [{mode}] on {', '.join(cfg['symbols'])}")

    try:
        while True:
            await bot.run_pass()
            for line in bot.status_lines():
                logger.info(line)
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
