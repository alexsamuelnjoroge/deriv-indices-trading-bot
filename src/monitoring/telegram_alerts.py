"""
Telegram alerts — sends trade results, daily summaries, and halt notifications.

Setup:
  1. Create a bot via @BotFather on Telegram → get TELEGRAM_BOT_TOKEN
  2. Send any message to the bot, then open:
     https://api.telegram.org/bot<TOKEN>/getUpdates
     to find your TELEGRAM_CHAT_ID
  3. Add both to your .env file:
       TELEGRAM_BOT_TOKEN=123456789:ABCdef...
       TELEGRAM_CHAT_ID=987654321
  4. Set telegram.enabled: true in config.yaml
"""

import asyncio
import json
import urllib.request
import urllib.error
from typing import Optional
from loguru import logger


class TelegramAlerter:
    _BASE = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self, token: str, chat_id: str, enabled: bool = True,
                 on_trade: bool = True, on_daily_summary: bool = True,
                 on_halt: bool = True):
        self.token   = token
        self.chat_id = chat_id
        self.enabled = enabled and bool(token) and bool(chat_id)
        self.on_trade         = on_trade
        self.on_daily_summary = on_daily_summary
        self.on_halt          = on_halt

    # ------------------------------------------------------------------ #
    #  Public alert methods
    # ------------------------------------------------------------------ #

    async def send_trade(
        self,
        symbol: str,
        action: str,        # BUY_RISE / BUY_FALL
        stake: float,
        profit: float,
        balance: float,
        win_rate: float,
        total_trades: int,
    ):
        if not self.enabled or not self.on_trade:
            return
        won   = profit > 0
        emoji = "✅" if won else "❌"
        direction = "RISE" if action == "BUY_RISE" else "FALL"
        msg = (
            f"{emoji} *{symbol}* — {direction}\n"
            f"{'Win' if won else 'Loss'}: `{profit:+.2f} USD`\n"
            f"Stake: `{stake:.2f}` | Balance: `{balance:.2f}`\n"
            f"WR: `{win_rate:.1f}%` over {total_trades} trades"
        )
        await self._send(msg)

    async def send_halt(self, symbol: str, reason: str, balance: float):
        if not self.enabled or not self.on_halt:
            return
        msg = (
            f"🛑 *BOT HALTED* — {symbol}\n"
            f"Reason: {reason}\n"
            f"Balance: `{balance:.2f} USD`"
        )
        await self._send(msg)

    async def send_daily_summary(
        self,
        symbol: str,
        trades: int,
        wins: int,
        losses: int,
        net_pnl: float,
        win_rate: float,
        balance: float,
    ):
        if not self.enabled or not self.on_daily_summary:
            return
        emoji = "📈" if net_pnl >= 0 else "📉"
        msg = (
            f"{emoji} *Daily Summary — {symbol}*\n"
            f"Trades: `{trades}` | W/L: `{wins}/{losses}`\n"
            f"Win rate: `{win_rate:.1f}%`\n"
            f"Net P&L: `{net_pnl:+.2f} USD`\n"
            f"Balance: `{balance:.2f} USD`"
        )
        await self._send(msg)

    async def send_message(self, text: str):
        if not self.enabled:
            return
        await self._send(text)

    # ------------------------------------------------------------------ #
    #  Internal HTTP send (runs in thread to avoid blocking the loop)
    # ------------------------------------------------------------------ #

    async def _send(self, text: str):
        try:
            await asyncio.to_thread(self._send_sync, text)
        except Exception as e:
            logger.warning(f"Telegram send failed: {e}")

    def _send_sync(self, text: str):
        url  = self._BASE.format(token=self.token)
        data = json.dumps({
            "chat_id":    self.chat_id,
            "text":       text,
            "parse_mode": "Markdown",
        }).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()


def build_alerter(config: dict) -> Optional[TelegramAlerter]:
    """Build alerter from config + env vars. Returns None if disabled."""
    import os
    cfg = config.get("telegram", {})
    if not cfg.get("enabled", False):
        return None
    token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        logger.warning("Telegram enabled in config but TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set in .env")
        return None
    return TelegramAlerter(
        token=token,
        chat_id=chat_id,
        enabled=True,
        on_trade=cfg.get("on_trade", True),
        on_daily_summary=cfg.get("on_daily_summary", True),
        on_halt=cfg.get("on_halt", True),
    )
