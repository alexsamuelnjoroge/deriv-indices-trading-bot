"""
Telegram notifications for pro_bot and the incubator, using the same bot and chat as the
Deriv bot (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID in .env). Silent when they are not set.
Messages are sent from a background thread so a slow network never delays trading.
"""

import os
import threading

from loguru import logger

from src.monitoring.telegram_alerts import TelegramAlerter

_alerter = None


def _get() -> TelegramAlerter:
    global _alerter
    if _alerter is None:
        token, chat = os.getenv("TELEGRAM_BOT_TOKEN", ""), os.getenv("TELEGRAM_CHAT_ID", "")
        _alerter = TelegramAlerter(token, chat, enabled=bool(token and chat))
        logger.info(f"Telegram notifications {'ON' if _alerter.enabled else 'OFF (TELEGRAM_* not in .env)'}")
    return _alerter


def notify(source: str, text: str) -> None:
    alerter = _get()
    if not alerter.enabled:
        return
    body = f"[{source}] {text}"
    for ch in "_*`[":                     # the alerter sends Markdown; strategy names contain _
        body = body.replace(ch, "\\" + ch)

    def send():
        try:
            alerter._send_sync(body)
        except Exception as e:
            logger.warning(f"Telegram send failed: {e}")

    threading.Thread(target=send, daemon=True).start()
