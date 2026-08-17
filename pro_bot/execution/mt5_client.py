"""
MT5 Client — connects to MetaTrader 5 terminal and handles orders.

Requirements:
  pip install MetaTrader5

Setup:
  1. Install MetaTrader 5 desktop app (IC Markets, Pepperstone, FTMO, etc.)
  2. Enable "Allow automated trading" in MT5 Tools → Options → Expert Advisors
  3. Set MT5_LOGIN, MT5_PASSWORD, MT5_SERVER in .env

Symbol name mapping (broker-specific — adjust to your broker):
  AUD/USD → AUDUSD
  GBP/USD → GBPUSD
  USD/JPY → USDJPY
  XAU/USD → XAUUSD  (Gold)
"""

import os
import time
from dataclasses import dataclass
from typing import Optional

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

from loguru import logger


@dataclass
class OrderResult:
    success:    bool
    ticket:     int   = 0
    symbol:     str   = ""
    action:     str   = ""
    volume:     float = 0.0
    price:      float = 0.0
    sl:         float = 0.0
    tp:         float = 0.0
    error:      str   = ""


class MT5Client:

    def __init__(self, login: int = 0, password: str = "",
                 server: str = "", path: str = ""):
        if not MT5_AVAILABLE:
            raise RuntimeError(
                "MetaTrader5 package not installed. Run: pip install MetaTrader5\n"
                "Note: only available on Windows with MT5 terminal installed."
            )
        self.login    = login    or int(os.getenv("MT5_LOGIN",    "0"))
        self.password = password or os.getenv("MT5_PASSWORD", "")
        self.server   = server   or os.getenv("MT5_SERVER",   "")
        self.path     = path     or os.getenv("MT5_PATH",     "")
        self._connected = False

    # ── Connection ───────────────────────────────────────────────────────────

    def connect(self) -> bool:
        kwargs = {}
        if self.path:
            kwargs["path"] = self.path
        if self.login:
            kwargs.update(login=self.login, password=self.password,
                          server=self.server)

        if not mt5.initialize(**kwargs):
            logger.error(f"MT5 init failed: {mt5.last_error()}")
            return False

        info = mt5.account_info()
        if info is None:
            logger.error(f"MT5 account info failed: {mt5.last_error()}")
            return False

        self._connected = True
        logger.info(f"MT5 connected | Account: {info.login} | "
                    f"Broker: {info.company} | Balance: {info.balance:.2f} "
                    f"{info.currency}")
        return True

    def disconnect(self):
        mt5.shutdown()
        self._connected = False
        logger.info("MT5 disconnected")

    def get_balance(self) -> float:
        info = mt5.account_info()
        return info.balance if info else 0.0

    def get_equity(self) -> float:
        info = mt5.account_info()
        return info.equity if info else 0.0

    # ── Market data ──────────────────────────────────────────────────────────

    def get_tick(self, symbol: str) -> Optional[dict]:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        return {"bid": tick.bid, "ask": tick.ask,
                "time": tick.time, "symbol": symbol}

    def get_bars(self, symbol: str, timeframe: int,
                 count: int = 500) -> Optional[list[dict]]:
        """
        timeframe: mt5.TIMEFRAME_M5, mt5.TIMEFRAME_H1, mt5.TIMEFRAME_D1, etc.
        Returns list of {open, high, low, close, epoch, tick_volume} dicts,
        oldest first. tick_volume is the number of price ticks in the bar —
        a proxy for activity on OTC instruments like forex/gold.
        """
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            logger.warning(f"No bars for {symbol}: {mt5.last_error()}")
            return None
        return [{"open":        float(r["open"]),
                 "high":        float(r["high"]),
                 "low":         float(r["low"]),
                 "close":       float(r["close"]),
                 "epoch":       int(r["time"]),
                 "tick_volume": int(r["tick_volume"])} for r in rates]

    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        return {
            "point":        info.point,       # smallest price change (e.g. 0.00001)
            "digits":       info.digits,
            "trade_tick_value": info.trade_tick_value,
            "volume_min":   info.volume_min,
            "volume_step":  info.volume_step,
            "volume_max":   info.volume_max,
            "spread":       info.spread,
        }

    # ── Order placement ──────────────────────────────────────────────────────

    def place_order(self, symbol: str, action: str,
                    volume: float, sl_price: float,
                    tp_price: float, comment: str = "pro_bot") -> OrderResult:
        """
        action: "BUY" or "SELL"
        volume: lot size (0.01 = micro lot)
        sl_price / tp_price: absolute prices
        """
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return OrderResult(False, error=f"No tick for {symbol}")

        order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
        price      = tick.ask if action == "BUY" else tick.bid

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       volume,
            "type":         order_type,
            "price":        price,
            "sl":           round(sl_price, mt5.symbol_info(symbol).digits),
            "tp":           round(tp_price, mt5.symbol_info(symbol).digits),
            "deviation":    10,
            "magic":        20260101,
            "comment":      comment,
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            err = result.comment if result else str(mt5.last_error())
            logger.error(f"Order failed [{symbol} {action}]: {err}")
            return OrderResult(False, error=err)

        logger.info(f"Order placed | {symbol} {action} {volume}L | "
                    f"price={price} SL={sl_price:.5f} TP={tp_price:.5f} | "
                    f"ticket={result.order}")
        return OrderResult(True, ticket=result.order, symbol=symbol,
                           action=action, volume=volume, price=price,
                           sl=sl_price, tp=tp_price)

    def close_position(self, ticket: int) -> bool:
        pos = mt5.positions_get(ticket=ticket)
        if not pos:
            return False
        p          = pos[0]
        close_type = mt5.ORDER_TYPE_SELL if p.type == 0 else mt5.ORDER_TYPE_BUY
        tick       = mt5.symbol_info_tick(p.symbol)
        price      = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       p.symbol,
            "volume":       p.volume,
            "type":         close_type,
            "position":     ticket,
            "price":        price,
            "deviation":    10,
            "magic":        20260101,
            "comment":      "pro_bot close",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        ok = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        if ok:
            logger.info(f"Position closed | ticket={ticket}")
        else:
            logger.error(f"Close failed | ticket={ticket}: "
                         f"{result.comment if result else mt5.last_error()}")
        return ok

    def modify_sl(self, ticket: int, new_sl: float) -> bool:
        """Move the stop-loss on an open position (break-even, trail, etc.)."""
        pos = mt5.positions_get(ticket=ticket)
        if not pos:
            logger.warning(f"modify_sl: ticket {ticket} not found")
            return False
        p = pos[0]
        request = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "symbol":   p.symbol,
            "sl":       round(new_sl, mt5.symbol_info(p.symbol).digits),
            "tp":       p.tp,
            "position": ticket,
        }
        result = mt5.order_send(request)
        ok = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        if ok:
            logger.info(f"SL modified | ticket={ticket} new_sl={new_sl:.5f}")
        else:
            logger.error(f"SL modify failed | ticket={ticket}: "
                         f"{result.comment if result else mt5.last_error()}")
        return ok

    def get_open_positions(self, magic: int = 20260101) -> list[dict]:
        positions = mt5.positions_get()
        if not positions:
            return []
        return [
            {"ticket":  p.ticket,
             "symbol":  p.symbol,
             "type":    "BUY" if p.type == 0 else "SELL",
             "volume":  p.volume,
             "price":   p.price_open,
             "sl":      p.sl,
             "tp":      p.tp,
             "profit":  p.profit,
             "pips":    p.profit / (p.volume * 100000) if p.volume > 0 else 0}
            for p in positions if p.magic == magic
        ]

    # ── Position sizing ──────────────────────────────────────────────────────

    def calc_lot_size(self, symbol: str, risk_usd: float,
                      sl_pips: float) -> float:
        """
        Calculate lot size so that SL hit = risk_usd loss.
        sl_pips: distance in price (not pips count).
        """
        info = mt5.symbol_info(symbol)
        if info is None or sl_pips <= 0:
            return info.volume_min if info else 0.01

        # pip value per lot: for most pairs 1 lot = 100,000 units
        # tick_value = profit per 1 tick move on 1 lot
        tick_value = info.trade_tick_value
        tick_size  = info.point
        if tick_size <= 0:
            return info.volume_min

        loss_per_lot = (sl_pips / tick_size) * tick_value
        if loss_per_lot <= 0:
            return info.volume_min

        raw_lots = risk_usd / loss_per_lot
        # Round to volume_step
        step = info.volume_step
        lots = round(raw_lots / step) * step
        lots = max(info.volume_min, min(info.volume_max, lots))
        return round(lots, 2)
