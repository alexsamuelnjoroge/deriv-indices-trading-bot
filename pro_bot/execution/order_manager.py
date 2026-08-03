"""
Order Manager — translates strategy Signals into MT5 orders.

Responsibilities:
  - Convert sl_pips / tp_pips (price distance) to absolute SL/TP prices
  - Size positions based on risk % of balance
  - Enforce max open trades limit
  - Track daily P&L and halt if daily loss limit exceeded
  - Log every trade result
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional

from loguru import logger

from pro_bot.strategies.base import Signal


@dataclass
class OpenTrade:
    ticket:    int
    symbol:    str
    action:    str
    entry:     float
    sl:        float
    tp:        float
    sl_pips:   float
    risk_usd:  float
    opened_at: datetime = field(default_factory=datetime.now)


class OrderManager:

    def __init__(self, client, config: dict):
        self.client           = client
        self.risk_pct         = config.get("max_risk_per_trade_pct", 1.0)
        self.max_open         = config.get("max_open_trades",         3)
        self.daily_loss_limit = config.get("max_daily_loss_pct",      5.0)

        self._open:     dict[int, OpenTrade] = {}   # ticket → trade
        self._today_pnl:  float = 0.0
        self._today_date: date  = date.today()
        self._halted:     bool  = False
        self._total_trades = 0
        self._wins         = 0
        self._losses       = 0

    # ── Daily reset ──────────────────────────────────────────────────────────

    def _check_day_reset(self) -> None:
        if date.today() != self._today_date:
            logger.info(f"Daily reset | P&L {self._today_pnl:+.2f} | "
                        f"Trades {self._total_trades} | "
                        f"W/L {self._wins}/{self._losses}")
            self._today_pnl  = 0.0
            self._today_date = date.today()
            self._halted     = False

    # ── Signal → order ───────────────────────────────────────────────────────

    def on_signal(self, symbol: str, signal: Signal,
                  htf_bars: Optional[list] = None) -> bool:
        """
        Called by the bot loop when a strategy fires a non-HOLD signal.
        Returns True if order was placed.
        """
        self._check_day_reset()

        if self._halted:
            logger.warning(f"[{symbol}] Halted — skipping signal")
            return False

        if signal.action not in ("BUY", "SELL"):
            return False

        if signal.sl_pips is None or signal.sl_pips <= 0:
            logger.warning(f"[{symbol}] Signal missing sl_pips — skipping")
            return False

        if len(self._open) >= self.max_open:
            logger.info(f"[{symbol}] Max open trades ({self.max_open}) reached")
            return False

        # Check we don't already have a position on this symbol
        open_symbols = {t.symbol for t in self._open.values()}
        if symbol in open_symbols:
            logger.info(f"[{symbol}] Already have open position — skipping")
            return False

        # Size the position
        balance  = self.client.get_balance()
        risk_usd = balance * self.risk_pct / 100
        volume   = self.client.calc_lot_size(symbol, risk_usd, signal.sl_pips)

        if volume <= 0:
            logger.warning(f"[{symbol}] Invalid lot size — skipping")
            return False

        # Get current price for absolute SL/TP
        tick = self.client.get_tick(symbol)
        if tick is None:
            logger.warning(f"[{symbol}] No tick — skipping")
            return False

        price = tick["ask"] if signal.action == "BUY" else tick["bid"]

        if signal.action == "BUY":
            sl_price = price - signal.sl_pips
            tp_price = price + signal.tp_pips
        else:
            sl_price = price + signal.sl_pips
            tp_price = price - signal.tp_pips

        result = self.client.place_order(
            symbol   = symbol,
            action   = signal.action,
            volume   = volume,
            sl_price = sl_price,
            tp_price = tp_price,
            comment  = f"pro_bot|{signal.reason[:20]}",
        )

        if not result.success:
            return False

        self._open[result.ticket] = OpenTrade(
            ticket   = result.ticket,
            symbol   = symbol,
            action   = signal.action,
            entry    = price,
            sl       = sl_price,
            tp       = tp_price,
            sl_pips  = signal.sl_pips,
            risk_usd = risk_usd,
        )
        self._total_trades += 1
        logger.info(f"Trade opened | {symbol} {signal.action} {volume}L | "
                    f"entry={price:.5f} SL={sl_price:.5f} TP={tp_price:.5f} | "
                    f"risk=${risk_usd:.2f} | {signal.reason}")
        return True

    # ── Position monitoring ──────────────────────────────────────────────────

    def sync_positions(self) -> None:
        """
        Call this periodically to detect closed positions (SL/TP hit).
        Compares tracked open tickets against MT5's actual open positions.
        """
        self._check_day_reset()
        live_tickets = {p["ticket"] for p in self.client.get_open_positions()}
        closed = [t for t in list(self._open.keys()) if t not in live_tickets]

        for ticket in closed:
            trade = self._open.pop(ticket)
            # Estimate result from MT5 history (simplified)
            pnl = self._get_trade_pnl(ticket, trade)
            self._today_pnl += pnl

            won = pnl > 0
            if won:
                self._wins += 1
            else:
                self._losses += 1

            logger.info(f"Trade closed | {trade.symbol} {trade.action} | "
                        f"P&L ${pnl:+.2f} | {'WIN' if won else 'LOSS'} | "
                        f"Today {self._today_pnl:+.2f}")

            # Check daily loss limit
            balance = self.client.get_balance()
            if balance > 0:
                loss_pct = abs(min(0, self._today_pnl)) / balance * 100
                if loss_pct >= self.daily_loss_limit:
                    self._halted = True
                    logger.warning(f"DAILY LOSS LIMIT HIT ({loss_pct:.1f}%) — "
                                   "bot halted until tomorrow")

    def _get_trade_pnl(self, ticket: int, trade: OpenTrade) -> float:
        """Estimate P&L from MT5 deal history for the ticket."""
        try:
            import MetaTrader5 as mt5
            deals = mt5.history_deals_get(position=ticket)
            if deals:
                return sum(d.profit for d in deals)
        except Exception:
            pass
        return 0.0

    # ── Status ───────────────────────────────────────────────────────────────

    @property
    def is_halted(self) -> bool:
        return self._halted

    def status(self) -> dict:
        return {
            "open_trades":    len(self._open),
            "today_pnl":      self._today_pnl,
            "total_trades":   self._total_trades,
            "wins":           self._wins,
            "losses":         self._losses,
            "win_rate":       round(self._wins / max(1, self._wins + self._losses) * 100, 1),
            "halted":         self._halted,
        }
