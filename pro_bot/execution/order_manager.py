"""
Order Manager — translates strategy Signals into MT5 orders.

Responsibilities:
  - Convert sl_pips / tp_pips (price distance) to absolute SL/TP prices
  - Size positions based on risk % of balance
  - Enforce max open trades limit
  - Track daily P&L and halt if daily loss limit exceeded
  - Log every trade result
  - Break-even: move SL to entry once trade reaches break_even_trigger_r × risk
  - Pyramid: open a scaled-in position once trade reaches pyramid_trigger_r × risk
  - Spread filter: skip entry if current spread exceeds max_spread_sl_ratio × SL
"""

import csv
import pathlib
import time
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional

from loguru import logger

from pro_bot.strategies.base import Signal


@dataclass
class OpenTrade:
    ticket:        int
    symbol:        str
    action:        str
    entry:         float
    sl:            float
    tp:            float
    sl_pips:       float
    risk_usd:      float
    strategy_name: str      = ""
    opened_at:     datetime = field(default_factory=datetime.now)
    be_done:       bool = False   # break-even SL already applied
    pyramid_done:  bool = False   # pyramid entry already opened
    is_pyramid:    bool = False   # this position IS a pyramid entry


class OrderManager:

    def __init__(self, client, config: dict):
        self.client           = client
        self.risk_pct         = config.get("max_risk_per_trade_pct", 1.0)
        self.max_open         = config.get("max_open_trades",         3)
        self.max_open_per_sym = config.get("max_open_per_symbol",     1)
        self.daily_loss_limit = config.get("max_daily_loss_pct",      5.0)

        # Break-even: move SL to entry after this many R profit (0 = disabled)
        self.be_trigger_r     = config.get("break_even_trigger_r",  0.0)
        # Pyramid: open scaled entry after this many R profit (0 = disabled)
        self.pyramid_trigger_r = config.get("pyramid_trigger_r",    0.0)
        self.pyramid_risk_pct  = config.get("pyramid_risk_pct",     0.5)
        # Spread filter: skip if spread > this ratio × SL distance (0 = disabled)
        self.max_spread_ratio  = config.get("max_spread_sl_ratio",  0.0)
        # Cooldown after a loss: block re-entry on the same symbol for this many seconds
        self.loss_cooldown_secs = config.get("loss_cooldown_secs", 600)

        # Drawdown-aware sizing: scale risk_pct down as equity draws from session peak
        dd_cfg = config.get("dd_sizing", {})
        self._dd_enabled    = dd_cfg.get("enabled", False)
        self._dd_thresholds = sorted(
            dd_cfg.get("thresholds", []),
            key=lambda x: x["drawdown_pct"]
        )
        self._peak_balance: float = 0.0

        self._open:     dict[int, OpenTrade] = {}   # ticket → trade
        self._today_pnl:  float = 0.0
        self._today_date: date  = date.today()
        self._halted:     bool  = False
        self._total_trades = 0
        self._wins         = 0
        self._losses       = 0
        self._loss_cooldown: dict[str, float] = {}  # symbol → cooldown-until timestamp

        pathlib.Path("logs").mkdir(exist_ok=True)
        self._pnl_log_path = pathlib.Path("logs/strategy_pnl.csv")

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

        # Count only non-pyramid entries toward the max_open limit
        open_count = sum(1 for t in self._open.values() if not t.is_pyramid)
        if open_count >= self.max_open:
            logger.info(f"[{symbol}] Max open trades ({self.max_open}) reached")
            return False

        # Per-symbol position cap — checked against both in-memory state and live
        # MT5 positions so restarts or concurrent instances don't open duplicates.
        live_positions = self.client.get_open_positions()
        sym_live  = sum(1 for p in live_positions if p["symbol"] == symbol)
        sym_mem   = sum(1 for t in self._open.values() if t.symbol == symbol)
        sym_count = max(sym_live, sym_mem)
        if sym_count >= self.max_open_per_sym:
            logger.info(f"[{symbol}] Symbol limit ({self.max_open_per_sym}) reached "
                        f"({sym_count} open)")
            return False

        # Cooldown: block re-entry after a SL hit to avoid revenge trading
        if self.loss_cooldown_secs > 0:
            until = self._loss_cooldown.get(symbol, 0)
            if time.time() < until:
                remaining = int(until - time.time())
                logger.info(f"[{symbol}] Loss cooldown — {remaining}s remaining before re-entry")
                return False

        # Spread filter: skip if spread is too wide relative to the SL
        if self.max_spread_ratio > 0:
            info = self.client.get_symbol_info(symbol)
            if info:
                spread_price = info["spread"] * info["point"]
                max_spread   = signal.sl_pips * self.max_spread_ratio
                if spread_price > max_spread:
                    logger.warning(
                        f"[{symbol}] Spread {spread_price:.5f} > max "
                        f"{max_spread:.5f} ({self.max_spread_ratio:.0%} of SL) — skipping"
                    )
                    return False

        # Size the position — apply drawdown scaling if enabled
        balance  = self.client.get_balance()
        if balance > self._peak_balance:
            self._peak_balance = balance
        risk_usd = balance * self._effective_risk_pct() / 100
        volume   = self.client.calc_lot_size(symbol, risk_usd, signal.sl_pips)

        if volume <= 0:
            logger.warning(f"[{symbol}] Invalid lot size — skipping")
            return False

        # Skip entry if minimum lot forces actual risk beyond 2× target.
        # Happens when ATR-based SL is wide and account is too small to size below min lot.
        info = self.client.get_symbol_info(symbol)
        if info and info.get("point", 0) > 0:
            actual_risk = (signal.sl_pips / info["point"]) * info["trade_tick_value"] * volume
            if actual_risk > risk_usd * 2:
                logger.info(
                    f"[{symbol}] Entry skipped | min-lot risk ${actual_risk:.2f} "
                    f"exceeds 2× target ${risk_usd:.2f} — account too small for this SL"
                )
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

        eff_pct = self._effective_risk_pct()
        self._open[result.ticket] = OpenTrade(
            ticket        = result.ticket,
            symbol        = symbol,
            action        = signal.action,
            entry         = price,
            sl            = sl_price,
            tp            = tp_price,
            sl_pips       = signal.sl_pips,
            risk_usd      = risk_usd,
            strategy_name = signal.strategy,
        )
        self._total_trades += 1
        dd_note = (f" [risk scaled {eff_pct:.2f}%]"
                   if self._dd_enabled and eff_pct != self.risk_pct else "")
        logger.info(f"Trade opened | {symbol} {signal.action} {volume}L | "
                    f"entry={price:.5f} SL={sl_price:.5f} TP={tp_price:.5f} | "
                    f"risk=${risk_usd:.2f}{dd_note} | [{signal.strategy}] {signal.reason}")
        return True

    # ── Position management (break-even + pyramid) ───────────────────────────

    def manage_positions(self) -> None:
        """
        Call every monitor cycle. Checks open trades for:
          - Break-even: move SL to entry once trade reaches be_trigger_r profit
          - Pyramid: open a scaled entry once trade reaches pyramid_trigger_r profit
        """
        if not self._open:
            return

        live = {p["ticket"]: p for p in self.client.get_open_positions()}

        for ticket, trade in list(self._open.items()):
            if ticket not in live:
                continue

            # Profit in R units based on floating P&L vs initial risk
            profit_r = (live[ticket]["profit"] / trade.risk_usd
                        if trade.risk_usd > 0 else 0.0)

            # Break-even
            if (not trade.be_done and not trade.is_pyramid
                    and self.be_trigger_r > 0
                    and profit_r >= self.be_trigger_r):
                if self.client.modify_sl(ticket, trade.entry):
                    trade.sl     = trade.entry
                    trade.be_done = True
                    logger.info(
                        f"Break-even | {trade.symbol} ticket={ticket} "
                        f"SL → entry {trade.entry:.5f} (+{profit_r:.2f}R)"
                    )

            # Pyramid
            if (not trade.pyramid_done and not trade.is_pyramid
                    and self.pyramid_trigger_r > 0
                    and profit_r >= self.pyramid_trigger_r):
                self._open_pyramid(trade)

    def _open_pyramid(self, parent: OpenTrade) -> None:
        """
        Open a second position in the same direction when the parent trade
        has moved pyramid_trigger_r in profit.

        Entry:  current market price
        SL:     parent's entry price (guaranteed scratch on the parent)
        TP:     entry + same RR ratio as parent
        Risk:   pyramid_risk_pct% of current balance
        """
        tick = self.client.get_tick(parent.symbol)
        if not tick:
            return

        current = tick["ask"] if parent.action == "BUY" else tick["bid"]

        if parent.action == "BUY":
            sl_price = parent.entry
            sl_dist  = current - sl_price
            rr       = (parent.tp - parent.entry) / parent.sl_pips
            tp_price = current + sl_dist * rr
        else:
            sl_price = parent.entry
            sl_dist  = sl_price - current
            rr       = (parent.entry - parent.tp) / parent.sl_pips
            tp_price = current - sl_dist * rr

        if sl_dist <= 0:
            logger.debug(f"Pyramid skipped | {parent.symbol} price retreated to entry")
            parent.pyramid_done = True   # don't keep retrying this cycle
            return

        balance  = self.client.get_balance()
        risk_usd = balance * self.pyramid_risk_pct / 100
        volume   = self.client.calc_lot_size(parent.symbol, risk_usd, sl_dist)

        if volume <= 0:
            return

        # Skip pyramid if minimum lot forces actual risk beyond 2× the target.
        # This happens when sl_dist is very small (e.g. only 2-3 points from entry).
        info = self.client.get_symbol_info(parent.symbol)
        if info and info.get("point", 0) > 0:
            actual_risk = (sl_dist / info["point"]) * info["trade_tick_value"] * volume
            if actual_risk > risk_usd * 2:
                logger.info(
                    f"Pyramid skipped | {parent.symbol} min-lot risk ${actual_risk:.2f} "
                    f"exceeds 2× target ${risk_usd:.2f} — SL too tight for min lot"
                )
                parent.pyramid_done = True
                return

        result = self.client.place_order(
            symbol   = parent.symbol,
            action   = parent.action,
            volume   = volume,
            sl_price = sl_price,
            tp_price = tp_price,
            comment  = "pro_bot|pyramid",
        )

        if result.success:
            self._open[result.ticket] = OpenTrade(
                ticket       = result.ticket,
                symbol       = parent.symbol,
                action       = parent.action,
                entry        = current,
                sl           = sl_price,
                tp           = tp_price,
                sl_pips      = sl_dist,
                risk_usd     = risk_usd,
                is_pyramid   = True,
                be_done      = True,  # SL already at parent entry, no further BE needed
            )
            parent.pyramid_done = True
            self._total_trades += 1
            logger.info(
                f"Pyramid opened | {parent.symbol} {parent.action} {volume}L | "
                f"entry={current:.5f} SL={sl_price:.5f} TP={tp_price:.5f} | "
                f"risk=${risk_usd:.2f} RR={rr:.1f}"
            )

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
            pnl = self._get_trade_pnl(ticket, trade)
            self._today_pnl += pnl

            r_outcome = round(pnl / trade.risk_usd, 3) if trade.risk_usd > 0 else 0.0
            won = pnl > 0
            if won:
                self._wins += 1
                self._loss_cooldown.pop(trade.symbol, None)  # win clears any cooldown
            else:
                self._losses += 1
                if self.loss_cooldown_secs > 0 and not trade.is_pyramid:
                    self._loss_cooldown[trade.symbol] = time.time() + self.loss_cooldown_secs
                    logger.info(f"[{trade.symbol}] Loss cooldown started — "
                                f"re-entry blocked for {self.loss_cooldown_secs}s")

            tag = "PYRAMID" if trade.is_pyramid else "TRADE"
            logger.info(f"{tag} closed | [{trade.strategy_name}] {trade.symbol} {trade.action} | "
                        f"P&L ${pnl:+.2f} ({r_outcome:+.2f}R) | {'WIN' if won else 'LOSS'} | "
                        f"Today {self._today_pnl:+.2f}")
            self._write_pnl_log(trade, pnl, r_outcome)

            # Update peak balance after a close (equity may have changed)
            balance = self.client.get_balance()
            if balance > self._peak_balance:
                self._peak_balance = balance

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

    # ── Drawdown-aware sizing ────────────────────────────────────────────────

    def _effective_risk_pct(self) -> float:
        """Return risk_pct scaled down by the current equity drawdown from session peak."""
        if not self._dd_enabled or not self._dd_thresholds or self._peak_balance <= 0:
            return self.risk_pct
        current = self.client.get_balance()
        dd_pct = (self._peak_balance - current) / self._peak_balance * 100
        scale = 1.0
        for threshold in self._dd_thresholds:
            if dd_pct >= threshold["drawdown_pct"]:
                scale = threshold["risk_scale"]
        return self.risk_pct * scale

    # ── Strategy P&L log ─────────────────────────────────────────────────────

    def _write_pnl_log(self, trade: OpenTrade, pnl: float, r_outcome: float) -> None:
        """Append one row to the strategy P&L CSV for post-session analysis."""
        new_file = not self._pnl_log_path.exists()
        try:
            with self._pnl_log_path.open("a", newline="") as f:
                w = csv.writer(f)
                if new_file:
                    w.writerow(["closed_at", "strategy", "symbol", "direction",
                                "r_outcome", "pnl_usd", "is_pyramid"])
                w.writerow([
                    datetime.now().isoformat(timespec="seconds"),
                    trade.strategy_name,
                    trade.symbol,
                    trade.action,
                    r_outcome,
                    round(pnl, 2),
                    trade.is_pyramid,
                ])
        except Exception as e:
            logger.warning(f"P&L log write failed: {e}")

    # ── Status ───────────────────────────────────────────────────────────────

    @property
    def is_halted(self) -> bool:
        return self._halted

    def status(self) -> dict:
        base_count    = sum(1 for t in self._open.values() if not t.is_pyramid)
        pyramid_count = sum(1 for t in self._open.values() if t.is_pyramid)
        return {
            "open_trades":    base_count,
            "open_pyramids":  pyramid_count,
            "today_pnl":      self._today_pnl,
            "total_trades":   self._total_trades,
            "wins":           self._wins,
            "losses":         self._losses,
            "win_rate":       round(self._wins / max(1, self._wins + self._losses) * 100, 1),
            "halted":         self._halted,
        }
