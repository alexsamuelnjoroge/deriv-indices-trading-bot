"""
Live terminal dashboard using Rich.
Updates every second showing price, RSI, trend, ATR, trade stats, and balance.
"""

from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich import box


console = Console()


def _rsi_color(rsi: float | None) -> str:
    if rsi is None:
        return "white"
    if rsi >= 70:
        return "red"
    if rsi <= 30:
        return "green"
    return "yellow"


def _trend_text(tick_store, ema_period: int) -> Text:
    price = tick_store.latest_price
    ema = tick_store.ema(ema_period) if ema_period > 0 else None
    if price is None or ema is None:
        return Text("---", style="dim")
    if price > ema:
        return Text(f"UP  (EMA{ema_period}: {ema:.5f})", style="green")
    return Text(f"DOWN (EMA{ema_period}: {ema:.5f})", style="red")


def build_dashboard(tick_store, risk, signal, ema_period: int = 50) -> Panel:
    now = datetime.now().strftime("%H:%M:%S")
    price = tick_store.latest_price
    rsi = tick_store.rsi()
    atr = tick_store.atr(14)

    # ── Price & Signal ──────────────────────────────────────────
    price_str = f"{price:.5f}" if price else "---"
    rsi_str = f"{rsi:.1f}" if rsi is not None else "warming up..."
    rsi_color = _rsi_color(rsi)
    atr_str = f"{atr:.6f}" if atr is not None else "---"

    action = signal.action if signal else "HOLD"
    action_color = {"BUY_RISE": "green", "BUY_FALL": "red", "HOLD": "white"}.get(action, "white")
    reason = signal.reason if signal else ""

    market_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    market_table.add_column(style="dim")
    market_table.add_column()
    market_table.add_row("Price", price_str)
    market_table.add_row("RSI(14)", Text(rsi_str, style=rsi_color))
    market_table.add_row("Trend", _trend_text(tick_store, ema_period))
    market_table.add_row("ATR(14)", Text(atr_str, style="cyan"))
    market_table.add_row("Ticks", str(tick_store.tick_count))
    market_table.add_row("Signal", Text(action, style=action_color))
    market_table.add_row("Reason", Text(reason, style="dim"))

    # ── Account & PnL ───────────────────────────────────────────
    balance_color = "green" if risk.net_profit >= 0 else "red"
    pnl_color = "green" if risk.daily_pnl >= 0 else "red"

    account_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    account_table.add_column(style="dim")
    account_table.add_column()
    account_table.add_row("Balance", Text(f"{risk.current_balance:.2f}", style=balance_color))
    account_table.add_row("Today P&L", Text(f"{risk.daily_pnl:+.2f}", style=pnl_color))
    account_table.add_row("Net P&L", Text(f"{risk.net_profit:+.2f}", style=balance_color))
    account_table.add_row("Trades", str(risk.total_trades))
    account_table.add_row("Win Rate", f"{risk.win_rate}%")
    account_table.add_row("Open", str(risk.open_contracts))

    halted_msg = Text(" [HALTED]", style="bold red") if risk.is_halted else Text("")

    return Panel(
        Columns([market_table, account_table]),
        title=Text(f"[bold]Deriv Trading Bot[/bold]  {now}") + halted_msg,
        border_style="blue",
    )


class Dashboard:
    def __init__(self, tick_store, risk, ema_period: int = 50):
        self.tick_store = tick_store
        self.risk = risk
        self.ema_period = ema_period
        self._last_signal = None
        self._live: Live | None = None

    def update_signal(self, signal):
        self._last_signal = signal

    def start(self):
        self._live = Live(
            build_dashboard(self.tick_store, self.risk, self._last_signal, self.ema_period),
            refresh_per_second=1,
            console=console,
        )
        self._live.start()

    def refresh(self):
        if self._live:
            self._live.update(
                build_dashboard(self.tick_store, self.risk, self._last_signal, self.ema_period)
            )

    def stop(self):
        if self._live:
            self._live.stop()
