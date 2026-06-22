"""
Renders backtest results to the terminal using Rich.
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich import box

from src.backtest.engine import BacktestResult, BacktestTrade

console = Console()


def _profit_text(value: float) -> Text:
    color = "green" if value >= 0 else "red"
    return Text(f"{value:+.2f}", style=color)


def _pct_text(value: float, threshold: float) -> Text:
    color = "green" if value >= threshold else "red"
    return Text(f"{value:.1f}%", style=color)


def print_report(result: BacktestResult, strategy_cfg: dict, risk_cfg: dict) -> None:
    console.print()

    # ── Summary panel ──────────────────────────────────────────
    summary = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    summary.add_column(style="dim", width=22)
    summary.add_column()

    summary.add_row("Symbol", result.symbol)
    summary.add_row("Ticks analysed", f"{result.ticks_total:,}")
    summary.add_row("Starting balance", f"{result.starting_balance:.2f}")
    summary.add_row("Final balance", f"{result.final_balance:.2f}")
    summary.add_row("Net P&L", _profit_text(result.net_profit))

    # ── Trade stats panel ───────────────────────────────────────
    be = result.breakeven_win_rate
    stats = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    stats.add_column(style="dim", width=22)
    stats.add_column()

    stats.add_row("Total trades", str(result.total_trades))
    stats.add_row(
        "Wins / Losses",
        Text(f"{result.wins}", style="green") +
        Text(f" / ") +
        Text(f"{result.losses}", style="red"),
    )
    stats.add_row(
        "Win rate",
        _pct_text(result.win_rate, be),
    )
    stats.add_row(
        "Breakeven needed",
        Text(f"{be:.1f}%  (@ {result.payout_pct*100:.0f}% payout)", style="dim"),
    )
    stats.add_row("Best win streak", Text(str(result.best_win_streak), style="green"))
    stats.add_row("Worst loss streak", Text(str(result.worst_loss_streak), style="red"))
    stats.add_row("Avg stake", f"{result.avg_stake:.2f}")

    # ── Settings panel ──────────────────────────────────────────
    settings = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    settings.add_column(style="dim", width=22)
    settings.add_column()

    settings.add_row("RSI period", str(strategy_cfg.get("rsi_period", 14)))
    settings.add_row(
        "OB / OS thresholds",
        f"{strategy_cfg.get('rsi_overbought', 70)} / {strategy_cfg.get('rsi_oversold', 30)}",
    )
    settings.add_row("Confirm ticks", str(strategy_cfg.get("confirm_ticks", 3)))
    settings.add_row(
        "EMA trend filter",
        str(strategy_cfg.get("ema_trend_period", 50)) or "off",
    )
    ema_off = strategy_cfg.get("ema_trend_period", 50) == 0
    settings.add_row("EMA filter", Text("OFF", style="red") if ema_off else Text("ON", style="green"))
    bb_on = strategy_cfg.get("use_bb_filter", True)
    settings.add_row(
        "BB filter",
        Text("ON", style="green") if bb_on else Text("OFF", style="red"),
    )
    settings.add_row(
        "BB period / std",
        f"{strategy_cfg.get('bb_period', 20)} / {strategy_cfg.get('bb_std_dev', 2.0)}",
    )
    atr_on = risk_cfg.get("use_atr_stake", True)
    settings.add_row(
        "ATR stake sizing",
        Text("ON", style="green") if atr_on else Text("OFF", style="red"),
    )
    settings.add_row(
        "Contract duration",
        f"{strategy_cfg.get('contract_duration', 5)} {strategy_cfg.get('contract_duration_unit', 't')}",
    )

    console.print(Panel(
        Columns([summary, stats, settings]),
        title="[bold]Backtest Results[/bold]",
        border_style="blue",
    ))

    # ── Verdict ─────────────────────────────────────────────────
    if result.total_trades == 0:
        console.print(
            Panel(
                "[yellow]No trades fired. The filters may be too strict "
                "or there were not enough ticks for warmup. "
                "Try increasing --count, lowering confirm_ticks, or disabling bb_filter.[/yellow]",
                border_style="yellow",
                title="Verdict",
            )
        )
        return

    if result.win_rate >= be:
        msg = Text(
            f"Profitable: {result.win_rate:.1f}% win rate exceeds breakeven ({be:.1f}%). "
            f"Net: {result.net_profit:+.2f}",
            style="bold green",
        )
    else:
        gap = round(be - result.win_rate, 1)
        msg = Text(
            f"Not yet profitable: {result.win_rate:.1f}% win rate is {gap}% below breakeven ({be:.1f}%). "
            f"Try tuning parameters.",
            style="bold red",
        )

    console.print(Panel(msg, border_style="green" if result.win_rate >= be else "red", title="Verdict"))

    # ── Last 20 trades ──────────────────────────────────────────
    if result.trades:
        recent = result.trades[-20:]
        trade_table = Table(
            title=f"Last {len(recent)} trades",
            box=box.SIMPLE_HEAD,
            show_lines=False,
        )
        trade_table.add_column("Tick", style="dim", justify="right")
        trade_table.add_column("Type")
        trade_table.add_column("Entry", justify="right")
        trade_table.add_column("Exit", justify="right")
        trade_table.add_column("Stake", justify="right")
        trade_table.add_column("Profit", justify="right")
        trade_table.add_column("Reason", style="dim", max_width=50)

        for t in recent:
            color = "green" if t.status == "won" else "red"
            trade_table.add_row(
                str(t.index),
                Text(t.contract_type, style=color),
                f"{t.entry_price:.5f}",
                f"{t.exit_price:.5f}",
                f"{t.stake:.2f}",
                Text(f"{t.profit:+.2f}", style=color),
                t.reason,
            )

        console.print(trade_table)


def print_multi_window_report(
    results: list,
    strategy_cfg: dict,
    risk_cfg: dict,
) -> None:
    """Print per-window and aggregate results for a multi-window backtest."""
    console.print()
    be = results[0].breakeven_win_rate if results else 53.5

    # ── Per-window table ───────────────────────────────────────
    win_table = Table(
        title="Per-Window Results",
        box=box.SIMPLE_HEAD,
        show_lines=False,
    )
    win_table.add_column("Window", justify="right", style="dim")
    win_table.add_column("Ticks", justify="right")
    win_table.add_column("Trades", justify="right")
    win_table.add_column("Win Rate", justify="right")
    win_table.add_column("Net P&L", justify="right")
    win_table.add_column("Worst streak", justify="right")

    for i, r in enumerate(results, 1):
        color = "green" if r.win_rate >= be else "red"
        win_table.add_row(
            str(i),
            f"{r.ticks_total:,}",
            str(r.total_trades),
            Text(f"{r.win_rate:.1f}%", style=color),
            Text(f"{r.net_profit:+.2f}", style=color),
            str(r.worst_loss_streak),
        )

    console.print(win_table)

    # ── Aggregate stats ────────────────────────────────────────
    n          = len(results)
    all_trades = sum(r.total_trades for r in results)
    all_wins   = sum(r.wins for r in results)
    avg_wr     = round(all_wins / all_trades * 100, 1) if all_trades else 0
    total_pnl  = round(sum(r.net_profit for r in results), 2)
    min_wr     = min(r.win_rate for r in results)
    max_wr     = max(r.win_rate for r in results)
    profitable = sum(1 for r in results if r.win_rate >= be)

    agg = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    agg.add_column(style="dim", width=26)
    agg.add_column()
    agg.add_row("Windows tested", str(n))
    agg.add_row("Profitable windows", Text(f"{profitable}/{n}", style="green" if profitable == n else "yellow"))
    agg.add_row("Total trades", str(all_trades))
    agg.add_row("Aggregate win rate", _pct_text(avg_wr, be))
    agg.add_row("Win rate range", f"{min_wr:.1f}% – {max_wr:.1f}%")
    agg.add_row("Total net P&L", _profit_text(total_pnl))
    agg.add_row("Breakeven needed", Text(f"{be:.1f}%", style="dim"))

    verdict_color = "green" if profitable == n else ("yellow" if profitable > n // 2 else "red")
    verdict_text  = (
        "Robust: above breakeven in every window."
        if profitable == n else
        f"Inconsistent: profitable in {profitable}/{n} windows. Needs more tuning."
    )

    console.print(Panel(
        Columns([agg]),
        title="[bold]Multi-Window Aggregate[/bold]",
        border_style="blue",
    ))
    console.print(Panel(Text(verdict_text, style=f"bold {verdict_color}"),
                        border_style=verdict_color, title="Verdict"))
