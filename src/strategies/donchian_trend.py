"""
Daily Donchian trend-following (Turtle-style), evaluated on completed daily bars.

Entry: close breaks the highest high / lowest low of the previous `entry` bars.
Exit:  close breaks the lowest low / highest high of the previous `exit` bars.

Validated in pro_bot/backtest_trend_portfolio.py and robustness_trend_cmc.py on
XAU, XAG, BTC, ETH (2013-2026, Deriv x100 stake-out cap modelled).
"""


def entry_signal(bars: list[dict], entry: int) -> int:
    """+1 long breakout, -1 short breakout, 0 none. bars = completed, oldest first."""
    if len(bars) < entry + 1:
        return 0
    prev = bars[-entry - 1:-1]
    close = bars[-1]["close"]
    if close > max(b["high"] for b in prev):
        return 1
    if close < min(b["low"] for b in prev):
        return -1
    return 0


def exit_signal(bars: list[dict], exit_: int, direction: int) -> bool:
    """True if the open position in `direction` should be closed on this bar."""
    if len(bars) < exit_ + 1:
        return False
    prev = bars[-exit_ - 1:-1]
    close = bars[-1]["close"]
    if direction == 1:
        return close < min(b["low"] for b in prev)
    return close > max(b["high"] for b in prev)
