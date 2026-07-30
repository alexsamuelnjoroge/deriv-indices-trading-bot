"""
Base classes for pro_bot strategies.

A "bar" is a dict: {open, high, low, close, epoch, volume(optional)}
epoch is a Unix timestamp (seconds UTC).

Signal carries the direction + suggested SL/TP in price distance (not %).
The execution layer converts these to actual orders on whatever platform
(MT4/MT5, Binance, Bybit, IBKR, etc.).
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Signal:
    action:     str              # "BUY" | "SELL" | "HOLD"
    reason:     str  = ""
    sl_pips:    Optional[float] = None   # stop loss distance in pips/points
    tp_pips:    Optional[float] = None   # take profit distance in pips/points
    sl_price:   Optional[float] = None   # absolute SL price (alternative)
    tp_price:   Optional[float] = None   # absolute TP price (alternative)
    confidence: float = 1.0              # 0–1, used for position sizing
    meta:       dict  = field(default_factory=dict)


class BaseProStrategy:
    """
    All pro strategies implement this interface.

    feed(bar)        — add one completed bar; returns Signal
    evaluate(bars)   — bulk evaluate a list of bars; returns list of (index, Signal)
    on_result(won)   — feedback after trade closes
    reset()          — clear internal state
    """

    name: str = "base"

    # Best platform / instruments for this strategy (informational)
    best_platform:    str = ""
    best_instruments: list[str] = []

    def __init__(self, config: dict):
        self.config = config
        self._bars: list[dict] = []

    def feed(self, bar: dict) -> Signal:
        self._bars.append(bar)
        return self._evaluate()

    def evaluate(self, bars: list[dict]) -> list[tuple[int, Signal]]:
        """Bulk evaluate — returns (bar_index, Signal) for every non-HOLD signal."""
        results = []
        self._bars = []
        for i, bar in enumerate(bars):
            sig = self.feed(bar)
            if sig.action != "HOLD":
                results.append((i, sig))
        return results

    def on_result(self, won: bool) -> None:
        pass

    def reset(self) -> None:
        self._bars = []

    def _evaluate(self) -> Signal:
        raise NotImplementedError
