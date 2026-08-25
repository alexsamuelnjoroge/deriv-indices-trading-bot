from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class Signal:
    action: str              # "BUY_RISE", "BUY_FALL", or "HOLD"
    reason: str              # human-readable explanation
    rsi: Optional[float] = None
    atr: Optional[float] = None           # current ATR at signal time
    atr_baseline: Optional[float] = None  # longer-period ATR for comparison
    contract_duration: Optional[int] = None  # override symbol default when RSI is extreme
    sl_pct: Optional[float] = None   # SL as fraction of price (multiplier contracts)
    tp_pct: Optional[float] = None   # TP as fraction of price (multiplier contracts)
    close_open_accus: bool = False   # spike detected — sell any open ACCU before it KOs


class BaseStrategy(ABC):
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def evaluate(self, tick_store) -> Signal:
        """Return a Signal based on current market data."""
        ...
