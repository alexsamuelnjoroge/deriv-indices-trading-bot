"""Cross-Symbol Divergence — lead symbol moves significantly while lag symbol stays flat → lag catches up."""
from collections import deque
from typing import Optional

from .research_base import ResearchDailyStrategy
from .base import Signal


class CSDStrategy(ResearchDailyStrategy):
    """
    Trades the LAG symbol (config['symbols'][0]).
    The LEAD symbol is config['lead_symbol'] — its bars are fed via feed_lead().
    Signal fires on lag bar close when lead has diverged from lag.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.lead_symbol   = config.get("lead_symbol", "EURUSD")
        self.window_bars   = config.get("window_bars", 6)
        self.min_lead_atr  = config.get("min_lead_atr", 1.5)
        self.max_lag_atr   = config.get("max_lag_atr", 0.5)
        self.session       = config.get("session", "london")  # "london" | "ny" | "both"

        self._h1_lead: deque = deque(maxlen=300)
        self._lead_seeded    = False

    def feed_lead(self, bar: dict) -> None:
        self._h1_lead.append(bar)

    def seed_lead(self, bars: list) -> None:
        for bar in bars:
            self._h1_lead.append(bar)
        self._lead_seeded = True

    def _lead_atr14(self) -> Optional[float]:
        bars = list(self._h1_lead)
        if len(bars) < 15:
            return None
        trs = []
        for i in range(max(1, len(bars) - 14), len(bars)):
            h, l, cp = bars[i]["high"], bars[i]["low"], bars[i - 1]["close"]
            trs.append(max(h - l, abs(h - cp), abs(l - cp)))
        return sum(trs) / len(trs) if trs else None

    def _check_signal(self, atr, allow_long, allow_short) -> Optional[Signal]:
        lag_bars  = list(self._h1)
        lead_bars = list(self._h1_lead)

        W = self.window_bars
        if len(lag_bars) < W + 1 or len(lead_bars) < W + 1:
            return None

        lag_bar = lag_bars[-1]
        epoch   = lag_bar.get("epoch", 0)
        h       = (epoch % 86400) // 3600

        in_london = 7 <= h <= 16
        in_ny     = 13 <= h <= 21

        if self.session == "london" and not in_london:
            return None
        if self.session == "ny" and not in_ny:
            return None
        if self.session == "both" and not (in_london or in_ny):
            return None

        # Find lead bar closest to current epoch (lead and lag bars are same-epoch 1H)
        # Use the last lead bar that's <= current epoch
        lead_bar     = lead_bars[-1]
        lead_atr_val = self._lead_atr14()
        if lead_atr_val is None or lead_atr_val <= 0:
            return None

        # ATR-normalized N-bar moves
        lead_ret = (lead_bar["close"] - lead_bars[-(W + 1)]["close"]) / lead_atr_val
        lag_ret  = (lag_bar["close"]  - lag_bars[-(W + 1)]["close"])  / atr

        sl, tp = self._sl_tp(atr)

        if lead_ret >= self.min_lead_atr and abs(lag_ret) <= self.max_lag_atr:
            if allow_long:
                return Signal("BUY", sl_pips=sl, tp_pips=tp,
                              reason=f"CSD buy lead={lead_ret:.2f}ATR lag={lag_ret:.2f}ATR")

        if lead_ret <= -self.min_lead_atr and abs(lag_ret) <= self.max_lag_atr:
            if allow_short:
                return Signal("SELL", sl_pips=sl, tp_pips=tp,
                              reason=f"CSD sell lead={lead_ret:.2f}ATR lag={lag_ret:.2f}ATR")

        return None
