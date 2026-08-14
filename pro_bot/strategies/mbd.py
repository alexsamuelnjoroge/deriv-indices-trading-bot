"""Multi-Bar Divergence — large displacement over L bars but velocity stalled over S bars → fade."""
from .research_base import ResearchDailyStrategy
from .base import Signal


class MBDStrategy(ResearchDailyStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.long_window   = config.get("long_window", 6)
        self.short_window  = config.get("short_window", 2)
        self.min_long_atr  = config.get("min_long_atr", 2.0)
        self.max_short_atr = config.get("max_short_atr", 0.2)

    def _check_signal(self, atr, allow_long, allow_short):
        bars = list(self._h1)
        L, S = self.long_window, self.short_window
        if len(bars) < L + 1:
            return None

        close   = bars[-1]["close"]
        close_L = bars[-(L + 1)]["close"]
        close_S = bars[-(S + 1)]["close"]

        long_ret  = (close - close_L) / atr   # big displacement over L bars
        short_ret = (close - close_S) / atr   # velocity over last S bars

        sl, tp = self._sl_tp(atr)

        # Strong up-move but momentum stalled → fade sell
        if long_ret >= self.min_long_atr and abs(short_ret) <= self.max_short_atr and allow_short:
            return Signal("SELL", sl_pips=sl, tp_pips=tp,
                          reason=f"MBD sell long={long_ret:.2f}ATR short={short_ret:.3f}ATR")

        # Strong down-move but momentum stalled → fade buy
        if long_ret <= -self.min_long_atr and abs(short_ret) <= self.max_short_atr and allow_long:
            return Signal("BUY", sl_pips=sl, tp_pips=tp,
                          reason=f"MBD buy long={long_ret:.2f}ATR short={short_ret:.3f}ATR")

        return None
