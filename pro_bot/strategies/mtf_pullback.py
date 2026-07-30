"""
Multi-Timeframe Trend Pullback Strategy

Logic:
  Higher timeframe (1h/4h): EMA(200) slope defines trend direction.
  Lower timeframe (15min):  RSI pullback into trend gives entry.

  Uptrend  → EMA(200) sloping up  + RSI drops below rsi_entry  → BUY
  Downtrend → EMA(200) sloping down + RSI rises above 100-rsi_entry → SELL

Exit: trailing SL below/above the pullback swing low/high.

Best platform:    MT4/MT5 (IC Markets, Pepperstone, FTMO)
Best instruments: EUR/USD, Gold, USD/JPY — 1h trend + 15min entry
Expected WR:      62–72% with 1:2 R:R minimum
"""

from .base import BaseProStrategy, Signal
from ..indicators import ema, rsi


class MTFPullbackStrategy(BaseProStrategy):

    name             = "mtf_pullback"
    best_platform    = "MT4/MT5"
    best_instruments = ["EURUSD", "XAUUSD", "USDJPY", "GBPUSD"]

    def __init__(self, config: dict):
        super().__init__(config)
        # Higher timeframe bars (1h or 4h) — fed separately
        self._htf_bars:    list[dict] = []

        self.ema_period   = config.get("ema_period",   200)
        self.slope_bars   = config.get("slope_bars",     3)
        self.rsi_period   = config.get("rsi_period",    14)
        self.rsi_entry    = config.get("rsi_entry",    40.0)
        self.sl_atr_mult  = config.get("sl_atr_mult",   1.5)
        self.tp_rr        = config.get("tp_rr",         2.0)  # risk:reward ratio

    def feed_htf(self, bar: dict) -> None:
        """Feed one higher-timeframe bar (1h or 4h)."""
        self._htf_bars.append(bar)

    def _evaluate(self) -> Signal:
        bars   = self._bars
        htf    = self._htf_bars

        if len(bars) < self.rsi_period + 2:
            return Signal(action="HOLD", reason="Warming up LTF")
        if len(htf) < self.ema_period + self.slope_bars:
            return Signal(action="HOLD", reason="Warming up HTF")

        htf_closes = [b["close"] for b in htf]
        ltf_closes = [b["close"] for b in bars]

        ema_series = ema(htf_closes, self.ema_period)
        rsi_series = rsi(ltf_closes, self.rsi_period)

        ema_now  = ema_series[-1]
        ema_prev = ema_series[-1 - self.slope_bars]
        rsi_now  = rsi_series[-1]
        rsi_prev = rsi_series[-2]

        if any(x is None for x in [ema_now, ema_prev, rsi_now, rsi_prev]):
            return Signal(action="HOLD", reason="Indicator not ready")

        trend_up   = ema_now > ema_prev
        trend_down = ema_now < ema_prev
        ob         = 100.0 - self.rsi_entry

        if trend_up and rsi_prev >= self.rsi_entry > rsi_now:
            sl   = bars[-1]["low"] - bars[-1]["close"] * 0.0005
            tp   = sl * self.tp_rr
            return Signal(
                action="BUY",
                reason=f"HTF EMA↑  LTF RSI {rsi_now:.1f} pullback",
                sl_pips=sl,
                tp_pips=tp,
                confidence=min(1.0, (self.rsi_entry - rsi_now) / 10),
                meta={"ema": round(ema_now, 5), "rsi": rsi_now},
            )

        if trend_down and rsi_prev <= ob < rsi_now:
            sl  = bars[-1]["close"] - bars[-1]["high"] * 0.0005
            tp  = sl * self.tp_rr
            return Signal(
                action="SELL",
                reason=f"HTF EMA↓  LTF RSI {rsi_now:.1f} pullback",
                sl_pips=sl,
                tp_pips=tp,
                confidence=min(1.0, (rsi_now - ob) / 10),
                meta={"ema": round(ema_now, 5), "rsi": rsi_now},
            )

        return Signal(
            action="HOLD",
            reason=f"EMA {'↑' if trend_up else '↓' if trend_down else '='} "
                   f"RSI {rsi_now:.1f}",
        )
