"""
Multi-Timeframe Trend Pullback Strategy — v5

Backtest-proven configs (v5, spread applied, min 100 trades, macro filter):
  frxXAUUSD: EMA100 RSI<35 RR1.5 ADX25 SESS_PEAK MACRO20  EV+0.280R WR 51.2%
  frxEURUSD: EMA200 RSI<35 RR2.0 ADX25 MACRO50            EV+0.388R WR 46.3%
  frxUSDJPY: EMA100 RSI<35 RR2.0 ADX20 ATR1.5 MACRO20     EV+0.305R WR 43.5%

Logic (5min entry, 1h trend, daily macro gate):
  1h EMA slope  -> trend direction
  5min RSI pullback crossing rsi_entry -> entry trigger
  ADX(14)       -> skip low-momentum bars
  ATR(14)*mult  -> dynamic SL distance (atr_mult_sl > 0)
  Daily EMA slope -> macro direction gate (macro_filter: true)
  Session window  -> peak London/NY hours only (session_peak: true)
"""

from datetime import datetime, timezone

from .base import BaseProStrategy, Signal
from ..indicators import ema, rsi, adx as _adx, atr as _atr


class MTFPullbackStrategy(BaseProStrategy):

    name             = "mtf_pullback"
    best_platform    = "MT4/MT5"
    best_instruments = ["frxEURUSD", "frxXAUUSD", "frxUSDJPY"]

    def __init__(self, config: dict):
        super().__init__(config)
        self._htf_bars:   list[dict] = []
        self._daily_bars: list[dict] = []

        self.ema_period       = config.get("ema_period",        100)
        self.slope_bars       = config.get("slope_bars",           3)
        self.rsi_period       = config.get("rsi_period",          14)
        self.rsi_entry        = config.get("rsi_entry",          35.0)
        self.tp_rr            = config.get("tp_rr",               1.5)
        self.adx_min          = config.get("adx_min",               0)
        self.session_only     = config.get("session_only",       False)
        self.session_peak     = config.get("session_peak",       False)
        self.atr_mult_sl      = config.get("atr_mult_sl",         0.0)
        self.macro_filter     = config.get("macro_filter",       False)
        self.macro_ema_period = config.get("macro_ema_period",     20)

    def feed_htf(self, bar: dict) -> None:
        self._htf_bars.append(bar)

    def feed_daily(self, bar: dict) -> None:
        self._daily_bars.append(bar)

    def _in_session(self, bar: dict) -> bool:
        dt = datetime.fromtimestamp(bar.get("epoch", 0), tz=timezone.utc)
        h  = dt.hour + dt.minute / 60.0
        if self.session_peak:
            return (7.0 <= h < 10.5) or (13.0 <= h < 16.5)
        if self.session_only:
            return 7.0 <= h < 20.0
        return True

    def _evaluate(self) -> Signal:
        bars = self._bars
        htf  = self._htf_bars

        if len(bars) < max(self.rsi_period, 14) + 2:
            return Signal(action="HOLD", reason="Warming up LTF")
        if len(htf) < self.ema_period + self.slope_bars:
            return Signal(action="HOLD", reason="Warming up HTF")

        cur = bars[-1]
        if not self._in_session(cur):
            return Signal(action="HOLD", reason="Outside session")

        htf_closes = [b["close"] for b in htf]
        ltf_closes = [b["close"] for b in bars]
        highs      = [b["high"]  for b in bars]
        lows       = [b["low"]   for b in bars]

        ema_vals = ema(htf_closes, self.ema_period)
        ema_now  = ema_vals[-1]
        ema_prev = ema_vals[-1 - self.slope_bars]

        rsi_vals = rsi(ltf_closes, self.rsi_period)
        rsi_now  = rsi_vals[-1]
        rsi_prev = rsi_vals[-2]

        if any(x is None for x in [ema_now, ema_prev, rsi_now, rsi_prev]):
            return Signal(action="HOLD", reason="Indicator not ready")

        if self.adx_min > 0:
            adx_vals = _adx(bars, 14)
            adx_val  = adx_vals[-1]
            if adx_val is None or adx_val < self.adx_min:
                return Signal(action="HOLD", reason=f"ADX {adx_val} < {self.adx_min}")

        if self.atr_mult_sl > 0:
            atr_vals = _atr(bars, 14)
            atr_val  = atr_vals[-1]
            sl_dist  = (atr_val * self.atr_mult_sl) if atr_val else (cur["close"] * 0.001)
        else:
            sl_dist = cur["close"] * 0.001

        allow_long = allow_short = True
        if self.macro_filter and len(self._daily_bars) >= self.macro_ema_period + 1:
            d_ema = ema([b["close"] for b in self._daily_bars], self.macro_ema_period)
            if d_ema[-1] is not None and d_ema[-2] is not None:
                macro_up    = d_ema[-1] > d_ema[-2]
                allow_long  = macro_up
                allow_short = not macro_up

        trend_up   = ema_now > ema_prev
        trend_down = ema_now < ema_prev
        ob         = 100.0 - self.rsi_entry
        tp_dist    = sl_dist * self.tp_rr

        if trend_up and rsi_prev >= self.rsi_entry > rsi_now and allow_long:
            return Signal(
                action="BUY",
                reason=f"EMA up | RSI {rsi_now:.1f} cross | macro {'up' if self.macro_filter else 'off'}",
                sl_pips=sl_dist,
                tp_pips=tp_dist,
                confidence=min(1.0, (self.rsi_entry - rsi_now) / 10),
                meta={"ema": round(ema_now, 5), "rsi": round(rsi_now, 1)},
            )

        if trend_down and rsi_prev <= ob < rsi_now and allow_short:
            return Signal(
                action="SELL",
                reason=f"EMA down | RSI {rsi_now:.1f} cross | macro {'down' if self.macro_filter else 'off'}",
                sl_pips=sl_dist,
                tp_pips=tp_dist,
                confidence=min(1.0, (rsi_now - ob) / 10),
                meta={"ema": round(ema_now, 5), "rsi": round(rsi_now, 1)},
            )

        return Signal(
            action="HOLD",
            reason=f"EMA {'up' if trend_up else 'down' if trend_down else 'flat'} | RSI {rsi_now:.1f}",
        )
