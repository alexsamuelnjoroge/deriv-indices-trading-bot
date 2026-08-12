"""
Bollinger Band + RSI Pullback Strategy

Backtest result (365-day, holdout 20%):
  frxEURUSD: EMA100 BB(20,2.0) RSI<55 RR1.5 SESS_PEAK MACRO20
             Train EV +0.299R WR 55.0%  |  Holdout EV +0.186R WR 49.3%

Logic (5min entry, 1h trend, daily macro gate):
  1h EMA slope  → trend direction
  5min price touches lower Bollinger Band + RSI < rsi_thresh  → BUY entry
  5min price touches upper Bollinger Band + RSI > (100-rsi_thresh)  → SELL entry
  ATR(14)*atr_mult_sl  → dynamic SL distance
  Daily EMA slope  → macro direction gate (macro_filter: true)
  Session window  → peak London/NY hours (session_peak: true)

Designed for instruments where RSI rarely dips to 35 (strong or volatile trends)
but price regularly touches the Bollinger Bands.
"""

from datetime import datetime, timezone, timedelta

from .base import BaseProStrategy, Signal
from ..indicators import ema, rsi, bollinger as _bb, atr as _atr


class BBRSIPullbackStrategy(BaseProStrategy):

    name             = "bb_rsi_pullback"
    best_platform    = "MT4/MT5"
    best_instruments = ["frxEURUSD"]

    def __init__(self, config: dict):
        super().__init__(config)
        self._htf_bars:   list[dict] = []
        self._daily_bars: list[dict] = []

        self.ema_period       = config.get("ema_period",         100)
        self.slope_bars       = config.get("slope_bars",            3)
        self.bb_period        = config.get("bb_period",            20)
        self.bb_std           = config.get("bb_std",              2.0)
        self.rsi_period       = config.get("rsi_period",           14)
        self.rsi_thresh       = config.get("rsi_thresh",          55.0)  # BUY when RSI < this
        self.tp_rr            = config.get("tp_rr",               1.5)
        self.atr_mult_sl      = config.get("atr_mult_sl",         1.5)
        self.session_only     = config.get("session_only",       False)
        self.session_peak     = config.get("session_peak",       False)
        self.tz_offset_hours  = config.get("tz_offset_hours",      0)
        self.macro_filter     = config.get("macro_filter",       False)
        self.macro_ema_period = config.get("macro_ema_period",     20)

    def feed_htf(self, bar: dict) -> None:
        self._htf_bars.append(bar)

    def feed_daily(self, bar: dict) -> None:
        self._daily_bars.append(bar)

    def _in_session(self, bar: dict) -> bool:
        tz = timezone(timedelta(hours=self.tz_offset_hours))
        dt = datetime.fromtimestamp(bar.get("epoch", 0), tz=tz)
        h  = dt.hour + dt.minute / 60.0
        if self.session_peak:
            london_start = 7.0  + self.tz_offset_hours
            london_end   = 10.5 + self.tz_offset_hours
            ny_start     = 13.0 + self.tz_offset_hours
            ny_end       = 16.5 + self.tz_offset_hours
            return (london_start <= h < london_end) or (ny_start <= h < ny_end)
        if self.session_only:
            return (7.0 + self.tz_offset_hours) <= h < (20.0 + self.tz_offset_hours)
        return True

    def _evaluate(self) -> Signal:
        bars = self._bars
        htf  = self._htf_bars

        warm = max(self.rsi_period + 2, self.bb_period + 1)
        if len(bars) < warm:
            return Signal(action="HOLD", reason="Warming up LTF")
        if len(htf) < self.ema_period + self.slope_bars:
            return Signal(action="HOLD", reason="Warming up HTF")

        cur = bars[-1]
        if not self._in_session(cur):
            return Signal(action="HOLD", reason="Outside session")

        htf_closes = [b["close"] for b in htf]
        ltf_closes = [b["close"] for b in bars]

        # 1h trend
        ema_vals = ema(htf_closes, self.ema_period)
        ema_now  = ema_vals[-1]
        ema_prev = ema_vals[-1 - self.slope_bars]
        if ema_now is None or ema_prev is None:
            return Signal(action="HOLD", reason="HTF EMA not ready")

        # 5m indicators
        rsi_vals = rsi(ltf_closes, self.rsi_period)
        rsi_now  = rsi_vals[-1]
        if rsi_now is None:
            return Signal(action="HOLD", reason="RSI not ready")

        bb_vals = _bb(ltf_closes, self.bb_period, self.bb_std)
        bb_now  = bb_vals[-1]
        if bb_now is None:
            return Signal(action="HOLD", reason="BB not ready")
        up_b, _mid, lo_b = bb_now

        # ATR-based SL
        atr_vals = _atr(bars, 14)
        atr_now  = atr_vals[-1]
        if atr_now and self.atr_mult_sl > 0:
            sl_dist = atr_now * self.atr_mult_sl
        else:
            sl_dist = cur["close"] * 0.001
        tp_dist = sl_dist * self.tp_rr

        # Macro direction gate
        allow_long = allow_short = True
        if self.macro_filter and len(self._daily_bars) >= self.macro_ema_period + 1:
            d_ema = ema([b["close"] for b in self._daily_bars], self.macro_ema_period)
            if d_ema[-1] is not None and d_ema[-2] is not None:
                macro_slope = d_ema[-1] - d_ema[-2]
                flat_thresh = abs(d_ema[-1]) * 0.0001
                if abs(macro_slope) >= flat_thresh:
                    if macro_slope > 0:
                        allow_short = False
                    else:
                        allow_long = False

        trend_up   = ema_now > ema_prev
        trend_down = ema_now < ema_prev
        price      = cur["close"]
        ob_thresh  = 100.0 - self.rsi_thresh   # overbought mirror

        if trend_up and price <= lo_b and rsi_now < self.rsi_thresh and allow_long:
            return Signal(
                action="BUY",
                reason=f"EMA up | price@lower-BB {lo_b:.5f} | RSI {rsi_now:.1f}<{self.rsi_thresh}",
                sl_pips=sl_dist,
                tp_pips=tp_dist,
                confidence=min(1.0, (self.rsi_thresh - rsi_now) / 20),
                meta={"ema": round(ema_now, 5), "rsi": round(rsi_now, 1),
                      "bb_lo": round(lo_b, 5)},
            )

        if trend_down and price >= up_b and rsi_now > ob_thresh and allow_short:
            return Signal(
                action="SELL",
                reason=f"EMA dn | price@upper-BB {up_b:.5f} | RSI {rsi_now:.1f}>{ob_thresh}",
                sl_pips=sl_dist,
                tp_pips=tp_dist,
                confidence=min(1.0, (rsi_now - ob_thresh) / 20),
                meta={"ema": round(ema_now, 5), "rsi": round(rsi_now, 1),
                      "bb_up": round(up_b, 5)},
            )

        return Signal(
            action="HOLD",
            reason=(f"EMA {'up' if trend_up else 'dn' if trend_down else 'flat'} | "
                    f"RSI {rsi_now:.1f} | price {price:.5f} "
                    f"BB[{lo_b:.5f}–{up_b:.5f}]"),
        )
