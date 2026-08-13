"""
MACD Momentum Strategy (1H entry, 4H trend filter, daily macro)

Backtest result (dual-holdout, 2-year dataset, USDJPY):
  MACD(8,21,9) EMA100 RR2.0 ATR×2.0 4H-slope sess=peak
  Win-A holdout: EV +0.600R WR 66.7% n=10
  Win-B holdout: EV +0.375R WR 50.0% n=16

Logic:
  1H EMA(100) slope → trend direction
  MACD(fast,slow,sig) histogram on 1H → flip negative→positive = BUY
                                         flip positive→negative = SELL
  4H EMA(ema_4h_period) slope → must agree with 1H trend (stored in _htf_bars)
  Daily EMA slope → macro direction gate (macro_filter: true)
  Session window → peak London/NY hours (session_peak: true)

Entry fires on bar close when histogram changes sign and all gates pass.
Designed for USDJPY where BoJ/carry dynamics make 5m noise unreliable;
the 1H timeframe captures genuine momentum shifts.
"""

from datetime import datetime, timezone, timedelta

from .base import BaseProStrategy, Signal
from ..indicators import ema, macd as _macd, atr as _atr


class MACDMomentumStrategy(BaseProStrategy):

    name             = "macd_momentum"
    best_platform    = "MT4/MT5"
    best_instruments = ["frxUSDJPY"]

    def __init__(self, config: dict):
        super().__init__(config)
        self._htf_bars:   list[dict] = []   # 4H bars — slope gate
        self._daily_bars: list[dict] = []   # daily bars — macro gate

        self.macd_fast        = config.get("macd_fast",        8)
        self.macd_slow        = config.get("macd_slow",       21)
        self.macd_signal      = config.get("macd_signal",      9)
        self.ema_period       = config.get("ema_period",     100)   # 1H trend EMA
        self.ema_4h_period    = config.get("ema_4h_period",   20)   # 4H slope EMA
        self.tp_rr            = config.get("tp_rr",          2.0)
        self.atr_mult_sl      = config.get("atr_mult_sl",    2.0)
        self.session_peak     = config.get("session_peak",  False)
        self.tz_offset_hours  = config.get("tz_offset_hours",  0)
        self.macro_filter     = config.get("macro_filter",  False)
        self.macro_ema_period = config.get("macro_ema_period", 20)
        self._session_mode    = config.get("session",
                                           "peak" if self.session_peak else "off")

    def feed_htf(self, bar: dict) -> None:
        self._htf_bars.append(bar)

    def feed_4h(self, bar: dict) -> None:
        pass  # 4H comes via _htf_bars (htf_granularity: 14400)

    def feed_daily(self, bar: dict) -> None:
        self._daily_bars.append(bar)

    def _in_session(self, bar: dict) -> bool:
        tz   = timezone(timedelta(hours=self.tz_offset_hours))
        dt   = datetime.fromtimestamp(bar.get("epoch", 0), tz=tz)
        h    = dt.hour + dt.minute / 60.0
        off  = self.tz_offset_hours
        mode = self._session_mode
        if mode == "london":
            return (7.0 + off) <= h < (11.0 + off)   # 10:00–14:00 EAT
        if mode == "ny":
            return (13.0 + off) <= h < (17.0 + off)  # 16:00–20:00 EAT
        if mode == "peak":
            return ((7.0 + off) <= h < (11.0 + off)) or \
                   ((13.0 + off) <= h < (17.0 + off))
        return True

    def _evaluate(self) -> Signal:
        bars = self._bars
        min_bars = self.macd_slow + self.macd_signal + 2
        if len(bars) < min_bars:
            return Signal(action="HOLD", reason="Warming up MACD")
        if len(bars) < self.ema_period + 2:
            return Signal(action="HOLD", reason="Warming up EMA")

        cur = bars[-1]
        if not self._in_session(cur):
            return Signal(action="HOLD", reason="Outside session")

        closes = [b["close"] for b in bars]

        # 1H EMA trend direction
        ema_vals = ema(closes, self.ema_period)
        if ema_vals[-1] is None or ema_vals[-2] is None:
            return Signal(action="HOLD", reason="EMA not ready")
        trend_up   = ema_vals[-1] > ema_vals[-2]
        trend_down = ema_vals[-1] < ema_vals[-2]

        # MACD histogram
        _, _, hist = _macd(closes, self.macd_fast, self.macd_slow, self.macd_signal)
        if hist[-1] is None or hist[-2] is None:
            return Signal(action="HOLD", reason="MACD not ready")
        hist_n = hist[-1]
        hist_p = hist[-2]

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

        # 4H EMA slope gate — HTF bars are 4H (htf_granularity: 14400)
        if len(self._htf_bars) >= self.ema_4h_period + 1:
            ema_4h = ema([b["close"] for b in self._htf_bars], self.ema_4h_period)
            if ema_4h[-1] is not None and ema_4h[-2] is not None:
                ema4h_up = ema_4h[-1] > ema_4h[-2]
                if trend_up and not ema4h_up:
                    return Signal(action="HOLD", reason="4H EMA opposes 1H uptrend")
                if trend_down and ema4h_up:
                    return Signal(action="HOLD", reason="4H EMA opposes 1H downtrend")

        # ATR-based SL sizing
        atr_vals = _atr(bars, 14)
        atr_now  = atr_vals[-1] if atr_vals else None
        if not atr_now:
            atr_now = cur["close"] * 0.001
        sl_dist = atr_now * self.atr_mult_sl
        tp_dist = sl_dist * self.tp_rr

        price = cur["close"]

        # MACD histogram flip: negative → positive = bullish momentum shift
        if trend_up and hist_p < 0 and hist_n >= 0 and allow_long:
            return Signal(
                action="BUY",
                reason=f"MACD hist flip↑ {hist_p:.5f}→{hist_n:.5f} | EMA trend up",
                sl_pips=sl_dist,
                tp_pips=tp_dist,
                meta={"ema": round(ema_vals[-1], 5), "hist": round(hist_n, 6)},
            )

        # MACD histogram flip: positive → negative = bearish momentum shift
        if trend_down and hist_p > 0 and hist_n <= 0 and allow_short:
            return Signal(
                action="SELL",
                reason=f"MACD hist flip↓ {hist_p:.5f}→{hist_n:.5f} | EMA trend dn",
                sl_pips=sl_dist,
                tp_pips=tp_dist,
                meta={"ema": round(ema_vals[-1], 5), "hist": round(hist_n, 6)},
            )

        return Signal(
            action="HOLD",
            reason=(f"EMA {'up' if trend_up else 'dn' if trend_down else 'flat'} | "
                    f"MACD hist {hist_n:.5f} (prev {hist_p:.5f})"),
        )
