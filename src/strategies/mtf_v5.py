"""
MTF v5 Strategy — 5min RSI pullback + 1h EMA trend + daily macro gate

Backtest-confirmed configs (pro_bot/backtest.py v5, spread applied, min 100 trades):
  frxXAUUSD: EMA100/1h RSI<35/5m ADX25 SESS_PEAK MACRO20  EV+0.280R WR 51.2%
  frxEURUSD: EMA200/1h RSI<35/5m ADX25        MACRO50     EV+0.388R WR 46.3%
  frxUSDJPY: EMA100/1h RSI<35/5m ADX20 ATR1.5 MACRO20     EV+0.305R WR 43.5%

Signal fires on 5-min bar close. Call seed_all() at startup with historical
candle data fetched from Deriv.

Macro filter: daily EMA slope auto-switches direction with the market regime
(no config change needed when Gold shifts from bearish to bullish).

Multiplier contract mode: sets sl_pct/tp_pct on the Signal so Trader places
MULTUP/MULTDOWN with computed SL/TP amounts.
"""

from datetime import datetime, timezone
from typing import Optional

from .base import BaseStrategy, Signal


class MTFV5Strategy(BaseStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self._ltf_bars:     list[dict]  = []   # 5min OHLCV bars
        self._htf_closes:   list[float] = []   # 1h closes for EMA trend
        self._daily_closes: list[float] = []   # daily closes for macro EMA

        # Bar tracking (builds current 5min bar from ticks)
        self._bar_start: Optional[int]   = None
        self._bar_open:  Optional[float] = None
        self._bar_high:  Optional[float] = None
        self._bar_low:   Optional[float] = None

        # Epoch of last HTF/daily bar appended (to throttle updates)
        self._last_htf_end:   int = 0
        self._last_daily_end: int = 0

        self.bar_seconds      = config.get("bar_seconds",     300)
        self.htf_seconds      = config.get("htf_seconds",    3600)
        self.daily_seconds    = config.get("daily_seconds", 86400)
        self.ema_period       = config.get("ema_period",      100)
        self.slope_bars       = config.get("slope_bars",        3)
        self.rsi_period       = config.get("rsi_period",       14)
        self.rsi_entry        = config.get("rsi_entry",       35.0)
        self.tp_rr            = config.get("tp_rr",            1.5)
        self.adx_min          = config.get("adx_min",           0)
        self.session_peak     = config.get("session_peak",   False)
        self.atr_mult_sl      = config.get("atr_mult_sl",     0.0)
        self.sl_pct           = config.get("sl_pct",         0.001)
        self.macro_filter     = config.get("macro_filter",   False)
        self.macro_ema_period = config.get("macro_ema_period", 20)

    # ── Startup seeding ──────────────────────────────────────────────────

    def seed_all(self, ltf_bars: list[dict], htf_closes: list[float],
                 daily_closes: list[float]) -> None:
        self._ltf_bars     = list(ltf_bars)
        self._htf_closes   = list(htf_closes)
        self._daily_closes = list(daily_closes)
        if self._ltf_bars:
            self._last_htf_end   = self._ltf_bars[-1]["epoch"]
            self._last_daily_end = self._ltf_bars[-1]["epoch"]

    # ── Live tick evaluation ─────────────────────────────────────────────

    def evaluate(self, tick_store) -> Signal:
        price = tick_store.latest_price
        epoch = tick_store.latest_epoch
        if price is None or epoch is None:
            return Signal(action="HOLD", reason="No ticks yet")

        if self._bar_start is None:
            self._bar_start = epoch
            self._bar_open = self._bar_high = self._bar_low = price
            return Signal(action="HOLD", reason="Bar started")

        if price > self._bar_high:
            self._bar_high = price
        if price < self._bar_low:
            self._bar_low = price

        if epoch - self._bar_start < self.bar_seconds:
            return Signal(action="HOLD",
                          reason=f"In bar ({epoch - self._bar_start}s/{self.bar_seconds}s)")

        # ── 5min bar closed ───────────────────────────────────────────
        self._ltf_bars.append({
            "epoch": epoch,
            "open":  self._bar_open,
            "high":  self._bar_high,
            "low":   self._bar_low,
            "close": price,
        })

        if epoch - self._last_htf_end >= self.htf_seconds:
            self._htf_closes.append(price)
            self._last_htf_end = epoch

        if epoch - self._last_daily_end >= self.daily_seconds:
            self._daily_closes.append(price)
            self._last_daily_end = epoch

        self._bar_start = epoch
        self._bar_open = self._bar_high = self._bar_low = price

        return self._evaluate_bar(price, epoch)

    # ── Signal logic ─────────────────────────────────────────────────────

    def _evaluate_bar(self, close: float, epoch: int) -> Signal:
        ltf = self._ltf_bars
        htf = self._htf_closes

        need_ltf = max(self.rsi_period, 14) + 2
        need_htf = self.ema_period + self.slope_bars
        if len(ltf) < need_ltf:
            return Signal(action="HOLD", reason=f"LTF warm-up ({len(ltf)}/{need_ltf})")
        if len(htf) < need_htf:
            return Signal(action="HOLD", reason=f"HTF warm-up ({len(htf)}/{need_htf})")

        if self.session_peak:
            dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
            h  = dt.hour + dt.minute / 60.0
            if not ((7.0 <= h < 10.5) or (13.0 <= h < 16.5)):
                return Signal(action="HOLD", reason="Outside session peak")

        ema_now  = _ema(htf,                    self.ema_period)
        ema_prev = _ema(htf[:-self.slope_bars], self.ema_period)
        if ema_now is None or ema_prev is None:
            return Signal(action="HOLD", reason="HTF EMA not ready")

        ltf_closes = [b["close"] for b in ltf]
        rsi_prev, rsi_now = _rsi_two(ltf_closes, self.rsi_period)
        if rsi_now is None:
            return Signal(action="HOLD", reason="RSI not ready")

        if self.adx_min > 0:
            highs   = [b["high"] for b in ltf]
            lows    = [b["low"]  for b in ltf]
            adx_val = _adx(highs, lows, ltf_closes, 14)
            if adx_val is None or adx_val < self.adx_min:
                return Signal(action="HOLD",
                              reason=f"ADX {adx_val:.1f} < {self.adx_min}" if adx_val else "ADX not ready")

        if self.atr_mult_sl > 0:
            highs   = [b["high"] for b in ltf]
            lows    = [b["low"]  for b in ltf]
            atr_val = _atr(highs, lows, ltf_closes, 14)
            sl_pct  = (atr_val * self.atr_mult_sl / close) if atr_val else self.sl_pct
        else:
            sl_pct = self.sl_pct
        tp_pct = sl_pct * self.tp_rr

        allow_long = allow_short = True
        if self.macro_filter and len(self._daily_closes) >= self.macro_ema_period + 1:
            d_now  = _ema(self._daily_closes,      self.macro_ema_period)
            d_prev = _ema(self._daily_closes[:-1], self.macro_ema_period)
            if d_now is not None and d_prev is not None:
                macro_up    = d_now > d_prev
                allow_long  = macro_up
                allow_short = not macro_up

        trend_up   = ema_now > ema_prev
        trend_down = ema_now < ema_prev
        ob         = 100.0 - self.rsi_entry

        if trend_up and rsi_prev >= self.rsi_entry > rsi_now and allow_long:
            return Signal(
                action="BUY_RISE",
                reason=f"EMA up | RSI {rsi_now:.1f} cross | macro {'up' if self.macro_filter else 'off'}",
                rsi=rsi_now,
                sl_pct=sl_pct,
                tp_pct=tp_pct,
            )

        if trend_down and rsi_prev <= ob < rsi_now and allow_short:
            return Signal(
                action="BUY_FALL",
                reason=f"EMA dn | RSI {rsi_now:.1f} cross | macro {'dn' if self.macro_filter else 'off'}",
                rsi=rsi_now,
                sl_pct=sl_pct,
                tp_pct=tp_pct,
            )

        return Signal(
            action="HOLD",
            reason=f"EMA {'up' if trend_up else 'dn' if trend_down else '='} | RSI {rsi_now:.1f}",
            rsi=rsi_now,
        )

    def on_result(self, won: bool) -> None:
        pass


# ── Indicator helpers ────────────────────────────────────────────────────

def _ema(closes: list[float], period: int) -> Optional[float]:
    if len(closes) < period:
        return None
    k   = 2.0 / (period + 1)
    val = sum(closes[:period]) / period
    for c in closes[period:]:
        val = c * k + val * (1 - k)
    return val


def _rsi_two(closes: list[float], period: int) -> tuple[Optional[float], Optional[float]]:
    """Returns (rsi_prev, rsi_now) using simple RSI on the last two completed bars."""
    if len(closes) < period + 2:
        return (None, None)

    def _one(window: list[float]) -> float:
        changes  = [window[i] - window[i - 1] for i in range(1, len(window))]
        avg_gain = sum(c for c in changes if c > 0) / period
        avg_loss = sum(abs(c) for c in changes if c < 0) / period
        if avg_loss == 0:
            return 100.0
        return 100 - (100 / (1 + avg_gain / avg_loss))

    return (
        _one(closes[-(period + 2):-1]),
        _one(closes[-(period + 1):]),
    )


def _atr(highs: list[float], lows: list[float],
         closes: list[float], period: int) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    trs = [
        max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        for i in range(-period, 0)
    ]
    return sum(trs) / period


def _adx(highs: list[float], lows: list[float],
         closes: list[float], period: int) -> Optional[float]:
    if len(closes) < period * 2:
        return None
    plus_dm, minus_dm, trs = [], [], []
    for i in range(-period, 0):
        up   = highs[i]   - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm.append(up   if up > down   and up > 0   else 0.0)
        minus_dm.append(down if down > up   and down > 0 else 0.0)
        h, l, pc = highs[i], lows[i], closes[i - 1]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    atr = sum(trs) / period
    if atr == 0:
        return 0.0
    plus_di  = (sum(plus_dm)  / period) / atr * 100
    minus_di = (sum(minus_dm) / period) / atr * 100
    denom    = plus_di + minus_di
    if denom == 0:
        return 0.0
    return abs(plus_di - minus_di) / denom * 100
