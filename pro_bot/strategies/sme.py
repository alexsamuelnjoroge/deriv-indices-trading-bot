"""Session Move Exhaustion — cumulative session displacement beyond ATR threshold → fade the session move."""
from typing import Optional

from .research_base import ResearchDailyStrategy
from .base import Signal


class SMEStrategy(ResearchDailyStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.exhaustion = config.get("exhaustion_thresh", 3.0)
        self.session    = config.get("session", "both")   # "london" | "ny" | "both"

        self._session_opens: dict = {}   # day_midnight → {london: price, ny: price}
        self._fired: dict         = {}   # day_midnight → {london: bool, ny: bool}

    def _record_session_open(self, bar: dict):
        epoch = bar.get("epoch", 0)
        h     = (epoch % 86400) // 3600
        day   = (epoch // 86400) * 86400

        if day not in self._session_opens:
            self._session_opens[day] = {"london": None, "ny": None}
        so = self._session_opens[day]

        if h == 7 and so["london"] is None:
            so["london"] = bar["open"]
        if h == 13 and so["ny"] is None:
            so["ny"] = bar["open"]

    def feed(self, bar: dict):
        self._record_session_open(bar)
        return super().feed(bar)

    def _seed_h1(self, bars: list) -> None:
        for bar in bars:
            self._record_session_open(bar)
            self._h1.append(bar)
            self._bar_idx += 1

    def _check_signal(self, atr, allow_long, allow_short) -> Optional[Signal]:
        bar   = list(self._h1)[-1]
        epoch = bar.get("epoch", 0)
        h     = (epoch % 86400) // 3600
        day   = (epoch // 86400) * 86400

        in_london = 7 <= h <= 16
        in_ny     = 13 <= h <= 21

        if self.session == "london" and not in_london:
            return None
        if self.session == "ny" and not in_ny:
            return None
        if self.session == "both" and not (in_london or in_ny):
            return None

        so = self._session_opens.get(day)
        if so is None:
            return None

        if day not in self._fired:
            self._fired[day] = {"london": False, "ny": False}

        sess_key  = None
        sess_open = None
        if in_london and self.session in ("london", "both"):
            if not self._fired[day]["london"]:
                sess_key  = "london"
                sess_open = so.get("london")
        if in_ny and self.session in ("ny", "both") and sess_key is None:
            if not self._fired[day]["ny"]:
                sess_key  = "ny"
                sess_open = so.get("ny")

        if sess_key is None or sess_open is None:
            return None

        displacement = (bar["close"] - sess_open) / atr
        if abs(displacement) < self.exhaustion:
            return None

        sl, tp = self._sl_tp(atr)

        if displacement > 0 and allow_short:
            self._fired[day][sess_key] = True
            return Signal("SELL", sl_pips=sl, tp_pips=tp,
                          reason=f"SME sell disp={displacement:.2f}ATR sess={sess_key}")

        if displacement < 0 and allow_long:
            self._fired[day][sess_key] = True
            return Signal("BUY", sl_pips=sl, tp_pips=tp,
                          reason=f"SME buy disp={displacement:.2f}ATR sess={sess_key}")

        return None
