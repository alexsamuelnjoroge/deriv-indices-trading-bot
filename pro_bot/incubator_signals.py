"""
Frozen signal rules for the XAUUSD incubator (research 2026-11-10).

Shared by the live runner (pro_bot/incubator.py) and the historical parity check
(research/incubator_parity.py), so live trades come from exactly the tested code.
Rules are frozen: do not tune them on incubation results until the review date.

  A_asia_dip    gold's 12-bar move > 3 sigma in the Asian session -> fade, hold 24 bars
                (research/xau_freq_battery.py F1 "k12 z>3.0 asia fade H=24")
  B_usd_resid   gold's 48-bar move unexplained by the USD basket > 3 sigma -> fade, hold 24
                (research/xau_xasset_battery.py G1 "usd W48 z>3.0 all fade H=24")
  C_mf_resid_eu 12-bar residual vs USD basket + silver + US500 > 2 sigma, EU session
                -> fade, hold 12 (G2 "W12 z>2.0 eu fade H=12" minus the T-bond factor,
                which IC Markets MT5 does not offer)

All times UTC. Bars are M5, indexed by bar open time. A signal on bar i is entered at
the open of bar i+1 and exited at the close of bar i+H. Entries 20:30-22:30 UTC are
blocked and a trade must exit inside the same session (session boundary 20:30 UTC),
so no position is held through the daily rollover.
"""

import numpy as np
import pandas as pd

GOLD = "XAUUSD"
FX_USD_SIGN = {"EURUSD": -1, "GBPUSD": -1, "AUDUSD": -1, "USDJPY": 1, "USDCHF": 1, "USDCAD": 1}
FACTOR_SYMBOL = {"xag": "XAGUSD", "spx": "US500", "bond": "USTBOND"}

RULES = {
    "A_asia_dip":    dict(kind="selfz", k=12, thr=3.0, sess="asia", H=24),
    "B_usd_resid":   dict(kind="resid", factors=("usd",), W=48, thr=3.0, sess="all", H=24),
    "C_mf_resid_eu": dict(kind="resid", factors=("usd", "xag", "spx"), W=12, thr=2.0, sess="eu", H=12),
}

# D_asia_range_bo — Asian-range breakout (research/xau_session_breakout_ea.py, 2026-11-11).
# Server time = New York + 7h. Range = high/low of 03:00-08:00 server; at 08:00 place a buy-stop
# at the high and a sell-stop at the low (one cancels the other). SL = opposite side of the
# range, TP = 1R. Unfilled orders cancelled at 22:00, open trades closed at 23:00 server.
# Duka 2008-16 +4.7bp t=4.1 | 2017-26 +3.4bp t=3.1 | IC MT5 2022-26 +6.8bp t=3.4 (cost 0.36bp).
BREAKOUT = dict(name="D_asia_range_bo", range_start=3 * 60, range_end=8 * 60,
                place_until=8 * 60 + 30, cancel=22 * 60, flat=23 * 60, tp_r=1.0)


def asia_range(server_bars: pd.DataFrame, server_date) -> tuple | None:
    """(high, low) of the BREAKOUT window on server_date from M5 bars indexed by server time,
    or None if the window is incomplete (holiday, missing data)."""
    day = server_bars[server_bars.index.date == server_date]
    mins = day.index.hour * 60 + day.index.minute
    win = day[(mins >= BREAKOUT["range_start"]) & (mins < BREAKOUT["range_end"])]
    need = (BREAKOUT["range_end"] - BREAKOUT["range_start"]) // 5
    if len(win) < need - 6:
        return None
    hi, lo = float(win["high"].max()), float(win["low"].min())
    return (hi, lo) if hi > lo else None


SESSION_OFFSET = int(20.5 * 3600)
BETA_LOOKBACK_SESSIONS = 20


def symbols_needed(rules=RULES) -> list[str]:
    syms = {GOLD}
    for r in rules.values():
        for f in r.get("factors", ()):
            if f == "usd":
                syms.update(FX_USD_SIGN)
            else:
                syms.add(FACTOR_SYMBOL[f])
    return sorted(syms)


def session_id(ts_seconds):
    return (np.asarray(ts_seconds, dtype=np.int64) - SESSION_OFFSET) // 86400


def align(gold_close: pd.Series, others: dict) -> pd.DataFrame:
    """Reindex every other symbol's close onto the gold bar index (ffill <= 1h)."""
    out = {GOLD: gold_close}
    for sym, s in others.items():
        out[sym] = s.reindex(gold_close.index, method="ffill", limit=12)
    return pd.DataFrame(out)


def _bp_returns(close: np.ndarray, gap: np.ndarray) -> np.ndarray:
    r = np.r_[np.nan, np.diff(np.log(close) * 1e4)]
    r[gap] = np.nan
    return r


def _first_bar(c):
    return c & ~np.r_[False, c[:-1]]


def _roll_sum(x, w):
    return pd.Series(np.nan_to_num(x)).rolling(w).sum().to_numpy()


def _session_masks(idx: pd.DatetimeIndex):
    hour = idx.hour.to_numpy()
    mod = hour * 60 + idx.minute.to_numpy()
    entry_ok = ~((mod >= 20 * 60 + 30) & (mod < 22 * 60 + 30))
    return {
        "all":  entry_ok,
        "asia": entry_ok & ((mod >= 22 * 60 + 30) | (hour < 7)),
        "eu":   entry_ok & (hour >= 7) & (hour < 12),
        "us":   entry_ok & (hour >= 12) & (hour < 20),
    }


def _residual(rg, X, sess_rank, n_sess):
    ok = np.isfinite(rg) & np.isfinite(X).all(1)
    Xo = np.where(ok[:, None], X, 0.0)
    yo = np.where(ok, rg, 0.0)
    k = X.shape[1]
    xtx = np.zeros((n_sess, k, k))
    xty = np.zeros((n_sess, k))
    np.add.at(xtx, sess_rank, np.einsum("ni,nj->nij", Xo, Xo))
    np.add.at(xty, sess_rank, Xo * yo[:, None])
    cx, cy = np.cumsum(xtx, 0), np.cumsum(xty, 0)
    beta = np.full((n_sess, k), np.nan)
    L = BETA_LOOKBACK_SESSIONS
    for s in range(L, n_sess):
        A = cx[s - 1] - cx[s - 1 - L]
        b = cy[s - 1] - cy[s - 1 - L]
        try:
            beta[s] = np.linalg.solve(A + 1e-6 * np.eye(k), b)
        except np.linalg.LinAlgError:
            pass
    fit = np.einsum("ni,ni->n", np.nan_to_num(X), beta[sess_rank])
    return np.where(ok, rg - fit, np.nan)


def compute_signals(df: pd.DataFrame, rules=RULES) -> pd.DataFrame:
    """
    df: aligned closes from align(); index = gold M5 bar open times (UTC, tz-aware).
    Returns per-bar columns <rule>_side (+1 buy / -1 sell / 0) and <rule>_z.
    """
    idx = df.index
    ts = idx.asi8 // 10**9
    gap = np.r_[True, np.diff(ts) > 3600]
    rg = _bp_returns(df[GOLD].to_numpy(float), gap)
    sig = pd.Series(rg).rolling(288, min_periods=200).std().to_numpy()
    sess = session_id(ts)
    sess_rank = np.unique(sess, return_inverse=True)[1]
    n_sess = int(sess_rank.max()) + 1
    masks = _session_masks(idx)

    fr = {}

    def factor(name):
        if name not in fr:
            if name == "usd":
                fx = [sgn * _bp_returns(df[s].to_numpy(float), gap) for s, sgn in FX_USD_SIGN.items()]
                with np.errstate(all="ignore"):
                    stack = np.vstack(fx)
                    cnt = np.isfinite(stack).sum(0)
                    fr[name] = np.where(cnt > 0, np.nansum(stack, 0) / np.maximum(cnt, 1), np.nan)
            else:
                fr[name] = _bp_returns(df[FACTOR_SYMBOL[name]].to_numpy(float), gap)
        return fr[name]

    out = {}
    for name, r in rules.items():
        if r["kind"] == "selfz":
            k = r["k"]
            lp = np.log(df[GOLD].to_numpy(float)) * 1e4
            z = (lp - np.r_[np.full(k, np.nan), lp[:-k]]) / (sig * np.sqrt(k))
        else:
            X = np.column_stack([factor(f) for f in r["factors"]])
            e = _residual(rg, X, sess_rank, n_sess)
            esd = pd.Series(e).rolling(288, min_periods=200).std().to_numpy()
            W = r["W"]
            z = _roll_sum(e, W) / (esd * np.sqrt(W))
        fire = _first_bar(np.nan_to_num(np.abs(z) > r["thr"], nan=False).astype(bool)) & masks[r["sess"]]
        side = np.where(fire, -np.sign(np.nan_to_num(z)), 0).astype(np.int8)
        out[f"{name}_side"] = side
        out[f"{name}_z"] = z
    return pd.DataFrame(out, index=idx)
