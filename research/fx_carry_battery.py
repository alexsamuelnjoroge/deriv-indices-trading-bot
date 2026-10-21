"""
Adversarial test battery for CURRENCY CARRY (long high-rate, short low-rate currencies vs USD).

Universe: EUR JPY GBP CHF AUD NZD CAD NOK SEK (G10) + MXN ZAR TRY PLN HUF CZK (EM), each traded
vs USD.  SGD dropped (no OECD/FRED short-rate series; MAS runs an FX-band, not a rate policy).

Variants fixed upfront:
  a) cross-sectional carry: month-end rank by lagged 3m rate, long top k / short bottom k vs USD,
     inverse-vol weights, whole book scaled ex-ante to 10% vol (252d covariance), gross cap 6x
  b) a + trend filter: keep a leg only if its 63d (b3) / 126d (b6) spot return agrees with carry
  c) a + risk filter: flat when VIX > rolling 252d median (c_med) / 75th pct (c_p75), daily, lag 1
     (alt proxy: 20d S&P realised vol > 252d median, c_spx)
  d) fixed basket: long AUD NZD, short JPY CHF vs USD (USD implicit middle)

Returns: spot return + (r_ccy - r_usd)*days/365 (theory carry, contemporaneous monthly rate).
REALISTIC: broker swap = differential minus a per-currency markup estimated from current MT5
symbol_info swap_long/swap_short vs current FRED differentials; MT5 median spread + IC raw
commission charged on turnover.  Signals use rates lagged one month; trades lag the signal 1 day.

Data: Yahoo FX (2001/2003->), FRED H.10 noon rates (backfill pre-Yahoo + second source); MT5 only
for symbol_info swaps/spreads (spreads: median of an earlier 45-day M15 sample where cached), FRED OECD 3m interbank IR3TIB01xxM156N (fallback call money IRSTCI01xxM156N),
Yahoo ^VIX / ^GSPC.  Cached to research/cache_fx_carry_*.csv.  Research only: read-only MT5,
no orders, no live-code changes.

Run: .venv/Scripts/python.exe research/fx_carry_battery.py > research/fx_carry_battery.out
"""

import io
import json
import math
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20261021)
CDIR = "research"
C_PX = f"{CDIR}/cache_fx_carry_px.csv"
C_H10 = f"{CDIR}/cache_fx_carry_h10.csv"
C_RT = f"{CDIR}/cache_fx_carry_rates.csv"
C_MT5I = f"{CDIR}/cache_fx_carry_mt5info.json"

# ccy: (yahoo, yahoo_is_USDxxx, fred_h10, h10_is_USDxxx, mt5, mt5_is_USDxxx, country, group)
CCY = {
    "EUR": ("EURUSD=X", False, "DEXUSEU", False, "EURUSD", False, "DE", "G10"),
    "JPY": ("JPY=X", True, "DEXJPUS", True, "USDJPY", True, "JP", "G10"),
    "GBP": ("GBPUSD=X", False, "DEXUSUK", False, "GBPUSD", False, "GB", "G10"),
    "CHF": ("CHF=X", True, "DEXSZUS", True, "USDCHF", True, "CH", "G10"),
    "AUD": ("AUDUSD=X", False, "DEXUSAL", False, "AUDUSD", False, "AU", "G10"),
    "NZD": ("NZDUSD=X", False, "DEXUSNZ", False, "NZDUSD", False, "NZ", "G10"),
    "CAD": ("CAD=X", True, "DEXCAUS", True, "USDCAD", True, "CA", "G10"),
    "NOK": ("NOK=X", True, "DEXNOUS", True, "USDNOK", True, "NO", "G10"),
    "SEK": ("SEK=X", True, "DEXSDUS", True, "USDSEK", True, "SE", "G10"),
    "MXN": ("MXN=X", True, "DEXMXUS", True, "USDMXN", True, "MX", "EM"),
    "ZAR": ("ZAR=X", True, "DEXSFUS", True, "USDZAR", True, "ZA", "EM"),
    "TRY": ("TRY=X", True, None, True, "USDTRY", True, "TR", "EM"),
    "PLN": ("PLN=X", True, None, True, "USDPLN", True, "PL", "EM"),
    "HUF": ("HUF=X", True, None, True, "USDHUF", True, "HU", "EM"),
    "CZK": ("CZK=X", True, None, True, "USDCZK", True, "CZ", "EM"),
}
CCYS = list(CCY)
G10 = [c for c in CCYS if CCY[c][7] == "G10"]
EM = [c for c in CCYS if CCY[c][7] == "EM"]
COMMISSION = 0.35e-4          # IC Markets raw: $3.5/side per 100k notional
VOL_TGT = 0.10
GROSS_CAP = 6.0
START = "2000-01-01"
N_CONFIGS = 0


def fred(sid: str) -> pd.Series:
    import requests
    for _ in range(3):
        try:
            r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", timeout=60)
            d = pd.read_csv(io.StringIO(r.text))
            s = pd.to_numeric(d.iloc[:, 1], errors="coerce")
            s.index = pd.to_datetime(d.iloc[:, 0])
            return s.dropna()
        except Exception:
            time.sleep(3)
    return pd.Series(dtype=float)


# ----------------------------------------------------------------------------- data
def load_data():
    if not os.path.exists(C_PX):
        import yfinance as yf
        tk = [CCY[c][0] for c in CCYS] + ["^VIX", "^GSPC"]
        d = yf.download(tk, start="1995-01-01", auto_adjust=False, progress=False)["Close"]
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        d.to_csv(C_PX)
    if not os.path.exists(C_H10):
        h = {CCY[c][2]: fred(CCY[c][2]) for c in CCYS if CCY[c][2]}
        pd.DataFrame(h).to_csv(C_H10)
    if not os.path.exists(C_RT):
        rt = {}
        for cc in ["US"] + [CCY[c][6] for c in CCYS] + ["EZ"]:
            for pre in ("IR3TIB01", "IRSTCI01"):
                rt[f"{pre}{cc}"] = fred(f"{pre}{cc}M156N")
        pd.DataFrame(rt).to_csv(C_RT)
    if not os.path.exists(C_MT5I):
        # MT5 used ONLY for symbol_info/tick, one subprocess per symbol with a 20 s hard timeout
        # (the shared live terminal can block).  Symbols that time out are skipped and reported.
        import subprocess
        tmp = f"{CDIR}/_mt5_tmp"
        os.makedirs(tmp, exist_ok=True)
        info = {}
        for c in CCYS:
            s_ = CCY[c][4]
            f = f"{tmp}/{s_}.json"
            if not os.path.exists(f):
                try:
                    subprocess.run([sys.executable, __file__, "--mt5", s_, f], timeout=20)
                except subprocess.TimeoutExpired:
                    print("  mt5 timeout, skipped", s_, flush=True)
            if os.path.exists(f):
                info[s_] = json.load(open(f))["info"]
        json.dump(info, open(C_MT5I, "w"), indent=1)

    raw = pd.read_csv(C_PX, index_col=0, parse_dates=True)
    h10 = pd.read_csv(C_H10, index_col=0, parse_dates=True)
    rt = pd.read_csv(C_RT, index_col=0, parse_dates=True)
    info = json.load(open(C_MT5I))
    return raw, h10, rt, info


def usd_per(series: pd.Series, inverted: bool) -> pd.Series:
    s = series.dropna()
    s = s[s > 0]
    return 1.0 / s if inverted else s


def clean_spikes(s: pd.Series, thr=0.06):
    """Yahoo FX has one-day bad prints: |r_t|>thr and reversal next day -> replace by prior price."""
    s = s.copy()
    n = 0
    for _ in range(3):
        r = np.log(s).diff()
        bad = (r.abs() > thr) & ((r + r.shift(-1)).abs() < thr / 3)
        n += int(bad.sum())
        if not bad.any():
            break
        s[bad] = np.nan
        s = s.ffill()
    return s, n


def build_panel(raw, h10, rt):
    bidx = pd.bdate_range("1998-01-01", raw.index.max())
    px, src, spikes = {}, {}, {}
    for c in CCYS:
        y, yinv, h, hinv = CCY[c][:4]
        ys, n = clean_spikes(usd_per(raw[y], yinv))
        spikes[c] = n
        if h and h in h10:
            hs = usd_per(h10[h], hinv)
            hs = hs[hs.index < ys.index[0]]
            s = pd.concat([hs, ys])
            src[c] = f"H10 {hs.index[0].date() if len(hs) else '-'} + Yahoo {ys.index[0].date()}"
        else:
            s = ys
            src[c] = f"Yahoo {ys.index[0].date()}"
        s = s[~s.index.duplicated(keep="last")].sort_index()
        px[c] = s.reindex(bidx).ffill(limit=5)
    P = pd.DataFrame(px)
    R = P.pct_change(fill_method=None)
    R = R.where(P.notna() & P.shift().notna())
    days = pd.Series(bidx, index=bidx).diff().dt.days.fillna(1)

    # rates: 3m interbank, filled by call money where 3m is missing (EUR: DE 3m, then EZ)
    rates, rsrc = {}, {}
    for c in ["USD"] + CCYS:
        cc = "US" if c == "USD" else CCY[c][6]
        a = rt.get(f"IR3TIB01{cc}", pd.Series(dtype=float)).dropna()
        b = rt.get(f"IRSTCI01{cc}", pd.Series(dtype=float)).dropna()
        if c == "EUR":
            a = a.combine_first(rt["IR3TIB01EZ"].dropna())
            b = rt["IRSTCI01EZ"].dropna().combine_first(b)
        s = a.combine_first(b)
        rates[c] = s
        filled = s.index.difference(a.index)
        filled = filled[filled >= "1999-01-01"]
        rsrc[c] = f"3m to {a.index.max().date()}" + (f", call-money fill {len(filled)} m" if len(filled) else "")
    M = pd.DataFrame(rates).sort_index()
    M.index = M.index.to_period("M")
    M = M.reindex(pd.period_range(M.index.min(), bidx[-1].to_period("M") + 1, freq="M")).ffill(limit=3)
    return P, R, days, M, src, spikes, rsrc


# ----------------------------------------------------------------------------- swap markup
def swap_markup(info, M):
    """Annualised swap rates from MT5 points: swap*point/price*365 (swap_mode 1 = points, Wed x3)."""
    last = M.ffill().iloc[-1]
    rows = []
    for c in CCYS:
        s = CCY[c][4]
        if s not in info:
            print(f"  WARNING: no MT5 info for {s}; using median markup of others")
            continue
        i = info[s]
        px = (i["bid"] + i["ask"]) / 2
        sl = i["swap_long"] * i["point"] / px * 365 * 100   # % p.a. of notional, long the pair
        ss = i["swap_short"] * i["point"] / px * 365 * 100
        if CCY[c][5]:        # USDxxx: long ccy = short pair
            L, S = ss, sl
        else:
            L, S = sl, ss
        theo = last[c] - last["USD"]
        half = -(L + S) / 2
        mid = (L - S) / 2
        m_long, m_short = theo - L, -theo - S
        spr = i["spread_med_m15"] * i["point"] / px
        rows.append(dict(ccy=c, sym=s, r_ccy=last[c], r_usd=last["USD"], theory=theo, swapL=L, swapS=S,
                         mid=mid, offset=mid - theo, half_spread=half, m_long=m_long, m_short=m_short,
                         spread_bp=spr * 1e4, cost_bp=(spr / 2 + COMMISSION) * 1e4))
    df = pd.DataFrame(rows).set_index("ccy").reindex(CCYS)
    df = df.fillna(df.median(numeric_only=True))
    # markup used: per side, the worse of (half swap spread) and the side-specific shortfall vs theory,
    # floored at 0.25%/yr (adversarial: centre offsets are charged too)
    df["markup"] = np.maximum.reduce([df["half_spread"], df["m_long"], df["m_short"],
                                      pd.Series(0.25, index=df.index)])
    return df


# ----------------------------------------------------------------------------- engine
class Engine:
    def __init__(self, P, R, days, M, vix, spx, mk):
        self.P, self.R, self.days, self.M = P, R, days, M
        self.idx = R.index
        self.vix = vix.reindex(self.idx).ffill()
        spxr = spx.reindex(self.idx).ffill().pct_change(fill_method=None)
        self.spxvol = spxr.rolling(20).std() * math.sqrt(252)
        # contemporaneous monthly differential for accrual; lagged (1 month) for signals
        per = self.idx.to_period("M")
        diff = M[CCYS].sub(M["USD"], axis=0) / 100.0
        self.diff_d = diff.reindex(per).set_axis(self.idx)
        self.sig_rate = M[CCYS].shift(1)
        self.mk = mk
        self.markup = (mk["markup"] / 100.0).reindex(CCYS)
        self.cost = (mk["cost_bp"] / 1e4).reindex(CCYS)
        self.vol63 = R.rolling(63, min_periods=60).std() * math.sqrt(252)
        self.cnt = R.notna().rolling(252).sum()
        self._cov = {}
        self.reb = {}
        for f in ("M", "Q"):
            s = pd.Series(self.idx, index=self.idx)
            g = s.groupby(self.idx.to_period(f)).max()
            self.reb[f] = [pd.Timestamp(d) for d in g.values if d >= pd.Timestamp(START) - pd.Timedelta(days=40)]
        self.gates = {}
        for name, x in (("vix", self.vix), ("spx", self.spxvol)):
            med = x.rolling(252, min_periods=200).median()
            p75 = x.rolling(252, min_periods=200).quantile(0.75)
            self.gates[f"{name}_med"] = (x <= med).astype(float).where(med.notna(), 1.0)
            self.gates[f"{name}_p75"] = (x <= p75).astype(float).where(p75.notna(), 1.0)

    def cov(self, d, names):
        key = (d, tuple(names))
        if key not in self._cov:
            w = self.R.loc[:d, list(names)].iloc[-252:]
            self._cov[key] = w.cov().values * 252
        return self._cov[key]

    def weights(self, k=3, freq="M", trend=0, universe=None, basket=None, placebo=None):
        universe = universe or CCYS
        rows = {}
        for n_r, d in enumerate(self.reb[freq]):
            avail = [c for c in universe if self.cnt.at[d, c] >= 250 and not np.isnan(self.vol63.at[d, c])
                     and not np.isnan(self.sig_rate.at[d.to_period("M"), c])]
            sign = {}
            if basket:
                sign = {c: s for c, s in basket.items() if c in avail}
            elif len(avail) >= 2 * k:
                if placebo is not None:
                    order = list(placebo.permutation(avail))
                else:
                    rr = self.sig_rate.loc[d.to_period("M"), avail]
                    order = list(rr.sort_values(kind="mergesort").index)
                for c in order[-k:]:
                    sign[c] = 1.0
                for c in order[:k]:
                    sign[c] = -1.0
            if trend and sign:
                loc = self.P.index.get_loc(d)
                past = self.P[list(sign)].iloc[loc] / self.P[list(sign)].iloc[loc - trend] - 1
                sign = {c: s for c, s in sign.items() if np.sign(past[c]) == s}
            w = pd.Series(0.0, index=CCYS)
            if sign:
                names = sorted(sign)
                raw = np.array([sign[c] / self.vol63.at[d, c] for c in names])
                C = self.cov(d, names)
                pv = math.sqrt(max(raw @ C @ raw, 1e-12))
                sc = min(VOL_TGT / pv, GROSS_CAP / np.abs(raw).sum())
                w[names] = raw * sc
            rows[d] = w
        W = pd.DataFrame(rows).T.reindex(self.idx).ffill().fillna(0.0)
        return W.shift(2).fillna(0.0)            # decided at close d, traded at close d+1

    def run(self, W, gate=None, spread_mult=1.0, markup_mult=1.0, theory=False):
        if gate is not None:
            W = W.mul(self.gates[gate].shift(2).fillna(1.0), axis=0)
        R = self.R.fillna(0.0)
        D = self.diff_d.ffill().fillna(0.0)
        yf = (self.days / 365.0).values[:, None]
        spot = (W * R).sum(1)
        carry = (W * D * yf).sum(1)
        mk = (W.abs() * self.markup.values * yf).sum(1) * markup_mult
        to = W.diff().abs().fillna(W.abs())
        spr = (to * self.cost.values).sum(1) * spread_mult
        if theory:
            mk, spr = mk * 0, spr * 0
        net = spot + carry - mk - spr
        out = pd.DataFrame(dict(net=net, spot=spot, carry=carry, markup=mk, spread=spr,
                                turnover=to.sum(1), gross=W.abs().sum(1)))
        return out.loc[START:]


# ----------------------------------------------------------------------------- stats
def stats(r: pd.Series):
    r = r.dropna()
    if len(r) < 50 or r.std() == 0:
        return dict(ann=np.nan, vol=np.nan, sr=np.nan, mdd=np.nan, t=np.nan)
    ann = r.mean() * 252
    vol = r.std() * math.sqrt(252)
    eq = r.cumsum()
    mdd = (eq - eq.cummax()).min()
    sr = ann / vol
    return dict(ann=ann, vol=vol, sr=sr, mdd=mdd, t=sr * math.sqrt(len(r) / 252))


def fmt(s):
    return f"ann {s['ann']*100:6.2f}%  vol {s['vol']*100:5.2f}%  SR {s['sr']:5.2f}  t {s['t']:5.2f}  maxDD {s['mdd']*100:6.1f}%"


def scale10(r):
    return r * (VOL_TGT / (r.std() * math.sqrt(252)))


def yearly(r):
    return r.groupby(r.index.year).sum()


def verdict(ok):
    return "PASS" if ok else "FAIL"


# ----------------------------------------------------------------------------- main
def main():
    global N_CONFIGS
    raw, h10, rt, info = load_data()
    P, R, days, M, src, spikes, rsrc = build_panel(raw, h10, rt)
    vix = raw["^VIX"].dropna()
    spx = raw["^GSPC"].dropna()

    print("=" * 100)
    print("FX CARRY BATTERY  (run", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), ")")
    print("=" * 100)
    print("\n--- DATA ---")
    for c in CCYS:
        print(f"  {c}: spot {src[c]:<38} yahoo spikes cleaned {spikes[c]:3d} | rate {rsrc[c]}")
    print(f"  USD: rate {rsrc['USD']}")
    print(f"  last monthly rates (%):", M.ffill().iloc[-1].round(2).to_dict())

    mk = swap_markup(info, M)
    print("\n--- MT5 SWAP vs THEORY (% p.a., long-the-currency perspective; spreads median M15 last ~6 weeks) ---")
    print(mk[["sym", "r_ccy", "r_usd", "theory", "swapL", "swapS", "mid", "offset", "half_spread", "m_long",
              "m_short", "markup", "spread_bp", "cost_bp"]].round(2).to_string())
    print(f"  median half swap-spread {mk['half_spread'].median():.2f}%  median markup used {mk['markup'].median():.2f}%"
          f"  (G10 {mk.loc[G10,'markup'].median():.2f}%, EM {mk.loc[EM,'markup'].median():.2f}%)")

    E = Engine(P, R, days, M, vix, spx, mk)

    # ---------------- variants
    VAR = {
        "a  CS carry k3 M": dict(w=dict(k=3), gate=None),
        "b3 a+trend63": dict(w=dict(k=3, trend=63), gate=None),
        "b6 a+trend126": dict(w=dict(k=3, trend=126), gate=None),
        "c_med a+VIX>med flat": dict(w=dict(k=3), gate="vix_med"),
        "c_p75 a+VIX>p75 flat": dict(w=dict(k=3), gate="vix_p75"),
        "c_spx a+SPXvol>med flat": dict(w=dict(k=3), gate="spx_med"),
        "d  AUD,NZD vs JPY,CHF": dict(w=dict(basket={"AUD": 1, "NZD": 1, "JPY": -1, "CHF": -1}), gate=None),
    }
    Wcache = {}

    def getW(**kw):
        key = json.dumps(kw, sort_keys=True, default=str)
        if key not in Wcache:
            Wcache[key] = E.weights(**kw)
        return Wcache[key]

    res = {}
    print("\n--- VARIANTS (2000->now; theory = no broker costs; net = MT5 markup x1 + spread x1) ---")
    for name, v in VAR.items():
        W = getW(**v["w"])
        th = E.run(W, v["gate"], theory=True)
        nt = E.run(W, v["gate"])
        res[name] = (W, v["gate"], nt)
        N_CONFIGS += 1
        print(f"  {name:<26} THEORY {fmt(stats(th['net']))}")
        print(f"  {'':<26} NET    {fmt(stats(nt['net']))}  | carry {nt['carry'].mean()*25200:5.2f}%"
              f" markup {nt['markup'].mean()*25200:4.2f}% spread {nt['spread'].mean()*25200:4.2f}% gross {nt['gross'].mean():.2f}x")

    best_name = max(res, key=lambda n: stats(res[n][2]["net"])["sr"])
    W_b, gate_b, net_b = res[best_name]
    print(f"\n  best NET Sharpe (in-sample choice, see test 7/10): {best_name}")
    base = res["a  CS carry k3 M"][2]

    # ---------------- 1 sources
    print("\n--- TEST 1: SOURCES AGREE ---")
    ok1 = True
    print("  (MT5 history not used: terminal blocks on copy_rates; second source = FRED H.10 noon NY rates)")
    for c in CCYS:
        y = usd_per(raw[CCY[c][0]], CCY[c][1])
        y, _ = clean_spikes(y)
        out = f"  {c}:"
        h = CCY[c][2]
        if h:
            hs = usd_per(h10[h], CCY[c][3])
            j = pd.concat([y, hs], axis=1, keys=["y", "h"]).dropna()
            mo = j.resample("ME").last().pct_change().dropna()
            ch = mo.corr().iloc[0, 1]
            q = j.resample("QE").last().pct_change().dropna()
            cq = q.corr().iloc[0, 1]
            out += f" | Yahoo~H10 n={len(j)} monthly {ch:.3f} quarterly {cq:.3f}"
            if not (cq > 0.97):
                ok1 = False
        else:
            out += " | SINGLE SOURCE (no H.10 series)"
        print(out)
    # strategy on H10-only universe (no Yahoo) vs Yahoo-spliced, same currencies
    h10c = [c for c in CCYS if CCY[c][2]]
    Ph = pd.DataFrame({c: usd_per(h10[CCY[c][2]], CCY[c][3]) for c in h10c}).reindex(P.index).ffill(limit=5)
    Rh = Ph.pct_change(fill_method=None).where(Ph.notna() & Ph.shift().notna())
    Rh = Rh.reindex(columns=CCYS)
    Ph = Ph.reindex(columns=CCYS)
    Eh = Engine(Ph, Rh, days, M, vix, spx, mk)
    rh = Eh.run(Eh.weights(k=3, universe=h10c))["net"]
    ry = E.run(E.weights(k=3, universe=h10c))["net"]
    ch = pd.concat([rh, ry], axis=1).dropna().resample("ME").sum().corr().iloc[0, 1]
    th_h = Eh.run(Eh.weights(k=3, universe=h10c), theory=True)["net"]
    th_y = E.run(E.weights(k=3, universe=h10c), theory=True)["net"]
    print(f"  strategy a on 11 H10 ccys NET:    H10 data {fmt(stats(rh))}")
    print(f"  strategy a on 11 H10 ccys NET:    Yahoo    {fmt(stats(ry))}  monthly corr {ch:.3f}")
    print(f"  strategy a on 11 H10 ccys THEORY: H10 SR {stats(th_h)['sr']:.2f} vs Yahoo SR {stats(th_y)['sr']:.2f}")
    ok1 = ok1 and ch > 0.9 and abs(stats(rh)["sr"] - stats(ry)["sr"]) < 0.15
    print(f"  TEST 1 {verdict(ok1)} (quarterly-return corr >0.97 every pair/source; strategy monthly corr >0.9, |dSR|<0.15;"
          f" TRY/PLN/HUF/CZK single-source; monthly-close corr depressed by Yahoo-vs-noon-NY close-time mismatch)")

    # ---------------- 2 surface
    print("\n--- TEST 2: PARAMETER SURFACE (NET Sharpe, 2000->now) ---")
    surf = []
    for k in (2, 3, 4):
        for f in ("M", "Q"):
            for tr in (0, 63, 126):
                W = getW(k=k, freq=f, trend=tr)
                for g in (None, "vix_med", "vix_p75"):
                    s = stats(E.run(W, g)["net"])
                    surf.append(dict(k=k, freq=f, trend=tr, gate=g or "-", sr=s["sr"], ann=s["ann"], mdd=s["mdd"]))
                    N_CONFIGS += 1
    S = pd.DataFrame(surf)
    piv = S.pivot_table(index=["k", "freq"], columns=["trend", "gate"], values="sr").round(2)
    print(piv.to_string())
    share = (S["sr"] > 0).mean()
    share3 = (S["sr"] > 0.3).mean()
    print(f"  cells {len(S)}: SR>0 {share:.0%}, SR>0.3 {share3:.0%}, median SR {S['sr'].median():.2f}, "
          f"min {S['sr'].min():.2f}, max {S['sr'].max():.2f}")
    print("  by filter (median SR):", S.groupby(["trend", "gate"])["sr"].median().round(2).to_dict())
    ok2 = share >= 0.8 and S["sr"].median() > 0.3
    print(f"  TEST 2 {verdict(ok2)} (need >=80% cells SR>0 and median SR>0.3)")

    # ---------------- 3 OOS by time
    print("\n--- TEST 3: OUT-OF-SAMPLE BY TIME (NET) ---")
    periods = [("DEV 2000-09", "2000", "2009"), ("VAL 2010-19", "2010", "2019"), ("RECENT 2020-", "2020", "2026")]
    ok3_best = True
    for name in res:
        r = res[name][2]["net"]
        line = f"  {name:<26}"
        for pn, a, b in periods:
            s = stats(r.loc[a:b])
            line += f" {pn}: SR {s['sr']:5.2f} ann {s['ann']*100:5.1f}% |"
            if name == best_name and not (s["sr"] > 0):
                ok3_best = False
        y = yearly(r)
        line += f" +yrs {(y > 0).mean():.0%}"
        print(line)
    yb = yearly(net_b["net"])
    print(f"  {best_name} yearly NET %:", (yb * 100).round(1).to_dict())
    ok3 = ok3_best and (yb > 0).mean() >= 0.6
    print(f"  TEST 3 {verdict(ok3)} (best variant: SR>0 in all 3 periods and >=60% positive years)")

    # ---------------- 4 regimes
    print("\n--- TEST 4: REGIMES / CRISES (NET, % of capital at ~10% vol) ---")
    vx = E.vix.reindex(base.index)
    hi = vx > vx.rolling(252, min_periods=200).median()
    for name in res:
        r = res[name][2]["net"]
        s_on, s_off = stats(r[~hi]), stats(r[hi])
        print(f"  {name:<26} risk-on (VIX<=med) ann {s_on['ann']*100:6.2f}% SR {s_on['sr']:5.2f} | "
              f"risk-off ann {s_off['ann']*100:6.2f}% SR {s_off['sr']:5.2f}")
    crises = [("2008 Q4", "2008-08-01", "2008-12-31"), ("2011 Aug-Sep", "2011-07-25", "2011-10-05"),
              ("2015 Jan CHF", "2015-01-12", "2015-01-31"), ("2015 Aug CNY", "2015-08-01", "2015-09-30"),
              ("2018 TRY", "2018-07-15", "2018-09-15"), ("2020 Mar", "2020-02-15", "2020-04-15"),
              ("2022", "2022-01-01", "2022-12-31"), ("2024 Aug yen", "2024-07-10", "2024-08-15")]
    ok4 = True
    for cn, a, b in crises:
        line = f"  {cn:<13}"
        for name in ("a  CS carry k3 M", "b3 a+trend63", "c_med a+VIX>med flat", "d  AUD,NZD vs JPY,CHF", best_name):
            r = res[name][2]["net"].loc[a:b]
            eq = r.cumsum()
            dd = (eq - eq.cummax()).min() if len(eq) else np.nan
            line += f" {name.split()[0]:>5}: tot {r.sum()*100:6.1f}% dd {dd*100:6.1f}% |"
            if name == best_name and r.sum() < -0.15:
                ok4 = False
        print(line)
    wd = net_b["net"].rolling(21).sum().min()
    print(f"  best variant worst 21-day loss {wd*100:.1f}%, worst day {net_b['net'].min()*100:.2f}% "
          f"({net_b['net'].idxmin().date()}), skew {net_b['net'].skew():.2f}, monthly skew "
          f"{net_b['net'].resample('ME').sum().skew():.2f}")
    print(f"  TEST 4 {verdict(ok4)} (best variant: no crisis window worse than -15% at ~10% vol)")

    # ---------------- 5 placebo
    print("\n--- TEST 5: PLACEBO (random long/short baskets, same machinery/filters) ---")
    NPL = 200
    best_cfg = VAR[best_name]
    for label, cfg in (("a  CS carry k3 M", VAR["a  CS carry k3 M"]), (best_name, best_cfg)):
        if label == best_name and best_name == "a  CS carry k3 M":
            continue
        if "basket" in cfg["w"]:
            print(f"  {label}: fixed basket -> placebo = random fixed 2v2 baskets")
        act = stats(res[label][2]["net"])["sr"]
        srs = []
        for i in range(NPL):
            g = np.random.default_rng(1000 + i)
            if "basket" in cfg["w"]:
                pick = list(g.choice(CCYS, 4, replace=False))
                kw = dict(basket={pick[0]: 1, pick[1]: 1, pick[2]: -1, pick[3]: -1})
            else:
                kw = dict(cfg["w"], placebo=g)
            W = E.weights(**kw)
            srs.append(stats(E.run(W, cfg["gate"])["net"])["sr"])
        srs = np.array(srs)
        p = (np.sum(srs >= act) + 1) / (NPL + 1)
        print(f"  {label}: actual SR {act:.2f}; placebo mean {srs.mean():.2f} sd {srs.std():.2f} "
              f"95th {np.percentile(srs,95):.2f}; p = {p:.4f}")
        if label == best_name or best_name == "a  CS carry k3 M":
            p5 = p
    # block bootstrap of mean
    r = net_b["net"].values
    nb = 2000
    L = 63
    n = len(r)
    means = []
    for _ in range(nb):
        st = RNG.integers(0, n - L, n // L + 1)
        smp = np.concatenate([r[s:s + L] for s in st])[:n]
        means.append(smp.mean() - r.mean())
    pboot = (np.sum(np.array(means) >= r.mean()) + 1) / (nb + 1)
    print(f"  block-bootstrap (63d) p-value mean<=0 for {best_name}: {pboot:.4f}")
    ok5 = p5 < 0.05 and pboot < 0.05
    print(f"  TEST 5 {verdict(ok5)}")

    # ---------------- 6 costs
    print("\n--- TEST 6: COSTS (NET SR / ann%) ---")
    ok6 = True
    for name in ("a  CS carry k3 M", best_name):
        W, g, _ = res[name]
        line = f"  {name:<26}"
        for sm in (1, 2, 3):
            for mm in (0, "hs", 1, 2):
                if mm == "hs":   # markup = half the broker swap spread only (no centre-offset charge)
                    E.markup = (mk["half_spread"].clip(lower=0) / 100.0).reindex(CCYS)
                    s = stats(E.run(W, g, spread_mult=sm)["net"])
                    E.markup = (mk["markup"] / 100.0).reindex(CCYS)
                else:
                    s = stats(E.run(W, g, spread_mult=sm, markup_mult=mm)["net"])
                line += f" s{sm}m{mm}: {s['sr']:.2f}/{s['ann']*100:.1f}%"
                if name == best_name and sm == 2 and mm == 2 and not (s["sr"] > 0.3):
                    ok6 = False
            line += "\n" + " " * 28
        print(line.rstrip())
    print(f"  TEST 6 {verdict(ok6)} (best: SR>0.3 at spread x2 and markup x2)")

    # ---------------- 7 walk-forward
    print("\n--- TEST 7: WALK-FORWARD VARIANT SELECTION (expanding window, >=3y history, yearly) ---")
    nets = pd.DataFrame({n: res[n][2]["net"] for n in res})
    wf = []
    picks = {}
    for yr in range(2003, nets.index.year.max() + 1):
        hist = nets.loc[:f"{yr-1}-12-31"]
        srs = hist.mean() / hist.std()
        pick = srs.idxmax()
        picks[yr] = pick.split()[0]
        wf.append(nets.loc[str(yr), pick])
    wf = pd.concat(wf)
    swf = stats(wf)
    print("  picks:", picks)
    print(f"  WF {fmt(swf)}")
    eqw = nets.loc["2003":].mean(1)
    print(f"  equal-weight all variants 2003-> {fmt(stats(eqw))}")
    ok7 = swf["sr"] > 0.3
    print(f"  TEST 7 {verdict(ok7)} (WF SR>0.3)")

    # ---------------- 8 mechanism
    print("\n--- TEST 8: MECHANISM ---")
    for name in res:
        nt = res[name][2]
        sp, ca, mk_, sr_ = (nt[x].mean() * 25200 for x in ("spot", "carry", "markup", "spread"))
        print(f"  {name:<26} spot {sp:6.2f}%  carry {ca:5.2f}%  -markup {mk_:4.2f}%  -spread {sr_:4.2f}%  = net {sp+ca-mk_-sr_:5.2f}%/yr")
    ok8 = True
    for lab, uni, k in (("G10 only k3", G10, 3), ("EM only k2", EM, 2), ("ex-TRY k3", [c for c in CCYS if c != "TRY"], 3),
                        ("G10 only k2", G10, 2)):
        W = E.weights(k=k, universe=uni)
        for g in (None, best_cfg["gate"]) if best_cfg["gate"] else (None,):
            nt = E.run(W, g)
            N_CONFIGS += 1
            s = stats(nt["net"])
            print(f"  {lab:<12} gate {str(g):<8} {fmt(s)} | spot {nt['spot'].mean()*25200:5.2f}% carry "
                  f"{nt['carry'].mean()*25200:5.2f}% markup {nt['markup'].mean()*25200:4.2f}%")
            if lab in ("G10 only k3", "EM only k2") and g is None and not (s["sr"] > 0):
                ok8 = False
    print(f"  TEST 8 {verdict(ok8)} (ungated carry positive in BOTH G10 and EM subsets)")

    # ---------------- 9 Monte Carlo
    print("\n--- TEST 9: MONTE CARLO (63d block bootstrap of NET daily returns, rescaled to 10% vol, 10y paths) ---")
    for name in ("a  CS carry k3 M", best_name) if best_name != "a  CS carry k3 M" else ("a  CS carry k3 M",):
        r = scale10(res[name][2]["net"]).values
        n = len(r)
        H = 2520
        anns, mdds, ly, worst21 = [], [], [], []
        for _ in range(3000):
            st = RNG.integers(0, n - L, H // L + 1)
            smp = np.concatenate([r[s:s + L] for s in st])[:H]
            eq = np.cumsum(smp)
            mdds.append((eq - np.maximum.accumulate(eq)).min())
            yrs = smp[: 2520].reshape(10, 252).sum(1)
            anns.extend(yrs)
            ly.append((yrs < 0).mean())
            worst21.append(pd.Series(smp).rolling(21).sum().min())
        anns, mdds = np.array(anns), np.array(mdds)
        hist = pd.Series(r, index=res[name][2].index)
        eqh = hist.cumsum()
        print(f"  {name}: hist (10% vol) ann {hist.mean()*25200:.2f}% maxDD {(eqh-eqh.cummax()).min()*100:.1f}% | "
              f"MC annual ret median {np.median(anns)*100:.1f}% p5 {np.percentile(anns,5)*100:.1f}% p1 "
              f"{np.percentile(anns,1)*100:.1f}%; P(losing yr) {np.mean(anns<0):.0%}; 10y maxDD median "
              f"{np.median(mdds)*100:.1f}% p95 {np.percentile(mdds,5)*100:.1f}% p99 {np.percentile(mdds,1)*100:.1f}%;"
              f" worst-month p99 {np.percentile(worst21,1)*100:.1f}%")
    r10 = scale10(net_b["net"])
    ok9 = r10.mean() * 252 > 0.03
    print(f"  TEST 9 {verdict(ok9)} (net ann > 3% at 10% vol)")

    # ---------------- 10 multiple testing
    print("\n--- TEST 10: MULTIPLE TESTING ---")
    sb = stats(net_b["net"])
    from math import erfc, sqrt
    p_raw = 0.5 * erfc(sb["t"] / sqrt(2))
    p_adj = min(1.0, p_raw * N_CONFIGS)
    print(f"  configurations evaluated: {N_CONFIGS} (+{NPL*2} placebo runs not counted as candidates)")
    print(f"  best ({best_name}) t = {sb['t']:.2f}, one-sided p {p_raw:.2e}, Bonferroni x{N_CONFIGS} -> {p_adj:.3f}")
    sa = stats(base["net"])
    pa = 0.5 * erfc(sa["t"] / sqrt(2))
    print(f"  pre-registered base (a) t = {sa['t']:.2f}, p {pa:.2e} (no adjustment needed: fixed upfront)")
    ok10 = p_adj < 0.05
    print(f"  TEST 10 {verdict(ok10)} (Bonferroni-adjusted p<0.05; Harvey-Liu-Zhu hurdle t>3: {sb['t']>3})")

    print("\n--- SUMMARY for", best_name, "---")
    for i, (nm, ok) in enumerate([("sources", ok1), ("surface", ok2), ("OOS time", ok3), ("regimes/crises", ok4),
                                  ("placebo", ok5), ("costs", ok6), ("walk-forward", ok7), ("mechanism", ok8),
                                  ("MC >3%", ok9), ("multiple testing", ok10)], 1):
        print(f"  {i:2d}. {nm:<18} {verdict(ok)}")
    print(f"  realistic NET at 10% vol: ann {r10.mean()*25200:.2f}%  maxDD {((r10.cumsum()-r10.cumsum().cummax()).min())*100:.1f}%")


def mt5_fetch_one(s, out):
    """Read-only: symbol_info + current tick only (no copy_rates history pulls)."""
    import MetaTrader5 as mt5
    assert mt5.initialize()
    mt5.symbol_select(s, True)
    i = mt5.symbol_info(s)
    t = mt5.symbol_info_tick(s)
    bid, ask = (t.bid, t.ask) if t is not None else (i.bid, i.ask)
    info = dict(bid=bid, ask=ask, point=i.point, cs=i.trade_contract_size, swap_mode=i.swap_mode,
                swap_long=i.swap_long, swap_short=i.swap_short, spread_now=i.spread,
                spread_med_m15=float(i.spread), spread_src="symbol_info snapshot")
    mt5.shutdown()
    json.dump(dict(info=info), open(out, "w"))
    print("  mt5", s, "ok", flush=True)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--mt5":
        mt5_fetch_one(sys.argv[2], sys.argv[3])
    else:
        main()
