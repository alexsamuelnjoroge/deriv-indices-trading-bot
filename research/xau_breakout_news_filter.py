"""
News-blackout filters for the gold Asian-range breakout (incubator rule D), pre-registered 2026-11-16.

Event calendars:
  NFP and CPI release dates from the BLS release archives (bls.gov/bls/news-release/empsit.htm,
  cpi.htm), both 08:30 ET. The list was transcribed by a summariser, so every date is checked
  against the market: a real 08:30 release shows an abnormal gold move 08:30-08:35 ET. Dates
  whose move is < 2.5x the median for that clock are replaced by the weekday within +/-4 days
  with the largest such move (if that one is >= 2.5x); counts are printed.
  FOMC announcement dates 2008-2026 from research/index_calendar_battery.py (14:00 ET).
Variants (server dates; server = ET + 7h):
  F1 skip NFP + CPI days      F2 skip NFP + CPI + FOMC days
  F3 NFP/CPI days: cancel orders and close at 08:15 ET (15:15 server), 15 min before release
  F4 skip FOMC days only
Judged like the loss-reduction test: mean R, R/yr, max drawdown in R on Dukascopy 2008-16 and
2017-26 and on IC MT5 2022-26. A filter helps only if it raises mean R without cutting R/yr
much, or cuts drawdown, in BOTH Dukascopy periods.
"""

import ast
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xau_session_breakout_ea as base  # noqa: E402
from xasset_breakout_battery import ASIA, load_mt5_m15, run  # noqa: E402

COST = 0.36
NFP_RAW = """2008-01-04 2008-02-01 2008-03-07 2008-04-04 2008-05-02 2008-06-06 2008-07-03 2008-08-01 2008-09-05 2008-10-03 2008-11-07 2008-12-05
2009-01-09 2009-02-06 2009-03-06 2009-04-03 2009-05-08 2009-06-05 2009-07-02 2009-08-07 2009-09-04 2009-10-02 2009-11-06 2009-12-04
2010-01-08 2010-02-05 2010-03-05 2010-04-02 2010-05-07 2010-06-04 2010-07-02 2010-08-06 2010-09-03 2010-10-08 2010-11-05 2010-12-03
2011-01-07 2011-02-04 2011-03-04 2011-04-01 2011-05-06 2011-06-03 2011-07-08 2011-08-05 2011-09-02 2011-10-07 2011-11-04 2011-12-02
2012-01-06 2012-02-03 2012-03-09 2012-04-06 2012-05-04 2012-06-01 2012-07-06 2012-08-03 2012-09-07 2012-10-05 2012-11-02 2012-12-07
2013-01-04 2013-02-01 2013-03-08 2013-04-05 2013-05-03 2013-06-07 2013-07-05 2013-08-02 2013-09-06 2013-10-22 2013-11-08 2013-12-06
2014-01-10 2014-02-07 2014-03-07 2014-04-04 2014-05-02 2014-06-06 2014-07-03 2014-08-01 2014-09-05 2014-10-03 2014-11-07 2014-12-05
2015-01-09 2015-02-06 2015-03-06 2015-04-03 2015-05-08 2015-06-05 2015-07-02 2015-08-07 2015-09-04 2015-10-02 2015-11-06 2015-12-04
2016-01-08 2016-02-05 2016-03-04 2016-04-01 2016-05-06 2016-06-03 2016-07-08 2016-08-05 2016-09-02 2016-10-07 2016-11-04 2016-12-02
2017-01-06 2017-02-03 2017-03-10 2017-04-07 2017-05-05 2017-06-02 2017-07-07 2017-08-04 2017-09-01 2017-10-06 2017-11-03 2017-12-08
2018-01-05 2018-02-02 2018-03-09 2018-04-06 2018-05-04 2018-06-01 2018-07-06 2018-08-03 2018-09-07 2018-10-05 2018-11-02 2018-12-07
2019-01-04 2019-02-01 2019-03-08 2019-04-05 2019-05-03 2019-06-07 2019-07-05 2019-08-02 2019-09-06 2019-10-04 2019-11-01 2019-12-06
2020-01-10 2020-02-07 2020-03-06 2020-04-03 2020-05-08 2020-06-05 2020-07-02 2020-08-07 2020-09-04 2020-10-02 2020-11-06 2020-12-04
2021-01-08 2021-02-05 2021-03-05 2021-04-02 2021-05-07 2021-06-04 2021-07-02 2021-08-06 2021-09-03 2021-10-08 2021-11-05 2021-12-03
2022-01-07 2022-02-04 2022-03-04 2022-04-01 2022-05-06 2022-06-03 2022-07-08 2022-08-05 2022-09-02 2022-10-07 2022-11-04 2022-12-02
2023-01-06 2023-02-03 2023-03-10 2023-04-07 2023-05-05 2023-06-02 2023-07-07 2023-08-04 2023-09-01 2023-10-06 2023-11-03 2023-12-08
2024-01-05 2024-02-02 2024-03-08 2024-04-05 2024-05-03 2024-06-07 2024-07-05 2024-08-02 2024-09-06 2024-10-04 2024-11-01 2024-12-06
2025-01-10 2025-02-07 2025-03-07 2025-04-04 2025-05-02 2025-06-06 2025-07-03 2025-08-01 2025-09-05 2025-11-20 2025-12-16
2026-01-09 2026-02-11 2026-03-06 2026-04-03 2026-05-08 2026-06-05 2026-07-02 2026-08-07 2026-09-04 2026-10-02"""
CPI_RAW = """2008-01-16 2008-02-20 2008-03-14 2008-04-16 2008-05-14 2008-06-13 2008-07-16 2008-08-14 2008-09-16 2008-10-16 2008-11-19 2008-12-16
2009-01-16 2009-02-20 2009-03-18 2009-04-15 2009-05-15 2009-06-17 2009-07-15 2009-08-14 2009-09-16 2009-10-15 2009-11-18 2009-12-16
2010-01-15 2010-02-19 2010-03-18 2010-04-14 2010-05-19 2010-06-17 2010-07-16 2010-08-13 2010-09-17 2010-10-15 2010-11-17 2010-12-15
2011-01-14 2011-02-17 2011-03-17 2011-04-15 2011-05-13 2011-06-15 2011-07-15 2011-08-18 2011-09-15 2011-10-19 2011-11-16 2011-12-16
2012-01-19 2012-02-17 2012-03-16 2012-04-13 2012-05-15 2012-06-14 2012-07-17 2012-08-15 2012-09-14 2012-10-16 2012-11-15 2012-12-14
2013-01-16 2013-02-21 2013-03-15 2013-04-16 2013-05-16 2013-06-18 2013-07-16 2013-08-15 2013-09-17 2013-10-30 2013-11-20 2013-12-17
2014-01-16 2014-02-20 2014-03-18 2014-04-15 2014-05-15 2014-06-17 2014-07-22 2014-08-19 2014-09-17 2014-10-22 2014-11-20 2014-12-17
2015-01-16 2015-02-26 2015-03-24 2015-04-17 2015-05-22 2015-06-18 2015-07-17 2015-08-19 2015-09-16 2015-10-15 2015-11-17 2015-12-15
2016-01-20 2016-02-19 2016-03-16 2016-04-14 2016-05-17 2016-06-16 2016-07-15 2016-08-16 2016-09-16 2016-10-18 2016-11-17 2016-12-15
2017-01-18 2017-02-15 2017-03-15 2017-04-14 2017-05-12 2017-06-14 2017-07-14 2017-08-11 2017-09-14 2017-10-13 2017-11-15 2017-12-13
2018-01-12 2018-02-14 2018-03-13 2018-04-11 2018-05-10 2018-06-12 2018-07-12 2018-08-10 2018-09-13 2018-10-11 2018-11-14 2018-12-12
2019-01-11 2019-02-13 2019-03-12 2019-04-10 2019-05-10 2019-06-12 2019-07-11 2019-08-13 2019-09-12 2019-10-10 2019-11-13 2019-12-11
2020-01-14 2020-02-13 2020-03-11 2020-04-10 2020-05-12 2020-06-10 2020-07-14 2020-08-12 2020-09-11 2020-10-13 2020-11-12 2020-12-10
2021-01-13 2021-02-10 2021-03-10 2021-04-13 2021-05-12 2021-06-10 2021-07-13 2021-08-11 2021-09-14 2021-10-13 2021-11-10 2021-12-10
2022-01-12 2022-02-10 2022-03-10 2022-04-12 2022-05-11 2022-06-10 2022-07-13 2022-08-10 2022-09-13 2022-10-13 2022-11-10 2022-12-13
2023-01-12 2023-02-14 2023-03-14 2023-04-12 2023-05-10 2023-06-13 2023-07-12 2023-08-10 2023-09-13 2023-10-12 2023-11-14 2023-12-12
2024-01-11 2024-02-13 2024-03-12 2024-04-10 2024-05-15 2024-06-12 2024-07-11 2024-08-14 2024-09-11 2024-10-10 2024-11-13 2024-12-11
2025-01-15 2025-02-12 2025-03-12 2025-04-10 2025-05-13 2025-06-11 2025-07-15 2025-08-12 2025-09-11 2025-10-24 2025-12-18
2026-01-13 2026-02-13 2026-03-11 2026-04-10 2026-05-12 2026-06-10 2026-07-14 2026-08-12 2026-09-11 2026-10-14"""


def fomc_dates():
    src = Path(__file__).resolve().parent.joinpath("index_calendar_battery.py").read_text(encoding="utf-8")
    raw = ast.literal_eval(re.search(r"FOMC_RAW = (\{.*?\n\})", src, re.S).group(1))
    return {pd.to_datetime(f"{d} {y}", format="%b %d %Y").date()
            for y, s in raw.items() if y >= 2008 for d in s.split(",")}


def release_move():
    """|gold move| over the 5-min bar starting 08:30 ET (15:30 server), per server date."""
    m5 = base.m5
    bars = m5[(m5.index.hour == 15) & (m5.index.minute == 30)]
    mv = (bars["close"] / bars["open"] - 1).abs() * 1e4
    return pd.Series(mv.values, index=bars.index.date)


def validated(raw, mv, label):
    med = mv.median()
    ratio = (mv / med).to_dict()
    kept = fixed = weak = 0
    out = set()
    for s in raw.split():
        d = pd.Timestamp(s).date()
        if ratio.get(d, 0) >= 2.5:
            out.add(d)
            kept += 1
            continue
        cands = [(ratio.get((pd.Timestamp(d) + pd.Timedelta(days=k)).date(), 0), (pd.Timestamp(d) + pd.Timedelta(days=k)).date())
                 for k in range(-4, 5) if (pd.Timestamp(d) + pd.Timedelta(days=k)).weekday() < 5]
        best = max(cands)
        if best[0] >= 2.5:
            out.add(best[1])
            fixed += 1
        else:
            out.add(d)
            weak += 1
    print(f"{label}: {kept} dates confirmed by the market, {fixed} corrected to a nearby release day, "
          f"{weak} kept without a clear spike (median 08:30 move {med:.1f}bp)")
    return out


def summarise(tr):
    r = np.array([(x[2] - COST) / x[3] for x in tr])
    if len(r) < 30:
        return "n<30"
    eq = np.cumsum(r)
    dd = (eq - np.maximum.accumulate(eq)).min()
    years = len({x[0].year for x in tr})
    t = r.mean() / r.std(ddof=1) * np.sqrt(len(r))
    return f"n={len(r):4d} mean={r.mean():+.3f}R t={t:+5.2f} R/yr={r.sum() / years:+5.1f} maxDD={dd:6.1f}R"


if __name__ == "__main__":
    mv = release_move()
    nfp = validated(NFP_RAW, mv, "NFP")
    cpi = validated(CPI_RAW, mv, "CPI")
    fomc = fomc_dates()
    data_days = nfp | cpi
    mt = load_mt5_m15("XAUUSD")

    base_tr = run(base.m15, ASIA)
    on = [x for x in base_tr if x[0] in data_days]
    off = [x for x in base_tr if x[0] not in data_days]
    on_f = [x for x in base_tr if x[0] in fomc]
    print("\nBaseline trades on event days vs other days (all 2008-2026):")
    print(f"   NFP/CPI days {summarise(on)}")
    print(f"   FOMC days    {summarise(on_f)}" if len(on_f) >= 30 else f"   FOMC days    n={len(on_f)}")
    print(f"   other days   {summarise(off)}")

    variants = {
        "V0 baseline": dict(),
        "F1 skip NFP+CPI": dict(skip=frozenset(data_days)),
        "F2 skip NFP+CPI+FOMC": dict(skip=frozenset(data_days | fomc)),
        "F3 flat 08:15 ET on NFP/CPI": dict(early_flat={d: 15 * 60 + 15 for d in data_days}),
        "F4 skip FOMC": dict(skip=frozenset(fomc)),
    }
    print()
    for name, kw in variants.items():
        tr = run(base.m15, ASIA, **kw)
        print(name)
        print(f"   Duka 08-16 {summarise([x for x in tr if x[0].year < 2017])}")
        print(f"   Duka 17-26 {summarise([x for x in tr if x[0].year >= 2017])}")
        print(f"   MT5  22-26 {summarise(run(mt, ASIA, **kw))}", flush=True)
