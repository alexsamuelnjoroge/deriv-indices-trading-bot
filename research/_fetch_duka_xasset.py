"""Fetch Dukascopy M5 bid into data/duka/<NAME>_M5_<YEAR>.csv.
Usage: .venv/Scripts/python.exe research/_fetch_duka_xasset.py [NAME ...] [--from YEAR] [--to YEAR]
Default: gold's cross-asset drivers (XAGUSD DXY USTBOND), 2018-2026.
"""
import sys
from datetime import datetime, timezone

import pandas as pd
from dukascopy_python import instruments as ins

from _fetch_duka_fx import CACHE, END, fetch_range, ok

ALL = {
    "XAGUSD": ins.INSTRUMENT_FX_METALS_XAG_USD,
    "DXY": ins.INSTRUMENT_IDX_AMERICA_DOLLAR_IDX_USD,
    "USTBOND": ins.INSTRUMENT_BND_CFD_USTBOND_TR_USD,
    "frxXAUUSD": ins.INSTRUMENT_FX_METALS_XAU_USD,
}
args = sys.argv[1:]
y0 = int(args[args.index("--from") + 1]) if "--from" in args else 2018
y1 = int(args[args.index("--to") + 1]) if "--to" in args else 2026
names = [a for a in args if a in ALL] or ["XAGUSD", "DXY", "USTBOND"]
TARGETS = {n: ALL[n] for n in names}

for name, inst in TARGETS.items():
    for y in range(y0, y1 + 1):
        f = CACHE / f"{name}_M5_{y}.csv"
        if f.exists() and ok(f, y):
            continue
        a = datetime(y, 1, 1, tzinfo=timezone.utc)
        b = min(datetime(y + 1, 1, 1, tzinfo=timezone.utc), END)
        parts, m = [], a
        while m < b:
            n = datetime(m.year + (m.month == 12), m.month % 12 + 1, 1, tzinfo=timezone.utc)
            parts.append(fetch_range(inst, m, min(n, b)))
            m = n
        df = pd.concat(parts)
        df = df[~df.index.duplicated()].sort_index()
        df.to_csv(f)
        print(f"{name} {y}: {len(df)} bars, ok={ok(f, y)}", flush=True)
