"""
Backtest the two optional mtf_pullback filters (Volume Profile, COT) on XAUUSD.

Variants (all driven through MTFPullbackStrategy itself, config from config.yaml):
  base  : use_vp_filter=false, use_cot_filter=false
  vp    : use_vp_filter=true
  cot   : use_cot_filter=true   (historical COT injected per bar, see below)
  both  : both on

Data
  * XAUUSD M5/H1/H4/D1 from the running MT5 terminal (read-only: copy_rates_from_pos).
    M5 depth is capped by the terminal's "max bars in chart" (100k) -> ~17 months.
    Cached to data/pro/mt5_XAUUSD_{M5,H1,H4,D1}.csv  (--refresh to re-pull).
    Epochs are broker server time, exactly what the live bot feeds the strategy.
  * COT: CFTC Legacy futures-only annual files (deacot{YEAR}.zip), market
    "GOLD - COMMODITY EXCHANGE INC.", net = Noncommercial long - short (the classic
    "large speculator" net). Report is as-of Tuesday; it is applied only from
    Friday 20:30 UTC (+3h to approximate broker server time) of the release week,
    i.e. effectively from the Sunday/Monday open. Cached to data/pro/cot_gold_legacy.csv.

Exits: SL = atr_mult_sl x ATR, TP = tp_rr x SL from the strategy Signal. Entry at the
signal-bar close +/- spread (SPREADS["frxXAUUSD"]). SL wins ties within a bar. No
break-even (config break_even_trigger_r: 0). Trades still open after MAX_HOLD bars
are marked to market at that bar's close (in R). Every signal is taken (overlaps
allowed) so filtered variants are exact subsets of base; a one-position-at-a-time
EV is reported as a secondary column.

Usage:
  .venv\\Scripts\\python.exe pro_bot/backtest_mtf_filters.py [--refresh]
"""

import csv
import io
import json
import math
import sys
import time
import zipfile
from bisect import bisect_right
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from pro_bot.strategies.mtf_pullback import MTFPullbackStrategy  # noqa: E402

try:
    from pro_bot.backtest import SPREADS
    SPREAD = SPREADS["frxXAUUSD"]
except Exception:                     # backtest.py imports websockets etc.
    SPREAD = 0.22

DATA_DIR  = ROOT / "data" / "pro"
SYMBOL    = "XAUUSD"
MAX_HOLD  = 288          # 24h of M5 bars; live has no time exit
N_WIN     = 4
SERVER_OFFSET_H = 3      # approx broker server time - UTC (EET/EEST)

# rolling window caps (live seeds 499 LTF / 299 H1 / 199 H4 / 99 D1 and grows)
KEEP = {"ltf": 700, "htf": 400, "h4": 300, "d1": 150}


# ── Data: MT5 ────────────────────────────────────────────────────────────────

def _read_csv(p: Path) -> list[dict]:
    with open(p, newline="") as f:
        return [{"epoch": int(r["epoch"]), "open": float(r["open"]),
                 "high": float(r["high"]), "low": float(r["low"]),
                 "close": float(r["close"]), "tick_volume": int(r["tick_volume"])}
                for r in csv.DictReader(f)]


def load_mt5(refresh: bool = False) -> dict[str, list[dict]]:
    names = ["M5", "H1", "H4", "D1"]
    paths = {n: DATA_DIR / f"mt5_{SYMBOL}_{n}.csv" for n in names}
    if not refresh and all(p.exists() for p in paths.values()):
        return {n: _read_csv(paths[n]) for n in names}

    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise RuntimeError(f"mt5.initialize failed: {mt5.last_error()}")
    try:
        tfs = {"M5": mt5.TIMEFRAME_M5, "H1": mt5.TIMEFRAME_H1,
               "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1}
        maxbars = mt5.terminal_info().maxbars
        out = {}
        for n, tf in tfs.items():
            rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, maxbars - 1)
            if rates is None or len(rates) == 0:
                raise RuntimeError(f"No {n} rates: {mt5.last_error()}")
            bars = [{"epoch": int(r["time"]), "open": float(r["open"]),
                     "high": float(r["high"]), "low": float(r["low"]),
                     "close": float(r["close"]), "tick_volume": int(r["tick_volume"])}
                    for r in rates[:-1]]            # drop the forming bar
            out[n] = bars
    finally:
        mt5.shutdown()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for n, bars in out.items():
        with open(paths[n], "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(bars[0].keys()))
            w.writeheader()
            w.writerows(bars)
    return out


# ── Data: COT ────────────────────────────────────────────────────────────────

COT_MARKET = "GOLD - COMMODITY EXCHANGE INC."


def load_cot(years: list[int], refresh: bool = False) -> list[tuple[int, int, str]]:
    """Returns sorted [(effective_epoch_server_time, net_noncomm, as_of)]."""
    cache = DATA_DIR / "cot_gold_legacy.csv"
    rows: dict[str, int] = {}
    if cache.exists() and not refresh:
        with open(cache, newline="") as f:
            rows = {r["as_of"]: int(r["net"]) for r in csv.DictReader(f)}
    have_years = {int(a[:4]) for a in rows}
    need = [y for y in years if y not in have_years or y == max(years)] if rows else years
    if need:
        import requests
        for y in need:
            url = f"https://www.cftc.gov/files/dea/history/deacot{y}.zip"
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            txt = z.read(z.namelist()[0]).decode("latin-1")
            for row in csv.DictReader(io.StringIO(txt)):
                if row["Market and Exchange Names"].strip() != COT_MARKET:
                    continue
                lng = int(row["Noncommercial Positions-Long (All)"].strip())
                sht = int(row["Noncommercial Positions-Short (All)"].strip())
                rows[row["As of Date in Form YYYY-MM-DD"].strip()] = lng - sht
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["as_of", "net"])
            for a in sorted(rows):
                w.writerow([a, rows[a]])

    out = []
    for a, net in rows.items():
        tue = datetime.strptime(a, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        release_utc = tue + timedelta(days=3, hours=20, minutes=30)   # Fri 15:30 ET
        eff = int(release_utc.timestamp()) + SERVER_OFFSET_H * 3600
        out.append((eff, net, a))
    out.sort()
    return out


def make_cot_lookup(cot, threshold=50_000):
    effs = [c[0] for c in cot]

    def bias_at(epoch: int) -> str:
        i = bisect_right(effs, epoch) - 1
        if i < 0:
            return "neutral"
        net = cot[i][1]
        if net > threshold:
            return "bullish"
        if net < -threshold:
            return "bearish"
        return "neutral"
    return bias_at


# ── Strategy driver ──────────────────────────────────────────────────────────

def load_cfg() -> dict:
    cfg = yaml.safe_load(open(ROOT / "pro_bot" / "config.yaml", encoding="utf-8"))
    for s in cfg["strategies"]:
        if s["name"] == "mtf_pullback" and "XAUUSD" in s.get("symbols", []):
            return dict(s)
    raise RuntimeError("mtf_pullback XAUUSD config not found")


def run_strategy(data, cfg, use_vp=False, use_cot=False, cot_bias_at=None):
    """Feed bars exactly as the live handlers do; returns [(m5_index, Signal)]."""
    c = dict(cfg)
    c["use_vp_filter"] = use_vp
    c["use_cot_filter"] = use_cot
    c["cot_cache_path"] = str(ROOT / "pro_bot" / "__no_such_cot_cache__.json")
    st = MTFPullbackStrategy(c)

    cur_epoch = [0]
    if use_cot:
        # The strategy re-reads COT via self._load_cot_bias(path) on every bar;
        # replace it with the point-in-time historical value.
        st._load_cot_bias = lambda _path: cot_bias_at(cur_epoch[0])

    m5, h1, h4, d1 = data["M5"], data["H1"], data["H4"], data["D1"]
    ih = i4 = idd = 0
    sigs = []
    for i, bar in enumerate(m5):
        close_t = bar["epoch"] + 300
        while ih < len(h1) and h1[ih]["epoch"] + 3600 <= close_t:
            st.feed_htf(h1[ih]); ih += 1
        while i4 < len(h4) and h4[i4]["epoch"] + 14400 <= close_t:
            st.feed_4h(h4[i4]); i4 += 1
        while idd < len(d1) and d1[idd]["epoch"] + 86400 <= close_t:
            st.feed_daily(d1[idd]); idd += 1
        cur_epoch[0] = close_t
        # trim rolling buffers (amortised)
        if len(st._bars) > 2 * KEEP["ltf"]:
            del st._bars[:-KEEP["ltf"]]
        if len(st._htf_bars) > 2 * KEEP["htf"]:
            del st._htf_bars[:-KEEP["htf"]]
        if len(st._4h_bars) > 2 * KEEP["h4"]:
            del st._4h_bars[:-KEEP["h4"]]
        if len(st._daily_bars) > 2 * KEEP["d1"]:
            del st._daily_bars[:-KEEP["d1"]]
        sig = st.feed(bar)
        if sig.action != "HOLD" and i >= 600:      # skip warm-up period
            sigs.append((i, sig))
    return sigs


# ── Exits / metrics ──────────────────────────────────────────────────────────

def simulate(bars, sigs, spread=SPREAD, max_hold=MAX_HOLD):
    trades = []
    for idx, sig in sigs:
        close = bars[idx]["close"]
        sl_d, tp_d = sig.sl_pips, sig.tp_pips
        if not sl_d or sl_d <= 0:
            continue
        if sig.action == "BUY":
            entry = close + spread; sl = entry - sl_d; tp = entry + tp_d
        else:
            entry = close - spread; sl = entry + sl_d; tp = entry - tp_d
        r, exit_i, res = None, None, None
        last = min(idx + max_hold, len(bars) - 1)
        for j in range(idx + 1, last + 1):
            h, l = bars[j]["high"], bars[j]["low"]
            if sig.action == "BUY":
                if l <= sl: r, res = -1.0, "LOSS"
                elif h >= tp: r, res = tp_d / sl_d, "WIN"
            else:
                if h >= sl: r, res = -1.0, "LOSS"
                elif l <= tp: r, res = tp_d / sl_d, "WIN"
            if r is not None:
                exit_i = j
                break
        if r is None:
            exit_i = last
            px = bars[last]["close"]
            r = ((px - entry) if sig.action == "BUY" else (entry - px)) / sl_d
            res = "TIMEOUT"
        trades.append({"idx": idx, "exit": exit_i, "side": sig.action,
                       "r": r, "res": res, "epoch": bars[idx]["epoch"]})
    return trades


def stats(trades):
    n = len(trades)
    if n == 0:
        return {"n": 0, "wr": float("nan"), "ev": float("nan"), "net": 0.0,
                "dd": 0.0, "se": float("nan"), "ev_seq": float("nan"), "n_seq": 0}
    rs = [t["r"] for t in trades]
    ev = sum(rs) / n
    sd = math.sqrt(sum((x - ev) ** 2 for x in rs) / (n - 1)) if n > 1 else float("nan")
    eq = peak = dd = 0.0
    for x in rs:
        eq += x; peak = max(peak, eq); dd = max(dd, peak - eq)
    # one position at a time
    seq, busy_until = [], -1
    for t in sorted(trades, key=lambda t: t["idx"]):
        if t["idx"] > busy_until:
            seq.append(t["r"]); busy_until = t["exit"]
    return {"n": n, "wr": sum(1 for t in trades if t["res"] == "WIN") / n,
            "ev": ev, "net": sum(rs), "dd": dd, "se": sd / math.sqrt(n) if n > 1 else float("nan"),
            "ev_seq": sum(seq) / len(seq) if seq else float("nan"), "n_seq": len(seq)}


def fmt_row(name, s):
    return (f"{name:<28} {s['n']:>4} {s['wr']*100:>6.1f}% {s['ev']:>+7.3f} "
            f"({s['se']:.3f}) {s['net']:>+7.1f} {s['dd']:>6.1f}  "
            f"{s['ev_seq']:>+7.3f} n={s['n_seq']}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    refresh = "--refresh" in sys.argv
    data = load_mt5(refresh)
    m5 = data["M5"]
    f = lambda e: datetime.fromtimestamp(e, timezone.utc).strftime("%Y-%m-%d")
    print(f"MT5 {SYMBOL}: M5 {len(m5)} bars {f(m5[0]['epoch'])} -> {f(m5[-1]['epoch'])} "
          f"| H1 {len(data['H1'])} | H4 {len(data['H4'])} | D1 {len(data['D1'])}")

    y0 = datetime.fromtimestamp(m5[0]["epoch"], timezone.utc).year - 1
    y1 = datetime.fromtimestamp(m5[-1]["epoch"], timezone.utc).year
    cot = load_cot(list(range(y0, y1 + 1)), refresh)
    in_rng = [c for c in cot if m5[0]["epoch"] - 14 * 86400 <= c[0] <= m5[-1]["epoch"]]
    nets = [c[1] for c in in_rng]
    print(f"COT legacy noncomm net ({COT_MARKET}): {len(in_rng)} weekly reports in range, "
          f"{in_rng[0][2]} .. {in_rng[-1][2]}, net min {min(nets):+,} max {max(nets):+,}")
    for th in (25_000, 50_000, 100_000):
        nb = sum(1 for x in nets if x > th); ns = sum(1 for x in nets if x < -th)
        print(f"  threshold {th:>7,}: bullish weeks {nb}, bearish {ns}, neutral {len(nets)-nb-ns}")

    cfg = load_cfg()
    print(f"Config: ema{cfg['ema_period']} rsi_entry{cfg['rsi_entry']} tp_rr{cfg['tp_rr']} "
          f"adx{cfg['adx_min']} atr_sl{cfg['atr_mult_sl']} vp_lookback{cfg['vp_lookback']} "
          f"vp_prox{cfg['vp_proximity_pct']} | spread {SPREAD} | max_hold {MAX_HOLD} bars\n")

    bias50 = make_cot_lookup(cot, 50_000)
    variants = {}
    for name, vp, ct in [("base", False, False), ("vp", True, False),
                         ("cot", False, True), ("both", True, True)]:
        t0 = time.time()
        sigs = run_strategy(data, cfg, vp, ct, bias50)
        variants[name] = simulate(m5, sigs)
        print(f"  ran {name:<5} signals={len(sigs):>4}  ({time.time()-t0:.0f}s)")

    # sanity: filters must be pure subsets of base
    base_idx = {t["idx"] for t in variants["base"]}
    for k in ("vp", "cot", "both"):
        extra = {t["idx"] for t in variants[k]} - base_idx
        if extra:
            print(f"  WARNING: {k} has {len(extra)} signals not in base")

    # windows
    start_e, end_e = m5[600]["epoch"], m5[-1]["epoch"]
    edges = [start_e + (end_e - start_e) * k / N_WIN for k in range(N_WIN + 1)]
    win_of = lambda e: min(N_WIN - 1, max(0, bisect_right(edges, e) - 1))

    print("\n=== Results (R per trade; EV se in parens; ev_seq = one position at a time) ===")
    print(f"{'variant':<28} {'n':>4} {'WR':>7} {'EV':>7} {'(se)':>7} {'netR':>7} {'maxDD':>6}  {'EV_seq':>7}")
    for k, tr in variants.items():
        print(fmt_row(k, stats(tr)))
    b = stats(variants["base"])
    print(f"(breakeven WR at {cfg['tp_rr']}R = {100/(1+cfg['tp_rr']):.1f}%)")

    print("\n=== Walk-forward: EV by chronological quarter (n in brackets) ===")
    hdr = "variant   " + "".join(
        f"  W{w+1} {f(int(edges[w]))[2:]}..{f(int(edges[w+1]))[5:]}  " for w in range(N_WIN))
    print(hdr)
    for k, tr in variants.items():
        cells = []
        for w in range(N_WIN):
            s = stats([t for t in tr if win_of(t["epoch"]) == w])
            cells.append(f"{s['ev']:>+7.3f} [{s['n']:>3}]       " if s["n"] else f"{'--':>7} [  0]       ")
        print(f"{k:<9} " + "".join(cells))
    for k in ("vp", "cot", "both"):
        better = 0
        for w in range(N_WIN):
            sb = stats([t for t in variants["base"] if win_of(t["epoch"]) == w])
            sv = stats([t for t in variants[k] if win_of(t["epoch"]) == w])
            if sv["n"] and sv["ev"] > sb["ev"]:
                better += 1
        print(f"  {k}: EV better than base in {better}/{N_WIN} windows")

    print("\n=== Trades REMOVED by each filter (vs base) ===")
    for k in ("vp", "cot", "both"):
        kept = {t["idx"] for t in variants[k]}
        rem = [t for t in variants["base"] if t["idx"] not in kept]
        s = stats(rem)
        per_w = " ".join(
            f"W{w+1}:{stats([t for t in rem if win_of(t['epoch'])==w])['ev']:+.2f}"
            f"[{len([t for t in rem if win_of(t['epoch'])==w])}]" for w in range(N_WIN))
        print(f"  {k:<5} removed {s['n']:>4}/{b['n']}  EV removed {s['ev']:+.3f}R (se {s['se']:.3f})"
              f"  WR {s['wr']*100:.1f}%  | {per_w}")

    # long/short split for base & cot (COT acts by side)
    print("\n=== Side split ===")
    for k in variants:
        for side in ("BUY", "SELL"):
            s = stats([t for t in variants[k] if t["side"] == side])
            print(f"  {k:<5} {side:<4} n={s['n']:>4} EV={s['ev']:+.3f} WR={s['wr']*100 if s['n'] else 0:.1f}%")

    # ── Sensitivity (NOT tuning; neighbours of configured values) ──
    print("\n=== Sensitivity check (neighbouring values, informational only) ===")
    base_sigs = run_strategy(data, cfg)            # keep Signal objects for predicates
    base_tr = {t["idx"]: t for t in simulate(m5, base_sigs)}
    probe = MTFPullbackStrategy(dict(cfg))

    def vp_pass(idx, sig, lookback, prox):
        bars = m5[max(0, idx + 1 - max(lookback, 30) - 5): idx + 1]
        poc, vah, val = probe._volume_profile(bars, lookback)
        lvl = val if sig.action == "BUY" else vah
        if lvl is None:
            return True
        return abs(m5[idx]["close"] - lvl) / m5[idx]["close"] <= prox

    # verify predicate reproduces the class result at configured values
    vp_cls = {t["idx"] for t in variants["vp"]}
    vp_pred = {i for i, s in base_sigs if vp_pass(i, s, cfg["vp_lookback"], cfg["vp_proximity_pct"])}
    print(f"  (predicate check: VP class={len(vp_cls)} predicate={len(vp_pred)} "
          f"match={vp_cls == vp_pred})")
    for lb, px in [(288, 0.002), (288, 0.005), (144, 0.003), (576, 0.003)]:
        keep = [base_tr[i] for i, s in base_sigs if vp_pass(i, s, lb, px)]
        print(fmt_row(f"vp lb={lb} prox={px}", stats(keep)))
    for th in (25_000, 100_000):
        ba = make_cot_lookup(cot, th)
        keep = []
        for i, s in base_sigs:
            bias = ba(m5[i]["epoch"] + 300)
            if (s.action == "BUY" and bias == "bearish") or (s.action == "SELL" and bias == "bullish"):
                continue
            keep.append(base_tr[i])
        print(fmt_row(f"cot threshold={th:,}", stats(keep)))


if __name__ == "__main__":
    main()
