"""
Live win-rate checker for ALL contract types (binary RISE/FALL + digit).

Groups closed trades by symbol and contract family, shows WR, net profit,
and sample longcodes so you can confirm the filter is working.

Usage:
  python check_live_wr.py               # last 1000 transactions
  python check_live_wr.py 500           # last 500
  python check_live_wr.py 500 JD75      # JD75 only
  python check_live_wr.py 1000 dump     # also print first raw transaction
"""
import asyncio
import os
import sys
from collections import defaultdict
from dotenv import load_dotenv
from src.api.client import DerivClient

LIMIT      = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 1000
FILTER_SYM = next((a for a in sys.argv[1:] if not a.isdigit() and a != "dump"), None)
DUMP       = "dump" in sys.argv

_LONGCODE_NAMES = {
    "Volatility 10 Index":      "R_10",
    "Volatility 25 Index":      "R_25",
    "Volatility 50 Index":      "R_50",
    "Volatility 75 Index":      "R_75",
    "Volatility 100 Index":     "R_100",
    "Jump 10 Index":            "JD10",
    "Jump 25 Index":            "JD25",
    "Jump 50 Index":            "JD50",
    "Jump 75 Index":            "JD75",
    "Jump 100 Index":           "JD100",
    "Crash 300 Index":          "CRASH300",
    "Crash 500 Index":          "CRASH500",
    "Crash 1000 Index":         "CRASH1000",
    "Boom 300 Index":           "BOOM300",
    "Boom 500 Index":           "BOOM500",
    "Boom 1000 Index":          "BOOM1000",
    "USD/JPY":                  "frxUSDJPY",
    "XAU/USD":                  "frxXAUUSD",
}

# Shortcode prefixes that identify binary contract types
_BINARY_PREFIXES = ("CALL_", "PUT_", "RISE_", "FALL_", "CALLE_", "PUTE_",
                    "ONETOUCH_", "NOTOUCH_", "RANGE_", "UPORDOWN_",
                    "DIGITOVER_", "DIGITUNDER_", "DIGITEVEN_", "DIGITODD_",
                    "DIGITMATCH_", "DIGITDIFF_")


def _contract_family(longcode: str, shortcode: str) -> str:
    """Return a short label for the contract type."""
    lc = longcode.lower()
    sc = shortcode.upper()
    if "digit" in lc:
        if "higher than" in lc:
            return "DIGITOVER"
        if "lower than" in lc:
            return "DIGITUNDER"
        if "even" in lc:
            return "DIGITEVEN"
        if "odd" in lc:
            return "DIGITODD"
        return "DIGIT"
    if sc.startswith("CALL_") or sc.startswith("RISE_"):
        return "CALL/RISE"
    if sc.startswith("PUT_") or sc.startswith("FALL_"):
        return "PUT/FALL"
    if "higher than" in lc or "above" in lc:
        return "CALL/RISE"
    if "lower than" in lc or "below" in lc:
        return "PUT/FALL"
    if "accumulator" in lc or "accu" in lc:
        return "ACCU"
    if "multiplier" in lc or "mult" in lc:
        return "MULT"
    return "OTHER"


def _symbol_from_tx(tx: dict) -> str:
    for field in ("symbol", "underlying_symbol", "underlying"):
        v = tx.get(field)
        if v:
            return str(v)
    sc = tx.get("shortcode", "")
    if sc:
        # CALL_JD75_... or DIGITOVER_R_50_...
        for part_count in (3, 2):
            parts = sc.split("_")
            for i in range(1, len(parts)):
                candidate = "_".join(parts[i:i + part_count])
                if candidate in _LONGCODE_NAMES.values():
                    return candidate
    lc = tx.get("longcode", "")
    for name, sym in _LONGCODE_NAMES.items():
        if name in lc:
            return sym
    return "unknown"


async def main():
    load_dotenv()
    token = os.getenv("DERIV_API_TOKEN") or os.getenv("DERIV_TOKEN")
    if not token:
        print("ERROR: DERIV_API_TOKEN not set")
        return

    client = DerivClient(token)
    await client.connect()

    all_tx = []
    offset = 0
    while len(all_tx) < LIMIT:
        fetch = min(100, LIMIT - len(all_tx))
        resp  = await client._send({
            "statement": 1,
            "description": 1,
            "limit": fetch,
            "offset": offset,
        })
        page = resp.get("statement", {}).get("transactions", [])
        if not page:
            break
        all_tx.extend(page)
        offset += len(page)
        if len(page) < fetch:
            break

    print(f"Fetched {len(all_tx)} transactions")

    if DUMP and all_tx:
        print("\n--- FIRST RAW TRANSACTION ---")
        print(dict(all_tx[0]))

    # Pair buy/sell by contract_id
    by_cid: dict = defaultdict(dict)
    for tx in all_tx:
        cid    = str(tx.get("contract_id", ""))
        action = tx.get("action_type", "")
        if action in ("buy", "sell"):
            by_cid[cid][action] = tx

    # Tally by (symbol, family)
    stats: dict = defaultdict(lambda: {"wins": 0, "losses": 0, "profit": 0.0,
                                        "open": 0, "samples": []})

    for cid, pair in by_cid.items():
        buy  = pair.get("buy", {})
        sell = pair.get("sell", {})

        if not buy:
            continue
        if not sell:
            # No sell entry = contract still open or cancelled
            symbol = _symbol_from_tx(buy)
            family = _contract_family(buy.get("longcode", ""),
                                       buy.get("shortcode", ""))
            key = (symbol, family)
            stats[key]["open"] += 1
            continue

        tx     = sell  # sell has the payout/outcome
        desc   = tx.get("longcode", "")
        sc     = tx.get("shortcode", "") or buy.get("shortcode", "")
        symbol = _symbol_from_tx(tx) or _symbol_from_tx(buy)
        family = _contract_family(desc, sc)

        if FILTER_SYM and symbol != FILTER_SYM:
            continue

        amount = float(sell.get("amount", 0))
        stake  = abs(float(buy.get("amount", 0)))
        profit = amount - stake

        key = (symbol, family)
        stats[key]["wins"]    += profit > 0
        stats[key]["losses"]  += profit <= 0
        stats[key]["profit"]  += profit
        if len(stats[key]["samples"]) < 2:
            stats[key]["samples"].append(desc[:90])

    if not stats:
        print("\nNo closed trades found" + (f" for {FILTER_SYM}" if FILTER_SYM else ""))
        await client.disconnect()
        return

    # Sort by total trades descending
    rows = sorted(stats.items(), key=lambda kv: kv[1]["wins"] + kv[1]["losses"], reverse=True)

    print(f"\n{'Symbol':<12} {'Type':<12} {'W':>5} {'L':>5} {'Total':>6} "
          f"{'WR%':>6} {'Net$':>8}  {'Status'}")
    print("-" * 72)

    for (sym, fam), s in rows:
        w, l = s["wins"], s["losses"]
        total = w + l
        if total == 0 and s["open"] == 0:
            continue
        wr  = w / total * 100 if total else 0
        net = s["profit"]

        # Break-even depends on contract type
        if fam in ("CALL/RISE", "PUT/FALL"):
            be = 53.5   # JD75 payout ~87% → BE=53.5%; adjust as needed
        elif "DIGIT" in fam:
            be = 51.3   # 95% payout → BE=51.3%
        elif fam == "ACCU":
            be = 0
        else:
            be = 52.0

        if total < 5:
            verdict = f"({total} trades — too few)"
        elif wr >= be + 3:
            verdict = "PROFITABLE"
        elif wr >= be:
            verdict = "MARGINAL"
        else:
            verdict = f"LOSING (BE={be:.1f}%)"

        open_str = f"  +{s['open']}open" if s["open"] else ""
        print(f"{sym:<12} {fam:<12} {w:>5} {l:>5} {total:>6} "
              f"{wr:>5.1f}%  {net:>+8.2f}  {verdict}{open_str}")

        for sample in s["samples"]:
            print(f"  eg: {sample}")

    print()
    await client.disconnect()


asyncio.run(main())
