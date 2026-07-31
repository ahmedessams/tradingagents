"""Regenerate the committed US stock list for the webapp's ticker search.

Fetches the daily-updated rreichel3/US-Stock-Symbols dataset (NASDAQ /
NYSE / AMEX), normalizes it via ``webapp.stocklist``, appends the curated
ETF set, and writes minified JSON to ``webapp/static/us_stocks.json``.

The output is committed to the repo so the server has no runtime network
dependency. Rerun whenever the list feels stale (new IPOs, delistings):

    python scripts/update_stock_list.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

from webapp.stocklist import EXCHANGE_URLS, curated_etf_entries, merge_exchange_lists

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "webapp" / "static" / "us_stocks.json"


def main() -> int:
    payloads: dict[str, list[dict]] = {}
    for exchange, url in EXCHANGE_URLS.items():
        print(f"Fetching {exchange} ...", flush=True)
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            payloads[exchange] = response.json()
        except (requests.RequestException, ValueError) as exc:
            print(f"Failed to fetch {exchange} list from {url}: {exc}", file=sys.stderr)
            return 1

    stocks = merge_exchange_lists(payloads)
    entries = stocks + curated_etf_entries()
    OUTPUT_PATH.write_text(json.dumps(entries, separators=(",", ":")) + "\n")

    counts = {ex: len(rows) for ex, rows in payloads.items()}
    print(
        f"Wrote {len(entries)} entries ({len(stocks)} stocks from {counts}, "
        f"{len(entries) - len(stocks)} curated ETFs) to {OUTPUT_PATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
