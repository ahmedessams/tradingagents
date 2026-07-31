"""US stock list normalization for the webapp's searchable ticker select.

The raw data comes from the daily-updated rreichel3/US-Stock-Symbols GitHub
dataset (NASDAQ/NYSE/AMEX). ``scripts/update_stock_list.py`` fetches it,
runs it through :func:`merge_exchange_lists`, and commits the result to
``webapp/static/us_stocks.json`` so the server never needs the network.

Entry shape is deliberately compact — ``{"s": symbol, "n": name, "x":
exchange}`` — because the whole list ships to the browser.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from tradingagents.dataflows.utils import safe_ticker_component

STOCKS_JSON_PATH = Path(__file__).parent / "static" / "us_stocks.json"

EXCHANGE_URLS = {
    "nasdaq": "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq_full_tickers.json",
    "nyse": "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nyse/nyse_full_tickers.json",
    "amex": "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/amex/amex_full_tickers.json",
}

# The source dataset is common stock only, but the UI presets (and most
# casual users) reach for the big index/commodity ETFs, so a small curated
# set rides along, tagged "etf".
CURATED_ETFS = [
    ("SPY", "SPDR S&P 500 ETF Trust"),
    ("VOO", "Vanguard S&P 500 ETF"),
    ("IVV", "iShares Core S&P 500 ETF"),
    ("QQQ", "Invesco QQQ Trust (Nasdaq-100)"),
    ("DIA", "SPDR Dow Jones Industrial Average ETF"),
    ("IWM", "iShares Russell 2000 ETF"),
    ("VTI", "Vanguard Total Stock Market ETF"),
    ("VT", "Vanguard Total World Stock ETF"),
    ("EFA", "iShares MSCI EAFE ETF"),
    ("EEM", "iShares MSCI Emerging Markets ETF"),
    ("AGG", "iShares Core U.S. Aggregate Bond ETF"),
    ("BND", "Vanguard Total Bond Market ETF"),
    ("TLT", "iShares 20+ Year Treasury Bond ETF"),
    ("HYG", "iShares iBoxx High Yield Corporate Bond ETF"),
    ("GLD", "SPDR Gold Shares"),
    ("SLV", "iShares Silver Trust"),
    ("USO", "United States Oil Fund"),
    ("XLE", "Energy Select Sector SPDR Fund"),
    ("XLF", "Financial Select Sector SPDR Fund"),
    ("XLK", "Technology Select Sector SPDR Fund"),
    ("SMH", "VanEck Semiconductor ETF"),
    ("ARKK", "ARK Innovation ETF"),
]

# Boilerplate suffixes the source appends to nearly every security name.
# Trimmed longest-first so "Class A Common Stock" goes before "Common Stock".
_NAME_SUFFIXES = (
    " Class A Common Stock",
    " Class B Common Stock",
    " Class C Common Stock",
    " Common Stock",
    " Common Shares",
    " Ordinary Shares",
    " American Depositary Shares",
    " Depositary Shares",
)


def _clean_name(name: str) -> str:
    name = " ".join(name.split())
    for suffix in _NAME_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)].rstrip(",").strip()
    return name


def normalize_symbol(raw: str) -> str | None:
    """Map a source symbol to the yfinance form, or None to drop it.

    Class shares use a slash upstream (``BRK/B``) but a dash on yfinance
    (``BRK-B``). Caret symbols (``ABR^D``) are preferred shares, which
    yfinance names differently (``ABR-PD``) and which aren't meaningful
    analysis targets — dropped rather than mis-mapped.
    """
    symbol = raw.strip().upper()
    if not symbol or "^" in symbol:
        return None
    symbol = symbol.replace("/", "-")
    try:
        safe_ticker_component(symbol)
    except ValueError:
        return None
    return symbol


def merge_exchange_lists(payloads: dict[str, list[dict]]) -> list[dict]:
    """Merge per-exchange payloads into the compact, sorted entry list.

    Duplicate symbols across exchanges keep the first occurrence in
    ``payloads`` iteration order.
    """
    merged: dict[str, dict] = {}
    for exchange, rows in payloads.items():
        for row in rows:
            symbol = normalize_symbol(str(row.get("symbol", "")))
            if symbol is None or symbol in merged:
                continue
            merged[symbol] = {
                "s": symbol,
                "n": _clean_name(str(row.get("name", ""))),
                "x": exchange,
            }
    return sorted(merged.values(), key=lambda e: e["s"])


def curated_etf_entries() -> list[dict]:
    return [{"s": s, "n": n, "x": "etf"} for s, n in CURATED_ETFS]


@lru_cache(maxsize=1)
def symbol_name_map() -> dict[str, str]:
    """Symbol -> company name from the committed list; {} if it's missing.

    Cached for the process lifetime — the file only changes on redeploy,
    which restarts the server anyway.
    """
    try:
        entries = json.loads(STOCKS_JSON_PATH.read_text())
    except (OSError, ValueError):
        return {}
    return {e["s"]: e["n"] for e in entries if e.get("s") and e.get("n")}


def lookup_company_name(ticker: str) -> str | None:
    return symbol_name_map().get(ticker)
