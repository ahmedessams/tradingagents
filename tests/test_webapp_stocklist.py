"""Tests for the US stock list normalization behind the webapp ticker search."""

import json
import unittest
from pathlib import Path

import pytest

from tradingagents.dataflows.utils import safe_ticker_component
from webapp.stocklist import (
    CURATED_ETFS,
    curated_etf_entries,
    merge_exchange_lists,
    normalize_symbol,
)

STOCKS_JSON = Path(__file__).resolve().parent.parent / "webapp" / "static" / "us_stocks.json"


@pytest.mark.unit
class TestNormalizeSymbol(unittest.TestCase):
    def test_class_share_slash_becomes_dash(self):
        self.assertEqual(normalize_symbol("BRK/B"), "BRK-B")

    def test_preferred_share_caret_dropped(self):
        self.assertIsNone(normalize_symbol("ABR^D"))

    def test_whitespace_and_case_normalized(self):
        self.assertEqual(normalize_symbol(" nvda "), "NVDA")

    def test_invalid_symbols_dropped(self):
        self.assertIsNone(normalize_symbol(""))
        self.assertIsNone(normalize_symbol("   "))
        self.assertIsNone(normalize_symbol("A" * 40))  # over max_len
        self.assertIsNone(normalize_symbol("ET C"))  # embedded whitespace
        # ".." is the traversal-dangerous path component; "../ETC" maps to
        # "..-ETC" which is character-valid and harmless as a path component.
        self.assertIsNone(normalize_symbol(".."))
        self.assertEqual(normalize_symbol("../ETC"), "..-ETC")


@pytest.mark.unit
class TestMergeExchangeLists(unittest.TestCase):
    def test_merge_normalizes_and_sorts(self):
        payloads = {
            "nyse": [
                {"symbol": "BRK/B", "name": "Berkshire Hathaway Inc."},
                {"symbol": "ABR^D", "name": "Arbor Realty Preferred D"},
            ],
            "nasdaq": [
                {"symbol": "NVDA", "name": "NVIDIA Corporation Common Stock"},
                {"symbol": "AAPL", "name": "Apple Inc. Common Stock"},
            ],
        }
        entries = merge_exchange_lists(payloads)
        self.assertEqual([e["s"] for e in entries], ["AAPL", "BRK-B", "NVDA"])
        self.assertEqual(entries[0]["n"], "Apple Inc.")
        self.assertEqual(entries[1]["x"], "nyse")

    def test_duplicates_keep_first_exchange(self):
        payloads = {
            "nyse": [{"symbol": "DUP", "name": "First Listing Inc."}],
            "nasdaq": [{"symbol": "DUP", "name": "Second Listing Inc."}],
        }
        entries = merge_exchange_lists(payloads)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["n"], "First Listing Inc.")
        self.assertEqual(entries[0]["x"], "nyse")

    def test_name_suffix_trimming(self):
        payloads = {
            "nyse": [
                {"symbol": "AKO/A", "name": "Embotelladora Andina S.A. Class A Common Stock"},
                {"symbol": "TSM", "name": "Taiwan Semiconductor American Depositary Shares"},
            ]
        }
        entries = merge_exchange_lists(payloads)
        names = {e["s"]: e["n"] for e in entries}
        self.assertEqual(names["AKO-A"], "Embotelladora Andina S.A.")
        self.assertEqual(names["TSM"], "Taiwan Semiconductor")


@pytest.mark.unit
class TestCuratedEtfs(unittest.TestCase):
    def test_all_symbols_valid_and_tagged(self):
        entries = curated_etf_entries()
        self.assertEqual(len(entries), len(CURATED_ETFS))
        for entry in entries:
            safe_ticker_component(entry["s"])  # raises on invalid
            self.assertEqual(entry["x"], "etf")
            self.assertTrue(entry["n"])

    def test_spy_present(self):
        # The UI preset buttons rely on SPY existing in the searchable list.
        self.assertIn("SPY", [s for s, _ in CURATED_ETFS])


@pytest.mark.unit
class TestCommittedStockList(unittest.TestCase):
    """Guards the generated webapp/static/us_stocks.json against a bad rerun
    of scripts/update_stock_list.py."""

    def test_file_parses_and_is_plausible(self):
        data = json.loads(STOCKS_JSON.read_text())
        self.assertGreater(len(data), 1000)
        for entry in data:
            self.assertEqual(set(entry), {"s", "n", "x"})
            safe_ticker_component(entry["s"])  # raises on invalid

    def test_contains_expected_symbols(self):
        symbols = {e["s"] for e in json.loads(STOCKS_JSON.read_text())}
        for expected in ("AAPL", "NVDA", "MSFT", "BRK-B", "SPY"):
            self.assertIn(expected, symbols)


if __name__ == "__main__":
    unittest.main()
