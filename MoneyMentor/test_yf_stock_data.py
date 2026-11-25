import time
import unittest
from unittest.mock import patch

import pandas as pd

import yf_stock_data


class StockQuoteTests(unittest.TestCase):
    def setUp(self):
        self.cache_key = "fetch_quote_('MSFT',)_{}"
        yf_stock_data.cache.pop(self.cache_key, None)
        yf_stock_data.cache_expiry.pop(self.cache_key, None)

    def tearDown(self):
        yf_stock_data.cache.pop(self.cache_key, None)
        yf_stock_data.cache_expiry.pop(self.cache_key, None)

    @patch("yf_stock_data.yf.Ticker")
    def test_realtime_quote_bypasses_cached_failure(self, ticker_factory):
        ticker = ticker_factory.return_value
        ticker.info = {
            "shortName": "Microsoft Corporation",
            "marketCap": 1_000,
            "trailingPE": 30,
        }
        ticker.history.return_value = pd.DataFrame(
            {
                "Close": [100.0, 110.0],
                "High": [101.0, 111.0],
                "Low": [99.0, 109.0],
                "Volume": [1_000, 2_000],
            },
            index=pd.date_range("2026-01-01", periods=2),
        )
        yf_stock_data.cache[self.cache_key] = None
        yf_stock_data.cache_expiry[self.cache_key] = time.time() + 300

        quote = yf_stock_data.get_stock_quote("MSFT", use_realtime=True)

        self.assertIsNotNone(quote)
        self.assertEqual(quote["price"], 110.0)
        ticker_factory.assert_called_once_with("MSFT")

    def test_none_response_is_not_cached(self):
        calls = []

        def fetch(symbol):
            calls.append(symbol)
            return None

        first_result = yf_stock_data.get_from_cache_or_api(fetch, "MSFT")
        second_result = yf_stock_data.get_from_cache_or_api(fetch, "MSFT")

        self.assertIsNone(first_result)
        self.assertIsNone(second_result)
        self.assertEqual(calls, ["MSFT", "MSFT"])


if __name__ == "__main__":
    unittest.main()
