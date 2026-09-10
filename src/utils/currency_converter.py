"""
Currency conversion utility for international assets
Converts all prices to USD using real-time forex rates from our forex pairs
"""

import logging
from datetime import datetime
from typing import Dict, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class MissingRateError(Exception):
    """
    Raised when a price cannot be converted to USD.

    Returning the unconverted price instead is what allowed a JPY 4,899 quote to be
    compared against USD-denominated barriers. Callers must drop the asset rather
    than trade a price of unknown denomination.
    """


class CurrencyConverter:
    """Converts prices from various currencies to USD"""

    # Currency pair mappings to our forex symbols
    CURRENCY_PAIRS = {
        'EUR': 'EURUSD=X',  # EUR to USD
        'GBP': 'GBPUSD=X',  # GBP to USD
        'JPY': 'USDJPY=X',  # USD to JPY (inverted)
        'CHF': 'USDCHF=X',  # USD to CHF (inverted)
        'CNY': 'USDCNY=X',  # USD to CNY (inverted)
        'HKD': 'USDHKD=X',  # USD to HKD (inverted); falls back to the 7.80 peg
        'AUD': 'AUDUSD=X',  # AUD to USD
        'CAD': 'USDCAD=X',  # USD to CAD (inverted)
        'NZD': 'NZDUSD=X',  # NZD to USD
    }

    # Inverted pairs (quoted as USD/XXX, so the XXX->USD rate is 1/rate)
    INVERTED_PAIRS = {'JPY', 'CHF', 'CNY', 'CAD', 'HKD'}

    # Fallback rates (units of currency per USD) used when no forex bar is available.
    # HKD is pegged inside a 7.75-7.85 band.
    PEGGED_RATES = {'HKD': 7.80}

    def __init__(self):
        self.exchange_rates = {}  # Cache of current exchange rates
        self.last_update = None

    def update_rates(self, forex_data: Dict[str, pd.DataFrame]):
        """
        Update exchange rates from forex market data

        Args:
            forex_data: Dictionary mapping forex symbols to DataFrames with OHLCV data
        """
        try:
            for currency, forex_symbol in self.CURRENCY_PAIRS.items():
                rate = None

                df = forex_data.get(forex_symbol)
                if df is not None and not df.empty:
                    quoted = float(df['Close'].iloc[-1])
                    if quoted > 0:
                        # Inverted pairs are quoted as USD/XXX, so XXX->USD is 1/quoted
                        rate = (1.0 / quoted) if currency in self.INVERTED_PAIRS else quoted

                # Fall back to a pegged rate when no bar is available. Pegs are quoted
                # as units-per-USD, so they invert the same way.
                if rate is None and currency in self.PEGGED_RATES:
                    rate = 1.0 / self.PEGGED_RATES[currency]
                    logger.debug(f"Using pegged fallback for {currency}: {rate:.6f}")

                if rate is not None and rate > 0:
                    self.exchange_rates[currency] = rate
                    logger.debug(f"Updated {currency}/USD rate: {rate:.6f}")

            self.last_update = datetime.now()
            logger.info(f"Updated {len(self.exchange_rates)} currency exchange rates")

        except Exception as e:
            logger.error(f"Error updating currency rates: {e}")

    def convert_to_usd(self, amount: float, from_currency: str) -> float:
        """
        Convert an amount from a given currency to USD.

        Raises:
            MissingRateError: if no rate is known for `from_currency`.
        """
        # Already in USD
        if from_currency == 'USD':
            return amount

        rate = self.exchange_rates.get(from_currency)
        if rate is None:
            raise MissingRateError(
                f"No exchange rate available for {from_currency}; refusing to treat a "
                f"{from_currency} price as USD"
            )

        return amount * rate

    def convert_price_series(self, prices: pd.DataFrame, from_currency: str) -> pd.DataFrame:
        """
        Convert a price DataFrame (OHLCV) from a given currency to USD.

        Raises:
            MissingRateError: if no rate is known for `from_currency`.
        """
        if from_currency == 'USD':
            return prices

        rate = self.exchange_rates.get(from_currency)
        if rate is None:
            raise MissingRateError(
                f"No exchange rate available for {from_currency}; refusing to treat "
                f"{from_currency} prices as USD"
            )

        # Convert price columns only. Volume is a share count, not a price.
        converted = prices.copy()
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in converted.columns:
                converted[col] = converted[col] * rate

        return converted

    def currency_for_symbol(self, symbol: str) -> str:
        """
        Infer the quote currency from a ticker suffix.

        A fallback for when asset metadata is unavailable; the authoritative source is
        the `currency` field from AssetSelector.get_all_assets().
        """
        suffix_map = {
            '.T': 'JPY', '.HK': 'HKD', '.SS': 'CNY', '.SZ': 'CNY',
            '.DE': 'EUR', '.PA': 'EUR', '.AS': 'EUR', '.MI': 'EUR', '.MC': 'EUR',
            '.L': 'GBP', '.SW': 'CHF', '.AX': 'AUD', '.TO': 'CAD', '.NZ': 'NZD',
        }
        for suffix, currency in suffix_map.items():
            if symbol.endswith(suffix):
                return currency
        return 'USD'

    def get_rate(self, from_currency: str) -> Optional[float]:
        """Get the current exchange rate for a currency to USD"""
        if from_currency == 'USD':
            return 1.0
        return self.exchange_rates.get(from_currency)

    def has_rate(self, currency: str) -> bool:
        """Check if we have an exchange rate for a currency"""
        return currency == 'USD' or currency in self.exchange_rates
