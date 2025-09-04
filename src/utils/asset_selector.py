"""
Asset selection and management
Handles S&P 500 stocks and top 50 cryptocurrencies.
"""

import logging
from typing import List, Dict


class AssetSelector:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # S&P 500 symbols (sample - full list would be 500)
        self.sp500_symbols = [
            'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'TSLA', 'GOOG', 'BRK-B',
            'META', 'UNH', 'XOM', 'LLY', 'JPM', 'JNJ', 'V', 'PG', 'MA', 'HD',
            'CVX', 'MRK', 'ABBV', 'PEP', 'KO', 'AVGO', 'PFE', 'TMO', 'COST',
            'WMT', 'BAC', 'CRM', 'ABT', 'LIN', 'CSCO', 'DIS', 'ACN', 'DHR',
            'TXN', 'VZ', 'WFC', 'ADBE', 'NEE', 'ORCL', 'BMY', 'PM', 'RTX',
            'T', 'CMCSA', 'COP', 'NFLX', 'AMD', 'LOW', 'HON', 'AMGN', 'UPS',
            'IBM', 'BA', 'GS', 'SPGI', 'DE', 'BKNG', 'BLK', 'ELV', 'CAT',
            'MDLZ', 'GILD', 'ADP', 'SYK', 'TJX', 'VRTX', 'AXP', 'ADI', 'CVS',
            'SCHW', 'MU', 'LRCX', 'PLD', 'ISRG', 'SO', 'TMUS', 'ZTS', 'REGN',
            'NOW', 'CB', 'MMC', 'FDX', 'EOG', 'MO', 'ETN', 'BSX', 'DUK', 'ITW'
        ]
        
        # Top 50 crypto symbols (by market cap)
        self.crypto_symbols = [
            'BTC', 'ETH', 'USDT', 'BNB', 'SOL', 'USDC', 'XRP', 'DOGE', 'TON',
            'ADA', 'SHIB', 'AVAX', 'TRX', 'WBTC', 'DOT', 'LINK', 'BCH', 'NEAR',
            'MATIC', 'ICP', 'UNI', 'LTC', 'DAI', 'ETC', 'XMR', 'APT', 'ATOM',
            'OKB', 'FIL', 'STX', 'MNT', 'CRO', 'VET', 'LDO', 'ARB', 'IMX',
            'GRT', 'MKR', 'HBAR', 'OP', 'INJ', 'SUI', 'REND', 'SAND', 'MANA',
            'ALGO', 'QNT', 'AAVE', 'FTM', 'THETA'
        ]
    
    async def get_all_assets(self) -> List[Dict[str, str]]:
        """Get all assets (stocks + crypto) to analyze"""
        assets = []
        
        # Add S&P 500 stocks
        for symbol in self.sp500_symbols:
            assets.append({
                'symbol': symbol,
                'type': 'stock',
                'exchange': 'NYSE/NASDAQ'
            })
        
        # Add top crypto
        for symbol in self.crypto_symbols:
            assets.append({
                'symbol': symbol,
                'type': 'crypto',
                'exchange': 'binance'
            })
        
        self.logger.info(f"Selected {len(assets)} assets ({len(self.sp500_symbols)} stocks, {len(self.crypto_symbols)} crypto)")
        return assets
    
    async def get_stocks_only(self) -> List[Dict[str, str]]:
        """Get only stock assets"""
        return [
            {'symbol': symbol, 'type': 'stock', 'exchange': 'NYSE/NASDAQ'}
            for symbol in self.sp500_symbols
        ]
    
    async def get_crypto_only(self) -> List[Dict[str, str]]:
        """Get only cryptocurrency assets"""
        return [
            {'symbol': symbol, 'type': 'crypto', 'exchange': 'binance'}
            for symbol in self.crypto_symbols
        ]