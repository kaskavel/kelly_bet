"""
Asset name mapping utilities
Maps stock symbols and crypto symbols to their full names.
"""

import logging
import yfinance as yf
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Static mapping for crypto names (more reliable than API calls)
CRYPTO_NAMES = {
    'BTC': 'Bitcoin',
    'ETH': 'Ethereum',
    'USDT': 'Tether USDt',
    'BNB': 'BNB',
    'SOL': 'Solana',
    'USDC': 'USD Coin',
    'XRP': 'XRP',
    'DOGE': 'Dogecoin',
    'TON': 'Toncoin',
    'ADA': 'Cardano',
    'SHIB': 'Shiba Inu',
    'AVAX': 'Avalanche',
    'TRX': 'TRON',
    'WBTC': 'Wrapped Bitcoin',
    'DOT': 'Polkadot',
    'LINK': 'Chainlink',
    'BCH': 'Bitcoin Cash',
    'NEAR': 'NEAR Protocol',
    'MATIC': 'Polygon',
    'ICP': 'Internet Computer',
    'UNI': 'Uniswap',
    'LTC': 'Litecoin',
    'DAI': 'Dai',
    'ETC': 'Ethereum Classic',
    'XMR': 'Monero',
    'APT': 'Aptos',
    'ATOM': 'Cosmos Hub',
    'OKB': 'OKB',
    'FIL': 'Filecoin',
    'STX': 'Stacks',
    'MNT': 'Mantle',
    'CRO': 'Cronos',
    'VET': 'VeChain',
    'LDO': 'Lido DAO',
    'ARB': 'Arbitrum',
    'IMX': 'Immutable',
    'GRT': 'The Graph',
    'MKR': 'Maker',
    'HBAR': 'Hedera',
    'OP': 'Optimism',
    'INJ': 'Injective',
    'SUI': 'Sui',
    'REND': 'Render',
    'SAND': 'The Sandbox',
    'MANA': 'Decentraland',
    'ALGO': 'Algorand',
    'QNT': 'Quant',
    'AAVE': 'Aave',
    'FTM': 'Fantom',
    'THETA': 'Theta Network'
}

def get_crypto_name(symbol: str) -> str:
    """Get full name for crypto symbol"""
    return CRYPTO_NAMES.get(symbol.upper(), symbol.upper())

def get_stock_name(symbol: str) -> Optional[str]:
    """Get full company name for stock symbol using yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # Try different name fields in order of preference
        for name_field in ['longName', 'shortName', 'displayName']:
            if name_field in info and info[name_field]:
                return info[name_field]

        # Fallback to symbol if no name found
        return symbol.upper()

    except Exception as e:
        logger.warning(f"Could not fetch name for stock {symbol}: {e}")
        return symbol.upper()

def get_asset_name(symbol: str, asset_type: str) -> str:
    """Get full name for any asset (stock or crypto)"""
    if asset_type.lower() == 'crypto':
        return get_crypto_name(symbol)
    elif asset_type.lower() == 'stock':
        return get_stock_name(symbol) or symbol.upper()
    else:
        return symbol.upper()

def get_display_name(symbol: str, asset_type: str) -> str:
    """Get display name in format 'SYMBOL - Full Name'"""
    full_name = get_asset_name(symbol, asset_type)
    if full_name and full_name.upper() != symbol.upper():
        return f"{symbol.upper()} - {full_name}"
    else:
        return symbol.upper()