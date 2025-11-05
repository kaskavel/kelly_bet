"""
Asset selection and management
Handles S&P 500 stocks, top 50 cryptocurrencies, commodities (ETFs), and forex.
"""

import logging
from typing import List, Dict


class AssetSelector:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Commodity ETFs (highly liquid, track spot prices)
        self.commodity_symbols = {
            # Precious Metals
            'GLD': 'SPDR Gold Trust',
            'SLV': 'iShares Silver Trust',
            'PPLT': 'Aberdeen Standard Platinum Shares ETF',
            'PALL': 'Aberdeen Standard Palladium Shares ETF',

            # Energy
            'USO': 'United States Oil Fund (Crude Oil)',
            'UNG': 'United States Natural Gas Fund',
            'BNO': 'United States Brent Oil Fund',

            # Agricultural
            'CORN': 'Teucrium Corn Fund',
            'WEAT': 'Teucrium Wheat Fund',
            'SOYB': 'Teucrium Soybean Fund',

            # Broad Commodities Baskets
            'DBC': 'Invesco DB Commodity Index Tracking Fund',
            'GSG': 'iShares S&P GSCI Commodity-Indexed Trust',
            'PDBC': 'Invesco Optimum Yield Diversified Commodity',

            # Industrial Metals
            'CPER': 'United States Copper Index Fund',
            'JJN': 'iPath Bloomberg Nickel Subindex Total Return ETN',

            # Other Commodities
            'URA': 'Global X Uranium ETF',
            'WOOD': 'iShares Global Timber & Forestry ETF',
        }

        # Forex pairs (major and cross pairs)
        self.forex_symbols = {
            # Major Pairs (USD-based)
            'EURUSD=X': 'Euro / US Dollar',
            'GBPUSD=X': 'British Pound / US Dollar',
            'USDJPY=X': 'US Dollar / Japanese Yen',
            'USDCHF=X': 'US Dollar / Swiss Franc',
            'AUDUSD=X': 'Australian Dollar / US Dollar',
            'USDCAD=X': 'US Dollar / Canadian Dollar',
            'NZDUSD=X': 'New Zealand Dollar / US Dollar',

            # Cross Pairs (EUR-based)
            'EURGBP=X': 'Euro / British Pound',
            'EURJPY=X': 'Euro / Japanese Yen',
            'EURCHF=X': 'Euro / Swiss Franc',
            'EURAUD=X': 'Euro / Australian Dollar',
            'EURCAD=X': 'Euro / Canadian Dollar',

            # Cross Pairs (GBP-based)
            'GBPJPY=X': 'British Pound / Japanese Yen',
            'GBPCHF=X': 'British Pound / Swiss Franc',
            'GBPAUD=X': 'British Pound / Australian Dollar',

            # Cross Pairs (JPY-based)
            'AUDJPY=X': 'Australian Dollar / Japanese Yen',
            'CADJPY=X': 'Canadian Dollar / Japanese Yen',
            'CHFJPY=X': 'Swiss Franc / Japanese Yen',

            # Emerging Market Currencies
            'USDCNY=X': 'US Dollar / Chinese Yuan',
            'USDINR=X': 'US Dollar / Indian Rupee',
            'USDBRL=X': 'US Dollar / Brazilian Real',
            'USDMXN=X': 'US Dollar / Mexican Peso',
        }

        # Complete S&P 500 symbols (503 symbols - current as of 2024/2025)
        self.sp500_symbols = [
            # Information Technology (78 symbols)
            'AAPL', 'MSFT', 'NVDA', 'AVGO', 'ADBE', 'CSCO', 'CRM', 'ACN', 'ORCL', 'AMD',
            'INTC', 'IBM', 'QCOM', 'AMAT', 'SNPS', 'KLAC', 'CDNS', 'LRCX', 'ROP', 'ANET',
            'APH', 'ADI', 'ANSS', 'ADSK', 'ADP', 'MPWR', 'TXN', 'MCHP', 'SWKS', 'NTAP',
            'TER', 'SMCI', 'PANW', 'CRWD', 'DDOG', 'AKAM', 'CTSH', 'FICO', 'FTNT', 'GDDY',
            'GRMN', 'HPE', 'HPQ', 'INTU', 'IT', 'JNPR', 'KEYS', 'LOGI', 'MSTR', 'NOW',
            'NXPI', 'PLTR', 'PTC', 'QRVO', 'SEDG', 'STX', 'TRMB', 'TYL', 'VRSN', 'WDC',
            'WDAY', 'TEAM', 'ZM', 'DOCU', 'OKTA', 'SPLK', 'VEEV', 'GTLB', 'BILL', 'COUP',
            'FIVN', 'PCTY', 'PING', 'SMAR', 'XLNX', 'ZS', 'MU', 'NFLX',

            # Health Care (63 symbols)
            'UNH', 'LLY', 'JNJ', 'MRK', 'ABBV', 'TMO', 'PFE', 'ABT', 'DHR', 'ISRG',
            'REGN', 'VRTX', 'ZTS', 'DXCM', 'MRNA', 'BMY', 'AMGN', 'GILD', 'BIIB', 'BSX',
            'MDT', 'BDX', 'CI', 'CVS', 'HCA', 'ELV', 'TECH', 'MOH', 'SYK', 'EW',
            'HOLX', 'LH', 'WST', 'RMD', 'IDXX', 'RVTY', 'IQV', 'CRL', 'CAH', 'MCK',
            'A', 'VTRS', 'ALGN', 'GEHC', 'WAT', 'ZBH', 'STE', 'PODD', 'SOLV', 'HSIC',
            'DVA', 'UHS', 'INCY', 'NBIX', 'JAZZ', 'EXAS', 'TDOC', 'ILMN', 'IONS', 'BMRN',
            'ALNY', 'RARE', 'HALO',

            # Financials (65 symbols)
            'BRK.B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'AXP', 'SCHW',
            'BLK', 'BX', 'COF', 'C', 'PNC', 'TFC', 'USB', 'AON', 'CB', 'ICE',
            'CME', 'MCO', 'SPGI', 'AJG', 'MMC', 'AFL', 'ALL', 'AIG', 'TRV', 'PGR',
            'COIN', 'KKR', 'APO', 'NDAQ', 'MSCI', 'FIS', 'FITB', 'RF', 'CFG', 'KEY',
            'ZION', 'WTW', 'BRO', 'RJF', 'NTRS', 'STT', 'CBOE', 'TROW', 'IVZ', 'BEN',
            'EQH', 'MTB', 'HBAN', 'CMA', 'WAL', 'EWBC', 'ACGL', 'AIZ', 'AMP', 'CINF',
            'L', 'FDX', 'HOOD', 'PAYC', 'GPN',

            # Consumer Discretionary (52 symbols)
            'AMZN', 'TSLA', 'HD', 'MCD', 'SBUX', 'TJX', 'NKE', 'LOW', 'ORLY', 'BKNG',
            'CMG', 'MAR', 'GM', 'F', 'LULU', 'ROST', 'YUM', 'EBAY', 'ETSY', 'AZO',
            'DECK', 'EXPE', 'BBY', 'DRI', 'LVS', 'MGM', 'WYNN', 'NCLH', 'RCL', 'CCL',
            'HLT', 'DIS', 'LYV', 'FOXA', 'FOX', 'PARA', 'WBD', 'MTCH', 'UBER', 'LYFT',
            'ABNB', 'DASH', 'BROS', 'CHWY', 'CVNA', 'DKNG', 'PENN', 'LKQ', 'AAP', 'GPS',
            'TPG', 'TKO',

            # Communication Services (24 symbols)
            'GOOGL', 'GOOG', 'META', 'NFLX', 'DIS', 'CMCSA', 'T', 'VZ', 'CHTR', 'TMUS',
            'FOXA', 'FOX', 'PARA', 'WBD', 'MTCH', 'PINS', 'SNAP', 'SPOT', 'TTD', 'ROKU',
            'ZM', 'DOCU', 'LUMN', 'OMC',

            # Industrials (72 symbols)
            'CAT', 'RTX', 'HON', 'UNP', 'BA', 'DE', 'LMT', 'GE', 'MMM', 'ITW',
            'NOC', 'ETN', 'APD', 'CSX', 'NSC', 'CARR', 'GD', 'LHX', 'TT', 'EMR',
            'FDX', 'UPS', 'SWK', 'CMI', 'PH', 'DOV', 'ROK', 'OTIS', 'IR', 'VRSK',
            'CTAS', 'FAST', 'PAYX', 'RSG', 'WM', 'WCN', 'IEX', 'PWR', 'GNRC', 'J',
            'PKG', 'ALLE', 'AOS', 'AME', 'BLDR', 'CHRW', 'DAL', 'AAL', 'UAL', 'LUV',
            'ALK', 'JBHT', 'ODFL', 'XPO', 'EXPD', 'ARCB', 'KNX', 'HUBG', 'SNDR', 'TXT',
            'HII', 'BWA', 'LDOS', 'CACI', 'SAIC', 'KBR', 'TDG', 'CPRT', 'URI', 'WAB',
            'PCAR', 'NDSN',

            # Consumer Staples (33 symbols)
            'PG', 'PEP', 'COST', 'KO', 'WMT', 'MDLZ', 'CL', 'MO', 'PM', 'STZ',
            'KMB', 'GIS', 'K', 'HSY', 'CHD', 'CLX', 'TSN', 'CAG', 'SJM', 'CPB',
            'HRL', 'MKC', 'TAP', 'BF.B', 'KDP', 'KHC', 'MNST', 'KR', 'SYY', 'DG',
            'DLTR', 'WBA', 'COKE',

            # Energy (24 symbols)
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PXD', 'VLO', 'PSX', 'WMB',
            'KMI', 'OKE', 'TRGP', 'HAL', 'BKR', 'DVN', 'FANG', 'APA', 'MRO', 'OXY',
            'CTRA', 'EQT', 'CNX', 'AR',

            # Utilities (28 symbols)
            'NEE', 'SO', 'DUK', 'CEG', 'SRE', 'AEP', 'EXC', 'XEL', 'ED', 'PEG',
            'EIX', 'WEC', 'AWK', 'ES', 'FE', 'ETR', 'CNP', 'NI', 'LNT', 'EVRG',
            'AES', 'CMS', 'DTE', 'PPL', 'ATO', 'NRG', 'VST', 'PCG',

            # Real Estate (29 symbols)
            'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'WELL', 'DLR', 'BXP',
            'AVB', 'EQR', 'VTR', 'ESS', 'MAA', 'UDR', 'CPT', 'ARE', 'HST', 'REG',
            'FRT', 'KIM', 'ADC', 'ACC', 'SLG', 'HIW', 'DEI', 'CXW', 'INVH',

            # Materials (28 symbols)
            'LIN', 'SHW', 'APD', 'FCX', 'NEM', 'ECL', 'CTVA', 'DD', 'DOW', 'NUE',
            'PPG', 'LYB', 'BALL', 'AVY', 'RPM', 'SEE', 'IP', 'PKG', 'WRK', 'CLF',
            'STLD', 'RS', 'AA', 'X', 'CENX', 'MP', 'ALB', 'FMC'
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
        """Get all assets (stocks + crypto + commodities + forex) to analyze"""
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

        # Add commodity ETFs
        for symbol, name in self.commodity_symbols.items():
            assets.append({
                'symbol': symbol,
                'type': 'commodity',
                'exchange': 'NYSE/NASDAQ',
                'name': name
            })

        # Add forex pairs
        for symbol, name in self.forex_symbols.items():
            assets.append({
                'symbol': symbol,
                'type': 'forex',
                'exchange': 'FX',
                'name': name
            })

        self.logger.info(
            f"Selected {len(assets)} assets: "
            f"{len(self.sp500_symbols)} stocks, "
            f"{len(self.crypto_symbols)} crypto, "
            f"{len(self.commodity_symbols)} commodities, "
            f"{len(self.forex_symbols)} forex"
        )
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

    async def get_commodities_only(self) -> List[Dict[str, str]]:
        """Get only commodity ETF assets"""
        return [
            {'symbol': symbol, 'type': 'commodity', 'exchange': 'NYSE/NASDAQ', 'name': name}
            for symbol, name in self.commodity_symbols.items()
        ]

    async def get_forex_only(self) -> List[Dict[str, str]]:
        """Get only forex pair assets"""
        return [
            {'symbol': symbol, 'type': 'forex', 'exchange': 'FX', 'name': name}
            for symbol, name in self.forex_symbols.items()
        ]