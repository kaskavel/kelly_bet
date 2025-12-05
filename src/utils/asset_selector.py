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

        # Additional US stocks beyond S&P 500 (NASDAQ-100 and mid-caps)
        self.us_additional_symbols = [
            # Top NASDAQ tech/growth not in S&P 500 or smaller positions
            'RIVN', 'LCID', 'SOFI', 'HOOD', 'RBLX', 'U', 'DKNG', 'FUBO',
            'ZI', 'DOCN', 'FRSH', 'S', 'MDB', 'CFLT', 'NET', 'ESTC',
            'SNOW', 'DDOG', 'ZS', 'CRWD', 'PANW', 'FTNT', 'OKTA', 'ZM',
            'DOCU', 'TWLO', 'VEEV', 'SPLK', 'WDAY', 'NOW', 'CRM', 'TEAM',

            # Mid-cap growth stocks
            'ENPH', 'SEDG', 'FSLR', 'RUN', 'PLUG', 'BE', 'CHPT', 'BLNK',
            'BYND', 'TTCF', 'CELH', 'MNST', 'OLPX', 'CVNA', 'CAVA',
            'BROS', 'WING', 'TXRH', 'BLMN', 'CAKE', 'CHUY', 'BJRI',

            # E-commerce and consumer
            'SHOP', 'MELI', 'SE', 'CPNG', 'BABA', 'JD', 'PDD', 'BEKE',

            # Biotech mid-caps
            'VRTX', 'REGN', 'ALNY', 'BMRN', 'IONS', 'TECH', 'UTHR', 'INCY',
            'JAZZ', 'NBIX', 'EXAS', 'TDOC', 'ILMN', 'PACB', 'NVTA', 'IRTC',

            # Semiconductors
            'ARM', 'MPWR', 'WOLF', 'SWKS', 'QRVO', 'CRUS', 'SLAB', 'LSCC',

            # Other sectors
            'FOUR', 'BILL', 'PCTY', 'SMAR', 'GTLB', 'FIVN', 'PING', 'DT'
        ]

        # Japan stocks (Nikkei top 100)
        self.japan_symbols = [
            # Top 50 by market cap
            '7203.T',   # Toyota Motor
            '6758.T',   # Sony Group
            '9984.T',   # SoftBank Group
            '6861.T',   # Keyence
            '6954.T',   # Fanuc
            '7974.T',   # Nintendo
            '4063.T',   # Shin-Etsu Chemical
            '9432.T',   # NTT (Nippon Telegraph)
            '8306.T',   # Mitsubishi UFJ Financial
            '6902.T',   # Denso
            '9433.T',   # KDDI
            '7267.T',   # Honda Motor
            '8035.T',   # Tokyo Electron
            '4502.T',   # Takeda Pharmaceutical
            '4503.T',   # Astellas Pharma
            '6367.T',   # Daikin Industries
            '4568.T',   # Daiichi Sankyo
            '4452.T',   # Kao Corporation
            '4911.T',   # Shiseido
            '6501.T',   # Hitachi
            '6594.T',   # Nidec
            '6645.T',   # Omron
            '6971.T',   # Kyocera
            '7751.T',   # Canon
            '7752.T',   # Ricoh
            '8001.T',   # Itochu
            '8002.T',   # Marubeni
            '8031.T',   # Mitsui & Co
            '8053.T',   # Sumitomo
            '8058.T',   # Mitsubishi Corp
            '8267.T',   # Aeon
            '8411.T',   # Mizuho Financial
            '8766.T',   # Tokio Marine
            '8801.T',   # Mitsui Fudosan
            '8802.T',   # Mitsubishi Estate
            '9020.T',   # JR East
            '9021.T',   # JR Central
            '9022.T',   # JR West
            '9101.T',   # NYK Line
            '9104.T',   # Mitsui O.S.K. Lines
            '9202.T',   # ANA Holdings
            '9301.T',   # Mitsubishi Logistics
            '9501.T',   # Tokyo Electric Power
            '9502.T',   # Chubu Electric Power
            '9503.T',   # Kansai Electric Power
            '9531.T',   # Tokyo Gas
            '9532.T',   # Osaka Gas

            # Additional 50 (51-100)
            '2914.T',   # JT (Japan Tobacco)
            '3382.T',   # Seven & i Holdings
            '3659.T',   # Nexon
            '4021.T',   # Nissan Chemical
            '4042.T',   # Tosoh
            '4061.T',   # Denka
            '4183.T',   # Mitsui Chemicals
            '4188.T',   # Mitsubishi Chemical
            '4324.T',   # Dentsu Group
            '4543.T',   # Terumo
            '4578.T',   # Otsuka Holdings
            '5020.T',   # ENEOS Holdings
            '5108.T',   # Bridgestone
            '5201.T',   # AGC
            '5332.T',   # TOTO
            '5333.T',   # NGK Insulators
            '5401.T',   # Nippon Steel
            '5406.T',   # Kobe Steel
            '5411.T',   # JFE Holdings
            '5631.T',   # Nippon Steel & Sumitomo Metal
            '5711.T',   # Mitsubishi Materials
            '5713.T',   # Sumitomo Metal Mining
            '5802.T',   # Sumitomo Electric
            '5803.T',   # Fujikura
            '6098.T',   # Recruit Holdings
            '6178.T',   # Japan Post Bank
            '6301.T',   # Komatsu
            '6326.T',   # Kubota
            '6370.T',   # Kurita Water Industries
            '6471.T',   # NSK
            '6472.T',   # NTN
            '6473.T',   # JTEKT
            '6503.T',   # Mitsubishi Electric
            '6504.T',   # Fuji Electric
            '6701.T',   # NEC
            '6702.T',   # Fujitsu
            '6724.T',   # Seiko Epson
            '6753.T',   # Sharp
            '6762.T',   # TDK
            '6857.T',   # Advantest
            '6952.T',   # Casio Computer
            '6965.T',   # Hoya
            '7011.T',   # Mitsubishi Heavy Industries
            '7012.T',   # Kawasaki Heavy Industries
            '7013.T',   # IHI
            '7201.T',   # Nissan Motor
            '7202.T',   # Isuzu Motors
            '7269.T',   # Suzuki Motor
            '7270.T',   # Subaru
        ]

        # China stocks (Hong Kong + mainland top 80)
        self.china_symbols = [
            # Hong Kong Exchange (most liquid, best for international access)
            '0700.HK',  # Tencent Holdings
            '9988.HK',  # Alibaba Group
            '1211.HK',  # BYD Company
            '2318.HK',  # Ping An Insurance
            '3690.HK',  # Meituan
            '0941.HK',  # China Mobile
            '1398.HK',  # ICBC
            '3988.HK',  # Bank of China
            '0939.HK',  # China Construction Bank
            '2388.HK',  # BOC Hong Kong
            '1299.HK',  # AIA Group
            '2628.HK',  # China Life Insurance
            '0883.HK',  # CNOOC
            '0386.HK',  # China Petroleum & Chemical
            '0857.HK',  # PetroChina
            '0688.HK',  # China Overseas Land
            '1109.HK',  # China Resources Land
            '2007.HK',  # Country Garden
            '1093.HK',  # CSPC Pharmaceutical
            '2269.HK',  # Wuxi Biologics
            '1177.HK',  # Sino Biopharmaceutical
            '0992.HK',  # Lenovo Group
            '0981.HK',  # Semiconductor Manufacturing Intl
            '2382.HK',  # Sunny Optical Technology
            '1810.HK',  # Xiaomi Corporation
            '9618.HK',  # JD.com
            '3968.HK',  # China Merchants Bank
            '9999.HK',  # NetEase
            '1024.HK',  # Kuaishou Technology
            '9961.HK',  # Trip.com Group
            '9896.HK',  # JD Health International
            '2020.HK',  # ANTA Sports
            '3319.HK',  # China Resources Pharmaceuticals
            '0868.HK',  # Xinyi Glass Holdings
            '1928.HK',  # Sands China
            '0027.HK',  # Galaxy Entertainment
            '0388.HK',  # Hong Kong Exchanges and Clearing
            '0016.HK',  # Sun Hung Kai Properties
            '1113.HK',  # CK Asset Holdings

            # Shanghai Stock Exchange (mainland)
            '600519.SS', # Kweichow Moutai
            '600036.SS', # China Merchants Bank
            '601318.SS', # Ping An Bank
            '600030.SS', # CITIC Securities
            '601166.SS', # Industrial Bank
            '601288.SS', # Agricultural Bank of China
            '601398.SS', # ICBC
            '601939.SS', # China Construction Bank
            '601988.SS', # Bank of China
            '600887.SS', # Inner Mongolia Yili
            '600276.SS', # Jiangsu Hengrui Medicine
            '600809.SS', # Shanxi Xinghuacun Fen Wine
            '601888.SS', # China Tourism Group
            '600900.SS', # China Yangtze Power
            '601012.SS', # Longfor Group Holdings
            '603259.SS', # WuXi AppTec
            '603288.SS', # Foshan Haitian Flavouring
            '600585.SS', # Anhui Conch Cement
            '600690.SS', # Haier Smart Home
            '600438.SS', # Tongwei

            # Shenzhen Stock Exchange (tech-focused)
            '000858.SZ', # Wuliangye Yibin
            '000333.SZ', # Midea Group
            '002594.SZ', # BYD Company (A-shares)
            '002415.SZ', # Hikvision
            '000651.SZ', # Gree Electric Appliances
            '002714.SZ', # Muyuan Foods
            '300750.SZ', # Contemporary Amperex Technology (CATL)
            '002475.SZ', # Luxshare Precision
            '300059.SZ', # East Money Information
            '000568.SZ', # Luzhou Laojiao
            '002027.SZ', # Shenzhen Yan Tian Port
            '002352.SZ', # Shenzhen Inovance Technology
            '300015.SZ', # Aier Eye Hospital Group
            '002624.SZ', # Perfect World
            '000002.SZ', # China Vanke
            '000001.SZ', # Ping An Bank (A-shares)
            '300142.SZ', # Wotu Software Engineering
            '002241.SZ', # GoerTek
            '000725.SZ', # BOE Technology Group
            '002460.SZ', # Jiangxi Ganfeng Lithium
        ]

        # Germany stocks (DAX 40 + additional)
        self.germany_symbols = [
            # DAX 40
            'SAP.DE',    # SAP
            'SIE.DE',    # Siemens
            'VOW3.DE',   # Volkswagen
            'BMW.DE',    # BMW
            'MBG.DE',    # Mercedes-Benz Group
            'DTE.DE',    # Deutsche Telekom
            'ALV.DE',    # Allianz
            'BAS.DE',    # BASF
            'ADS.DE',    # Adidas
            'AIR.DE',    # Airbus
            'BAYN.DE',   # Bayer
            'BEI.DE',    # Beiersdorf
            'CBK.DE',    # Commerzbank
            'CON.DE',    # Continental
            'DB1.DE',    # Deutsche Börse
            'DBK.DE',    # Deutsche Bank
            'DHL.DE',    # Deutsche Post
            'DPW.DE',    # Deutsche Post
            'EOAN.DE',   # E.ON
            'FME.DE',    # Fresenius Medical Care
            'FRE.DE',    # Fresenius
            'HEI.DE',    # HeidelbergCement
            'HEN3.DE',   # Henkel
            'HNR1.DE',   # Hannover Re
            'IFX.DE',    # Infineon Technologies
            'LIN.DE',    # Linde
            'MRK.DE',    # Merck
            'MTX.DE',    # MTU Aero Engines
            'MUV2.DE',   # Munich Re
            'P911.DE',   # Porsche
            'PUM.DE',    # Puma
            'QIA.DE',    # Qiagen
            'RWE.DE',    # RWE
            'SHL.DE',    # Siemens Healthineers
            'SRT3.DE',   # Sartorius
            'VNA.DE',    # Vonovia
            'WDI.DE',    # Wirecard (if still listed)
            'ZAL.DE',    # Zalando
            'HFG.DE',    # HelloFresh
            'PAH3.DE',   # Porsche Automobil Holding
        ]

        # UK stocks (FTSE top 80)
        self.uk_symbols = [
            # FTSE 100 top stocks
            'HSBA.L',    # HSBC Holdings
            'BP.L',      # BP
            'SHEL.L',    # Shell
            'AZN.L',     # AstraZeneca
            'ULVR.L',    # Unilever
            'GSK.L',     # GlaxoSmithKline
            'DGE.L',     # Diageo
            'RIO.L',     # Rio Tinto
            'LSEG.L',    # London Stock Exchange Group
            'NG.L',      # National Grid
            'BARC.L',    # Barclays
            'LLOY.L',    # Lloyds Banking Group
            'VOD.L',     # Vodafone
            'BT-A.L',    # BT Group
            'PRU.L',     # Prudential
            'STAN.L',    # Standard Chartered
            'BA.L',      # BAE Systems
            'IMB.L',     # Imperial Brands
            'CNA.L',     # Centrica
            'SSE.L',     # SSE
            'GLEN.L',    # Glencore
            'AAL.L',     # Anglo American
            'ANTO.L',    # Antofagasta
            'BATS.L',    # British American Tobacco
            'REL.L',     # RELX
            'III.L',     # 3i Group
            'EXPN.L',    # Experian
            'RKT.L',     # Reckitt Benckiser
            'ABF.L',     # Associated British Foods
            'SBRY.L',    # Sainsbury's
            'TSCO.L',    # Tesco
            'MKS.L',     # Marks & Spencer
            'NXT.L',     # Next
            'FRES.L',    # Fresnillo
            'BRBY.L',    # Burberry
            'RR.L',      # Rolls-Royce Holdings
            'IAG.L',     # International Airlines Group
            'EZJ.L',     # easyJet
            'OCDO.L',    # Ocado Group
            'AUTO.L',    # Auto Trader Group
            'BNZL.L',    # Bunzl
            'CRH.L',     # CRH
            'CRDA.L',    # Croda International
            'DCC.L',     # DCC
            'FCIT.L',    # F&C Investment Trust
            'FLTR.L',    # Flutter Entertainment
            'HLMA.L',    # Halma
            'HIK.L',     # Hikma Pharmaceuticals
            'ITRK.L',    # Intertek Group
            'IHG.L',     # InterContinental Hotels
            'JD.L',      # JD Sports Fashion
            'KGF.L',     # Kingfisher
            'LAND.L',    # Land Securities
            'LGEN.L',    # Legal & General
            'MNDI.L',    # Mondi
            'NWG.L',     # NatWest Group
            'PSN.L',     # Persimmon
            'PSON.L',    # Pearson
            'SGE.L',     # Sage Group
            'SDR.L',     # Schroders
            'SGRO.L',    # Segro
            'SVT.L',     # Severn Trent
            'SMDS.L',    # Smith & Nephew
            'SMIN.L',    # Smiths Group
            'SPX.L',     # Spirax-Sarco Engineering
            'SSE.L',     # SSE
            'STJ.L',     # St James's Place
            'TW.L',      # Taylor Wimpey
            'ULVR.L',    # Unilever
            'UU.L',      # United Utilities
            'WTB.L',     # Whitbread
            'WPP.L',     # WPP
            'AHT.L',     # Ashtead Group
            'AVV.L',     # Aviva
            'BDEV.L',    # Barratt Developments
            'BKG.L',     # Berkeley Group
            'CCH.L',     # Coca-Cola HBC
            'CPG.L',     # Compass Group
        ]

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

        # Add S&P 500 stocks (USD)
        for symbol in self.sp500_symbols:
            assets.append({
                'symbol': symbol,
                'type': 'stock',
                'exchange': 'NYSE/NASDAQ',
                'currency': 'USD',
                'country': 'US'
            })

        # Add additional US stocks (USD)
        for symbol in self.us_additional_symbols:
            assets.append({
                'symbol': symbol,
                'type': 'stock',
                'exchange': 'NASDAQ',
                'currency': 'USD',
                'country': 'US'
            })

        # Add Japan stocks (JPY)
        for symbol in self.japan_symbols:
            assets.append({
                'symbol': symbol,
                'type': 'stock',
                'exchange': 'TSE',
                'currency': 'JPY',
                'country': 'Japan'
            })

        # Add China stocks (HKD/CNY)
        for symbol in self.china_symbols:
            # Determine currency based on exchange
            if '.HK' in symbol:
                currency = 'HKD'
                exchange = 'HKEX'
            elif '.SS' in symbol:
                currency = 'CNY'
                exchange = 'SSE'
            elif '.SZ' in symbol:
                currency = 'CNY'
                exchange = 'SZSE'
            else:
                currency = 'CNY'
                exchange = 'China'

            assets.append({
                'symbol': symbol,
                'type': 'stock',
                'exchange': exchange,
                'currency': currency,
                'country': 'China'
            })

        # Add Germany stocks (EUR)
        for symbol in self.germany_symbols:
            assets.append({
                'symbol': symbol,
                'type': 'stock',
                'exchange': 'XETRA',
                'currency': 'EUR',
                'country': 'Germany'
            })

        # Add UK stocks (GBP - pence actually, but we'll handle conversion)
        for symbol in self.uk_symbols:
            assets.append({
                'symbol': symbol,
                'type': 'stock',
                'exchange': 'LSE',
                'currency': 'GBP',
                'country': 'UK'
            })

        # Add top crypto (USD)
        for symbol in self.crypto_symbols:
            assets.append({
                'symbol': symbol,
                'type': 'crypto',
                'exchange': 'binance',
                'currency': 'USD',
                'country': 'Global'
            })

        # Add commodity ETFs (USD)
        for symbol, name in self.commodity_symbols.items():
            assets.append({
                'symbol': symbol,
                'type': 'commodity',
                'exchange': 'NYSE/NASDAQ',
                'currency': 'USD',
                'country': 'US',
                'name': name
            })

        # Add forex pairs (quoted in second currency)
        for symbol, name in self.forex_symbols.items():
            assets.append({
                'symbol': symbol,
                'type': 'forex',
                'exchange': 'FX',
                'currency': 'USD',  # All forex normalized to USD base
                'country': 'Global',
                'name': name
            })

        total_stocks = (len(self.sp500_symbols) + len(self.us_additional_symbols) +
                       len(self.japan_symbols) + len(self.china_symbols) +
                       len(self.germany_symbols) + len(self.uk_symbols))

        self.logger.info(
            f"Selected {len(assets)} assets: "
            f"{total_stocks} stocks (US: {len(self.sp500_symbols) + len(self.us_additional_symbols)}, "
            f"Japan: {len(self.japan_symbols)}, China: {len(self.china_symbols)}, "
            f"Germany: {len(self.germany_symbols)}, UK: {len(self.uk_symbols)}), "
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