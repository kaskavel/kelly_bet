#!/usr/bin/env python3
"""
Streamlit-based trading dashboard for Kelly Criterion Trading System
"""

import asyncio
import logging
import struct
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
# NOTE: streamlit_autorefresh deliberately NOT imported. Timed auto-refresh was
# re-fetching the whole universe unattended and exhausting the yfinance rate limit;
# market data is now fetched only on an explicit button press.

try:
    from src.core.trading_system import TradingSystem
    from src.cli.bet_monitor import BetMonitor
    from src.cli.bet_analyzer import BetAnalyzer
    from src.portfolio.manager import PortfolioManager
    from src.data.market_data import MarketDataManager
    from src.utils.asset_names import get_display_name, get_asset_name
    from src.utils.currency_converter import CurrencyConverter
    import yaml
    REAL_DATA_AVAILABLE = True
except ImportError as e:
    st.error(f"Missing dependencies: {e}")
    REAL_DATA_AVAILABLE = False
    # For development/testing - mock these classes
    class TradingSystem:
        def __init__(self, *args, **kwargs):
            pass
    
    class BetMonitor:
        def __init__(self, *args, **kwargs):
            pass
    
    class BetAnalyzer:
        def __init__(self, *args, **kwargs):
            pass
    
    class PortfolioManager:
        def __init__(self, *args, **kwargs):
            pass
    
    class MarketDataManager:
        def __init__(self, *args, **kwargs):
            pass


class TradingDashboard:
    """Main dashboard class for the trading system UI"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.setup_logging()
        
        # Load config
        self.config = {}
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            st.warning(f"Config file not found: {config_path}")
        
        # Initialize managers
        self.portfolio_manager = None
        self.bet_monitor = None
        self.market_data = None
        
        if REAL_DATA_AVAILABLE:
            try:
                self.logger.info("Initializing real data managers...")
                self.market_data = MarketDataManager(self.config)
                self.portfolio_manager = PortfolioManager(self.config)
                self.bet_monitor = BetMonitor(config_path)
                self.logger.info("✅ All managers initialized successfully")
            except Exception as e:
                error_msg = f"❌ CRITICAL: Failed to initialize data managers: {e}"
                st.error(error_msg)
                st.error("🔧 Please check:")
                st.error("1. config/config.yaml exists and is valid")
                st.error("2. Database file is accessible")
                st.error("3. All required packages are installed")
                st.stop()  # Stop execution - don't continue with broken setup
                self.logger.error(f"Manager initialization error: {e}")
        else:
            error_msg = "❌ CRITICAL: Required dependencies not available!"
            st.error(error_msg)
            st.error("Missing required packages. Please install them and restart.")
            st.stop()  # Stop execution
        
        # Initialize session state
        if 'trading_system' not in st.session_state:
            st.session_state.trading_system = None
        if 'auto_mode' not in st.session_state:
            st.session_state.auto_mode = False
        if 'auto_threshold' not in st.session_state:
            st.session_state.auto_threshold = 60.0
        if 'last_update' not in st.session_state:
            st.session_state.last_update = None
        if 'opportunities_data' not in st.session_state:
            st.session_state.opportunities_data = []
        if 'portfolio_data' not in st.session_state:
            st.session_state.portfolio_data = {}
        if 'active_bets_data' not in st.session_state:
            st.session_state.active_bets_data = []
        if 'all_bets_data' not in st.session_state:
            st.session_state.all_bets_data = ([], [])  # (alive_bets, closed_bets)
        if 'needs_refresh' not in st.session_state:
            st.session_state.needs_refresh = False
    
    def setup_logging(self):
        """Setup logging for the dashboard"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def initialize_trading_system(self):
        """Initialize the trading system if not already done"""
        try:
            if st.session_state.trading_system is None:
                mode = "automated" if st.session_state.auto_mode else "manual"
                st.session_state.trading_system = TradingSystem(
                    config_path=self.config_path,
                    mode=mode,
                    auto_threshold=st.session_state.auto_threshold
                )
            return st.session_state.trading_system
        except Exception as e:
            st.error(f"Failed to initialize trading system: {e}")
            return None
    
    async def get_opportunities_data(self) -> List[Dict]:
        """Get all opportunities from trading system with individual algorithm predictions"""
        try:
            self.logger.info("Getting comprehensive opportunities data...")

            if not self.market_data or not self.portfolio_manager:
                error_msg = "❌ CRITICAL: Market data or portfolio manager not initialized properly!"
                self.logger.error(error_msg)
                st.error(error_msg)
                st.error("Please check your configuration file (config/config.yaml) and restart the dashboard.")
                return []

            self.logger.info("Initializing trading system components...")

            # Initialize components
            from src.prediction.predictor import PredictionEngine
            from src.kelly.calculator import KellyCalculator
            from src.utils.asset_selector import AssetSelector

            predictor = PredictionEngine(self.config)
            kelly_calc = KellyCalculator(self.config)
            asset_selector = AssetSelector(self.config)

            self.logger.info("Initializing market data and portfolio...")
            await self.market_data.initialize()
            await self.portfolio_manager.initialize()

            self.logger.info("Getting available capital...")
            available_capital = await self.portfolio_manager.get_available_capital()
            # Open positions, for the Kelly correlation haircut.
            try:
                open_positions = await self.portfolio_manager._count_alive_bets()
            except Exception:
                open_positions = 0
            self.logger.info(f"Available capital: ${available_capital:.2f}")

            # Get ALL assets (stocks and crypto)
            self.logger.info("Getting complete asset list...")
            all_assets = await asset_selector.get_all_assets()
            self.logger.info(f"Total assets available: {len(all_assets)}")

            opportunities = []

            # Process all assets with REAL ML predictions
            assets_to_process = all_assets
            self.logger.info(f"Processing {len(assets_to_process)} assets with REAL ML predictions...")

            # Get market data for all assets first
            market_data = {}
            self.logger.info("Fetching market data for all assets...")
            for asset in assets_to_process:
                symbol = asset['symbol']
                try:
                    stock_data = await self.market_data.get_stock_data(symbol, days=150, extend_history=True)  # Incremental updates (accounts for weekends/holidays)
                    if stock_data is not None and not stock_data.empty:
                        market_data[symbol] = stock_data
                        self.logger.info(f"Loaded {len(stock_data)} days of data for {symbol}")
                    else:
                        self.logger.warning(f"No market data for {symbol}")
                except Exception as e:
                    self.logger.warning(f"Failed to get market data for {symbol}: {e}")

            self.logger.info(f"Successfully loaded market data for {len(market_data)} assets")

            # Currency conversion: Convert all non-USD prices to USD
            self.logger.info("Initializing currency converter...")
            currency_converter = CurrencyConverter()

            # Extract forex data and update converter rates
            forex_data = {}
            for symbol, data in market_data.items():
                if symbol.endswith('=X'):  # Forex pair
                    forex_data[symbol] = data

            if forex_data:
                currency_converter.update_rates(forex_data)
                self.logger.info(f"Updated currency rates from {len(forex_data)} forex pairs")

            # Convert all non-USD asset prices to USD
            converted_market_data = {}
            for symbol, data in market_data.items():
                # Find asset currency
                asset = next((a for a in assets_to_process if a['symbol'] == symbol), None)
                currency = asset.get('currency', 'USD') if asset else 'USD'

                if currency == 'USD' or symbol.endswith('=X'):
                    # Already in USD or is a forex pair
                    converted_market_data[symbol] = data
                else:
                    # Convert to USD
                    try:
                        converted_data = currency_converter.convert_price_series(data, currency)
                        converted_market_data[symbol] = converted_data
                        rate = currency_converter.get_rate(currency)
                        self.logger.debug(f"Converted {symbol} from {currency} to USD (rate: {rate:.4f})")
                    except Exception as e:
                        self.logger.warning(f"Failed to convert {symbol} from {currency} to USD: {e}")
                        # Use original data if conversion fails
                        converted_market_data[symbol] = data

            self.logger.info(f"All prices normalized to USD for ML processing")

            # Now generate REAL predictions using the PredictionEngine
            if converted_market_data:
                self.logger.info("Generating REAL ML predictions...")
                try:
                    # Use the actual prediction engine to generate real predictions
                    self.logger.info("Initializing prediction engine...")
                    await predictor.initialize()

                    # Handle force retrain if requested
                    if st.session_state.get('force_retrain', False):
                        self.logger.info("🔄 Force retraining requested - clearing existing models...")
                        # Clear model cache to force retrain
                        # This would need to be implemented in the predictor
                        st.session_state.force_retrain = False

                    self.logger.info("Generating real ML predictions (this may take time for training)...")
                    predictions = await predictor.predict_all(converted_market_data)
                    self.logger.info(f"✅ Generated {len(predictions)} real ML predictions (all in USD)")

                    for prediction in predictions:
                        symbol = prediction['symbol']
                        asset = next((a for a in assets_to_process if a['symbol'] == symbol), None)
                        asset_type = asset.get('type', 'stock') if asset else 'stock'

                        try:
                            current_price = prediction['current_price']

                            # Extract individual algorithm predictions with error handling
                            algorithms_dict = {}
                            final_probability = prediction['probability']
                            failed_algorithms = []

                            # Get individual algorithm results if available
                            if 'algorithms' in prediction:
                                for algo_result in prediction['algorithms']:
                                    algo_name = algo_result.get('algorithm', 'unknown')
                                    algo_prob = algo_result.get('probability', 0.0)
                                    algo_error = algo_result.get('error', None)

                                    # Map algorithm names to display names
                                    if algo_name == 'lstm':
                                        if algo_error:
                                            failed_algorithms.append(f"LSTM: {algo_error}")
                                            algorithms_dict['lstm'] = None  # No hallucination
                                        else:
                                            algorithms_dict['lstm'] = algo_prob
                                    elif algo_name == 'rf':
                                        if algo_error:
                                            failed_algorithms.append(f"Random Forest: {algo_error}")
                                            algorithms_dict['random_forest'] = None  # No hallucination
                                        else:
                                            algorithms_dict['random_forest'] = algo_prob
                                    elif algo_name == 'sma':
                                        if algo_error:
                                            failed_algorithms.append(f"SMA: {algo_error}")
                                            algorithms_dict['sma'] = None  # No hallucination
                                        else:
                                            algorithms_dict['sma'] = algo_prob
                                    elif algo_name == 'rsi':
                                        if algo_error:
                                            failed_algorithms.append(f"RSI: {algo_error}")
                                            algorithms_dict['rsi'] = None  # No hallucination
                                        else:
                                            algorithms_dict['rsi'] = algo_prob
                                    elif algo_name == 'regression':
                                        if algo_error:
                                            failed_algorithms.append(f"Regression: {algo_error}")
                                            algorithms_dict['regression'] = None  # No hallucination
                                        else:
                                            algorithms_dict['regression'] = algo_prob
                                    elif algo_name == 'svm':
                                        if algo_error:
                                            failed_algorithms.append(f"SVM: {algo_error}")
                                            algorithms_dict['svm'] = None  # No hallucination
                                        else:
                                            algorithms_dict['svm'] = algo_prob

                            # Ensure we have all six algorithms (mark missing as None)
                            if 'lstm' not in algorithms_dict:
                                algorithms_dict['lstm'] = None
                                failed_algorithms.append("LSTM: Model not available")
                            if 'random_forest' not in algorithms_dict:
                                algorithms_dict['random_forest'] = None
                                failed_algorithms.append("Random Forest: Model not available")
                            if 'sma' not in algorithms_dict:
                                algorithms_dict['sma'] = None
                                failed_algorithms.append("SMA: Model not available")
                            if 'rsi' not in algorithms_dict:
                                algorithms_dict['rsi'] = None
                                failed_algorithms.append("RSI: Model not available")
                            if 'regression' not in algorithms_dict:
                                algorithms_dict['regression'] = None
                                failed_algorithms.append("Regression: Model not available")
                            if 'svm' not in algorithms_dict:
                                algorithms_dict['svm'] = None
                                failed_algorithms.append("SVM: Model not available")

                            # Calculate Kelly recommendation using REAL probability,
                            # on THIS asset's barriers and against the open book.
                            # Omitting the thresholds sized every asset on the config
                            # defaults, which under volatility-scaled barriers prices
                            # a different bet than the one being offered.
                            kelly_rec = kelly_calc.calculate_bet_size(
                                probability=final_probability,
                                current_price=current_price,
                                available_capital=available_capital,
                                win_threshold=prediction.get('win_threshold'),
                                loss_threshold=prediction.get('loss_threshold'),
                                concurrent_positions=open_positions + 1,
                            )

                            opportunities.append({
                                "symbol": symbol,
                                "asset_type": asset_type,
                                "raw_score": prediction.get('raw_score', final_probability),
                                "win_threshold": prediction.get('win_threshold'),
                                "loss_threshold": prediction.get('loss_threshold'),
                                "sigma_pct": prediction.get('sigma_pct'),
                                "break_even_pct": prediction.get('break_even_pct'),
                                "required_edge_pct": self._required_edge_for(prediction),
                                "expected_days_to_win": self._expected_days(prediction),
                                "currency": asset.get('currency', 'USD') if asset else 'USD',  # Track original currency
                                "current_price": current_price,  # Already in USD after conversion
                                "final_probability": final_probability,
                                "algorithms": algorithms_dict,
                                "kelly_fraction": kelly_rec.fraction_of_capital if kelly_rec.is_favorable else 0.0,
                                "recommended_amount": kelly_rec.recommended_amount if kelly_rec.is_favorable else 0.0,
                                "is_favorable": kelly_rec.is_favorable,
                                "prediction_confidence": prediction.get('confidence', 0.0),
                                "risk_warning": kelly_rec.risk_warning if hasattr(kelly_rec, 'risk_warning') else "",
                                "failed_algorithms": failed_algorithms
                            })

                            # Log with algorithm status
                            if failed_algorithms:
                                self.logger.warning(f"⚠️  {symbol}: {final_probability:.1f}% (Real ML) - Some algorithms failed: {', '.join(failed_algorithms)}")
                            else:
                                self.logger.info(f"✅ {symbol}: {final_probability:.1f}% probability (Real ML - all algorithms working)")

                        except Exception as e:
                            self.logger.error(f"Error processing prediction for {symbol}: {e}")
                            continue

                except Exception as e:
                    self.logger.error(f"Error generating real predictions: {e}")
                    st.error(f"Error generating real ML predictions: {e}")
                    return []

            self.logger.info(f"Generated {len(opportunities)} real opportunities")

            # Sort by final probability descending
            opportunities.sort(key=lambda x: x['final_probability'], reverse=True)
            return opportunities

        except Exception as e:
            self.logger.error(f"Critical error in get_opportunities_data: {e}", exc_info=True)
            st.error(f"Error getting opportunities: {e}")
            return []
    
    async def get_portfolio_data(self, fetch_prices: bool = True) -> Dict:
        """Get portfolio status data"""
        try:
            if not self.portfolio_manager:
                error_msg = "❌ Portfolio manager not available - cannot load real portfolio data!"
                self.logger.error(error_msg)
                st.error(error_msg)
                return {
                    "total_capital": 0.0,
                    "available_capital": 0.0,
                    "active_bets_value": 0.0,
                    "total_pnl": 0.0,
                    "win_rate": 0.0,
                    "total_bets": 0,
                    "completed_bets": 0,
                    "won_bets": 0,
                    "lost_bets": 0,
                    "active_bets": 0
                }
            
            # Get real portfolio data
            await self.portfolio_manager.initialize()

            # Refresh portfolio state to ensure in-memory data matches database after any settlements
            await self.portfolio_manager._load_portfolio_state()

            # Update current prices for active bets to get accurate unrealized P&L.
            # This is a network call, so it is skipped on the DB-only path; equity
            # then reflects the last marked-to-market prices.
            if fetch_prices:
                await self._update_active_bet_prices()

            portfolio_summary = await self.portfolio_manager.get_portfolio_summary()
            bet_statistics = await self.portfolio_manager.get_bet_statistics()

            # Debug logging to see what values we're actually getting
            self.logger.info(f"Portfolio data - Cash: ${portfolio_summary.cash_balance:.2f}, Total: ${portfolio_summary.total_capital:.2f}, Active: ${portfolio_summary.active_bets_value:.2f}")

            return {
                "total_capital": portfolio_summary.total_capital,
                "available_capital": portfolio_summary.cash_balance,
                "active_bets_value": portfolio_summary.active_bets_value,
                "total_pnl": portfolio_summary.unrealized_pnl + portfolio_summary.realized_pnl,
                "total_return": portfolio_summary.total_capital - self.config.get('trading', {}).get('initial_capital', 10000.0),
                "win_rate": bet_statistics["win_rate"],
                "total_bets": bet_statistics["total_bets"],
                "completed_bets": bet_statistics["completed_bets"],
                "won_bets": bet_statistics["won_bets"],
                "lost_bets": bet_statistics["lost_bets"],
                "active_bets": bet_statistics["active_bets"]
            }
        except Exception as e:
            st.error(f"Error getting portfolio data: {e}")
            self.logger.error(f"Portfolio data error: {e}")
            return {}
    
    async def get_active_bets_data(self, fetch_prices: bool = True) -> List[Dict]:
        """Get active bets data"""
        try:
            if not self.portfolio_manager:
                error_msg = "❌ Portfolio manager not available - cannot load real active bets!"
                self.logger.error(error_msg)
                st.error(error_msg)
                return []

            # Get real active bets
            await self.portfolio_manager.initialize()
            active_bets = await self.portfolio_manager.get_alive_bets()

            # Get current market prices for all active bets (with USD conversion).
            # Skipped entirely when fetch_prices is False: the caller then relies on
            # the last marked-to-market price stored on each bet, so rendering the
            # page costs no API quota.
            symbols = list(set(bet.symbol for bet in active_bets))
            current_prices = {}

            if not fetch_prices:
                current_prices = {bet.symbol: bet.current_price for bet in active_bets
                                  if bet.current_price}
                self.logger.debug(f"Using {len(current_prices)} stored prices "
                                  f"(no API calls)")
            elif symbols and self.market_data:
                try:
                    await self.market_data.initialize()

                    # Initialize currency converter
                    currency_converter = CurrencyConverter()

                    # Get forex rates first
                    forex_symbols = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X', 'USDCNY=X',
                                   'AUDUSD=X', 'USDCAD=X', 'NZDUSD=X']
                    forex_data = {}
                    for fx_symbol in forex_symbols:
                        try:
                            fx_data = await self.market_data.get_stock_data(fx_symbol, days=2)
                            if not fx_data.empty:
                                forex_data[fx_symbol] = fx_data
                        except:
                            pass

                    if forex_data:
                        currency_converter.update_rates(forex_data)

                    # Get prices for each symbol and convert to USD
                    for symbol in symbols:
                        try:
                            recent_data = await self.market_data.get_stock_data(symbol, days=2)
                            if not recent_data.empty:
                                raw_price = float(recent_data['Close'].iloc[-1])

                                # Get currency for this bet
                                bet_with_symbol = next((b for b in active_bets if b.symbol == symbol), None)
                                if bet_with_symbol and hasattr(bet_with_symbol, 'currency'):
                                    currency = bet_with_symbol.currency
                                else:
                                    # Fallback: detect from symbol
                                    if symbol.endswith('.T'):
                                        currency = 'JPY'
                                    elif symbol.endswith('.HK'):
                                        currency = 'HKD'
                                    elif symbol.endswith(('.SS', '.SZ')):
                                        currency = 'CNY'
                                    elif symbol.endswith('.DE'):
                                        currency = 'EUR'
                                    elif symbol.endswith('.L'):
                                        currency = 'GBP'
                                    else:
                                        currency = 'USD'

                                # Convert to USD if needed
                                if currency != 'USD':
                                    usd_price = currency_converter.convert_to_usd(raw_price, currency)
                                    self.logger.debug(f"Converted {symbol}: {raw_price:.2f} {currency} → ${usd_price:.2f} USD")
                                    current_prices[symbol] = usd_price
                                else:
                                    current_prices[symbol] = raw_price
                        except Exception as e:
                            self.logger.warning(f"Failed to get current price for {symbol}: {e}")
                except Exception as e:
                    self.logger.error(f"Error fetching current prices: {e}")

            bets_data = []
            for bet in active_bets:
                # Update current price if we have fresh market data
                current_price = current_prices.get(bet.symbol, bet.current_price)

                # Calculate current P&L
                pnl_dollars = (current_price - bet.entry_price) * bet.shares
                pnl_pct = ((current_price - bet.entry_price) / bet.entry_price) * 100

                bets_data.append({
                    "symbol": bet.symbol,
                    "entry_price": bet.entry_price,
                    "current_price": current_price,
                    "amount": bet.amount,
                    "pnl": pnl_dollars,
                    "pnl_pct": pnl_pct,
                    "entry_time": bet.entry_time,
                    # Prices and percentages are kept under distinct keys. These two
                    # code paths previously both wrote "win_threshold" -- one a price,
                    # the other a percentage -- and the table formatted both as
                    # dollars, so an 8.5% barrier rendered as "$8.50".
                    "win_price": bet.win_price,
                    "loss_price": bet.loss_price,
                    "win_pct": bet.win_threshold,
                    "loss_pct": bet.loss_threshold,
                    "bet_id": bet.bet_id,
                    "asset_type": bet.asset_type,
                    "shares": bet.shares,
                    "algorithm_used": bet.algorithm_used,
                    "probability_when_placed": bet.probability_when_placed
                })

            return bets_data

        except Exception as e:
            st.error(f"Error getting active bets: {e}")
            self.logger.error(f"Active bets error: {e}")
            return []

    async def get_all_bets_data(self, fetch_prices: bool = True) -> Tuple[List[Dict], List[Dict]]:
        """Get all bets data split into alive and closed bets"""
        try:
            if not self.portfolio_manager:
                error_msg = "❌ Portfolio manager not available - cannot load real bet history!"
                self.logger.error(error_msg)
                st.error(error_msg)
                return [], []

            await self.portfolio_manager.initialize()

            # Get all bets from database
            import sqlite3
            conn = sqlite3.connect(self.portfolio_manager.db_path)
            cursor = conn.cursor()

            try:
                # Get all bets ordered by entry time
                cursor.execute('''
                SELECT bet_id, symbol, asset_type, entry_price, entry_time, amount, shares,
                       win_threshold, loss_threshold, win_price, loss_price, current_price,
                       status, algorithm_used, probability_when_placed, exit_time, exit_price, realized_pnl
                FROM bets
                ORDER BY entry_time DESC
                ''')

                rows = cursor.fetchall()
                alive_bets = []
                closed_bets = []

                # Collect all alive bet symbols to fetch current prices
                alive_symbols = set()
                bet_rows = []

                for row in rows:
                    bet_id_full = row[0]
                    bet_data = {
                        "bet_id": bet_id_full[:8],  # Short ID for display
                        "full_bet_id": bet_id_full,  # Keep full ID for operations
                        "symbol": row[1],
                        "asset_type": row[2],
                        "entry_price": float(row[3]),
                        "entry_time": datetime.fromisoformat(row[4]),
                        "amount": float(row[5]),
                        "shares": float(row[6]),
                        "win_pct": float(row[7]),
                        "loss_pct": float(row[8]),
                        "win_price": float(row[9]),
                        "loss_price": float(row[10]),
                        "current_price": float(row[11]) if row[11] else float(row[3]),
                        "status": row[12],
                        "algorithm_used": row[13] or "unknown",
                        "probability_when_placed": float(row[14]) if row[14] else 0.0,
                        "exit_time": datetime.fromisoformat(row[15]) if row[15] else None,
                        "exit_price": float(row[16]) if row[16] else None,
                        "realized_pnl": float(row[17]) if row[17] else None
                    }

                    # Fetch individual algorithm predictions for this bet
                    cursor.execute('''
                    SELECT algorithm, probability
                    FROM bet_predictions
                    WHERE bet_id = ?
                    ORDER BY algorithm
                    ''', (bet_id_full,))

                    predictions = cursor.fetchall()
                    # Convert probability to float, handling both binary and text formats
                    def safe_convert_prob(prob):
                        if prob is None:
                            return 0.0
                        if isinstance(prob, bytes):
                            # Binary format - decode as 4-byte float (little-endian)
                            try:
                                return struct.unpack('<f', prob)[0]
                            except:
                                return 0.0
                        try:
                            return float(prob)
                        except:
                            return 0.0

                    bet_data["algorithm_predictions"] = {
                        algo: safe_convert_prob(prob)
                        for algo, prob in predictions
                    }

                    bet_rows.append(bet_data)
                    if bet_data["status"] == "alive":
                        alive_symbols.add(bet_data["symbol"])

                # Get current market prices for alive bets (with USD conversion).
                # Skipped when fetch_prices is False so the page can render from the
                # database without spending API quota.
                current_prices = {}
                if not fetch_prices:
                    current_prices = {r["symbol"]: r.get("current_price")
                                      for r in bet_rows
                                      if r["status"] == "alive" and r.get("current_price")}
                elif alive_symbols and self.market_data:
                    try:
                        await self.market_data.initialize()

                        # Initialize currency converter
                        currency_converter = CurrencyConverter()

                        # Get forex rates
                        forex_symbols = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X', 'USDCNY=X',
                                       'AUDUSD=X', 'USDCAD=X', 'NZDUSD=X']
                        forex_data = {}
                        for fx_symbol in forex_symbols:
                            try:
                                fx_data = await self.market_data.get_stock_data(fx_symbol, days=2)
                                if not fx_data.empty:
                                    forex_data[fx_symbol] = fx_data
                            except:
                                pass

                        if forex_data:
                            currency_converter.update_rates(forex_data)

                        # Get prices and convert to USD
                        for symbol in alive_symbols:
                            try:
                                recent_data = await self.market_data.get_stock_data(symbol, days=2)
                                if not recent_data.empty:
                                    raw_price = float(recent_data['Close'].iloc[-1])

                                    # Detect currency from symbol
                                    if symbol.endswith('.T'):
                                        currency = 'JPY'
                                    elif symbol.endswith('.HK'):
                                        currency = 'HKD'
                                    elif symbol.endswith(('.SS', '.SZ')):
                                        currency = 'CNY'
                                    elif symbol.endswith('.DE'):
                                        currency = 'EUR'
                                    elif symbol.endswith('.L'):
                                        currency = 'GBP'
                                    else:
                                        currency = 'USD'

                                    # Convert to USD if needed
                                    if currency != 'USD':
                                        usd_price = currency_converter.convert_to_usd(raw_price, currency)
                                        self.logger.debug(f"Converted {symbol}: {raw_price:.2f} {currency} → ${usd_price:.2f} USD")
                                        current_prices[symbol] = usd_price
                                    else:
                                        current_prices[symbol] = raw_price
                            except Exception as e:
                                self.logger.warning(f"Failed to get current price for {symbol}: {e}")
                    except Exception as e:
                        self.logger.error(f"Error fetching current prices: {e}")

                # Process bets with updated current prices
                for bet_data in bet_rows:
                    if bet_data["status"] == "alive":
                        # Update current price if we have fresh market data
                        if bet_data["symbol"] in current_prices:
                            bet_data["current_price"] = current_prices[bet_data["symbol"]]

                        # Calculate P&L with current price
                        bet_data["pnl"] = (bet_data["current_price"] - bet_data["entry_price"]) * bet_data["shares"]
                        bet_data["pnl_pct"] = ((bet_data["current_price"] - bet_data["entry_price"]) / bet_data["entry_price"]) * 100
                        alive_bets.append(bet_data)
                    else:
                        bet_data["pnl"] = bet_data["realized_pnl"] or 0.0
                        if bet_data["exit_price"]:
                            bet_data["pnl_pct"] = ((bet_data["exit_price"] - bet_data["entry_price"]) / bet_data["entry_price"]) * 100
                        else:
                            bet_data["pnl_pct"] = 0.0
                        closed_bets.append(bet_data)

                return alive_bets, closed_bets

            finally:
                conn.close()

        except Exception as e:
            st.error(f"Error getting all bets data: {e}")
            self.logger.error(f"All bets data error: {e}")
            return [], []
    
    async def settle_bet_manually(self, bet_id: str, symbol: str, current_price: float) -> bool:
        """Manually settle a bet at current market price"""
        try:
            if not self.portfolio_manager:
                st.error("Portfolio manager not available")
                return False

            await self.portfolio_manager.initialize()

            # Find the bet and close it manually
            if bet_id in self.portfolio_manager.active_bets:
                # Update the current price first
                bet = self.portfolio_manager.active_bets[bet_id]
                bet.current_price = current_price
                bet.current_value = bet.shares * current_price
                bet.unrealized_pnl = bet.current_value - bet.amount

                # Close the bet manually (user initiated)
                await self.portfolio_manager.close_bet(bet_id, "manual_settlement", current_price)

                # Only refresh portfolio and bets, NOT opportunities (no polling/retraining)
                await self.refresh_portfolio_only()
                return True
            else:
                st.error(f"Bet {bet_id} not found in active bets")
                return False

        except Exception as e:
            st.error(f"Failed to settle bet: {e}")
            return False

    async def place_bet(self, symbol: str, probability: float, current_price: float,
                        algorithms_dict: dict = None, currency: str = 'USD',
                        opportunity: Dict = None) -> bool:
        """
        Place a bet through the real portfolio manager.

        `opportunity` carries the row the user actually clicked, and passing it is not
        optional under volatility-scaled barriers: the barriers travel on it, and
        PortfolioManager.place_bet refuses to fall back to config defaults rather
        than silently size a different bet than the one on screen. Without it, every
        placement path raised.

        Args:
            symbol: Asset symbol
            probability: Predicted probability
            current_price: Current price in USD (already converted)
            algorithms_dict: Algorithm predictions
            currency: Original currency (for reference, price is already in USD)
            opportunity: The full opportunity row, source of the barriers
        """
        try:
            if not self.portfolio_manager:
                st.error("Portfolio manager not available")
                return False

            opportunity = opportunity or {}

            # Create prediction dict for portfolio manager with full algorithm data
            prediction = {
                'symbol': symbol,
                'probability': probability,
                'current_price': current_price,  # Already in USD
                'currency': currency,  # Original currency for reference
                # This asset's own barriers, so sizing and the stored win/loss prices
                # match the bet that was displayed and agreed to.
                'win_threshold': opportunity.get('win_threshold'),
                'loss_threshold': opportunity.get('loss_threshold'),
                'barrier_mode': opportunity.get('barrier_mode', 'volatility'),
                'sigma_pct': opportunity.get('sigma_pct'),
                'asset_type': opportunity.get('asset_type'),
                'algorithms': []
            }

            # If we have individual algorithm predictions, include them
            if algorithms_dict:
                for algo_name, algo_prob in algorithms_dict.items():
                    # Map display names back to internal names
                    internal_name_map = {
                        'lstm': 'lstm',
                        'random_forest': 'rf',
                        'sma': 'sma',
                        'rsi': 'rsi',
                        'regression': 'regression',
                        'svm': 'svm'
                    }

                    internal_name = internal_name_map.get(algo_name, algo_name)

                    if algo_prob is not None:
                        prediction['algorithms'].append({
                            'algorithm': internal_name,
                            'probability': algo_prob,
                            'error': None
                        })
                    else:
                        # Algorithm failed or unavailable
                        prediction['algorithms'].append({
                            'algorithm': internal_name,
                            'probability': None,
                            'error': 'Not available'
                        })
            else:
                # Fallback for manual placement without algorithm data
                prediction['algorithms'] = [{'algorithm': 'dashboard_manual', 'probability': probability, 'error': None}]

            # Place the bet using portfolio manager
            await self.portfolio_manager.initialize()
            bet_id = await self.portfolio_manager.place_bet(prediction)

            st.success(f"SUCCESS: Bet placed successfully!")
            st.info(f"Bet ID: {bet_id}")
            st.info(f"Symbol: {symbol} at ${current_price:.2f}")
            st.info(f"Probability: {probability:.1f}%")

            # Only refresh portfolio and bets data, NOT opportunities (no polling/retraining)
            await self.refresh_portfolio_only()

            return True

        except Exception as e:
            st.error(f"ERROR: Failed to place bet: {e}")
            self.logger.error(f"Bet placement error: {e}")
            return False

    async def check_and_settle_bets(self, fetch_prices: bool = True) -> List[Dict]:
        """
        Check open positions and settle any that qualify.

        Runs on **every** entry into the app and on every refresh, because a position
        that should have closed distorts equity, the win-rate statistics and the
        correlation haircut applied to new bets.

        `fetch_prices=False` is the page-load path: it settles using the last stored
        marks, with no network calls. That still catches **every time-barrier exit** --
        the time barrier needs no quote at all -- plus any price barrier already
        visible in the last known marks. Live price barriers need fresh data, which is
        what the "Refresh prices" button is for.
        """
        try:
            from src.trading.settlement import settle_positions

            if not self.portfolio_manager:
                self.logger.warning("Portfolio manager not available for settlement")
                return []

            max_hold_days = int(self.config.get('trading', {}).get('max_hold_days', 30))

            if fetch_prices and self.bet_monitor:
                # Full check: the monitor fetches live prices, marks to market, then
                # settles through the same shared rule.
                await self.bet_monitor._monitor_and_settle_positions()
                self.logger.info("Bet settlement check completed (live prices)")
                return []

            # Offline check: last known marks only.
            await self.portfolio_manager.initialize()
            alive = await self.portfolio_manager.get_alive_bets()
            if not alive:
                return []

            prices = {bet.symbol: bet.current_price for bet in alive
                      if bet.current_price}
            settled = await settle_positions(self.portfolio_manager, prices,
                                             max_hold_days)
            if settled:
                self.logger.info(f"Settled {len(settled)} position(s) from stored "
                                 f"marks, no API calls")
            return settled

        except Exception as e:
            self.logger.error(f"Error during bet settlement check: {e}")
            # Don't raise - settlement failures shouldn't break the dashboard
            return []

    async def _update_active_bet_prices(self):
        """Update current prices for active bets to ensure accurate unrealized P&L calculations"""
        try:
            if not self.portfolio_manager or not self.market_data:
                return

            # Get current active bets from portfolio manager
            active_bet_symbols = list(self.portfolio_manager.active_bets.keys())
            if not active_bet_symbols:
                return

            self.logger.info(f"Updating prices for {len(active_bet_symbols)} active bets...")

            # Initialize currency converter
            currency_converter = CurrencyConverter()

            # Get forex data to update exchange rates
            forex_symbols = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X', 'USDCNY=X',
                           'AUDUSD=X', 'USDCAD=X', 'NZDUSD=X']
            forex_data = {}
            for fx_symbol in forex_symbols:
                try:
                    fx_data = await self.market_data.get_stock_data(fx_symbol, days=2)
                    if not fx_data.empty:
                        forex_data[fx_symbol] = fx_data
                except:
                    pass

            if forex_data:
                currency_converter.update_rates(forex_data)

            # Update prices for each active bet
            for bet_id, bet in self.portfolio_manager.active_bets.items():
                try:
                    # Get current market price (raw)
                    recent_data = await self.market_data.get_stock_data(bet.symbol, days=2)
                    if not recent_data.empty:
                        raw_price = float(recent_data['Close'].iloc[-1])

                        # Convert to USD if needed
                        bet_currency = bet.currency if hasattr(bet, 'currency') else 'USD'
                        if bet_currency != 'USD':
                            if not currency_converter.has_rate(bet_currency):
                                self.logger.warning(f"No exchange rate available for {bet_currency}, using entry price")
                                current_price_usd = bet.entry_price  # Fallback to entry price if no rate
                            else:
                                current_price_usd = currency_converter.convert_to_usd(raw_price, bet_currency)
                                self.logger.info(f"Converted {bet.symbol}: {raw_price:.2f} {bet_currency} → ${current_price_usd:.2f} USD")
                        else:
                            current_price_usd = raw_price

                        # Sanity check: USD price should be reasonable (not in thousands for most stocks)
                        if current_price_usd > 10000:
                            self.logger.error(f"CURRENCY BUG DETECTED: {bet.symbol} price ${current_price_usd:.2f} seems wrong! Raw: {raw_price}, Currency: {bet_currency}")
                            current_price_usd = bet.entry_price  # Use entry price as safe fallback

                        # Update bet's current price and calculated values (ALL IN USD)
                        bet.current_price = current_price_usd
                        bet.current_value = bet.shares * current_price_usd
                        bet.unrealized_pnl = bet.current_value - bet.amount

                        self.logger.info(f"Updated {bet.symbol}: ${bet.current_price:.2f} USD (P&L: ${bet.unrealized_pnl:+.2f})")
                    else:
                        self.logger.warning(f"No recent data available for {bet.symbol}")

                except Exception as e:
                    self.logger.error(f"Error updating price for {bet.symbol}: {e}")

            self.logger.info("Active bet price update completed")

        except Exception as e:
            self.logger.error(f"Error during active bet price update: {e}")

    async def refresh_portfolio_only(self):
        """Lightweight refresh - only update portfolio and bets, NO polling or ML retraining"""
        try:
            self.logger.info("Refreshing portfolio and bets (no polling/retraining)...")

            # Check and settle bets that hit thresholds
            await self.check_and_settle_bets()

            # Update portfolio data
            st.session_state.portfolio_data = await self.get_portfolio_data()

            # Update active bets
            st.session_state.active_bets_data = await self.get_active_bets_data()

            # Update all bets data
            st.session_state.all_bets_data = await self.get_all_bets_data()

            # Do NOT update opportunities_data - keep existing cached predictions

            self.logger.info("Portfolio and bets refreshed (opportunities cached)")

        except Exception as e:
            self.logger.error(f"Portfolio refresh failed: {e}", exc_info=True)
            st.error(f"Failed to refresh portfolio: {e}")

    @staticmethod
    def _hurdle_label(rake_pct: Optional[float]) -> Optional[str]:
        """
        Plain reading of how hard a bet is to win, from the fee's share of the pot.

        The number itself ("2.40 pts of required edge") is meaningless to most
        people. What it means is: how much better than luck you have to be. For
        calibration, professional systematic funds typically run on a couple of
        points of edge, so 5+ is not a realistic target for anyone.
        """
        if rake_pct is None:
            return None
        if rake_pct < 1.5:
            return "Low"
        if rake_pct < 3.0:
            return "Moderate"
        if rake_pct < 5.0:
            return "High"
        return "Very high"

    @staticmethod
    def _opportunity_status(opp: Dict) -> str:
        """
        One word on why a row is or is not actionable.

        Most rows are not, and silently showing a Kelly of 0.0% with no reason is
        unhelpful. The usual causes are an uneconomic barrier (fees eat the target)
        or a Kelly size below the minimum bet.
        """
        if opp.get('is_favorable'):
            return "Tradeable"
        if opp.get('tradeable') is False:
            return "Fees too big"
        warning = (opp.get('risk_warning') or '').lower()
        if 'below minimum' in warning:
            return "Too small"
        if 'negative expected value' in warning:
            return "Below break-even"
        return "No bet"

    def _required_edge_for(self, prediction: Dict) -> Optional[float]:
        """Points of edge this bet's barriers demand over the geometry: 2c/(w+l)."""
        win, loss = prediction.get('win_threshold'), prediction.get('loss_threshold')
        if not win or not loss:
            return None
        from src.trading.barriers import BarrierPolicy
        return BarrierPolicy(self.config).required_edge(win, loss)

    def _expected_days(self, prediction: Dict) -> Optional[float]:
        """
        Expected trading days for price to travel to the profit target.

        A barrier d daily-sigmas away is reached in about d^2 bars.
        """
        win = prediction.get('win_threshold')
        sigma = prediction.get('sigma_pct')
        if not win or not sigma:
            return None
        from src.trading.barriers import BarrierPolicy
        horizon = BarrierPolicy(self.config).horizon_days
        daily_sigma = sigma / (horizon ** 0.5)
        return (win / daily_sigma) ** 2 if daily_sigma > 0 else None

    async def load_cached_state(self):
        """
        Populate the dashboard from the DATABASE only. No network calls.

        Used on page load and after any rerun, so browsing the dashboard costs no API
        quota. Prices shown are the last marked-to-market values; the header states
        how old they are.
        """
        try:
            self.logger.info("Loading cached state from database (no API calls)...")

            # Settle BEFORE reading anything: an overdue position would otherwise
            # inflate equity, distort the win rate and tighten the correlation
            # haircut on new bets. Uses stored marks only, so this costs no API quota
            # and still catches every time-barrier exit.
            settled = await self.check_and_settle_bets(fetch_prices=False)
            if settled:
                st.session_state.settled_on_load = settled

            st.session_state.portfolio_data = await self.get_portfolio_data(
                fetch_prices=False)
            st.session_state.active_bets_data = await self.get_active_bets_data(
                fetch_prices=False)
            st.session_state.all_bets_data = await self.get_all_bets_data(
                fetch_prices=False)
            st.session_state.opportunities_data = await self.get_cached_opportunities()
            st.session_state.prices_as_of = await self.get_price_data_asof()

            self.logger.info(
                f"Cached state loaded: {len(st.session_state.opportunities_data)} "
                f"stored opportunities, prices as of {st.session_state.prices_as_of}")

        except Exception as e:
            self.logger.error(f"Failed to load cached state: {e}", exc_info=True)
            st.error(f"Could not load saved data: {e}")

    async def get_price_data_asof(self):
        """Timestamp of the newest cached price bar, for the staleness indicator."""
        try:
            conn = sqlite3.connect(self.portfolio_manager.db_path)
            try:
                row = conn.execute(
                    "SELECT MAX(substr(timestamp,1,10)) FROM price_data").fetchone()
                return row[0] if row else None
            finally:
                conn.close()
        except Exception as e:
            self.logger.debug(f"Could not read price data age: {e}")
            return None

    async def get_cached_opportunities(self) -> List[Dict]:
        """
        Rebuild the last computed ranking from the stored `predictions` rows.

        Lets the page show the most recent opportunity list without re-running the
        models or hitting any API. Barriers and break-even are recomputed locally
        from cached bars, which is free.
        """
        try:
            conn = sqlite3.connect(self.portfolio_manager.db_path)
            try:
                latest = conn.execute(
                    "SELECT MAX(timestamp) FROM predictions").fetchone()[0]
                if not latest:
                    return []

                # Predictions from one scoring cycle share a timestamp prefix.
                rows = conn.execute("""
                    SELECT symbol, algorithm, probability
                    FROM predictions
                    WHERE substr(timestamp, 1, 13) = substr(?, 1, 13)
                """, (latest,)).fetchall()
            finally:
                conn.close()

            if not rows:
                return []

            by_symbol: Dict[str, Dict[str, float]] = {}
            for symbol, algorithm, probability in rows:
                if probability is None:
                    continue
                by_symbol.setdefault(symbol, {})[algorithm] = float(probability)

            # Asset class per symbol, so the table is not mislabelled.
            conn = sqlite3.connect(self.portfolio_manager.db_path)
            try:
                asset_types = dict(conn.execute(
                    "SELECT symbol, asset_type FROM assets").fetchall())
            finally:
                conn.close()

            opportunities = []
            for symbol, algos in by_symbol.items():
                if not algos:
                    continue
                score = sum(algos.values()) / len(algos)
                # Must carry the SAME keys as the live refresh path in
                # get_opportunities_data(), or the renderers KeyError. Kelly fields
                # are filled in by _annotate_opportunity_economics once the
                # per-asset barriers are known.
                opportunities.append({
                    'symbol': symbol,
                    'asset_type': asset_types.get(symbol, 'stock'),
                    'currency': 'USD',
                    'raw_score': score,
                    'final_probability': score,
                    'algorithms': {
                        'lstm': algos.get('lstm'),
                        'random_forest': algos.get('rf'),
                        'sma': algos.get('sma'),
                        'rsi': algos.get('rsi'),
                        'regression': algos.get('regression'),
                        'svm': algos.get('svm'),
                    },
                    'n_algorithms': len(algos),
                    'computed_at': latest,
                    'prediction_confidence': 0.0,
                    'failed_algorithms': [a for a in ('lstm', 'rf', 'sma', 'rsi',
                                                      'regression', 'svm')
                                          if a not in algos],
                    # Placeholders; overwritten during annotation.
                    'current_price': 0.0,
                    'kelly_fraction': 0.0,
                    'recommended_amount': 0.0,
                    'is_favorable': False,
                    'risk_warning': '',
                })

            opportunities.sort(key=lambda o: o['raw_score'], reverse=True)
            top = opportunities[:60]
            await self._annotate_opportunity_economics(top)
            return top

        except Exception as e:
            self.logger.error(f"Could not rebuild cached opportunities: {e}")
            return []

    async def _annotate_opportunity_economics(self, opportunities: List[Dict]):
        """
        Attach the exactly-knowable economics to each opportunity.

        These do NOT depend on the model being calibrated: barrier distances,
        break-even and required edge follow from the asset's own volatility and the
        fee, and are computed from cached bars. They are the part of this screen a
        user can actually act on.
        """
        if not opportunities:
            return

        from src.kelly.calculator import KellyCalculator
        from src.trading.barriers import BarrierPolicy
        policy = BarrierPolicy(self.config)
        kelly = KellyCalculator(self.config)

        # Sizing context: real cash and the real open-position count, so the
        # correlation haircut is the one that would actually apply.
        try:
            available_capital = await self.portfolio_manager.get_cash_balance()
            open_positions = await self.portfolio_manager._count_alive_bets()
        except Exception as e:
            self.logger.debug(f"Could not read sizing context: {e}")
            available_capital, open_positions = 0.0, 0

        conn = sqlite3.connect(self.portfolio_manager.db_path)
        try:
            for opp in opportunities:
                try:
                    frame = pd.read_sql_query("""
                        SELECT p.timestamp, p.open, p.high, p.low, p.close, p.volume
                        FROM price_data p JOIN assets a ON a.asset_id = p.asset_id
                        WHERE a.symbol = ?
                        ORDER BY p.timestamp DESC LIMIT 120
                    """, conn, params=(opp['symbol'],))
                    if frame.empty:
                        continue

                    frame = frame.iloc[::-1].rename(columns={
                        'open': 'Open', 'high': 'High', 'low': 'Low',
                        'close': 'Close', 'volume': 'Volume'})

                    opp.setdefault('current_price', float(frame['Close'].iloc[-1]))

                    spec = policy.for_series(frame)
                    if spec is None:
                        continue

                    opp['win_threshold'] = spec.win_pct
                    opp['loss_threshold'] = spec.loss_pct
                    opp['sigma_pct'] = spec.sigma_pct
                    opp['break_even_pct'] = policy.break_even(
                        spec.win_pct, spec.loss_pct) * 100.0
                    opp['required_edge_pct'] = policy.required_edge(
                        spec.win_pct, spec.loss_pct)
                    opp['tradeable'] = policy.is_economic(spec)
                    opp['reject_reason'] = policy.rejection_reason(spec)
                    # Expected bars to touch a barrier d sigma_daily away is ~d^2.
                    if spec.sigma_pct:
                        daily_sigma = spec.sigma_pct / (policy.horizon_days ** 0.5)
                        if daily_sigma > 0:
                            opp['expected_days_to_win'] = (spec.win_pct / daily_sigma) ** 2
                            opp['expected_days_to_loss'] = (spec.loss_pct / daily_sigma) ** 2

                    # Size on THIS asset's barriers, not on config defaults, and
                    # against the book that is actually open.
                    if opp['tradeable'] and available_capital > 0:
                        recommendation = kelly.calculate_bet_size(
                            probability=opp['final_probability'],
                            current_price=opp['current_price'],
                            available_capital=available_capital,
                            win_threshold=spec.win_pct,
                            loss_threshold=spec.loss_pct,
                            concurrent_positions=open_positions + 1,
                        )
                        opp['is_favorable'] = recommendation.is_favorable
                        opp['kelly_fraction'] = (recommendation.fraction_of_capital
                                                 if recommendation.is_favorable else 0.0)
                        opp['recommended_amount'] = (recommendation.recommended_amount
                                                     if recommendation.is_favorable else 0.0)
                        opp['risk_warning'] = recommendation.risk_warning or ''
                    else:
                        opp['is_favorable'] = False
                        opp['kelly_fraction'] = 0.0
                        opp['recommended_amount'] = 0.0
                        opp['risk_warning'] = opp.get('reject_reason') or ''
                except Exception as e:
                    self.logger.debug(f"Could not annotate {opp['symbol']}: {e}")
        finally:
            conn.close()

    async def refresh_data(self):
        """Refresh all dashboard data with real-time progress"""
        try:
            self.logger.info("Starting dashboard data refresh...")

            # Create progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Step 0: Check and settle bets that hit thresholds (FIRST PRIORITY)
            status_text.text("🎯 Checking for bet settlements...")
            progress_bar.progress(10)
            await self.check_and_settle_bets()

            # Step 1: Portfolio data
            status_text.text("Loading portfolio data...")
            progress_bar.progress(25)
            st.session_state.portfolio_data = await self.get_portfolio_data()

            # Step 2: Active bets
            status_text.text("Loading active bets...")
            progress_bar.progress(45)
            st.session_state.active_bets_data = await self.get_active_bets_data()

            # Step 3: All bets data
            status_text.text("Loading bet history...")
            progress_bar.progress(65)
            st.session_state.all_bets_data = await self.get_all_bets_data()

            # Step 4: Real ML predictions (this is the slow part)
            status_text.text("🧠 Generating REAL ML predictions (this may take a few minutes)...")
            progress_bar.progress(85)

            # Generate opportunities with optional log display
            if st.session_state.get('show_ml_logs', False):
                # Create a real-time log display
                log_container = st.container()
                with log_container:
                    st.write("**ML Processing Log:**")
                    log_placeholder = st.empty()

                    # Capture logs during prediction generation
                    import io
                    import logging

                    # Create string buffer to capture logs
                    log_capture_string = io.StringIO()
                    ch = logging.StreamHandler(log_capture_string)
                    ch.setLevel(logging.INFO)
                    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                    ch.setFormatter(formatter)

                    # Add handler to capture logs
                    self.logger.addHandler(ch)

                    try:
                        st.session_state.opportunities_data = await self.get_opportunities_data()

                        # Display captured logs
                        log_contents = log_capture_string.getvalue()
                        if log_contents:
                            log_placeholder.code(log_contents, language="text")

                    finally:
                        # Remove the handler
                        self.logger.removeHandler(ch)
            else:
                # Generate without log display
                st.session_state.opportunities_data = await self.get_opportunities_data()

            # Final step
            status_text.text("✅ Dashboard refresh completed!")
            progress_bar.progress(100)
            st.session_state.last_update = datetime.now()

            # Clear progress indicators after a moment
            import time
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()

            self.logger.info(f"Dashboard refresh completed at {st.session_state.last_update}")

        except Exception as e:
            self.logger.error(f"Dashboard refresh failed: {e}", exc_info=True)
            st.error(f"❌ Refresh failed: {e}")
            # Set empty data on failure
            st.session_state.opportunities_data = []
            st.session_state.portfolio_data = {}
            st.session_state.active_bets_data = []
            st.session_state.all_bets_data = ([], [])
    
    def render_header(self):
        """Render dashboard header"""
        st.title("Kelly Criterion Trading Dashboard")
        
        # Data source indicator - should always be real data now
        if REAL_DATA_AVAILABLE and self.portfolio_manager and self.market_data:
            st.success("🟢 CONNECTED: Real trading data with live ML predictions")
            if hasattr(self.portfolio_manager, 'initial_capital'):
                st.caption(f"Portfolio initialized with ${self.portfolio_manager.initial_capital:,.2f}")
        else:
            st.error("🔴 NOT CONNECTED: Dashboard initialization failed!")
            st.error("This should not happen if setup is correct. Check logs.")
        
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            prices_asof = st.session_state.get('prices_as_of')
            if st.session_state.last_update:
                st.caption(f"Prices fetched this session at "
                           f"{st.session_state.last_update.strftime('%H:%M:%S')}")
            elif prices_asof:
                st.caption(f"Showing saved data. Newest cached price bar: {prices_asof}")
            else:
                st.caption("Showing saved data. No cached prices found yet.")

        with col2:
            # The ONLY thing on this page that calls an external API.
            if st.button("Refresh prices", type="primary",
                         help="Fetches live market data and re-scores the universe. "
                              "This is the only action that calls an external API, so "
                              "nothing else on this page consumes rate limit."):
                st.session_state.needs_refresh = True
                st.rerun()

        with col3:
            st.caption("Auto-refresh is off by design — it was exhausting the "
                       "market-data rate limit. Browsing and switching tabs never "
                       "re-fetches. Settlement is still checked every time.")

        # A position closing is never silent, even when it happened during a
        # background settlement check on page load.
        settled = st.session_state.pop('settled_on_load', None)
        if settled:
            lines = " · ".join(
                f"**{s['symbol']}** ({s['exit_type'].replace('_', ' ')})"
                for s in settled)
            st.warning(f"Settled {len(settled)} overdue position(s) on load: {lines}")
    
    def render_portfolio_overview(self):
        """Render portfolio overview section"""
        st.header("Portfolio Overview")
        
        portfolio = st.session_state.portfolio_data
        if not portfolio:
            st.warning("No portfolio data available")
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Capital",
                f"${portfolio.get('total_capital', 0):,.2f}",
                delta=f"${portfolio.get('total_return', 0):+.2f}"
            )
        
        with col2:
            st.metric(
                "Available Capital",
                f"${portfolio.get('available_capital', 0):,.2f}"
            )
        
        with col3:
            st.metric(
                "Active Bets Value",
                f"${portfolio.get('active_bets_value', 0):,.2f}"
            )
        
        with col4:
            st.metric(
                "Win Rate",
                f"{portfolio.get('win_rate', 0)*100:.1f}%",
                delta=f"{portfolio.get('won_bets', 0)}/{portfolio.get('completed_bets', 0)} bets"
            )
    
    def render_trading_controls(self):
        """Render trading controls section"""
        st.header("Trading Controls")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            auto_mode = st.toggle(
                "Automated Mode",
                value=st.session_state.auto_mode,
                help="Enable automated bet placement based on threshold"
            )
            if auto_mode != st.session_state.auto_mode:
                st.session_state.auto_mode = auto_mode
        
        with col2:
            threshold = st.slider(
                "Auto Threshold (%)",
                min_value=50.0,
                max_value=90.0,
                value=st.session_state.auto_threshold,
                step=1.0,
                help="Minimum probability for automated betting"
            )
            if threshold != st.session_state.auto_threshold:
                st.session_state.auto_threshold = threshold
        
        with col3:
            if st.button("🛑 Emergency Stop", type="secondary"):
                st.warning("Emergency stop activated - all automated trading paused")
    
    def render_proposals(self, opportunities: List[Dict]):
        """
        The shortlist: the few bets actually worth putting in front of someone.

        Ranked by how little forecasting skill they demand (the fee's share of the
        pot), NOT by ensemble score. The score has been measured as mildly
        anti-predictive out of sample -- z = -3.11 across 321 assets -- so ordering by
        it descending would rank by the thing that precedes worse outcomes.

        Crucially this section is allowed to come back empty, with reasons. A screen
        that always finds ten opportunities because it has ten slots is a screen that
        tells you nothing.
        """
        from src.trading.selection import select_proposals

        st.subheader("Today's proposals")

        held = {bet['symbol'] for bet in (st.session_state.get('active_bets_data') or [])}

        control_left, control_right = st.columns([1, 3])
        with control_left:
            max_hurdle = st.slider(
                "Max skill required (pts)", min_value=1.0, max_value=8.0,
                value=float(self.config.get('trading', {}).get('max_proposal_edge_pct', 3.0)),
                step=0.1, key="proposal_max_hurdle",
                help="The rake ceiling. A bet needing more than ~3 points of edge "
                     "asks for better forecasting than professional funds deliver.")

        shortlist = select_proposals(
            opportunities,
            limit=int(self.config.get('trading', {}).get('top_n_display', 10)),
            max_required_edge_pct=max_hurdle,
            held_symbols=held,
        )

        if shortlist.is_empty:
            st.info(f"**Nothing worth proposing right now.** "
                    f"{shortlist.considered} assets considered.")
            if shortlist.rejected:
                st.caption("Why: " + " · ".join(
                    f"**{count}** {reason}"
                    for reason, count in sorted(shortlist.rejected.items(),
                                                key=lambda kv: -kv[1])))
                if any('below minimum' in r for r in shortlist.rejected):
                    st.caption("Assets rejected for size are usually the *cheapest* "
                               "tables — Kelly shrinks as barriers widen. Adding cash, "
                               "or lowering `min_bet_amount`, would let them through.")
            return

        with control_right:
            st.caption(f"**{len(shortlist.proposals)} of {shortlist.considered}** "
                       f"assets clear the bar. Ranked by **lowest skill required** — "
                       f"the score is shown but does not set the order, because out of "
                       f"sample it has been mildly *anti*-predictive.")

        rows = []
        for opp in shortlist.proposals:
            rows.append({
                '#': opp['proposal_rank'],
                'Symbol': opp['symbol'],
                'Type': opp.get('asset_type', 'stock').upper(),
                'Need right': f"{opp['break_even_pct']:.0f} in 100",
                'Fees take': opp['required_edge_pct'],
                'Hurdle': opp['hurdle'],
                'Risk / Reward': f"-${opp['loss_threshold']:.2f} / +${opp['win_threshold']:.2f}",
                'Score': opp.get('raw_score'),
                'Suggested size': opp['recommended_amount'],
            })

        st.caption("Select a row to review and place the bet.")
        event = st.dataframe(
            pd.DataFrame(rows), use_container_width=True, hide_index=True,
            on_select="rerun", selection_mode="single-row",
            key="proposal_table",
            column_config={
                '#': st.column_config.NumberColumn('#', width='small'),
                'Fees take': st.column_config.NumberColumn(
                    'Fees take', format="%.1f%% of pot",
                    help="The broker's cut as a share of everything at stake. This is "
                         "what sets the order: a lower rake is a cheaper table, needing "
                         "fewer extra correct calls per hundred to break even."),
                'Need right': st.column_config.TextColumn(
                    'Need right', width='small',
                    help="Of every 100 such bets, how many must win to break even "
                         "after fees. Luck alone already wins about 40."),
                'Score': st.column_config.NumberColumn(
                    'Score', format="%.1f",
                    help="Ensemble ranking signal, for information only. It does not "
                         "set the order here."),
                'Suggested size': st.column_config.NumberColumn(
                    'Suggested size', format="$%.0f",
                    help="Fractional Kelly on this asset's own barriers, after the "
                         "correlation haircut for positions already open."),
            })

        st.caption("These are the **cheapest tables**, not predicted winners. No "
                   "signal in this system has yet demonstrated an edge, so treat the "
                   "suggested sizes as research positions.")

        if event.selection and event.selection.rows:
            chosen = shortlist.proposals[event.selection.rows[0]]
            self.show_bet_placement_dialog(chosen, held)

    def render_opportunities(self):
        """Render comprehensive opportunities section with algorithm breakdowns"""
        st.header("Market Opportunities")

        opportunities = st.session_state.opportunities_data
        if not opportunities:
            st.info("No opportunities available. Click 'Refresh prices' to update.")
            return

        # The shortlist comes first: it is the answer to "what should I do?", where
        # the full table below is the evidence behind it.
        self.render_proposals(opportunities)

        st.divider()
        with st.expander(f"Full ranking — all {len(opportunities)} assets scored",
                         expanded=False):
            self._render_full_opportunity_list(opportunities)

    def _render_full_opportunity_list(self, opportunities: List[Dict]):
        """The complete scored universe, with filters. Evidence, not recommendation."""
        # Filter controls
        col1, col2, col3 = st.columns(3)

        with col1:
            asset_filter = st.selectbox(
                "Asset Type:",
                options=["all", "stock", "crypto", "commodity", "forex"],
                key="asset_type_filter"
            )

        with col2:
            min_probability = st.slider(
                "Min Probability:",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                step=1.0,
                key="min_prob_filter"
            )

        with col3:
            show_algorithm_details = st.toggle(
                "Show Algorithm Details",
                value=False,
                key="show_algo_details"
            )

        # Add ML processing logs toggle
        col1, col2 = st.columns(2)
        with col1:
            show_ml_logs = st.toggle(
                "Show ML Processing Logs",
                value=False,
                key="show_ml_logs"
            )
        with col2:
            if st.button("🔄 Force ML Retrain", help="Force retrain all ML models"):
                if st.session_state.get('force_retrain_confirm', False):
                    st.session_state.force_retrain = True
                    st.session_state.force_retrain_confirm = False
                    st.rerun()
                else:
                    st.session_state.force_retrain_confirm = True
                    st.warning("Click again to confirm ML model retraining")

        if st.session_state.get('force_retrain_confirm', False):
            st.caption("⚠️ This will retrain all models from scratch (may take several minutes)")

        # Filter opportunities
        filtered_opps = opportunities
        if asset_filter != "all":
            filtered_opps = [opp for opp in filtered_opps if opp.get('asset_type', 'stock') == asset_filter]

        filtered_opps = [opp for opp in filtered_opps if opp['final_probability'] >= min_probability]

        st.write(f"Showing {len(filtered_opps)} of {len(opportunities)} assets")

        # Create expandable sections for better organization
        if show_algorithm_details:
            self.render_detailed_opportunities(filtered_opps)
        else:
            self.render_compact_opportunities(filtered_opps)

    @st.dialog("Place Bet Confirmation")
    def show_bet_placement_dialog(self, opp: Dict, active_bet_symbols: set):
        """Show bet placement confirmation dialog"""
        symbol = opp['symbol']

        # Check if there's already an active bet
        if symbol in active_bet_symbols:
            st.warning(f"⚠️ You already have an active bet for {symbol}")
            st.write("Are you sure you want to place another bet on this asset?")

        # Display asset information
        st.subheader(f"{symbol} - {opp.get('asset_type', 'stock').upper()}")

        stake = opp.get('recommended_amount') or 0.0
        win_pct = opp.get('win_threshold')
        loss_pct = opp.get('loss_threshold')
        break_even = opp.get('break_even_pct')
        rake = opp.get('required_edge_pct')

        # State the bet in money before stating it in probability. This is the part
        # the user is actually agreeing to, and it is exact.
        if win_pct and loss_pct and stake:
            price = opp['current_price']
            fee = self.config.get('trading', {}).get('trading_fee_percentage', 0.25) / 100
            st.markdown(
                f"**You are staking ${stake:,.2f} on {symbol} at ${price:,.2f}.**\n\n"
                f"- Sell automatically at **${price * (1 + win_pct / 100):,.2f}** "
                f"(+{win_pct:.2f}%) for a gain of about "
                f"**${stake * win_pct / 100:,.2f}**\n"
                f"- Or at **${price * (1 - loss_pct / 100):,.2f}** "
                f"(−{loss_pct:.2f}%) for a loss of about "
                f"**${stake * loss_pct / 100:,.2f}**\n"
                f"- Either way you pay about **${stake * 2 * fee:,.2f}** in fees\n"
                f"- If neither level is reached, it closes at market after "
                f"**{self.config.get('trading', {}).get('max_hold_days', 30)} days**"
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Suggested stake", f"${stake:,.0f}",
                      help=f"{opp['kelly_fraction'] * 100:.2f}% of cash — fractional "
                           f"Kelly on this asset's barriers, after the correlation "
                           f"haircut for positions already open.")
        with col2:
            if break_even is not None:
                st.metric("Must win", f"{break_even:.0f} in 100",
                          help=f"Break-even after fees is {break_even:.2f}%. Pure luck "
                               f"already wins about 40 in 100 from the barrier shape.")
        with col3:
            if rake is not None:
                st.metric("Fees take", f"{rake:.1f}% of pot",
                          help="The broker's cut as a share of everything at stake. "
                               "This is the only hurdle that is known exactly.")

        if not opp.get('is_calibrated', self._calibration_is_fitted()):
            st.info(f"The score of **{opp['final_probability']:.1f}** is a ranking "
                    f"signal, not a probability — it has not been calibrated, and out "
                    f"of sample it has been mildly *anti*-predictive. Nothing here "
                    f"forecasts that this bet will win; the numbers above describe "
                    f"what it costs and what it pays.")

        # Show algorithm predictions
        st.write("**Individual Algorithm Predictions:**")

        algo_cols = st.columns(3)
        algorithms = [
            ('LSTM', opp['algorithms']['lstm']),
            ('Random Forest', opp['algorithms']['random_forest']),
            ('SMA', opp['algorithms']['sma']),
            ('RSI', opp['algorithms']['rsi']),
            ('Regression', opp['algorithms']['regression']),
            ('SVM', opp['algorithms']['svm'])
        ]

        for idx, (algo_name, algo_prob) in enumerate(algorithms):
            with algo_cols[idx % 3]:
                if algo_prob is not None:
                    st.write(f"**{algo_name}:** {algo_prob:.1f}%")
                else:
                    st.write(f"**{algo_name}:** N/A")

        # Show failed algorithms if any
        if opp.get('failed_algorithms'):
            with st.expander("⚠️ Algorithm Warnings", expanded=False):
                for failure in opp['failed_algorithms']:
                    st.caption(failure)

        # Show risk warning if present
        if opp.get('risk_warning'):
            st.warning(f"⚠️ {opp['risk_warning']}")

        st.divider()

        # Action buttons
        if not opp['is_favorable']:
            st.error("This bet is not favorable according to Kelly Criterion (probability too low or odds unfavorable)")
            if st.button("Close", type="secondary", use_container_width=True):
                st.rerun()
        else:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Cancel", type="secondary", use_container_width=True):
                    st.rerun()
            with col2:
                if st.button("Confirm & Place Bet", type="primary", use_container_width=True):
                    success = asyncio.run(self.place_bet(
                        symbol=symbol,
                        probability=opp['final_probability'],
                        current_price=opp['current_price'],
                        algorithms_dict=opp['algorithms'],  # Pass the full algorithms dictionary
                        currency=opp.get('currency', 'USD'),  # Pass original currency
                        opportunity=opp,  # Carries this asset's own barriers
                    ))
                    if success:
                        st.success(f"✓ Bet placed successfully for {symbol}!")
                        time.sleep(1)  # Brief pause to show success message
                        st.rerun()
                    else:
                        st.error("Failed to place bet. Please check logs.")

    def render_compact_opportunities(self, opportunities: List[Dict]):
        """Render opportunities in compact table format with clickable rows"""
        if not opportunities:
            st.info("No opportunities match the current filters.")
            return

        # Get list of symbols with active bets for highlighting
        active_bet_symbols = set()
        if hasattr(st.session_state, 'active_bets_data') and st.session_state.active_bets_data:
            active_bet_symbols = {bet['symbol'] for bet in st.session_state.active_bets_data}

        # Create DataFrame with sortable columns
        df_data = []
        for idx, opp in enumerate(opportunities):
            # Add red color indicator for assets with active bets
            symbol_display = opp['symbol']
            if opp['symbol'] in active_bet_symbols:
                symbol_display = f"🔴 {opp['symbol']}"

            # Handle None values for display - show as "N/A"
            def format_algo_value(val):
                return val if val is not None else None  # Keep None for proper sorting

            # Per-asset economics. These do NOT depend on the model being calibrated
            # -- they follow from the asset's own volatility and the fee -- so they
            # are the columns a user can actually act on today.
            score = opp.get('raw_score', opp.get('final_probability'))
            break_even = opp.get('break_even_pct')
            margin = (score - break_even) if (score is not None and break_even) else None

            win_pct = opp.get('win_threshold')
            loss_pct = opp.get('loss_threshold')
            rake = opp.get('required_edge_pct')

            df_data.append({
                'Symbol': symbol_display,
                'Type': opp.get('asset_type', 'stock').upper(),
                'Price': opp['current_price'],
                'Score': score,
                # Plain-language versions of the same arithmetic. "43.28% break-even"
                # and "3.26 pts of required edge" are quant units that mean nothing
                # to someone deciding whether to place a bet; these say the same
                # thing as a hit rate, a rake, and a risk/reward in money.
                'Need right': (f"{break_even:.0f} in 100" if break_even else None),
                'Fees take': rake,
                'Hurdle': self._hurdle_label(rake),
                'Risk / Reward': (f"-${loss_pct:.2f} / +${win_pct:.2f}"
                                  if win_pct and loss_pct else None),
                'Status': self._opportunity_status(opp),
                'Kelly %': opp['kelly_fraction']*100,
                'Recommended': opp['recommended_amount'] if opp['is_favorable'] else 0,
                'LSTM': format_algo_value(opp['algorithms']['lstm']),
                'RF': format_algo_value(opp['algorithms']['random_forest']),
                'SMA': format_algo_value(opp['algorithms']['sma']),
                'RSI': format_algo_value(opp['algorithms']['rsi']),
                'Regression': format_algo_value(opp['algorithms']['regression']),
                'SVM': format_algo_value(opp['algorithms']['svm']),
            })

        if df_data:
            df = pd.DataFrame(df_data)

            # Display instructions
            st.write("**Market Opportunities Table**")
            st.caption("🔴 Red indicator = Active bet already placed | Click column headers to sort | Select row below to place bet")

            # Use st.dataframe with column configuration and selection enabled
            event = st.dataframe(
                df,
                use_container_width=True,
                column_config={
                    "Symbol": st.column_config.TextColumn("Symbol", width="medium"),
                    "Type": st.column_config.TextColumn("Type", width="small"),
                    "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                    "Score": st.column_config.NumberColumn(
                        "Score", format="%.1f",
                        help="Ensemble ranking signal. Compares assets against each "
                             "other; its level is not a probability unless the "
                             "Reliability tab says calibration is fitted."),
                    "Need right": st.column_config.TextColumn(
                        "Need right", width="small",
                        help="How many of every 100 such bets must win just to break "
                             "even, after fees. Pure luck already wins about 40 in "
                             "100 from the barrier shape alone, so the gap above 40 "
                             "is what your judgement has to supply. Exact — does not "
                             "depend on the model."),
                    "Fees take": st.column_config.NumberColumn(
                        "Fees take", format="%.1f%% of pot",
                        help="The broker's cut, as a share of everything at stake in "
                             "the bet. A 0.50% round trip on a bet that only swings "
                             "10% means fees eat 5% of the pot before you play — like "
                             "a casino rake. Lower is a cheaper table."),
                    "Hurdle": st.column_config.TextColumn(
                        "Hurdle", width="small",
                        help="Plain reading of the rake. Low means the fees barely "
                             "matter and modest skill can win. Very high means you "
                             "would need professional-grade forecasting just to "
                             "break even."),
                    "Risk / Reward": st.column_config.TextColumn(
                        "Risk / Reward", width="small",
                        help="Per $100 staked: what you lose if the stop hits, versus "
                             "what you make if the target hits. Both scale with the "
                             "asset's own volatility."),
                    "Status": st.column_config.TextColumn(
                        "Status", width="small",
                        help="Why this row is or is not actionable. 'Too small' means "
                             "Kelly sized it below the minimum bet, which happens on "
                             "wide-barrier assets once the correlation haircut for "
                             "the open book is applied."),
                    "Kelly %": st.column_config.NumberColumn("Kelly %", format="%.1f%%"),
                    "Recommended": st.column_config.NumberColumn("Recommended", format="$%.0f"),
                    "LSTM": st.column_config.NumberColumn("LSTM", format="%.1f%%"),
                    "RF": st.column_config.NumberColumn("RF", format="%.1f%%"),
                    "SMA": st.column_config.NumberColumn("SMA", format="%.1f%%"),
                    "RSI": st.column_config.NumberColumn("RSI", format="%.1f%%"),
                    "Regression": st.column_config.NumberColumn("Regression", format="%.1f%%"),
                    "SVM": st.column_config.NumberColumn("SVM", format="%.1f%%"),
                },
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row"
            )

            # Handle row selection
            if event.selection and event.selection.rows:
                selected_idx = event.selection.rows[0]
                selected_opp = opportunities[selected_idx]

                # Show bet placement dialog
                self.show_bet_placement_dialog(selected_opp, active_bet_symbols)


    def render_detailed_opportunities(self, opportunities: List[Dict]):
        """Render opportunities with detailed algorithm breakdowns"""
        if not opportunities:
            st.info("No opportunities match the current filters.")
            return

        for idx, opp in enumerate(opportunities):
            # Create expandable section for each asset
            asset_type = opp.get('asset_type', 'stock')
            if asset_type == 'crypto':
                asset_badge = "🪙"
            elif asset_type == 'commodity':
                asset_badge = "🥇"  # Gold medal for commodities
            elif asset_type == 'forex':
                asset_badge = "💱"  # Currency exchange
            else:
                asset_badge = "📈"  # Stock/default

            prob_color = "🟢" if opp['final_probability'] > 70 else "🟡" if opp['final_probability'] > 60 else "🔴"

            # Get full asset name for display
            display_name = get_display_name(opp['symbol'], opp.get('asset_type', 'stock'))

            with st.expander(
                f"{asset_badge} {display_name} - {prob_color} {opp['final_probability']:.1f}% "
                f"(${opp['current_price']:,.2f})"
            ):
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Asset Information:**")
                    full_name = get_asset_name(opp['symbol'], opp.get('asset_type', 'stock'))
                    st.write(f"• Symbol: {opp['symbol']}")
                    st.write(f"• Name: {full_name}")
                    st.write(f"• Type: {opp.get('asset_type', 'stock').title()}")
                    st.write(f"• Current Price: ${opp['current_price']:,.2f}")
                    st.write(f"• Final Probability: {opp['final_probability']:.1f}%")

                    st.write("**Kelly Recommendation:**")
                    if opp['is_favorable']:
                        st.write(f"• Kelly Fraction: {opp['kelly_fraction']*100:.1f}%")
                        st.write(f"• Recommended Amount: ${opp['recommended_amount']:,.0f}")
                        st.write("• Status: 🟢 Favorable")
                    else:
                        st.write("• Status: 🔴 Not Favorable")

                with col2:
                    st.write("**Algorithm Predictions:**")

                    # LSTM
                    lstm_prob = opp['algorithms']['lstm']
                    lstm_failed = any("LSTM" in fail for fail in opp.get('failed_algorithms', []))
                    if lstm_failed:
                        lstm_color = "⚠️"
                        lstm_status = " (Fallback - model error)"
                    else:
                        lstm_color = "🟢" if lstm_prob > 60 else "🟡" if lstm_prob > 55 else "🔴"
                        lstm_status = ""
                    st.write(f"• LSTM: {lstm_color} {lstm_prob:.1f}%{lstm_status}")

                    # Random Forest
                    rf_prob = opp['algorithms']['random_forest']
                    rf_failed = any("Random Forest" in fail for fail in opp.get('failed_algorithms', []))
                    if rf_failed:
                        rf_color = "⚠️"
                        rf_status = " (Fallback - model error)"
                    else:
                        rf_color = "🟢" if rf_prob > 60 else "🟡" if rf_prob > 55 else "🔴"
                        rf_status = ""
                    st.write(f"• Random Forest: {rf_color} {rf_prob:.1f}%{rf_status}")

                    # SMA
                    sma_prob = opp['algorithms']['sma']
                    sma_failed = any("SMA" in fail for fail in opp.get('failed_algorithms', []))
                    if sma_failed:
                        sma_color = "⚠️"
                        sma_status = " (Fallback - model error)"
                    else:
                        sma_color = "🟢" if sma_prob > 60 else "🟡" if sma_prob > 55 else "🔴"
                        sma_status = ""
                    st.write(f"• SMA: {sma_color} {sma_prob:.1f}%{sma_status}")

                    # RSI
                    rsi_prob = opp['algorithms']['rsi']
                    rsi_failed = any("RSI" in fail for fail in opp.get('failed_algorithms', []))
                    if rsi_failed:
                        rsi_color = "⚠️"
                        rsi_status = " (Fallback - model error)"
                    else:
                        rsi_color = "🟢" if rsi_prob > 60 else "🟡" if rsi_prob > 55 else "🔴"
                        rsi_status = ""
                    st.write(f"• RSI: {rsi_color} {rsi_prob:.1f}%{rsi_status}")

                    # Regression
                    regression_prob = opp['algorithms']['regression']
                    regression_failed = any("Regression" in fail for fail in opp.get('failed_algorithms', []))
                    if regression_failed:
                        regression_color = "⚠️"
                        regression_status = " (Fallback - model error)"
                    else:
                        regression_color = "🟢" if regression_prob > 60 else "🟡" if regression_prob > 55 else "🔴"
                        regression_status = ""
                    st.write(f"• Regression: {regression_color} {regression_prob:.1f}%{regression_status}")

                    # SVM
                    svm_prob = opp['algorithms']['svm']
                    svm_failed = any("SVM" in fail for fail in opp.get('failed_algorithms', []))
                    if svm_failed:
                        svm_color = "⚠️"
                        svm_status = " (Fallback - model error)"
                    else:
                        svm_color = "🟢" if svm_prob > 60 else "🟡" if svm_prob > 55 else "🔴"
                        svm_status = ""
                    st.write(f"• SVM: {svm_color} {svm_prob:.1f}%{svm_status}")

                    # Show algorithm failures if any
                    if opp.get('failed_algorithms'):
                        st.write("**Algorithm Issues:**")
                        for failure in opp['failed_algorithms']:
                            st.write(f"⚠️ {failure}")

                    st.write("**Ensemble Calculation:**")
                    st.write(f"• LSTM × 25%: {lstm_prob:.1f}% × 0.25 = {lstm_prob * 0.25:.1f}%")
                    st.write(f"• Random Forest × 20%: {rf_prob:.1f}% × 0.20 = {rf_prob * 0.20:.1f}%")
                    st.write(f"• SMA × 15%: {sma_prob:.1f}% × 0.15 = {sma_prob * 0.15:.1f}%")
                    st.write(f"• RSI × 15%: {rsi_prob:.1f}% × 0.15 = {rsi_prob * 0.15:.1f}%")
                    st.write(f"• Regression × 15%: {regression_prob:.1f}% × 0.15 = {regression_prob * 0.15:.1f}%")
                    st.write(f"• SVM × 10%: {svm_prob:.1f}% × 0.10 = {svm_prob * 0.10:.1f}%")
                    st.write(f"• **Final: {opp['final_probability']:.1f}%**")

                # Bet placement button
                if opp['is_favorable']:
                    if st.button(f"Place Bet for {opp['symbol']}", key=f"detailed_bet_{idx}"):
                        success = asyncio.run(self.place_bet(
                            symbol=opp['symbol'],
                            probability=opp['final_probability'],
                            current_price=opp['current_price'],
                            algorithms_dict=opp.get('algorithms'),
                            currency=opp.get('currency', 'USD'),
                            opportunity=opp,  # Carries this asset's own barriers
                        ))
                        if success:
                            st.rerun()
    
    def render_active_bets(self):
        """Render active bets monitoring section"""
        st.header("Active Bets")
        
        bets = st.session_state.active_bets_data
        if not bets:
            st.info("No active bets")
            return
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        total_invested = sum(bet['amount'] for bet in bets)
        total_pnl = sum(bet['pnl'] for bet in bets)
        avg_pnl_pct = sum(bet['pnl_pct'] for bet in bets) / len(bets) if bets else 0
        
        with col1:
            st.metric("Total Invested", f"${total_invested:,.0f}")
        with col2:
            st.metric("Unrealized P&L", f"${total_pnl:+,.2f}")
        with col3:
            st.metric("Avg Return", f"{avg_pnl_pct:+.1f}%")
        
        # Detailed table with fees calculation
        df = pd.DataFrame(bets)

        # Calculate fees (entry + exit fees from config)
        fee_percentage = self.config.get('trading', {}).get('trading_fee_percentage', 0.25)  # Default 0.25% if not in config

        # Format the dataframe for display
        display_df = df.copy()
        # Add asset names
        display_df['Asset'] = display_df.apply(
            lambda row: get_display_name(row['symbol'], row.get('asset_type', 'stock')),
            axis=1
        )
        display_df['Entry'] = display_df['entry_price'].apply(lambda x: f"${x:,.2f}")
        display_df['Current'] = display_df['current_price'].apply(lambda x: f"${x:,.2f}")
        display_df['Amount'] = display_df['amount'].apply(lambda x: f"${x:,.0f}")

        # Calculate fees paid (entry fee + estimated exit fee)
        display_df['Fees Paid'] = display_df.apply(
            lambda row: f"${(row['amount'] * fee_percentage / 100) + ((row['shares'] * row['current_price']) * fee_percentage / 100):.2f}",
            axis=1
        )

        display_df['P&L'] = display_df.apply(lambda x: f"${x['pnl']:+.2f} ({x['pnl_pct']:+.1f}%)", axis=1)

        # Calculate P&L minus fees
        display_df['P&L - Fees'] = display_df.apply(
            lambda row: f"${(row['pnl'] - (row['amount'] * fee_percentage / 100) - ((row['shares'] * row['current_price']) * fee_percentage / 100)):+.2f}",
            axis=1
        )

        # Show the barrier as both a price and a percentage. Under volatility-scaled
        # barriers the percentage differs per asset, so the price alone tells you
        # nothing about how far the bet has to travel.
        display_df['Win Target'] = display_df.apply(
            lambda x: f"${x['win_price']:,.2f} (+{x['win_pct']:.2f}%)", axis=1)
        display_df['Stop Loss'] = display_df.apply(
            lambda x: f"${x['loss_price']:,.2f} (-{x['loss_pct']:.2f}%)", axis=1)
        display_df['Duration'] = display_df['entry_time'].apply(
            lambda x: str(datetime.now() - x).split('.')[0] if isinstance(x, datetime) else "N/A"
        )

        # Add algorithm and probability if available
        if 'algorithm_used' in df.columns:
            display_df['Algorithm'] = df['algorithm_used']
        if 'probability_when_placed' in df.columns:
            display_df['Entry Prob'] = df['probability_when_placed'].apply(lambda x: f"{x:.1f}%")

        # Show the table with new columns
        columns_to_show = ['Asset', 'Entry', 'Current', 'Amount', 'Fees Paid', 'P&L', 'P&L - Fees', 'Win Target', 'Stop Loss', 'Duration']
        if 'Algorithm' in display_df.columns:
            columns_to_show.append('Algorithm')
        if 'Entry Prob' in display_df.columns:
            columns_to_show.append('Entry Prob')
            
        # Display each bet with settle button
        for idx, bet in df.iterrows():
            with st.expander(f"📈 {get_display_name(bet['symbol'], bet.get('asset_type', 'stock'))} - {bet['pnl']:+.2f} P&L"):
                col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])

                with col1:
                    st.write(f"**Amount:** ${bet['amount']:,.0f}")
                    st.write(f"**Entry:** ${bet['entry_price']:,.2f} → **Current:** ${bet['current_price']:,.2f}")
                    fee_paid = (bet['amount'] * fee_percentage / 100) + ((bet['shares'] * bet['current_price']) * fee_percentage / 100)
                    st.write(f"**Fees Paid:** ${fee_paid:.2f}")

                with col2:
                    st.write(f"**P&L:** ${bet['pnl']:+.2f}")
                    pnl_after_fees = bet['pnl'] - fee_paid
                    st.write(f"**P&L - Fees:** ${pnl_after_fees:+.2f}")

                with col3:
                    st.write(f"**Win Target:** ${bet['win_price']:,.2f} "
                             f"(+{bet['win_pct']:.2f}%)")
                    st.write(f"**Stop Loss:** ${bet['loss_price']:,.2f} "
                             f"(-{bet['loss_pct']:.2f}%)")

                with col4:
                    duration = str(datetime.now() - bet['entry_time']).split('.')[0] if isinstance(bet['entry_time'], datetime) else "N/A"
                    st.write(f"**Duration:** {duration}")
                    if 'algorithm_used' in bet and bet['algorithm_used']:
                        st.write(f"**Algorithm:** {bet['algorithm_used']}")

                with col5:
                    # Settle Bet button
                    if st.button(f"🔄 Settle Bet", key=f"settle_{bet['bet_id']}", help="Manually settle this bet at current market price"):
                        if asyncio.run(self.settle_bet_manually(bet['bet_id'], bet['symbol'], bet['current_price'])):
                            st.success(f"Bet {bet['symbol']} settled successfully!")
                            st.rerun()

        # Summary row with totals
        total_fees = sum((bet['amount'] * fee_percentage / 100) + ((bet['shares'] * bet['current_price']) * fee_percentage / 100) for _, bet in df.iterrows())
        total_pnl_after_fees = total_pnl - total_fees

        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Invested", f"${sum(bet['amount'] for _, bet in df.iterrows()):,.2f}")
        with col2:
            st.metric("Total Fees", f"${total_fees:.2f}")
        with col3:
            st.metric("Total P&L", f"${total_pnl:+,.2f}")
        with col4:
            st.metric("Total P&L - Fees", f"${total_pnl_after_fees:+,.2f}")
    
    def render_performance_charts(self):
        """Render performance visualization section"""
        st.header("Performance Analysis")

        # Get real portfolio history
        portfolio_history = asyncio.run(self.get_portfolio_history())

        if not portfolio_history:
            st.info("No portfolio history data available yet")
            return

        # Create DataFrame from history
        df = pd.DataFrame(portfolio_history)

        # Create the chart
        fig = go.Figure()

        # Portfolio value line
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['total_capital'],
            mode='lines',
            name='Total Portfolio Value',
            line=dict(color='#1f77b4', width=2)
        ))

        # Cash balance line
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['cash_balance'],
            mode='lines',
            name='Cash Balance',
            line=dict(color='#2ca02c', width=2, dash='dash')
        ))

        # Active bets value line
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['active_bets_value'],
            mode='lines',
            name='Active Bets Value',
            line=dict(color='#ff7f0e', width=2, dash='dot')
        ))

        # Add initial capital reference line
        if self.portfolio_manager:
            initial_capital = self.portfolio_manager.initial_capital
            fig.add_hline(
                y=initial_capital,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Initial Capital: ${initial_capital:,.0f}"
            )

        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            height=400,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        st.plotly_chart(fig, width='stretch')

        # Portfolio statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            # Calculate total return from initial capital to current value
            if self.portfolio_manager:
                initial_capital = self.portfolio_manager.initial_capital
                current_portfolio = st.session_state.portfolio_data.get('total_capital', initial_capital)
                total_return = current_portfolio - initial_capital
                total_return_pct = (total_return / initial_capital) * 100 if initial_capital > 0 else 0
            else:
                total_return = 0
                total_return_pct = 0
            st.metric("Total Return", f"${total_return:+,.2f}", f"{total_return_pct:+.2f}%")

        with col2:
            if len(df) > 1:
                max_value = df['total_capital'].max()
                current_value = df['total_capital'].iloc[-1]
                max_drawdown = ((current_value - max_value) / max_value) * 100 if max_value > 0 else 0
                st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
            else:
                st.metric("Max Drawdown", "0.00%")

        with col3:
            total_realized = st.session_state.portfolio_data.get('total_pnl', 0)
            st.metric("Total P&L", f"${total_realized:+,.2f}")

    def render_bets_tab(self):
        """Render the bets tab with alive and closed bets"""
        st.header("All Bets")

        alive_bets, closed_bets = st.session_state.all_bets_data

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Active Bets", len(alive_bets))
        with col2:
            st.metric("Closed Bets", len(closed_bets))
        with col3:
            total_alive_amount = sum(bet['amount'] for bet in alive_bets)
            st.metric("Active Amount", f"${total_alive_amount:,.0f}")
        with col4:
            total_closed_pnl = sum(bet['pnl'] for bet in closed_bets)
            st.metric("Total Realized P&L", f"${total_closed_pnl:+,.2f}")

        st.divider()

        # Active Bets section (on top)
        self.render_alive_bets_section(alive_bets)

        st.divider()

        # Closed Bets section (underneath Active Bets)
        self.render_closed_bets_section(closed_bets)

    def render_alive_bets_section(self, alive_bets: List[Dict]):
        """Render alive bets section"""
        st.subheader(f"Active Bets ({len(alive_bets)})")

        if not alive_bets:
            st.info("No active bets")
            return

        # Calculate fees and P&L summary
        fee_percentage = self.config.get('trading', {}).get('trading_fee_percentage', 0.25)  # Default 0.25% if not in config
        total_unrealized_pnl = sum(bet['pnl'] for bet in alive_bets)

        # Calculate total fees for active bets (entry + estimated exit fees)
        total_fees = sum((bet['amount'] * fee_percentage / 100) + ((bet['shares'] * bet['current_price']) * fee_percentage / 100)
                        for bet in alive_bets)
        total_pnl_after_fees = total_unrealized_pnl - total_fees

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Unrealized P&L", f"${total_unrealized_pnl:+,.2f}")
        with col2:
            st.metric("Total Fees (Est.)", f"${total_fees:.2f}")
        with col3:
            st.metric("P&L After Fees", f"${total_pnl_after_fees:+,.2f}")

        # Create dataframe for alive bets
        df = pd.DataFrame(alive_bets)

        # Format display
        display_df = df.copy()
        display_df['Asset'] = display_df.apply(
            lambda row: get_display_name(row['symbol'], row.get('asset_type', 'stock')),
            axis=1
        )
        display_df['Entry Price'] = display_df['entry_price'].apply(lambda x: f"${x:,.2f}")
        display_df['Current Price'] = display_df['current_price'].apply(lambda x: f"${x:,.2f}")
        display_df['Amount'] = display_df['amount'].apply(lambda x: f"${x:,.0f}")

        # Calculate fees for each bet (entry + estimated exit)
        display_df['Fees (Est.)'] = display_df.apply(
            lambda row: f"${(row['amount'] * fee_percentage / 100) + (row.get('current_value', row['amount']) * fee_percentage / 100):.2f}",
            axis=1
        )

        display_df['P&L'] = display_df.apply(lambda x: f"${x['pnl']:+.2f} ({x['pnl_pct']:+.1f}%)", axis=1)

        # Calculate P&L minus fees
        display_df['P&L - Fees'] = display_df.apply(
            lambda row: f"${(row['pnl'] - (row['amount'] * fee_percentage / 100) - (row.get('current_value', row['amount']) * fee_percentage / 100)):+.2f}",
            axis=1
        )

        display_df['Entry Date'] = display_df['entry_time'].apply(lambda x: x.strftime('%m/%d %H:%M'))
        display_df['Algorithm'] = display_df['algorithm_used']
        display_df['Entry Prob'] = display_df['probability_when_placed'].apply(lambda x: f"{x:.1f}%")

        # Add individual algorithm prediction columns
        display_df['Algo Predictions'] = display_df['algorithm_predictions'].apply(
            lambda preds: ', '.join([f"{algo.split()[0]}: {prob:.1f}%" for algo, prob in preds.items()]) if preds else "N/A"
        )

        # Show table with fees columns and algorithm predictions
        columns_to_show = ['Asset', 'Entry Price', 'Current Price', 'Amount', 'Fees (Est.)', 'P&L', 'P&L - Fees', 'Entry Date', 'Algorithm', 'Entry Prob', 'Algo Predictions']
        st.dataframe(
            display_df[columns_to_show],
            width='stretch',
            height=400
        )

    def render_closed_bets_section(self, closed_bets: List[Dict]):
        """Render closed bets section"""
        # Filter out cancelled bets (they have no meaningful data)
        closed_bets = [bet for bet in closed_bets if bet.get('status') != 'cancelled']

        st.subheader(f"Closed Bets ({len(closed_bets)})")

        if not closed_bets:
            st.info("No closed bets")
            return

        # Show filter options
        status_filter = st.selectbox(
            "Filter by result:",
            options=["all", "won", "lost"],
            key="closed_bets_filter",
            help="Filter bets by outcome"
        )

        # Filter bets
        filtered_bets = closed_bets
        if status_filter != "all":
            filtered_bets = [bet for bet in closed_bets if bet['status'] == status_filter]

        if not filtered_bets:
            st.info(f"No {status_filter} bets found")
            return

        # Calculate fees and P&L summary for filtered bets
        fee_percentage = self.config.get('trading', {}).get('trading_fee_percentage', 0.25)  # Default 0.25% if not in config
        total_realized_pnl = sum(bet['pnl'] for bet in filtered_bets)

        # Calculate total fees for closed bets (entry + exit fees)
        # Handle cancelled bets with no exit_price
        total_fees = sum(
            (bet['amount'] * fee_percentage / 100) +
            ((bet['shares'] * (bet.get('exit_price') or bet.get('current_price', 0))) * fee_percentage / 100)
            for bet in filtered_bets
            if bet.get('exit_price') or bet.get('current_price')  # Skip bets with no price data
        )
        total_pnl_after_fees = total_realized_pnl - total_fees

        won_bets = [bet for bet in filtered_bets if bet['status'] == 'won']
        lost_bets = [bet for bet in filtered_bets if bet['status'] == 'lost']

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Realized P&L", f"${total_realized_pnl:+,.2f}")
        with col2:
            st.metric("Total Fees Paid", f"${total_fees:.2f}")
        with col3:
            st.metric("P&L After Fees", f"${total_pnl_after_fees:+,.2f}")
        with col4:
            if won_bets and lost_bets:
                win_rate = len(won_bets) / len(filtered_bets) * 100
                st.metric("Win Rate", f"{win_rate:.1f}%", delta=f"{len(won_bets)}W/{len(lost_bets)}L")

        # Create dataframe for closed bets
        df = pd.DataFrame(filtered_bets)

        # Format display
        display_df = df.copy()
        display_df['Asset'] = display_df.apply(
            lambda row: get_display_name(row['symbol'], row.get('asset_type', 'stock')),
            axis=1
        )
        display_df['Entry Price'] = display_df['entry_price'].apply(lambda x: f"${x:,.2f}")
        display_df['Exit Price'] = display_df['exit_price'].apply(lambda x: f"${x:,.2f}" if x else "N/A")
        display_df['Amount'] = display_df['amount'].apply(lambda x: f"${x:,.0f}")

        # Calculate fees for each bet (entry + exit)
        display_df['Fees Paid'] = display_df.apply(
            lambda row: f"${(row['amount'] * fee_percentage / 100) + (row.get('exit_value', row['amount']) * fee_percentage / 100):.2f}",
            axis=1
        )

        display_df['P&L'] = display_df.apply(lambda x: f"${x['pnl']:+.2f} ({x['pnl_pct']:+.1f}%)", axis=1)

        # Calculate P&L minus fees
        display_df['P&L - Fees'] = display_df.apply(
            lambda row: f"${(row['pnl'] - (row['amount'] * fee_percentage / 100) - (row.get('exit_value', row['amount']) * fee_percentage / 100)):+.2f}",
            axis=1
        )

        display_df['Entry Date'] = display_df['entry_time'].apply(lambda x: x.strftime('%m/%d %H:%M'))
        display_df['Exit Date'] = display_df['exit_time'].apply(lambda x: x.strftime('%m/%d %H:%M') if x else "N/A")
        display_df['Status'] = display_df['status'].apply(lambda x: x.upper())
        display_df['Duration'] = display_df.apply(lambda x:
            str(x['exit_time'] - x['entry_time']).split('.')[0] if x['exit_time'] else "N/A", axis=1)

        # Add individual algorithm prediction columns
        display_df['Algo Predictions'] = display_df['algorithm_predictions'].apply(
            lambda preds: ', '.join([f"{algo.split()[0]}: {prob:.1f}%" for algo, prob in preds.items()]) if preds else "N/A"
        )

        # Show table with fees columns and algorithm predictions
        columns_to_show = ['Asset', 'Entry Price', 'Exit Price', 'Amount', 'Fees Paid', 'P&L', 'P&L - Fees', 'Entry Date', 'Exit Date', 'Status', 'Duration', 'Algo Predictions']
        st.dataframe(
            display_df[columns_to_show],
            width='stretch',
            height=400
        )

        # Portfolio value change summary for closed bets
        if filtered_bets:
            st.subheader("Portfolio Impact")

            # Calculate win/loss stats
            won_bets = [bet for bet in filtered_bets if bet['status'] == 'won']
            lost_bets = [bet for bet in filtered_bets if bet['status'] == 'lost']

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                win_rate = len(won_bets) / len(filtered_bets) * 100 if filtered_bets else 0
                st.metric("Win Rate", f"{win_rate:.1f}%")

            with col2:
                total_won = sum(bet['pnl'] for bet in won_bets)
                st.metric("Total Won", f"${total_won:+,.2f}")

            with col3:
                total_lost = sum(bet['pnl'] for bet in lost_bets)
                st.metric("Total Lost", f"${total_lost:+,.2f}")

            with col4:
                net_pnl = sum(bet['pnl'] for bet in filtered_bets)
                st.metric("Net P&L", f"${net_pnl:+,.2f}")

    async def get_portfolio_history(self) -> List[Dict]:
        """Get portfolio value history from database"""
        try:
            if not self.portfolio_manager:
                return []

            import sqlite3
            conn = sqlite3.connect(self.portfolio_manager.db_path)
            cursor = conn.cursor()

            try:
                cursor.execute('''
                SELECT timestamp, total_capital, cash_balance, active_bets_value, realized_pnl, notes
                FROM portfolio_history
                ORDER BY timestamp ASC
                ''')

                rows = cursor.fetchall()
                history = []

                for row in rows:
                    history.append({
                        'timestamp': datetime.fromisoformat(row[0]),
                        'total_capital': float(row[1]),
                        'cash_balance': float(row[2]),
                        'active_bets_value': float(row[3]),
                        'realized_pnl': float(row[4]),
                        'notes': row[5] or ""
                    })

                return history

            finally:
                conn.close()

        except Exception as e:
            self.logger.error(f"Error getting portfolio history: {e}")
            return []

    def render_market_data_tab(self):
        """Render market data visualization tab"""
        st.header("Market Data Visualization")

        try:
            # Get available assets from database
            available_assets = asyncio.run(self.get_available_assets())

            if not available_assets:
                st.warning("No market data available in database")
                return

            # Asset selector with full names
            asset_options = [
                f"{get_display_name(asset['symbol'], asset['asset_type'])} ({asset['asset_type']})"
                for asset in available_assets
            ]
            selected_asset_display = st.selectbox(
                "Select an asset to visualize:",
                asset_options,
                index=0 if asset_options else None
            )

            if selected_asset_display:
                # Extract symbol from display string (format: "SYMBOL - Full Name (type)" or "SYMBOL (type)")
                if ' - ' in selected_asset_display:
                    selected_symbol = selected_asset_display.split(' - ')[0]
                else:
                    selected_symbol = selected_asset_display.split(' (')[0]
                selected_asset = next(asset for asset in available_assets if asset['symbol'] == selected_symbol)

                # Time period selector
                time_periods = {
                    "Last 30 days": 30,
                    "Last 60 days": 60,
                    "Last 90 days": 90,
                    "All available data": None
                }

                selected_period = st.selectbox(
                    "Select time period:",
                    list(time_periods.keys()),
                    index=1  # Default to 60 days
                )

                days_back = time_periods[selected_period]

                # Get and display price data
                price_data = asyncio.run(self.get_asset_price_data(selected_asset['asset_id'], days_back))

                if price_data:
                    self.render_price_chart(selected_asset, price_data)
                    self.render_price_statistics(selected_asset, price_data)
                else:
                    st.warning(f"No price data available for {selected_symbol}")

        except Exception as e:
            st.error(f"Error loading market data: {e}")
            self.logger.error(f"Market data tab error: {e}")

    async def get_available_assets(self) -> List[Dict]:
        """Get list of assets with available price data"""
        try:
            if not self.market_data:
                return []

            await self.market_data.initialize()

            # Query database for assets with price data
            import sqlite3
            conn = sqlite3.connect(self.market_data.db_path)
            cursor = conn.cursor()

            cursor.execute('''
            SELECT DISTINCT a.asset_id, a.symbol, a.asset_type, COUNT(p.price_id) as record_count
            FROM assets a
            INNER JOIN price_data p ON a.asset_id = p.asset_id
            GROUP BY a.asset_id, a.symbol, a.asset_type
            ORDER BY a.symbol
            ''')

            results = cursor.fetchall()
            conn.close()

            return [
                {
                    'asset_id': row[0],
                    'symbol': row[1],
                    'asset_type': row[2],
                    'record_count': row[3]
                }
                for row in results
            ]

        except Exception as e:
            self.logger.error(f"Error getting available assets: {e}")
            return []

    async def get_asset_price_data(self, asset_id: int, days_back: int = None) -> List[Dict]:
        """Get price data for specific asset"""
        try:
            if not self.market_data:
                return []

            await self.market_data.initialize()

            import sqlite3
            from datetime import datetime, timedelta

            conn = sqlite3.connect(self.market_data.db_path)
            cursor = conn.cursor()

            # Build query with optional date filter
            if days_back:
                cutoff_date = datetime.now() - timedelta(days=days_back)
                cursor.execute('''
                SELECT timestamp, open, high, low, close, volume
                FROM price_data
                WHERE asset_id = ? AND timestamp >= ?
                ORDER BY timestamp
                ''', (asset_id, cutoff_date.isoformat()))
            else:
                cursor.execute('''
                SELECT timestamp, open, high, low, close, volume
                FROM price_data
                WHERE asset_id = ?
                ORDER BY timestamp
                ''', (asset_id,))

            results = cursor.fetchall()
            conn.close()

            return [
                {
                    'timestamp': row[0],
                    'open': row[1],
                    'high': row[2],
                    'low': row[3],
                    'close': row[4],
                    'volume': row[5]
                }
                for row in results
            ]

        except Exception as e:
            self.logger.error(f"Error getting price data: {e}")
            return []

    def render_price_chart(self, asset: Dict, price_data: List[Dict]):
        """Render price chart for asset"""
        try:
            import pandas as pd
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            # Convert to DataFrame
            df = pd.DataFrame(price_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Create subplots for price and volume
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                subplot_titles=(f"{asset['symbol']} Price", "Volume"),
                vertical_spacing=0.03
            )

            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=df['timestamp'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name="Price"
                ),
                row=1, col=1
            )

            # Volume bar chart
            fig.add_trace(
                go.Bar(
                    x=df['timestamp'],
                    y=df['volume'],
                    name="Volume",
                    marker_color='rgba(158,202,225,0.6)'
                ),
                row=2, col=1
            )

            # Update layout
            fig.update_layout(
                title=f"{asset['symbol']} ({asset['asset_type'].upper()}) - Historical Data",
                xaxis_rangeslider_visible=False,
                height=600,
                showlegend=False
            )

            # Update axes
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)

            st.plotly_chart(fig, width='stretch')

        except Exception as e:
            st.error(f"Error creating price chart: {e}")
            self.logger.error(f"Price chart error: {e}")

    def render_price_statistics(self, asset: Dict, price_data: List[Dict]):
        """Render price statistics summary"""
        try:
            import pandas as pd

            df = pd.DataFrame(price_data)

            if df.empty:
                return

            # Ensure timestamp is properly converted to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Calculate statistics
            current_price = df['close'].iloc[-1]
            start_price = df['close'].iloc[0]
            price_change = current_price - start_price
            price_change_pct = (price_change / start_price) * 100

            high_price = df['high'].max()
            low_price = df['low'].min()
            avg_volume = df['volume'].mean()

            # Calculate volatility (standard deviation of daily returns)
            df['daily_return'] = df['close'].pct_change()
            volatility = df['daily_return'].std() * (252 ** 0.5) * 100  # Annualized

            st.subheader("Price Statistics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Current Price", f"${current_price:.2f}")
                st.metric("Period High", f"${high_price:.2f}")

            with col2:
                st.metric(
                    "Period Change",
                    f"${price_change:+.2f}",
                    f"{price_change_pct:+.1f}%"
                )
                st.metric("Period Low", f"${low_price:.2f}")

            with col3:
                st.metric("Avg Volume", f"{avg_volume:,.0f}")
                st.metric("Data Points", f"{len(df):,}")

            with col4:
                st.metric("Volatility (Annual)", f"{volatility:.1f}%")
                # Format dates safely
                start_date = df['timestamp'].iloc[0].strftime('%Y-%m-%d')
                end_date = df['timestamp'].iloc[-1].strftime('%Y-%m-%d')
                st.metric("Date Range", f"{start_date} to {end_date}")

        except Exception as e:
            st.error(f"Error calculating statistics: {e}")
            self.logger.error(f"Statistics error: {e}")

    def _kelly_reference(self):
        """Kelly calculator instance, for the break-even figure."""
        from src.kelly.calculator import KellyCalculator
        return KellyCalculator(self.config)

    def _calibration_is_fitted(self) -> bool:
        """Whether a fitted ensemble calibration exists on disk."""
        try:
            from src.prediction.calibration import EnsembleCalibrator
            return EnsembleCalibrator().is_fitted
        except Exception:
            return False

    def render_calibration_status(self):
        """
        Show what the user needs to decide: the bar this payoff has to clear, the
        skill it demands, what has actually been delivered, and what the score means.

        Without this the dashboard renders an uncalibrated score in a column called
        "Probability" with no reference point, which is how a 62% reading was read as
        an edge for six months while the real break-even sat at 43.75%.
        """
        from src.trading.barriers import BarrierPolicy
        from src.ui.reliability_panel import render_decision_panel

        kelly = self._kelly_reference()
        policy = BarrierPolicy(self.config)

        # Portfolio-level reference barriers. Per-asset figures appear in the
        # opportunities table, where they differ by volatility.
        if policy.is_volatility_scaled:
            spec = policy._spec_from_sigma(
                0.018 * (policy.horizon_days ** 0.5) * 100.0)  # a typical 1.8%-vol name
            reference_win, reference_loss = spec.win_pct, spec.loss_pct
        else:
            reference_win, reference_loss = policy.fixed_win_pct, policy.fixed_loss_pct

        # Expected days to target is a constant under volatility scaling:
        # (win_sigma * sqrt(horizon))^2 bars. Stated once here rather than repeated
        # identically down every row of the table.
        expected_days = None
        if policy.is_volatility_scaled:
            expected_days = (policy.win_sigma ** 2) * policy.horizon_days

        open_positions = 0
        try:
            open_positions = asyncio.run(self.portfolio_manager._count_alive_bets())
        except Exception as e:
            self.logger.debug(f"Could not count open positions: {e}")

        render_decision_panel(
            db_path=str(self.portfolio_manager.db_path),
            break_even_pct=policy.break_even(reference_win, reference_loss) * 100.0,
            geometry_pct=policy.geometry_probability * 100.0,
            required_edge_pct=policy.required_edge(reference_win, reference_loss),
            is_calibrated=self._calibration_is_fitted(),
            expected_days=expected_days,
            open_positions=open_positions,
            correlation_haircut=kelly.correlation_haircut(open_positions + 1),
        )

    def render_reliability_tab(self):
        """Reliability and execution-quality panels."""
        from src.ui.reliability_panel import render_execution_quality, render_reliability

        st.header("Reliability")
        st.caption(
            "Does a predicted probability match the frequency actually delivered, and "
            "do exits land where they were designed to? These are the two questions "
            "that decide whether the strategy can work."
        )

        self.render_calibration_status()
        st.divider()

        db_path = str(self.portfolio_manager.db_path)
        render_reliability(db_path, self._kelly_reference().break_even_probability * 100.0)
        st.divider()
        render_execution_quality(db_path)

    def _admin(self):
        """Portfolio admin API for the current database."""
        from src.portfolio.admin import PortfolioAdmin
        return PortfolioAdmin(self.portfolio_manager.db_path)

    def render_portfolio_admin_tab(self):
        """
        Cash movements and database lifecycle.

        These were loose scripts run by hand, which is how a portfolio ends up in a
        state nobody can reconstruct. Two rules are enforced here rather than trusted
        to the operator: every cash movement goes through the transaction ledger (so
        reconciliation keeps working), and every destructive action snapshots first
        (so it can be undone).
        """
        from src.portfolio.admin import AdminError

        admin = self._admin()
        st.header("Portfolio & Data")

        try:
            summary = admin.summary()
        except Exception as e:
            st.error(f"Could not read the database: {e}")
            return

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Cash", f"${summary['cash_balance']:,.2f}")
        col2.metric("Open bets", summary['open_bets'])
        col3.metric("Realised P&L", f"${summary['realised_pnl']:+,.2f}")
        col4.metric("Database", f"{summary['db_size_mb']:.0f} MB",
                    help=f"{summary['price_bars']:,} price bars, "
                         f"{summary['stored_predictions']:,} stored predictions")
        st.caption(f"Deposited ${summary['deposited']:,.2f} · "
                   f"withdrawn ${summary['withdrawn']:,.2f} · "
                   f"{summary['snapshots']} saved snapshot(s)")

        st.divider()

        # ------------------------------------------------------------ cash
        st.subheader("Cash")
        cash_left, cash_right = st.columns(2)

        with cash_left:
            st.markdown("**Add cash**")
            deposit_amount = st.number_input(
                "Amount to add", min_value=0.01, value=1000.0, step=100.0,
                key="admin_deposit_amount")
            deposit_note = st.text_input("Note (optional)", key="admin_deposit_note")
            if st.button("Add cash", key="admin_deposit_btn"):
                try:
                    balance = admin.deposit(deposit_amount, deposit_note)
                    st.success(f"Added ${deposit_amount:,.2f}. Cash is now ${balance:,.2f}.")
                    st.session_state.cached_state_loaded = False
                    st.rerun()
                except AdminError as e:
                    st.error(str(e))

        with cash_right:
            st.markdown("**Withdraw cash**")
            st.caption(f"Available now: **${summary['cash_balance']:,.2f}**. Money in "
                       f"open positions can only be withdrawn after they close.")
            withdraw_amount = st.number_input(
                "Amount to withdraw", min_value=0.01,
                value=min(1000.0, max(0.01, summary['cash_balance'])), step=100.0,
                key="admin_withdraw_amount")
            withdraw_note = st.text_input("Note (optional)", key="admin_withdraw_note")
            if st.button("Withdraw cash", key="admin_withdraw_btn"):
                try:
                    balance = admin.withdraw(withdraw_amount, withdraw_note)
                    st.success(f"Withdrew ${withdraw_amount:,.2f}. Cash is now ${balance:,.2f}.")
                    st.session_state.cached_state_loaded = False
                    st.rerun()
                except AdminError as e:
                    st.error(str(e))

        history = admin.cash_history(limit=20)
        if history:
            with st.expander("Cash in / out history"):
                st.dataframe(
                    pd.DataFrame([{
                        'When': h['timestamp'][:19].replace('T', ' '),
                        'Type': h['type'],
                        'Amount': h['amount'],
                        'Balance after': h['balance_after'],
                        'Note': h['description'],
                    } for h in history]),
                    use_container_width=True, hide_index=True,
                    column_config={
                        'Amount': st.column_config.NumberColumn(format="$%+,.2f"),
                        'Balance after': st.column_config.NumberColumn(format="$%,.2f"),
                    })

        st.divider()

        # -------------------------------------------------------- snapshots
        st.subheader("Snapshots")
        st.caption("A snapshot is a complete copy of the database — positions, cash "
                   "ledger, predictions and cached prices. Restore and reset both "
                   "take one automatically first, so either can be undone.")

        snap_left, snap_right = st.columns([1, 2])

        with snap_left:
            snapshot_label = st.text_input("Label", value="manual",
                                           key="admin_snapshot_label")
            if st.button("Save snapshot now", key="admin_snapshot_btn"):
                try:
                    snapshot = admin.create_snapshot(snapshot_label)
                    st.success(f"Saved {snapshot.name} ({snapshot.size_mb:.0f} MB)")
                    st.rerun()
                except AdminError as e:
                    st.error(str(e))

        snapshots = admin.list_snapshots()
        with snap_right:
            if not snapshots:
                st.info("No snapshots yet.")
            else:
                st.dataframe(
                    pd.DataFrame([{
                        'Snapshot': s.name,
                        'Taken': s.created.strftime('%Y-%m-%d %H:%M:%S'),
                        'Label': s.label,
                        'Size (MB)': round(s.size_mb, 1),
                    } for s in snapshots]),
                    use_container_width=True, hide_index=True, height=200)

        if snapshots:
            st.markdown("**Restore from a snapshot**")
            st.caption("Replaces the live database. The current state is saved as a "
                       "`pre-restore` snapshot first, so this is reversible.")
            chosen = st.selectbox("Snapshot to restore",
                                  [s.name for s in snapshots],
                                  key="admin_restore_choice")
            confirm_restore = st.checkbox(
                f"I understand this replaces the current book "
                f"({summary['open_bets']} open bets, ${summary['cash_balance']:,.2f} cash)",
                key="admin_restore_confirm")
            if st.button("Restore", key="admin_restore_btn",
                         disabled=not confirm_restore):
                try:
                    safety = admin.restore_snapshot(chosen)
                    st.success(f"Restored {chosen}. Previous state saved as "
                               f"{safety.name if safety else 'n/a'}.")
                    st.session_state.cached_state_loaded = False
                    st.rerun()
                except AdminError as e:
                    st.error(str(e))

        st.divider()

        # ------------------------------------------------------------ reset
        st.subheader("Start over")
        st.caption("Clears positions, the cash ledger, predictions and performance "
                   "history, then opens with the capital you specify. **Your saved "
                   "snapshots are not touched**, and the current state is snapshotted "
                   "first.")

        reset_left, reset_right = st.columns(2)
        with reset_left:
            new_capital = st.number_input(
                "Opening capital", min_value=0.0,
                value=float(self.config.get('trading', {}).get('initial_capital', 10000.0)),
                step=1000.0, key="admin_reset_capital")
            keep_prices = st.checkbox(
                "Keep cached price history", value=True, key="admin_reset_keep_prices",
                help="Recommended. Price bars are an expensive, rate-limited cache "
                     "with nothing to do with your trading record — clearing them "
                     "forces a full re-download.")

        with reset_right:
            st.warning(f"This will clear **{summary['total_bets']} bets** and a cash "
                       f"ledger of **${summary['cash_balance']:,.2f}**.")
            typed = st.text_input('Type RESET to confirm', key="admin_reset_typed")
            if st.button("Clear everything and start over", key="admin_reset_btn",
                         type="primary", disabled=(typed.strip().upper() != "RESET")):
                try:
                    safety = admin.reset(new_capital, keep_price_history=keep_prices)
                    st.success(f"Portfolio reset. Opening capital "
                               f"${new_capital:,.2f}. Previous state saved as "
                               f"{safety.name} — restore it above to undo this.")
                    st.session_state.cached_state_loaded = False
                    st.rerun()
                except AdminError as e:
                    st.error(str(e))

    def run(self):
        """Main dashboard runner"""
        st.set_page_config(
            page_title="Kelly Trading Dashboard",
            page_icon=":chart_with_upwards_trend:",
            layout="wide",
            initial_sidebar_state="collapsed"
        )

        # NO AUTO-REFRESH. Market data is fetched only when the user asks for it.
        #
        # This page used to call st_autorefresh() on a timer, which re-fetched the
        # whole universe unattended and burned through the yfinance rate limit. Every
        # Streamlit rerun -- including switching tabs, sorting a table or clicking a
        # row -- re-runs this method, so anything that fetches here fetches constantly.
        #
        # The only path that touches an external API is the explicit "Refresh prices"
        # button. Everything else reads the local database.
        if st.session_state.needs_refresh:
            asyncio.run(self.refresh_data())
            st.session_state.needs_refresh = False

        # First load reads the DATABASE only: portfolio, bets, and the most recent
        # stored predictions. No network calls, so opening the page (or reopening it
        # in a new browser session) costs nothing in API quota.
        if not st.session_state.get('cached_state_loaded'):
            asyncio.run(self.load_cached_state())
            st.session_state.cached_state_loaded = True

        # Render dashboard components
        self.render_header()

        # Create persistent tab selection using radio buttons
        if 'active_tab' not in st.session_state:
            st.session_state.active_tab = "Trading Dashboard"

        tab_options = ["Trading Dashboard", "All Bets", "Reliability",
                       "Market Data", "Portfolio & Data"]
        if st.session_state.active_tab not in tab_options:
            st.session_state.active_tab = tab_options[0]

        selected_tab = st.radio(
            "Dashboard Sections:",
            options=tab_options,
            index=tab_options.index(st.session_state.active_tab),
            horizontal=True,
            key="dashboard_tab_selector"
        )

        # Update session state if tab changed
        if selected_tab != st.session_state.active_tab:
            st.session_state.active_tab = selected_tab

        st.divider()

        # Render content based on selected tab
        if selected_tab == "Trading Dashboard":
            # State up front whether the numbers below are probabilities or scores.
            self.render_calibration_status()
            st.divider()

            # Main trading dashboard content
            self.render_portfolio_overview()
            st.divider()

            self.render_trading_controls()
            st.divider()

            # Market Opportunities section
            self.render_opportunities()

            st.divider()

            # Active Bets section (underneath Market Opportunities)
            self.render_active_bets()

            st.divider()
            self.render_performance_charts()

        elif selected_tab == "All Bets":
            # Bets tab content
            self.render_bets_tab()

        elif selected_tab == "Reliability":
            self.render_reliability_tab()

        elif selected_tab == "Portfolio & Data":
            self.render_portfolio_admin_tab()

        elif selected_tab == "Market Data":
            # Market data visualization tab
            self.render_market_data_tab()


def main():
    """Entry point for the Streamlit dashboard"""
    dashboard = TradingDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()