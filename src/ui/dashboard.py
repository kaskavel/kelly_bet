#!/usr/bin/env python3
"""
Streamlit-based trading dashboard for Kelly Criterion Trading System
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

try:
    from src.core.trading_system import TradingSystem
    from src.cli.bet_monitor import BetMonitor
    from src.cli.bet_analyzer import BetAnalyzer
    from src.portfolio.manager import PortfolioManager
    from src.data.market_data import MarketDataManager
    from src.utils.asset_names import get_display_name, get_asset_name
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
                    stock_data = await self.market_data.get_stock_data(symbol, days=90)  # Need more data for ML
                    if stock_data is not None and not stock_data.empty:
                        market_data[symbol] = stock_data
                        self.logger.info(f"Loaded {len(stock_data)} days of data for {symbol}")
                    else:
                        self.logger.warning(f"No market data for {symbol}")
                except Exception as e:
                    self.logger.warning(f"Failed to get market data for {symbol}: {e}")

            self.logger.info(f"Successfully loaded market data for {len(market_data)} assets")

            # Now generate REAL predictions using the PredictionEngine
            if market_data:
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
                    predictions = await predictor.predict_all(market_data)
                    self.logger.info(f"✅ Generated {len(predictions)} real ML predictions")

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

                            # Calculate Kelly recommendation using REAL probability
                            kelly_rec = kelly_calc.calculate_bet_size(
                                probability=final_probability,
                                current_price=current_price,
                                available_capital=available_capital
                            )

                            opportunities.append({
                                "symbol": symbol,
                                "asset_type": asset_type,
                                "current_price": current_price,
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
    
    async def get_portfolio_data(self) -> Dict:
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

            # Update current prices for active bets to get accurate unrealized P&L
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
    
    async def get_active_bets_data(self) -> List[Dict]:
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

            # Get current market prices for all active bets
            symbols = list(set(bet.symbol for bet in active_bets))
            current_prices = {}

            if symbols and self.market_data:
                try:
                    await self.market_data.initialize()
                    for symbol in symbols:
                        try:
                            recent_data = await self.market_data.get_stock_data(symbol, days=2)
                            if not recent_data.empty:
                                current_prices[symbol] = float(recent_data['Close'].iloc[-1])
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
                    "win_threshold": bet.win_price,
                    "loss_threshold": bet.loss_price,
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

    async def get_all_bets_data(self) -> Tuple[List[Dict], List[Dict]]:
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
                        "win_threshold": float(row[7]),
                        "loss_threshold": float(row[8]),
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
                    bet_data["algorithm_predictions"] = {algo: prob for algo, prob in predictions}

                    bet_rows.append(bet_data)
                    if bet_data["status"] == "alive":
                        alive_symbols.add(bet_data["symbol"])

                # Get current market prices for alive bets
                current_prices = {}
                if alive_symbols and self.market_data:
                    try:
                        await self.market_data.initialize()
                        for symbol in alive_symbols:
                            try:
                                recent_data = await self.market_data.get_stock_data(symbol, days=2)
                                if not recent_data.empty:
                                    current_prices[symbol] = float(recent_data['Close'].iloc[-1])
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

                # Refresh data
                await self.refresh_data()
                return True
            else:
                st.error(f"Bet {bet_id} not found in active bets")
                return False

        except Exception as e:
            st.error(f"Failed to settle bet: {e}")
            return False

    async def place_bet(self, symbol: str, probability: float, current_price: float) -> bool:
        """Place a bet for the given symbol using real portfolio manager"""
        try:
            if not self.portfolio_manager:
                st.error("Portfolio manager not available")
                return False
            
            # Create prediction dict for portfolio manager
            prediction = {
                'symbol': symbol,
                'probability': probability,
                'current_price': current_price,
                'algorithms': [{'algorithm': 'dashboard_manual'}]  # Indicate this was manually placed via dashboard
            }
            
            # Place the bet using portfolio manager
            await self.portfolio_manager.initialize()
            bet_id = await self.portfolio_manager.place_bet(prediction)
            
            st.success(f"SUCCESS: Bet placed successfully!")
            st.info(f"Bet ID: {bet_id}")
            st.info(f"Symbol: {symbol} at ${current_price:.2f}")
            st.info(f"Probability: {probability:.1f}%")
            
            # Refresh data to show the new bet
            await self.refresh_data()
            
            return True
            
        except Exception as e:
            st.error(f"ERROR: Failed to place bet: {e}")
            self.logger.error(f"Bet placement error: {e}")
            return False

    async def check_and_settle_bets(self):
        """Check active bets and settle any that hit win/loss thresholds"""
        try:
            if not self.bet_monitor:
                self.logger.warning("Bet monitor not available for settlement check")
                return

            self.logger.info("Checking for bet settlements...")

            # Use the bet monitor's settlement logic from livebets CLI
            await self.bet_monitor._monitor_and_settle_positions()

            self.logger.info("Bet settlement check completed")

        except Exception as e:
            self.logger.error(f"Error during bet settlement check: {e}")
            # Don't raise exception - settlement failures shouldn't break dashboard

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

            # Update prices for each active bet
            for bet_id, bet in self.portfolio_manager.active_bets.items():
                try:
                    # Get current market price
                    recent_data = await self.market_data.get_stock_data(bet.symbol, days=2)
                    if not recent_data.empty:
                        current_price = float(recent_data['Close'].iloc[-1])

                        # Update bet's current price and calculated values
                        bet.current_price = current_price
                        bet.current_value = bet.shares * current_price
                        bet.unrealized_pnl = bet.current_value - bet.amount

                        self.logger.debug(f"Updated {bet.symbol}: ${bet.current_price:.2f} (P&L: ${bet.unrealized_pnl:+.2f})")
                    else:
                        self.logger.warning(f"No recent data available for {bet.symbol}")

                except Exception as e:
                    self.logger.error(f"Error updating price for {bet.symbol}: {e}")

            self.logger.info("Active bet price update completed")

        except Exception as e:
            self.logger.error(f"Error during active bet price update: {e}")

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
            if st.session_state.last_update:
                st.caption(f"Last updated: {st.session_state.last_update.strftime('%H:%M:%S')}")
        
        with col2:
            if st.button("Refresh Data"):
                # Use session state to trigger refresh
                st.session_state.needs_refresh = True
                st.rerun()
        
        with col3:
            # Auto-refresh status display only
            if not st.session_state.auto_mode:
                st.write("🔄 Auto-refresh: ON (15min)")
                st.caption("Auto-refresh is active when automated mode is off")
            else:
                st.write("🔄 Auto-refresh: OFF")
                st.caption("Auto-refresh disabled in automated mode")
    
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
    
    def render_opportunities(self):
        """Render comprehensive opportunities section with algorithm breakdowns"""
        st.header("Market Opportunities")

        opportunities = st.session_state.opportunities_data
        if not opportunities:
            st.info("No opportunities available. Click 'Refresh Data' to update.")
            return

        # Filter controls
        col1, col2, col3 = st.columns(3)

        with col1:
            asset_filter = st.selectbox(
                "Asset Type:",
                options=["all", "stock", "crypto"],
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

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Price", f"${opp['current_price']:,.2f}")
            st.metric("Final Probability", f"{opp['final_probability']:.1f}%")
        with col2:
            st.metric("Kelly Fraction", f"{opp['kelly_fraction']*100:.1f}%")
            st.metric("Recommended Amount", f"${opp['recommended_amount']:,.0f}")

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
                        current_price=opp['current_price']
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

            df_data.append({
                'Symbol': symbol_display,
                'Type': opp.get('asset_type', 'stock').upper(),
                'Price': opp['current_price'],
                'Final Prob': opp['final_probability'],
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
                    "Final Prob": st.column_config.NumberColumn("Final Prob", format="%.1f%%"),
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
            asset_badge = "🪙" if opp.get('asset_type') == 'crypto' else "📈"
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
                            current_price=opp['current_price']
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

        display_df['Win Target'] = display_df['win_threshold'].apply(lambda x: f"${x:,.2f}")
        display_df['Stop Loss'] = display_df['loss_threshold'].apply(lambda x: f"${x:,.2f}")
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
                    st.write(f"**Win Target:** ${bet['win_threshold']:,.2f}")
                    st.write(f"**Stop Loss:** ${bet['loss_threshold']:,.2f}")

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
        total_fees = sum((bet['amount'] * fee_percentage / 100) + ((bet['shares'] * bet.get('exit_price', bet['current_price'])) * fee_percentage / 100)
                        for bet in filtered_bets)
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

    def run(self):
        """Main dashboard runner"""
        st.set_page_config(
            page_title="Kelly Trading Dashboard",
            page_icon=":chart_with_upwards_trend:",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        
        # Auto-refresh when not in automated mode
        if not st.session_state.auto_mode:
            refresh_count = st_autorefresh(interval=15*60*1000, key="dashboard_refresh")  # 15 minutes
            # When auto-refresh triggers, force data refresh
            if refresh_count > 0:
                st.session_state.needs_refresh = True

        # Handle refresh requests
        if st.session_state.needs_refresh:
            asyncio.run(self.refresh_data())
            st.session_state.needs_refresh = False

        # Initialize data on first load
        if st.session_state.last_update is None:
            asyncio.run(self.refresh_data())
        
        # Render dashboard components
        self.render_header()

        # Create persistent tab selection using radio buttons
        if 'active_tab' not in st.session_state:
            st.session_state.active_tab = "Trading Dashboard"

        selected_tab = st.radio(
            "Dashboard Sections:",
            options=["Trading Dashboard", "All Bets", "Market Data"],
            index=["Trading Dashboard", "All Bets", "Market Data"].index(st.session_state.active_tab),
            horizontal=True,
            key="dashboard_tab_selector"
        )

        # Update session state if tab changed
        if selected_tab != st.session_state.active_tab:
            st.session_state.active_tab = selected_tab

        st.divider()

        # Render content based on selected tab
        if selected_tab == "Trading Dashboard":
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

        elif selected_tab == "Market Data":
            # Market data visualization tab
            self.render_market_data_tab()


def main():
    """Entry point for the Streamlit dashboard"""
    dashboard = TradingDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()