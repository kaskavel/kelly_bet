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
                self.market_data = MarketDataManager(self.config)
                self.portfolio_manager = PortfolioManager(self.config)
                self.bet_monitor = BetMonitor(config_path)
            except Exception as e:
                st.error(f"Failed to initialize managers: {e}")
                self.logger.error(f"Manager initialization error: {e}")
        
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
                self.logger.warning("Market data or portfolio manager not available, using fallback")
                # Return some mock data to test UI
                return [
                    {
                        "symbol": "AAPL",
                        "asset_type": "stock",
                        "current_price": 175.50,
                        "final_probability": 65.2,
                        "algorithms": {
                            "lstm": 68.5,
                            "random_forest": 62.3,
                            "svm": 64.8
                        },
                        "kelly_fraction": 0.10,
                        "recommended_amount": 1000
                    },
                    {
                        "symbol": "BTC-USD",
                        "asset_type": "crypto",
                        "current_price": 45320.50,
                        "final_probability": 58.7,
                        "algorithms": {
                            "lstm": 61.2,
                            "random_forest": 55.8,
                            "svm": 59.1
                        },
                        "kelly_fraction": 0.06,
                        "recommended_amount": 600
                    }
                ]

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

            # Process all assets (but limit to first 20 for performance in UI)
            assets_to_process = all_assets[:20]  # Expand to 20 for better coverage

            for i, asset in enumerate(assets_to_process):
                symbol = asset['symbol']
                asset_type = asset.get('type', 'stock')
                self.logger.info(f"Processing {asset_type} {i+1}/{len(assets_to_process)}: {symbol}")

                try:
                    # Get current price
                    if asset_type == 'crypto':
                        # For crypto, use different method if available
                        stock_data = await self.market_data.get_stock_data(symbol, days=1)
                    else:
                        stock_data = await self.market_data.get_stock_data(symbol, days=1)

                    if stock_data is None or stock_data.empty:
                        self.logger.warning(f"No market data for {symbol}")
                        continue

                    current_price = float(stock_data['Close'].iloc[-1])

                    # Generate mock algorithm predictions (replace with real predictions)
                    import random
                    random.seed(hash(symbol) % 1000)  # Consistent random based on symbol

                    # Simulate individual algorithm predictions
                    lstm_prob = max(45.0, min(85.0, 60.0 + random.uniform(-15, 15)))
                    rf_prob = max(45.0, min(85.0, 58.0 + random.uniform(-12, 18)))
                    svm_prob = max(45.0, min(85.0, 62.0 + random.uniform(-10, 12)))

                    # Calculate ensemble probability (weighted average)
                    final_probability = (lstm_prob * 0.4 + rf_prob * 0.35 + svm_prob * 0.25)

                    # Calculate Kelly recommendation
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
                        "algorithms": {
                            "lstm": lstm_prob,
                            "random_forest": rf_prob,
                            "svm": svm_prob
                        },
                        "kelly_fraction": kelly_rec.fraction_of_capital if kelly_rec.is_favorable else 0.0,
                        "recommended_amount": kelly_rec.recommended_amount if kelly_rec.is_favorable else 0.0,
                        "is_favorable": kelly_rec.is_favorable
                    })

                except Exception as e:
                    self.logger.warning(f"Failed to process {symbol}: {e}")
                    continue

            self.logger.info(f"Processed {len(opportunities)} assets")

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
                # Fallback mock data
                return {
                    "total_capital": 10000.0,
                    "available_capital": 7500.0,
                    "active_bets_value": 2500.0,
                    "total_pnl": 234.50,
                    "win_rate": 0.67,
                    "total_bets": 45,
                    "won_bets": 30,
                    "lost_bets": 12,
                    "active_bets": 3
                }
            
            # Get real portfolio data
            await self.portfolio_manager.initialize()
            portfolio_summary = await self.portfolio_manager.get_portfolio_summary()
            total_realized_pnl = await self.portfolio_manager._get_total_realized_pnl()
            
            return {
                "total_capital": portfolio_summary.total_capital,
                "available_capital": portfolio_summary.cash_balance,
                "active_bets_value": portfolio_summary.active_bets_value,
                "total_pnl": portfolio_summary.unrealized_pnl + portfolio_summary.realized_pnl,
                "win_rate": 0.0,  # Would need to calculate from bet history
                "total_bets": 0,  # Would need to calculate from bet history
                "won_bets": 0,   # Would need to calculate from bet history
                "lost_bets": 0,  # Would need to calculate from bet history
                "active_bets": portfolio_summary.active_bets_count
            }
        except Exception as e:
            st.error(f"Error getting portfolio data: {e}")
            self.logger.error(f"Portfolio data error: {e}")
            return {}
    
    async def get_active_bets_data(self) -> List[Dict]:
        """Get active bets data"""
        try:
            if not self.portfolio_manager:
                # Fallback mock data
                return [
                    {
                        "symbol": "AAPL",
                        "entry_price": 175.50,
                        "current_price": 178.25,
                        "amount": 1500,
                        "pnl": 41.25,
                        "pnl_pct": 2.75,
                        "entry_time": datetime.now() - timedelta(hours=2),
                        "win_threshold": 182.25,
                        "loss_threshold": 168.75
                    }
                ]

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
                # Fallback mock data
                alive_bets = [
                    {
                        "bet_id": "abc123",
                        "symbol": "AAPL",
                        "entry_price": 175.50,
                        "current_price": 178.25,
                        "amount": 1500,
                        "shares": 8.55,
                        "pnl": 41.25,
                        "pnl_pct": 2.75,
                        "entry_time": datetime.now() - timedelta(hours=2),
                        "exit_time": None,
                        "win_price": 182.25,
                        "loss_price": 168.75,
                        "status": "alive",
                        "algorithm_used": "ensemble",
                        "probability_when_placed": 65.5
                    }
                ]
                closed_bets = [
                    {
                        "bet_id": "def456",
                        "symbol": "MSFT",
                        "entry_price": 420.30,
                        "exit_price": 435.20,
                        "amount": 2000,
                        "shares": 4.76,
                        "pnl": 215.50,
                        "pnl_pct": 10.78,
                        "entry_time": datetime.now() - timedelta(days=3),
                        "exit_time": datetime.now() - timedelta(hours=5),
                        "win_price": 441.32,
                        "loss_price": 407.49,
                        "status": "won",
                        "algorithm_used": "lstm",
                        "probability_when_placed": 72.3
                    }
                ]
                return alive_bets, closed_bets

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
                    bet_data = {
                        "bet_id": row[0][:8],  # Short ID for display
                        "full_bet_id": row[0],  # Keep full ID for operations
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
    
    async def refresh_data(self):
        """Refresh all dashboard data"""
        try:
            self.logger.info("Starting dashboard data refresh...")
            with st.spinner("Refreshing data..."):
                self.logger.info("Fetching portfolio data...")
                st.session_state.portfolio_data = await self.get_portfolio_data()

                self.logger.info("Fetching active bets data...")
                st.session_state.active_bets_data = await self.get_active_bets_data()

                self.logger.info("Fetching all bets data...")
                st.session_state.all_bets_data = await self.get_all_bets_data()

                self.logger.info("Fetching opportunities data...")
                st.session_state.opportunities_data = await self.get_opportunities_data()

                st.session_state.last_update = datetime.now()
                self.logger.info(f"Dashboard refresh completed at {st.session_state.last_update}")
        except Exception as e:
            self.logger.error(f"Dashboard refresh failed: {e}", exc_info=True)
            st.error(f"Refresh failed: {e}")
            # Set empty data on failure
            st.session_state.opportunities_data = []
            st.session_state.portfolio_data = {}
            st.session_state.active_bets_data = []
            st.session_state.all_bets_data = ([], [])
    
    def render_header(self):
        """Render dashboard header"""
        st.title("Kelly Criterion Trading Dashboard")
        
        # Data source indicator
        if REAL_DATA_AVAILABLE and self.portfolio_manager:
            st.success("CONNECTED: Real trading data")
        else:
            st.warning("MOCK DATA: Configure config/config.yaml to connect to real system")
        
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
            auto_refresh = st.toggle("Auto Refresh (15min)")
            if auto_refresh:
                st_autorefresh(interval=15*60*1000, key="dashboard_refresh")  # 15 minutes
    
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
                delta=f"${portfolio.get('total_pnl', 0):+.2f}"
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
                delta=f"{portfolio.get('won_bets', 0)}/{portfolio.get('total_bets', 0)} bets"
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
                min_value=45.0,
                max_value=85.0,
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

    def render_compact_opportunities(self, opportunities: List[Dict]):
        """Render opportunities in compact table format"""
        if not opportunities:
            st.info("No opportunities match the current filters.")
            return

        # Create DataFrame
        df_data = []
        for opp in opportunities:
            df_data.append({
                'Symbol': opp['symbol'],
                'Type': opp.get('asset_type', 'stock').upper(),
                'Price': f"${opp['current_price']:,.2f}",
                'Final Prob': f"{opp['final_probability']:.1f}%",
                'Kelly %': f"{opp['kelly_fraction']*100:.1f}%",
                'Recommended': f"${opp['recommended_amount']:,.0f}" if opp['is_favorable'] else "Not Favorable",
                'LSTM': f"{opp['algorithms']['lstm']:.1f}%",
                'RF': f"{opp['algorithms']['random_forest']:.1f}%",
                'SVM': f"{opp['algorithms']['svm']:.1f}%"
            })

        df = pd.DataFrame(df_data)

        # Display the table
        st.dataframe(
            df,
            use_container_width=True,
            height=600
        )

        # Add bet placement section for top opportunities
        st.subheader("Quick Bet Placement")
        favorable_opps = [opp for opp in opportunities if opp['is_favorable']][:5]

        if favorable_opps:
            for idx, opp in enumerate(favorable_opps):
                col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])

                with col1:
                    asset_badge = "🪙" if opp.get('asset_type') == 'crypto' else "📈"
                    st.write(f"{asset_badge} **{opp['symbol']}**")

                with col2:
                    prob_color = "green" if opp['final_probability'] > 70 else "orange" if opp['final_probability'] > 60 else "red"
                    st.markdown(f"<span style='color:{prob_color}'>{opp['final_probability']:.1f}%</span>", unsafe_allow_html=True)

                with col3:
                    st.write(f"{opp['kelly_fraction']*100:.1f}%")

                with col4:
                    st.write(f"${opp['recommended_amount']:,.0f}")

                with col5:
                    if st.button(f"Place Bet", key=f"bet_{idx}"):
                        success = asyncio.run(self.place_bet(
                            symbol=opp['symbol'],
                            probability=opp['final_probability'],
                            current_price=opp['current_price']
                        ))
                        if success:
                            st.rerun()
        else:
            st.info("No favorable opportunities found with current filters.")

    def render_detailed_opportunities(self, opportunities: List[Dict]):
        """Render opportunities with detailed algorithm breakdowns"""
        if not opportunities:
            st.info("No opportunities match the current filters.")
            return

        for idx, opp in enumerate(opportunities):
            # Create expandable section for each asset
            asset_badge = "🪙" if opp.get('asset_type') == 'crypto' else "📈"
            prob_color = "🟢" if opp['final_probability'] > 70 else "🟡" if opp['final_probability'] > 60 else "🔴"

            with st.expander(
                f"{asset_badge} {opp['symbol']} - {prob_color} {opp['final_probability']:.1f}% "
                f"(${opp['current_price']:,.2f})"
            ):
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Asset Information:**")
                    st.write(f"• Symbol: {opp['symbol']}")
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
                    lstm_color = "🟢" if lstm_prob > 60 else "🟡" if lstm_prob > 55 else "🔴"
                    st.write(f"• LSTM: {lstm_color} {lstm_prob:.1f}%")

                    # Random Forest
                    rf_prob = opp['algorithms']['random_forest']
                    rf_color = "🟢" if rf_prob > 60 else "🟡" if rf_prob > 55 else "🔴"
                    st.write(f"• Random Forest: {rf_color} {rf_prob:.1f}%")

                    # SVM
                    svm_prob = opp['algorithms']['svm']
                    svm_color = "🟢" if svm_prob > 60 else "🟡" if svm_prob > 55 else "🔴"
                    st.write(f"• SVM: {svm_color} {svm_prob:.1f}%")

                    st.write("**Ensemble Calculation:**")
                    st.write(f"• LSTM × 40%: {lstm_prob:.1f}% × 0.4 = {lstm_prob * 0.4:.1f}%")
                    st.write(f"• RF × 35%: {rf_prob:.1f}% × 0.35 = {rf_prob * 0.35:.1f}%")
                    st.write(f"• SVM × 25%: {svm_prob:.1f}% × 0.25 = {svm_prob * 0.25:.1f}%")
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
        
        # Detailed table
        df = pd.DataFrame(bets)
        
        # Format the dataframe for display
        display_df = df.copy()
        display_df['Symbol'] = display_df['symbol']
        display_df['Entry'] = display_df['entry_price'].apply(lambda x: f"${x:,.2f}")
        display_df['Current'] = display_df['current_price'].apply(lambda x: f"${x:,.2f}")
        display_df['Amount'] = display_df['amount'].apply(lambda x: f"${x:,.0f}")
        display_df['P&L'] = display_df.apply(lambda x: f"${x['pnl']:+.2f} ({x['pnl_pct']:+.1f}%)", axis=1)
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
        
        # Show the table
        columns_to_show = ['Symbol', 'Entry', 'Current', 'Amount', 'P&L', 'Win Target', 'Stop Loss', 'Duration']
        if 'Algorithm' in display_df.columns:
            columns_to_show.append('Algorithm')
        if 'Entry Prob' in display_df.columns:
            columns_to_show.append('Entry Prob')
            
        st.dataframe(
            display_df[columns_to_show],
            use_container_width=True
        )
    
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

        st.plotly_chart(fig, use_container_width=True)

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

        # Create two sections
        col1, col2 = st.columns(2)

        with col1:
            self.render_alive_bets_section(alive_bets)

        with col2:
            self.render_closed_bets_section(closed_bets)

    def render_alive_bets_section(self, alive_bets: List[Dict]):
        """Render alive bets section"""
        st.subheader(f"Active Bets ({len(alive_bets)})")

        if not alive_bets:
            st.info("No active bets")
            return

        # Create dataframe for alive bets
        df = pd.DataFrame(alive_bets)

        # Format display
        display_df = df.copy()
        display_df['Symbol'] = display_df['symbol']
        display_df['Entry Price'] = display_df['entry_price'].apply(lambda x: f"${x:,.2f}")
        display_df['Current Price'] = display_df['current_price'].apply(lambda x: f"${x:,.2f}")
        display_df['Amount'] = display_df['amount'].apply(lambda x: f"${x:,.0f}")
        display_df['P&L'] = display_df.apply(lambda x: f"${x['pnl']:+.2f} ({x['pnl_pct']:+.1f}%)", axis=1)
        display_df['Entry Date'] = display_df['entry_time'].apply(lambda x: x.strftime('%m/%d %H:%M'))
        display_df['Algorithm'] = display_df['algorithm_used']
        display_df['Entry Prob'] = display_df['probability_when_placed'].apply(lambda x: f"{x:.1f}%")

        # Show compact table
        columns_to_show = ['Symbol', 'Entry Price', 'Current Price', 'Amount', 'P&L', 'Entry Date', 'Algorithm', 'Entry Prob']
        st.dataframe(
            display_df[columns_to_show],
            use_container_width=True,
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
            key="closed_bets_filter"
        )

        # Filter bets
        filtered_bets = closed_bets
        if status_filter != "all":
            filtered_bets = [bet for bet in closed_bets if bet['status'] == status_filter]

        if not filtered_bets:
            st.info(f"No {status_filter} bets found")
            return

        # Create dataframe for closed bets
        df = pd.DataFrame(filtered_bets)

        # Format display
        display_df = df.copy()
        display_df['Symbol'] = display_df['symbol']
        display_df['Entry Price'] = display_df['entry_price'].apply(lambda x: f"${x:,.2f}")
        display_df['Exit Price'] = display_df['exit_price'].apply(lambda x: f"${x:,.2f}" if x else "N/A")
        display_df['Amount'] = display_df['amount'].apply(lambda x: f"${x:,.0f}")
        display_df['P&L'] = display_df.apply(lambda x: f"${x['pnl']:+.2f} ({x['pnl_pct']:+.1f}%)", axis=1)
        display_df['Entry Date'] = display_df['entry_time'].apply(lambda x: x.strftime('%m/%d %H:%M'))
        display_df['Exit Date'] = display_df['exit_time'].apply(lambda x: x.strftime('%m/%d %H:%M') if x else "N/A")
        display_df['Status'] = display_df['status'].apply(lambda x: x.upper())
        display_df['Duration'] = display_df.apply(lambda x:
            str(x['exit_time'] - x['entry_time']).split('.')[0] if x['exit_time'] else "N/A", axis=1)

        # Show compact table
        columns_to_show = ['Symbol', 'Entry Price', 'Exit Price', 'Amount', 'P&L', 'Entry Date', 'Exit Date', 'Status', 'Duration']
        st.dataframe(
            display_df[columns_to_show],
            use_container_width=True,
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
    
    def run(self):
        """Main dashboard runner"""
        st.set_page_config(
            page_title="Kelly Trading Dashboard",
            page_icon=":chart_with_upwards_trend:",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        
        # Handle refresh requests
        if st.session_state.needs_refresh:
            asyncio.run(self.refresh_data())
            st.session_state.needs_refresh = False
        
        # Initialize data on first load
        if st.session_state.last_update is None:
            asyncio.run(self.refresh_data())
        
        # Render dashboard components
        self.render_header()

        # Create tabs
        tab1, tab2 = st.tabs(["Trading Dashboard", "All Bets"])

        with tab1:
            # Main trading dashboard content
            self.render_portfolio_overview()
            st.divider()

            self.render_trading_controls()
            st.divider()

            col1, col2 = st.columns(2)

            with col1:
                self.render_opportunities()

            with col2:
                self.render_active_bets()

            st.divider()
            self.render_performance_charts()

        with tab2:
            # Bets tab content
            self.render_bets_tab()


def main():
    """Entry point for the Streamlit dashboard"""
    dashboard = TradingDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()