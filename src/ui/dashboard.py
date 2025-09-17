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
except ImportError:
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


class TradingDashboard:
    """Main dashboard class for the trading system UI"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.setup_logging()
        
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
        """Get top opportunities from trading system"""
        try:
            trading_system = await self.initialize_trading_system()
            if not trading_system:
                return []
            
            # This would call your existing prediction logic
            # For now, returning mock data - you'll need to integrate with your actual system
            opportunities = [
                {"symbol": "AAPL", "probability": 75.2, "kelly_fraction": 0.15, "recommended_amount": 1500},
                {"symbol": "MSFT", "probability": 68.9, "kelly_fraction": 0.12, "recommended_amount": 1200},
                {"symbol": "GOOGL", "probability": 62.4, "kelly_fraction": 0.08, "recommended_amount": 800},
                {"symbol": "BTC-USD", "probability": 71.8, "kelly_fraction": 0.14, "recommended_amount": 1400},
                {"symbol": "ETH-USD", "probability": 59.3, "kelly_fraction": 0.06, "recommended_amount": 600},
            ]
            
            return opportunities
        except Exception as e:
            st.error(f"Error getting opportunities: {e}")
            return []
    
    async def get_portfolio_data(self) -> Dict:
        """Get portfolio status data"""
        try:
            # Mock data - integrate with your portfolio manager
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
        except Exception as e:
            st.error(f"Error getting portfolio data: {e}")
            return {}
    
    async def get_active_bets_data(self) -> List[Dict]:
        """Get active bets data"""
        try:
            # Mock data - integrate with your bet monitor
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
                },
                {
                    "symbol": "BTC-USD",
                    "entry_price": 43500,
                    "current_price": 44200,
                    "amount": 1400,
                    "pnl": 22.54,
                    "pnl_pct": 1.61,
                    "entry_time": datetime.now() - timedelta(hours=5),
                    "win_threshold": 45225,
                    "loss_threshold": 41775
                }
            ]
        except Exception as e:
            st.error(f"Error getting active bets: {e}")
            return []
    
    async def place_bet(self, symbol: str, amount: float) -> bool:
        """Place a bet for the given symbol"""
        try:
            trading_system = await self.initialize_trading_system()
            if not trading_system:
                return False
            
            # Integrate with your actual trading system
            st.success(f"Bet placed: {symbol} for ${amount:.2f}")
            return True
        except Exception as e:
            st.error(f"Failed to place bet: {e}")
            return False
    
    async def refresh_data(self):
        """Refresh all dashboard data"""
        with st.spinner("Refreshing data..."):
            st.session_state.opportunities_data = await self.get_opportunities_data()
            st.session_state.portfolio_data = await self.get_portfolio_data()
            st.session_state.active_bets_data = await self.get_active_bets_data()
            st.session_state.last_update = datetime.now()
    
    def render_header(self):
        """Render dashboard header"""
        st.title("🎯 Kelly Criterion Trading Dashboard")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.session_state.last_update:
                st.caption(f"Last updated: {st.session_state.last_update.strftime('%H:%M:%S')}")
        
        with col2:
            if st.button("🔄 Refresh Data"):
                asyncio.run(self.refresh_data())
                st.rerun()
        
        with col3:
            auto_refresh = st.toggle("Auto Refresh (15min)")
            if auto_refresh:
                st_autorefresh(interval=15*60*1000, key="dashboard_refresh")  # 15 minutes
    
    def render_portfolio_overview(self):
        """Render portfolio overview section"""
        st.header("📊 Portfolio Overview")
        
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
        st.header("🎮 Trading Controls")
        
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
        """Render top opportunities section"""
        st.header("🚀 Top Opportunities")
        
        opportunities = st.session_state.opportunities_data
        if not opportunities:
            st.info("No opportunities available. Click 'Refresh Data' to update.")
            return
        
        # Create DataFrame for display
        df = pd.DataFrame(opportunities)
        
        for idx, opp in enumerate(opportunities):
            col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
            
            with col1:
                st.write(f"**{opp['symbol']}**")
            
            with col2:
                # Color code probability
                prob_color = "green" if opp['probability'] > 70 else "orange" if opp['probability'] > 60 else "red"
                st.markdown(f"<span style='color:{prob_color}'>{opp['probability']:.1f}%</span>", unsafe_allow_html=True)
            
            with col3:
                st.write(f"{opp['kelly_fraction']:.2f}")
            
            with col4:
                st.write(f"${opp['recommended_amount']:,.0f}")
            
            with col5:
                if st.button(f"Place Bet", key=f"bet_{idx}"):
                    asyncio.run(self.place_bet(opp['symbol'], opp['recommended_amount']))
                    st.rerun()
    
    def render_active_bets(self):
        """Render active bets monitoring section"""
        st.header("📈 Active Bets")
        
        bets = st.session_state.active_bets_data
        if not bets:
            st.info("No active bets")
            return
        
        df = pd.DataFrame(bets)
        
        # Format the dataframe for display
        display_df = df.copy()
        display_df['Entry Price'] = display_df['entry_price'].apply(lambda x: f"${x:,.2f}")
        display_df['Current Price'] = display_df['current_price'].apply(lambda x: f"${x:,.2f}")
        display_df['Amount'] = display_df['amount'].apply(lambda x: f"${x:,.0f}")
        display_df['P&L'] = display_df.apply(lambda x: f"${x['pnl']:+.2f} ({x['pnl_pct']:+.1f}%)", axis=1)
        display_df['Duration'] = display_df['entry_time'].apply(
            lambda x: str(datetime.now() - x).split('.')[0]
        )
        
        # Show the table
        st.dataframe(
            display_df[['symbol', 'Entry Price', 'Current Price', 'Amount', 'P&L', 'Duration']],
            use_container_width=True
        )
    
    def render_performance_charts(self):
        """Render performance visualization section"""
        st.header("📊 Performance Analysis")
        
        # Mock data for charts - integrate with your actual data
        dates = pd.date_range(start='2024-01-01', end=datetime.now(), freq='D')
        portfolio_values = [10000 + i * 5 + (i % 7 - 3) * 20 for i in range(len(dates))]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=portfolio_values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def run(self):
        """Main dashboard runner"""
        st.set_page_config(
            page_title="Kelly Trading Dashboard",
            page_icon="🎯",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        
        # Initialize data on first load
        if st.session_state.last_update is None:
            asyncio.run(self.refresh_data())
        
        # Render dashboard components
        self.render_header()
        
        # Main content
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


def main():
    """Entry point for the Streamlit dashboard"""
    dashboard = TradingDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()