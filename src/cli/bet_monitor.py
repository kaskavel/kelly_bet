"""
Live bet monitoring CLI interface
Shows current active bets with real-time status updates.
"""

import asyncio
import logging
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass
from ..data.market_data import MarketDataManager
from ..portfolio.manager import PortfolioManager
import yaml


@dataclass
class LiveBetDisplay:
    bet_id: int
    symbol: str
    bet_type: str
    amount: float
    probability: float
    current_price: float
    entry_price: float
    win_threshold: float
    loss_threshold: float
    status: str
    days_active: int
    current_return_pct: float
    potential_win: float
    potential_loss: float
    created_at: str


class BetMonitor:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = logging.getLogger(__name__)
        self.db_path = Path(self.config['database']['sqlite']['path'])
        
        # Initialize market data manager for current prices
        self.market_data = MarketDataManager(self.config)
        
    async def show_live_bets(self, refresh_interval: int = 0):
        """Display live bets with optional auto-refresh"""
        await self.market_data.initialize()
        
        if refresh_interval > 0:
            self.logger.info(f"Starting live bet monitor with {refresh_interval}s refresh")
            try:
                while True:
                    await self._display_live_bets()
                    print(f"\n--- Refreshing in {refresh_interval} seconds (Ctrl+C to stop) ---")
                    await asyncio.sleep(refresh_interval)
            except KeyboardInterrupt:
                print("\nLive monitoring stopped.")
        else:
            await self._display_live_bets()
    
    async def _display_live_bets(self):
        """Display current live bets"""
        live_bets = await self._get_live_bets()
        
        if not live_bets:
            print("\nLIVE BETS MONITOR")
            print("=" * 60)
            print("No active bets found.")
            return
        
        # Get current market prices
        symbols = list(set(bet.symbol for bet in live_bets))
        current_prices = await self._get_current_prices(symbols)
        
        # Update bet displays with current prices
        for bet in live_bets:
            if bet.symbol in current_prices:
                bet.current_price = current_prices[bet.symbol]
                bet.current_return_pct = ((bet.current_price - bet.entry_price) / bet.entry_price) * 100
        
        # Display header
        print(f"\nLIVE BETS MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 120)
        
        # Summary stats
        total_amount = sum(bet.amount for bet in live_bets)
        total_current_value = sum(bet.amount * (1 + bet.current_return_pct/100) for bet in live_bets)
        unrealized_pnl = total_current_value - total_amount
        
        print(f"Active Bets: {len(live_bets)} | Total Amount: ${total_amount:,.2f} | "
              f"Current Value: ${total_current_value:,.2f} | Unrealized P&L: ${unrealized_pnl:+,.2f}")
        print()
        
        # Detailed bet table
        self._print_bet_table(live_bets)
        
        # Risk warnings
        await self._show_risk_warnings(live_bets)
    
    def _print_bet_table(self, live_bets: List[LiveBetDisplay]):
        """Print formatted table of live bets"""
        header = (
            f"{'ID':<4} {'Symbol':<8} {'Type':<5} {'Amount':<10} {'Prob':<5} "
            f"{'Entry':<8} {'Current':<8} {'Return%':<8} {'Days':<5} "
            f"{'Win@':<8} {'Loss@':<8} {'Status':<10}"
        )
        print(header)
        print("-" * len(header))
        
        for bet in sorted(live_bets, key=lambda x: x.days_active, reverse=True):
            # Status indicators for returns
            return_indicator = ""
            if bet.current_return_pct > 5:
                return_indicator = "+"  # Positive returns
            elif bet.current_return_pct < -5:
                return_indicator = "-"  # Negative returns
            else:
                return_indicator = "="  # Neutral returns
            
            # Status indicator
            status_indicator = ""
            if bet.days_active > 7:
                status_indicator = "!"  # Warning for old bets
            elif abs(bet.current_return_pct) > 10:
                status_indicator = "*"  # Star for significant moves
            
            row = (
                f"{bet.bet_id:<4} {bet.symbol:<8} {bet.bet_type:<5} "
                f"${bet.amount:<9.2f} {bet.probability:<5.0f} "
                f"${bet.entry_price:<7.2f} ${bet.current_price:<7.2f} "
                f"{return_indicator}{bet.current_return_pct:<+7.1f}% {bet.days_active:<5} "
                f"${bet.win_threshold:<7.2f} ${bet.loss_threshold:<7.2f} "
                f"{status_indicator}{bet.status:<9}"
            )
            print(row)
    
    async def _show_risk_warnings(self, live_bets: List[LiveBetDisplay]):
        """Show risk warnings for concerning bets"""
        warnings = []
        
        # Check for old bets
        old_bets = [bet for bet in live_bets if bet.days_active > 10]
        if old_bets:
            warnings.append(f"WARNING: {len(old_bets)} bets are older than 10 days")
        
        # Check for large losses
        losing_bets = [bet for bet in live_bets if bet.current_return_pct < -10]
        if losing_bets:
            total_loss = sum(bet.amount * abs(bet.current_return_pct/100) for bet in losing_bets)
            warnings.append(f"ALERT: {len(losing_bets)} bets with >10% losses (${total_loss:.2f} unrealized loss)")
        
        # Check concentration risk
        symbol_counts = {}
        for bet in live_bets:
            symbol_counts[bet.symbol] = symbol_counts.get(bet.symbol, 0) + 1
        
        concentrated_symbols = [symbol for symbol, count in symbol_counts.items() if count > 3]
        if concentrated_symbols:
            warnings.append(f"WARNING: High concentration in: {', '.join(concentrated_symbols)}")
        
        if warnings:
            print("\nRISK WARNINGS:")
            for warning in warnings:
                print(f"   {warning}")
    
    async def _get_live_bets(self) -> List[LiveBetDisplay]:
        """Fetch current live bets from database"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            query = """
            SELECT 
                b.bet_id,
                b.symbol,
                b.bet_type,
                b.amount,
                b.probability,
                b.entry_price,
                b.win_threshold,
                b.loss_threshold,
                b.status,
                b.created_at,
                b.entry_fee,
                JULIANDAY('now') - JULIANDAY(b.created_at) as days_active
            FROM bets b
            WHERE b.status = 'alive'
            ORDER BY b.created_at DESC
            """
            
            df = pd.read_sql_query(query, conn)
            
            if df.empty:
                return []
            
            live_bets = []
            for _, row in df.iterrows():
                # Calculate potential win/loss
                if row['bet_type'] == 'long':
                    potential_win = (row['win_threshold'] - row['entry_price']) / row['entry_price'] * row['amount']
                    potential_loss = (row['entry_price'] - row['loss_threshold']) / row['entry_price'] * row['amount']
                else:  # short
                    potential_win = (row['entry_price'] - row['win_threshold']) / row['entry_price'] * row['amount']
                    potential_loss = (row['loss_threshold'] - row['entry_price']) / row['entry_price'] * row['amount']
                
                bet_display = LiveBetDisplay(
                    bet_id=int(row['bet_id']),
                    symbol=row['symbol'],
                    bet_type=row['bet_type'],
                    amount=float(row['amount']),
                    probability=float(row['probability']),
                    current_price=float(row['entry_price']),  # Will be updated with real price
                    entry_price=float(row['entry_price']),
                    win_threshold=float(row['win_threshold']),
                    loss_threshold=float(row['loss_threshold']),
                    status=row['status'],
                    days_active=int(row['days_active']),
                    current_return_pct=0.0,  # Will be calculated
                    potential_win=potential_win,
                    potential_loss=potential_loss,
                    created_at=row['created_at']
                )
                live_bets.append(bet_display)
            
            return live_bets
            
        except Exception as e:
            self.logger.error(f"Error fetching live bets: {e}")
            return []
        finally:
            conn.close()
    
    async def _get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current market prices for symbols"""
        current_prices = {}
        
        try:
            # Get recent data (last few data points) for current prices
            for symbol in symbols:
                try:
                    recent_data = await self.market_data.get_stock_data(symbol, days=2)
                    if not recent_data.empty:
                        current_prices[symbol] = float(recent_data['Close'].iloc[-1])
                    else:
                        self.logger.warning(f"No recent data for {symbol}")
                except Exception as e:
                    self.logger.error(f"Error getting current price for {symbol}: {e}")
            
        except Exception as e:
            self.logger.error(f"Error fetching current prices: {e}")
        
        return current_prices