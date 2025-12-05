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
from ..utils.currency_converter import CurrencyConverter
import yaml


@dataclass
class LiveBetDisplay:
    bet_id: str
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
        
        # Initialize portfolio manager for performance tracking
        self.portfolio_manager = None
        
    async def show_live_bets(self, refresh_interval: int = 0):
        """Display live bets with optional auto-refresh and automatic settlement"""
        await self.market_data.initialize()
        
        # Initialize portfolio manager for performance tracking
        if not self.portfolio_manager:
            self.portfolio_manager = PortfolioManager(self.config)
            await self.portfolio_manager.initialize()
        
        # Default to 5-minute monitoring if no interval specified
        if refresh_interval == 0:
            refresh_interval = 300  # 5 minutes
            
        print(f"Starting live bet monitor with {refresh_interval//60}-minute refresh and auto-settlement")
        print("Press Ctrl+C to stop monitoring\n")
        
        try:
            while True:
                # FIRST: Check and settle any positions that hit thresholds
                await self._monitor_and_settle_positions()
                
                # THEN: Display current live bets
                await self._display_live_bets()
                
                if refresh_interval > 0:
                    print(f"\n--- Next check in {refresh_interval//60} minutes ({refresh_interval} seconds) - Ctrl+C to stop ---")
                    await asyncio.sleep(refresh_interval)
                else:
                    break
                    
        except KeyboardInterrupt:
            print("\nLive monitoring stopped.")
            await self._cleanup()
    
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
        
        # Summary stats - use PortfolioManager's authoritative calculation
        if self.portfolio_manager:
            portfolio_summary = await self.portfolio_manager.get_portfolio_summary()
            total_amount = portfolio_summary.total_invested
            total_current_value = portfolio_summary.active_bets_value
            unrealized_pnl = portfolio_summary.unrealized_pnl
        else:
            # Fallback to local calculation if portfolio manager not available
            total_amount = sum(bet.amount for bet in live_bets)
            total_current_value = sum(bet.amount * (1 + bet.current_return_pct/100) for bet in live_bets)
            unrealized_pnl = total_current_value - total_amount

        print(f"Active Bets: {len(live_bets)} | Total Amount: ${total_amount:,.2f} | "
              f"Current Value: ${total_current_value:,.2f} | Unrealized P&L: ${unrealized_pnl:+,.2f}")
        print()
        
        # Detailed bet table
        self._print_bet_table(live_bets)
        
        # Portfolio performance summary
        await self._show_portfolio_performance()
        
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
                'long' as bet_type,
                b.amount,
                b.probability_when_placed as probability,
                b.entry_price,
                b.win_price as win_threshold,
                b.loss_price as loss_threshold,
                b.status,
                b.created_at,
                0.0 as entry_fee,
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
                    bet_id=row['bet_id'][:8],  # Show first 8 chars of UUID
                    symbol=row['symbol'],
                    bet_type=row['bet_type'],
                    amount=float(row['amount']),
                    probability=float(row['probability']) if row['probability'] else 0.0,
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
        """Get current market prices for symbols (converted to USD)"""
        current_prices = {}

        try:
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

                        # Detect currency from symbol
                        if symbol.endswith('.T'):  # Japan
                            currency = 'JPY'
                        elif symbol.endswith('.HK'):  # Hong Kong
                            currency = 'HKD'
                        elif symbol.endswith(('.SS', '.SZ')):  # China mainland
                            currency = 'CNY'
                        elif symbol.endswith('.DE'):  # Germany
                            currency = 'EUR'
                        elif symbol.endswith('.L'):  # UK
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
                    else:
                        self.logger.warning(f"No recent data for {symbol}")
                except Exception as e:
                    self.logger.error(f"Error getting current price for {symbol}: {e}")

        except Exception as e:
            self.logger.error(f"Error fetching current prices: {e}")

        return current_prices
    
    async def _monitor_and_settle_positions(self):
        """Monitor existing alive bets and close positions that hit win/loss thresholds"""
        try:
            # Get all alive bets using the same query as display
            alive_bets = await self._get_live_bets()
            
            if not alive_bets:
                return
            
            self.logger.info(f"Monitoring {len(alive_bets)} positions for threshold triggers...")
            
            # Get current prices for all symbols
            symbols_to_check = list(set(bet.symbol for bet in alive_bets))
            current_prices = await self._get_current_prices(symbols_to_check)
            
            # Track settlements for reporting
            settlements = []
            
            # Check each bet against thresholds
            for bet in alive_bets:
                if bet.symbol not in current_prices:
                    self.logger.warning(f"Skipping {bet.symbol} - no current price available")
                    continue
                
                current_price = current_prices[bet.symbol]
                
                # Calculate current return
                current_return_pct = ((current_price - bet.entry_price) / bet.entry_price) * 100
                
                # Check thresholds (assuming long positions)  
                hit_win_threshold = current_price >= bet.win_threshold
                hit_loss_threshold = current_price <= bet.loss_threshold
                
                self.logger.debug(f"Checking {bet.symbol}: Entry=${bet.entry_price:.2f}, "
                                f"Current=${current_price:.2f}, Return={current_return_pct:+.2f}%")
                
                # Determine if we need to close the position
                should_close = False
                close_reason = ""
                
                if hit_win_threshold:
                    should_close = True
                    close_reason = f"WIN THRESHOLD HIT: {current_return_pct:+.2f}% (target: {((bet.win_threshold/bet.entry_price - 1) * 100):+.1f}%)"
                elif hit_loss_threshold:
                    should_close = True
                    close_reason = f"LOSS THRESHOLD HIT: {current_return_pct:+.2f}% (stop: {((bet.loss_threshold/bet.entry_price - 1) * 100):+.1f}%)"
                
                if should_close:
                    self.logger.info(f"SETTLING POSITION - {bet.symbol}: {close_reason}")
                    
                    try:
                        # We need to close the bet through the portfolio manager
                        # Import and initialize it temporarily
                        from ..portfolio.manager import PortfolioManager
                        portfolio = PortfolioManager(self.config)
                        await portfolio.initialize()
                        
                        # Get the full bet_id from the database (we only have first 8 chars in display)
                        full_bet_id = await self._get_full_bet_id(bet.bet_id)
                        if full_bet_id:
                            await portfolio.close_bet(full_bet_id, current_price, close_reason)
                            
                            settlement_info = {
                                'symbol': bet.symbol,
                                'reason': close_reason,
                                'entry_price': bet.entry_price,
                                'exit_price': current_price,
                                'amount': bet.amount,
                                'return_pct': current_return_pct
                            }
                            settlements.append(settlement_info)
                        else:
                            self.logger.error(f"Could not find full bet ID for {bet.bet_id}")
                        
                        await portfolio.cleanup()
                        
                    except Exception as e:
                        self.logger.error(f"Error settling position {bet.symbol}: {e}")
            
            # Report any settlements
            if settlements:
                print(f"\n{'='*80}")
                print(f"POSITIONS SETTLED ({len(settlements)})")
                print(f"{'='*80}")
                
                for settlement in settlements:
                    print(f"CLOSED: {settlement['symbol']}")
                    print(f"  Reason: {settlement['reason']}")
                    print(f"  Entry: ${settlement['entry_price']:.2f} -> Exit: ${settlement['exit_price']:.2f}")
                    print(f"  Amount: ${settlement['amount']:.2f} | Return: {settlement['return_pct']:+.2f}%")
                    
                print(f"{'='*80}")
            else:
                self.logger.debug("No positions required settlement at this time")
                
        except Exception as e:
            self.logger.error(f"Error in position monitoring: {e}")
    
    async def _get_full_bet_id(self, short_bet_id: str) -> str:
        """Get the full bet ID from the short display ID"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT bet_id FROM bets WHERE bet_id LIKE ? AND status = ?', 
                         (f'{short_bet_id}%', 'alive'))
            result = cursor.fetchone()
            return result[0] if result else None
        except Exception as e:
            self.logger.error(f"Error getting full bet ID: {e}")
            return None
        finally:
            conn.close()
    
    async def _show_portfolio_performance(self):
        """Display portfolio performance summary"""
        if not self.portfolio_manager:
            return
            
        try:
            # Refresh portfolio state to get latest cash balance and active bets
            await self.portfolio_manager._load_portfolio_state()
            
            # Get portfolio summary
            portfolio = await self.portfolio_manager.get_portfolio_summary()
            
            # Calculate trading performance correctly
            initial_capital = self.portfolio_manager.initial_capital
            
            # Total P&L should be realized + unrealized (actual trading results)
            total_trading_pnl = portfolio.realized_pnl + portfolio.unrealized_pnl
            trading_return_pct = (total_trading_pnl / initial_capital) * 100
            
            # Portfolio value comparison (for reference, but not the main metric)
            portfolio_value_change = portfolio.total_capital - initial_capital
            portfolio_change_pct = (portfolio_value_change / initial_capital) * 100
            
            print(f"\n{'='*60}")
            print("PORTFOLIO PERFORMANCE")
            print(f"{'='*60}")
            print(f"Initial Capital: ${initial_capital:,.2f}")
            print(f"Current Value:   ${portfolio.total_capital:,.2f}")
            print(f"Cash Balance:    ${portfolio.cash_balance:,.2f}")
            print(f"Active Positions: ${portfolio.active_bets_value:,.2f}")
            print(f"")
            print(f"TRADING PERFORMANCE:")
            print(f"Realized P&L:    ${portfolio.realized_pnl:+,.2f}")
            print(f"Unrealized P&L:  ${portfolio.unrealized_pnl:+,.2f}")
            print(f"Total Trading P&L: ${total_trading_pnl:+,.2f} ({trading_return_pct:+.2f}%)")
            print(f"")
            print(f"Portfolio Change: ${portfolio_value_change:+,.2f} ({portfolio_change_pct:+.2f}%)")
            
        except Exception as e:
            self.logger.error(f"Error showing portfolio performance: {e}")
    
    async def _cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'market_data'):
            await self.market_data.cleanup()
        if hasattr(self, 'portfolio_manager') and self.portfolio_manager:
            await self.portfolio_manager.cleanup()
        self.logger.info("Bet monitor cleanup complete")