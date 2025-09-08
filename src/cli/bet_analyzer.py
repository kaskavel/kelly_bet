"""
Bet history analysis CLI interface
Analyzes historical betting performance with detailed statistics and insights.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import yaml


@dataclass
class BetSummary:
    bet_id: str
    symbol: str
    bet_type: str
    amount: float
    probability: float
    entry_price: float
    exit_price: float
    win_threshold: float
    loss_threshold: float
    status: str
    pnl: float
    return_pct: float
    duration_days: int
    entry_fee: float
    exit_fee: float
    total_fees: float
    created_at: str
    closed_at: str


@dataclass
class PerformanceStats:
    total_bets: int
    won_bets: int
    lost_bets: int
    alive_bets: int
    win_rate: float
    total_invested: float
    total_pnl: float
    total_fees: float
    net_return_pct: float
    avg_bet_amount: float
    avg_return_per_bet: float
    avg_duration_days: float
    best_bet_return: float
    worst_bet_return: float
    current_drawdown: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float


class BetAnalyzer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.db_path = Path(self.config['database']['sqlite']['path'])
        
        # Initialize portfolio manager for performance tracking
        self.portfolio_manager = None
    
    async def show_bet_history(self, limit: int = 50, status_filter: str = 'all', show_stats: bool = False):
        """Display bet history with optional filtering and statistics"""
        
        # Initialize portfolio manager for performance tracking
        if not self.portfolio_manager:
            from ..portfolio.manager import PortfolioManager
            self.portfolio_manager = PortfolioManager(self.config)
            await self.portfolio_manager.initialize()
        
        # Get bet data
        bet_history = await self._get_bet_history(limit, status_filter)
        
        if not bet_history:
            print(f"\nBET HISTORY ANALYSIS")
            print("=" * 80)
            print("No bets found matching the criteria.")
            return
        
        # Display header
        print(f"\nBET HISTORY ANALYSIS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 120)
        
        # Show portfolio performance
        await self._show_portfolio_performance()
        
        if show_stats:
            # Calculate and display comprehensive statistics
            stats = self._calculate_performance_stats(bet_history)
            self._display_performance_stats(stats)
            print("\n" + "=" * 120)
        
        # Display bet table
        print(f"\nBet History (Showing {len(bet_history)} bets, filter: {status_filter})")
        self._display_bet_table(bet_history)
        
        if show_stats and len(bet_history) > 10:
            print("\n" + "=" * 120)
            self._display_detailed_analysis(bet_history)
    
    def _display_performance_stats(self, stats: PerformanceStats):
        """Display comprehensive performance statistics"""
        print("PERFORMANCE SUMMARY")
        print("-" * 50)
        
        # Basic metrics
        print(f"Total Bets: {stats.total_bets} | Won: {stats.won_bets} | Lost: {stats.lost_bets} | Active: {stats.alive_bets}")
        print(f"Win Rate: {stats.win_rate:.1f}% | Net Return: {stats.net_return_pct:+.1f}%")
        print(f"Total Invested: ${stats.total_invested:,.2f} | P&L: ${stats.total_pnl:+,.2f} | Fees: ${stats.total_fees:,.2f}")
        
        # Advanced metrics
        print(f"\nAvg Bet: ${stats.avg_bet_amount:,.2f} | Avg Return/Bet: {stats.avg_return_per_bet:+.1f}%")
        print(f"Avg Duration: {stats.avg_duration_days:.1f} days")
        print(f"Best Bet: {stats.best_bet_return:+.1f}% | Worst Bet: {stats.worst_bet_return:+.1f}%")
        
        # Risk metrics
        if stats.sharpe_ratio is not None:
            print(f"Sharpe Ratio: {stats.sharpe_ratio:.2f} | Profit Factor: {stats.profit_factor:.2f}")
        print(f"Max Drawdown: {stats.max_drawdown:.1f}% | Current Drawdown: {stats.current_drawdown:.1f}%")
        
        # Performance assessment
        self._display_performance_assessment(stats)
    
    def _display_performance_assessment(self, stats: PerformanceStats):
        """Display qualitative assessment of performance"""
        print(f"\nPERFORMANCE ASSESSMENT")
        print("-" * 50)
        
        assessments = []
        
        # Win rate assessment
        if stats.win_rate >= 60:
            assessments.append("+ Strong win rate")
        elif stats.win_rate >= 50:
            assessments.append("= Decent win rate")
        else:
            assessments.append("- Low win rate - review strategy")
        
        # Return assessment
        if stats.net_return_pct >= 15:
            assessments.append("+ Excellent returns")
        elif stats.net_return_pct >= 5:
            assessments.append("= Solid returns")
        elif stats.net_return_pct >= 0:
            assessments.append("= Break-even performance")
        else:
            assessments.append("- Losing money - strategy needs work")
        
        # Risk assessment
        if stats.max_drawdown <= 10:
            assessments.append("+ Low drawdown - good risk control")
        elif stats.max_drawdown <= 20:
            assessments.append("= Moderate drawdown")
        else:
            assessments.append("- High drawdown - risky strategy")
        
        # Sharpe ratio assessment
        if stats.sharpe_ratio is not None:
            if stats.sharpe_ratio >= 1.5:
                assessments.append("+ Excellent risk-adjusted returns")
            elif stats.sharpe_ratio >= 1.0:
                assessments.append("= Good risk-adjusted returns")
            elif stats.sharpe_ratio >= 0.5:
                assessments.append("= Decent risk-adjusted returns")
            else:
                assessments.append("- Poor risk-adjusted returns")
        
        for assessment in assessments:
            print(f"   {assessment}")
    
    def _display_bet_table(self, bet_history: List[BetSummary]):
        """Display formatted table of bet history"""
        header = (
            f"{'ID':<4} {'Symbol':<8} {'Type':<5} {'Amount':<10} {'Prob':<5} "
            f"{'Entry':<8} {'Exit':<8} {'Return%':<8} {'P&L':<10} "
            f"{'Days':<5} {'Status':<7} {'Date':<12}"
        )
        print(header)
        print("-" * len(header))
        
        for bet in bet_history:
            # Status indicators
            if bet.status == 'won':
                status_symbol = "+"
            elif bet.status == 'lost':
                status_symbol = "-"
            else:  # alive
                status_symbol = "="
            
            # Return color
            if bet.return_pct > 0:
                return_color = ""
            else:
                return_color = ""
            
            exit_price_str = f"${bet.exit_price:.2f}" if bet.exit_price > 0 else "N/A"
            date_str = bet.created_at[:10] if bet.created_at else "N/A"
            
            row = (
                f"{bet.bet_id:<4} {bet.symbol:<8} {bet.bet_type:<5} "
                f"${bet.amount:<9.2f} {bet.probability:<5.0f} "
                f"${bet.entry_price:<7.2f} {exit_price_str:<8} "
                f"{return_color}{bet.return_pct:<+7.1f}% ${bet.pnl:<+9.2f} "
                f"{bet.duration_days:<5} {status_symbol}{bet.status:<6} {date_str:<12}"
            )
            print(row)
    
    def _display_detailed_analysis(self, bet_history: List[BetSummary]):
        """Display detailed analysis insights"""
        print("DETAILED ANALYSIS")
        print("-" * 50)
        
        # Symbol performance
        symbol_stats = self._analyze_by_symbol(bet_history)
        if symbol_stats:
            print("Top performing symbols:")
            for i, (symbol, stats) in enumerate(symbol_stats[:5], 1):
                print(f"   {i}. {symbol}: {stats['return']:.1f}% ({stats['count']} bets, {stats['win_rate']:.0f}% win rate)")
        
        # Bet type performance
        type_stats = self._analyze_by_bet_type(bet_history)
        if type_stats:
            print(f"\nBet type performance:")
            for bet_type, stats in type_stats.items():
                print(f"   {bet_type.upper()}: {stats['return']:.1f}% ({stats['count']} bets, {stats['win_rate']:.0f}% win rate)")
        
        # Time-based patterns
        monthly_stats = self._analyze_by_month(bet_history)
        if len(monthly_stats) > 1:
            print(f"\nMonthly performance (last 6 months):")
            for month, stats in list(monthly_stats.items())[-6:]:
                print(f"   {month}: {stats['return']:.1f}% ({stats['count']} bets)")
        
        # Duration analysis
        duration_stats = self._analyze_by_duration(bet_history)
        if duration_stats:
            print(f"\nBest performing duration ranges:")
            for duration_range, stats in duration_stats.items():
                if stats['count'] >= 3:  # Only show ranges with sufficient data
                    print(f"   {duration_range}: {stats['return']:.1f}% avg return ({stats['count']} bets)")
    
    def _analyze_by_symbol(self, bet_history: List[BetSummary]) -> List[Tuple[str, Dict]]:
        """Analyze performance by symbol"""
        symbol_stats = {}
        
        for bet in bet_history:
            if bet.status in ['won', 'lost']:  # Only closed bets
                if bet.symbol not in symbol_stats:
                    symbol_stats[bet.symbol] = {'returns': [], 'won': 0, 'total': 0}
                
                symbol_stats[bet.symbol]['returns'].append(bet.return_pct)
                symbol_stats[bet.symbol]['total'] += 1
                if bet.status == 'won':
                    symbol_stats[bet.symbol]['won'] += 1
        
        # Calculate aggregated stats
        result = []
        for symbol, stats in symbol_stats.items():
            if stats['total'] >= 2:  # At least 2 bets
                avg_return = np.mean(stats['returns'])
                win_rate = (stats['won'] / stats['total']) * 100
                result.append((symbol, {
                    'return': avg_return,
                    'win_rate': win_rate,
                    'count': stats['total']
                }))
        
        return sorted(result, key=lambda x: x[1]['return'], reverse=True)
    
    def _analyze_by_bet_type(self, bet_history: List[BetSummary]) -> Dict[str, Dict]:
        """Analyze performance by bet type (long/short)"""
        type_stats = {}
        
        for bet in bet_history:
            if bet.status in ['won', 'lost']:
                if bet.bet_type not in type_stats:
                    type_stats[bet.bet_type] = {'returns': [], 'won': 0, 'total': 0}
                
                type_stats[bet.bet_type]['returns'].append(bet.return_pct)
                type_stats[bet.bet_type]['total'] += 1
                if bet.status == 'won':
                    type_stats[bet.bet_type]['won'] += 1
        
        # Calculate aggregated stats
        result = {}
        for bet_type, stats in type_stats.items():
            if stats['total'] > 0:
                result[bet_type] = {
                    'return': np.mean(stats['returns']),
                    'win_rate': (stats['won'] / stats['total']) * 100,
                    'count': stats['total']
                }
        
        return result
    
    def _analyze_by_month(self, bet_history: List[BetSummary]) -> Dict[str, Dict]:
        """Analyze performance by month"""
        monthly_stats = {}
        
        for bet in bet_history:
            if bet.status in ['won', 'lost'] and bet.created_at:
                try:
                    month_key = bet.created_at[:7]  # YYYY-MM format
                    
                    if month_key not in monthly_stats:
                        monthly_stats[month_key] = {'returns': [], 'count': 0}
                    
                    monthly_stats[month_key]['returns'].append(bet.return_pct)
                    monthly_stats[month_key]['count'] += 1
                except:
                    continue
        
        # Calculate averages
        result = {}
        for month, stats in monthly_stats.items():
            if stats['count'] > 0:
                result[month] = {
                    'return': np.mean(stats['returns']),
                    'count': stats['count']
                }
        
        return dict(sorted(result.items()))
    
    def _analyze_by_duration(self, bet_history: List[BetSummary]) -> Dict[str, Dict]:
        """Analyze performance by bet duration"""
        duration_ranges = {
            '0-2 days': (0, 2),
            '3-7 days': (3, 7),
            '8-14 days': (8, 14),
            '15+ days': (15, 1000)
        }
        
        range_stats = {range_name: {'returns': [], 'count': 0} for range_name in duration_ranges}
        
        for bet in bet_history:
            if bet.status in ['won', 'lost']:
                for range_name, (min_days, max_days) in duration_ranges.items():
                    if min_days <= bet.duration_days <= max_days:
                        range_stats[range_name]['returns'].append(bet.return_pct)
                        range_stats[range_name]['count'] += 1
                        break
        
        # Calculate averages
        result = {}
        for range_name, stats in range_stats.items():
            if stats['count'] > 0:
                result[range_name] = {
                    'return': np.mean(stats['returns']),
                    'count': stats['count']
                }
        
        return result
    
    def _calculate_performance_stats(self, bet_history: List[BetSummary]) -> PerformanceStats:
        """Calculate comprehensive performance statistics"""
        if not bet_history:
            return PerformanceStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # Basic counts
        total_bets = len(bet_history)
        won_bets = len([b for b in bet_history if b.status == 'won'])
        lost_bets = len([b for b in bet_history if b.status == 'lost'])
        alive_bets = len([b for b in bet_history if b.status == 'alive'])
        
        # Financial metrics
        total_invested = sum(b.amount for b in bet_history)
        total_pnl = sum(b.pnl for b in bet_history if b.status in ['won', 'lost'])
        total_fees = sum(b.total_fees for b in bet_history)
        
        # Performance metrics
        win_rate = (won_bets / max(1, won_bets + lost_bets)) * 100
        net_return_pct = (total_pnl / max(1, total_invested)) * 100
        
        # Averages
        avg_bet_amount = total_invested / max(1, total_bets)
        closed_bets = [b for b in bet_history if b.status in ['won', 'lost']]
        avg_return_per_bet = np.mean([b.return_pct for b in closed_bets]) if closed_bets else 0
        avg_duration_days = np.mean([b.duration_days for b in closed_bets]) if closed_bets else 0
        
        # Extremes
        returns = [b.return_pct for b in closed_bets]
        best_bet_return = max(returns) if returns else 0
        worst_bet_return = min(returns) if returns else 0
        
        # Risk metrics
        current_drawdown, max_drawdown = self._calculate_drawdown(bet_history)
        sharpe_ratio = self._calculate_sharpe_ratio(closed_bets)
        profit_factor = self._calculate_profit_factor(closed_bets)
        
        return PerformanceStats(
            total_bets=total_bets,
            won_bets=won_bets,
            lost_bets=lost_bets,
            alive_bets=alive_bets,
            win_rate=win_rate,
            total_invested=total_invested,
            total_pnl=total_pnl,
            total_fees=total_fees,
            net_return_pct=net_return_pct,
            avg_bet_amount=avg_bet_amount,
            avg_return_per_bet=avg_return_per_bet,
            avg_duration_days=avg_duration_days,
            best_bet_return=best_bet_return,
            worst_bet_return=worst_bet_return,
            current_drawdown=current_drawdown,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor
        )
    
    def _calculate_drawdown(self, bet_history: List[BetSummary]) -> Tuple[float, float]:
        """Calculate current and maximum drawdown"""
        if not bet_history:
            return 0.0, 0.0
        
        # Sort by date
        sorted_bets = sorted([b for b in bet_history if b.status in ['won', 'lost'] and b.created_at], 
                           key=lambda x: x.created_at)
        
        if not sorted_bets:
            return 0.0, 0.0
        
        # Calculate cumulative returns
        cumulative_pnl = 0
        peak_pnl = 0
        max_drawdown = 0
        
        for bet in sorted_bets:
            cumulative_pnl += bet.pnl
            peak_pnl = max(peak_pnl, cumulative_pnl)
            drawdown = ((peak_pnl - cumulative_pnl) / max(1, abs(peak_pnl))) * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        # Current drawdown
        current_drawdown = ((peak_pnl - cumulative_pnl) / max(1, abs(peak_pnl))) * 100
        
        return current_drawdown, max_drawdown
    
    def _calculate_sharpe_ratio(self, closed_bets: List[BetSummary]) -> Optional[float]:
        """Calculate Sharpe ratio for the betting strategy"""
        if len(closed_bets) < 10:  # Need sufficient data
            return None
        
        returns = [b.return_pct / 100 for b in closed_bets]  # Convert to decimal
        
        if not returns:
            return None
        
        avg_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        if std_return == 0:
            return None
        
        # Assuming risk-free rate of 2% annually, convert to per-bet basis
        # This is very rough - ideally we'd use actual duration
        risk_free_rate = 0.02 / (365 / 7)  # Assume 7-day average bet duration
        
        sharpe_ratio = (avg_return - risk_free_rate) / std_return
        return sharpe_ratio
    
    def _calculate_profit_factor(self, closed_bets: List[BetSummary]) -> float:
        """Calculate profit factor (total profits / total losses)"""
        total_profits = sum(b.pnl for b in closed_bets if b.pnl > 0)
        total_losses = abs(sum(b.pnl for b in closed_bets if b.pnl < 0))
        
        if total_losses == 0:
            return float('inf') if total_profits > 0 else 0
        
        return total_profits / total_losses
    
    async def _get_bet_history(self, limit: int, status_filter: str) -> List[BetSummary]:
        """Fetch bet history from database"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Build query with filter
            where_clause = ""
            if status_filter != 'all':
                where_clause = f"WHERE b.status = '{status_filter}'"
            
            query = f"""
            SELECT 
                b.bet_id,
                b.symbol,
                'long' as bet_type,
                b.amount,
                b.probability_when_placed as probability,
                b.entry_price,
                b.exit_price,
                b.win_threshold,
                b.loss_threshold,
                b.status,
                b.realized_pnl as pnl,
                0.0 as entry_fee,
                0.0 as exit_fee,
                b.created_at,
                b.exit_time as closed_at,
                CASE 
                    WHEN b.exit_time IS NOT NULL AND b.created_at IS NOT NULL 
                    THEN JULIANDAY(b.exit_time) - JULIANDAY(b.created_at)
                    WHEN b.status = 'alive' AND b.created_at IS NOT NULL
                    THEN JULIANDAY('now') - JULIANDAY(b.created_at)
                    ELSE 0 
                END as duration_days
            FROM bets b
            {where_clause}
            ORDER BY b.created_at DESC
            LIMIT {limit}
            """
            
            df = pd.read_sql_query(query, conn)
            
            if df.empty:
                return []
            
            bet_summaries = []
            for _, row in df.iterrows():
                # Calculate return percentage
                if row['exit_price'] and row['exit_price'] > 0 and row['entry_price'] > 0:
                    if row['bet_type'] == 'long':
                        return_pct = ((row['exit_price'] - row['entry_price']) / row['entry_price']) * 100
                    else:  # short
                        return_pct = ((row['entry_price'] - row['exit_price']) / row['entry_price']) * 100
                else:
                    return_pct = 0.0
                
                # Total fees
                total_fees = (row['entry_fee'] or 0) + (row['exit_fee'] or 0)
                
                bet_summary = BetSummary(
                    bet_id=row['bet_id'][:8],  # Show first 8 chars of UUID for display
                    symbol=row['symbol'],
                    bet_type=row['bet_type'],
                    amount=float(row['amount']),
                    probability=float(row['probability']) if row['probability'] else 0.0,
                    entry_price=float(row['entry_price']),
                    exit_price=float(row['exit_price']) if row['exit_price'] else 0.0,
                    win_threshold=float(row['win_threshold']),
                    loss_threshold=float(row['loss_threshold']),
                    status=row['status'],
                    pnl=float(row['pnl']) if row['pnl'] is not None else 0.0,
                    return_pct=return_pct,
                    duration_days=int(row['duration_days']),
                    entry_fee=float(row['entry_fee']) if row['entry_fee'] else 0.0,
                    exit_fee=float(row['exit_fee']) if row['exit_fee'] else 0.0,
                    total_fees=total_fees,
                    created_at=row['created_at'] or '',
                    closed_at=row['closed_at'] or ''
                )
                bet_summaries.append(bet_summary)
            
            return bet_summaries
            
        except Exception as e:
            print(f"Error fetching bet history: {e}")
            return []
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
            print(f"Error showing portfolio performance: {e}")