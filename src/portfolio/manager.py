"""
Portfolio Manager
Manages cash balance, active bets, and portfolio tracking.
"""

import asyncio
import logging
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum


class BetStatus(Enum):
    ALIVE = "alive"
    WON = "won"
    LOST = "lost"
    CANCELLED = "cancelled"


@dataclass
class Bet:
    """Represents a single bet in the portfolio"""
    bet_id: str
    symbol: str
    asset_type: str           # 'stock' or 'crypto'
    entry_price: float        # ALWAYS in USD (converted if needed)
    entry_time: datetime
    amount: float             # Dollar amount invested
    shares: float             # Number of shares/units
    win_threshold: float      # Percentage gain to trigger win
    loss_threshold: float     # Percentage loss to trigger loss
    win_price: float          # Price at which bet wins (in USD)
    loss_price: float         # Price at which bet loses (in USD)
    current_price: float      # Latest price (in USD)
    current_value: float      # Current market value
    unrealized_pnl: float     # Unrealized P&L
    status: BetStatus
    algorithm_used: str       # Which algorithm generated this bet
    probability_when_placed: float  # Probability when bet was placed
    currency: str = 'USD'     # Original currency (for reference)
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None  # ALWAYS in USD
    realized_pnl: Optional[float] = None


@dataclass
class PortfolioSnapshot:
    """Snapshot of portfolio state"""
    timestamp: datetime
    total_capital: float      # Cash + market value of all positions
    cash_balance: float       # Available cash
    active_bets_count: int    # Number of active bets
    active_bets_value: float  # Market value of active bets
    total_invested: float     # Total amount invested in active bets
    unrealized_pnl: float     # Total unrealized P&L
    realized_pnl: float       # Total realized P&L since inception


class PortfolioManager:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.db_path = Path(config['database']['sqlite']['path'])
        
        # Portfolio state
        self.active_bets: Dict[str, Bet] = {}
        self.initial_capital = config.get('trading', {}).get('initial_capital', 10000.0)
        
        # Portfolio constraints
        self.max_concurrent_bets = config.get('trading', {}).get('max_concurrent_bets', 10)
        self.max_single_bet_fraction = config.get('trading', {}).get('max_bet_fraction', 0.1)  # 10%
        
        # Trading fees
        self.trading_fee_percentage = config.get('trading', {}).get('trading_fee_percentage', 0.25)  # 0.25%
    
    async def initialize(self):
        """Initialize portfolio manager and load existing state"""
        self.logger.info("Initializing portfolio manager...")
        
        # Create database tables
        await self._create_tables()
        
        # Load existing portfolio state
        await self._load_portfolio_state()

        # If no cash balance exists, initialize with starting capital
        cash_balance = await self.get_cash_balance()
        if cash_balance == 0.0:
            await self._record_cash_transaction(
                amount=self.initial_capital,
                description="Initial capital - Fresh start",
                transaction_type='initial_capital'
            )
        
        cash_balance = await self.get_cash_balance()
        self.logger.info(f"Portfolio initialized: ${cash_balance:.2f} cash, "
                        f"{len(self.active_bets)} active bets")

    async def get_cash_balance(self) -> float:
        """
        Get current cash balance from database (single source of truth).
        Calculates from sum of all transactions for accuracy.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Get sum of all cash transactions
            cursor.execute('SELECT SUM(amount) FROM cash_transactions')
            result = cursor.fetchone()
            balance = result[0] if result and result[0] is not None else 0.0
            return balance
        finally:
            conn.close()

    async def _create_tables(self):
        """Create database tables for portfolio management"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enhanced bets table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS bets (
            bet_id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            asset_type TEXT NOT NULL,
            entry_price REAL NOT NULL,
            entry_time TIMESTAMP NOT NULL,
            amount REAL NOT NULL,
            shares REAL NOT NULL,
            win_threshold REAL NOT NULL,
            loss_threshold REAL NOT NULL,
            win_price REAL NOT NULL,
            loss_price REAL NOT NULL,
            current_price REAL,
            status TEXT NOT NULL,
            algorithm_used TEXT,
            probability_when_placed REAL,
            exit_time TIMESTAMP,
            exit_price REAL,
            realized_pnl REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Portfolio history table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolio_history (
            history_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP NOT NULL,
            total_capital REAL NOT NULL,
            cash_balance REAL NOT NULL,
            active_bets_count INTEGER NOT NULL,
            active_bets_value REAL NOT NULL,
            total_invested REAL NOT NULL,
            unrealized_pnl REAL NOT NULL,
            realized_pnl REAL NOT NULL,
            notes TEXT
        )
        ''')
        
        # Cash transactions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS cash_transactions (
            transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            amount REAL NOT NULL,
            balance_after REAL NOT NULL,
            description TEXT NOT NULL,
            bet_id TEXT,
            transaction_type TEXT NOT NULL  -- 'deposit', 'withdrawal', 'bet_entry', 'bet_exit'
        )
        ''')

        # Bet predictions table - stores individual algorithm predictions for each bet
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS bet_predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            bet_id TEXT NOT NULL,
            algorithm TEXT NOT NULL,
            probability REAL NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (bet_id) REFERENCES bets(bet_id)
        )
        ''')
        
        # Additive migrations for columns introduced after the table was first
        # created. `currency` predates this block and is added the same way.
        cursor.execute("PRAGMA table_info(bets)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        for column, ddl in (
            ('currency', "ALTER TABLE bets ADD COLUMN currency TEXT DEFAULT 'USD'"),
            # How the position actually left the book: win_barrier, loss_barrier,
            # time_barrier or manual. Needed to build a reliability curve -- a
            # time-barrier exit is not evidence about the barrier probability.
            ('exit_reason', "ALTER TABLE bets ADD COLUMN exit_reason TEXT"),
            # Signed gap between the barrier price and the actual fill, in percent.
            # Makes execution quality measurable instead of invisible.
            ('exit_slippage_pct', "ALTER TABLE bets ADD COLUMN exit_slippage_pct REAL"),
        ):
            if column not in existing_columns:
                try:
                    cursor.execute(ddl)
                    self.logger.info(f"Added bets.{column} column")
                except sqlite3.OperationalError as e:
                    self.logger.debug(f"Could not add bets.{column}: {e}")

        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_bets_status ON bets(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_bets_symbol ON bets(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_portfolio_history_time ON portfolio_history(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_bet_predictions_bet_id ON bet_predictions(bet_id)')
        
        conn.commit()
        conn.close()
    
    async def _load_portfolio_state(self):
        """Load existing portfolio state from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Load active bets
            cursor.execute('''
            SELECT * FROM bets WHERE status = ?
            ''', (BetStatus.ALIVE.value,))

            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]

            for row in rows:
                bet_data = dict(zip(columns, row))
                bet = self._row_to_bet(bet_data)

                # Calculate current value and unrealized PnL
                bet.current_value = bet.shares * bet.current_price
                bet.unrealized_pnl = bet.current_value - bet.amount

                self.active_bets[bet.bet_id] = bet

            cash_balance = await self.get_cash_balance()
            self.logger.info(f"Loaded portfolio state: ${cash_balance:.2f} cash, "
                           f"{len(self.active_bets)} active bets")

        except Exception as e:
            self.logger.error(f"Error loading portfolio state: {e}")
        finally:
            conn.close()
    
    async def _count_alive_bets(self) -> int:
        """Number of open bets, from the database."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM bets WHERE status = ?',
                           (BetStatus.ALIVE.value,))
            return int(cursor.fetchone()[0])
        finally:
            conn.close()

    @staticmethod
    def _infer_asset_type(symbol: str) -> str:
        """Fallback asset class from the symbol shape, when metadata is absent."""
        if symbol.endswith('=X'):
            return 'forex'
        if symbol.endswith(('-USD', '/USD', 'USDT')):
            return 'crypto'
        return 'stock'

    def _row_to_bet(self, row: Dict) -> Bet:
        """Convert database row to Bet object"""
        return Bet(
            bet_id=row['bet_id'],
            symbol=row['symbol'],
            asset_type=row['asset_type'],
            entry_price=row['entry_price'],
            entry_time=datetime.fromisoformat(row['entry_time']),
            amount=row['amount'],
            shares=row['shares'],
            win_threshold=row['win_threshold'],
            loss_threshold=row['loss_threshold'],
            win_price=row['win_price'],
            loss_price=row['loss_price'],
            current_price=row['current_price'] or row['entry_price'],
            current_value=0.0,  # Will be calculated
            unrealized_pnl=0.0,  # Will be calculated
            status=BetStatus(row['status']),
            algorithm_used=row['algorithm_used'] or 'unknown',
            probability_when_placed=row['probability_when_placed'] or 0.0,
            currency=row.get('currency', 'USD'),  # Original currency
            exit_time=datetime.fromisoformat(row['exit_time']) if row['exit_time'] else None,
            exit_price=row['exit_price'],
            realized_pnl=row['realized_pnl']
        )
    
    async def place_bet(self, prediction: Dict) -> str:
        """
        Place a new bet based on prediction
        
        Args:
            prediction: Prediction dictionary with symbol, probability, current_price, algorithms
            
        Returns:
            Bet ID of the placed bet
        """
        symbol = prediction['symbol']
        probability = prediction['probability']
        current_price = prediction['current_price']  # Already in USD
        currency = prediction.get('currency', 'USD')  # Original currency for reference
        algorithms = prediction.get('algorithms', [])

        # A missing or zero price must fail here, with a message that names the cause.
        # It previously reached `shares = net_bet_amount / current_price` and surfaced
        # as a bare "float division by zero", which says nothing about what went wrong.
        if not current_price or current_price <= 0:
            raise ValueError(
                f"{symbol}: refusing to place a bet at a price of {current_price}. "
                f"The opportunity carries no usable price, so position size and both "
                f"barrier levels would be meaningless."
            )

        self.logger.info(f"Placing bet for {symbol} at ${current_price:.2f} USD "
                        f"(original currency: {currency}) with {probability:.1f}% probability")
        
        # Check if we can place more bets. Counted from the database, not from the
        # in-memory cache, which used to drift upward as bets were closed.
        open_count = await self._count_alive_bets()
        if open_count >= self.max_concurrent_bets:
            raise ValueError(f"Maximum concurrent bets reached ({self.max_concurrent_bets})")
        
        # Barriers come from the PREDICTION, not from config. In volatility mode they
        # are scaled to this asset's own realised vol, and sizing must use exactly the
        # barriers the probability was estimated against -- otherwise Kelly is pricing
        # a different bet than the model forecast.
        win_threshold = prediction.get('win_threshold')
        loss_threshold = prediction.get('loss_threshold')

        if win_threshold is None or loss_threshold is None:
            from ..trading.barriers import BarrierPolicy
            policy = BarrierPolicy(self.config)
            if policy.is_volatility_scaled:
                raise ValueError(
                    f"{symbol}: volatility-scaled barriers are configured but the "
                    f"prediction carries no win/loss thresholds. Refusing to fall "
                    f"back to fixed barriers, which would size a different bet than "
                    f"was forecast."
                )
            win_threshold = policy.fixed_win_pct
            loss_threshold = policy.fixed_loss_pct

        # Import here to avoid circular import
        from ..kelly.calculator import KellyCalculator

        # Get current cash balance from DB
        cash_balance = await self.get_cash_balance()

        # Calculate bet size using Kelly, on THIS asset's barriers
        kelly_calc = KellyCalculator(self.config)
        recommendation = kelly_calc.calculate_bet_size(
            probability=probability,
            current_price=current_price,
            available_capital=cash_balance,
            win_threshold=win_threshold,
            loss_threshold=loss_threshold,
            # Positions already open. Kelly sizes a bet as if it were the only thing
            # at risk; with a book of correlated longs that over-levers everything.
            concurrent_positions=open_count + 1,
        )

        if not recommendation.is_favorable or recommendation.recommended_amount <= 0:
            raise ValueError(f"Kelly calculator recommends no bet: {recommendation.risk_warning}")

        bet_amount = recommendation.recommended_amount

        # Check if we have enough cash
        if bet_amount > cash_balance:
            raise ValueError(f"Insufficient cash: need ${bet_amount:.2f}, have ${cash_balance:.2f}")
        
        # Apply trading fees to bet amount
        trading_fee = bet_amount * (self.trading_fee_percentage / 100)
        net_bet_amount = bet_amount - trading_fee
        
        # Calculate number of shares and prices
        shares = net_bet_amount / current_price
        win_price = current_price * (1 + win_threshold / 100)
        loss_price = current_price * (1 - loss_threshold / 100)
        
        # Create bet
        bet_id = str(uuid.uuid4())
        algorithm_used = algorithms[0]['algorithm'] if algorithms else 'ensemble'
        
        # Asset class comes from the prediction (AssetSelector supplies it). It was
        # hardcoded to 'stock', which mislabelled every commodity ETF and forex pair
        # in the book and broke all per-asset-class reporting.
        asset_type = prediction.get('asset_type') or self._infer_asset_type(symbol)

        bet = Bet(
            bet_id=bet_id,
            symbol=symbol,
            asset_type=asset_type,
            entry_price=current_price,  # Already in USD
            entry_time=datetime.now(),
            amount=net_bet_amount,
            shares=shares,
            win_threshold=win_threshold,
            loss_threshold=loss_threshold,
            win_price=win_price,  # In USD
            loss_price=loss_price,  # In USD
            current_price=current_price,  # In USD
            current_value=net_bet_amount,
            unrealized_pnl=0.0,
            status=BetStatus.ALIVE,
            algorithm_used=algorithm_used,
            probability_when_placed=probability,
            currency=currency  # Store original currency for reference
        )
        
        # Store bet
        await self._store_bet(bet)

        # Store individual algorithm predictions for this bet
        await self._store_bet_predictions(bet_id, algorithms)

        # Record cash transaction (negative amount = cash leaving)
        await self._record_cash_transaction(
            amount=-bet_amount,
            description=f"Bet entry: {symbol}",
            bet_id=bet_id,
            transaction_type='bet_entry'
        )

        # Add to active bets
        self.active_bets[bet_id] = bet

        # Record portfolio snapshot
        await self._record_portfolio_snapshot(f"Placed bet: {symbol}")
        
        self.logger.info(f"Bet placed: {bet_id} - {symbol} ${bet_amount:.2f} "
                        f"(${net_bet_amount:.2f} after ${trading_fee:.2f} fee) "
                        f"({shares:.2f} shares at ${current_price:.2f}) "
                        f"barriers +{win_threshold:.2f}%/-{loss_threshold:.2f}%"
                        + (f" [{prediction['barrier_mode']}, sigma={prediction['sigma_pct']:.2f}%]"
                           if prediction.get('sigma_pct') else ""))
        
        return bet_id
    
    async def update_bet_prices(self, market_data: Dict[str, float]):
        """
        Update current prices for all active bets and check for win/loss conditions
        
        Args:
            market_data: Dictionary mapping symbol to current price
        """
        bets_to_close = []
        
        for bet_id, bet in self.active_bets.items():
            if bet.symbol in market_data:
                new_price = market_data[bet.symbol]
                
                # Update bet prices and P&L
                bet.current_price = new_price
                bet.current_value = bet.shares * new_price
                bet.unrealized_pnl = bet.current_value - bet.amount
                
                # Check win/loss conditions
                if new_price >= bet.win_price:
                    bets_to_close.append((bet_id, 'won', new_price))
                elif new_price <= bet.loss_price:
                    bets_to_close.append((bet_id, 'lost', new_price))
        
        # Close bets that hit win/loss thresholds
        for bet_id, outcome, exit_price in bets_to_close:
            await self._close_bet(bet_id, outcome, exit_price)
    
    async def _close_bet(self, bet_id: str, outcome: str, exit_price: float):
        """Close a bet with win/loss outcome"""
        if bet_id not in self.active_bets:
            return
        
        bet = self.active_bets[bet_id]
        
        # Calculate final values with exit fees
        gross_final_value = bet.shares * exit_price
        exit_fee = gross_final_value * (self.trading_fee_percentage / 100)
        final_value = gross_final_value - exit_fee
        realized_pnl = final_value - bet.amount
        
        # Update bet
        bet.status = BetStatus.WON if outcome == 'won' else BetStatus.LOST
        bet.exit_time = datetime.now()
        bet.exit_price = exit_price
        bet.current_price = exit_price
        bet.current_value = final_value
        bet.realized_pnl = realized_pnl
        
        # Update cash balance
        new_balance = self.cash_balance + final_value
        await self._update_cash_balance(
            new_balance, 
            f"Bet exit ({outcome}): {bet.symbol}", 
            bet_id, 
            'bet_exit'
        )
        self.cash_balance = new_balance
        
        # Update bet in database
        await self._update_bet(bet)
        
        # Remove from active bets
        del self.active_bets[bet_id]
        
        # Record portfolio snapshot
        await self._record_portfolio_snapshot(f"Closed bet ({outcome}): {bet.symbol}")
        
        self.logger.info(f"Bet closed: {bet_id} - {bet.symbol} {outcome.upper()} "
                        f"P&L: ${realized_pnl:+.2f} (${bet.amount:.2f} → ${final_value:.2f} after ${exit_fee:.2f} exit fee)")
    
    async def get_available_capital(self) -> float:
        """Get available capital for new bets"""
        return await self.get_cash_balance()

    async def get_portfolio_summary(self) -> PortfolioSnapshot:
        """
        Get current portfolio summary.

        Open positions are read from the DATABASE, not from the in-memory
        active_bets dict. The dict is a cache that two divergent close paths kept out
        of sync, which inflated reported equity by the full value of every settled
        position. The database is the single source of truth.

        Positions are valued at their last known price (`current_price`), which
        mark_to_market() refreshes. Where no price has been seen yet this falls back
        to entry price, i.e. cost.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
            SELECT amount, shares, entry_price, current_price
            FROM bets WHERE status = ?
            ''', (BetStatus.ALIVE.value,))
            rows = cursor.fetchall()
        finally:
            conn.close()

        active_bets_count = len(rows)
        total_invested = 0.0
        active_bets_value = 0.0

        for amount, shares, entry_price, current_price in rows:
            price = current_price if current_price else entry_price
            total_invested += float(amount)
            active_bets_value += float(shares) * float(price)

        unrealized_pnl = active_bets_value - total_invested

        # Get realized P&L from database
        realized_pnl = await self._get_total_realized_pnl()

        # Get cash balance from database
        cash_balance = await self.get_cash_balance()
        total_capital = cash_balance + active_bets_value

        return PortfolioSnapshot(
            timestamp=datetime.now(),
            total_capital=total_capital,
            cash_balance=cash_balance,
            active_bets_count=active_bets_count,
            active_bets_value=active_bets_value,
            total_invested=total_invested,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl
        )

    async def mark_to_market(self, prices: Dict[str, float]) -> int:
        """
        Refresh `current_price` on every open bet from a USD price map.

        Without this, current_price stays equal to entry_price for a position's whole
        life, so unrealized P&L is permanently zero and reported equity values open
        positions at cost.

        Args:
            prices: symbol -> latest price in USD

        Returns:
            Number of positions repriced.
        """
        if not prices:
            return 0

        conn = sqlite3.connect(self.db_path, timeout=60.0)
        cursor = conn.cursor()
        updated = 0

        try:
            cursor.execute('SELECT bet_id, symbol FROM bets WHERE status = ?',
                           (BetStatus.ALIVE.value,))
            for bet_id, symbol in cursor.fetchall():
                price = prices.get(symbol)
                if price is None:
                    continue
                cursor.execute('''
                UPDATE bets SET current_price = ?, updated_at = CURRENT_TIMESTAMP
                WHERE bet_id = ?
                ''', (float(price), bet_id))
                updated += 1

                # Keep the in-memory cache consistent where it is populated.
                bet = self.active_bets.get(bet_id)
                if bet is not None:
                    bet.current_price = float(price)
                    bet.current_value = bet.shares * float(price)
                    bet.unrealized_pnl = bet.current_value - bet.amount

            conn.commit()
        except Exception as e:
            self.logger.error(f"Error marking positions to market: {e}")
            conn.rollback()
        finally:
            conn.close()

        self.logger.info(f"Marked {updated} open position(s) to market")
        return updated

    async def reconcile(self, raise_on_mismatch: bool = False,
                        tolerance: float = 0.01) -> Dict:
        """
        Check that the books balance before trading.

        Cash must equal the sum of the transaction ledger, and realized P&L must
        equal proceeds minus cost basis on closed bets. A mismatch means position
        state and cash state have diverged and no sizing decision can be trusted.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('SELECT COALESCE(SUM(amount), 0) FROM cash_transactions')
            ledger_cash = float(cursor.fetchone()[0])

            cursor.execute('''
            SELECT COALESCE(SUM(balance_after), 0) FROM cash_transactions
            WHERE transaction_id = (SELECT MAX(transaction_id) FROM cash_transactions)
            ''')
            last_balance_after = float(cursor.fetchone()[0])

            cursor.execute('''
            SELECT COUNT(*), COALESCE(SUM(realized_pnl), 0) FROM bets
            WHERE status IN (?, ?)
            ''', (BetStatus.WON.value, BetStatus.LOST.value))
            closed_count, realized_pnl = cursor.fetchone()

            cursor.execute('SELECT COUNT(*) FROM bets WHERE status = ?',
                           (BetStatus.ALIVE.value,))
            alive_count = cursor.fetchone()[0]
        finally:
            conn.close()

        cash_drift = abs(ledger_cash - last_balance_after)
        memory_drift = abs(alive_count - len(self.active_bets)) if self.active_bets else 0

        result = {
            'ledger_cash': ledger_cash,
            'last_balance_after': last_balance_after,
            'cash_drift': cash_drift,
            'alive_bets_in_db': alive_count,
            'alive_bets_in_memory': len(self.active_bets),
            'memory_drift': memory_drift,
            'closed_bets': closed_count,
            'realized_pnl': float(realized_pnl),
            'balanced': cash_drift <= tolerance,
        }

        if not result['balanced']:
            message = (f"Books do not balance: ledger sum ${ledger_cash:.2f} vs "
                       f"last recorded balance ${last_balance_after:.2f} "
                       f"(drift ${cash_drift:.2f})")
            self.logger.error(message)
            if raise_on_mismatch:
                raise ValueError(message)
        else:
            self.logger.info(f"Reconciliation OK: cash ${ledger_cash:.2f}, "
                             f"{alive_count} open, {closed_count} closed, "
                             f"realized ${realized_pnl:+.2f}")

        if memory_drift:
            self.logger.warning(f"In-memory active_bets out of sync with DB "
                                f"({len(self.active_bets)} vs {alive_count})")

        return result

    async def get_expired_bets(self, max_hold_days: int) -> List[Bet]:
        """
        Open bets that have passed the time barrier and must be force-closed.

        Without a time barrier a position on a low-volatility asset can never reach
        either price barrier: one forex bet in this book sat open for 4.6 months.
        """
        if not max_hold_days or max_hold_days <= 0:
            return []

        cutoff = datetime.now() - timedelta(days=max_hold_days)
        return [bet for bet in await self.get_alive_bets() if bet.entry_time <= cutoff]
    
    async def _store_bet(self, bet: Bet):
        """Store bet in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            INSERT INTO bets (
                bet_id, symbol, asset_type, entry_price, entry_time, amount, shares,
                win_threshold, loss_threshold, win_price, loss_price, current_price,
                status, algorithm_used, probability_when_placed, currency
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                bet.bet_id, bet.symbol, bet.asset_type, bet.entry_price,
                bet.entry_time.isoformat(), bet.amount, bet.shares,
                bet.win_threshold, bet.loss_threshold, bet.win_price, bet.loss_price,
                bet.current_price, bet.status.value, bet.algorithm_used,
                bet.probability_when_placed, bet.currency
            ))
            
            conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error storing bet: {e}")
            conn.rollback()
        finally:
            conn.close()

    async def _store_bet_predictions(self, bet_id: str, algorithms: list):
        """Store individual algorithm predictions for a bet"""
        if not algorithms:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            timestamp = datetime.now().isoformat()
            for algo_pred in algorithms:
                # Ensure probability is a proper float, not bytes or numpy type
                prob_value = algo_pred['probability']
                if prob_value is not None:
                    prob_value = float(prob_value)

                cursor.execute('''
                INSERT INTO bet_predictions (bet_id, algorithm, probability, timestamp)
                VALUES (?, ?, ?, ?)
                ''', (bet_id, algo_pred['algorithm'], prob_value, timestamp))

            conn.commit()
            self.logger.debug(f"Stored {len(algorithms)} algorithm predictions for bet {bet_id}")

        except Exception as e:
            self.logger.error(f"Error storing bet predictions: {e}")
            conn.rollback()
        finally:
            conn.close()

    async def _update_bet(self, bet: Bet):
        """Update existing bet in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            UPDATE bets SET
                current_price = ?, status = ?, exit_time = ?, exit_price = ?,
                realized_pnl = ?, updated_at = CURRENT_TIMESTAMP
            WHERE bet_id = ?
            ''', (
                bet.current_price, bet.status.value,
                bet.exit_time.isoformat() if bet.exit_time else None,
                bet.exit_price, bet.realized_pnl, bet.bet_id
            ))
            
            conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error updating bet: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    async def _record_cash_transaction(self, amount: float, description: str,
                                        bet_id: Optional[str] = None, transaction_type: str = 'manual'):
        """
        Record a cash transaction to the database.
        The balance_after is calculated from the cumulative sum of all transactions.

        Args:
            amount: Transaction amount (positive = cash in, negative = cash out)
            description: Description of the transaction
            bet_id: Optional bet ID this transaction relates to
            transaction_type: Type of transaction (bet_entry, bet_close, initial_capital, etc.)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            timestamp = datetime.now().isoformat()

            # Calculate new balance as sum of all transactions including this one
            cursor.execute('SELECT SUM(amount) FROM cash_transactions')
            result = cursor.fetchone()
            current_sum = result[0] if result and result[0] is not None else 0.0
            balance_after = current_sum + amount

            cursor.execute('''
            INSERT INTO cash_transactions (timestamp, amount, balance_after, description, bet_id, transaction_type)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (timestamp, amount, balance_after, description, bet_id, transaction_type))

            conn.commit()
            self.logger.debug(f"Recorded transaction: {amount:+.2f} -> balance: {balance_after:.2f}")

        except Exception as e:
            self.logger.error(f"Error recording cash transaction: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    async def _record_portfolio_snapshot(self, notes: str = ""):
        """Record current portfolio state"""
        summary = await self.get_portfolio_summary()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            INSERT INTO portfolio_history (
                timestamp, total_capital, cash_balance, active_bets_count,
                active_bets_value, total_invested, unrealized_pnl, realized_pnl, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                summary.timestamp.isoformat(), summary.total_capital, summary.cash_balance,
                summary.active_bets_count, summary.active_bets_value, summary.total_invested,
                summary.unrealized_pnl, summary.realized_pnl, notes
            ))
            
            conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error recording portfolio snapshot: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    async def _get_total_realized_pnl(self) -> float:
        """Get total realized P&L from closed bets"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            SELECT COALESCE(SUM(realized_pnl), 0) FROM bets 
            WHERE status IN (?, ?) AND realized_pnl IS NOT NULL
            ''', (BetStatus.WON.value, BetStatus.LOST.value))
            
            result = cursor.fetchone()
            return result[0] if result else 0.0
            
        except Exception as e:
            self.logger.error(f"Error getting realized P&L: {e}")
            return 0.0
        finally:
            conn.close()

    async def get_bet_statistics(self) -> Dict:
        """Get betting statistics including win rate"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Get total bet counts by status
            cursor.execute('''
            SELECT status, COUNT(*) FROM bets GROUP BY status
            ''')
            status_counts = dict(cursor.fetchall())

            won_bets = status_counts.get(BetStatus.WON.value, 0)
            lost_bets = status_counts.get(BetStatus.LOST.value, 0)
            active_bets = status_counts.get(BetStatus.ALIVE.value, 0)

            total_bets = won_bets + lost_bets + active_bets
            completed_bets = won_bets + lost_bets
            win_rate = (won_bets / completed_bets) if completed_bets > 0 else 0.0

            return {
                "total_bets": total_bets,
                "won_bets": won_bets,
                "lost_bets": lost_bets,
                "active_bets": active_bets,
                "completed_bets": completed_bets,
                "win_rate": win_rate
            }

        except Exception as e:
            self.logger.error(f"Error getting bet statistics: {e}")
            return {
                "total_bets": 0,
                "won_bets": 0,
                "lost_bets": 0,
                "active_bets": 0,
                "completed_bets": 0,
                "win_rate": 0.0
            }
        finally:
            conn.close()
    
    async def get_alive_bets(self) -> List[Bet]:
        """Get all alive bets for monitoring"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            SELECT bet_id, symbol, asset_type, entry_price, entry_time, amount, shares,
                   win_threshold, loss_threshold, win_price, loss_price, current_price,
                   status, algorithm_used, probability_when_placed, exit_time, exit_price, realized_pnl
            FROM bets
            WHERE status = ?
            ORDER BY entry_time ASC
            ''', (BetStatus.ALIVE.value,))
            
            rows = cursor.fetchall()
            alive_bets = []
            
            for row in rows:
                # Map database columns to Bet object
                bet = Bet(
                    bet_id=row[0],
                    symbol=row[1],
                    asset_type=row[2],
                    entry_price=float(row[3]),
                    entry_time=datetime.fromisoformat(row[4]),
                    amount=float(row[5]),
                    shares=float(row[6]),
                    win_threshold=float(row[7]),
                    loss_threshold=float(row[8]),
                    win_price=float(row[9]),
                    loss_price=float(row[10]),
                    current_price=float(row[11]) if row[11] else float(row[3]),  # Use entry price if no current price
                    current_value=0.0,  # Will be calculated
                    unrealized_pnl=0.0,  # Will be calculated
                    status=BetStatus.ALIVE,
                    algorithm_used=row[13] or 'unknown',
                    probability_when_placed=float(row[14]) if row[14] else 0.0,
                    exit_time=datetime.fromisoformat(row[15]) if row[15] else None,
                    exit_price=float(row[16]) if row[16] else None,
                    realized_pnl=float(row[17]) if row[17] else None
                )
                
                # Add bet_type attribute (needed by monitoring logic)
                bet.bet_type = 'long'  # Assuming long positions for now - can be enhanced later
                
                alive_bets.append(bet)
            
            self.logger.info(f"Retrieved {len(alive_bets)} alive bets for monitoring")
            return alive_bets
            
        except Exception as e:
            self.logger.error(f"Error retrieving alive bets: {e}")
            return []
        finally:
            conn.close()
    
    async def close_bet(self, bet_id: str, current_price: float, close_reason: str):
        """Close a bet when it hits win/loss thresholds"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Get the bet details
            cursor.execute('''
            SELECT symbol, amount, shares, entry_price, win_price, loss_price, status
            FROM bets WHERE bet_id = ?
            ''', (bet_id,))

            bet_row = cursor.fetchone()
            if not bet_row:
                raise Exception(f"Bet {bet_id} not found")

            symbol, amount, shares, entry_price, win_price, loss_price, current_status = bet_row

            if current_status != BetStatus.ALIVE.value:
                self.logger.warning(f"Bet {bet_id} is already closed (status: {current_status})")
                return

            # Calculate current value and P&L
            current_value = shares * current_price
            exit_fee = current_value * (self.trading_fee_percentage / 100)  # Fee on exit value, not original amount
            net_exit_value = current_value - exit_fee
            realized_pnl = net_exit_value - amount  # P&L after fees

            # Determine new status based on thresholds first, then P&L.
            # exit_reason records WHICH barrier resolved the bet: a time-barrier exit
            # that happens to be profitable is not evidence that the win barrier was
            # reached, and must not be counted as such when calibrating.
            if current_price >= win_price:
                new_status = BetStatus.WON
                exit_reason = 'win_barrier'
                barrier_price = win_price
            elif current_price <= loss_price:
                new_status = BetStatus.LOST
                exit_reason = 'loss_barrier'
                barrier_price = loss_price
            else:
                # Between thresholds - closed early (time barrier or manual).
                new_status = BetStatus.WON if realized_pnl > 0 else BetStatus.LOST
                exit_reason = 'time_barrier' if 'TIME' in close_reason.upper() else 'manual'
                barrier_price = None

            # Slippage: how far past the barrier the fill actually landed. Negative
            # means the exit was worse than designed. Polling daily closes has
            # historically overshot a -3% stop to -7% on average.
            if barrier_price:
                slippage_pct = ((current_price - barrier_price) / barrier_price) * 100.0
                if exit_reason == 'loss_barrier':
                    # For a stop, a lower fill is worse; report worse as negative.
                    slippage_pct = -abs(slippage_pct) if current_price < barrier_price else abs(slippage_pct)
            else:
                slippage_pct = None

            # Update the bet in database
            cursor.execute('''
            UPDATE bets SET
                current_price = ?,
                status = ?,
                exit_time = ?,
                exit_price = ?,
                realized_pnl = ?,
                exit_reason = ?,
                exit_slippage_pct = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE bet_id = ?
            ''', (
                current_price,
                new_status.value,
                datetime.now().isoformat(),
                current_price,
                realized_pnl,
                exit_reason,
                slippage_pct,
                bet_id
            ))

            if slippage_pct is not None and abs(slippage_pct) > 1.0:
                self.logger.warning(
                    f"{symbol}: exit filled {slippage_pct:+.2f}% past the "
                    f"{exit_reason} at ${barrier_price:.4f} (fill ${current_price:.4f})")

            conn.commit()

            self.logger.info(f"Bet {bet_id} closed successfully - {new_status.value.upper()}: "
                           f"${realized_pnl:+.2f} P&L (after ${exit_fee:.2f} exit fee)")

        except Exception as e:
            self.logger.error(f"Error closing bet {bet_id}: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

        # Record cash transaction (outside the DB transaction for safety)
        await self._record_cash_transaction(
            amount=net_exit_value,
            description=f"Bet closed: {symbol} - {close_reason}",
            bet_id=bet_id,
            transaction_type='bet_close'
        )

        # Drop the settled position from the in-memory cache. Omitting this left
        # closed bets counted in active_bets_value, so reported total_capital
        # double-counted every settled position (the account read +10% when it was
        # +1.1%). It also let the max_concurrent_bets counter drift upward forever.
        self.active_bets.pop(bet_id, None)
        self.cash_balance = await self.get_cash_balance()

        # Record portfolio snapshot after closing bet
        await self._record_portfolio_snapshot(f"Bet closed: {symbol} - {close_reason}")
    
    async def cleanup(self):
        """Clean up portfolio manager"""
        # Record final portfolio snapshot
        await self._record_portfolio_snapshot("System shutdown")
        self.logger.info("Portfolio manager cleanup complete")