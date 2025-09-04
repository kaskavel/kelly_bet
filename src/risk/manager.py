"""
Risk Management System
Implements circuit breakers, drawdown controls, and safety mechanisms.
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TradingStatus(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class RiskMetrics:
    """Current risk assessment metrics"""
    drawdown_percentage: float          # Current drawdown from peak
    consecutive_losses: int             # Number of consecutive losing bets
    losing_streak_days: int            # Days since last winning bet
    capital_at_risk_percentage: float  # Percentage of capital in active bets
    daily_loss_percentage: float       # Loss percentage in last 24 hours
    weekly_loss_percentage: float      # Loss percentage in last 7 days
    risk_level: RiskLevel              # Overall risk assessment
    trading_status: TradingStatus      # Current trading status
    warnings: List[str]                # Active risk warnings


@dataclass
class RiskAlert:
    """Risk alert/warning"""
    alert_id: str
    timestamp: datetime
    risk_level: RiskLevel
    category: str                      # 'drawdown', 'losses', 'exposure', etc.
    message: str
    current_value: float
    threshold_value: float
    is_resolved: bool = False


class RiskManager:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.db_path = Path(config['database']['sqlite']['path'])
        
        # Risk thresholds from config
        risk_config = config.get('risk', {})
        self.max_drawdown = risk_config.get('max_drawdown', 20.0)          # 20% max drawdown
        self.loss_streak_limit = risk_config.get('loss_streak_limit', 5)   # 5 consecutive losses
        self.daily_loss_limit = risk_config.get('daily_loss_limit', 5.0)   # 5% daily loss limit
        self.weekly_loss_limit = risk_config.get('weekly_loss_limit', 10.0) # 10% weekly loss limit
        self.max_exposure = risk_config.get('max_exposure', 50.0)          # 50% max capital at risk
        self.min_capital = risk_config.get('min_capital', 1000.0)          # $1000 minimum capital
        
        # Risk state
        self.trading_status = TradingStatus.ACTIVE
        self.emergency_stop_reason: Optional[str] = None
        self.last_risk_check = datetime.now()
        self.peak_capital = 0.0
        self.active_alerts: List[RiskAlert] = []
        
        # Pause conditions
        self.pause_until: Optional[datetime] = None
        self.pause_reason: Optional[str] = None
    
    async def initialize(self):
        """Initialize risk management system"""
        self.logger.info("Initializing risk management system...")
        
        # Create database tables
        await self._create_tables()
        
        # Load risk state
        await self._load_risk_state()
        
        self.logger.info(f"Risk management initialized - Status: {self.trading_status.value}")
    
    async def _create_tables(self):
        """Create database tables for risk management"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Risk events table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS risk_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP NOT NULL,
            event_type TEXT NOT NULL,  -- 'drawdown', 'loss_streak', 'emergency_stop', etc.
            risk_level TEXT NOT NULL,
            description TEXT NOT NULL,
            trigger_value REAL,
            threshold_value REAL,
            action_taken TEXT,
            is_resolved BOOLEAN DEFAULT FALSE,
            resolved_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Risk metrics history table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS risk_metrics_history (
            metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP NOT NULL,
            drawdown_percentage REAL NOT NULL,
            consecutive_losses INTEGER NOT NULL,
            losing_streak_days INTEGER NOT NULL,
            capital_at_risk_percentage REAL NOT NULL,
            daily_loss_percentage REAL NOT NULL,
            weekly_loss_percentage REAL NOT NULL,
            risk_level TEXT NOT NULL,
            trading_status TEXT NOT NULL,
            total_capital REAL NOT NULL,
            peak_capital REAL NOT NULL
        )
        ''')
        
        # Trading status log
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trading_status_log (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP NOT NULL,
            old_status TEXT,
            new_status TEXT NOT NULL,
            reason TEXT NOT NULL,
            duration_minutes INTEGER,  -- How long in previous status
            triggered_by TEXT  -- What triggered the change
        )
        ''')
        
        conn.commit()
        conn.close()
    
    async def _load_risk_state(self):
        """Load existing risk state from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Load latest trading status
            cursor.execute('''
            SELECT new_status, reason, timestamp FROM trading_status_log 
            ORDER BY timestamp DESC LIMIT 1
            ''')
            result = cursor.fetchone()
            
            if result:
                status_str, reason, timestamp = result
                self.trading_status = TradingStatus(status_str)
                if self.trading_status != TradingStatus.ACTIVE:
                    self.pause_reason = reason
                    self.logger.warning(f"Loaded non-active status: {status_str} - {reason}")
            
            # Load peak capital
            cursor.execute('''
            SELECT MAX(total_capital) FROM risk_metrics_history
            ''')
            result = cursor.fetchone()
            if result and result[0]:
                self.peak_capital = result[0]
            
        except Exception as e:
            self.logger.error(f"Error loading risk state: {e}")
        finally:
            conn.close()
    
    async def assess_risk(self, portfolio_summary) -> RiskMetrics:
        """
        Perform comprehensive risk assessment
        
        Args:
            portfolio_summary: Current portfolio snapshot
            
        Returns:
            RiskMetrics with current risk assessment
        """
        self.logger.debug("Performing risk assessment...")
        
        # Update peak capital
        if portfolio_summary.total_capital > self.peak_capital:
            self.peak_capital = portfolio_summary.total_capital
        
        # Calculate risk metrics
        drawdown_pct = self._calculate_drawdown(portfolio_summary.total_capital)
        consecutive_losses = await self._get_consecutive_losses()
        losing_streak_days = await self._get_losing_streak_days()
        capital_at_risk_pct = self._calculate_capital_at_risk(portfolio_summary)
        daily_loss_pct = await self._calculate_daily_loss(portfolio_summary.total_capital)
        weekly_loss_pct = await self._calculate_weekly_loss(portfolio_summary.total_capital)
        
        # Assess overall risk level
        risk_level = self._determine_risk_level(
            drawdown_pct, consecutive_losses, daily_loss_pct, 
            weekly_loss_pct, capital_at_risk_pct
        )
        
        # Generate warnings
        warnings = self._generate_risk_warnings(
            drawdown_pct, consecutive_losses, losing_streak_days,
            capital_at_risk_pct, daily_loss_pct, weekly_loss_pct
        )
        
        # Create risk metrics
        risk_metrics = RiskMetrics(
            drawdown_percentage=drawdown_pct,
            consecutive_losses=consecutive_losses,
            losing_streak_days=losing_streak_days,
            capital_at_risk_percentage=capital_at_risk_pct,
            daily_loss_percentage=daily_loss_pct,
            weekly_loss_percentage=weekly_loss_pct,
            risk_level=risk_level,
            trading_status=self.trading_status,
            warnings=warnings
        )
        
        # Store risk metrics
        await self._store_risk_metrics(risk_metrics, portfolio_summary.total_capital)
        
        self.last_risk_check = datetime.now()
        return risk_metrics
    
    async def can_continue_trading(self, portfolio_summary=None) -> bool:
        """
        Check if trading can continue based on risk assessment
        
        Returns:
            True if trading can continue, False if should pause/stop
        """
        # Check if we're in a paused state with time limit
        if self.pause_until and datetime.now() < self.pause_until:
            remaining = self.pause_until - datetime.now()
            self.logger.info(f"Trading paused for {remaining.total_seconds()/60:.1f} more minutes")
            return False
        elif self.pause_until and datetime.now() >= self.pause_until:
            # Pause period expired, resume trading
            await self._resume_trading("Pause period expired")
        
        # Emergency stop overrides everything
        if self.trading_status == TradingStatus.EMERGENCY_STOP:
            return False
        
        # If no portfolio summary provided, perform quick checks
        if portfolio_summary is None:
            return self.trading_status == TradingStatus.ACTIVE
        
        # Perform full risk assessment
        risk_metrics = await self.assess_risk(portfolio_summary)
        
        # Check for emergency stop conditions
        if await self._should_emergency_stop(risk_metrics, portfolio_summary):
            await self._trigger_emergency_stop("Critical risk thresholds exceeded")
            return False
        
        # Check for pause conditions
        if await self._should_pause_trading(risk_metrics):
            await self._pause_trading("Risk thresholds exceeded", duration_minutes=60)
            return False
        
        return self.trading_status == TradingStatus.ACTIVE
    
    def _calculate_drawdown(self, current_capital: float) -> float:
        """Calculate current drawdown percentage from peak"""
        if self.peak_capital <= 0:
            return 0.0
        
        drawdown = (self.peak_capital - current_capital) / self.peak_capital * 100
        return max(0.0, drawdown)
    
    async def _get_consecutive_losses(self) -> int:
        """Get number of consecutive losing bets"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get recent bets ordered by exit time
            cursor.execute('''
            SELECT status FROM bets 
            WHERE status IN ('won', 'lost') AND exit_time IS NOT NULL
            ORDER BY exit_time DESC
            LIMIT 20
            ''')
            
            results = cursor.fetchall()
            consecutive_losses = 0
            
            for (status,) in results:
                if status == 'lost':
                    consecutive_losses += 1
                else:
                    break  # Stop at first win
            
            return consecutive_losses
            
        except Exception as e:
            self.logger.error(f"Error calculating consecutive losses: {e}")
            return 0
        finally:
            conn.close()
    
    async def _get_losing_streak_days(self) -> int:
        """Get number of days since last winning bet"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            SELECT MAX(exit_time) FROM bets 
            WHERE status = 'won' AND exit_time IS NOT NULL
            ''')
            
            result = cursor.fetchone()
            if not result or not result[0]:
                return 0
            
            last_win_time = datetime.fromisoformat(result[0])
            days_since = (datetime.now() - last_win_time).days
            return max(0, days_since)
            
        except Exception as e:
            self.logger.error(f"Error calculating losing streak days: {e}")
            return 0
        finally:
            conn.close()
    
    def _calculate_capital_at_risk(self, portfolio_summary) -> float:
        """Calculate percentage of capital currently at risk in active bets"""
        if portfolio_summary.total_capital <= 0:
            return 0.0
        
        return (portfolio_summary.active_bets_value / portfolio_summary.total_capital) * 100
    
    async def _calculate_daily_loss(self, current_capital: float) -> float:
        """Calculate loss percentage in last 24 hours"""
        return await self._calculate_period_loss(current_capital, hours=24)
    
    async def _calculate_weekly_loss(self, current_capital: float) -> float:
        """Calculate loss percentage in last 7 days"""
        return await self._calculate_period_loss(current_capital, hours=24*7)
    
    async def _calculate_period_loss(self, current_capital: float, hours: int) -> float:
        """Calculate loss percentage over specified period"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            cursor.execute('''
            SELECT total_capital FROM portfolio_history 
            WHERE timestamp >= ? 
            ORDER BY timestamp ASC 
            LIMIT 1
            ''', (cutoff_time.isoformat(),))
            
            result = cursor.fetchone()
            if not result:
                return 0.0
            
            start_capital = result[0]
            if start_capital <= 0:
                return 0.0
            
            loss_pct = ((start_capital - current_capital) / start_capital) * 100
            return max(0.0, loss_pct)  # Only return positive losses
            
        except Exception as e:
            self.logger.error(f"Error calculating period loss: {e}")
            return 0.0
        finally:
            conn.close()
    
    def _determine_risk_level(self, drawdown: float, consecutive_losses: int, 
                            daily_loss: float, weekly_loss: float, 
                            capital_at_risk: float) -> RiskLevel:
        """Determine overall risk level based on metrics"""
        
        # Critical risk conditions
        if (drawdown >= self.max_drawdown * 0.8 or  # 80% of max drawdown
            consecutive_losses >= self.loss_streak_limit or
            daily_loss >= self.daily_loss_limit or
            weekly_loss >= self.weekly_loss_limit):
            return RiskLevel.CRITICAL
        
        # High risk conditions
        if (drawdown >= self.max_drawdown * 0.6 or  # 60% of max drawdown
            consecutive_losses >= self.loss_streak_limit * 0.8 or
            daily_loss >= self.daily_loss_limit * 0.8 or
            weekly_loss >= self.weekly_loss_limit * 0.8 or
            capital_at_risk >= self.max_exposure * 0.9):
            return RiskLevel.HIGH
        
        # Medium risk conditions
        if (drawdown >= self.max_drawdown * 0.4 or  # 40% of max drawdown
            consecutive_losses >= self.loss_streak_limit * 0.6 or
            daily_loss >= self.daily_loss_limit * 0.6 or
            weekly_loss >= self.weekly_loss_limit * 0.6 or
            capital_at_risk >= self.max_exposure * 0.7):
            return RiskLevel.MEDIUM
        
        return RiskLevel.LOW
    
    def _generate_risk_warnings(self, drawdown: float, consecutive_losses: int,
                              losing_streak_days: int, capital_at_risk: float,
                              daily_loss: float, weekly_loss: float) -> List[str]:
        """Generate list of active risk warnings"""
        warnings = []
        
        if drawdown >= self.max_drawdown * 0.5:
            warnings.append(f"High drawdown: {drawdown:.1f}% (limit: {self.max_drawdown:.1f}%)")
        
        if consecutive_losses >= self.loss_streak_limit * 0.6:
            warnings.append(f"Loss streak: {consecutive_losses} consecutive losses")
        
        if losing_streak_days >= 7:
            warnings.append(f"No wins in {losing_streak_days} days")
        
        if daily_loss >= self.daily_loss_limit * 0.5:
            warnings.append(f"Daily loss: {daily_loss:.1f}% (limit: {self.daily_loss_limit:.1f}%)")
        
        if weekly_loss >= self.weekly_loss_limit * 0.5:
            warnings.append(f"Weekly loss: {weekly_loss:.1f}% (limit: {self.weekly_loss_limit:.1f}%)")
        
        if capital_at_risk >= self.max_exposure * 0.8:
            warnings.append(f"High exposure: {capital_at_risk:.1f}% of capital at risk")
        
        return warnings
    
    async def _should_emergency_stop(self, risk_metrics: RiskMetrics, 
                                   portfolio_summary) -> bool:
        """Check if emergency stop should be triggered"""
        
        # Critical drawdown
        if risk_metrics.drawdown_percentage >= self.max_drawdown:
            self.logger.critical(f"Emergency stop: Drawdown {risk_metrics.drawdown_percentage:.1f}% "
                               f">= limit {self.max_drawdown:.1f}%")
            return True
        
        # Extreme daily loss
        if risk_metrics.daily_loss_percentage >= self.daily_loss_limit:
            self.logger.critical(f"Emergency stop: Daily loss {risk_metrics.daily_loss_percentage:.1f}% "
                               f">= limit {self.daily_loss_limit:.1f}%")
            return True
        
        # Below minimum capital
        if portfolio_summary.total_capital < self.min_capital:
            self.logger.critical(f"Emergency stop: Capital ${portfolio_summary.total_capital:.2f} "
                               f"< minimum ${self.min_capital:.2f}")
            return True
        
        return False
    
    async def _should_pause_trading(self, risk_metrics: RiskMetrics) -> bool:
        """Check if trading should be paused"""
        
        # Too many consecutive losses
        if risk_metrics.consecutive_losses >= self.loss_streak_limit:
            return True
        
        # High weekly loss
        if risk_metrics.weekly_loss_percentage >= self.weekly_loss_limit * 0.8:
            return True
        
        # High risk level
        if risk_metrics.risk_level == RiskLevel.CRITICAL:
            return True
        
        return False
    
    async def _trigger_emergency_stop(self, reason: str):
        """Trigger emergency stop"""
        old_status = self.trading_status
        self.trading_status = TradingStatus.EMERGENCY_STOP
        self.emergency_stop_reason = reason
        
        await self._log_status_change(old_status, self.trading_status, reason, "emergency_condition")
        
        self.logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
    
    async def _pause_trading(self, reason: str, duration_minutes: int = 60):
        """Pause trading for specified duration"""
        old_status = self.trading_status
        self.trading_status = TradingStatus.PAUSED
        self.pause_reason = reason
        self.pause_until = datetime.now() + timedelta(minutes=duration_minutes)
        
        await self._log_status_change(old_status, self.trading_status, 
                                    f"{reason} (paused for {duration_minutes} minutes)", 
                                    "risk_threshold")
        
        self.logger.warning(f"TRADING PAUSED: {reason} - Resuming at {self.pause_until}")
    
    async def _resume_trading(self, reason: str):
        """Resume trading"""
        old_status = self.trading_status
        self.trading_status = TradingStatus.ACTIVE
        self.pause_reason = None
        self.pause_until = None
        
        await self._log_status_change(old_status, self.trading_status, reason, "automatic_resume")
        
        self.logger.info(f"TRADING RESUMED: {reason}")
    
    async def _log_status_change(self, old_status: TradingStatus, new_status: TradingStatus,
                               reason: str, triggered_by: str):
        """Log trading status change"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            INSERT INTO trading_status_log 
            (timestamp, old_status, new_status, reason, triggered_by)
            VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                old_status.value if old_status else None,
                new_status.value,
                reason,
                triggered_by
            ))
            
            conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error logging status change: {e}")
        finally:
            conn.close()
    
    async def _store_risk_metrics(self, metrics: RiskMetrics, total_capital: float):
        """Store risk metrics in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            INSERT INTO risk_metrics_history (
                timestamp, drawdown_percentage, consecutive_losses, losing_streak_days,
                capital_at_risk_percentage, daily_loss_percentage, weekly_loss_percentage,
                risk_level, trading_status, total_capital, peak_capital
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                metrics.drawdown_percentage,
                metrics.consecutive_losses,
                metrics.losing_streak_days,
                metrics.capital_at_risk_percentage,
                metrics.daily_loss_percentage,
                metrics.weekly_loss_percentage,
                metrics.risk_level.value,
                metrics.trading_status.value,
                total_capital,
                self.peak_capital
            ))
            
            conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error storing risk metrics: {e}")
        finally:
            conn.close()
    
    async def force_resume_trading(self, reason: str = "Manual override"):
        """Force resume trading (manual intervention)"""
        await self._resume_trading(reason)
        self.logger.warning(f"MANUAL TRADING RESUME: {reason}")
    
    async def get_risk_summary(self) -> Dict:
        """Get summary of risk management state"""
        return {
            'trading_status': self.trading_status.value,
            'emergency_stop_reason': self.emergency_stop_reason,
            'pause_reason': self.pause_reason,
            'pause_until': self.pause_until.isoformat() if self.pause_until else None,
            'peak_capital': self.peak_capital,
            'last_risk_check': self.last_risk_check.isoformat(),
            'thresholds': {
                'max_drawdown': self.max_drawdown,
                'loss_streak_limit': self.loss_streak_limit,
                'daily_loss_limit': self.daily_loss_limit,
                'weekly_loss_limit': self.weekly_loss_limit,
                'max_exposure': self.max_exposure,
                'min_capital': self.min_capital
            }
        }
    
    async def cleanup(self):
        """Clean up risk manager"""
        await self._log_status_change(
            self.trading_status, 
            TradingStatus.PAUSED, 
            "System shutdown", 
            "system_shutdown"
        )
        self.logger.info("Risk manager cleanup complete")