#!/usr/bin/env python3
"""
Tests that the risk controls actually fire.

Every risk control in this system was dead code: trading_system.py called
`can_continue_trading()` with no argument, which returns on the
`portfolio_summary is None` shortcut before drawdown, loss streaks, exposure or
minimum capital are ever evaluated. risk_events and risk_metrics_history both held
zero rows after six months of live trading.

These tests force each breach and assert the halt, so the shortcut cannot silently
come back.
"""

import asyncio
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.risk.manager import RiskManager, TradingStatus
from src.portfolio.manager import PortfolioSnapshot


def snapshot(total_capital: float, active_value: float = 0.0,
             realized: float = 0.0) -> PortfolioSnapshot:
    """A portfolio snapshot with the fields the risk manager reads."""
    return PortfolioSnapshot(
        timestamp=datetime.now(),
        total_capital=total_capital,
        cash_balance=total_capital - active_value,
        active_bets_count=1 if active_value else 0,
        active_bets_value=active_value,
        total_invested=active_value,
        unrealized_pnl=0.0,
        realized_pnl=realized,
    )


class RiskManagerTestCase(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        db_path = Path(self.tmp.name) / 'risk_test.db'

        self.config = {
            'database': {'sqlite': {'path': str(db_path)}},
            'risk': {
                'max_drawdown': 20.0,
                'loss_streak_limit': 5,
                'min_capital': 1000.0,
                'daily_loss_limit': 5.0,
                'weekly_loss_limit': 10.0,
                'max_exposure': 50.0,
            },
        }

        self._create_bets_table(db_path)
        self.manager = RiskManager(self.config)
        asyncio.run(self.manager.initialize())

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def _create_bets_table(db_path: Path):
        """The risk manager reads bets for loss streaks."""
        conn = sqlite3.connect(db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS bets (
                bet_id TEXT PRIMARY KEY, symbol TEXT, status TEXT,
                exit_time TIMESTAMP, realized_pnl REAL, entry_time TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def _insert_losses(self, count: int):
        """Record `count` consecutive losing bets."""
        conn = sqlite3.connect(self.config['database']['sqlite']['path'])
        now = datetime.now()
        for i in range(count):
            conn.execute(
                'INSERT INTO bets (bet_id, symbol, status, exit_time, realized_pnl, entry_time) '
                'VALUES (?, ?, ?, ?, ?, ?)',
                (f'loss-{i}', 'TEST', 'lost',
                 (now - timedelta(hours=count - i)).isoformat(), -50.0,
                 (now - timedelta(days=count - i)).isoformat()),
            )
        conn.commit()
        conn.close()

    # --- the bug that made all of this unreachable ------------------------

    def test_no_summary_takes_the_shortcut(self):
        """
        Documents the trap: called with no argument, no risk metric is evaluated.

        This is why trading_system.py must pass the portfolio summary.
        """
        allowed = asyncio.run(self.manager.can_continue_trading())
        self.assertTrue(allowed)

        # Even a catastrophic portfolio is not detected without the argument.
        self.manager.peak_capital = 10000.0
        still_allowed = asyncio.run(self.manager.can_continue_trading())
        self.assertTrue(still_allowed, "shortcut should return on status alone")

    # --- each control, with the summary passed ----------------------------

    def test_drawdown_breach_halts_trading(self):
        """A drawdown past max_drawdown triggers an emergency stop."""
        self.manager.peak_capital = 10000.0

        # 25% drawdown, past the 20% limit.
        allowed = asyncio.run(self.manager.can_continue_trading(snapshot(7500.0)))

        self.assertFalse(allowed)
        self.assertEqual(self.manager.trading_status, TradingStatus.EMERGENCY_STOP)

    def test_drawdown_within_limit_allows_trading(self):
        """A drawdown inside the limit does not halt."""
        self.manager.peak_capital = 10000.0

        allowed = asyncio.run(self.manager.can_continue_trading(snapshot(9500.0)))

        self.assertTrue(allowed)
        self.assertEqual(self.manager.trading_status, TradingStatus.ACTIVE)

    def test_below_minimum_capital_halts_trading(self):
        """Falling under min_capital triggers an emergency stop."""
        self.manager.peak_capital = 1000.0

        allowed = asyncio.run(self.manager.can_continue_trading(snapshot(500.0)))

        self.assertFalse(allowed)
        self.assertEqual(self.manager.trading_status, TradingStatus.EMERGENCY_STOP)

    def test_loss_streak_pauses_trading(self):
        """Hitting loss_streak_limit consecutive losses pauses trading."""
        self._insert_losses(6)  # limit is 5
        self.manager.peak_capital = 10000.0

        allowed = asyncio.run(self.manager.can_continue_trading(snapshot(9900.0)))

        self.assertFalse(allowed)
        self.assertIsNotNone(self.manager.pause_until)

    def test_emergency_stop_is_sticky(self):
        """Once stopped, a healthy portfolio does not silently resume trading."""
        self.manager.peak_capital = 10000.0
        asyncio.run(self.manager.can_continue_trading(snapshot(7000.0)))
        self.assertEqual(self.manager.trading_status, TradingStatus.EMERGENCY_STOP)

        allowed = asyncio.run(self.manager.can_continue_trading(snapshot(10000.0)))
        self.assertFalse(allowed, "emergency stop must require an explicit override")

    def test_risk_assessment_is_recorded(self):
        """Assessments must leave an audit trail; the tables were empty before."""
        self.manager.peak_capital = 10000.0
        asyncio.run(self.manager.can_continue_trading(snapshot(9800.0)))

        conn = sqlite3.connect(self.config['database']['sqlite']['path'])
        try:
            count = conn.execute('SELECT COUNT(*) FROM risk_metrics_history').fetchone()[0]
        finally:
            conn.close()

        self.assertGreater(count, 0, "risk assessment produced no stored metrics")


if __name__ == '__main__':
    unittest.main()
