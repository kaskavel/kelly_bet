#!/usr/bin/env python3
"""
Tests for when a bet settles.

There used to be three settlement paths and they disagreed. The dashboard's path
checked ONLY the price barriers -- no time barrier -- so a position that reached
neither price level stayed open indefinitely. A real forex bet sat open for 293 days
against a 60-day setting. These tests pin the single shared rule.
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

from src.portfolio.manager import PortfolioManager
from src.trading.settlement import decide, settle_positions


NOW = datetime(2026, 9, 10, 12, 0, 0)


def call(current_price=None, held_days=10, max_hold_days=60,
         entry=100.0, win=110.0, loss=95.0, bet_type='long'):
    return decide(
        entry_price=entry, win_price=win, loss_price=loss,
        entry_time=NOW - timedelta(days=held_days),
        current_price=current_price, max_hold_days=max_hold_days,
        now=NOW, bet_type=bet_type)


class DecisionTestCase(unittest.TestCase):

    # --- price barriers ----------------------------------------------------

    def test_win_barrier_closes(self):
        decision = call(current_price=111.0)
        self.assertTrue(decision.should_close)
        self.assertEqual(decision.exit_type, 'win_barrier')
        self.assertEqual(decision.exit_price, 111.0)
        self.assertIn("WIN THRESHOLD HIT", decision.reason)

    def test_loss_barrier_closes(self):
        decision = call(current_price=94.0)
        self.assertTrue(decision.should_close)
        self.assertEqual(decision.exit_type, 'loss_barrier')
        self.assertIn("LOSS THRESHOLD HIT", decision.reason)

    def test_barriers_are_inclusive(self):
        self.assertEqual(call(current_price=110.0).exit_type, 'win_barrier')
        self.assertEqual(call(current_price=95.0).exit_type, 'loss_barrier')

    def test_between_barriers_stays_open(self):
        decision = call(current_price=102.0, held_days=10)
        self.assertFalse(decision.should_close)

    # --- the time barrier, i.e. the bug ------------------------------------

    def test_time_barrier_closes_a_stale_position(self):
        """
        The 293-day bet. Price between the barriers, well past the limit.

        The dashboard's settlement path had no time barrier at all, so this position
        had nothing that could ever close it.
        """
        decision = call(current_price=101.0, held_days=293, max_hold_days=60)
        self.assertTrue(decision.should_close)
        self.assertEqual(decision.exit_type, 'time_barrier')
        self.assertIn("293d", decision.reason)
        self.assertIn("60d", decision.reason)
        self.assertEqual(decision.exit_price, 101.0)

    def test_time_barrier_is_inclusive_at_the_limit(self):
        self.assertFalse(call(current_price=101.0, held_days=59, max_hold_days=60).should_close)
        self.assertTrue(call(current_price=101.0, held_days=60, max_hold_days=60).should_close)

    def test_price_barrier_wins_over_a_simultaneous_time_barrier(self):
        """
        A position that hits a barrier on its expiry day is a BARRIER hit.

        The distinction matters: a time-barrier exit says nothing about whether the
        win barrier would have been reached, so counting one as the other would
        corrupt the calibration data.
        """
        decision = call(current_price=115.0, held_days=300, max_hold_days=60)
        self.assertEqual(decision.exit_type, 'win_barrier')

        decision = call(current_price=90.0, held_days=300, max_hold_days=60)
        self.assertEqual(decision.exit_type, 'loss_barrier')

    def test_time_barrier_disabled_when_zero(self):
        self.assertFalse(call(current_price=101.0, held_days=999, max_hold_days=0).should_close)

    # --- missing prices ----------------------------------------------------

    def test_no_price_keeps_a_fresh_position_open(self):
        decision = call(current_price=None, held_days=5, max_hold_days=60)
        self.assertFalse(decision.should_close)
        self.assertIn("no current price", decision.reason)

    def test_no_price_still_expires_a_stale_position(self):
        """
        A position must not become immortal because its quote went missing.

        This is the realistic failure for a delisted or unsupported symbol, and it is
        exactly how a bet reaches 293 days.
        """
        decision = call(current_price=None, held_days=293, max_hold_days=60)
        self.assertTrue(decision.should_close)
        self.assertEqual(decision.exit_type, 'time_barrier')
        self.assertIn("no current price", decision.reason)
        self.assertEqual(decision.exit_price, 100.0)   # falls back to entry

    # --- shorts ------------------------------------------------------------

    def test_short_barriers_are_inverted(self):
        self.assertEqual(call(current_price=89.0, win=90.0, loss=105.0,
                              bet_type='short').exit_type, 'win_barrier')
        self.assertEqual(call(current_price=106.0, win=90.0, loss=105.0,
                              bet_type='short').exit_type, 'loss_barrier')


class SettlePositionsTestCase(unittest.TestCase):
    """The runner, against a real PortfolioManager."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / 'settle.db'
        self.config = {
            'database': {'sqlite': {'path': str(self.db)}},
            'trading': {
                'initial_capital': 100000.0,
                'win_threshold': 5.0, 'loss_threshold': 3.0,
                'kelly_fraction': 0.005, 'max_bet_fraction': 0.1,
                'max_risk_fraction': 0.0035, 'min_bet_amount': 100.0,
                'max_bet_amount': 10000.0, 'min_probability': 66.0,
                'max_loss_percentage': 5.0, 'max_concurrent_bets': 30,
                'trading_fee_percentage': 0.25, 'max_hold_days': 60,
                'barrier': {'mode': 'volatility'},
            },
        }
        self.manager = PortfolioManager(self.config)
        asyncio.run(self.manager.initialize())

    def tearDown(self):
        self.tmp.cleanup()

    def _place(self, symbol, price=100.0, win=10.0, loss=6.0, days_ago=0):
        bet_id = asyncio.run(self.manager.place_bet({
            'symbol': symbol, 'probability': 75.0, 'current_price': price,
            'asset_type': 'stock', 'algorithms': [],
            'win_threshold': win, 'loss_threshold': loss}))
        if days_ago:
            conn = sqlite3.connect(self.db)
            try:
                conn.execute('UPDATE bets SET entry_time = ? WHERE bet_id = ?',
                             ((datetime.now() - timedelta(days=days_ago)).isoformat(),
                              bet_id))
                conn.commit()
            finally:
                conn.close()
        return bet_id

    def test_stale_position_is_settled_by_the_runner(self):
        self._place('OLD', days_ago=293)
        settled = asyncio.run(settle_positions(self.manager, {'OLD': 101.0}, 60))

        self.assertEqual(len(settled), 1)
        self.assertEqual(settled[0]['exit_type'], 'time_barrier')
        self.assertEqual(asyncio.run(self.manager._count_alive_bets()), 0)

    def test_fresh_positions_are_left_alone(self):
        self._place('NEW', days_ago=3)
        settled = asyncio.run(settle_positions(self.manager, {'NEW': 101.0}, 60))

        self.assertEqual(settled, [])
        self.assertEqual(asyncio.run(self.manager._count_alive_bets()), 1)

    def test_mixed_book_settles_only_what_qualifies(self):
        self._place('STALE', days_ago=100)
        self._place('WINNER', price=100.0, win=10.0, loss=6.0, days_ago=5)
        self._place('OPEN', days_ago=5)

        settled = asyncio.run(settle_positions(
            self.manager,
            {'STALE': 100.5, 'WINNER': 112.0, 'OPEN': 100.5},
            60))

        by_symbol = {s['symbol']: s['exit_type'] for s in settled}
        self.assertEqual(by_symbol.get('STALE'), 'time_barrier')
        self.assertEqual(by_symbol.get('WINNER'), 'win_barrier')
        self.assertNotIn('OPEN', by_symbol)
        self.assertEqual(asyncio.run(self.manager._count_alive_bets()), 1)

    def test_settlement_keeps_the_books_balanced(self):
        self._place('OLD', days_ago=293)
        asyncio.run(settle_positions(self.manager, {'OLD': 101.0}, 60))

        result = asyncio.run(self.manager.reconcile())
        self.assertTrue(result['balanced'])

    def test_unpriced_stale_position_is_still_settled(self):
        """The realistic 293-day case: expired AND unpriceable."""
        self._place('GONE', days_ago=293)
        settled = asyncio.run(settle_positions(self.manager, {}, 60))

        self.assertEqual(len(settled), 1)
        self.assertEqual(settled[0]['exit_type'], 'time_barrier')
        self.assertEqual(asyncio.run(self.manager._count_alive_bets()), 0)


if __name__ == '__main__':
    unittest.main()


class EntryPointCoverageTestCase(unittest.TestCase):
    """
    Settlement must be checked on EVERY way into the app.

    The 293-day bet survived because the dashboard's entry path never checked. These
    tests assert the wiring rather than the arithmetic, so a future refactor that
    drops a call is caught.
    """

    def test_every_entry_path_settles(self):
        import inspect
        import src.ui.dashboard as dashboard
        import src.cli.bet_analyzer as analyzer
        import src.cli.bet_monitor as monitor
        import src.core.trading_system as trading

        checks = [
            # page load / every Streamlit rerun -- offline, no API calls
            (dashboard.TradingDashboard.load_cached_state, 'check_and_settle_bets'),
            # the "Refresh prices" button
            (dashboard.TradingDashboard.refresh_data, 'check_and_settle_bets'),
            # the lightweight portfolio refresh
            (dashboard.TradingDashboard.refresh_portfolio_only, 'check_and_settle_bets'),
            # `python main.py --bets`
            (analyzer.BetAnalyzer.show_bet_history, '_settle_overdue'),
            # `python main.py --livebets`
            (monitor.BetMonitor.show_live_bets, '_monitor_and_settle_positions'),
            # `python main.py --mode manual|automated`
            (trading.TradingSystem.run, '_monitor_existing_bets'),
        ]

        for func, expected_call in checks:
            source = inspect.getsource(func)
            self.assertIn(expected_call, source,
                          f"{func.__qualname__} does not check settlement")

    def test_all_settlement_paths_use_the_shared_rule(self):
        """No caller may reimplement the decision."""
        import inspect
        import src.cli.bet_monitor as monitor
        import src.core.trading_system as trading
        import src.ui.dashboard as dashboard

        for func in (monitor.BetMonitor._monitor_and_settle_positions,
                     trading.TradingSystem._monitor_existing_bets,
                     dashboard.TradingDashboard.check_and_settle_bets):
            source = inspect.getsource(func)
            self.assertIn('settle_positions', source,
                          f"{func.__qualname__} does not delegate to the shared rule")
            # The old hand-rolled comparisons must be gone.
            self.assertNotIn('hit_win_threshold', source)
            self.assertNotIn('hit_loss_threshold', source)

    def test_offline_settlement_makes_no_network_calls(self):
        """
        The page-load check must not touch an API, or it defeats the rate-limit fix.

        Verified structurally: the offline branch works from stored marks on the bets
        themselves and never consults market_data.
        """
        import inspect
        import src.ui.dashboard as dashboard

        source = inspect.getsource(dashboard.TradingDashboard.check_and_settle_bets)
        offline = source.split('# Offline check')[1]
        for forbidden in ('get_stock_data', 'get_latest_data',
                          'get_current_prices_usd', 'get_crypto_data'):
            self.assertNotIn(forbidden, offline)
        self.assertIn('bet.current_price', offline)
