#!/usr/bin/env python3
"""
Tests for portfolio accounting invariants.

The bug these exist to prevent: `close_bet()` (the path the monitoring loop calls)
settled a bet in the database but never removed it from the in-memory `active_bets`
dict, while `get_portfolio_summary()` valued open positions from that dict. Closed
positions were therefore counted as assets AND as cash, and reported equity read
$11,001.32 (+10.0%) on an account actually worth $10,112.82 (+1.1%).

The same drift made the max_concurrent_bets counter climb forever.
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

from src.portfolio.manager import BetStatus, PortfolioManager


class PortfolioAccountingTestCase(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        db_path = Path(self.tmp.name) / 'portfolio_test.db'

        self.config = {
            'database': {'sqlite': {'path': str(db_path)}},
            'trading': {
                'initial_capital': 10000.0,
                'win_threshold': 5.0,
                'loss_threshold': 3.0,
                'kelly_fraction': 0.005,
                'max_bet_fraction': 0.1,
                'max_risk_fraction': 0.0035,
                'min_bet_amount': 100.0,
                'max_bet_amount': 10000.0,
                'min_probability': 66.0,
                'max_loss_percentage': 5.0,
                'max_concurrent_bets': 30,
                'trading_fee_percentage': 0.25,
                'max_hold_days': 15,
            },
        }

        self.manager = PortfolioManager(self.config)
        asyncio.run(self.manager.initialize())

    def tearDown(self):
        self.tmp.cleanup()

    def _place(self, symbol='TEST', probability=80.0, price=100.0, asset_type=None):
        prediction = {
            'symbol': symbol,
            'probability': probability,
            'current_price': price,
            'algorithms': [{'algorithm': 'sma', 'probability': probability}],
        }
        if asset_type:
            prediction['asset_type'] = asset_type
        return asyncio.run(self.manager.place_bet(prediction))

    # --- the equity bug ----------------------------------------------------

    def test_closing_a_bet_removes_it_from_active_value(self):
        """After a close, the position must not still be counted as an asset."""
        bet_id = self._place()

        before = asyncio.run(self.manager.get_portfolio_summary())
        self.assertEqual(before.active_bets_count, 1)

        asyncio.run(self.manager.close_bet(bet_id, 105.0, 'WIN THRESHOLD HIT'))

        after = asyncio.run(self.manager.get_portfolio_summary())
        self.assertEqual(after.active_bets_count, 0)
        self.assertEqual(after.active_bets_value, 0.0)
        self.assertNotIn(bet_id, self.manager.active_bets)

    def test_total_capital_equals_cash_plus_positions(self):
        """
        Equity must equal the cash ledger plus marked positions -- never more.

        This is the invariant that was violated: total_capital counted settled
        positions twice.
        """
        first = self._place(symbol='AAA')
        self._place(symbol='BBB')
        asyncio.run(self.manager.close_bet(first, 105.0, 'WIN THRESHOLD HIT'))

        summary = asyncio.run(self.manager.get_portfolio_summary())
        ledger_cash = asyncio.run(self.manager.get_cash_balance())

        self.assertAlmostEqual(summary.cash_balance, ledger_cash, places=6)
        self.assertAlmostEqual(
            summary.total_capital,
            summary.cash_balance + summary.active_bets_value,
            places=6,
        )

        # And equity must be within a sane band of the starting capital -- a
        # double-count showed up as a ~9x overstatement of the return.
        self.assertLess(summary.total_capital, 10600.0)
        self.assertGreater(summary.total_capital, 9400.0)

    def test_reconcile_reports_balanced_books(self):
        """The reconciliation check passes on a consistent book."""
        self._place(symbol='AAA')
        result = asyncio.run(self.manager.reconcile())

        self.assertTrue(result['balanced'])
        self.assertEqual(result['alive_bets_in_db'], 1)
        self.assertLessEqual(result['cash_drift'], 0.01)

    def test_concurrency_limit_counts_from_database(self):
        """The open-bet count must come from the DB, not a drifting cache."""
        bet_id = self._place(symbol='AAA')
        asyncio.run(self.manager.close_bet(bet_id, 105.0, 'WIN THRESHOLD HIT'))

        # A stale cache entry must not be trusted.
        self.manager.active_bets['ghost'] = object()
        self.assertEqual(asyncio.run(self.manager._count_alive_bets()), 0)

    # --- mark to market ----------------------------------------------------

    def test_mark_to_market_updates_unrealized_pnl(self):
        """
        Open positions must be repriced.

        current_price stayed equal to entry_price for a position's whole life, so
        unrealized P&L was permanently zero and equity valued positions at cost.
        """
        self._place(symbol='AAA', price=100.0)

        flat = asyncio.run(self.manager.get_portfolio_summary())
        self.assertAlmostEqual(flat.unrealized_pnl, 0.0, places=6)

        updated = asyncio.run(self.manager.mark_to_market({'AAA': 110.0}))
        self.assertEqual(updated, 1)

        marked = asyncio.run(self.manager.get_portfolio_summary())
        self.assertGreater(marked.unrealized_pnl, 0.0)
        self.assertGreater(marked.active_bets_value, marked.total_invested)

    def test_mark_to_market_ignores_unknown_symbols(self):
        """A missing price leaves the position at its last known mark."""
        self._place(symbol='AAA', price=100.0)
        updated = asyncio.run(self.manager.mark_to_market({'ZZZ': 50.0}))
        self.assertEqual(updated, 0)

    # --- metadata ----------------------------------------------------------

    def test_asset_type_is_not_hardcoded_to_stock(self):
        """
        asset_type must come from the prediction.

        It was hardcoded 'stock', which mislabelled every commodity ETF and forex
        pair -- all 87 bets in the live book are recorded as stocks, including SLV
        and EURGBP=X.
        """
        bet_id = self._place(symbol='EURGBP=X', asset_type='forex')

        conn = sqlite3.connect(self.config['database']['sqlite']['path'])
        try:
            stored = conn.execute(
                'SELECT asset_type FROM bets WHERE bet_id = ?', (bet_id,)).fetchone()[0]
        finally:
            conn.close()

        self.assertEqual(stored, 'forex')

    def test_asset_type_inferred_when_absent(self):
        """Without metadata, the symbol shape is used rather than defaulting."""
        self.assertEqual(PortfolioManager._infer_asset_type('EURGBP=X'), 'forex')
        self.assertEqual(PortfolioManager._infer_asset_type('BTC-USD'), 'crypto')
        self.assertEqual(PortfolioManager._infer_asset_type('AAPL'), 'stock')

    # --- exit quality ------------------------------------------------------

    def test_exit_reason_and_slippage_are_recorded(self):
        """
        A barrier exit records which barrier and how far past it the fill landed.

        Barrier overshoot was the largest measurable cost in this strategy's history
        and there was nowhere to see it.
        """
        bet_id = self._place(symbol='AAA', price=100.0)

        # Stop is at 97.00; simulate a gap that fills at 91.00.
        asyncio.run(self.manager.close_bet(bet_id, 91.0, 'LOSS THRESHOLD HIT: -9.00%'))

        conn = sqlite3.connect(self.config['database']['sqlite']['path'])
        try:
            reason, slippage, status = conn.execute(
                'SELECT exit_reason, exit_slippage_pct, status FROM bets WHERE bet_id = ?',
                (bet_id,)).fetchone()
        finally:
            conn.close()

        self.assertEqual(reason, 'loss_barrier')
        self.assertEqual(status, 'lost')
        self.assertIsNotNone(slippage)
        self.assertLess(slippage, 0.0, "overshooting a stop must read as negative slippage")

    def test_time_barrier_exit_is_labelled_distinctly(self):
        """
        A time-barrier exit must not masquerade as a barrier hit.

        A profitable timeout is not evidence that the win barrier was reached, and
        counting it as one would corrupt calibration.
        """
        bet_id = self._place(symbol='AAA', price=100.0)

        asyncio.run(self.manager.close_bet(bet_id, 101.0, 'TIME BARRIER: held 15d'))

        conn = sqlite3.connect(self.config['database']['sqlite']['path'])
        try:
            reason, slippage = conn.execute(
                'SELECT exit_reason, exit_slippage_pct FROM bets WHERE bet_id = ?',
                (bet_id,)).fetchone()
        finally:
            conn.close()

        self.assertEqual(reason, 'time_barrier')
        self.assertIsNone(slippage, "a timeout has no barrier to slip against")

    def test_expired_bets_are_detected(self):
        """Positions past the time barrier are reported for force-closing."""
        bet_id = self._place(symbol='AAA', price=100.0)

        conn = sqlite3.connect(self.config['database']['sqlite']['path'])
        try:
            conn.execute('UPDATE bets SET entry_time = ? WHERE bet_id = ?',
                         ((datetime.now() - timedelta(days=40)).isoformat(), bet_id))
            conn.commit()
        finally:
            conn.close()

        expired = asyncio.run(self.manager.get_expired_bets(max_hold_days=15))
        self.assertEqual(len(expired), 1)
        self.assertEqual(expired[0].bet_id, bet_id)

        # Nothing is expired under a longer limit.
        self.assertEqual(len(asyncio.run(self.manager.get_expired_bets(max_hold_days=90))), 0)

    # --- fees --------------------------------------------------------------

    def test_both_fee_legs_are_charged(self):
        """Fees are charged on entry and on exit."""
        bet_id = self._place(symbol='AAA', price=100.0)

        conn = sqlite3.connect(self.config['database']['sqlite']['path'])
        try:
            entry_cash, = conn.execute(
                "SELECT amount FROM cash_transactions WHERE bet_id = ? AND "
                "transaction_type = 'bet_entry'", (bet_id,)).fetchone()
            stake, = conn.execute(
                'SELECT amount FROM bets WHERE bet_id = ?', (bet_id,)).fetchone()
        finally:
            conn.close()

        # Cash out exceeds the net stake by the entry fee.
        fee_rate = self.config['trading']['trading_fee_percentage'] / 100.0
        self.assertAlmostEqual(-entry_cash, stake / (1 - fee_rate), places=2)

        # A flat exit loses roughly the round-trip fee, not zero.
        asyncio.run(self.manager.close_bet(bet_id, 100.0, 'manual close'))
        summary = asyncio.run(self.manager.get_portfolio_summary())
        self.assertLess(summary.realized_pnl, 0.0)


if __name__ == '__main__':
    unittest.main()


class BarrierPassThroughTestCase(unittest.TestCase):
    """
    A bet must be placed on the barriers it was displayed with.

    Under volatility-scaled barriers the dashboard's place_bet() did not forward
    win/loss thresholds, so every placement path raised (correctly — the alternative
    is silently sizing a different bet than the one the user agreed to). These tests
    pin both halves of that contract.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        db_path = Path(self.tmp.name) / 'barrier_test.db'
        self.config = {
            'database': {'sqlite': {'path': str(db_path)}},
            'trading': {
                'initial_capital': 100000.0,
                'win_threshold': 5.0, 'loss_threshold': 3.0,
                'kelly_fraction': 0.005, 'max_bet_fraction': 0.1,
                'max_risk_fraction': 0.0035, 'min_bet_amount': 100.0,
                'max_bet_amount': 10000.0, 'min_probability': 66.0,
                'max_loss_percentage': 5.0, 'max_concurrent_bets': 30,
                'trading_fee_percentage': 0.25, 'max_hold_days': 30,
                'barrier': {'mode': 'volatility', 'win_sigma': 1.5,
                            'loss_sigma': 1.0, 'horizon_days': 10},
            },
        }
        self.manager = PortfolioManager(self.config)
        asyncio.run(self.manager.initialize())

    def tearDown(self):
        self.tmp.cleanup()

    def test_supplied_barriers_are_stored_and_priced(self):
        """The stored win/loss prices must follow the barriers that were passed."""
        bet_id = asyncio.run(self.manager.place_bet({
            'symbol': 'WIDE', 'probability': 70.0, 'current_price': 100.0,
            'asset_type': 'stock', 'algorithms': [],
            'win_threshold': 18.0, 'loss_threshold': 12.0,
        }))

        conn = sqlite3.connect(self.config['database']['sqlite']['path'])
        try:
            win_pct, loss_pct, win_price, loss_price = conn.execute(
                'SELECT win_threshold, loss_threshold, win_price, loss_price '
                'FROM bets WHERE bet_id = ?', (bet_id,)).fetchone()
        finally:
            conn.close()

        self.assertAlmostEqual(win_pct, 18.0, places=6)
        self.assertAlmostEqual(loss_pct, 12.0, places=6)
        self.assertAlmostEqual(win_price, 118.0, places=6)
        self.assertAlmostEqual(loss_price, 88.0, places=6)

    def test_missing_barriers_are_refused_not_defaulted(self):
        """
        Falling back to the config defaults would size a different bet than the one
        displayed. Refusing loudly is the correct behaviour.
        """
        with self.assertRaises(ValueError) as caught:
            asyncio.run(self.manager.place_bet({
                'symbol': 'NOBARRIERS', 'probability': 70.0, 'current_price': 100.0,
                'asset_type': 'stock', 'algorithms': [],
            }))
        self.assertIn('barriers', str(caught.exception).lower())

    def test_wider_barriers_get_a_smaller_stake(self):
        """
        Capped-loss Kelly shrinks as the stop widens, so a wide-barrier bet is sized
        smaller for the same probability. Worth pinning: it is counter-intuitive and
        it is why the cheapest tables often fall under the minimum bet.
        """
        narrow = asyncio.run(self.manager.place_bet({
            'symbol': 'NARROW', 'probability': 70.0, 'current_price': 100.0,
            'asset_type': 'stock', 'algorithms': [],
            'win_threshold': 6.0, 'loss_threshold': 4.0}))
        wide = asyncio.run(self.manager.place_bet({
            'symbol': 'WIDE', 'probability': 70.0, 'current_price': 100.0,
            'asset_type': 'stock', 'algorithms': [],
            'win_threshold': 24.0, 'loss_threshold': 16.0}))

        conn = sqlite3.connect(self.config['database']['sqlite']['path'])
        try:
            amounts = dict(conn.execute(
                'SELECT bet_id, amount FROM bets WHERE bet_id IN (?, ?)',
                (narrow, wide)).fetchall())
        finally:
            conn.close()

        self.assertLess(amounts[wide], amounts[narrow])

    def test_placement_keeps_the_books_balanced(self):
        asyncio.run(self.manager.place_bet({
            'symbol': 'AAA', 'probability': 70.0, 'current_price': 100.0,
            'asset_type': 'stock', 'algorithms': [],
            'win_threshold': 9.0, 'loss_threshold': 6.0}))

        result = asyncio.run(self.manager.reconcile())
        self.assertTrue(result['balanced'])
        self.assertEqual(result['alive_bets_in_db'], 1)
