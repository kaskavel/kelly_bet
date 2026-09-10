#!/usr/bin/env python3
"""
Tests for cash movements, snapshots and reset.

These operations used to be loose scripts run by hand. The invariants that matter:
cash only ever moves through the ledger (so reconciliation holds), destructive
actions are reversible, and snapshots are never collateral damage.
"""

import sqlite3
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.portfolio.admin import AdminError, PortfolioAdmin


def build_db(path: Path, capital: float = 10000.0, bets: int = 0):
    """A minimal trading database with the tables the admin API touches."""
    if path.exists():
        path.unlink()          # rebuild from scratch so helpers are re-callable
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE bets (
            bet_id TEXT PRIMARY KEY, symbol TEXT, asset_type TEXT, status TEXT,
            amount REAL, realized_pnl REAL, entry_time TIMESTAMP);
        CREATE TABLE cash_transactions (
            transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP, amount REAL, balance_after REAL,
            description TEXT, bet_id TEXT, transaction_type TEXT);
        CREATE TABLE portfolio_history (history_id INTEGER PRIMARY KEY, timestamp TIMESTAMP);
        CREATE TABLE predictions (prediction_id INTEGER PRIMARY KEY, symbol TEXT);
        CREATE TABLE bet_predictions (prediction_id INTEGER PRIMARY KEY, bet_id TEXT);
        CREATE TABLE assets (asset_id INTEGER PRIMARY KEY, symbol TEXT);
        CREATE TABLE price_data (price_id INTEGER PRIMARY KEY, asset_id INTEGER);
    """)
    conn.execute("""INSERT INTO cash_transactions
        (timestamp, amount, balance_after, description, transaction_type)
        VALUES (?, ?, ?, 'Initial capital', 'initial_capital')""",
                 (datetime.now().isoformat(), capital, capital))
    for i in range(bets):
        conn.execute("""INSERT INTO bets
            (bet_id, symbol, asset_type, status, amount, realized_pnl, entry_time)
            VALUES (?, 'TEST', 'stock', 'alive', 100.0, NULL, ?)""",
                     (f"bet-{i}", datetime.now().isoformat()))
    conn.execute("INSERT INTO price_data (asset_id) VALUES (1)")
    conn.execute("INSERT INTO assets (symbol) VALUES ('TEST')")
    conn.commit()
    conn.close()


class AdminTestCase(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = self.root / 'trading.db'
        build_db(self.db)
        self.admin = PortfolioAdmin(self.db, snapshot_dir=self.root / 'snapshots')

    def tearDown(self):
        self.tmp.cleanup()

    # --------------------------------------------------------------- cash in

    def test_deposit_increases_balance(self):
        self.assertEqual(self.admin.cash_balance(), 10000.0)
        self.assertAlmostEqual(self.admin.deposit(2500.0, "payday"), 12500.0)
        self.assertAlmostEqual(self.admin.cash_balance(), 12500.0)

    def test_deposit_appends_to_the_ledger(self):
        """
        Cash is derived from SUM(cash_transactions). A deposit that edited a stored
        balance instead would silently break reconciliation.
        """
        self.admin.deposit(1000.0)
        conn = sqlite3.connect(self.db)
        try:
            rows = conn.execute(
                "SELECT amount, balance_after, transaction_type FROM cash_transactions "
                "ORDER BY transaction_id").fetchall()
        finally:
            conn.close()

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1][2], 'deposit')
        self.assertAlmostEqual(rows[1][0], 1000.0)
        self.assertAlmostEqual(rows[1][1], 11000.0)

    def test_deposit_rejects_non_positive(self):
        for bad in (0.0, -100.0):
            with self.assertRaises(AdminError):
                self.admin.deposit(bad)

    # -------------------------------------------------------------- cash out

    def test_withdraw_decreases_balance(self):
        self.assertAlmostEqual(self.admin.withdraw(2500.0), 7500.0)
        self.assertAlmostEqual(self.admin.cash_balance(), 7500.0)

    def test_withdraw_cannot_exceed_cash(self):
        """
        Capped at cash, not equity. Money committed to open positions is not
        available until they close; allowing it would leave the ledger unable to
        fund the exits.
        """
        with self.assertRaises(AdminError) as caught:
            self.admin.withdraw(15000.0)
        self.assertIn("only", str(caught.exception))
        self.assertAlmostEqual(self.admin.cash_balance(), 10000.0)

    def test_withdraw_exactly_the_balance_is_allowed(self):
        self.assertAlmostEqual(self.admin.withdraw(10000.0), 0.0)

    def test_withdraw_rejects_non_positive(self):
        with self.assertRaises(AdminError):
            self.admin.withdraw(-50.0)

    def test_cash_history_excludes_bet_flows(self):
        """The audit trail shows money in and out, not every trade."""
        self.admin.deposit(500.0)
        self.admin.withdraw(200.0)
        conn = sqlite3.connect(self.db)
        conn.execute("""INSERT INTO cash_transactions
            (timestamp, amount, balance_after, description, transaction_type)
            VALUES (?, -100.0, 0.0, 'Bet entry: X', 'bet_entry')""",
                     (datetime.now().isoformat(),))
        conn.commit()
        conn.close()

        history = self.admin.cash_history()
        self.assertEqual({h['type'] for h in history},
                         {'deposit', 'withdrawal', 'initial_capital'})

    # -------------------------------------------------------------- snapshots

    def test_snapshot_round_trip(self):
        self.admin.deposit(5000.0)
        snapshot = self.admin.create_snapshot('before-change')
        self.assertTrue(snapshot.path.exists())
        self.assertGreater(snapshot.size_bytes, 0)

        self.admin.withdraw(9000.0)
        self.assertAlmostEqual(self.admin.cash_balance(), 6000.0)

        self.admin.restore_snapshot(snapshot.name)
        self.assertAlmostEqual(self.admin.cash_balance(), 15000.0)

    def test_restore_snapshots_current_state_first(self):
        """A restore must be undoable."""
        self.admin.deposit(1000.0)
        snapshot = self.admin.create_snapshot('point-a')

        self.admin.deposit(7000.0)          # now 18000
        safety = self.admin.restore_snapshot(snapshot.name)

        self.assertAlmostEqual(self.admin.cash_balance(), 11000.0)
        self.assertIsNotNone(safety)
        self.admin.restore_snapshot(safety.name)
        self.assertAlmostEqual(self.admin.cash_balance(), 18000.0)

    def test_restore_refuses_a_foreign_file(self):
        """Do not copy an arbitrary SQLite file over the live database."""
        junk = self.admin.snapshot_dir / 'trading-20260101-000000-junk.db'
        conn = sqlite3.connect(junk)
        conn.execute('CREATE TABLE unrelated (x INTEGER)')
        conn.commit()
        conn.close()

        with self.assertRaises(AdminError):
            self.admin.restore_snapshot(junk.name)
        self.assertAlmostEqual(self.admin.cash_balance(), 10000.0)

    def test_restore_missing_snapshot_raises(self):
        with self.assertRaises(AdminError):
            self.admin.restore_snapshot('trading-19990101-000000-nope.db')

    def test_snapshots_listed_newest_first(self):
        first = self.admin.create_snapshot('one')
        second = self.admin.create_snapshot('two')
        names = [s.name for s in self.admin.list_snapshots()]
        self.assertEqual(len(names), 2)
        self.assertLessEqual(names.index(second.name), names.index(first.name))

    # ------------------------------------------------------------------ reset

    def test_reset_clears_the_book_and_sets_capital(self):
        build_db(self.db, capital=10000.0, bets=3)
        self.admin.deposit(500.0)

        self.admin.reset(initial_capital=25000.0)

        self.assertAlmostEqual(self.admin.cash_balance(), 25000.0)
        conn = sqlite3.connect(self.db)
        try:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM bets').fetchone()[0], 0)
            self.assertEqual(
                conn.execute('SELECT COUNT(*) FROM cash_transactions').fetchone()[0], 1)
        finally:
            conn.close()

    def test_reset_is_undoable(self):
        self.admin.deposit(4000.0)
        safety = self.admin.reset(initial_capital=100.0)

        self.assertAlmostEqual(self.admin.cash_balance(), 100.0)
        self.admin.restore_snapshot(safety.name)
        self.assertAlmostEqual(self.admin.cash_balance(), 14000.0)

    def test_reset_never_deletes_snapshots(self):
        """
        The whole value of a snapshot is surviving a reset. Snapshots live in their
        own directory and reset must not go near it.
        """
        keep_a = self.admin.create_snapshot('keep-a')
        keep_b = self.admin.create_snapshot('keep-b')

        self.admin.reset(initial_capital=1000.0)

        names = {s.name for s in self.admin.list_snapshots()}
        self.assertIn(keep_a.name, names)
        self.assertIn(keep_b.name, names)

    def test_reset_keeps_the_price_cache_by_default(self):
        """
        Price bars are an expensive rate-limited cache with nothing to do with the
        trading record. Wiping them would force a full re-download.
        """
        self.admin.reset(initial_capital=1000.0)

        conn = sqlite3.connect(self.db)
        try:
            self.assertGreater(conn.execute('SELECT COUNT(*) FROM price_data').fetchone()[0], 0)
            self.assertGreater(conn.execute('SELECT COUNT(*) FROM assets').fetchone()[0], 0)
        finally:
            conn.close()

    def test_reset_can_clear_the_price_cache_on_request(self):
        self.admin.reset(initial_capital=1000.0, keep_price_history=False)

        conn = sqlite3.connect(self.db)
        try:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM price_data').fetchone()[0], 0)
        finally:
            conn.close()

    def test_reset_rejects_negative_capital(self):
        with self.assertRaises(AdminError):
            self.admin.reset(initial_capital=-1.0)

    # ---------------------------------------------------------------- summary

    def test_summary_reports_the_book(self):
        build_db(self.db, capital=10000.0, bets=2)
        self.admin.deposit(1000.0)
        self.admin.withdraw(250.0)

        summary = self.admin.summary()
        self.assertAlmostEqual(summary['cash_balance'], 10750.0)
        self.assertEqual(summary['open_bets'], 2)
        self.assertAlmostEqual(summary['deposited'], 11000.0)
        self.assertAlmostEqual(summary['withdrawn'], 250.0)
        self.assertGreater(summary['db_size_mb'], 0)

    def test_summary_counts_are_not_silently_zero(self):
        """
        Regression: the counts were built in the return statement, after `finally`
        had closed the connection, and the error was swallowed as a zero. A 121 MB
        database with 245k price bars reported "0 bars, 0 bets".
        """
        build_db(self.db, capital=5000.0, bets=4)
        summary = self.admin.summary()

        self.assertEqual(summary['total_bets'], 4)
        self.assertEqual(summary['open_bets'], 4)
        self.assertGreater(summary['price_bars'], 0)
        # open_bets and total_bets come from different queries; they must agree.
        self.assertLessEqual(summary['open_bets'], summary['total_bets'])



if __name__ == '__main__':
    unittest.main()


class SnapshotCollisionTestCase(unittest.TestCase):
    """
    Snapshots taken in the same second must not overwrite each other.

    This was a real bug: filenames used second-resolution timestamps, and
    restore_snapshot() takes an automatic safety snapshot, so restoring twice in
    quick succession could clobber the snapshot being restored -- destroying it and
    turning the restore into a silent no-op.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = self.root / 'trading.db'
        build_db(self.db)
        self.admin = PortfolioAdmin(self.db, snapshot_dir=self.root / 'snapshots')

    def tearDown(self):
        self.tmp.cleanup()

    def test_same_second_snapshots_get_distinct_files(self):
        snapshots = [self.admin.create_snapshot('burst') for _ in range(5)]
        names = {s.name for s in snapshots}
        self.assertEqual(len(names), 5, f"snapshot names collided: {names}")
        for snapshot in snapshots:
            self.assertTrue(snapshot.path.exists())

    def test_repeated_restore_preserves_each_state(self):
        """Restore, then undo the restore, and land back where we were."""
        self.admin.deposit(1000.0)                 # 11000
        point_a = self.admin.create_snapshot('point-a')

        self.admin.deposit(7000.0)                 # 18000
        safety = self.admin.restore_snapshot(point_a.name)
        self.assertAlmostEqual(self.admin.cash_balance(), 11000.0)

        self.admin.restore_snapshot(safety.name)
        self.assertAlmostEqual(self.admin.cash_balance(), 18000.0)

        # And the original snapshot is still intact and still usable.
        self.assertTrue(point_a.path.exists())
        self.admin.restore_snapshot(point_a.name)
        self.assertAlmostEqual(self.admin.cash_balance(), 11000.0)
