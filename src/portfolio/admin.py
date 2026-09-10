"""
Portfolio administration: cash movements, database snapshots, and reset.

These operations were previously loose scripts (`scripts/fix_balance.py`,
`reset_portfolio_clean_start.py`) run by hand, which is how a portfolio ends up in a
state nobody can reconstruct. They live here as one audited API so that:

  * **every cash movement goes through the transaction ledger.** Cash is derived from
    `SUM(cash_transactions.amount)`, so a deposit that edits a balance directly would
    silently break reconciliation. Add and withdraw append rows; nothing is mutated.
  * **destructive actions snapshot first.** Restore and reset both take an automatic
    snapshot of the current database before touching it, so any of them can be undone.
  * **snapshots are never destroyed.** Reset clears the trading database and leaves
    the snapshot directory alone -- that is the whole point of having snapshots.
"""

import logging
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

SNAPSHOT_DIR = Path('data/snapshots')


class AdminError(Exception):
    """A requested administrative action was refused."""


@dataclass
class Snapshot:
    """One saved copy of the trading database."""
    path: Path
    created: datetime
    size_bytes: int
    label: str

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)

    @property
    def name(self) -> str:
        return self.path.name


class PortfolioAdmin:
    """Cash movements and database lifecycle for one trading database."""

    def __init__(self, db_path, snapshot_dir: Path = SNAPSHOT_DIR):
        self.db_path = Path(db_path)
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------- cash

    def cash_balance(self) -> float:
        """Cash from the ledger, which is the single source of truth."""
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute('SELECT SUM(amount) FROM cash_transactions').fetchone()
            return float(row[0]) if row and row[0] is not None else 0.0
        finally:
            conn.close()

    def deposit(self, amount: float, note: str = "") -> float:
        """
        Add cash to the portfolio. Returns the new balance.

        Appends to the ledger rather than adjusting a stored balance, so
        reconciliation continues to hold.
        """
        if amount <= 0:
            raise AdminError(f"Deposit must be positive, got {amount}")

        description = f"Deposit{f': {note}' if note else ''}"
        return self._record(amount, description, 'deposit')

    def withdraw(self, amount: float, note: str = "") -> float:
        """
        Withdraw cash. Returns the new balance.

        Capped at the CASH balance, not at total equity: money committed to open
        positions is not available until those positions close. Allowing a withdrawal
        against unrealised value would leave the ledger unable to fund the exits.
        """
        if amount <= 0:
            raise AdminError(f"Withdrawal must be positive, got {amount}")

        available = self.cash_balance()
        if amount > available:
            raise AdminError(
                f"Cannot withdraw ${amount:,.2f}: only ${available:,.2f} is in cash. "
                f"Funds committed to open positions become available when they close."
            )

        description = f"Withdrawal{f': {note}' if note else ''}"
        return self._record(-amount, description, 'withdrawal')

    def _record(self, amount: float, description: str, transaction_type: str) -> float:
        conn = sqlite3.connect(self.db_path, timeout=60.0)
        try:
            cursor = conn.cursor()
            row = cursor.execute('SELECT SUM(amount) FROM cash_transactions').fetchone()
            current = float(row[0]) if row and row[0] is not None else 0.0
            balance_after = current + amount

            cursor.execute("""
                INSERT INTO cash_transactions
                    (timestamp, amount, balance_after, description, bet_id, transaction_type)
                VALUES (?, ?, ?, ?, NULL, ?)
            """, (datetime.now().isoformat(), amount, balance_after,
                  description, transaction_type))
            conn.commit()

            logger.info(f"{transaction_type}: {amount:+,.2f} -> balance {balance_after:,.2f}")
            return balance_after
        finally:
            conn.close()

    def cash_history(self, limit: int = 50) -> List[Dict]:
        """Recent deposits and withdrawals only, for an audit trail in the UI."""
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute("""
                SELECT timestamp, amount, balance_after, description, transaction_type
                FROM cash_transactions
                WHERE transaction_type IN ('deposit', 'withdrawal', 'initial_capital')
                ORDER BY transaction_id DESC LIMIT ?
            """, (limit,)).fetchall()
        finally:
            conn.close()

        return [{'timestamp': r[0], 'amount': r[1], 'balance_after': r[2],
                 'description': r[3], 'type': r[4]} for r in rows]

    # -------------------------------------------------------------- snapshots

    def create_snapshot(self, label: str = "manual") -> Snapshot:
        """
        Save a consistent copy of the database.

        Uses SQLite's own backup API rather than a file copy: a copy taken while a
        write is in flight, or while a WAL holds committed pages, can be torn or stale.
        """
        if not self.db_path.exists():
            raise AdminError(f"No database at {self.db_path} to snapshot")

        safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)[:40]
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        target = self.snapshot_dir / f"trading-{stamp}-{safe_label or 'manual'}.db"

        # Never overwrite an existing snapshot. Second-resolution timestamps collide
        # when two snapshots are taken in quick succession -- and restore_snapshot()
        # takes one automatically, so a same-second collision could have clobbered
        # the very snapshot being restored, destroying it and silently making the
        # restore a no-op.
        if target.exists():
            for attempt in range(2, 1000):
                candidate = self.snapshot_dir / (
                    f"trading-{stamp}-{safe_label or 'manual'}-{attempt}.db")
                if not candidate.exists():
                    target = candidate
                    break
            else:
                raise AdminError(f"Cannot find a free snapshot name for {stamp}")

        source = sqlite3.connect(self.db_path)
        try:
            destination = sqlite3.connect(target)
            try:
                source.backup(destination)
            finally:
                destination.close()
        finally:
            source.close()

        logger.info(f"Snapshot written to {target}")
        return self._describe(target)

    def list_snapshots(self) -> List[Snapshot]:
        """Snapshots, newest first."""
        found = [self._describe(p) for p in self.snapshot_dir.glob('trading-*.db')]
        return sorted(found, key=lambda s: s.created, reverse=True)

    def _describe(self, path: Path) -> Snapshot:
        stat = path.stat()
        parts = path.stem.split('-', 3)
        label = parts[3] if len(parts) > 3 else 'manual'
        return Snapshot(path=path, created=datetime.fromtimestamp(stat.st_mtime),
                        size_bytes=stat.st_size, label=label)

    def restore_snapshot(self, snapshot_name: str) -> Snapshot:
        """
        Replace the live database with a snapshot.

        The current database is snapshotted first under the label
        `pre-restore`, so this is reversible. Returns that safety snapshot.
        """
        source = self.snapshot_dir / snapshot_name
        if not source.exists():
            raise AdminError(f"Snapshot not found: {snapshot_name}")
        if not self._looks_like_trading_db(source):
            raise AdminError(
                f"{snapshot_name} does not look like a trading database "
                f"(missing the expected tables); refusing to restore it.")

        safety = None
        if self.db_path.exists():
            safety = self.create_snapshot(label='pre-restore')

        # Belt and braces: the safety snapshot must never be the file we are about to
        # read from, or the restore would copy the wrong state over the database.
        if safety is not None and safety.path.resolve() == source.resolve():
            raise AdminError(
                f"Safety snapshot collided with the restore source ({source.name}); "
                f"aborting rather than risk overwriting it.")

        # Clear WAL/journal siblings so the restored file is what actually loads.
        for suffix in ('-wal', '-shm'):
            sibling = Path(str(self.db_path) + suffix)
            if sibling.exists():
                sibling.unlink()

        shutil.copy2(source, self.db_path)
        logger.info(f"Restored {snapshot_name} over {self.db_path}")
        return safety

    def delete_snapshot(self, snapshot_name: str):
        """Remove one snapshot. Only ever called explicitly by the user."""
        target = self.snapshot_dir / snapshot_name
        if not target.exists():
            raise AdminError(f"Snapshot not found: {snapshot_name}")
        target.unlink()
        logger.info(f"Deleted snapshot {snapshot_name}")

    @staticmethod
    def _looks_like_trading_db(path: Path) -> bool:
        """Sanity check before restoring a file over the live database."""
        try:
            conn = sqlite3.connect(path)
            try:
                tables = {row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'")}
            finally:
                conn.close()
            return {'bets', 'cash_transactions'}.issubset(tables)
        except Exception:
            return False

    # ------------------------------------------------------------------ reset

    def reset(self, initial_capital: float, keep_price_history: bool = True) -> Snapshot:
        """
        Start over with an empty book.

        Snapshots the current database first and returns that snapshot, so a reset is
        always undoable. **Existing snapshots are never touched** -- they live in a
        separate directory and this method does not go near it.

        `keep_price_history` preserves the `assets` and `price_data` tables by default.
        Those are an expensive, rate-limited cache with nothing to do with the trading
        record, and re-downloading them is exactly the API burn we are trying to avoid.
        """
        if initial_capital < 0:
            raise AdminError(f"Initial capital cannot be negative, got {initial_capital}")

        safety = self.create_snapshot(label='pre-reset')

        conn = sqlite3.connect(self.db_path, timeout=60.0)
        try:
            cursor = conn.cursor()
            existing = {row[0] for row in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}

            # Trading record: always cleared.
            wipe = ['bets', 'bet_predictions', 'cash_transactions', 'portfolio_history',
                    'predictions', 'algorithm_performance', 'algorithm_weights',
                    'risk_events', 'risk_metrics_history', 'trading_status_log']
            # Market-data cache: cleared only on request.
            if not keep_price_history:
                wipe += ['price_data', 'assets']

            for table in wipe:
                if table in existing:
                    cursor.execute(f'DELETE FROM "{table}"')

            if 'sqlite_sequence' in existing:
                cursor.execute('DELETE FROM sqlite_sequence')

            if initial_capital > 0:
                cursor.execute("""
                    INSERT INTO cash_transactions
                        (timestamp, amount, balance_after, description, bet_id, transaction_type)
                    VALUES (?, ?, ?, ?, NULL, 'initial_capital')
                """, (datetime.now().isoformat(), initial_capital, initial_capital,
                      'Initial capital - portfolio reset'))

            conn.commit()
        finally:
            conn.close()

        logger.warning(f"Portfolio reset. Opening capital ${initial_capital:,.2f}. "
                       f"Previous state saved as {safety.name}")
        return safety

    # ----------------------------------------------------------------- status

    def summary(self) -> Dict:
        """Counts and balances, for the admin screen."""
        conn = sqlite3.connect(self.db_path)
        try:
            def count(table: str) -> int:
                try:
                    return int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
                except sqlite3.Error:
                    return 0

            open_bets = int(conn.execute(
                "SELECT COUNT(*) FROM bets WHERE status='alive'").fetchone()[0] or 0)
            realised = conn.execute(
                "SELECT COALESCE(SUM(realized_pnl),0) FROM bets "
                "WHERE status IN ('won','lost')").fetchone()[0] or 0.0

            deposits = conn.execute(
                "SELECT COALESCE(SUM(amount),0) FROM cash_transactions "
                "WHERE transaction_type IN ('deposit','initial_capital')").fetchone()[0] or 0.0
            withdrawals = conn.execute(
                "SELECT COALESCE(SUM(-amount),0) FROM cash_transactions "
                "WHERE transaction_type='withdrawal'").fetchone()[0] or 0.0

            # All counts must be taken while the connection is still open. Building
            # them in the return statement ran them after `finally` had closed it, and
            # count() swallowed the resulting ProgrammingError as a zero -- so a
            # 121 MB database with 245k price bars reported "0 bars, 0 bets".
            total_bets = count('bets')
            price_bars = count('price_data')
            stored_predictions = count('predictions')
        finally:
            conn.close()

        return {
            'cash_balance': self.cash_balance(),
            'open_bets': open_bets,
            'total_bets': total_bets,
            'realised_pnl': float(realised),
            'deposited': float(deposits),
            'withdrawn': float(withdrawals),
            'price_bars': price_bars,
            'stored_predictions': stored_predictions,
            'db_size_mb': self.db_path.stat().st_size / (1024 * 1024)
            if self.db_path.exists() else 0.0,
            'snapshots': len(self.list_snapshots()),
        }
