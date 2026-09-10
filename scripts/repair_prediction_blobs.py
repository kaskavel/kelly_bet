#!/usr/bin/env python3
"""
Repair prediction rows stored as raw float32 BLOBs instead of SQLite REALs.

numpy float32 values (from Keras and some sklearn paths) do not adapt to a SQLite
REAL: sqlite3 stores their raw 4 bytes as a BLOB. 82,429 of 698,233 rows in
`predictions` were affected. Any aggregate silently mixes types -- AVG(probability)
for the LSTM returned 6035.52, and MIN > MAX because SQLite orders BLOBs above
numbers.

The insert path is fixed (predictor._store_predictions now casts with float()), so
this only repairs history.

    python scripts/repair_prediction_blobs.py --dry-run
    python scripts/repair_prediction_blobs.py
"""

import argparse
import sqlite3
import struct
import sys
from pathlib import Path

import yaml


def decode(blob: bytes):
    """
    Decode a stored numpy scalar.

    4 bytes is a float32, 8 is a float64. Byte order follows the machine that wrote
    the row; little-endian is assumed and the result sanity-checked against the
    0-100 range a probability must occupy.
    """
    for size, fmt in ((4, '<f'), (8, '<d')):
        if len(blob) == size:
            value = struct.unpack(fmt, blob)[0]
            if 0.0 <= value <= 100.0:
                return float(value)
            # Retry big-endian before giving up.
            value = struct.unpack('>' + fmt[1], blob)[0]
            if 0.0 <= value <= 100.0:
                return float(value)
            return None
    return None


def main():
    parser = argparse.ArgumentParser(description='Repair BLOB-encoded prediction probabilities')
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--dry-run', action='store_true',
                        help='Report what would change without writing')
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    db_path = Path(config['database']['sqlite']['path'])

    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return 1

    conn = sqlite3.connect(db_path, timeout=60.0)
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT typeof(probability), COUNT(*) FROM predictions GROUP BY 1
        """)
        print("Current type distribution in predictions.probability:")
        for type_name, count in cursor.fetchall():
            print(f"  {type_name:8s} {count:>8,}")

        cursor.execute("""
            SELECT prediction_id, probability FROM predictions
            WHERE typeof(probability) = 'blob'
        """)
        rows = cursor.fetchall()

        if not rows:
            print("\nNothing to repair.")
            return 0

        print(f"\nFound {len(rows):,} BLOB rows")

        repairs = []
        undecodable = 0
        for prediction_id, blob in rows:
            value = decode(blob)
            if value is None:
                undecodable += 1
            else:
                repairs.append((value, prediction_id))

        print(f"  decodable:   {len(repairs):,}")
        print(f"  undecodable: {undecodable:,}")

        if repairs:
            sample = repairs[:5]
            print("\n  sample decoded values: " +
                  ", ".join(f"{value:.2f}" for value, _ in sample))

        if args.dry_run:
            print("\nDry run: no changes written.")
            return 0

        cursor.executemany(
            "UPDATE predictions SET probability = ? WHERE prediction_id = ?", repairs)

        # Rows that cannot be decoded are not probabilities and must not be left as
        # BLOBs where aggregates would pick them up.
        if undecodable:
            cursor.execute("""
                DELETE FROM predictions WHERE typeof(probability) = 'blob'
            """)
            print(f"  deleted {undecodable:,} undecodable row(s)")

        conn.commit()

        cursor.execute("SELECT typeof(probability), COUNT(*) FROM predictions GROUP BY 1")
        print("\nAfter repair:")
        for type_name, count in cursor.fetchall():
            print(f"  {type_name:8s} {count:>8,}")

        cursor.execute("""
            SELECT algorithm, COUNT(*), ROUND(AVG(probability), 2),
                   ROUND(MIN(probability), 2), ROUND(MAX(probability), 2)
            FROM predictions GROUP BY algorithm ORDER BY 3 DESC
        """)
        print("\nPer-algorithm distribution (now consistent):")
        print(f"  {'algorithm':<12}{'n':>10}{'mean':>9}{'min':>8}{'max':>8}")
        for algorithm, count, mean, low, high in cursor.fetchall():
            print(f"  {algorithm:<12}{count:>10,}{mean:>9}{low:>8}{high:>8}")

        return 0

    finally:
        conn.close()


if __name__ == '__main__':
    sys.exit(main())
