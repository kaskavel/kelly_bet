#!/usr/bin/env python3
"""
Script to fix binary probability data in bet_predictions table.

This script converts any binary-encoded probabilities back to proper REAL values.
"""

import sqlite3
import struct
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def fix_binary_probabilities(db_path: str):
    """Fix binary probability values in bet_predictions table"""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all records from bet_predictions
    cursor.execute('SELECT prediction_id, probability FROM bet_predictions')
    records = cursor.fetchall()

    fixed_count = 0
    error_count = 0

    for prediction_id, prob in records:
        if isinstance(prob, bytes):
            try:
                # Decode binary float (little-endian, 4 bytes)
                if len(prob) == 4:
                    decoded_prob = struct.unpack('<f', prob)[0]
                    cursor.execute(
                        'UPDATE bet_predictions SET probability = ? WHERE prediction_id = ?',
                        (decoded_prob, prediction_id)
                    )
                    fixed_count += 1
                    print(f"Fixed prediction_id {prediction_id}: {prob.hex()} -> {decoded_prob:.4f}")
                else:
                    print(f"ERROR: prediction_id {prediction_id} has unexpected byte length: {len(prob)}")
                    error_count += 1
            except Exception as e:
                print(f"ERROR: Failed to decode prediction_id {prediction_id}: {e}")
                error_count += 1

    conn.commit()
    conn.close()

    print(f"\nSummary:")
    print(f"  Records fixed: {fixed_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total checked: {len(records)}")

    return fixed_count, error_count


if __name__ == '__main__':
    # Default database path
    db_path = 'data/trading.db'

    if len(sys.argv) > 1:
        db_path = sys.argv[1]

    print(f"Fixing binary probabilities in: {db_path}")
    print("-" * 60)

    fixed, errors = fix_binary_probabilities(db_path)

    if fixed > 0:
        print("\nDatabase has been updated. Please restart the dashboard.")
    elif errors > 0:
        print("\nSome errors occurred. Please review the output above.")
    else:
        print("\nNo binary probabilities found. Database is clean.")
