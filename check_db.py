#!/usr/bin/env python3
import sqlite3

conn = sqlite3.connect('data/trading.db')
cursor = conn.cursor()

# Check what bets exist
cursor.execute('SELECT bet_id, symbol, status, entry_price, amount, created_at FROM bets')
rows = cursor.fetchall()

print('Current bets in database:')
for row in rows:
    print(f"ID: {row[0]}, Symbol: {row[1]}, Status: {row[2]}, Entry: ${row[3]:.2f}, Amount: ${row[4]:.2f}, Created: {row[5]}")

print(f"\nTotal bets: {len(rows)}")

# Check alive bets specifically
cursor.execute("SELECT COUNT(*) FROM bets WHERE status = 'alive'")
alive_count = cursor.fetchone()[0]
print(f"Alive bets: {alive_count}")

conn.close()