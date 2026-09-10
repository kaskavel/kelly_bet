import sqlite3

conn = sqlite3.connect('data/trading.db')
cursor = conn.cursor()

# Get the entry amount for the cancelled bet
cursor.execute("SELECT amount FROM cash_transactions WHERE description = 'Bet entry: 8053.T'")
entry_amount = cursor.fetchone()

if entry_amount:
    refund = -entry_amount[0]

    # Get current balance
    cursor.execute('SELECT SUM(amount) FROM cash_transactions')
    current_balance = cursor.fetchone()[0]

    # New balance after refund
    balance_after = current_balance + refund

    cursor.execute(
        'INSERT INTO cash_transactions (amount, balance_after, description, transaction_type, timestamp) VALUES (?, ?, ?, ?, datetime("now"))',
        (refund, balance_after, 'Refund: 8053.T bet cancelled', 'refund')
    )
    conn.commit()
    print(f'Refunded: {refund:.2f}')
    print(f'Balance after refund: {balance_after:.2f}')
else:
    print("No entry transaction found")

conn.close()
