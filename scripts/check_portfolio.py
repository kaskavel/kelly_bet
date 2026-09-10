import sqlite3

conn = sqlite3.connect('data/trading.db')
cursor = conn.cursor()

# Check cash balance
cursor.execute('SELECT SUM(amount) FROM cash_transactions')
cash = cursor.fetchone()[0]
print(f'Cash balance: ${cash:.2f}')

# Check active bets
cursor.execute('SELECT symbol, current_price, shares, amount, currency FROM bets WHERE status="alive"')
print('\nActive bets:')
print('SYMBOL | CURRENT_PRICE | SHARES | AMOUNT | CURRENCY | VALUE')
total_value = 0
for row in cursor.fetchall():
    symbol, current_price, shares, amount, currency = row
    value = current_price * shares
    total_value += value
    print(f'{symbol} | ${current_price:.2f} | {shares:.4f} | ${amount:.2f} | {currency or "USD"} | ${value:.2f}')

print(f'\nTotal active bets value (DB): ${total_value:.2f}')
print(f'Total capital (DB): ${cash + total_value:.2f}')

conn.close()
