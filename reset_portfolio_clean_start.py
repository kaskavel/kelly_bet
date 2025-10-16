import sqlite3
import pandas as pd
from datetime import datetime

def reset_portfolio_for_clean_start():
    """
    Reset portfolio to clean start while preserving market data
    - Keep: assets, price_data tables (all stock/crypto historical data)
    - Clear: bets, cash_transactions, portfolio_history
    - Reset: Initial capital to $10,000
    """

    print('=== PORTFOLIO CLEAN START RESET ===\n')

    conn = sqlite3.connect('data/trading.db')
    cursor = conn.cursor()

    try:
        # First, let's see what we're about to clear
        print('CURRENT DATA TO BE CLEARED:')

        # Count bets
        bets_count = cursor.execute('SELECT COUNT(*) FROM bets').fetchone()[0]
        print(f'Bets: {bets_count} records')

        # Count cash transactions
        cash_count = cursor.execute('SELECT COUNT(*) FROM cash_transactions').fetchone()[0]
        print(f'Cash transactions: {cash_count} records')

        # Count portfolio history
        portfolio_count = cursor.execute('SELECT COUNT(*) FROM portfolio_history').fetchone()[0]
        print(f'Portfolio history: {portfolio_count} records')

        print('\nDATA TO BE PRESERVED:')

        # Count assets
        assets_count = cursor.execute('SELECT COUNT(*) FROM assets').fetchone()[0]
        print(f'Assets: {assets_count} records (stocks + crypto)')

        # Count price data
        price_count = cursor.execute('SELECT COUNT(*) FROM price_data').fetchone()[0]
        print(f'Price data: {price_count} records (historical prices)')

        # Show date range of price data
        date_range = cursor.execute('''
            SELECT MIN(timestamp) as earliest, MAX(timestamp) as latest
            FROM price_data
        ''').fetchone()
        print(f'Price data range: {date_range[0]} to {date_range[1]}')

        print(f'\n' + '='*50)
        response = input('Proceed with portfolio reset? (yes/no): ')

        if response.lower() != 'yes':
            print('Reset cancelled.')
            return

        print('\nResetting portfolio data...')

        # Clear betting and portfolio data
        cursor.execute('DELETE FROM bets')
        print('✓ Cleared all bets')

        cursor.execute('DELETE FROM cash_transactions')
        print('✓ Cleared all cash transactions')

        cursor.execute('DELETE FROM portfolio_history')
        print('✓ Cleared all portfolio history')

        # Reset auto-increment counters
        cursor.execute('DELETE FROM sqlite_sequence WHERE name IN ("bets", "cash_transactions", "portfolio_history")')
        print('✓ Reset ID counters')

        # Create initial capital transaction
        initial_capital = 10000.0
        timestamp = datetime.now().isoformat()

        cursor.execute('''
            INSERT INTO cash_transactions (
                timestamp, amount, balance_after, description,
                bet_id, transaction_type
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            timestamp, initial_capital, initial_capital,
            'Initial capital - Fresh start', None, 'initial_capital'
        ))
        print(f'✓ Created initial capital transaction: ${initial_capital:,.2f}')

        # Commit all changes
        conn.commit()

        print(f'\n' + '='*50)
        print('PORTFOLIO RESET COMPLETE!')
        print(f'✓ Starting fresh with ${initial_capital:,.2f}')
        print('✓ All market data preserved')
        print('✓ Portfolio tracking with fixed synchronization')
        print(f'\nYou can now:')
        print('1. Restart the dashboard')
        print('2. Begin placing new bets with corrected portfolio tracking')
        print('3. All future portfolio calculations will be accurate')

    except Exception as e:
        conn.rollback()
        print(f'❌ Error during reset: {e}')
        print('Portfolio reset failed - no changes made')

    finally:
        conn.close()

if __name__ == "__main__":
    reset_portfolio_for_clean_start()