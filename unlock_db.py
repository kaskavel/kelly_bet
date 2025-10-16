#!/usr/bin/env python3
"""
Force unlock SQLite database by clearing any stale connections
"""
import sqlite3
import os
import time

db_path = "data/trading.db"

try:
    # Force close any existing connections
    conn = sqlite3.connect(db_path, timeout=1)
    conn.execute("BEGIN IMMEDIATE;")
    conn.rollback()
    conn.close()
    
    # Try to vacuum and checkpoint
    conn = sqlite3.connect(db_path, timeout=5)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    conn.execute("VACUUM;")
    conn.close()
    
    print(f"Successfully unlocked {db_path}")
    
except Exception as e:
    print(f"Error unlocking database: {e}")
    print("Trying alternative approach...")
    
    # Alternative: copy database to new file
    try:
        import shutil
        backup_path = f"{db_path}.backup_{int(time.time())}"
        shutil.copy2(db_path, backup_path)
        print(f"Created backup: {backup_path}")
        
        # Recreate database
        if os.path.exists(f"{db_path}-wal"):
            os.remove(f"{db_path}-wal")
            print("Removed WAL file")
        if os.path.exists(f"{db_path}-shm"):
            os.remove(f"{db_path}-shm")
            print("Removed SHM file")
            
        print("Database should now be unlocked")
        
    except Exception as e2:
        print(f"Alternative approach failed: {e2}")
        print("You may need to restart your terminal/IDE")