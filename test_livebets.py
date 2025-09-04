#!/usr/bin/env python3
"""
Quick test of enhanced livebets monitoring (single run)
"""

import asyncio
import yaml
from src.cli.bet_monitor import BetMonitor

async def test_livebets():
    """Test the enhanced live bets monitoring"""
    monitor = BetMonitor('config/config.yaml')
    
    # Single run test (refresh=0)
    await monitor.show_live_bets(refresh_interval=0)

if __name__ == '__main__':
    asyncio.run(test_livebets())