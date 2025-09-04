#!/usr/bin/env python3
"""
Kelly Criterion Trading System
Main entry point for manual and automated trading modes.
"""

import argparse
import asyncio
import logging
import sys
from typing import Optional

from src.core.trading_system import TradingSystem
from src.utils.logger import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description='Kelly Criterion Trading System')
    parser.add_argument(
        '--mode', 
        choices=['manual', 'automated'], 
        required=True,
        help='Operating mode: manual (user selects bets) or automated (system places bets)'
    )
    parser.add_argument(
        '--threshold', 
        type=float, 
        default=60.0,
        help='Minimum probability threshold for automated betting (default: 60%%)'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--log-level', 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
        default='INFO',
        help='Logging level'
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting Kelly Trading System in {args.mode} mode")
    
    try:
        # Initialize trading system
        trading_system = TradingSystem(
            config_path=args.config,
            mode=args.mode,
            auto_threshold=args.threshold
        )
        
        # Start the system
        await trading_system.run()
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down gracefully...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("System shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())