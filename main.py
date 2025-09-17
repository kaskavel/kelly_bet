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
    
    # Main action flags (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        '--mode', 
        choices=['manual', 'automated'], 
        help='Trading mode: manual (user selects bets) or automated (system places bets)'
    )
    action_group.add_argument(
        '--livebets',
        action='store_true',
        help='Monitor active bets with auto-settlement (continuous monitoring with 5-min intervals)'
    )
    action_group.add_argument(
        '--bets',
        action='store_true',
        help='Show betting history and performance'
    )
    action_group.add_argument(
        '--dashboard',
        action='store_true',
        help='Launch web-based trading dashboard (recommended)'
    )
    
    # Trading system options
    parser.add_argument(
        '--threshold', 
        type=float, 
        default=60.0,
        help='Minimum probability threshold for automated betting (default: 60%%)'
    )
    
    # Live bets options
    parser.add_argument(
        '--refresh', 
        type=int, 
        default=300,
        help='Monitor interval in seconds for --livebets (default: 300 = 5 minutes, 0 = single check)'
    )
    
    # Bet history options
    parser.add_argument(
        '--limit', 
        type=int, 
        default=50,
        help='Number of recent bets to show for --bets (default: 50)'
    )
    parser.add_argument(
        '--filter', 
        choices=['all', 'won', 'lost', 'alive'],
        default='all',
        help='Filter bets by status for --bets'
    )
    parser.add_argument(
        '--stats',
        action='store_true', 
        help='Show detailed performance statistics for --bets'
    )
    
    # Global arguments
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
    
    # Handle different actions
    if args.dashboard:
        await handle_dashboard_command(args, logger)
    elif args.livebets:
        await handle_livebets_command(args, logger)
    elif args.bets:
        await handle_bets_command(args, logger)
    elif args.mode:
        await handle_trading_command(args, logger)
    else:
        print("Please specify an action. Use --help for available options.")
        print("\nExample usage:")
        print("  python main.py --dashboard                      # Launch web UI (recommended)")
        print("  python main.py --mode manual")
        print("  python main.py --mode automated --threshold 65")
        print("  python main.py --livebets")
        print("  python main.py --livebets --refresh 30")
        print("  python main.py --bets")
        print("  python main.py --bets --stats --limit 100")


async def handle_trading_command(args, logger):
    """Handle trading system execution"""
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


async def handle_livebets_command(args, logger):
    """Handle live bets monitoring"""
    from src.cli.bet_monitor import BetMonitor
    
    try:
        monitor = BetMonitor(args.config)
        await monitor.show_live_bets(refresh_interval=args.refresh)
        
    except KeyboardInterrupt:
        logger.info("Live bets monitoring stopped")
    except Exception as e:
        logger.error(f"Error in live bets monitoring: {e}", exc_info=True)
        sys.exit(1)


async def handle_bets_command(args, logger):
    """Handle bet history analysis"""
    from src.cli.bet_analyzer import BetAnalyzer
    
    try:
        analyzer = BetAnalyzer(args.config)
        await analyzer.show_bet_history(
            limit=args.limit,
            status_filter=args.filter,
            show_stats=args.stats
        )
        
    except Exception as e:
        logger.error(f"Error in bet analysis: {e}", exc_info=True)
        sys.exit(1)


async def handle_dashboard_command(args, logger):
    """Handle dashboard launch"""
    import subprocess
    import os
    
    logger.info("Launching Kelly Trading Dashboard...")
    
    try:
        # Get project directory
        project_dir = os.path.dirname(os.path.abspath(__file__))
        dashboard_script = os.path.join(project_dir, "run_dashboard.py")
        
        # Launch dashboard
        subprocess.run([sys.executable, dashboard_script], check=True)
        
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
    except Exception as e:
        logger.error(f"Error launching dashboard: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())