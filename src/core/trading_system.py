"""
Core trading system orchestrator
Manages the main trading loop for both manual and automated modes.
"""

import asyncio
import logging
import yaml
from typing import Dict, List, Optional
from pathlib import Path

from ..data.market_data import MarketDataManager
from ..prediction.predictor import PredictionEngine
from ..kelly.calculator import KellyCalculator
from ..portfolio.manager import PortfolioManager
from ..risk.manager import RiskManager
from ..utils.asset_selector import AssetSelector


class TradingSystem:
    def __init__(self, config_path: str, mode: str, auto_threshold: float):
        self.mode = mode
        self.auto_threshold = auto_threshold
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.market_data = MarketDataManager(self.config)
        self.predictor = PredictionEngine(self.config)
        self.kelly_calc = KellyCalculator(self.config)
        self.portfolio = PortfolioManager(self.config)
        self.risk_manager = RiskManager(self.config)
        self.asset_selector = AssetSelector(self.config)
        
        # System state
        self.running = True
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.error(f"Config file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Invalid YAML config: {e}")
            raise
    
    async def run(self):
        """Main trading loop"""
        self.logger.info(f"Starting trading system in {self.mode} mode")
        
        # Initialize all components
        await self._initialize_components()
        
        try:
            while self.running:
                # Check risk conditions
                if not self.risk_manager.can_continue_trading():
                    self.logger.warning("Risk manager paused trading")
                    await asyncio.sleep(300)  # Wait 5 minutes before checking again
                    continue
                
                # Get asset predictions and rankings
                predictions = await self._get_predictions()
                
                if not predictions:
                    self.logger.info("No valid predictions, waiting...")
                    await asyncio.sleep(self.config['system']['polling_interval'])
                    continue
                
                # Process based on mode
                if self.mode == 'manual':
                    await self._handle_manual_mode(predictions)
                else:
                    await self._handle_automated_mode(predictions)
                    
                # Wait before next cycle
                await asyncio.sleep(self.config['system']['polling_interval'])
                
        except KeyboardInterrupt:
            self.logger.info("Shutdown signal received")
            self.running = False
        finally:
            await self._cleanup()
    
    async def _initialize_components(self):
        """Initialize all system components"""
        self.logger.info("Initializing system components...")
        
        await self.market_data.initialize()
        await self.predictor.initialize()
        await self.portfolio.initialize()
        await self.risk_manager.initialize()
        
        self.logger.info("All components initialized successfully")
    
    async def _get_predictions(self) -> List[Dict]:
        """Get predictions for all assets and rank by probability"""
        self.logger.info("Fetching market data and generating predictions...")
        
        # Get latest market data
        assets = await self.asset_selector.get_all_assets()
        market_data = await self.market_data.get_latest_data(assets)
        
        # Generate predictions
        predictions = await self.predictor.predict_all(market_data)
        
        # Filter and rank by probability
        valid_predictions = [
            p for p in predictions 
            if p['probability'] is not None and p['probability'] > 0
        ]
        
        # Sort by probability descending
        valid_predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        self.logger.info(f"Generated {len(valid_predictions)} valid predictions")
        return valid_predictions
    
    async def _handle_manual_mode(self, predictions: List[Dict]):
        """Handle manual mode interaction"""
        top_n = self.config['trading']['top_n_display']
        top_predictions = predictions[:top_n]
        
        self.logger.info(f"Top {len(top_predictions)} investment opportunities:")
        
        # Display top predictions
        print(f"\n{'='*60}")
        print("TOP INVESTMENT OPPORTUNITIES")
        print(f"{'='*60}")
        
        for i, pred in enumerate(top_predictions, 1):
            print(f"{i:2d}. {pred['symbol']:10s} | "
                  f"Probability: {pred['probability']:6.2f}% | "
                  f"Price: ${pred['current_price']:8.2f}")
        
        print(f"{'='*60}")
        
        # Get user selection
        try:
            choice = input(f"\nSelect bet (1-{len(top_predictions)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                self.running = False
                return
            
            bet_index = int(choice) - 1
            if 0 <= bet_index < len(top_predictions):
                selected_prediction = top_predictions[bet_index]
                
                # Show bet details and confirm
                await self._confirm_and_place_bet(selected_prediction)
            else:
                print("Invalid selection")
                
        except (ValueError, KeyboardInterrupt):
            print("Invalid input or cancelled")
    
    async def _handle_automated_mode(self, predictions: List[Dict]):
        """Handle automated mode logic"""
        if not predictions:
            self.logger.info("No predictions available")
            return
        
        best_prediction = predictions[0]
        best_prob = best_prediction['probability']
        
        self.logger.info(f"Best opportunity: {best_prediction['symbol']} "
                        f"with {best_prob:.2f}% probability")
        
        # Check thresholds
        if best_prob < 50.0:
            self.logger.info("Best probability <50%, no bets placed")
            return
        
        if best_prob >= self.auto_threshold:
            self.logger.info(f"Probability {best_prob:.2f}% >= threshold {self.auto_threshold}%, "
                           f"placing automatic bet")
            await self._place_bet(best_prediction)
        else:
            self.logger.info(f"Best probability {best_prob:.2f}% below threshold {self.auto_threshold}%")
    
    async def _confirm_and_place_bet(self, prediction: Dict):
        """Confirm bet with user and place if approved"""
        symbol = prediction['symbol']
        probability = prediction['probability']
        current_price = prediction['current_price']
        
        # Calculate bet size using Kelly
        bet_size = self.kelly_calc.calculate_bet_size(
            probability=probability / 100.0,  # Convert to decimal
            current_price=current_price,
            available_capital=await self.portfolio.get_available_capital()
        )
        
        print(f"\nBET DETAILS:")
        print(f"Asset: {symbol}")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Win Probability: {probability:.2f}%")
        print(f"Recommended Bet Size: ${bet_size:.2f}")
        
        confirm = input("Proceed with this bet? (y/N): ").strip().lower()
        
        if confirm == 'y':
            await self._place_bet(prediction)
        else:
            print("Bet cancelled")
    
    async def _place_bet(self, prediction: Dict):
        """Place the actual bet"""
        try:
            bet_id = await self.portfolio.place_bet(prediction)
            self.logger.info(f"Bet placed successfully: {bet_id}")
            print(f"✓ Bet placed: {prediction['symbol']} (ID: {bet_id})")
            
        except Exception as e:
            self.logger.error(f"Failed to place bet: {e}")
            print(f"✗ Failed to place bet: {e}")
    
    async def _cleanup(self):
        """Clean up resources"""
        self.logger.info("Cleaning up resources...")
        
        if hasattr(self, 'market_data'):
            await self.market_data.cleanup()
        if hasattr(self, 'portfolio'):
            await self.portfolio.cleanup()
        
        self.logger.info("Cleanup complete")