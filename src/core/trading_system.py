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
                if not await self.risk_manager.can_continue_trading():
                    self.logger.warning("Risk manager paused trading")
                    await asyncio.sleep(300)  # Wait 5 minutes before checking again
                    continue
                
                # CRITICAL: Monitor existing bets first - check and close positions that hit thresholds
                await self._monitor_existing_bets()
                
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
        bet_recommendation = self.kelly_calc.calculate_bet_size(
            probability=probability,
            current_price=current_price,
            available_capital=await self.portfolio.get_available_capital()
        )
        
        print(f"\n" + "="*80)
        print(f"KELLY CRITERION BET ANALYSIS - {symbol}")
        print(f"="*80)
        
        # Basic bet information
        print(f"Asset: {symbol}")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Available Capital: ${bet_recommendation.available_capital:,.2f}")
        
        # Probability analysis
        print(f"\nPROBABILITY ANALYSIS:")
        print(f"  Win Probability (p): {bet_recommendation.win_probability:.1%} ({probability:.2f}%)")
        print(f"  Loss Probability (q): {bet_recommendation.loss_probability:.1%}")
        
        # Threshold information
        win_threshold_pct = ((prediction.get('win_threshold', current_price) - current_price) / current_price) * 100
        loss_threshold_pct = ((prediction.get('loss_threshold', current_price) - current_price) / current_price) * 100
        print(f"\nTHRESHOLD SETUP:")
        print(f"  Win Target: +{win_threshold_pct:.1f}% (${prediction.get('win_threshold', current_price):.2f})")
        print(f"  Loss Stop: {loss_threshold_pct:.1f}% (${prediction.get('loss_threshold', current_price):.2f})")
        
        # Kelly formula breakdown
        print(f"\nKELLY FORMULA CALCULATION:")
        print(f"  Formula: f = (bp - q) / b")
        print(f"  where:")
        print(f"    b (odds ratio) = {bet_recommendation.kelly_formula_b:.3f} (win/loss ratio)")
        print(f"    p (win probability) = {bet_recommendation.kelly_formula_p:.3f}")
        print(f"    q (loss probability) = {bet_recommendation.kelly_formula_q:.3f}")
        print(f"  ")
        print(f"  Calculation: f = ({bet_recommendation.kelly_formula_b:.3f} × {bet_recommendation.kelly_formula_p:.3f} - {bet_recommendation.kelly_formula_q:.3f}) / {bet_recommendation.kelly_formula_b:.3f}")
        print(f"  Raw Kelly Fraction = {bet_recommendation.kelly_fraction_raw:.1%}")
        
        # Expected value breakdown
        print(f"\nEXPECTED VALUE ANALYSIS:")
        print(f"  Expected Win: {bet_recommendation.win_probability:.1%} × {bet_recommendation.win_amount_ratio:.1%} = {bet_recommendation.expected_win:.3f}")
        print(f"  Expected Loss: {bet_recommendation.loss_probability:.1%} × {bet_recommendation.loss_amount_ratio:.1%} = {bet_recommendation.expected_loss:.3f}")
        print(f"  Net Expected Value: {bet_recommendation.expected_win:.3f} - {bet_recommendation.expected_loss:.3f} = {bet_recommendation.expected_value:.3f}")
        
        # Risk adjustments
        print(f"\nRISK ADJUSTMENTS:")
        if bet_recommendation.kelly_fraction_raw != bet_recommendation.fraction_of_capital:
            kelly_multiplier = self.config.get('trading', {}).get('kelly_fraction', 0.25)
            print(f"  Conservative Multiplier: {kelly_multiplier:.1%} (reduces risk)")
            print(f"  Max Position Size Cap: {self.config.get('trading', {}).get('max_bet_fraction', 0.1):.1%}")
            print(f"  After Adjustments: {bet_recommendation.fraction_of_capital:.1%}")
        else:
            print(f"  No adjustments applied")
        
        # Final recommendation
        print(f"\nFINAL RECOMMENDATION:")
        print(f"  Bet Amount: ${bet_recommendation.recommended_amount:,.2f}")
        print(f"  Position Size: {bet_recommendation.fraction_of_capital:.1%} of capital")
        print(f"  Confidence Level: {bet_recommendation.confidence_level}")
        
        if bet_recommendation.risk_warning:
            print(f"\nWARNINGS:")
            for warning in bet_recommendation.risk_warning.split(';'):
                print(f"  WARNING: {warning.strip()}")
        
        print(f"\n" + "="*80)
        
        confirm = input("Proceed with this bet? (y/N): ").strip().lower()
        
        if confirm == 'y':
            if bet_recommendation.is_favorable and bet_recommendation.recommended_amount > 0:
                await self._place_bet(prediction)
            else:
                print("Bet not favorable - Kelly recommends no bet")
        else:
            print("Bet cancelled")
    
    async def _place_bet(self, prediction: Dict):
        """Place the actual bet"""
        try:
            bet_id = await self.portfolio.place_bet(prediction)
            self.logger.info(f"Bet placed successfully: {bet_id}")
            print(f"Bet placed: {prediction['symbol']} (ID: {bet_id})")
            
        except Exception as e:
            self.logger.error(f"Failed to place bet: {e}")
            print(f"Failed to place bet: {e}")
    
    async def _monitor_existing_bets(self):
        """Monitor existing alive bets and close positions that hit win/loss thresholds"""
        try:
            self.logger.info("Checking existing bets for threshold triggers...")
            
            # Get all alive bets
            alive_bets = await self.portfolio.get_alive_bets()
            
            if not alive_bets:
                self.logger.debug("No alive bets to monitor")
                return
                
            self.logger.info(f"Monitoring {len(alive_bets)} active positions")
            
            # Get current prices for all symbols with alive bets
            symbols_to_check = list(set(bet.symbol for bet in alive_bets))
            current_prices = {}
            
            for symbol in symbols_to_check:
                try:
                    recent_data = await self.market_data.get_stock_data(symbol, days=1)
                    if not recent_data.empty:
                        current_prices[symbol] = float(recent_data['Close'].iloc[-1])
                        self.logger.debug(f"Current price for {symbol}: ${current_prices[symbol]:.2f}")
                    else:
                        self.logger.warning(f"No recent data available for {symbol}")
                except Exception as e:
                    self.logger.error(f"Error fetching current price for {symbol}: {e}")
            
            # Check each bet against thresholds
            bets_closed = 0
            for bet in alive_bets:
                if bet.symbol not in current_prices:
                    self.logger.warning(f"Skipping {bet.symbol} - no current price available")
                    continue
                
                current_price = current_prices[bet.symbol]
                
                # Calculate current return
                if bet.bet_type == 'long':
                    current_return_pct = ((current_price - bet.entry_price) / bet.entry_price) * 100
                    hit_win_threshold = current_price >= bet.win_threshold
                    hit_loss_threshold = current_price <= bet.loss_threshold
                else:  # short
                    current_return_pct = ((bet.entry_price - current_price) / bet.entry_price) * 100
                    hit_win_threshold = current_price <= bet.win_threshold
                    hit_loss_threshold = current_price >= bet.loss_threshold
                
                self.logger.debug(f"Bet {bet.bet_id} ({bet.symbol}): Entry=${bet.entry_price:.2f}, "
                                f"Current=${current_price:.2f}, Return={current_return_pct:+.2f}%")
                
                # Check if we need to close the position
                should_close = False
                close_reason = ""
                
                if hit_win_threshold:
                    should_close = True
                    close_reason = f"WIN THRESHOLD HIT: {current_return_pct:+.2f}% (target: {((bet.win_threshold/bet.entry_price - 1) * 100):+.2f}%)"
                elif hit_loss_threshold:
                    should_close = True
                    close_reason = f"LOSS THRESHOLD HIT: {current_return_pct:+.2f}% (stop: {((bet.loss_threshold/bet.entry_price - 1) * 100):+.2f}%)"
                
                if should_close:
                    self.logger.info(f"CLOSING POSITION - {bet.symbol}: {close_reason}")
                    
                    try:
                        # Close the bet
                        await self.portfolio.close_bet(bet.bet_id, current_price, close_reason)
                        bets_closed += 1
                        
                        # Print notification if in manual mode (user is watching)
                        if self.mode == 'manual':
                            print(f"\n*** POSITION CLOSED ***")
                            print(f"Symbol: {bet.symbol}")
                            print(f"Reason: {close_reason}")
                            print(f"Entry: ${bet.entry_price:.2f} -> Exit: ${current_price:.2f}")
                            print(f"Amount: ${bet.amount:.2f}")
                        
                    except Exception as e:
                        self.logger.error(f"Error closing bet {bet.bet_id}: {e}")
            
            if bets_closed > 0:
                self.logger.info(f"Successfully closed {bets_closed} position(s)")
            else:
                self.logger.debug("No positions required closing at this time")
                
        except Exception as e:
            self.logger.error(f"Error in bet monitoring: {e}")
    
    async def _cleanup(self):
        """Clean up resources"""
        self.logger.info("Cleaning up resources...")
        
        if hasattr(self, 'market_data'):
            await self.market_data.cleanup()
        if hasattr(self, 'portfolio'):
            await self.portfolio.cleanup()
        
        self.logger.info("Cleanup complete")