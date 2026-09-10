"""
Core trading system orchestrator
Manages the main trading loop for both manual and automated modes.
"""

import asyncio
import logging
import yaml
from datetime import datetime
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

        # Asset metadata cache (symbol -> {type, currency}), populated on startup so
        # the monitoring loop can price positions in USD without re-deriving it.
        self._asset_meta: Dict[str, Dict[str, str]] = {}

        # Time barrier: force-close positions that reach neither price barrier.
        self.max_hold_days = self.config.get('trading', {}).get('max_hold_days', 15)

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
                # Monitor existing bets FIRST so risk is assessed against fresh
                # marked-to-market equity, and so exits happen even when risk
                # controls have halted new entries.
                await self._monitor_existing_bets()

                # Check risk conditions. The portfolio summary MUST be passed:
                # can_continue_trading() returns on a shortcut when it is None, which
                # bypassed drawdown, loss-streak, exposure and minimum-capital checks
                # entirely -- every risk control in the system was dead code.
                portfolio_summary = await self.portfolio.get_portfolio_summary()
                if not await self.risk_manager.can_continue_trading(portfolio_summary):
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

        # Cache asset metadata (type and quote currency) for the monitoring loop.
        try:
            assets = await self.asset_selector.get_all_assets()
            self._asset_meta = {
                asset['symbol']: {
                    'type': asset.get('type', 'stock'),
                    'currency': asset.get('currency', 'USD'),
                }
                for asset in assets
            }
            self.logger.info(f"Cached metadata for {len(self._asset_meta)} assets")
        except Exception as e:
            self.logger.error(f"Could not cache asset metadata: {e}")

        # Refuse to trade on books that do not balance -- a sizing decision made
        # against a wrong equity figure is worse than no decision.
        recon = await self.portfolio.reconcile()
        if not recon['balanced']:
            raise ValueError(
                f"Refusing to start: portfolio does not reconcile "
                f"(cash drift ${recon['cash_drift']:.2f}). Investigate before trading."
            )

        self.logger.info("All components initialized successfully")

    def asset_selector_type_for(self, symbol: str) -> str:
        """Asset class for a symbol, from cached metadata."""
        meta = self._asset_meta.get(symbol)
        if meta:
            return meta['type']
        return 'forex' if symbol.endswith('=X') else 'stock'

    def asset_currency_for(self, symbol: str) -> str:
        """Quote currency for a symbol, from cached metadata."""
        meta = self._asset_meta.get(symbol)
        if meta:
            return meta['currency']
        return self.market_data.currency_converter.currency_for_symbol(symbol)
    
    async def _get_predictions(self) -> List[Dict]:
        """Get predictions for all assets and rank by probability"""
        self.logger.info("Fetching market data and generating predictions...")
        
        # Get latest market data. get_latest_data() normalises everything to USD and
        # drops assets whose currency cannot be converted.
        assets = await self.asset_selector.get_all_assets()
        market_data = await self.market_data.get_latest_data(assets)

        # Create asset metadata mappings
        asset_type_map = {asset['symbol']: asset['type'] for asset in assets}
        currency_map = {asset['symbol']: asset.get('currency', 'USD') for asset in assets}

        # Generate predictions
        predictions = await self.predictor.predict_all(market_data)

        # Enrich predictions with asset metadata so the portfolio records the real
        # asset class and original currency rather than defaulting both.
        for prediction in predictions:
            symbol = prediction['symbol']
            prediction['asset_type'] = asset_type_map.get(symbol, 'unknown')
            prediction['currency'] = currency_map.get(symbol, 'USD')

        # Filter and rank by probability
        valid_predictions = [
            p for p in predictions 
            if p['probability'] is not None and p['probability'] > 0
        ]
        
        # Sort by probability descending
        valid_predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        self.logger.info(f"Generated {len(valid_predictions)} valid predictions")
        return valid_predictions
    
    async def _get_active_bet_symbols(self) -> List[str]:
        """Get list of symbols that currently have active bets"""
        try:
            alive_bets = await self.portfolio.get_alive_bets()
            active_symbols = [bet.symbol for bet in alive_bets]
            self.logger.debug(f"Active bet symbols: {active_symbols}")
            return active_symbols
        except Exception as e:
            self.logger.error(f"Error getting active bet symbols: {e}")
            return []
    
    async def _handle_manual_mode(self, predictions: List[Dict]):
        """Handle manual mode interaction"""
        top_n = self.config['trading']['top_n_display']
        top_predictions = predictions[:top_n]
        
        # Get active bet symbols to show warnings
        active_symbols = await self._get_active_bet_symbols()
        
        self.logger.info(f"Top {len(top_predictions)} investment opportunities:")
        
        # Display top predictions
        print(f"\n{'='*60}")
        print("TOP INVESTMENT OPPORTUNITIES")
        print(f"{'='*60}")
        
        for i, pred in enumerate(top_predictions, 1):
            duplicate_warning = ""
            if pred['symbol'] in active_symbols:
                duplicate_warning = " [ACTIVE BET]"
            
            asset_type = pred.get('asset_type', 'unknown').upper()
            
            print(f"{i:2d}. {pred['symbol']:10s} | "
                  f"{asset_type:6s} | "
                  f"Probability: {pred['probability']:6.2f}% | "
                  f"Price: ${pred['current_price']:8.2f}{duplicate_warning}")
        
        if active_symbols:
            print(f"\nNOTE: [ACTIVE BET] indicates you already have an active bet for this symbol.")
            print(f"Active bets: {', '.join(active_symbols)}")
        
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
                
                # Check for duplicate bet and confirm
                if selected_prediction['symbol'] in active_symbols:
                    print(f"\n⚠️  WARNING: You already have an active bet for {selected_prediction['symbol']}")
                    confirm_duplicate = input("Do you want to place another bet on the same symbol? (y/N): ").strip().lower()
                    
                    if confirm_duplicate != 'y':
                        print("Bet cancelled - avoiding duplicate position")
                        return
                    
                    print("Proceeding with duplicate bet as requested...")
                
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
        
        # Display top 10 predictions for reference (like manual mode)
        top_n = self.config['trading']['top_n_display']
        top_predictions = predictions[:top_n]
        
        print(f"\n{'='*70}")
        print("TOP INVESTMENT OPPORTUNITIES (AUTOMATED MODE)")
        print(f"{'='*70}")
        
        for i, pred in enumerate(top_predictions, 1):
            asset_type = pred.get('asset_type', 'unknown').upper()
            print(f"{i:2d}. {pred['symbol']:10s} | "
                  f"{asset_type:6s} | "
                  f"Probability: {pred['probability']:6.2f}% | "
                  f"Price: ${pred['current_price']:8.2f}")
        
        print(f"{'='*70}")
        
        # Get active bet symbols to prevent duplicates
        active_symbols = await self._get_active_bet_symbols()
        
        # Filter out predictions for symbols with active bets
        available_predictions = [
            p for p in predictions 
            if p['symbol'] not in active_symbols
        ]
        
        if not available_predictions:
            if active_symbols:
                self.logger.info(f"All top predictions have active bets ({len(active_symbols)} active). "
                               f"Active symbols: {', '.join(active_symbols)}")
                print(f"\nAll top predictions have active bets. Active symbols: {', '.join(active_symbols)}")
            else:
                self.logger.info("No predictions available")
            return
        
        best_prediction = available_predictions[0]
        best_prob = best_prediction['probability']
        best_asset_type = best_prediction.get('asset_type', 'unknown').upper()
        
        if active_symbols:
            self.logger.info(f"Best opportunity (excluding {len(active_symbols)} active bets): "
                           f"{best_prediction['symbol']} ({best_asset_type}) with {best_prob:.2f}% probability")
            print(f"\nBest available opportunity (excluding {len(active_symbols)} active bets):")
        else:
            self.logger.info(f"Best opportunity: {best_prediction['symbol']} ({best_asset_type}) "
                           f"with {best_prob:.2f}% probability")
            print(f"\nBest opportunity:")
        
        print(f">>> {best_prediction['symbol']} ({best_asset_type}) - {best_prob:.2f}% probability")

        # The hard floor is BREAK-EVEN for this bet's own barriers, not 50%. For a
        # barrier bet, 50% was never the neutral point: the driftless geometry is
        # around 40% and the fee-adjusted break-even for a 5%/3% bet is 43.75%.
        # Comparing against 50% both rejected genuinely favourable bets and, far
        # worse, made a "60% probability" read as a 10-point edge when it was 16.
        break_even_pct = best_prediction.get('break_even_pct')
        if break_even_pct is None:
            break_even_pct = self.kelly_calc.break_even_probability * 100.0

        if best_prob < break_even_pct:
            self.logger.info(f"Best probability {best_prob:.2f}% below break-even "
                             f"{break_even_pct:.2f}%, no bets placed")
            print(f"Probability {best_prob:.2f}% < break-even {break_even_pct:.2f}% "
                  f"- no bet placed")
            return

        if not best_prediction.get('is_calibrated', True):
            self.logger.warning(
                "Betting on an UNCALIBRATED score. Run scripts/backtest.py then "
                "scripts/fit_calibration.py before trusting these as probabilities.")

        if best_prob >= self.auto_threshold:
            self.logger.info(f"Probability {best_prob:.2f}% >= threshold {self.auto_threshold}%, "
                           f"placing automatic bet")
            print(f"Probability {best_prob:.2f}% >= threshold {self.auto_threshold}% - placing automatic bet...")
            await self._place_bet(best_prediction)
        else:
            self.logger.info(f"Best probability {best_prob:.2f}% below threshold {self.auto_threshold}%")
            print(f"Probability {best_prob:.2f}% < threshold {self.auto_threshold}% - no bet placed")
    
    async def _confirm_and_place_bet(self, prediction: Dict):
        """Confirm bet with user and place if approved"""
        symbol = prediction['symbol']
        probability = prediction['probability']
        current_price = prediction['current_price']
        
        # Calculate bet size using Kelly, on THIS asset's barriers. Passing them
        # explicitly matters in volatility mode: sizing must price the same bet the
        # probability was estimated for.
        bet_recommendation = self.kelly_calc.calculate_bet_size(
            probability=probability,
            current_price=current_price,
            available_capital=await self.portfolio.get_available_capital(),
            win_threshold=prediction.get('win_threshold'),
            loss_threshold=prediction.get('loss_threshold'),
        )
        
        print(f"\n" + "="*80)
        print(f"KELLY CRITERION BET ANALYSIS - {symbol}")
        print(f"="*80)
        
        # Basic bet information
        print(f"Asset: {symbol}")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Available Capital: ${bet_recommendation.available_capital:,.2f}")
        
        # Probability analysis, against the break-even that actually applies to this
        # bet's barriers. 50% is not the neutral point for a barrier bet.
        break_even_pct = self.kelly_calc.break_even_probability * 100.0
        calibrated_note = "" if prediction.get('is_calibrated', True) else "  [UNCALIBRATED SCORE]"
        print(f"\nPROBABILITY ANALYSIS:")
        print(f"  Win Probability (p): {bet_recommendation.win_probability:.1%} "
              f"({probability:.2f}%){calibrated_note}")
        print(f"  Loss Probability (q): {bet_recommendation.loss_probability:.1%}")
        print(f"  Break-even for these barriers: {break_even_pct:.2f}% "
              f"(margin {probability - break_even_pct:+.2f} pts)")

        # Threshold information. win_threshold/loss_threshold are PERCENTAGES, scaled
        # to this asset's own volatility when barrier.mode is "volatility".
        win_threshold_pct = prediction.get('win_threshold')
        loss_threshold_pct = prediction.get('loss_threshold')
        print(f"\nTHRESHOLD SETUP:")
        if win_threshold_pct is not None and loss_threshold_pct is not None:
            win_price = current_price * (1 + win_threshold_pct / 100.0)
            loss_price = current_price * (1 - loss_threshold_pct / 100.0)
            print(f"  Win Target: +{win_threshold_pct:.2f}% (${win_price:.2f})")
            print(f"  Loss Stop: -{loss_threshold_pct:.2f}% (${loss_price:.2f})")
            if prediction.get('sigma_pct'):
                print(f"  Barriers scaled to this asset: "
                      f"sigma_{self.predictor.barrier_policy.horizon_days}d = "
                      f"{prediction['sigma_pct']:.2f}% "
                      f"({prediction.get('barrier_mode', 'volatility')} mode)")
        else:
            print(f"  (no barrier information on this prediction)")

        # Kelly formula breakdown
        print(f"\nKELLY FORMULA CALCULATION:")
        print(f"  Formula: f = p/l - q/w  (capped-loss Kelly, net of fees)")
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

            # Get current prices IN USD for all symbols with alive bets.
            #
            # This previously called get_stock_data(symbol, days=1) directly, which
            # returns the raw quote in the asset's native currency -- so a JPY or HKD
            # price was compared against USD-denominated barriers. It also requested
            # a 1-day window from a daily-bar feed, which frequently returned nothing.
            symbols_to_check = sorted({bet.symbol for bet in alive_bets})
            monitor_assets = [
                {
                    'symbol': symbol,
                    'type': self.asset_selector_type_for(symbol),
                    'currency': self.asset_currency_for(symbol),
                }
                for symbol in symbols_to_check
            ]

            current_prices = await self.market_data.get_current_prices_usd(monitor_assets)

            # Reprice open positions so unrealized P&L and reported equity are real.
            await self.portfolio.mark_to_market(current_prices)

            # Delegate the decision to the shared settlement module, so this path
            # and the dashboard/CLI path cannot drift apart again.
            from ..trading.settlement import settle_positions

            settlements = await settle_positions(
                self.portfolio, current_prices, self.max_hold_days)
            bets_closed = len(settlements)

            if self.mode == 'manual':
                for record in settlements:
                    print()
                    print("*** POSITION CLOSED ***")
                    print(f"Symbol: {record['symbol']}  [{record['exit_type']}]")
                    print(f"Reason: {record['reason']}")
                    print(f"Entry: ${record['entry_price']:.4f} -> "
                          f"Exit: ${record['exit_price']:.4f}")

            if bets_closed:
                # Feed the resolved outcomes back into algorithm weights. This is
                # the loop that never ran: update_algorithm_performance() was a stub
                # and weights sat frozen at 0.2 from 2025-09-04 onward.
                try:
                    scored = await self.predictor.resolve_bet_outcomes()
                    if scored:
                        self.logger.info(f"Fed {scored} resolved bet(s) into algorithm weights")
                except Exception as e:
                    self.logger.error(f"Error updating algorithm performance: {e}")

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