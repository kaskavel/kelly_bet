"""
RSI (Relative Strength Index) based prediction algorithm
Predicts based on RSI overbought/oversold levels and momentum.
"""

import pandas as pd
import numpy as np
from typing import Optional
from .base_algorithm import BasePredictionAlgorithm


class RSIAlgorithm(BasePredictionAlgorithm):
    def __init__(self, config: dict):
        super().__init__("Relative Strength Index", config)
        
        # Algorithm parameters
        self.rsi_period = config.get('rsi_period', 14)
        self.oversold_level = config.get('oversold_level', 30)
        self.overbought_level = config.get('overbought_level', 70)
        
    async def predict(self, data: pd.DataFrame) -> Optional[float]:
        """
        Predict based on RSI levels and momentum
        """
        if len(data) < self.get_required_data_points():
            self.logger.warning(f"Insufficient data: {len(data)} < {self.get_required_data_points()}")
            return None
        
        try:
            # Calculate technical indicators
            df = self._calculate_technical_indicators(data)
            
            # Get latest values
            latest = df.iloc[-1]
            previous = df.iloc[-2] if len(df) > 1 else latest
            
            current_rsi = latest['RSI']
            previous_rsi = previous['RSI']
            
            if pd.isna(current_rsi):
                self.logger.warning("RSI calculation failed (NaN)")
                return None
            
            # Base probability from RSI level
            if current_rsi <= self.oversold_level:
                # Oversold - likely to bounce up
                base_probability = 70
            elif current_rsi >= self.overbought_level:
                # Overbought - likely to go down
                base_probability = 20
            elif current_rsi < 50:
                # Below neutral - slight bearish
                base_probability = 40
            else:
                # Above neutral - slight bullish
                base_probability = 60
            
            # RSI momentum (improving or deteriorating)
            rsi_momentum = current_rsi - previous_rsi
            if rsi_momentum > 2:  # RSI improving
                momentum_boost = 15
            elif rsi_momentum < -2:  # RSI deteriorating
                momentum_boost = -15
            else:
                momentum_boost = 0
            
            # RSI divergence check (simplified)
            price_change = latest['Price_Change_1']
            if current_rsi > 50 and price_change > 0:
                # Both price and RSI moving up - bullish
                divergence_signal = 10
            elif current_rsi < 50 and price_change < 0:
                # Both price and RSI moving down - bearish continuation
                divergence_signal = -5
            else:
                divergence_signal = 0
            
            # Volume confirmation
            volume_signal = 0
            if latest['Volume_Ratio'] > 1.3:  # High volume
                if current_rsi <= self.oversold_level:
                    volume_signal = 10  # High volume on oversold = bullish
                elif current_rsi >= self.overbought_level:
                    volume_signal = -10  # High volume on overbought = bearish
            
            # Combine all signals
            probability = base_probability + momentum_boost + divergence_signal + volume_signal
            
            # Apply volatility adjustment
            volatility = latest['Volatility']
            if volatility > 0.05:  # High volatility
                # Reduce confidence in extreme RSI readings
                if current_rsi <= self.oversold_level or current_rsi >= self.overbought_level:
                    probability *= 0.9
            
            # Ensure probability is in valid range
            probability = max(0, min(100, probability))
            
            self.logger.debug(f"RSI prediction: {probability:.2f}% "
                            f"(RSI={current_rsi:.1f}, momentum={rsi_momentum:.1f})")
            
            return probability
            
        except Exception as e:
            self.logger.error(f"Error in RSI prediction: {e}")
            return None
    
    async def train(self, data: pd.DataFrame, target_data: pd.DataFrame = None):
        """
        RSI algorithm training - optimize thresholds based on historical data
        """
        self.logger.info("RSI algorithm training")
        
        try:
            # Calculate technical indicators
            df = self._calculate_technical_indicators(data)
            
            # Simple optimization: find best RSI levels based on historical performance
            # This is a simplified version - in practice, you'd do more sophisticated optimization
            
            # For now, just use default values and mark as trained
            self.is_trained = True
            self.logger.info("RSI algorithm training complete")
            
        except Exception as e:
            self.logger.error(f"Error in RSI training: {e}")
            self.is_trained = False
    
    def get_required_data_points(self) -> int:
        """Need enough points for RSI calculation"""
        return max(self.rsi_period + 10, 30)