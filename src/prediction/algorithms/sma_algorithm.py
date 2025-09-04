"""
Simple Moving Average based prediction algorithm
Predicts based on SMA crossovers and momentum indicators.
"""

import pandas as pd
import numpy as np
from typing import Optional
from .base_algorithm import BasePredictionAlgorithm


class SMAAlgorithm(BasePredictionAlgorithm):
    def __init__(self, config: dict):
        super().__init__("Simple Moving Average", config)
        
        # Algorithm parameters
        self.short_window = config.get('short_window', 5)
        self.long_window = config.get('long_window', 20)
        self.momentum_window = config.get('momentum_window', 10)
        
    async def predict(self, data: pd.DataFrame) -> Optional[float]:
        """
        Predict based on moving average crossovers and momentum
        """
        if len(data) < self.get_required_data_points():
            self.logger.warning(f"Insufficient data: {len(data)} < {self.get_required_data_points()}")
            return None
        
        try:
            # Calculate technical indicators
            df = self._calculate_technical_indicators(data)
            
            # Get latest values
            latest = df.iloc[-1]
            
            # SMA signals
            sma_short = latest['SMA_5']
            sma_long = latest['SMA_20']
            current_price = latest['Close']
            
            # Check for golden cross (short MA > long MA)
            sma_signal = 1 if sma_short > sma_long else 0
            
            # Price momentum (current price vs short MA)
            momentum_signal = 1 if current_price > sma_short else 0
            
            # Price position relative to moving averages
            price_above_short = 1 if current_price > sma_short else 0
            price_above_long = 1 if current_price > sma_long else 0
            
            # Volume signal
            volume_signal = 1 if latest['Volume_Ratio'] > 1.2 else 0
            
            # Recent performance
            recent_change = latest['Price_Change_5']
            momentum_boost = 1 if recent_change > 0.02 else 0  # 2% positive change
            
            # Combine signals
            signals = [
                sma_signal * 25,      # SMA crossover: 25%
                momentum_signal * 20, # Price momentum: 20%
                price_above_short * 15, # Above short MA: 15%
                price_above_long * 15,  # Above long MA: 15%
                volume_signal * 10,     # Volume: 10%
                momentum_boost * 15     # Recent momentum: 15%
            ]
            
            # Base probability
            probability = sum(signals)
            
            # Apply volatility adjustment (higher volatility = lower confidence)
            volatility = latest['Volatility']
            if volatility > 0.05:  # High volatility (>5%)
                probability *= 0.8
            elif volatility < 0.02:  # Low volatility (<2%)
                probability *= 1.1
            
            # Ensure probability is in valid range
            probability = max(0, min(100, probability))
            
            self.logger.debug(f"SMA prediction: {probability:.2f}% "
                            f"(signals: SMA={sma_signal}, momentum={momentum_signal}, "
                            f"volume={volume_signal})")
            
            return probability
            
        except Exception as e:
            self.logger.error(f"Error in SMA prediction: {e}")
            return None
    
    async def train(self, data: pd.DataFrame, target_data: pd.DataFrame = None):
        """
        SMA algorithm doesn't require explicit training,
        but we can use this to optimize parameters
        """
        self.logger.info("SMA algorithm training (parameter optimization)")
        
        # Simple training: just mark as trained
        # In a more sophisticated version, we could optimize the windows
        # based on historical performance
        
        self.is_trained = True
        self.logger.info("SMA algorithm training complete")
    
    def get_required_data_points(self) -> int:
        """Need enough points for the longest moving average"""
        return max(self.long_window, 30)