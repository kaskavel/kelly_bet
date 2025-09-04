"""
Base class for prediction algorithms
All prediction algorithms should inherit from this class.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Optional, Any
import logging


class BasePredictionAlgorithm(ABC):
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.is_trained = False
        
    @abstractmethod
    async def predict(self, data: pd.DataFrame) -> Optional[float]:
        """
        Make a prediction based on the provided data
        
        Args:
            data: OHLCV price data as pandas DataFrame
            
        Returns:
            Probability (0-100) that price will increase by target percentage,
            or None if prediction cannot be made
        """
        pass
    
    @abstractmethod
    async def train(self, data: pd.DataFrame, target_data: pd.DataFrame = None):
        """
        Train the algorithm on historical data
        
        Args:
            data: Historical OHLCV data
            target_data: Target outcomes for supervised learning (optional)
        """
        pass
    
    def get_required_data_points(self) -> int:
        """Return minimum number of data points required for prediction"""
        return 30  # Default to 30 periods
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Return information about this algorithm"""
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'is_trained': self.is_trained,
            'required_data_points': self.get_required_data_points(),
            'config': self.config
        }
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate common technical indicators"""
        df = data.copy()
        
        # Simple Moving Averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price change indicators
        df['Price_Change_1'] = df['Close'].pct_change(1)
        df['Price_Change_5'] = df['Close'].pct_change(5)
        df['Price_Change_10'] = df['Close'].pct_change(10)
        
        # Volatility
        df['Volatility'] = df['Price_Change_1'].rolling(window=20).std()
        
        return df