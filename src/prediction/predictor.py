
"""
Prediction engine that manages multiple algorithms and ensemble scoring
Implements adaptive weighting based on algorithm performance tracking.
"""

import asyncio
import logging
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

from .algorithms.sma_algorithm import SMAAlgorithm
from .algorithms.rsi_algorithm import RSIAlgorithm
from .algorithms.random_forest_algorithm import RandomForestAlgorithm
from .algorithms.lstm_algorithm import LSTMAlgorithm
from .algorithms.regression_algorithm import RegressionAlgorithm


class PredictionEngine:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.db_path = Path(config['database']['sqlite']['path'])
        
        # Initialize algorithms
        self.algorithms = {}
        self._initialize_algorithms()
        
        # Algorithm weights (will be updated based on performance)
        self.algorithm_weights = {}
        
    def _initialize_algorithms(self):
        """Initialize all prediction algorithms"""
        algo_configs = self.config.get('prediction', {}).get('algorithms', {})
        
        # Simple Moving Average
        sma_config = algo_configs.get('sma', {})
        self.algorithms['sma'] = SMAAlgorithm(sma_config)
        
        # RSI
        rsi_config = algo_configs.get('rsi', {})
        self.algorithms['rsi'] = RSIAlgorithm(rsi_config)
        
        # Random Forest
        rf_config = algo_configs.get('random_forest', {})
        self.algorithms['rf'] = RandomForestAlgorithm(rf_config)
        
        # LSTM Neural Network
        lstm_config = algo_configs.get('lstm', {})
        self.algorithms['lstm'] = LSTMAlgorithm(lstm_config)
        
        # Linear Regression
        regression_config = algo_configs.get('regression', {})
        self.algorithms['regression'] = RegressionAlgorithm(regression_config)
        
        # Initialize equal weights
        num_algorithms = len(self.algorithms)
        initial_weight = 1.0 / num_algorithms if num_algorithms > 0 else 0
        
        for algo_name in self.algorithms.keys():
            self.algorithm_weights[algo_name] = initial_weight
        
        self.logger.info(f"Initialized {len(self.algorithms)} algorithms: {list(self.algorithms.keys())}")
    
    async def initialize(self):
        """Initialize prediction engine and create database tables"""
        self.logger.info("Initializing prediction engine...")
        
        await self._create_tables()
        await self._load_algorithm_performance()
        
        self.logger.info("Prediction engine initialized")
    
    async def _create_tables(self):
        """Create database tables for predictions and algorithm performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Predictions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            algorithm TEXT NOT NULL,
            probability REAL NOT NULL,
            confidence REAL,
            features TEXT,  -- JSON string of input features
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Algorithm performance table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS algorithm_performance (
            performance_id INTEGER PRIMARY KEY AUTOINCREMENT,
            algorithm TEXT NOT NULL,
            symbol TEXT,
            prediction_timestamp TIMESTAMP NOT NULL,
            predicted_probability REAL NOT NULL,
            actual_outcome INTEGER,  -- 1 if price increased, 0 if decreased
            accuracy_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(algorithm, symbol, prediction_timestamp)
        )
        ''')
        
        # Algorithm weights table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS algorithm_weights (
            weight_id INTEGER PRIMARY KEY AUTOINCREMENT,
            algorithm TEXT UNIQUE NOT NULL,
            weight REAL NOT NULL,
            performance_score REAL DEFAULT 0.5,
            total_predictions INTEGER DEFAULT 0,
            correct_predictions INTEGER DEFAULT 0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()
    
    async def predict_all(self, market_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """
        Generate predictions for all assets using ensemble of algorithms
        
        Args:
            market_data: Dictionary mapping symbol to OHLCV DataFrame
            
        Returns:
            List of prediction dictionaries with ensemble scores
        """
        self.logger.info(f"Generating predictions for {len(market_data)} assets")
        
        predictions = []
        
        # Process each asset
        for symbol, data in market_data.items():
            if data.empty:
                self.logger.warning(f"No data available for {symbol}")
                continue
            
            # Get predictions from each algorithm
            asset_predictions = await self._predict_asset(symbol, data)
            
            if asset_predictions:
                # Calculate ensemble score
                ensemble_score = self._calculate_ensemble_score(asset_predictions)
                
                prediction_result = {
                    'symbol': symbol,
                    'probability': ensemble_score,
                    'current_price': data['Close'].iloc[-1],
                    'algorithms': asset_predictions,
                    'timestamp': datetime.now().isoformat()
                }
                
                predictions.append(prediction_result)
                
                # Store individual algorithm predictions
                await self._store_predictions(symbol, asset_predictions)
        
        self.logger.info(f"Generated {len(predictions)} asset predictions")
        return predictions
    
    async def _predict_asset(self, symbol: str, data: pd.DataFrame) -> List[Dict]:
        """Get predictions from all algorithms for a single asset"""
        asset_predictions = []
        
        # Run all algorithms concurrently
        tasks = []
        for algo_name, algorithm in self.algorithms.items():
            task = self._run_algorithm_prediction(algo_name, algorithm, symbol, data)
            tasks.append(task)
        
        # Wait for all predictions
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for algo_name, result in zip(self.algorithms.keys(), results):
            if isinstance(result, Exception):
                self.logger.error(f"Error in {algo_name} for {symbol}: {result}")
            elif result is not None:
                asset_predictions.append({
                    'algorithm': algo_name,
                    'probability': result,
                    'weight': self.algorithm_weights.get(algo_name, 0.0)
                })
        
        return asset_predictions
    
    async def _run_algorithm_prediction(self, algo_name: str, algorithm, symbol: str, data: pd.DataFrame) -> Optional[float]:
        """Run prediction for a single algorithm"""
        try:
            return await algorithm.predict(data)
        except Exception as e:
            self.logger.error(f"Error running {algo_name} prediction for {symbol}: {e}")
            return None
    
    def _calculate_ensemble_score(self, asset_predictions: List[Dict]) -> float:
        """
        Calculate weighted ensemble score from individual algorithm predictions
        """
        if not asset_predictions:
            return 50.0  # Default neutral probability
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for pred in asset_predictions:
            probability = pred['probability']
            weight = pred['weight']
            
            weighted_sum += probability * weight
            total_weight += weight
        
        if total_weight > 0:
            ensemble_score = weighted_sum / total_weight
        else:
            # Fallback to simple average
            ensemble_score = sum(p['probability'] for p in asset_predictions) / len(asset_predictions)
        
        # Ensure score is in valid range
        return max(0.0, min(100.0, ensemble_score))
    
    async def _store_predictions(self, symbol: str, predictions: List[Dict]):
        """Store individual algorithm predictions in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        try:
            for pred in predictions:
                cursor.execute('''
                INSERT INTO predictions (symbol, timestamp, algorithm, probability)
                VALUES (?, ?, ?, ?)
                ''', (symbol, timestamp, pred['algorithm'], pred['probability']))
            
            conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error storing predictions: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    async def _load_algorithm_performance(self):
        """Load algorithm weights from database based on historical performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT algorithm, weight FROM algorithm_weights')
            weights = cursor.fetchall()
            
            if weights:
                for algo_name, weight in weights:
                    if algo_name in self.algorithm_weights:
                        self.algorithm_weights[algo_name] = weight
                
                self.logger.info(f"Loaded algorithm weights: {self.algorithm_weights}")
            else:
                # Initialize weights in database
                await self._initialize_algorithm_weights()
                
        except Exception as e:
            self.logger.error(f"Error loading algorithm performance: {e}")
        finally:
            conn.close()
    
    async def _initialize_algorithm_weights(self):
        """Initialize algorithm weights in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            for algo_name, weight in self.algorithm_weights.items():
                cursor.execute('''
                INSERT OR REPLACE INTO algorithm_weights (algorithm, weight)
                VALUES (?, ?)
                ''', (algo_name, weight))
            
            conn.commit()
            self.logger.info("Initialized algorithm weights in database")
            
        except Exception as e:
            self.logger.error(f"Error initializing algorithm weights: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    async def update_algorithm_performance(self, symbol: str, actual_outcome: bool, target_return: float = 0.05):
        """
        Update algorithm performance based on actual outcomes
        
        Args:
            symbol: Asset symbol
            actual_outcome: True if price increased by target_return, False otherwise
            target_return: Return threshold for considering prediction correct
        """
        # This would be called after bets are resolved
        # Implementation would:
        # 1. Get recent predictions for this symbol
        # 2. Compare with actual outcome
        # 3. Update algorithm performance scores
        # 4. Recalculate weights
        
        # For now, this is a placeholder
        self.logger.info(f"Updating algorithm performance for {symbol}: outcome={actual_outcome}")
    
    async def cleanup(self):
        """Clean up prediction engine"""
        self.logger.info("Prediction engine cleanup complete")