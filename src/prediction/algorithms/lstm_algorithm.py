"""
LSTM (Long Short-Term Memory) based prediction algorithm
Uses TensorFlow/Keras for time series prediction of price movements.
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging
from pathlib import Path
import joblib

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    keras = None

from .base_algorithm import BasePredictionAlgorithm


class LSTMAlgorithm(BasePredictionAlgorithm):
    def __init__(self, config: dict):
        super().__init__("LSTM Neural Network", config)
        
        if not TENSORFLOW_AVAILABLE:
            self.logger.error("TensorFlow not available. Please install: pip install tensorflow>=2.13.0")
            return
        
        # Algorithm parameters
        self.sequence_length = config.get('sequence_length', 60)  # 60 time steps
        self.lstm_units = config.get('lstm_units', 50)
        self.dropout_rate = config.get('dropout_rate', 0.2)
        self.epochs = config.get('epochs', 50)
        self.batch_size = config.get('batch_size', 32)
        self.target_return = config.get('target_return', 0.03)  # 3% target return
        
        # Model components
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Model persistence
        self.model_dir = Path('models/lstm')
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing model if available
        self._load_model()
    
    async def predict(self, data: pd.DataFrame) -> Optional[float]:
        """
        Predict using trained LSTM model
        """
        if not TENSORFLOW_AVAILABLE:
            self.logger.warning("TensorFlow not available")
            return None
            
        if not self.is_trained or self.model is None:
            self.logger.warning("LSTM model not trained")
            return None
            
        if len(data) < self.get_required_data_points():
            self.logger.warning(f"Insufficient data: {len(data)} < {self.get_required_data_points()}")
            return None
        
        try:
            # Prepare sequence data
            sequence_data = self._prepare_sequence_data(data)
            if sequence_data is None or len(sequence_data) == 0:
                return None

            # Get the most recent sequence (already shaped as [num_sequences, sequence_length, num_features])
            # We just need the last sequence: [1, sequence_length, num_features]
            latest_sequence = sequence_data[-1:]

            # Make prediction
            prediction = self.model.predict(latest_sequence, verbose=0)[0][0]
            
            # Convert regression output to probability
            # LSTM predicts future return, convert to probability of positive return
            probability = self._return_to_probability(prediction)
            
            self.logger.debug(f"LSTM prediction: {probability:.2f}% (raw output: {prediction:.4f})")
            return probability
            
        except Exception as e:
            self.logger.error(f"Error in LSTM prediction: {e}")
            return None
    
    async def train(self, data: pd.DataFrame, target_data: pd.DataFrame = None):
        """
        Train LSTM model on historical data
        """
        if not TENSORFLOW_AVAILABLE:
            self.logger.error("TensorFlow not available for training")
            return
            
        self.logger.info("Training LSTM model...")
        
        if len(data) < 50:
            self.logger.warning("Insufficient data for LSTM training")
            return
        
        try:
            # Prepare training data
            X_train, y_train = self._prepare_training_data(data)
            if X_train is None or y_train is None:
                return
            
            # Build model
            self._build_model(X_train.shape[1], X_train.shape[2])
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.2,
                verbose=1,
                shuffle=False  # Keep time series order
            )
            
            # Evaluate training
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            
            self.logger.info(f"LSTM training complete - "
                           f"Final loss: {final_loss:.4f}, Val loss: {final_val_loss:.4f}")
            
            # Save model
            self._save_model()
            
            self.is_trained = True
            
        except Exception as e:
            self.logger.error(f"Error in LSTM training: {e}")
            self.is_trained = False
    
    def _build_model(self, sequence_length: int, n_features: int):
        """Build LSTM model architecture"""
        self.model = Sequential([
            # First LSTM layer
            LSTM(
                units=self.lstm_units,
                return_sequences=True,
                input_shape=(sequence_length, n_features)
            ),
            Dropout(self.dropout_rate),
            
            # Second LSTM layer
            LSTM(
                units=self.lstm_units // 2,
                return_sequences=False
            ),
            Dropout(self.dropout_rate),
            
            # Dense layers
            Dense(25, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(1, activation='linear')  # Regression output
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.logger.info(f"LSTM model built: {self.model.count_params()} parameters")
    
    def _prepare_training_data(self, data: pd.DataFrame):
        """Prepare training data with sequences and targets"""
        try:
            # Calculate technical indicators
            df = self._calculate_technical_indicators(data)
            
            # Select features for LSTM
            feature_columns = [
                'Close', 'Volume',
                'SMA_5', 'SMA_10', 'SMA_20',
                'EMA_12', 'EMA_26',
                'MACD', 'RSI',
                'BB_Position',
                'Volume_Ratio',
                'Price_Change_1', 'Volatility'
            ]
            
            # Prepare feature matrix
            features = df[feature_columns].dropna()
            if len(features) < self.sequence_length + 10:
                self.logger.warning("Not enough clean data for LSTM training")
                return None, None

            # Scale features - use .values to convert DataFrame to numpy array
            # This ensures consistent behavior during training and prediction
            features_scaled = self.scaler.fit_transform(features.values)
            
            # Create sequences and targets
            X, y = [], []
            
            for i in range(self.sequence_length, len(features_scaled) - 5):  # -5 for forward prediction
                # Input sequence
                X.append(features_scaled[i-self.sequence_length:i])
                
                # Target: future return (5 periods ahead)
                current_price = features.iloc[i]['Close']
                future_price = features.iloc[i+5]['Close']
                future_return = (future_price - current_price) / current_price
                y.append(future_return)
            
            X = np.array(X)
            y = np.array(y)
            
            self.logger.info(f"Prepared LSTM training data: X shape {X.shape}, y shape {y.shape}")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing LSTM training data: {e}")
            return None, None
    
    def _prepare_sequence_data(self, data: pd.DataFrame):
        """Prepare sequence data for prediction"""
        try:
            # Calculate technical indicators
            df = self._calculate_technical_indicators(data)
            
            # Select same features as training
            feature_columns = [
                'Close', 'Volume',
                'SMA_5', 'SMA_10', 'SMA_20',
                'EMA_12', 'EMA_26',
                'MACD', 'RSI',
                'BB_Position',
                'Volume_Ratio',
                'Price_Change_1', 'Volatility'
            ]
            
            # Prepare feature matrix
            features = df[feature_columns].dropna()

            if len(features) < self.sequence_length:
                return None

            # Scale features using fitted scaler
            # Use .values to convert DataFrame to numpy array (consistent with training)
            features_scaled = self.scaler.transform(features.values)
            
            # Create sequences
            sequences = []
            for i in range(self.sequence_length, len(features_scaled) + 1):
                sequences.append(features_scaled[i-self.sequence_length:i])
            
            return np.array(sequences)
            
        except Exception as e:
            self.logger.error(f"Error preparing sequence data: {e}")
            return None
    
    def _return_to_probability(self, predicted_return: float) -> float:
        """Convert predicted return to probability of positive movement"""
        # Use sigmoid-like transformation
        # Adjust the scaling factor based on your target return
        scaling_factor = 10.0  # Adjust this based on typical return magnitudes
        
        # Convert to probability using sigmoid
        probability = 1 / (1 + np.exp(-predicted_return * scaling_factor))
        
        # Convert to percentage and ensure reasonable range
        probability_pct = probability * 100
        
        # Apply some bounds to keep it reasonable
        probability_pct = max(10, min(90, probability_pct))
        
        return probability_pct
    
    def _save_model(self):
        """Save trained model and scaler"""
        try:
            model_path = self.model_dir / 'lstm_model.h5'
            scaler_path = self.model_dir / 'lstm_scaler.joblib'
            
            self.model.save(model_path)
            joblib.dump(self.scaler, scaler_path)
            
            self.logger.info(f"LSTM model saved to {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving LSTM model: {e}")
    
    def _load_model(self):
        """Load saved model and scaler"""
        if not TENSORFLOW_AVAILABLE:
            return False
            
        try:
            model_path = self.model_dir / 'lstm_model.h5'
            scaler_path = self.model_dir / 'lstm_scaler.joblib'
            
            if model_path.exists() and scaler_path.exists():
                self.model = keras.models.load_model(model_path)
                self.scaler = joblib.load(scaler_path)
                self.is_trained = True
                self.logger.info("LSTM model loaded from disk")
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading LSTM model: {e}")
            return False
    
    def get_required_data_points(self) -> int:
        """Need enough points for sequence + feature calculation"""
        return self.sequence_length + 30