"""
Linear Regression based prediction algorithm
Uses multiple regression models with technical indicators to predict price movements.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from pathlib import Path
from .base_algorithm import BasePredictionAlgorithm


class RegressionAlgorithm(BasePredictionAlgorithm):
    def __init__(self, config: dict):
        super().__init__("Linear Regression", config)
        
        # Algorithm parameters
        self.model_type = config.get('model_type', 'ridge')  # linear, ridge, lasso
        self.alpha = config.get('alpha', 1.0)  # Regularization strength
        self.target_return = config.get('target_return', 0.03)  # 3% target return
        self.prediction_horizon = config.get('prediction_horizon', 5)  # 5 periods ahead
        
        # Model components
        self.models = {}  # Will store multiple models
        self.scaler = StandardScaler()
        self.feature_importance = {}
        
        # Model persistence
        self.model_dir = Path('models/regression')
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing model if available
        self._load_model()
    
    async def predict(self, data: pd.DataFrame) -> Optional[float]:
        """
        Predict using trained regression models
        """
        if not self.is_trained or not self.models:
            self.logger.warning("Regression model not trained")
            return None
            
        if len(data) < self.get_required_data_points():
            self.logger.warning(f"Insufficient data: {len(data)} < {self.get_required_data_points()}")
            return None
        
        try:
            # Prepare features
            features = self._prepare_features(data)
            if features is None:
                return None
            
            # Get latest feature vector
            latest_features = features.iloc[-1:].values
            
            # Scale features
            latest_features_scaled = self.scaler.transform(latest_features)
            
            # Get predictions from all models
            predictions = {}
            for model_name, model in self.models.items():
                pred = model.predict(latest_features_scaled)[0]
                predictions[model_name] = pred
            
            # Ensemble prediction (average of all models)
            ensemble_pred = np.mean(list(predictions.values()))
            
            # Convert regression output to probability
            probability = self._return_to_probability(ensemble_pred)
            
            self.logger.debug(f"Regression prediction: {probability:.2f}% "
                            f"(ensemble return: {ensemble_pred:.4f})")
            return probability
            
        except Exception as e:
            self.logger.error(f"Error in regression prediction: {e}")
            return None
    
    async def train(self, data: pd.DataFrame, target_data: pd.DataFrame = None):
        """
        Train regression models on historical data
        """
        self.logger.info("Training regression models...")
        
        if len(data) < 50:
            self.logger.warning("Insufficient data for regression training")
            return
        
        try:
            # Prepare features and targets
            features = self._prepare_features(data)
            if features is None:
                return
            
            targets = self._prepare_targets(data)
            if targets is None:
                return
            
            # Remove rows with NaN values
            valid_mask = ~(features.isna().any(axis=1) | targets.isna())
            features_clean = features[valid_mask]
            targets_clean = targets[valid_mask]
            
            if len(features_clean) < 50:
                self.logger.warning("Insufficient clean data for training")
                return
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features_clean, targets_clean, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train multiple models
            models_to_train = {
                'linear': LinearRegression(),
                'ridge': Ridge(alpha=self.alpha, random_state=42),
                'lasso': Lasso(alpha=self.alpha, random_state=42, max_iter=2000)
            }
            
            best_score = -np.inf
            best_model_name = None
            
            for model_name, model in models_to_train.items():
                try:
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Evaluate
                    train_pred = model.predict(X_train_scaled)
                    test_pred = model.predict(X_test_scaled)
                    
                    train_r2 = r2_score(y_train, train_pred)
                    test_r2 = r2_score(y_test, test_pred)
                    train_mse = mean_squared_error(y_train, train_pred)
                    test_mse = mean_squared_error(y_test, test_pred)
                    
                    self.logger.info(f"{model_name.upper()} - Train R²: {train_r2:.3f}, "
                                   f"Test R²: {test_r2:.3f}, Test MSE: {test_mse:.6f}")
                    
                    # Store model
                    self.models[model_name] = model
                    
                    # Track best model
                    if test_r2 > best_score:
                        best_score = test_r2
                        best_model_name = model_name
                        
                    # Store feature importance for linear models
                    if hasattr(model, 'coef_'):
                        self.feature_importance[model_name] = {
                            'coefficients': model.coef_,
                            'feature_names': features_clean.columns.tolist()
                        }
                
                except Exception as e:
                    self.logger.error(f"Error training {model_name}: {e}")
            
            if best_model_name:
                self.logger.info(f"Best performing model: {best_model_name} (R² = {best_score:.3f})")
                
                # Log feature importance for best model
                if best_model_name in self.feature_importance:
                    self._log_feature_importance(best_model_name)
            
            # Save models
            self._save_model()
            
            self.is_trained = True
            
        except Exception as e:
            self.logger.error(f"Error in regression training: {e}")
            self.is_trained = False
    
    def _prepare_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prepare feature matrix for regression model"""
        try:
            # Calculate technical indicators
            df = self._calculate_technical_indicators(data)
            
            # Select features
            feature_columns = [
                'SMA_5', 'SMA_10', 'SMA_20',
                'EMA_12', 'EMA_26',
                'MACD', 'MACD_Signal', 'MACD_Hist',
                'RSI',
                'BB_Position',
                'Volume_Ratio',
                'Price_Change_1', 'Price_Change_5', 'Price_Change_10',
                'Volatility'
            ]
            
            # Add derived features
            df['Price_to_SMA5'] = df['Close'] / df['SMA_5']
            df['Price_to_SMA20'] = df['Close'] / df['SMA_20']
            df['SMA5_to_SMA20'] = df['SMA_5'] / df['SMA_20']
            df['RSI_Change'] = df['RSI'].diff()
            df['Volume_Change'] = df['Volume'].pct_change()
            df['High_Low_Ratio'] = df['High'] / df['Low']
            
            # Add lagged features
            for lag in [1, 2, 3]:
                df[f'Price_Change_lag_{lag}'] = df['Price_Change_1'].shift(lag)
                df[f'RSI_lag_{lag}'] = df['RSI'].shift(lag)
                df[f'Volume_Ratio_lag_{lag}'] = df['Volume_Ratio'].shift(lag)
            
            extended_features = feature_columns + [
                'Price_to_SMA5', 'Price_to_SMA20', 'SMA5_to_SMA20',
                'RSI_Change', 'Volume_Change', 'High_Low_Ratio',
                'Price_Change_lag_1', 'Price_Change_lag_2', 'Price_Change_lag_3',
                'RSI_lag_1', 'RSI_lag_2', 'RSI_lag_3',
                'Volume_Ratio_lag_1', 'Volume_Ratio_lag_2', 'Volume_Ratio_lag_3'
            ]
            
            # Select and return features
            features = df[extended_features].copy()
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return None
    
    def _prepare_targets(self, data: pd.DataFrame) -> Optional[pd.Series]:
        """Prepare target variable (future returns)"""
        try:
            # Calculate forward returns
            future_returns = data['Close'].shift(-self.prediction_horizon) / data['Close'] - 1
            
            return future_returns
            
        except Exception as e:
            self.logger.error(f"Error preparing targets: {e}")
            return None
    
    def _return_to_probability(self, predicted_return: float) -> float:
        """Convert predicted return to probability of positive movement"""
        # More sophisticated probability conversion than simple sigmoid
        
        # Calibration based on historical return distributions
        # Assume normal distribution of returns with some typical volatility
        typical_volatility = 0.02  # 2% daily volatility assumption
        
        # Calculate z-score
        z_score = predicted_return / typical_volatility
        
        # Convert to probability using cumulative normal distribution
        from scipy.stats import norm
        probability = norm.cdf(z_score)
        
        # Convert to percentage
        probability_pct = probability * 100
        
        # Apply bounds to keep it reasonable (10% to 90%)
        probability_pct = max(10, min(90, probability_pct))
        
        return probability_pct
    
    def _log_feature_importance(self, model_name: str):
        """Log feature importance for interpretability"""
        if model_name not in self.feature_importance:
            return
        
        importance_data = self.feature_importance[model_name]
        coeffs = importance_data['coefficients']
        features = importance_data['feature_names']
        
        # Sort by absolute coefficient value
        importance_pairs = list(zip(features, coeffs))
        importance_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        
        self.logger.info(f"Top 10 features for {model_name}:")
        for i, (feature, coeff) in enumerate(importance_pairs[:10]):
            self.logger.info(f"  {i+1:2d}. {feature:20s}: {coeff:8.4f}")
    
    def _save_model(self):
        """Save trained models and scaler"""
        try:
            models_path = self.model_dir / 'regression_models.joblib'
            scaler_path = self.model_dir / 'regression_scaler.joblib'
            importance_path = self.model_dir / 'feature_importance.joblib'
            
            joblib.dump(self.models, models_path)
            joblib.dump(self.scaler, scaler_path)
            joblib.dump(self.feature_importance, importance_path)
            
            self.logger.info(f"Regression models saved to {models_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving regression models: {e}")
    
    def _load_model(self):
        """Load saved models and scaler"""
        try:
            models_path = self.model_dir / 'regression_models.joblib'
            scaler_path = self.model_dir / 'regression_scaler.joblib'
            importance_path = self.model_dir / 'feature_importance.joblib'
            
            if models_path.exists() and scaler_path.exists():
                self.models = joblib.load(models_path)
                self.scaler = joblib.load(scaler_path)
                
                if importance_path.exists():
                    self.feature_importance = joblib.load(importance_path)
                
                self.is_trained = True
                self.logger.info("Regression models loaded from disk")
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading regression models: {e}")
            return False
    
    def get_required_data_points(self) -> int:
        """Need enough points for feature calculation including lags"""
        return 50