"""
Support Vector Machine (SVM) based prediction algorithm
Uses scikit-learn SVM for price direction prediction with RBF kernel.
"""

import pandas as pd
import numpy as np
from typing import Optional
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
from .base_algorithm import BasePredictionAlgorithm


class SVMAlgorithm(BasePredictionAlgorithm):
    def __init__(self, config: dict):
        super().__init__("Support Vector Machine", config)

        # Algorithm parameters
        self.kernel = config.get('kernel', 'rbf')  # rbf, linear, poly
        self.C = config.get('C', 1.0)  # Regularization parameter
        self.gamma = config.get('gamma', 'scale')  # Kernel coefficient
        self.target_return = config.get('target_return', 0.03)  # 3% target return

        # Model components
        self.model = None
        self.scaler = StandardScaler()

        # Model persistence
        self.model_dir = Path('models/svm')
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Load existing model if available
        self._load_model()

    async def predict(self, data: pd.DataFrame) -> Optional[float]:
        """
        Predict using trained SVM model
        """
        if not self.is_trained or self.model is None:
            self.logger.warning("SVM model not trained")
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

            # Check for NaN values
            if np.isnan(latest_features).any():
                self.logger.warning("Features contain NaN values, skipping prediction")
                return None

            # Scale features
            latest_features_scaled = self.scaler.transform(latest_features)

            # Get prediction probabilities
            probabilities = self.model.predict_proba(latest_features_scaled)[0]

            # Return probability of positive class (price increase)
            if len(probabilities) >= 2:
                probability = probabilities[1] * 100  # Convert to percentage
            else:
                probability = 50  # Default if only one class

            self.logger.debug(f"SVM prediction: {probability:.2f}%")
            return probability

        except Exception as e:
            self.logger.error(f"Error in SVM prediction: {e}")
            return None

    async def train(self, data: pd.DataFrame, target_data: pd.DataFrame = None):
        """
        Train SVM model on historical data
        """
        self.logger.info("Training SVM model...")

        if len(data) < 50:
            self.logger.warning("Insufficient data for SVM training")
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

            # Scale features - convert DataFrames to numpy arrays
            X_train_scaled = self.scaler.fit_transform(X_train.values)
            X_test_scaled = self.scaler.transform(X_test.values)

            # Train base SVM model (without probability for calibration)
            base_svm = SVC(
                kernel=self.kernel,
                C=self.C,
                gamma=self.gamma,
                probability=False,  # Disable for better calibration
                random_state=42,
                class_weight='balanced'
            )

            # Wrap with CalibratedClassifierCV for better probability estimates
            # Uses sigmoid calibration (Platt scaling) with cross-validation
            self.model = CalibratedClassifierCV(
                base_svm,
                method='sigmoid',  # Platt scaling
                cv=5  # 5-fold cross-validation for calibration
            )

            self.model.fit(X_train_scaled, y_train)

            # Evaluate model
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)

            self.logger.info(f"SVM training complete with probability calibration - "
                           f"Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")

            # Save model
            self._save_model()

            self.is_trained = True

        except Exception as e:
            self.logger.error(f"Error in SVM training: {e}")
            self.is_trained = False

    def _prepare_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prepare feature matrix for SVM model"""
        try:
            # Calculate technical indicators
            df = self._calculate_technical_indicators(data)

            # Select features - same as Random Forest for consistency
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

            # Add price ratios
            df['Price_to_SMA5'] = df['Close'] / df['SMA_5']
            df['Price_to_SMA20'] = df['Close'] / df['SMA_20']
            df['SMA5_to_SMA20'] = df['SMA_5'] / df['SMA_20']

            feature_columns.extend(['Price_to_SMA5', 'Price_to_SMA20', 'SMA5_to_SMA20'])

            # Select and return features
            features = df[feature_columns].copy()

            return features

        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return None

    def _prepare_targets(self, data: pd.DataFrame) -> Optional[pd.Series]:
        """Prepare target variable (future price movement)"""
        try:
            # Calculate forward returns
            future_returns = data['Close'].shift(-5) / data['Close'] - 1  # 5-period forward return

            # Create binary target: 1 if return > target_return, 0 otherwise
            targets = (future_returns > self.target_return).astype(int)

            return targets

        except Exception as e:
            self.logger.error(f"Error preparing targets: {e}")
            return None

    def _save_model(self):
        """Save trained model and scaler"""
        try:
            model_path = self.model_dir / 'svm_model.joblib'
            scaler_path = self.model_dir / 'svm_scaler.joblib'

            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)

            self.logger.info(f"SVM model saved to {model_path}")

        except Exception as e:
            self.logger.error(f"Error saving SVM model: {e}")

    def _load_model(self):
        """Load saved model and scaler"""
        try:
            model_path = self.model_dir / 'svm_model.joblib'
            scaler_path = self.model_dir / 'svm_scaler.joblib'

            if model_path.exists() and scaler_path.exists():
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.is_trained = True
                self.logger.info("SVM model loaded from disk")
                return True
            else:
                return False

        except Exception as e:
            self.logger.error(f"Error loading SVM model: {e}")
            return False

    def get_required_data_points(self) -> int:
        """Need enough points for feature calculation"""
        return 50
