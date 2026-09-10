"""
Random Forest based prediction algorithm
Uses scikit-learn Random Forest for price direction prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
import joblib
from pathlib import Path
from .base_algorithm import BasePredictionAlgorithm


class RandomForestAlgorithm(BasePredictionAlgorithm):
    def __init__(self, config: dict):
        super().__init__("Random Forest", config)
        
        # Algorithm parameters
        self.n_estimators = config.get('n_estimators', 100)
        self.max_depth = config.get('max_depth', 10)
        self.min_samples_split = config.get('min_samples_split', 10)
        self.target_return = config.get('target_return', 0.03)  # 3% target return
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        
        # Model persistence
        self.model_dir = Path('models/random_forest')
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Load existing model if available (and schema-compatible)
        self._load_model()

    async def predict(self, data: pd.DataFrame) -> Optional[float]:
        """
        Predict using trained Random Forest model
        """
        if not self.is_trained or self.model is None:
            self.logger.warning("Random Forest model not trained")
            return None
            
        if len(data) < self.get_required_data_points():
            self.logger.warning(f"Insufficient data: {len(data)} < {self.get_required_data_points()}")
            return None
        
        try:
            # Prepare features
            features = self._prepare_features(data)
            if features is None:
                return None
            
            # Get latest feature vector (already as numpy array)
            latest_features = features.iloc[-1:].values

            # Scale features
            latest_features_scaled = self.scaler.transform(latest_features)
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(latest_features_scaled)[0]
            
            # Return probability of positive class (price increase)
            if len(probabilities) >= 2:
                probability = probabilities[1] * 100  # Convert to percentage
            else:
                probability = 50  # Default if only one class
            
            self.logger.debug(f"Random Forest prediction: {probability:.2f}%")
            return probability
            
        except Exception as e:
            self.logger.error(f"Error in Random Forest prediction: {e}")
            return None
    
    async def train(self, data: pd.DataFrame, target_data: pd.DataFrame = None):
        """
        Train Random Forest on the first-touch barrier event.

        Three things changed from the previous version:
          1. The label is which barrier is touched FIRST, not a 5-bar terminal
             return crossing +3%. The old label ignored the stop entirely.
          2. The split is time-ordered with a purge embargo. The old
             train_test_split(random_state=42) shuffled, leaking overlapping label
             windows between train and test.
          3. class_weight='balanced' is gone and the forest is wrapped in
             CalibratedClassifierCV. Reweighting the classes shifts predict_proba
             away from the true base rate, and that output was being fed straight
             into Kelly as though it were a calibrated probability.
        """
        self.logger.info("Training Random Forest model...")

        try:
            features, targets = self._prepare_barrier_dataset(data)
            if features is None:
                self.logger.warning("Insufficient data for Random Forest training")
                return

            X_train, X_test, y_train, y_test = self._purged_split(features, targets)

            if len(X_train) < 100 or y_train.nunique() < 2:
                self.logger.warning("Training split unusable (too small or single-class)")
                return

            # Scale features - convert DataFrames to numpy arrays to avoid feature name warnings
            X_train_scaled = self.scaler.fit_transform(X_train.values)
            X_test_scaled = self.scaler.transform(X_test.values)

            base_forest = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=42,
                n_jobs=-1,
            )

            # Calibrate on time-ordered inner folds so predict_proba means what it
            # says. TimeSeriesSplit, not KFold: random folds would reintroduce the
            # leakage the outer split just removed.
            n_splits = max(2, min(4, len(X_train) // 250))
            self.model = CalibratedClassifierCV(
                base_forest,
                method='isotonic',
                cv=TimeSeriesSplit(n_splits=n_splits),
            )
            self.model.fit(X_train_scaled, y_train)

            self._log_calibration("Random Forest", X_train_scaled, y_train, X_test_scaled, y_test)

            # Save model
            self._save_model()

            self.is_trained = True

        except Exception as e:
            self.logger.error(f"Error in Random Forest training: {e}")
            self.is_trained = False

    def _prepare_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Scale-free feature matrix.

        The previous feature set led with raw price levels (SMA_5, EMA_12, MACD in
        dollars). Those cannot be pooled across assets: a scaler fitted on US
        mega-caps produced meaningless inputs for JPY, HKD and crypto quotes.
        """
        return self._stationary_feature_frame(data)

    def _save_model(self):
        """Save trained model, scaler, and the schema they were trained against"""
        try:
            model_path = self.model_dir / 'rf_model.joblib'
            scaler_path = self.model_dir / 'rf_scaler.joblib'

            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            joblib.dump(self.model_signature(), self.model_dir / 'rf_signature.joblib')

            self.logger.info(f"Model saved to {model_path}")

        except Exception as e:
            self.logger.error(f"Error saving model: {e}")

    def _load_model(self):
        """
        Load saved model and scaler, but only if trained on the current schema.

        Models on disk predate the switch to stationary features and barrier labels.
        Loading one and feeding it the new feature matrix would either raise or, worse,
        silently score the wrong columns.
        """
        try:
            model_path = self.model_dir / 'rf_model.joblib'
            scaler_path = self.model_dir / 'rf_scaler.joblib'
            signature_path = self.model_dir / 'rf_signature.joblib'

            if not (model_path.exists() and scaler_path.exists()):
                return False

            if not signature_path.exists():
                self.logger.warning("Saved Random Forest predates the current feature "
                                    "schema (no signature) - ignoring it; retrain needed")
                return False

            saved_signature = joblib.load(signature_path)
            if saved_signature != self.model_signature():
                self.logger.warning(
                    f"Saved Random Forest schema mismatch - ignoring it; retrain needed. "
                    f"saved={saved_signature}, current={self.model_signature()}")
                return False

            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.is_trained = True
            self.logger.info("Model loaded from disk")
            return True

        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False

    def model_signature(self) -> Dict:
        """
        Identifies what this model was trained to predict, and from what.

        Any change here invalidates models on disk, which is the intent.
        """
        return {
            'schema': 2,
            'label': 'first_touch_barrier',
            'features': tuple(self.STATIONARY_FEATURES),
            'win_threshold': float(self.config.get('win_threshold', 5.0)),
            'loss_threshold': float(self.config.get('loss_threshold', 3.0)),
            'max_hold_days': int(self.config.get('max_hold_days', 15)),
        }

    def get_required_data_points(self) -> int:
        """Need enough points for feature calculation"""
        return 50