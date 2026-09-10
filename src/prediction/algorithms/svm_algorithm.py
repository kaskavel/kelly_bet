"""
Support Vector Machine (SVM) based prediction algorithm
Uses scikit-learn SVM for price direction prediction with RBF kernel.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
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
        self.target_return = config.get('target_return', 0.03)  # legacy, unused
        # SVC scales poorly; cap the training set to keep fits tractable on a
        # pooled multi-asset dataset.
        self.max_training_rows = config.get('max_training_rows', 20000)

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

        try:
            features, targets = self._prepare_barrier_dataset(data)
            if features is None:
                self.logger.warning("Insufficient data for SVM training")
                return

            X_train, X_test, y_train, y_test = self._purged_split(features, targets)

            if len(X_train) < 100 or y_train.nunique() < 2:
                self.logger.warning("Training split unusable (too small or single-class)")
                return

            # SVC is O(n^2)-ish; cap the training set so a wide universe stays viable.
            if len(X_train) > self.max_training_rows:
                self.logger.info(f"Subsampling {len(X_train)} -> {self.max_training_rows} "
                                 f"rows for SVM (keeping the most recent)")
                X_train = X_train.iloc[-self.max_training_rows:]
                y_train = y_train.iloc[-self.max_training_rows:]

            # Scale features - convert DataFrames to numpy arrays
            X_train_scaled = self.scaler.fit_transform(X_train.values)
            X_test_scaled = self.scaler.transform(X_test.values)

            # class_weight='balanced' is deliberately absent. Reweighting the classes
            # moves predict_proba away from the true base rate, and calibrating a
            # reweighted model calibrates it to the reweighted problem, not the real
            # one. The old configuration reported a 4.5% mean probability across
            # 76,515 predictions -- unusable as a probability.
            base_svm = SVC(
                kernel=self.kernel,
                C=self.C,
                gamma=self.gamma,
                probability=False,  # Disable for external calibration
                random_state=42,
            )

            # Platt scaling on TIME-ORDERED folds. cv=5 used random folds, which
            # leaked overlapping label windows into every calibration fold.
            n_splits = max(2, min(4, len(X_train) // 250))
            self.model = CalibratedClassifierCV(
                base_svm,
                method='sigmoid',  # Platt scaling
                cv=TimeSeriesSplit(n_splits=n_splits),
            )

            self.model.fit(X_train_scaled, y_train)

            self._log_calibration("SVM", X_train_scaled, y_train, X_test_scaled, y_test)

            # Save model
            self._save_model()

            self.is_trained = True

        except Exception as e:
            self.logger.error(f"Error in SVM training: {e}")
            self.is_trained = False

    def _prepare_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Scale-free feature matrix (shared with the other supervised models)."""
        return self._stationary_feature_frame(data)

    def model_signature(self) -> Dict:
        """What this model predicts, and from what. Invalidates stale saved models."""
        return {
            'schema': 2,
            'label': 'first_touch_barrier',
            'features': tuple(self.STATIONARY_FEATURES),
            'win_threshold': float(self.config.get('win_threshold', 5.0)),
            'loss_threshold': float(self.config.get('loss_threshold', 3.0)),
            'max_hold_days': int(self.config.get('max_hold_days', 15)),
        }

    def _save_model(self):
        """Save trained model, scaler and schema signature"""
        try:
            model_path = self.model_dir / 'svm_model.joblib'
            scaler_path = self.model_dir / 'svm_scaler.joblib'

            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            joblib.dump(self.model_signature(), self.model_dir / 'svm_signature.joblib')

            self.logger.info(f"SVM model saved to {model_path}")

        except Exception as e:
            self.logger.error(f"Error saving SVM model: {e}")

    def _load_model(self):
        """Load saved model and scaler, only if trained on the current schema"""
        try:
            model_path = self.model_dir / 'svm_model.joblib'
            scaler_path = self.model_dir / 'svm_scaler.joblib'
            signature_path = self.model_dir / 'svm_signature.joblib'

            if not (model_path.exists() and scaler_path.exists()):
                return False

            if not signature_path.exists():
                self.logger.warning("Saved SVM predates the current feature schema "
                                    "- ignoring it; retrain needed")
                return False

            if joblib.load(signature_path) != self.model_signature():
                self.logger.warning("Saved SVM schema mismatch - ignoring it; retrain needed")
                return False

            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.is_trained = True
            self.logger.info("SVM model loaded from disk")
            return True

        except Exception as e:
            self.logger.error(f"Error loading SVM model: {e}")
            return False

    def get_required_data_points(self) -> int:
        """Need enough points for feature calculation"""
        return 50
