"""
Regularised linear prediction algorithm.

Predicts P(the +win barrier is touched before the -loss barrier) with a penalised
logistic regression over scale-free technical features.

Previously this fitted Ridge/Lasso/OLS to a 5-bar forward RETURN and then mapped the
point forecast to a "probability" with norm.cdf(r / 0.02). Two problems with that:
the quantity being estimated was P(return > 0) rather than P(win barrier first), and
a linear model's forward-return forecast is so close to the sample mean that the CDF
squash produced a near-constant output -- the stored predictions averaged 78.1% with
84.7% of all calls above 60%, on an event whose base rate is around 37%.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
import joblib
from pathlib import Path
from .base_algorithm import BasePredictionAlgorithm


class RegressionAlgorithm(BasePredictionAlgorithm):
    def __init__(self, config: dict):
        super().__init__("Logistic Regression", config)

        # Algorithm parameters. `alpha` is kept as the config name for continuity;
        # for logistic regression the equivalent knob is C = 1/alpha.
        self.alpha = config.get('alpha', 1.0)
        self.penalty = config.get('penalty', 'l2')  # l1 or l2
        self.C = 1.0 / self.alpha if self.alpha > 0 else 1.0

        # Model components
        self.model = None
        self.scaler = StandardScaler()

        # Model persistence
        self.model_dir = Path('models/regression')
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Load existing model if available (and schema-compatible)
        self._load_model()

    async def predict(self, data: pd.DataFrame) -> Optional[float]:
        """Probability (0-100) that the win barrier is touched first."""
        if not self.is_trained or self.model is None:
            self.logger.warning("Regression model not trained")
            return None

        if len(data) < self.get_required_data_points():
            self.logger.warning(f"Insufficient data: {len(data)} < {self.get_required_data_points()}")
            return None

        try:
            features = self._prepare_features(data)
            if features is None:
                return None

            latest_features = features.iloc[-1:].values
            if not np.isfinite(latest_features).all():
                self.logger.warning("Features contain NaN/inf values, skipping prediction")
                return None

            latest_features_scaled = self.scaler.transform(latest_features)
            if not np.isfinite(latest_features_scaled).all():
                self.logger.warning("Scaled features contain NaN/inf values, skipping prediction")
                return None

            probability = float(self.model.predict_proba(latest_features_scaled)[0][1]) * 100.0

            self.logger.debug(f"Regression prediction: {probability:.2f}%")
            return probability

        except Exception as e:
            self.logger.error(f"Error in regression prediction: {e}")
            return None

    async def train(self, data: pd.DataFrame, target_data: pd.DataFrame = None):
        """Fit the logistic model on first-touch barrier labels."""
        self.logger.info("Training regression model...")

        try:
            features, targets = self._prepare_barrier_dataset(data)
            if features is None:
                self.logger.warning("Insufficient data for regression training")
                return

            X_train, X_test, y_train, y_test = self._purged_split(features, targets)

            if len(X_train) < 100 or y_train.nunique() < 2:
                self.logger.warning("Training split unusable (too small or single-class)")
                return

            X_train_scaled = self.scaler.fit_transform(X_train.values)
            X_test_scaled = self.scaler.transform(X_test.values)

            base_model = LogisticRegression(
                C=self.C,
                penalty=self.penalty,
                solver='liblinear' if self.penalty == 'l1' else 'lbfgs',
                max_iter=2000,
                random_state=42,
            )

            n_splits = max(2, min(4, len(X_train) // 250))
            self.model = CalibratedClassifierCV(
                base_model,
                method='sigmoid',
                cv=TimeSeriesSplit(n_splits=n_splits),
            )
            self.model.fit(X_train_scaled, y_train)

            self._log_calibration("Regression", X_train_scaled, y_train,
                                  X_test_scaled, y_test)
            self._log_feature_influence(X_train_scaled, y_train)

            self._save_model()
            self.is_trained = True

        except Exception as e:
            self.logger.error(f"Error in regression training: {e}")
            self.is_trained = False

    def _prepare_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Scale-free feature matrix (shared with the other supervised models)."""
        return self._stationary_feature_frame(data)

    def _log_feature_influence(self, X_train, y_train):
        """Log standardised coefficients from an uncalibrated refit, for readability."""
        try:
            probe = LogisticRegression(
                C=self.C,
                penalty=self.penalty,
                solver='liblinear' if self.penalty == 'l1' else 'lbfgs',
                max_iter=2000,
                random_state=42,
            )
            probe.fit(X_train, y_train)
            pairs = sorted(
                zip(self.STATIONARY_FEATURES, probe.coef_[0]),
                key=lambda kv: abs(kv[1]),
                reverse=True,
            )
            self.logger.info("Top features by standardised coefficient:")
            for i, (feature, coefficient) in enumerate(pairs[:10], 1):
                self.logger.info(f"  {i:2d}. {feature:20s}: {coefficient:+8.4f}")
        except Exception as e:
            self.logger.debug(f"Could not log feature influence: {e}")

    def model_signature(self) -> Dict:
        """What this model predicts, and from what. Invalidates stale saved models."""
        return {
            'schema': 2,
            'label': 'first_touch_barrier',
            'model': 'logistic',
            'features': tuple(self.STATIONARY_FEATURES),
            'win_threshold': float(self.config.get('win_threshold', 5.0)),
            'loss_threshold': float(self.config.get('loss_threshold', 3.0)),
            'max_hold_days': int(self.config.get('max_hold_days', 15)),
        }

    def _save_model(self):
        """Save trained model, scaler and schema signature"""
        try:
            joblib.dump(self.model, self.model_dir / 'regression_model.joblib')
            joblib.dump(self.scaler, self.model_dir / 'regression_scaler.joblib')
            joblib.dump(self.model_signature(), self.model_dir / 'regression_signature.joblib')
            self.logger.info(f"Regression model saved to {self.model_dir}")
        except Exception as e:
            self.logger.error(f"Error saving regression model: {e}")

    def _load_model(self):
        """Load saved model and scaler, only if trained on the current schema"""
        try:
            model_path = self.model_dir / 'regression_model.joblib'
            scaler_path = self.model_dir / 'regression_scaler.joblib'
            signature_path = self.model_dir / 'regression_signature.joblib'

            if not (model_path.exists() and scaler_path.exists()):
                return False

            if not signature_path.exists():
                self.logger.warning("Saved regression model predates the current schema "
                                    "- ignoring it; retrain needed")
                return False

            if joblib.load(signature_path) != self.model_signature():
                self.logger.warning("Saved regression schema mismatch - ignoring it; "
                                    "retrain needed")
                return False

            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.is_trained = True
            self.logger.info("Regression model loaded from disk")
            return True

        except Exception as e:
            self.logger.error(f"Error loading regression model: {e}")
            return False

    def get_required_data_points(self) -> int:
        """Need enough points for feature calculation"""
        return 50
