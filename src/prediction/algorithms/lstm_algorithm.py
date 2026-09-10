"""
LSTM (Long Short-Term Memory) based prediction algorithm.

Predicts P(the +win barrier is touched before the -loss barrier) from a sequence of
scale-free technical features, with a sigmoid output trained on binary cross-entropy.

Two defects in the previous version made this model actively harmful:

  1. It regressed a 5-bar forward return with MSE and then mapped the point forecast
     through `1 / (1 + exp(-r * 10))`. That scaling factor was arbitrary, and it
     answers P(return > 0), not P(win barrier first).

  2. Its MinMaxScaler was fitted on raw `Close` and `Volume` for five US mega-caps,
     then applied at prediction time to every asset in the universe. A JPY 4,899 or
     BTC 60,000 quote scaled far outside [0, 1], the network saturated, and the
     sigmoid pinned at its 90% ceiling: the stored predictions averaged 88.9%, with
     98.6% of all calls above 60%, on an event whose base rate is around 37%.

Both are fixed here: features are ratios and z-scores (scale-free by construction),
and the target is the barrier label itself.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from pathlib import Path
import joblib

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import StandardScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    keras = None

from .base_algorithm import BasePredictionAlgorithm


class LSTMAlgorithm(BasePredictionAlgorithm):
    def __init__(self, config: dict):
        super().__init__("LSTM Neural Network", config)

        # Parameters are set before the TensorFlow check so that every attribute
        # exists even when TF is missing. An early return here previously left
        # `sequence_length` undefined, and get_required_data_points() raised
        # AttributeError as soon as anything inspected the algorithm.
        self.sequence_length = config.get('sequence_length', 30)
        self.lstm_units = config.get('lstm_units', 50)
        self.dropout_rate = config.get('dropout_rate', 0.2)
        self.epochs = config.get('epochs', 50)
        self.batch_size = config.get('batch_size', 32)

        self.model = None
        self.model_dir = Path('models/lstm')

        if not TENSORFLOW_AVAILABLE:
            self.logger.warning("TensorFlow not available - LSTM disabled. "
                                "Install with: pip install tensorflow>=2.13.0")
            self.scaler = None
            return

        # StandardScaler, not MinMaxScaler: min/max are set by the extremes of the
        # training sample and clip anything outside them, which is precisely how
        # out-of-range assets got mangled before.
        self.scaler = StandardScaler()

        # Model persistence
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Load existing model if available (and schema-compatible)
        self._load_model()

    async def predict(self, data: pd.DataFrame) -> Optional[float]:
        """Probability (0-100) that the win barrier is touched first."""
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
            features = self._stationary_feature_frame(data)
            if features is None:
                return None

            features = features.dropna()
            if len(features) < self.sequence_length:
                return None

            scaled = self.scaler.transform(features.values[-self.sequence_length:])
            if not np.isfinite(scaled).all():
                self.logger.warning("Scaled sequence contains NaN/inf, skipping prediction")
                return None

            latest_sequence = scaled.reshape(1, self.sequence_length, scaled.shape[1])

            # Sigmoid output IS the probability. No post-hoc squash.
            probability = float(self.model.predict(latest_sequence, verbose=0)[0][0]) * 100.0
            probability = max(0.0, min(100.0, probability))

            self.logger.debug(f"LSTM prediction: {probability:.2f}%")
            return probability

        except Exception as e:
            self.logger.error(f"Error in LSTM prediction: {e}")
            return None

    async def train(self, data: pd.DataFrame, target_data: pd.DataFrame = None):
        """Train the LSTM classifier on first-touch barrier labels."""
        if not TENSORFLOW_AVAILABLE:
            self.logger.error("TensorFlow not available for training")
            return

        self.logger.info("Training LSTM model...")

        try:
            X, y = self._prepare_sequences(data)
            if X is None or len(X) < 200:
                self.logger.warning("Insufficient sequence data for LSTM training")
                return

            if len(np.unique(y)) < 2:
                self.logger.warning("LSTM training data is single-class")
                return

            # Time-ordered hold-out with a purge gap, so no training sequence's label
            # window overlaps the validation period.
            embargo = int(self.config.get('max_hold_days', 15))
            n_val = max(1, int(len(X) * 0.2))
            split = len(X) - n_val
            train_end = max(1, split - embargo)

            X_train, y_train = X[:train_end], y[:train_end]
            X_val, y_val = X[split:], y[split:]

            self.logger.info(f"LSTM split: {len(X_train)} train, {len(X_val)} val, "
                             f"{embargo}-sequence embargo; "
                             f"train base rate {y_train.mean():.1%}")

            self._build_model(X_train.shape[1], X_train.shape[2])

            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=0,
                shuffle=False,  # Keep time series order
                callbacks=[EarlyStopping(monitor='val_loss', patience=8,
                                         restore_best_weights=True)],
            )

            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            self.logger.info(f"LSTM training complete - "
                             f"loss: {final_loss:.4f}, val_loss: {final_val_loss:.4f}")

            self._log_sequence_calibration(X_val, y_val, y_train.mean())

            self._save_model()
            self.is_trained = True

        except Exception as e:
            self.logger.error(f"Error in LSTM training: {e}")
            self.is_trained = False

    def _log_sequence_calibration(self, X_val, y_val, base_rate: float):
        """Reliability of the validation predictions, against the base-rate baseline."""
        try:
            from sklearn.metrics import brier_score_loss

            if len(X_val) == 0 or len(np.unique(y_val)) < 2:
                self.logger.info("LSTM validation split unusable for calibration")
                return

            probs = self.model.predict(X_val, verbose=0).ravel()
            brier = brier_score_loss(y_val, probs)
            baseline = brier_score_loss(y_val, np.full(len(y_val), base_rate))

            self.logger.info(f"LSTM Brier: {brier:.4f} vs base-rate baseline "
                             f"{baseline:.4f} "
                             f"({'BETTER' if brier < baseline else 'NO BETTER'})")

            for lo, hi in ((0.0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 1.01)):
                mask = (probs >= lo) & (probs < hi)
                if mask.sum() >= 10:
                    self.logger.info(f"  predicted {lo:.0%}-{hi:.0%}: "
                                     f"n={int(mask.sum()):5d} observed {y_val[mask].mean():.1%}")
        except Exception as e:
            self.logger.debug(f"Could not log LSTM calibration: {e}")

    def _build_model(self, sequence_length: int, n_features: int):
        """Build LSTM classifier architecture"""
        self.model = Sequential([
            LSTM(units=self.lstm_units, return_sequences=True,
                 input_shape=(sequence_length, n_features)),
            Dropout(self.dropout_rate),

            LSTM(units=max(4, self.lstm_units // 2), return_sequences=False),
            Dropout(self.dropout_rate),

            Dense(25, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(1, activation='sigmoid'),  # Probability output
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )

        self.logger.info(f"LSTM classifier built: {self.model.count_params()} parameters")

    def _prepare_sequences(self, data: pd.DataFrame):
        """
        Build (sequences, labels) for the barrier event.

        Sequences never span an asset boundary: on a pooled multi-asset frame each
        asset is windowed separately and the results concatenated.
        """
        try:
            if 'symbol' in data.columns and data['symbol'].nunique() > 1:
                groups = [group for _, group in data.groupby('symbol', sort=False)]
            else:
                groups = [data]

            # Fit the scaler once, across the pooled feature rows.
            feature_frames = []
            label_series = []
            for group in groups:
                features = self._stationary_feature_frame(group)
                if features is None:
                    continue
                labels = self._barrier_labels(
                    group,
                    win_pct=float(self.config.get('win_threshold', 5.0)),
                    loss_pct=float(self.config.get('loss_threshold', 3.0)),
                    max_hold=int(self.config.get('max_hold_days', 15)),
                )
                valid = ~(features.isna().any(axis=1) | labels.isna())
                if valid.sum() < self.sequence_length + 10:
                    continue
                feature_frames.append(features[valid])
                label_series.append(labels[valid])

            if not feature_frames:
                return None, None

            self.scaler.fit(pd.concat(feature_frames, ignore_index=True).values)

            sequences, targets = [], []
            for features, labels in zip(feature_frames, label_series):
                scaled = self.scaler.transform(features.values)
                label_values = labels.to_numpy(dtype=float)
                for i in range(self.sequence_length, len(scaled)):
                    window = scaled[i - self.sequence_length:i]
                    if not np.isfinite(window).all():
                        continue
                    sequences.append(window)
                    targets.append(label_values[i - 1])

            if not sequences:
                return None, None

            X = np.asarray(sequences, dtype=np.float32)
            y = np.asarray(targets, dtype=np.float32)

            self.logger.info(f"Prepared LSTM data: X {X.shape}, y {y.shape}, "
                             f"base rate {y.mean():.1%}")
            return X, y

        except Exception as e:
            self.logger.error(f"Error preparing LSTM training data: {e}")
            return None, None

    def model_signature(self) -> Dict:
        """What this model predicts, and from what. Invalidates stale saved models."""
        return {
            'schema': 2,
            'label': 'first_touch_barrier',
            'output': 'sigmoid_probability',
            'features': tuple(self.STATIONARY_FEATURES),
            'sequence_length': int(self.sequence_length),
            'win_threshold': float(self.config.get('win_threshold', 5.0)),
            'loss_threshold': float(self.config.get('loss_threshold', 3.0)),
            'max_hold_days': int(self.config.get('max_hold_days', 15)),
        }

    def _save_model(self):
        """Save trained model, scaler and schema signature"""
        try:
            model_path = self.model_dir / 'lstm_model.keras'
            self.model.save(model_path)
            joblib.dump(self.scaler, self.model_dir / 'lstm_scaler.joblib')
            joblib.dump(self.model_signature(), self.model_dir / 'lstm_signature.joblib')
            self.logger.info(f"LSTM model saved to {model_path}")
        except Exception as e:
            self.logger.error(f"Error saving LSTM model: {e}")

    def _load_model(self):
        """Load saved model and scaler, only if trained on the current schema"""
        if not TENSORFLOW_AVAILABLE:
            return False

        try:
            model_path = self.model_dir / 'lstm_model.keras'
            scaler_path = self.model_dir / 'lstm_scaler.joblib'
            signature_path = self.model_dir / 'lstm_signature.joblib'

            if not (model_path.exists() and scaler_path.exists()):
                return False

            if not signature_path.exists():
                self.logger.warning("Saved LSTM predates the current feature schema "
                                    "- ignoring it; retrain needed")
                return False

            if joblib.load(signature_path) != self.model_signature():
                self.logger.warning("Saved LSTM schema mismatch - ignoring it; retrain needed")
                return False

            self.model = keras.models.load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            self.is_trained = True
            self.logger.info("LSTM model loaded from disk")
            return True

        except Exception as e:
            self.logger.error(f"Error loading LSTM model: {e}")
            return False

    def get_required_data_points(self) -> int:
        """Need enough points for the sequence plus indicator warm-up"""
        return self.sequence_length + 30
