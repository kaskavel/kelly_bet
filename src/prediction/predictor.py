
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
from .algorithms.svm_algorithm import SVMAlgorithm
from .calibration import EnsembleCalibrator
from ..trading.barriers import BarrierPolicy


class PredictionEngine:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.db_path = Path(config['database']['sqlite']['path'])
        
        # Initialize algorithms and weights
        self.algorithms = {}
        self.algorithm_weights = {}
        self._initialize_algorithms()

        # Where the win/loss barriers sit for each asset. Single source of truth
        # shared with sizing and execution.
        self.barrier_policy = BarrierPolicy(config)
        self.logger.info(self.barrier_policy.describe())

        # Maps the raw ensemble score onto observed win frequency. Falls through to
        # the raw score when no calibration has been fitted yet.
        self.calibrator = EnsembleCalibrator()
        if not self.calibrator.is_fitted:
            self.logger.warning(
                "No ensemble calibration fitted - reported probabilities are RAW "
                "SCORES and must not be read as probabilities. "
                "Run: python scripts/fit_calibration.py"
            )
        
    def _initialize_algorithms(self):
        """Initialize all prediction algorithms"""
        algo_configs = self.config.get('prediction', {}).get('algorithms', {})

        # Inject the bet definition into every algorithm's config. The supervised
        # models must be trained on the event actually being wagered on -- which
        # barrier is touched first -- and that requires the same win/loss/max-hold
        # parameters the trading system uses.
        trading = self.config.get('trading', {})
        barrier = {
            'win_threshold': trading.get('win_threshold', 5.0),
            'loss_threshold': trading.get('loss_threshold', 3.0),
            'max_hold_days': trading.get('max_hold_days', 30),
            'trading_fee_percentage': trading.get('trading_fee_percentage', 0.25),
            # Volatility-scaled barrier settings. Must reach the algorithms because
            # each bar's label depends on that bar's trailing volatility.
            'barrier': trading.get('barrier', {}) or {},
        }
        algo_configs = {
            name: {**barrier, **(cfg or {})}
            for name, cfg in algo_configs.items()
        }
        for name in ('sma', 'rsi', 'rf', 'lstm', 'regression', 'svm'):
            algo_configs.setdefault(name, dict(barrier))

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

        # Support Vector Machine
        svm_config = algo_configs.get('svm', {})
        self.algorithms['svm'] = SVMAlgorithm(svm_config)

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
        
        # Auto-train models on first run if needed
        await self._auto_train_models(market_data)
        
        predictions = []
        skipped_uneconomic = 0

        # Process each asset
        for symbol, data in market_data.items():
            if data.empty:
                self.logger.warning(f"No data available for {symbol}")
                continue

            # Place this asset's barriers before scoring it. In volatility mode the
            # barriers depend on the asset's own trailing volatility, and an asset
            # whose barriers cannot cover the round-trip fee is not tradeable at any
            # probability -- so it is dropped here rather than ranked and rejected
            # later.
            spec = self.barrier_policy.for_series(data)
            if spec is None:
                self.logger.debug(f"{symbol}: no volatility estimate, skipping")
                continue
            if not self.barrier_policy.is_economic(spec):
                skipped_uneconomic += 1
                self.logger.debug(f"{symbol}: {self.barrier_policy.rejection_reason(spec)}")
                continue

            # Get predictions from each algorithm
            asset_predictions = await self._predict_asset(symbol, data)

            if asset_predictions:
                # Calculate ensemble score
                ensemble_score = self._calculate_ensemble_score(asset_predictions)
                
                calibrated = self._calibrated_probability(ensemble_score)

                prediction_result = {
                    'symbol': symbol,
                    # `probability` is the calibrated figure -- this is what sizing
                    # and thresholds consume. The raw score is kept alongside it so
                    # the dashboard can show both and so calibration can be refitted.
                    'probability': calibrated,
                    'raw_score': ensemble_score,
                    'is_calibrated': self.calibrator.is_fitted,
                    'current_price': float(data['Close'].iloc[-1]),
                    # This asset's own barriers. Carried on the prediction so that
                    # sizing and execution use exactly the barriers the score was
                    # produced against, rather than re-deriving them and drifting.
                    'win_threshold': spec.win_pct,
                    'loss_threshold': spec.loss_pct,
                    'barrier_mode': spec.mode,
                    'sigma_pct': spec.sigma_pct,
                    'break_even_pct': self.barrier_policy.break_even(
                        spec.win_pct, spec.loss_pct) * 100.0,
                    'algorithms': asset_predictions,
                    'timestamp': datetime.now().isoformat()
                }

                predictions.append(prediction_result)
                
                # Store individual algorithm predictions
                await self._store_predictions(symbol, asset_predictions)
        
        if skipped_uneconomic:
            self.logger.info(
                f"Skipped {skipped_uneconomic} asset(s) whose barriers cannot cover "
                f"the round-trip fee (break-even above "
                f"{self.barrier_policy.max_break_even_pct:.1f}%)")

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
        Weighted ensemble score from individual algorithm predictions.

        NOTE: this is a SCORE, not a probability. Use _calibrated_probability() to get
        a number that can legitimately be compared against a threshold or fed into
        Kelly sizing.
        """
        if not asset_predictions:
            return 50.0  # Default neutral score

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

    def _calibrated_probability(self, raw_score: float) -> float:
        """
        Map the raw ensemble score onto observed win frequency.

        The raw score is centred near 50 with sigma around 11 across the universe, so
        the top-ranked asset reads 67-72% on every cycle regardless of whether any
        information exists. Calibration is what turns that ordering into a number
        that can be compared against the 43.75% break-even.
        """
        return self.calibrator.calibrate(raw_score)
    
    async def _store_predictions(self, symbol: str, predictions: List[Dict]):
        """Store individual algorithm predictions in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        try:
            for pred in predictions:
                # Cast to a native float. numpy float32 (from the LSTM and other
                # sklearn/keras outputs) does not adapt to a SQLite REAL and lands as
                # a 4-byte BLOB -- 82,429 of 698,233 stored rows were affected, and
                # any aggregate over them silently returns nonsense.
                probability = pred['probability']
                if probability is None:
                    continue
                cursor.execute('''
                INSERT INTO predictions (symbol, timestamp, algorithm, probability)
                VALUES (?, ?, ?, ?)
                ''', (symbol, timestamp, pred['algorithm'], float(probability)))

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
    
    async def update_algorithm_performance(self, bet_id: str, symbol: str,
                                           actual_outcome: bool) -> Dict[str, float]:
        """
        Score each algorithm against a resolved bet and recompute ensemble weights.

        This was a stub ("For now, this is a placeholder") and was never called, so
        the "performance tracking: system learns which algorithms work best over
        time" behaviour in the README never existed. Weights sat frozen at 0.2 from
        2025-09-04 onward and algorithm_performance held zero rows.

        Called from resolve_bet_outcomes() once a bet closes at a price barrier.
        """
        conn = sqlite3.connect(self.db_path, timeout=60.0)
        cursor = conn.cursor()

        try:
            # The per-algorithm predictions recorded when the bet was placed.
            cursor.execute('''
            SELECT algorithm, probability, timestamp FROM bet_predictions
            WHERE bet_id = ?
            ''', (bet_id,))
            rows = cursor.fetchall()

            if not rows:
                self.logger.debug(f"No stored algorithm predictions for bet {bet_id}")
                return {}

            outcome = 1 if actual_outcome else 0

            for algorithm, probability, timestamp in rows:
                if probability is None:
                    continue
                probability = float(probability)

                # Brier score for this single prediction: squared error between the
                # stated probability and what happened. Lower is better. This scores
                # CALIBRATION, not just direction -- an algorithm that says 90% and
                # is right 50% of the time is penalised heavily, which is exactly the
                # failure mode that went undetected.
                brier = (probability / 100.0 - outcome) ** 2

                cursor.execute('''
                INSERT OR REPLACE INTO algorithm_performance
                (algorithm, symbol, prediction_timestamp, predicted_probability,
                 actual_outcome, accuracy_score)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (algorithm, symbol, timestamp, probability, outcome, brier))

            conn.commit()

        except Exception as e:
            self.logger.error(f"Error updating algorithm performance: {e}")
            conn.rollback()
            return {}
        finally:
            conn.close()

        return await self._recalculate_weights()

    async def _recalculate_weights(self) -> Dict[str, float]:
        """
        Recompute ensemble weights from each algorithm's Brier score.

        Weight is inversely proportional to mean squared error, so a better-calibrated
        algorithm earns more of the vote. Weights are floored so no algorithm is fully
        silenced on a small sample, and left equal until there is enough evidence.
        """
        min_predictions = self.config.get('prediction', {}).get('ensemble', {}).get(
            'minimum_predictions', 10)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
            SELECT algorithm, COUNT(*), AVG(accuracy_score)
            FROM algorithm_performance
            WHERE accuracy_score IS NOT NULL
            GROUP BY algorithm
            ''')
            stats = {row[0]: {'n': row[1], 'brier': row[2]} for row in cursor.fetchall()}
        except Exception as e:
            self.logger.error(f"Error reading algorithm performance: {e}")
            return self.algorithm_weights
        finally:
            conn.close()

        eligible = {name: s for name, s in stats.items()
                    if name in self.algorithms
                    and s['n'] >= min_predictions
                    and s['brier'] is not None
                    and s['brier'] > 0}

        if not eligible:
            self.logger.info(f"Not enough resolved predictions to reweight "
                             f"(need {min_predictions} per algorithm); keeping current weights")
            return self.algorithm_weights

        # Inverse-Brier weighting with a floor.
        raw = {name: 1.0 / s['brier'] for name, s in eligible.items()}
        total = sum(raw.values())
        floor = 0.02

        new_weights = dict(self.algorithm_weights)
        for name in self.algorithms:
            if name in raw:
                new_weights[name] = max(floor, raw[name] / total)
            else:
                # No evidence yet: hold at the floor rather than a full share.
                new_weights[name] = floor

        # Renormalise to sum to 1.
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            new_weights = {k: v / total_weight for k, v in new_weights.items()}

        self.algorithm_weights = new_weights
        await self._persist_weights(stats)

        parts = []
        for name, weight in sorted(new_weights.items(), key=lambda kv: -kv[1]):
            stat = stats.get(name)
            if stat and stat.get('brier') is not None:
                parts.append(f"{name}={weight:.3f} "
                             f"(brier {stat['brier']:.4f}, n={stat['n']})")
            else:
                parts.append(f"{name}={weight:.3f} (no data)")
        self.logger.info("Updated ensemble weights: " + ", ".join(parts))
        return new_weights

    async def _persist_weights(self, stats: Dict[str, Dict]):
        """Write current weights and their supporting statistics to the database."""
        conn = sqlite3.connect(self.db_path, timeout=60.0)
        cursor = conn.cursor()

        try:
            for name, weight in self.algorithm_weights.items():
                stat = stats.get(name, {})
                cursor.execute('''
                INSERT INTO algorithm_weights
                    (algorithm, weight, performance_score, total_predictions, last_updated)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(algorithm) DO UPDATE SET
                    weight = excluded.weight,
                    performance_score = excluded.performance_score,
                    total_predictions = excluded.total_predictions,
                    last_updated = CURRENT_TIMESTAMP
                ''', (name, weight, stat.get('brier'), stat.get('n', 0)))
            conn.commit()
        except Exception as e:
            self.logger.error(f"Error persisting algorithm weights: {e}")
            conn.rollback()
        finally:
            conn.close()

    async def resolve_bet_outcomes(self) -> int:
        """
        Score every closed bet that has not yet been fed back into the weights.

        Only bets that resolved at a PRICE barrier are used. A time-barrier exit says
        nothing about whether the win barrier would have been touched, so counting it
        as a win or loss would corrupt the very calibration this feeds.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
            SELECT b.bet_id, b.symbol, b.status
            FROM bets b
            WHERE b.status IN ('won', 'lost')
              AND (b.exit_reason IS NULL OR b.exit_reason IN ('win_barrier', 'loss_barrier'))
              AND EXISTS (SELECT 1 FROM bet_predictions p WHERE p.bet_id = b.bet_id)
              AND NOT EXISTS (
                  SELECT 1 FROM algorithm_performance ap
                  WHERE ap.symbol = b.symbol
                    AND ap.prediction_timestamp = (
                        SELECT MIN(p2.timestamp) FROM bet_predictions p2 WHERE p2.bet_id = b.bet_id
                    )
              )
            ''')
            pending = cursor.fetchall()
        except Exception as e:
            self.logger.error(f"Error finding unresolved bets: {e}")
            return 0
        finally:
            conn.close()

        if not pending:
            self.logger.debug("No newly resolved bets to score")
            return 0

        self.logger.info(f"Scoring {len(pending)} resolved bet(s) into algorithm performance")
        for bet_id, symbol, status in pending:
            await self.update_algorithm_performance(bet_id, symbol, status == 'won')

        return len(pending)
    
    async def _auto_train_models(self, market_data: Dict[str, pd.DataFrame]):
        """Auto-train models if they haven't been trained yet"""
        untrained_models = []
        
        for algo_name, algorithm in self.algorithms.items():
            if not algorithm.is_trained and algo_name in ['rf', 'lstm', 'regression', 'svm']:
                untrained_models.append((algo_name, algorithm))
        
        if not untrained_models:
            return
        
        self.logger.info(f"Auto-training {len(untrained_models)} untrained models...")
        
        # Build a training set across many assets.
        #
        # This used to take five mega-cap US stocks and pd.concat(ignore_index=True)
        # them into one frame. That fabricates a price discontinuity at each seam --
        # rolling indicators computed across the boundary mix two unrelated assets --
        # and it means every scaler is fitted to a $100-$400 price range, then applied
        # at prediction time to HK$, JPY and crypto levels far outside it.
        #
        # Instead: compute per-asset features separately, then concatenate the
        # feature rows. Each asset's indicators stay within that asset, and a wide
        # cross-section gives the scalers a representative range.
        training_frames = self._build_training_frames(market_data)

        if not training_frames:
            self.logger.warning("Insufficient data for model training")
            return

        combined_training_data = pd.concat(training_frames, ignore_index=False)
        self.logger.info(f"Assembled training set: {len(combined_training_data)} rows "
                         f"from {len(training_frames)} assets")

        # Train each untrained model
        for algo_name, algorithm in untrained_models:
            try:
                self.logger.info(f"Training {algo_name} model...")
                await algorithm.train(combined_training_data)
                if algorithm.is_trained:
                    self.logger.info(f"{algo_name} model trained successfully")
                else:
                    self.logger.warning(f"{algo_name} model training failed")
            except Exception as e:
                self.logger.error(f"Error training {algo_name}: {e}")

    # Minimum bars an asset must have to contribute to training.
    MIN_TRAINING_BARS = 80
    # Cap on assets used for training, to keep fit times reasonable.
    MAX_TRAINING_ASSETS = 120

    def _build_training_frames(self, market_data: Dict[str, pd.DataFrame]) -> List[pd.DataFrame]:
        """
        Select a broad, per-asset-clean training set.

        Each returned frame is one asset's own OHLCV history, tagged with its symbol
        so downstream code can group by asset instead of treating the concatenation
        as a single continuous series.
        """
        eligible = [
            (symbol, data) for symbol, data in market_data.items()
            if data is not None and not data.empty and len(data) >= self.MIN_TRAINING_BARS
        ]

        # Prefer the longest histories, then take a deterministic spread across the
        # universe rather than five names from one sector.
        eligible.sort(key=lambda item: (-len(item[1]), item[0]))
        selected = eligible[:self.MAX_TRAINING_ASSETS]

        frames = []
        for symbol, data in selected:
            frame = data.copy()
            frame['symbol'] = symbol
            frames.append(frame)

        if frames:
            self.logger.info(f"Training on {len(frames)} assets "
                             f"(>= {self.MIN_TRAINING_BARS} bars each)")
        return frames
    
    async def cleanup(self):
        """Clean up prediction engine"""
        self.logger.info("Prediction engine cleanup complete")