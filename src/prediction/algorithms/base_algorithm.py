"""
Base class for prediction algorithms
All prediction algorithms should inherit from this class.
"""

from abc import ABC, abstractmethod
import numpy as np
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
    
    # Feature columns that are scale-free: safe to pool across assets and currencies.
    STATIONARY_FEATURES = [
        'RSI',
        'BB_Position',
        'Volume_Ratio',
        'Price_Change_1', 'Price_Change_5', 'Price_Change_10',
        'Volatility',
        'Price_to_SMA5', 'Price_to_SMA20', 'SMA5_to_SMA20',
        'EMA12_to_EMA26',
        'MACD_Norm', 'MACD_Hist_Norm',
        'High_Low_Range',
        'Return_Z_20',
        'Vol_Ratio_5_20',
    ]

    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate common technical indicators.

        When the frame carries a `symbol` column (a pooled multi-asset training set),
        indicators are computed PER ASSET. Computing them across a concatenation
        blends one asset's tail into the next asset's head and produces features that
        describe nothing real.
        """
        if 'symbol' in data.columns and data['symbol'].nunique() > 1:
            parts = [
                self._indicators_single_asset(group)
                for _, group in data.groupby('symbol', sort=False)
            ]
            return pd.concat(parts, ignore_index=False)

        return self._indicators_single_asset(data)

    def _indicators_single_asset(self, data: pd.DataFrame) -> pd.DataFrame:
        """Indicators for a single asset's contiguous history."""
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

        # --- Scale-free derivations ---------------------------------------
        # Everything above that is denominated in price (SMA_5, EMA_12, MACD, the
        # Bollinger bands) cannot be pooled across assets: a scaler fitted on a $200
        # US stock produces garbage when applied to a JPY 4,899 quote or a $60,000
        # crypto. These ratios carry the same information without the units.
        df['Price_to_SMA5'] = df['Close'] / df['SMA_5']
        df['Price_to_SMA20'] = df['Close'] / df['SMA_20']
        df['SMA5_to_SMA20'] = df['SMA_5'] / df['SMA_20']
        df['EMA12_to_EMA26'] = df['EMA_12'] / df['EMA_26']

        # MACD normalised by price, so it is comparable across assets.
        df['MACD_Norm'] = df['MACD'] / df['Close']
        df['MACD_Hist_Norm'] = df['MACD_Hist'] / df['Close']

        # Intraday range as a fraction of price.
        df['High_Low_Range'] = (df['High'] - df['Low']) / df['Close']

        # Recent return in units of its own volatility.
        df['Return_Z_20'] = df['Price_Change_1'] / df['Volatility']

        # Short-horizon volatility relative to longer-horizon volatility.
        short_vol = df['Price_Change_1'].rolling(window=5).std()
        df['Vol_Ratio_5_20'] = short_vol / df['Volatility']

        # Guard against divide-by-zero producing infinities that survive dropna().
        df = df.replace([np.inf, -np.inf], np.nan)

        return df

    def _log_calibration(self, name, X_train, y_train, X_test, y_test):
        """
        Report accuracy AND calibration for a probabilistic classifier.

        Accuracy alone is nearly useless here: always predicting the majority class
        on a 37% base rate scores 63%. What Kelly needs is for a predicted 60% to win
        60% of the time, which is what the Brier score and reliability table measure.
        A model that cannot beat the base-rate baseline has no usable signal.
        """
        from sklearn.metrics import brier_score_loss

        train_acc = self.model.score(X_train, y_train)

        if len(y_test) == 0 or y_test.nunique() < 2:
            self.logger.info(f"{name} trained - train accuracy {train_acc:.3f}; "
                             f"test split unusable for evaluation")
            return

        probs = self.model.predict_proba(X_test)[:, 1]
        test_acc = self.model.score(X_test, y_test)
        brier = brier_score_loss(y_test, probs)

        base_rate = float(y_train.mean())
        baseline_brier = brier_score_loss(y_test, np.full(len(y_test), base_rate))

        self.logger.info(
            f"{name} training complete - "
            f"Train accuracy: {train_acc:.3f}, Test accuracy: {test_acc:.3f}, "
            f"Brier: {brier:.4f} vs base-rate baseline {baseline_brier:.4f} "
            f"({'BETTER' if brier < baseline_brier else 'NO BETTER'})"
        )

        y_test_values = y_test.to_numpy()
        for lo, hi in ((0.0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 1.01)):
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() >= 10:
                self.logger.info(f"  predicted {lo:.0%}-{hi:.0%}: n={int(mask.sum()):5d} "
                                 f"observed {y_test_values[mask].mean():.1%}")

    def _purged_split(self, features: pd.DataFrame, targets: pd.Series,
                      test_size: float = 0.2, embargo: int = None):
        """
        Time-ordered train/test split with a purge gap between the two.

        `train_test_split(..., random_state=42)` shuffles by default. For this data
        that is severe leakage on two counts: adjacent bars share overlapping
        rolling-indicator windows, AND each label looks forward up to max_hold bars,
        so a shuffled test row's outcome window overlaps training rows. Reported test
        accuracy under shuffling is meaningless.

        The embargo drops `max_hold` rows between train and test so no training
        label's forward window can reach into the test period.
        """
        if embargo is None:
            embargo = int(self.config.get('max_hold_days', 15))

        n = len(features)
        n_test = max(1, int(n * test_size))
        split = n - n_test
        train_end = max(0, split - embargo)

        X_train = features.iloc[:train_end]
        y_train = targets.iloc[:train_end]
        X_test = features.iloc[split:]
        y_test = targets.iloc[split:]

        self.logger.info(f"Purged time-ordered split: {len(X_train)} train, "
                         f"{len(X_test)} test, {embargo}-row embargo")
        return X_train, X_test, y_train, y_test

    @property
    def barrier_policy(self):
        """
        Barrier placement policy, built from config on first use.

        Volatility-scaled barriers mean the label at each bar depends on that bar's
        trailing volatility, so the policy has to be available wherever labels are
        produced.
        """
        if getattr(self, '_barrier_policy', None) is None:
            from ...trading.barriers import BarrierPolicy
            # Algorithm configs are flattened, so the policy's expected shape is
            # reconstructed here.
            self._barrier_policy = BarrierPolicy({'trading': {
                'win_threshold': self.config.get('win_threshold', 5.0),
                'loss_threshold': self.config.get('loss_threshold', 3.0),
                'trading_fee_percentage': self.config.get('trading_fee_percentage', 0.25),
                'barrier': self.config.get('barrier', {}) or {},
            }})
        return self._barrier_policy

    def _barrier_labels_for(self, data: pd.DataFrame) -> Optional[pd.Series]:
        """Labels for `data` using the configured barrier policy."""
        policy = self.barrier_policy
        max_hold = int(self.config.get('max_hold_days', 30))

        if 'symbol' in data.columns and data['symbol'].nunique() > 1:
            parts = []
            for _, group in data.groupby('symbol', sort=False):
                win, loss = policy.barrier_series(group)
                parts.append(self._barrier_labels_single(group, win, loss, max_hold))
            return pd.concat(parts, ignore_index=False)

        win, loss = policy.barrier_series(data)
        return self._barrier_labels_single(data, win, loss, max_hold)

    def _prepare_barrier_dataset(self, data: pd.DataFrame):
        """
        Build (features, labels) for the first-touch barrier event.

        Returns (None, None) if there is not enough usable data.
        """
        features = self._stationary_feature_frame(data)
        if features is None:
            return None, None

        targets = self._barrier_labels_for(data)
        if targets is None:
            return None, None

        valid = ~(features.isna().any(axis=1) | targets.isna())
        features_clean = features[valid]
        targets_clean = targets[valid].astype(int)

        if len(features_clean) < 100:
            self.logger.warning(f"Only {len(features_clean)} clean labelled rows; "
                                f"need at least 100")
            return None, None

        base_rate = targets_clean.mean()
        self.logger.info(f"Barrier dataset: {len(features_clean)} rows, "
                         f"base rate {base_rate:.1%} "
                         f"(random-walk geometry would give "
                         f"{self.barrier_policy.geometry_probability:.1%}) "
                         f"| {self.barrier_policy.describe()}")

        return features_clean, targets_clean

    def _stationary_feature_frame(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Scale-free feature matrix, safe to pool across assets and currencies.

        Returns None on failure so callers can skip rather than train on nonsense.
        """
        try:
            df = self._calculate_technical_indicators(data)
            missing = [c for c in self.STATIONARY_FEATURES if c not in df.columns]
            if missing:
                self.logger.error(f"Missing expected features: {missing}")
                return None
            return df[self.STATIONARY_FEATURES].copy()
        except Exception as e:
            self.logger.error(f"Error preparing stationary features: {e}")
            return None

    def _barrier_labels(self, data: pd.DataFrame, win_pct: float, loss_pct: float,
                        max_hold: int) -> Optional[pd.Series]:
        """
        Triple-barrier labels: which barrier does this bar's bet touch FIRST?

        1 = the +win_pct barrier is touched before -loss_pct
        0 = the -loss_pct barrier is touched first, or neither within max_hold bars

        This is the event the system actually bets on. The previous label was
        `Close.shift(-5)/Close - 1 > 0.03` -- a fixed-horizon terminal return, which
        is path-independent, ignores the stop entirely, and answers a question nobody
        wagers on. Intraday High/Low are used for touch detection where available.

        Bars whose outcome is not yet determined (the tail of the series) are NaN and
        must be dropped, not filled.
        """
        if 'symbol' in data.columns and data['symbol'].nunique() > 1:
            parts = [
                self._barrier_labels_single(group, win_pct, loss_pct, max_hold)
                for _, group in data.groupby('symbol', sort=False)
            ]
            return pd.concat(parts, ignore_index=False)

        return self._barrier_labels_single(data, win_pct, loss_pct, max_hold)

    def barrier_outcomes(self, data: pd.DataFrame, win_pct: float, loss_pct: float,
                         max_hold: int) -> pd.DataFrame:
        """
        Full economics of the bet opened at each bar, not just the binary label.

        Columns:
            label          1 win barrier first, 0 loss barrier first, -1 neither
            exit_return    realised gross return, decimal (signed)
            bars_held      bars from entry to exit

        The binary label is the right TRAINING target (P(win barrier first)), but a
        simulation needs this: a bet that reaches neither barrier is closed at market
        for whatever the terminal move happens to be, which is usually a small gain
        or loss -- not a full stop-out. Booking timeouts at -loss_pct would badly
        overstate losses.
        """
        if 'symbol' in data.columns and data['symbol'].nunique() > 1:
            parts = [
                self._barrier_outcomes_single(group, win_pct, loss_pct, max_hold)
                for _, group in data.groupby('symbol', sort=False)
            ]
            return pd.concat(parts, ignore_index=False)

        return self._barrier_outcomes_single(data, win_pct, loss_pct, max_hold)

    @staticmethod
    def _as_barrier_array(value, n: int, index) -> np.ndarray:
        """
        Broadcast a barrier percentage to one value per bar.

        Accepts a scalar (fixed barriers) or a per-bar Series (volatility-scaled
        barriers), so the same labelling code serves both modes.
        """
        if np.isscalar(value):
            return np.full(n, float(value))

        series = pd.Series(value)
        if index is not None and not series.index.equals(pd.Index(index)):
            series = series.reindex(index)
        return series.to_numpy(dtype=float)

    def _barrier_outcomes_single(self, data: pd.DataFrame, win_pct, loss_pct,
                                 max_hold: int) -> pd.DataFrame:
        """
        Barrier economics for one asset's contiguous history.

        `win_pct` and `loss_pct` may be scalars or per-bar Series. In volatility mode
        each bar carries its own barrier distance, derived from the trailing
        volatility known at that bar.
        """
        close = data['Close'].to_numpy(dtype=float)
        high = data['High'].to_numpy(dtype=float) if 'High' in data else close
        low = data['Low'].to_numpy(dtype=float) if 'Low' in data else close

        n = len(close)
        win_arr = self._as_barrier_array(win_pct, n, data.index)
        loss_arr = self._as_barrier_array(loss_pct, n, data.index)

        labels = np.full(n, np.nan)
        returns = np.full(n, np.nan)
        held = np.full(n, np.nan)
        win_used = np.full(n, np.nan)
        loss_used = np.full(n, np.nan)

        for i in range(n):
            entry = close[i]
            if not np.isfinite(entry) or entry <= 0:
                continue
            if i + 1 + max_hold > n:
                continue  # forward window incomplete

            bar_win = win_arr[i]
            bar_loss = loss_arr[i]
            if not (np.isfinite(bar_win) and np.isfinite(bar_loss)):
                continue  # volatility unavailable at this bar: not labelled
            if bar_win <= 0 or bar_loss <= 0:
                continue

            upper = entry * (1.0 + bar_win / 100.0)
            lower = entry * (1.0 - bar_loss / 100.0)
            end = min(i + 1 + max_hold, n)

            label = -1
            exit_return = None
            bars = end - 1 - i

            for j in range(i + 1, end):
                hit_up = high[j] >= upper
                hit_down = low[j] <= lower

                if hit_up and hit_down:
                    # Both in one bar: OHLC cannot order them. Assume adverse first.
                    label, exit_return, bars = 0, -bar_loss / 100.0, j - i
                    break
                if hit_up:
                    label, exit_return, bars = 1, bar_win / 100.0, j - i
                    break
                if hit_down:
                    label, exit_return, bars = 0, -bar_loss / 100.0, j - i
                    break

            if exit_return is None:
                # Time barrier: close at market on the last bar of the window.
                exit_return = close[end - 1] / entry - 1.0

            labels[i] = label
            returns[i] = exit_return
            held[i] = bars
            win_used[i] = bar_win
            loss_used[i] = bar_loss

        return pd.DataFrame(
            {'label': labels, 'exit_return': returns, 'bars_held': held,
             'win_pct': win_used, 'loss_pct': loss_used},
            index=data.index,
        )

    def _barrier_labels_single(self, data: pd.DataFrame, win_pct, loss_pct,
                               max_hold: int) -> pd.Series:
        """
        Triple-barrier labels for one asset's contiguous history.

        Derived from barrier_outcomes so there is exactly one implementation of the
        touch logic: 1 if the win barrier came first, 0 for the loss barrier or a
        timeout, NaN where the outcome is undetermined.
        """
        outcomes = self._barrier_outcomes_single(data, win_pct, loss_pct, max_hold)
        labels = (outcomes['label'] == 1).astype(float)
        labels[outcomes['label'].isna()] = np.nan
        return pd.Series(labels, index=data.index, name='barrier_label')