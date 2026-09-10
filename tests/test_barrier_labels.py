#!/usr/bin/env python3
"""
Tests for the first-touch barrier labelling and the stationary feature set.

These replace the old label, `Close.shift(-5)/Close - 1 > 0.03`, which was a
fixed-horizon terminal return: path-independent, blind to the stop, and measured over
5 bars while the average position was held 18.8 days. The models were being trained
to predict something nobody was betting on.

The features are tested for scale invariance because the previous set led with raw
price levels. A MinMaxScaler fitted on five US mega-caps was applied to JPY 4,899 and
crypto quotes, which saturated the LSTM.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.prediction.algorithms.base_algorithm import BasePredictionAlgorithm


class Probe(BasePredictionAlgorithm):
    """Concrete subclass so the shared helpers can be exercised."""

    async def predict(self, data):
        return None

    async def train(self, data, target_data=None):
        return None


def series(close, spread=0.001, symbol=None):
    """OHLCV frame from a close series, with a small intraday spread."""
    close = np.asarray(close, dtype=float)
    frame = pd.DataFrame({
        'Open': close,
        'High': close * (1 + spread),
        'Low': close * (1 - spread),
        'Close': close,
        'Volume': np.full(len(close), 1_000_000.0),
    })
    if symbol:
        frame['symbol'] = symbol
    return frame


class BarrierLabelTestCase(unittest.TestCase):

    def setUp(self):
        self.probe = Probe('probe', {'win_threshold': 5.0, 'loss_threshold': 3.0,
                                     'max_hold_days': 10})

    def test_rising_series_touches_win_barrier(self):
        """A steady 1%/bar climb reaches +5% before -3%."""
        labels = self.probe._barrier_labels(series(100 * 1.01 ** np.arange(60)),
                                            5.0, 3.0, 10).dropna()
        self.assertTrue((labels == 1).all())

    def test_falling_series_touches_loss_barrier(self):
        """A steady 1%/bar decline reaches -3% first."""
        labels = self.probe._barrier_labels(series(100 * 0.99 ** np.arange(60)),
                                            5.0, 3.0, 10).dropna()
        self.assertTrue((labels == 0).all())

    def test_flat_series_times_out_as_not_a_win(self):
        """Reaching neither barrier is not a win."""
        labels = self.probe._barrier_labels(series(np.full(60, 100.0), spread=0.0),
                                            5.0, 3.0, 10).dropna()
        self.assertTrue((labels == 0).all())

    def test_unresolved_tail_is_not_labelled(self):
        """Bars without a full forward window must be NaN, never filled."""
        labels = self.probe._barrier_labels(series(np.full(40, 100.0), spread=0.0),
                                            5.0, 3.0, 10)
        self.assertTrue(labels.tail(10).isna().all())

    def test_random_walk_matches_barrier_geometry(self):
        """
        A driftless walk with a generous time barrier approaches l/(w+l).

        For a 5%/3% bet that is 37.5% -- the real neutral point, and the number every
        threshold in this system should have been measured against instead of 50%.
        """
        rng = np.random.default_rng(7)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 6000)))
        labels = self.probe._barrier_labels(series(close, spread=0.004),
                                            5.0, 3.0, 120).dropna()

        self.assertGreater(len(labels), 1000)
        self.assertAlmostEqual(labels.mean(), 0.375, delta=0.04)

    def test_shorter_time_barrier_lowers_the_win_rate(self):
        """
        A tight time barrier makes timeouts dominate.

        This is why max_hold_days is the dominant parameter: shrink it and the win
        barrier is rarely reached, so the strategy is negative-EV regardless of the
        model.
        """
        rng = np.random.default_rng(11)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, 4000)))
        frame = series(close, spread=0.003)

        short = self.probe._barrier_labels(frame, 5.0, 3.0, 10).dropna().mean()
        long = self.probe._barrier_labels(frame, 5.0, 3.0, 120).dropna().mean()

        self.assertLess(short, long)

    def test_labels_never_span_two_assets(self):
        """On a pooled frame, each asset is labelled from its own history only."""
        rising = series(100 * 1.01 ** np.arange(60), symbol='UP')
        falling = series(100 * 0.99 ** np.arange(60), symbol='DOWN')
        pooled = pd.concat([rising, falling], ignore_index=True)

        labels = self.probe._barrier_labels(pooled, 5.0, 3.0, 10)

        up = labels[pooled['symbol'] == 'UP'].dropna()
        down = labels[pooled['symbol'] == 'DOWN'].dropna()

        self.assertTrue((up == 1).all())
        self.assertTrue((down == 0).all())


class BarrierEconomicsTestCase(unittest.TestCase):

    def setUp(self):
        self.probe = Probe('probe', {'win_threshold': 5.0, 'loss_threshold': 3.0,
                                     'max_hold_days': 10})

    def test_win_exit_returns_the_win_threshold(self):
        outcomes = self.probe.barrier_outcomes(
            series(100 * 1.01 ** np.arange(60)), 5.0, 3.0, 10).dropna()
        self.assertTrue((outcomes['label'] == 1).all())
        self.assertTrue(np.allclose(outcomes['exit_return'], 0.05))

    def test_loss_exit_returns_the_loss_threshold(self):
        outcomes = self.probe.barrier_outcomes(
            series(100 * 0.99 ** np.arange(60)), 5.0, 3.0, 10).dropna()
        self.assertTrue((outcomes['label'] == 0).all())
        self.assertTrue(np.allclose(outcomes['exit_return'], -0.03))

    def test_timeout_books_the_terminal_move_not_a_stop_out(self):
        """
        A bet that reaches neither barrier closes at market.

        Booking timeouts at the full stop distance would badly overstate losses -- it
        is the difference between a small drift and a -3% hit.
        """
        # Drifts up 0.1%/bar: never reaches +5% or -3% within 10 bars.
        outcomes = self.probe.barrier_outcomes(
            series(100 * 1.001 ** np.arange(60), spread=0.0), 5.0, 3.0, 10).dropna()

        self.assertTrue((outcomes['label'] == -1).all())
        self.assertTrue((outcomes['exit_return'] > 0).all())
        self.assertTrue((outcomes['exit_return'] < 0.05).all())

    def test_bars_held_is_recorded(self):
        outcomes = self.probe.barrier_outcomes(
            series(100 * 1.01 ** np.arange(60)), 5.0, 3.0, 10).dropna()
        self.assertTrue((outcomes['bars_held'] >= 1).all())
        self.assertTrue((outcomes['bars_held'] <= 10).all())


class StationaryFeatureTestCase(unittest.TestCase):

    def setUp(self):
        self.probe = Probe('probe', {})
        rng = np.random.default_rng(3)
        close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.018, 400)))
        self.frame = pd.DataFrame({
            'Open': close,
            'High': close * (1 + abs(rng.normal(0, 0.006, 400))),
            'Low': close * (1 - abs(rng.normal(0, 0.006, 400))),
            'Close': close,
            'Volume': rng.integers(1_000_000, 5_000_000, 400).astype(float),
        })

    def test_features_are_scale_invariant(self):
        """
        Rescaling price must not change any feature.

        This is what lets a single scaler serve US equities, JPY quotes and crypto.
        """
        baseline = self.probe._stationary_feature_frame(self.frame)
        self.assertIsNotNone(baseline)

        for scale in (4899.0, 0.0002, 600.0):
            scaled = self.frame.copy()
            for column in ('Open', 'High', 'Low', 'Close'):
                scaled[column] = scaled[column] * scale

            features = self.probe._stationary_feature_frame(scaled)
            drift = (baseline - features).abs().max().max()
            self.assertLess(drift, 1e-8, f"features drifted at scale {scale}")

    def test_features_are_order_one(self):
        """No feature should be denominated in price units."""
        features = self.probe._stationary_feature_frame(self.frame).dropna()
        self.assertFalse(features.empty)

        # RSI is 0-100 by construction; everything else should be small.
        others = features.drop(columns=['RSI'])
        self.assertLess(others.abs().max().max(), 50.0)

    def test_no_infinities_survive(self):
        """Divide-by-zero must become NaN, so dropna() removes it."""
        flat = self.frame.copy()
        flat['Close'] = 100.0
        flat['Open'] = 100.0
        flat['High'] = 100.0
        flat['Low'] = 100.0

        features = self.probe._stationary_feature_frame(flat)
        self.assertFalse(np.isinf(features.to_numpy(dtype=float)).any())


class PurgedSplitTestCase(unittest.TestCase):

    def setUp(self):
        self.probe = Probe('probe', {'max_hold_days': 15})

    def test_split_is_time_ordered_with_an_embargo(self):
        """
        Train comes strictly before test, with a gap.

        train_test_split(random_state=42) shuffles by default, which leaked
        overlapping label windows between train and test and made every reported test
        accuracy meaningless.
        """
        features = pd.DataFrame({'a': np.arange(1000, dtype=float)})
        targets = pd.Series(np.tile([0, 1], 500))

        X_train, X_test, y_train, y_test = self.probe._purged_split(features, targets)

        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_test), 0)

        # Every training index precedes every test index...
        self.assertLess(X_train.index.max(), X_test.index.min())
        # ...with at least max_hold_days of gap between them.
        self.assertGreaterEqual(X_test.index.min() - X_train.index.max(), 15)

        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))


if __name__ == '__main__':
    unittest.main()
