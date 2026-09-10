#!/usr/bin/env python3
"""
Tests for the ensemble probability calibration.

This is the fix for the system's central defect: over 84 closed bets the mean
reported probability at placement was 63.7% while the realised win rate was 44.1%,
rejected at z = -3.75. The ranking carried signal but the level did not, and the
level is what Kelly sizing and every threshold consumed.
"""

import tempfile
import unittest
from pathlib import Path
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.prediction.calibration import EnsembleCalibrator


class CalibrationTestCase(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'calibration.json'

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def _overstated_sample(n=4000, seed=5):
        """
        Scores that rank correctly but overstate the level, as this system's did.

        True probability rises from ~25% to ~55% while the reported score runs
        50-75 -- the same shape as the observed reliability table.
        """
        rng = np.random.default_rng(seed)
        scores = rng.uniform(50, 75, n)
        true_probability = 0.25 + (scores - 50) / 25 * 0.30
        outcomes = (rng.random(n) < true_probability).astype(int)
        return scores, outcomes, true_probability

    def test_unfitted_calibrator_passes_scores_through(self):
        """With no fit, behaviour degrades to the status quo rather than to silence."""
        calibrator = EnsembleCalibrator(self.path)
        self.assertFalse(calibrator.is_fitted)
        self.assertEqual(calibrator.calibrate(62.0), 62.0)

    def test_fit_corrects_a_systematic_overstatement(self):
        """A score of 62 that wins 39% of the time must calibrate down towards 39."""
        scores, outcomes, _ = self._overstated_sample()

        calibrator = EnsembleCalibrator(self.path)
        calibrator.fit(scores, outcomes)

        self.assertTrue(calibrator.is_fitted)

        calibrated = calibrator.calibrate(62.0)
        true_at_62 = (0.25 + (62 - 50) / 25 * 0.30) * 100

        self.assertLess(calibrated, 62.0, "calibration must remove the overstatement")
        self.assertAlmostEqual(calibrated, true_at_62, delta=8.0)

    def test_calibration_beats_reading_the_score_as_a_probability(self):
        """The whole point: a calibrated probability scores better than the raw score."""
        scores, outcomes, _ = self._overstated_sample()

        calibrator = EnsembleCalibrator(self.path)
        report = calibrator.fit(scores, outcomes)

        self.assertLess(report.brier_calibrated, report.brier_raw)
        self.assertTrue(report.improves_on_baseline)

    def test_monotonicity_is_preserved(self):
        """Isotonic keeps the ordering the ensemble gets right."""
        scores, outcomes, _ = self._overstated_sample()

        calibrator = EnsembleCalibrator(self.path)
        calibrator.fit(scores, outcomes)

        calibrated = [calibrator.calibrate(s) for s in range(50, 76)]
        for earlier, later in zip(calibrated, calibrated[1:]):
            self.assertLessEqual(earlier, later)

    def test_pure_noise_does_not_beat_the_baseline(self):
        """
        A score with no information must not appear skilful.

        fit_calibration.py refuses to save in this case, which is what stops a
        constant being dressed up as a forecast.
        """
        rng = np.random.default_rng(9)
        scores = rng.uniform(40, 80, 4000)
        outcomes = (rng.random(4000) < 0.375).astype(int)  # independent of score

        calibrator = EnsembleCalibrator(self.path)
        report = calibrator.fit(scores, outcomes)

        self.assertFalse(report.improves_on_baseline)

    def test_output_stays_in_range(self):
        """Calibrated probabilities are always 0-100, including outside the fit range."""
        scores, outcomes, _ = self._overstated_sample()

        calibrator = EnsembleCalibrator(self.path)
        calibrator.fit(scores, outcomes)

        for raw in (-50, 0, 20, 50, 62, 100, 250):
            value = calibrator.calibrate(raw)
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 100.0)

    def test_round_trip_through_disk(self):
        """A saved calibration reloads to the same mapping."""
        scores, outcomes, _ = self._overstated_sample()

        calibrator = EnsembleCalibrator(self.path)
        calibrator.fit(scores, outcomes)
        calibrator.save()

        reloaded = EnsembleCalibrator(self.path)
        self.assertTrue(reloaded.is_fitted)

        # The knot table is stored rounded for readability, so agreement is to
        # within the storage precision rather than exact.
        for raw in (52, 58, 62, 68, 74):
            self.assertAlmostEqual(reloaded.calibrate(raw), calibrator.calibrate(raw),
                                   places=4)

    def test_too_few_samples_is_refused(self):
        """A calibration fitted on a handful of points would be noise."""
        calibrator = EnsembleCalibrator(self.path)
        with self.assertRaises(ValueError):
            calibrator.fit([60, 61, 62], [1, 0, 1])

    def test_single_class_is_refused(self):
        """Outcomes with no variation cannot calibrate anything."""
        calibrator = EnsembleCalibrator(self.path)
        with self.assertRaises(ValueError):
            calibrator.fit(np.linspace(50, 70, 400), np.ones(400))


if __name__ == '__main__':
    unittest.main()
