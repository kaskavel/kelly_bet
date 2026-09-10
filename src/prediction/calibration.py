"""
Ensemble probability calibration.

The system's central defect was that its "probability" was not one. Over 84 closed
bets the reported mean probability at placement was 63.7% and the realised win rate
was 44.1% -- a 20-point overstatement, rejected at z = -3.75. The ranking carried
signal (realised win rate rose monotonically across score bands: 27.3%, 41.7%,
45.0%, 56.7%) but the LEVEL was meaningless, and it was the level that Kelly sizing
and every threshold consumed.

This module fits a monotonic map

    raw ensemble score  ->  P(win barrier touched first)

using isotonic regression against realised first-touch barrier outcomes. Isotonic is
the right choice because it preserves the ordering the ensemble gets right while
discarding the scale it gets wrong, and it needs no parametric assumption about the
shape of the miscalibration.

Fit it with `python scripts/fit_calibration.py`. Until a calibrator is fitted the
system reports raw scores and says so, rather than quietly presenting them as
probabilities.
"""

import json
import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

CALIBRATION_PATH = Path('models/calibration/ensemble_calibration.json')


@dataclass
class CalibrationReport:
    """How good the fitted map is, out of sample."""
    n_samples: int
    base_rate: float
    brier_raw: float
    brier_calibrated: float
    brier_baseline: float
    bands: List[Dict] = field(default_factory=list)
    fitted_at: str = ""

    @property
    def improves_on_baseline(self) -> bool:
        """Does the calibrated model beat simply predicting the base rate?"""
        return self.brier_calibrated < self.brier_baseline

    def summary(self) -> str:
        verdict = "BETTER than base rate" if self.improves_on_baseline else "NO BETTER than base rate"
        lines = [
            f"Calibration fitted on {self.n_samples} samples "
            f"(base rate {self.base_rate:.1%})",
            f"  Brier  raw={self.brier_raw:.4f}  "
            f"calibrated={self.brier_calibrated:.4f}  "
            f"base-rate baseline={self.brier_baseline:.4f}  -> {verdict}",
        ]
        for band in self.bands:
            lines.append(
                f"    raw {band['lo']:.0f}-{band['hi']:.0f}%: n={band['n']:6d}  "
                f"observed {band['observed'] * 100:5.1f}%  "
                f"calibrated {band['calibrated'] * 100:5.1f}%"
            )
        return "\n".join(lines)


class EnsembleCalibrator:
    """
    Monotonic map from raw ensemble score (0-100) to calibrated probability (0-100).

    Loads a fitted map from disk when available. `is_fitted` is False otherwise, and
    callers must treat scores as scores.
    """

    def __init__(self, path: Path = CALIBRATION_PATH):
        self.path = Path(path)
        self._x: Optional[np.ndarray] = None   # raw score knots, ascending
        self._y: Optional[np.ndarray] = None   # calibrated probability knots
        self.report: Optional[Dict] = None
        self.load()

    @property
    def is_fitted(self) -> bool:
        return self._x is not None and self._y is not None and len(self._x) >= 2

    def calibrate(self, raw_score: float) -> float:
        """
        Map a raw ensemble score to a calibrated probability, both on 0-100.

        Returns the raw score unchanged when no calibrator is fitted, so behaviour
        degrades to the status quo rather than to silence.
        """
        if raw_score is None:
            return None
        if not self.is_fitted:
            return raw_score

        # np.interp clamps outside the knot range, which is what we want: the fit
        # says nothing about scores never observed in training.
        return float(np.interp(float(raw_score), self._x, self._y))

    def fit(self, raw_scores, outcomes, n_bands: int = 8) -> CalibrationReport:
        """
        Fit the isotonic map.

        Args:
            raw_scores: iterable of raw ensemble scores (0-100)
            outcomes: iterable of 1 (win barrier first) / 0 (loss barrier or timeout)
        """
        from sklearn.isotonic import IsotonicRegression
        from sklearn.metrics import brier_score_loss

        x = np.asarray(list(raw_scores), dtype=float)
        y = np.asarray(list(outcomes), dtype=float)

        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]

        if len(x) < 200:
            raise ValueError(f"Need at least 200 labelled samples to calibrate, got {len(x)}")
        if len(np.unique(y)) < 2:
            raise ValueError("Outcomes are single-class; cannot calibrate")

        # Time-ordered holdout: the caller passes samples in chronological order.
        split = int(len(x) * 0.75)
        x_train, y_train = x[:split], y[:split]
        x_test, y_test = x[split:], y[split:]

        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
        iso.fit(x_train, y_train)

        # Store as a knot table so the fitted map is inspectable and portable.
        knots = np.linspace(x.min(), x.max(), 101)
        self._x = knots
        self._y = np.clip(iso.predict(knots), 0.0, 1.0) * 100.0

        base_rate = float(y_train.mean())
        calibrated_test = np.array([self.calibrate(v) / 100.0 for v in x_test])

        report = CalibrationReport(
            n_samples=int(len(x)),
            base_rate=base_rate,
            # The raw score interpreted as a probability -- what the system did before.
            brier_raw=float(brier_score_loss(y_test, np.clip(x_test / 100.0, 0, 1))),
            brier_calibrated=float(brier_score_loss(y_test, calibrated_test)),
            brier_baseline=float(brier_score_loss(y_test, np.full(len(y_test), base_rate))),
            fitted_at=datetime.now().isoformat(),
        )

        for lo, hi in self._band_edges(x_test, n_bands):
            mask = (x_test >= lo) & (x_test < hi)
            if mask.sum() >= 10:
                report.bands.append({
                    'lo': float(lo),
                    'hi': float(hi),
                    'n': int(mask.sum()),
                    'observed': float(y_test[mask].mean()),
                    'calibrated': float(calibrated_test[mask].mean()),
                })

        self.report = report.__dict__
        return report

    @staticmethod
    def _band_edges(values: np.ndarray, n_bands: int):
        """Equal-count bands over the observed score range."""
        quantiles = np.quantile(values, np.linspace(0, 1, n_bands + 1))
        edges = np.unique(np.round(quantiles, 4))
        return list(zip(edges[:-1], edges[1:]))

    def save(self):
        """Persist the fitted map and its report."""
        if not self.is_fitted:
            raise ValueError("Nothing to save: calibrator is not fitted")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'schema': 1,
            'raw_scores': [round(v, 6) for v in self._x.tolist()],
            'calibrated': [round(v, 6) for v in self._y.tolist()],
            'report': self.report,
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        logger.info(f"Saved ensemble calibration to {self.path}")

    def load(self) -> bool:
        """Load a fitted map from disk, if one exists."""
        try:
            if not self.path.exists():
                return False

            payload = json.loads(self.path.read_text(encoding='utf-8'))
            self._x = np.asarray(payload['raw_scores'], dtype=float)
            self._y = np.asarray(payload['calibrated'], dtype=float)
            self.report = payload.get('report')
            logger.info(f"Loaded ensemble calibration from {self.path}")
            return True

        except Exception as e:
            logger.error(f"Could not load calibration from {self.path}: {e}")
            self._x = self._y = None
            return False
