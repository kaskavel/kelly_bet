#!/usr/bin/env python3
"""
Fit the ensemble calibration map from backtest output.

Run `scripts/backtest.py` first: it writes data/backtest_scores.csv, one row per
out-of-sample (symbol, bar) with the raw ensemble score and the realised first-touch
barrier outcome. This script fits a monotonic map from score to observed win
frequency, so that the number the system calls a probability is one.

    python scripts/backtest.py
    python scripts/fit_calibration.py

The fit is only saved if it beats simply predicting the base rate. A calibration that
does not is worse than none, because it would dress up a constant as a forecast.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.kelly.calculator import KellyCalculator
from src.prediction.calibration import EnsembleCalibrator

logger = logging.getLogger('fit_calibration')


def main():
    parser = argparse.ArgumentParser(description='Fit ensemble probability calibration')
    parser.add_argument('--scores', default='data/backtest_scores.csv',
                        help='Backtest output (default: data/backtest_scores.csv)')
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--force', action='store_true',
                        help='Save even if the fit does not beat the base rate')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    scores_path = Path(args.scores)
    if not scores_path.exists():
        logger.error(f"{scores_path} not found. Run: python scripts/backtest.py")
        return 1

    frame = pd.read_csv(scores_path)
    for column in ('raw_score', 'outcome'):
        if column not in frame.columns:
            logger.error(f"{scores_path} is missing the '{column}' column")
            return 1

    # Preserve chronological order: the calibrator's holdout is time-ordered.
    if 'timestamp' in frame.columns:
        frame['timestamp'] = pd.to_datetime(frame['timestamp'], utc=True, errors='coerce')
        frame = frame.sort_values('timestamp')

    frame = frame.dropna(subset=['raw_score', 'outcome'])

    calibrator = EnsembleCalibrator()
    try:
        report = calibrator.fit(frame['raw_score'], frame['outcome'])
    except ValueError as e:
        logger.error(str(e))
        return 1

    print()
    print(report.summary())
    print()

    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    kelly = KellyCalculator(config)
    break_even = kelly.break_even_probability * 100.0

    print(f"Fee-adjusted break-even probability: {break_even:.2f}%")
    print("Calibrated probability by raw score:")
    for raw in (40, 45, 50, 55, 60, 65, 70, 75, 80):
        calibrated = calibrator.calibrate(raw)
        verdict = "TRADE" if calibrated > break_even else "skip"
        print(f"  raw {raw:3d}%  ->  calibrated {calibrated:6.2f}%   {verdict}")

    # The raw score that first clears break-even: the threshold to configure.
    crossing = None
    for raw in [x / 10 for x in range(0, 1001)]:
        if calibrator.calibrate(raw) > break_even:
            crossing = raw
            break

    print()
    if crossing is None:
        print("NO raw score maps to a calibrated probability above break-even.")
        print("On this evidence the strategy has no tradeable edge at these barriers.")
        print("Consider different barriers, a longer time barrier, or better features")
        print("before trading it.")
    else:
        print(f"Raw scores above {crossing:.1f} map above break-even.")
        print(f"Set trading.min_probability accordingly (with a margin), and note that")
        print(f"min_probability is compared against the CALIBRATED probability once a")
        print(f"calibration is fitted.")

    if not report.improves_on_baseline and not args.force:
        print()
        print("NOT SAVED: the calibrated model does not beat predicting the base rate,")
        print("so the ensemble has no measurable skill on this sample. Saving it would")
        print("present a constant as a forecast. Re-run with --force to override.")
        return 2

    calibrator.save()
    print()
    print(f"Saved calibration to {calibrator.path}")
    print("The system will now report calibrated probabilities.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
