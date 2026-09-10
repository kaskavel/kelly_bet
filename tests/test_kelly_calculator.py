#!/usr/bin/env python3
"""
Tests for Kelly Criterion Calculator: thresholds, sizing chain, and warnings.
"""

import unittest
import yaml
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kelly.calculator import KellyCalculator


class TestKellyCalculator(unittest.TestCase):

    def setUp(self):
        """Set up test configuration"""
        config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.kelly = KellyCalculator(self.config)
        self.min_prob_pct = self.config['trading']['min_probability']

    def test_probability_unit_conversion(self):
        """Percentage config values are converted to decimals"""
        self.assertEqual(self.kelly.min_probability, self.min_prob_pct / 100.0)

    def test_high_probability_no_warning(self):
        """A probability well above the threshold does not warn about being low"""
        result = self.kelly.calculate_bet_size(
            probability=88.0,
            current_price=100.0,
            available_capital=10000.0
        )

        if result.risk_warning:
            self.assertNotIn("Low win probability", result.risk_warning)

        self.assertTrue(result.is_favorable)
        self.assertGreater(result.recommended_amount, 0)

    def test_low_probability_warning(self):
        """A probability below the configured threshold warns"""
        result = self.kelly.calculate_bet_size(
            probability=self.min_prob_pct - 10.0,
            current_price=100.0,
            available_capital=10000.0
        )

        self.assertIsNotNone(result.risk_warning)
        self.assertIn("Low win probability", result.risk_warning)

    def test_boundary_probability(self):
        """Probability exactly at the threshold is not flagged as low"""
        result = self.kelly.calculate_bet_size(
            probability=self.min_prob_pct,
            current_price=100.0,
            available_capital=10000.0
        )

        if result.risk_warning:
            self.assertNotIn("Low win probability", result.risk_warning)

    def test_thin_margin_warning(self):
        """A bet barely above break-even is flagged, even if EV is positive"""
        be = self.kelly.break_even_probability * 100.0
        result = self.kelly.calculate_bet_size(
            probability=be + 1.0,
            current_price=100.0,
            available_capital=100000.0
        )

        self.assertIsNotNone(result.risk_warning)
        self.assertIn("Thin margin over break-even", result.risk_warning)

    def test_position_size_increases_with_edge(self):
        """
        Size must be monotonically increasing in probability until a cap binds.

        The old sizing chain pinned ~half of all bets at max_bet_fraction, making the
        Kelly derivation decorative.
        """
        capital = 100000.0
        fractions = []
        for p in (50.0, 55.0, 60.0, 66.0, 72.0):
            r = self.kelly.calculate_bet_size(
                probability=p, current_price=100.0, available_capital=capital)
            fractions.append(r.fraction_of_capital)

        for earlier, later in zip(fractions, fractions[1:]):
            self.assertLess(earlier, later,
                            f"size did not grow with edge: {fractions}")

        # And it stays inside the hard cap throughout.
        self.assertLessEqual(max(fractions), self.config['trading']['max_bet_fraction'] + 1e-9)

    def test_caps_are_respected(self):
        """Neither the position cap nor the risk budget may be exceeded"""
        max_fraction = self.config['trading']['max_bet_fraction']
        max_risk = self.config['trading']['max_risk_fraction']

        result = self.kelly.calculate_bet_size(
            probability=99.0, current_price=100.0, available_capital=100000.0)

        self.assertLessEqual(result.fraction_of_capital, max_fraction + 1e-9)
        capital_at_risk = result.fraction_of_capital * result.loss_amount_ratio
        self.assertLessEqual(capital_at_risk, max_risk + 1e-9)


class TestKellyEdgeCases(unittest.TestCase):

    def setUp(self):
        """Set up minimal test configuration"""
        self.test_config = {
            'trading': {
                'kelly_fraction': 0.005,
                'min_bet_amount': 100.0,
                'max_bet_amount': 10000.0,
                'max_bet_fraction': 0.1,
                'max_risk_fraction': 0.0035,
                'min_probability': 66.0,
                'max_loss_percentage': 5.0,
                'win_threshold': 5.0,
                'loss_threshold': 3.0,
                'trading_fee_percentage': 0.25,
            }
        }

        self.kelly = KellyCalculator(self.test_config)

    def test_extreme_probabilities(self):
        """Extreme probabilities resolve sensibly at both ends"""
        result_low = self.kelly.calculate_bet_size(
            probability=10.0, current_price=100.0, available_capital=10000.0)
        self.assertFalse(result_low.is_favorable)
        self.assertEqual(result_low.recommended_amount, 0.0)

        result_high = self.kelly.calculate_bet_size(
            probability=95.0, current_price=100.0, available_capital=10000.0)
        self.assertTrue(result_high.is_favorable)
        self.assertGreater(result_high.recommended_amount, 0)

    def test_zero_capital(self):
        """Zero capital yields no bet"""
        result = self.kelly.calculate_bet_size(
            probability=70.0, current_price=100.0, available_capital=0.0)

        self.assertEqual(result.recommended_amount, 0.0)
        self.assertEqual(result.fraction_of_capital, 0.0)

    def test_minimum_bet_is_a_skip_threshold_not_a_floor(self):
        """
        When Kelly sizes below min_bet_amount the bet is SKIPPED.

        Previously `max(min_bet_amount, ...)` raised the stake up to the minimum,
        which breached max_bet_fraction precisely when capital was lowest.
        """
        # $900 capital at p=50% sizes to ~1.6% = ~$14, well under the $100 minimum.
        result = self.kelly.calculate_bet_size(
            probability=50.0, current_price=100.0, available_capital=900.0)

        self.assertEqual(result.recommended_amount, 0.0)
        self.assertFalse(result.is_favorable)
        self.assertIn("below minimum", result.risk_warning)

    def test_min_bet_never_breaches_max_fraction(self):
        """Across a range of small balances, size never exceeds the position cap"""
        max_fraction = self.test_config['trading']['max_bet_fraction']

        for capital in (150.0, 300.0, 500.0, 900.0, 1500.0, 3000.0):
            result = self.kelly.calculate_bet_size(
                probability=80.0, current_price=100.0, available_capital=capital)
            self.assertLessEqual(
                result.fraction_of_capital, max_fraction + 1e-9,
                f"capital={capital} produced {result.fraction_of_capital:.3%}")

    def test_fees_can_make_a_bet_impossible(self):
        """If the round-trip fee exceeds the profit target, no bet is favorable"""
        config = {'trading': dict(self.test_config['trading'],
                                  win_threshold=0.4, trading_fee_percentage=0.25)}
        kelly = KellyCalculator(config)

        result = kelly.calculate_bet_size(
            probability=95.0, current_price=100.0, available_capital=100000.0)

        self.assertFalse(result.is_favorable)
        self.assertEqual(result.recommended_amount, 0.0)

    def test_validate_uses_break_even_not_fifty(self):
        """Parameter validation rejects against break-even, not a hardcoded 50%"""
        ok, msg = self.kelly.validate_bet_parameters(
            probability=46.0, win_threshold=5.0, loss_threshold=3.0,
            available_capital=10000.0)
        self.assertTrue(ok, msg)

        bad, msg = self.kelly.validate_bet_parameters(
            probability=40.0, win_threshold=5.0, loss_threshold=3.0,
            available_capital=10000.0)
        self.assertFalse(bad)
        self.assertIn("break-even", msg)


if __name__ == '__main__':
    unittest.main()


class TestCorrelationHaircut(unittest.TestCase):
    """
    Kelly sizes a bet as if it were the only position at risk. With up to 30
    concurrent long positions that badly over-levers the book: 30 correlated longs
    behave much like one leveraged index bet.
    """

    def setUp(self):
        config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.kelly = KellyCalculator(self.config)

    def test_single_position_is_unhaircut(self):
        self.assertEqual(self.kelly.correlation_haircut(1), 1.0)

    def test_haircut_matches_the_effective_bet_count(self):
        """1/(1+(n-1)*rho), i.e. n divided by the effective number of independent bets."""
        rho = self.kelly.assumed_correlation
        for n in (2, 5, 10, 30):
            self.assertAlmostEqual(self.kelly.correlation_haircut(n),
                                   1.0 / (1.0 + (n - 1) * rho), places=9)

    def test_size_shrinks_as_the_book_grows(self):
        fractions = []
        for n in (1, 2, 5, 10, 30):
            result = self.kelly.calculate_bet_size(
                probability=66.0, current_price=100.0, available_capital=100000.0,
                concurrent_positions=n)
            fractions.append(result.fraction_of_capital)

        for larger, smaller in zip(fractions, fractions[1:]):
            self.assertLess(smaller, larger,
                            f"size did not shrink with book size: {fractions}")

    def test_total_book_risk_stays_bounded(self):
        """
        The point of the haircut: n positions together should risk about what one
        un-haircut position risks, not n times as much.
        """
        solo = self.kelly.calculate_bet_size(
            probability=66.0, current_price=100.0, available_capital=100000.0,
            concurrent_positions=1)

        n = 10
        each = self.kelly.calculate_bet_size(
            probability=66.0, current_price=100.0, available_capital=100000.0,
            concurrent_positions=n)

        # Naive sizing would deploy n x solo. With the haircut the whole book stays
        # within a small multiple of a single standalone position.
        book = n * each.fraction_of_capital
        self.assertLess(book, 3 * solo.fraction_of_capital)
        self.assertLess(book, 10 * each.fraction_of_capital + 1e-9)

    def test_haircut_is_reported(self):
        result = self.kelly.calculate_bet_size(
            probability=66.0, current_price=100.0, available_capital=100000.0,
            concurrent_positions=10)
        self.assertIsNotNone(result.risk_warning)
        self.assertIn("Correlation haircut", result.risk_warning)

    def test_zero_correlation_disables_the_haircut(self):
        config = {'trading': dict(self.config['trading']),
                  'risk': dict(self.config.get('risk', {}), assumed_correlation=0.0)}
        kelly = KellyCalculator(config)
        self.assertEqual(kelly.correlation_haircut(30), 1.0)
