#!/usr/bin/env python3
"""
Tests for the Kelly Criterion Calculator's detailed calculation breakdown.

These tests pin the CAPPED-LOSS Kelly formula, net of fees:

    f* = p/l_net - q/w_net,   w_net = w - 2c,  l_net = l + 2c

The previous version of this file asserted f = (bp - q)/b, which is the Kelly for a
bet that loses its entire stake. This bet loses only the stop distance, so that form
understated the optimum by roughly 30x and ignored fees entirely.
"""

import unittest
import yaml
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kelly.calculator import KellyCalculator


class TestEnhancedKellyCalculator(unittest.TestCase):

    def setUp(self):
        """Set up test configuration"""
        config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.kelly = KellyCalculator(self.config)
        self.fee = self.config['trading']['trading_fee_percentage'] / 100.0

    def _net_legs(self, win_pct=5.0, loss_pct=3.0):
        """Fee-adjusted win and loss legs, as decimals."""
        round_trip = 2 * self.fee
        return win_pct / 100.0 - round_trip, loss_pct / 100.0 + round_trip

    def test_detailed_calculation_fields(self):
        """Test that all detailed calculation fields are populated"""
        result = self.kelly.calculate_bet_size(
            probability=70.0,
            current_price=100.0,
            available_capital=1000.0
        )

        for field in ('win_probability', 'loss_probability', 'win_amount_ratio',
                      'loss_amount_ratio', 'expected_win', 'expected_loss',
                      'kelly_formula_b', 'kelly_formula_p', 'kelly_formula_q',
                      'available_capital'):
            self.assertIsNotNone(getattr(result, field), f"{field} is None")

        self.assertAlmostEqual(result.win_probability + result.loss_probability, 1.0, places=3)
        self.assertAlmostEqual(result.win_probability, 0.7, places=3)
        self.assertAlmostEqual(result.loss_probability, 0.3, places=3)
        self.assertEqual(result.available_capital, 1000.0)

    def test_reported_legs_are_net_of_fees(self):
        """Win/loss ratios must be reported net of the round-trip fee, not gross."""
        w_net, l_net = self._net_legs()
        result = self.kelly.calculate_bet_size(
            probability=60.0, current_price=100.0, available_capital=1000.0,
            win_threshold=5.0, loss_threshold=3.0
        )

        self.assertAlmostEqual(result.win_amount_ratio, w_net, places=6)
        self.assertAlmostEqual(result.loss_amount_ratio, l_net, places=6)
        # Fees are real: they shrink the win leg and grow the loss leg.
        self.assertLess(result.win_amount_ratio, 0.05)
        self.assertGreater(result.loss_amount_ratio, 0.03)

    def test_expected_value_includes_fees(self):
        """EV must be computed on the net legs."""
        w_net, l_net = self._net_legs()
        result = self.kelly.calculate_bet_size(
            probability=60.0, current_price=100.0, available_capital=1000.0,
            win_threshold=5.0, loss_threshold=3.0
        )

        expected_ev = 0.6 * w_net - 0.4 * l_net
        self.assertAlmostEqual(result.expected_value, expected_ev, places=6)
        self.assertAlmostEqual(result.expected_win, 0.6 * w_net, places=6)
        self.assertAlmostEqual(result.expected_loss, 0.4 * l_net, places=6)

        # The pre-fee EV would have been 0.018; fees take a real bite out of it.
        self.assertLess(result.expected_value, 0.6 * 0.05 - 0.4 * 0.03)

    def test_capped_loss_kelly_formula(self):
        """f* = p/l_net - q/w_net for a bet that risks only the stop distance."""
        w_net, l_net = self._net_legs()
        result = self.kelly.calculate_bet_size(
            probability=60.0, current_price=100.0, available_capital=1000.0,
            win_threshold=5.0, loss_threshold=3.0
        )

        self.assertAlmostEqual(result.kelly_formula_p, 0.6, places=3)
        self.assertAlmostEqual(result.kelly_formula_q, 0.4, places=3)
        self.assertAlmostEqual(result.kelly_formula_b, w_net / l_net, places=6)

        expected_kelly = (0.6 / l_net) - (0.4 / w_net)
        self.assertAlmostEqual(result.kelly_fraction_raw, expected_kelly, places=6)

        # Sanity: capped-loss Kelly is far above 1.0 here, which is why the
        # multiplier and caps are the real risk controls.
        self.assertGreater(result.kelly_fraction_raw, 1.0)

    def test_break_even_probability_is_not_fifty_percent(self):
        """
        The neutral point for a barrier bet is l/(w+l), fee-adjusted.

        For 5%/3% at 0.25% per side that is 43.75%, not 50%.
        """
        self.assertAlmostEqual(self.kelly.break_even_probability, 0.4375, places=4)

        # Without fees it would be exactly 3/(5+3).
        zero_fee_config = {'trading': dict(self.config['trading'], trading_fee_percentage=0.0)}
        zero_fee = KellyCalculator(zero_fee_config)
        self.assertAlmostEqual(zero_fee.break_even_probability, 0.375, places=6)

    def test_expected_value_sign_flips_at_break_even(self):
        """EV must be negative below break-even and positive above it."""
        be = self.kelly.break_even_probability * 100.0

        below = self.kelly.calculate_bet_size(
            probability=be - 2.0, current_price=100.0, available_capital=100000.0)
        at = self.kelly.calculate_bet_size(
            probability=be, current_price=100.0, available_capital=100000.0)
        above = self.kelly.calculate_bet_size(
            probability=be + 2.0, current_price=100.0, available_capital=100000.0)

        self.assertLess(below.expected_value, 0)
        self.assertFalse(below.is_favorable)
        self.assertAlmostEqual(at.expected_value, 0.0, places=9)
        self.assertFalse(at.is_favorable)
        self.assertGreater(above.expected_value, 0)

    def test_random_walk_baseline_is_rejected(self):
        """
        A driftless asset touches +5% before -3% with probability 3/(5+3) = 37.5%.

        That is a zero-EV bet before fees and negative after, so it must be refused.
        """
        result = self.kelly.calculate_bet_size(
            probability=37.5, current_price=100.0, available_capital=100000.0)
        self.assertFalse(result.is_favorable)
        self.assertLess(result.expected_value, 0)
        self.assertEqual(result.recommended_amount, 0.0)

    def test_real_world_example(self):
        """A high-confidence bet sizes up but stays inside the caps."""
        result = self.kelly.calculate_bet_size(
            probability=78.68,
            current_price=118.71,
            available_capital=9000.0,
            win_threshold=5.0,
            loss_threshold=3.0
        )

        self.assertTrue(result.is_favorable)
        self.assertGreater(result.expected_value, 0)
        self.assertGreater(result.recommended_amount, 0)

        self.assertAlmostEqual(result.win_probability, 0.7868, places=3)
        self.assertAlmostEqual(result.loss_probability, 0.2132, places=3)

        max_fraction = self.config['trading']['max_bet_fraction']
        self.assertLessEqual(result.fraction_of_capital, max_fraction + 1e-9)


if __name__ == '__main__':
    unittest.main()
