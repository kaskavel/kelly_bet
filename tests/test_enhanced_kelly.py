#!/usr/bin/env python3
"""
Tests for enhanced Kelly Criterion Calculator with detailed explanations
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
        # Load actual config
        config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.kelly = KellyCalculator(self.config)
    
    def test_detailed_calculation_fields(self):
        """Test that all detailed calculation fields are populated"""
        result = self.kelly.calculate_bet_size(
            probability=70.0,
            current_price=100.0,
            available_capital=1000.0
        )
        
        # Check that all new fields exist and have reasonable values
        self.assertIsNotNone(result.win_probability)
        self.assertIsNotNone(result.loss_probability)
        self.assertIsNotNone(result.win_amount_ratio)
        self.assertIsNotNone(result.loss_amount_ratio)
        self.assertIsNotNone(result.expected_win)
        self.assertIsNotNone(result.expected_loss)
        self.assertIsNotNone(result.kelly_formula_b)
        self.assertIsNotNone(result.kelly_formula_p)
        self.assertIsNotNone(result.kelly_formula_q)
        self.assertIsNotNone(result.available_capital)
        
        # Check probability calculations
        self.assertAlmostEqual(result.win_probability + result.loss_probability, 1.0, places=3)
        self.assertAlmostEqual(result.win_probability, 0.7, places=3)
        self.assertAlmostEqual(result.loss_probability, 0.3, places=3)
        
        # Check available capital matches
        self.assertEqual(result.available_capital, 1000.0)
    
    def test_expected_value_calculation(self):
        """Test expected value calculation accuracy"""
        result = self.kelly.calculate_bet_size(
            probability=60.0,  # 60% win
            current_price=100.0,
            available_capital=1000.0,
            win_threshold=5.0,   # 5% win
            loss_threshold=3.0   # 3% loss
        )
        
        # Manual EV calculation: (0.6 × 0.05) - (0.4 × 0.03) = 0.03 - 0.012 = 0.018
        expected_ev = (0.6 * 0.05) - (0.4 * 0.03)
        self.assertAlmostEqual(result.expected_value, expected_ev, places=3)
        
        # Check individual components
        self.assertAlmostEqual(result.expected_win, 0.6 * 0.05, places=3)
        self.assertAlmostEqual(result.expected_loss, 0.4 * 0.03, places=3)
    
    def test_kelly_formula_components(self):
        """Test Kelly formula component calculations"""
        result = self.kelly.calculate_bet_size(
            probability=60.0,
            current_price=100.0,
            available_capital=1000.0,
            win_threshold=5.0,   # 5% win
            loss_threshold=3.0   # 3% loss
        )
        
        # Kelly formula components
        # b = win_return / loss_risk = 0.05 / 0.03 = 1.667
        # p = 0.6, q = 0.4
        # f = (bp - q) / b = (1.667 * 0.6 - 0.4) / 1.667
        
        expected_b = 0.05 / 0.03
        self.assertAlmostEqual(result.kelly_formula_b, expected_b, places=3)
        self.assertAlmostEqual(result.kelly_formula_p, 0.6, places=3)
        self.assertAlmostEqual(result.kelly_formula_q, 0.4, places=3)
        
        # Check raw Kelly calculation
        expected_kelly = (expected_b * 0.6 - 0.4) / expected_b
        self.assertAlmostEqual(result.kelly_fraction_raw, expected_kelly, places=3)
    
    def test_real_world_example(self):
        """Test with a real-world example similar to your MU bet"""
        result = self.kelly.calculate_bet_size(
            probability=78.68,  # High confidence prediction
            current_price=118.71,
            available_capital=9000.0,  # $9k available
            win_threshold=5.0,   # 5% win target
            loss_threshold=3.0   # 3% stop loss
        )
        
        # Should be favorable with high probability
        self.assertTrue(result.is_favorable)
        self.assertGreater(result.expected_value, 0)
        self.assertGreater(result.recommended_amount, 0)
        
        # Check detailed calculations make sense
        self.assertAlmostEqual(result.win_probability, 0.7868, places=3)
        self.assertAlmostEqual(result.loss_probability, 0.2132, places=3)
        
        # Kelly b ratio should be 5/3 = 1.667
        self.assertAlmostEqual(result.kelly_formula_b, 5.0/3.0, places=3)


if __name__ == '__main__':
    unittest.main()