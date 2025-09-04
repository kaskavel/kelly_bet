#!/usr/bin/env python3
"""
Tests for Kelly Criterion Calculator
Ensures proper probability handling and warning logic.
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
        # Load actual config
        config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.kelly = KellyCalculator(self.config)
    
    def test_probability_unit_conversion(self):
        """Test that percentage config values are correctly converted to decimals"""
        # Config stores min_probability as percentage (55.0)
        self.assertEqual(self.config['trading']['min_probability'], 55.0)
        
        # Kelly calculator should convert to decimal (0.55)
        self.assertEqual(self.kelly.min_probability, 0.55)
    
    def test_high_probability_no_warning(self):
        """Test that high probability (78.7%) does not trigger low probability warning"""
        result = self.kelly.calculate_bet_size(
            probability=78.7,
            current_price=100.0,
            available_capital=1000.0
        )
        
        # Should not contain "Low win probability" warning
        if result.risk_warning:
            self.assertNotIn("Low win probability", result.risk_warning)
        
        # Should be favorable
        self.assertTrue(result.is_favorable)
        
        # Should have reasonable recommendation
        self.assertGreater(result.recommended_amount, 0)
    
    def test_low_probability_warning(self):
        """Test that genuinely low probability (45%) triggers warning"""
        result = self.kelly.calculate_bet_size(
            probability=45.0,  # Below 55% threshold
            current_price=100.0,
            available_capital=1000.0
        )
        
        # Should contain "Low win probability" warning
        self.assertIsNotNone(result.risk_warning)
        self.assertIn("Low win probability", result.risk_warning)
    
    def test_boundary_probability(self):
        """Test probability exactly at threshold (55%)"""
        result = self.kelly.calculate_bet_size(
            probability=55.0,  # Exactly at threshold
            current_price=100.0,
            available_capital=1000.0
        )
        
        # Should not trigger warning (55% = minimum, not below)
        if result.risk_warning:
            self.assertNotIn("Low win probability", result.risk_warning)
    
    def test_kelly_fraction_calculation(self):
        """Test Kelly fraction calculation with known values"""
        result = self.kelly.calculate_bet_size(
            probability=60.0,  # 60% win probability
            current_price=100.0,
            available_capital=1000.0,
            win_threshold=5.0,   # 5% win
            loss_threshold=3.0   # 3% loss
        )
        
        # Kelly formula: f = (bp - q) / b
        # where b = win_amount/loss_amount, p = 0.6, q = 0.4
        # b = 5/3 = 1.667, f = (1.667*0.6 - 0.4) / 1.667 = 0.36
        # But we apply multipliers and caps, so just check it's reasonable
        self.assertGreater(result.kelly_fraction_raw, 0)
        self.assertLess(result.fraction_of_capital, 1.0)
    
    def test_risk_warnings_comprehensive(self):
        """Test comprehensive risk warning scenarios"""
        # High probability, large Kelly fraction - should show fraction reduction
        result = self.kelly.calculate_bet_size(
            probability=80.0,  # Very high probability
            current_price=100.0,
            available_capital=1000.0
        )
        
        # Should show Kelly fraction reduction warning (capped for safety)
        if result.risk_warning:
            self.assertTrue(
                "Kelly fraction reduced" in result.risk_warning or
                "Large bet size" in result.risk_warning,
                f"Expected risk warnings, got: {result.risk_warning}"
            )


class TestKellyEdgeCases(unittest.TestCase):
    
    def setUp(self):
        """Set up test configuration"""
        # Create minimal test config
        self.test_config = {
            'trading': {
                'kelly_fraction': 0.25,
                'min_bet_amount': 100.0,
                'max_bet_amount': 10000.0,
                'max_bet_fraction': 0.1,
                'min_probability': 55.0,  # Test the conversion
                'max_loss_percentage': 5.0,
                'win_threshold': 5.0,
                'loss_threshold': 3.0
            }
        }
        
        self.kelly = KellyCalculator(self.test_config)
    
    def test_extreme_probabilities(self):
        """Test extreme probability values"""
        # Very low probability
        result_low = self.kelly.calculate_bet_size(
            probability=10.0,
            current_price=100.0,
            available_capital=1000.0
        )
        self.assertFalse(result_low.is_favorable)
        
        # Very high probability
        result_high = self.kelly.calculate_bet_size(
            probability=95.0,
            current_price=100.0,
            available_capital=1000.0
        )
        self.assertTrue(result_high.is_favorable)
        self.assertGreater(result_high.recommended_amount, 0)
    
    def test_zero_capital(self):
        """Test with zero available capital"""
        result = self.kelly.calculate_bet_size(
            probability=70.0,
            current_price=100.0,
            available_capital=0.0
        )
        
        self.assertEqual(result.recommended_amount, 0.0)
        self.assertEqual(result.fraction_of_capital, 0.0)


if __name__ == '__main__':
    unittest.main()