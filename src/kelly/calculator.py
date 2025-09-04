"""
Kelly Criterion Calculator
Calculates optimal bet sizes using the Kelly Criterion formula.
"""

import logging
import math
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class BetParameters:
    """Structure for bet parameters"""
    probability_win: float      # Probability of winning (0.0 to 1.0)
    win_percentage: float       # Winning return percentage (e.g., 5.0 for 5%)
    loss_percentage: float      # Loss percentage (e.g., 3.0 for 3%)
    current_price: float        # Current asset price
    available_capital: float    # Available capital for betting


@dataclass
class BetRecommendation:
    """Structure for Kelly bet recommendation"""
    recommended_amount: float   # Dollar amount to bet
    fraction_of_capital: float  # Fraction of capital (Kelly fraction)
    expected_value: float       # Expected value of the bet
    is_favorable: bool         # Whether bet has positive expected value
    kelly_fraction_raw: float  # Raw Kelly fraction before adjustments
    confidence_level: str      # High/Medium/Low confidence
    risk_warning: Optional[str] # Any risk warnings


class KellyCalculator:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Kelly parameters from config
        self.kelly_fraction_multiplier = config.get('trading', {}).get('kelly_fraction', 0.25)
        self.min_bet_amount = config.get('trading', {}).get('min_bet_amount', 100.0)
        self.max_bet_amount = config.get('trading', {}).get('max_bet_amount', 10000.0)
        self.max_bet_fraction = config.get('trading', {}).get('max_bet_fraction', 0.1)  # 10% max
        
        # Risk management
        self.min_probability = config.get('trading', {}).get('min_probability', 0.55)  # 55% minimum
        self.max_loss_percentage = config.get('trading', {}).get('max_loss_percentage', 5.0)  # 5% max loss
        
    def calculate_bet_size(self, 
                          probability: float, 
                          current_price: float, 
                          available_capital: float,
                          win_threshold: Optional[float] = None,
                          loss_threshold: Optional[float] = None) -> BetRecommendation:
        """
        Calculate optimal bet size using Kelly Criterion
        
        Args:
            probability: Win probability as percentage (0-100)
            current_price: Current asset price
            available_capital: Available capital for betting
            win_threshold: Win threshold percentage (default from config)
            loss_threshold: Loss threshold percentage (default from config)
            
        Returns:
            BetRecommendation with all bet details
        """
        # Use defaults from config if not provided
        if win_threshold is None:
            win_threshold = self.config.get('trading', {}).get('win_threshold', 5.0)
        if loss_threshold is None:
            loss_threshold = self.config.get('trading', {}).get('loss_threshold', 3.0)
        
        # Create bet parameters
        bet_params = BetParameters(
            probability_win=probability / 100.0,  # Convert to decimal
            win_percentage=win_threshold,
            loss_percentage=loss_threshold,
            current_price=current_price,
            available_capital=available_capital
        )
        
        self.logger.debug(f"Calculating Kelly bet size: prob={probability:.1f}%, "
                         f"win={win_threshold:.1f}%, loss={loss_threshold:.1f}%, "
                         f"capital=${available_capital:.2f}")
        
        # Calculate Kelly fraction
        return self._calculate_kelly_bet(bet_params)
    
    def _calculate_kelly_bet(self, params: BetParameters) -> BetRecommendation:
        """Calculate Kelly bet using the classic formula"""
        
        # Kelly Criterion formula: f = (bp - q) / b
        # Where:
        # f = fraction of capital to wager
        # b = odds received (net odds = win_return / loss_risk)
        # p = probability of winning
        # q = probability of losing (1 - p)
        
        p = params.probability_win
        q = 1 - p
        
        # Calculate odds ratio
        # If we win: gain win_percentage
        # If we lose: lose loss_percentage
        win_return = params.win_percentage / 100.0  # Convert to decimal
        loss_risk = params.loss_percentage / 100.0   # Convert to decimal
        
        # Net odds received = win_return / loss_risk
        b = win_return / loss_risk
        
        # Kelly fraction: f = (bp - q) / b = p - q/b
        kelly_fraction_raw = (b * p - q) / b
        
        # Alternative formulation: f = p - q/b
        # kelly_fraction_raw = p - (q / b)
        
        self.logger.debug(f"Kelly calculation: p={p:.3f}, q={q:.3f}, b={b:.3f}, "
                         f"raw_kelly={kelly_fraction_raw:.3f}")
        
        # Check if bet is favorable (positive expected value)
        expected_value = p * win_return - q * loss_risk
        is_favorable = expected_value > 0 and kelly_fraction_raw > 0
        
        if not is_favorable:
            return BetRecommendation(
                recommended_amount=0.0,
                fraction_of_capital=0.0,
                expected_value=expected_value,
                is_favorable=False,
                kelly_fraction_raw=kelly_fraction_raw,
                confidence_level="N/A",
                risk_warning="Negative expected value - no bet recommended"
            )
        
        # Apply Kelly fraction multiplier (conservative approach)
        adjusted_kelly_fraction = kelly_fraction_raw * self.kelly_fraction_multiplier
        
        # Apply maximum bet fraction limit
        final_kelly_fraction = min(adjusted_kelly_fraction, self.max_bet_fraction)
        
        # Calculate dollar amount
        raw_bet_amount = final_kelly_fraction * params.available_capital
        
        # Apply bet size limits
        recommended_amount = max(self.min_bet_amount, 
                               min(raw_bet_amount, self.max_bet_amount))
        
        # Ensure we don't exceed available capital
        recommended_amount = min(recommended_amount, params.available_capital)
        
        # Recalculate actual fraction used
        actual_fraction = recommended_amount / params.available_capital if params.available_capital > 0 else 0
        
        # Determine confidence level
        confidence_level = self._determine_confidence_level(params.probability_win, kelly_fraction_raw)
        
        # Generate risk warnings
        risk_warning = self._generate_risk_warnings(params, kelly_fraction_raw, adjusted_kelly_fraction)
        
        recommendation = BetRecommendation(
            recommended_amount=recommended_amount,
            fraction_of_capital=actual_fraction,
            expected_value=expected_value,
            is_favorable=is_favorable,
            kelly_fraction_raw=kelly_fraction_raw,
            confidence_level=confidence_level,
            risk_warning=risk_warning
        )
        
        self.logger.info(f"Kelly recommendation: ${recommended_amount:.2f} "
                        f"({actual_fraction:.1%} of capital), EV={expected_value:.3f}")
        
        return recommendation
    
    def _determine_confidence_level(self, probability: float, kelly_fraction: float) -> str:
        """Determine confidence level based on probability and Kelly fraction"""
        if probability >= 0.75 and kelly_fraction >= 0.2:
            return "High"
        elif probability >= 0.65 and kelly_fraction >= 0.1:
            return "Medium"
        elif probability >= 0.55 and kelly_fraction >= 0.05:
            return "Low"
        else:
            return "Very Low"
    
    def _generate_risk_warnings(self, params: BetParameters, 
                               raw_kelly: float, adjusted_kelly: float) -> Optional[str]:
        """Generate risk warnings for the bet"""
        warnings = []
        
        # Check probability threshold
        if params.probability_win < self.min_probability:
            warnings.append(f"Low win probability ({params.probability_win:.1%})")
        
        # Check if Kelly fraction was significantly reduced
        if adjusted_kelly < raw_kelly * 0.5:
            warnings.append(f"Kelly fraction reduced from {raw_kelly:.1%} to {adjusted_kelly:.1%}")
        
        # Check loss threshold
        if params.loss_percentage > self.max_loss_percentage:
            warnings.append(f"High loss risk ({params.loss_percentage:.1%})")
        
        # Check if betting large fraction of capital
        fraction_of_capital = adjusted_kelly
        if fraction_of_capital > 0.05:  # More than 5%
            warnings.append(f"Large bet size ({fraction_of_capital:.1%} of capital)")
        
        return "; ".join(warnings) if warnings else None
    
    def validate_bet_parameters(self, 
                               probability: float,
                               win_threshold: float,
                               loss_threshold: float,
                               available_capital: float) -> Tuple[bool, str]:
        """
        Validate bet parameters before calculation
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not (0 <= probability <= 100):
            return False, f"Invalid probability: {probability}% (must be 0-100)"
        
        if win_threshold <= 0:
            return False, f"Invalid win threshold: {win_threshold}% (must be positive)"
        
        if loss_threshold <= 0:
            return False, f"Invalid loss threshold: {loss_threshold}% (must be positive)"
        
        if available_capital <= 0:
            return False, f"Invalid available capital: ${available_capital} (must be positive)"
        
        if available_capital < self.min_bet_amount:
            return False, f"Insufficient capital: ${available_capital} < ${self.min_bet_amount} minimum"
        
        if probability < 50:
            return False, f"Probability too low: {probability}% (below 50% not recommended)"
        
        return True, "Valid"
    
    def get_kelly_info(self) -> Dict:
        """Get information about Kelly calculator configuration"""
        return {
            'kelly_fraction_multiplier': self.kelly_fraction_multiplier,
            'min_bet_amount': self.min_bet_amount,
            'max_bet_amount': self.max_bet_amount,
            'max_bet_fraction': self.max_bet_fraction,
            'min_probability': self.min_probability,
            'max_loss_percentage': self.max_loss_percentage
        }