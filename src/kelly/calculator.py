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
    is_favorable: bool         # Whether bet has positive equivalent
    kelly_fraction_raw: float  # Raw Kelly fraction before adjustments
    confidence_level: str      # High/Medium/Low confidence
    risk_warning: Optional[str] # Any risk warnings
    
    # Detailed calculation breakdown
    win_probability: float     # Win probability (decimal)
    loss_probability: float    # Loss probability (decimal)
    win_amount_ratio: float    # Win amount / bet amount ratio (b in Kelly formula)
    loss_amount_ratio: float   # Loss amount / bet amount ratio
    expected_win: float        # Expected win amount
    expected_loss: float       # Expected loss amount
    kelly_formula_b: float     # The 'b' parameter in Kelly formula
    kelly_formula_p: float     # The 'p' parameter in Kelly formula 
    kelly_formula_q: float     # The 'q' parameter in Kelly formula
    available_capital: float   # Available capital for betting


class KellyCalculator:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Kelly parameters from config
        self.kelly_fraction_multiplier = config.get('trading', {}).get('kelly_fraction', 0.25)
        self.min_bet_amount = config.get('trading', {}).get('min_bet_amount', 100.0)
        self.max_bet_amount = config.get('trading', {}).get('max_bet_amount', 10000.0)
        self.max_bet_fraction = config.get('trading', {}).get('max_bet_fraction', 0.1)  # 10% max
        # Risk budget: cap capital-at-risk per bet (fraction * net loss). The corrected
        # capped-loss Kelly formula is unbounded (it can exceed 1.0), so this and
        # max_bet_fraction are the constraints that actually govern position size.
        self.max_risk_fraction = config.get('trading', {}).get('max_risk_fraction', 0.003)
        # Fee per side, as a decimal. Charged on entry and on exit.
        self.fee_rate = config.get('trading', {}).get('trading_fee_percentage', 0.25) / 100.0
        # Assumed average pairwise correlation between concurrent positions, used to
        # haircut a standalone Kelly fraction. See correlation_haircut().
        self.assumed_correlation = float(
            config.get('risk', {}).get('assumed_correlation', 0.30))

        # Risk management
        min_prob_config = config.get('trading', {}).get('min_probability', 55.0)  # 55% minimum
        # Convert percentage to decimal if needed
        self.min_probability = min_prob_config / 100.0 if min_prob_config > 1.0 else min_prob_config
        self.max_loss_percentage = config.get('trading', {}).get('max_loss_percentage', 5.0)  # 5% max loss

        # Fee-adjusted break-even probability for the configured payoff. Recomputed on
        # every calculation; seeded here so it is always defined.
        self.break_even_probability = self._break_even_probability(
            config.get('trading', {}).get('win_threshold', 5.0),
            config.get('trading', {}).get('loss_threshold', 3.0),
        )

    def _break_even_probability(self, win_threshold: float, loss_threshold: float) -> float:
        """
        The win probability at which this bet has exactly zero expected value.

        For a barrier bet this - not 50% - is the neutral point. With no fees it is
        l/(w+l), which for a 5%/3% bet is 37.5%; the round-trip fee raises it to
        43.75%. Any threshold the system compares a probability against should be
        derived from here.
        """
        round_trip_fee = 2 * self.fee_rate
        win_net = win_threshold / 100.0 - round_trip_fee
        loss_net = loss_threshold / 100.0 + round_trip_fee
        if win_net <= 0:
            return 1.0
        return loss_net / (win_net + loss_net)

    def correlation_haircut(self, concurrent_positions: int) -> float:
        """
        Scale a standalone Kelly fraction down for positions held alongside it.

        Kelly for a single bet assumes that bet is the only thing at risk. It is not:
        this system runs up to 30 concurrent long positions, and 30 correlated longs
        behave much like one leveraged index bet. Sizing each as though it were
        independent over-levers the portfolio by roughly the correlation factor.

        With `n` simultaneous bets of average pairwise correlation rho, the effective
        number of independent bets is n / (1 + (n-1)*rho), so each position is scaled
        by 1 / (1 + (n-1)*rho).

        At rho = 0.30, ten open positions size at 27% of standalone Kelly. That is not
        excessive conservatism -- it is what keeps total portfolio risk at the level a
        single Kelly bet was supposed to represent.
        """
        n = max(1, int(concurrent_positions))
        if n == 1 or self.assumed_correlation <= 0:
            return 1.0
        return 1.0 / (1.0 + (n - 1) * self.assumed_correlation)

    def calculate_bet_size(self,
                          probability: float,
                          current_price: float,
                          available_capital: float,
                          win_threshold: Optional[float] = None,
                          loss_threshold: Optional[float] = None,
                          concurrent_positions: int = 1) -> BetRecommendation:
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
        
        self.concurrent_positions = max(1, int(concurrent_positions))

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
        """
        Calculate the growth-optimal bet fraction for a CAPPED-LOSS bet.

        This bet does not risk the whole stake: it wins w% of the position or loses
        l% of it. Maximising E[log(wealth)] over f gives

            d/df [ p*log(1 + f*w) + q*log(1 - f*l) ] = 0
            =>  f* = (p*w - q*l) / (w*l)  =  p/l - q/w

        The all-or-nothing form f = (bp - q)/b with b = w/l is the Kelly for a bet
        that loses the entire stake, and understates this payoff substantially.

        Both w and l are taken NET OF FEES, since fees are paid on entry and exit:
            w_net = w - 2c,  l_net = l + 2c
        On a 5%/3% bet at 0.25% per side this moves break-even from 37.5% to 43.75%.
        """
        p = params.probability_win
        q = 1 - p

        win_return = params.win_percentage / 100.0   # gross win, decimal
        loss_risk = params.loss_percentage / 100.0   # gross loss, decimal

        # Net the round-trip fee out of both legs.
        round_trip_fee = 2 * self.fee_rate
        win_net = win_return - round_trip_fee
        loss_net = loss_risk + round_trip_fee

        # Reported odds ratio, on the net legs.
        b = win_net / loss_net if loss_net > 0 else 0.0

        if win_net <= 0:
            # Fees exceed the entire profit target: no size can make this positive.
            kelly_fraction_raw = -1.0
            expected_value = -loss_net
        else:
            # f* = p/l - q/w  (capped-loss Kelly, fee-adjusted)
            kelly_fraction_raw = (p / loss_net) - (q / win_net)
            expected_value = p * win_net - q * loss_net

        # Break-even probability, for logging and warnings.
        self.break_even_probability = loss_net / (win_net + loss_net) if win_net > 0 else 1.0

        self.logger.debug(f"Kelly calculation: p={p:.3f}, q={q:.3f}, "
                          f"w_net={win_net:.4f}, l_net={loss_net:.4f}, b={b:.3f}, "
                          f"break_even={self.break_even_probability:.3f}, "
                          f"raw_kelly={kelly_fraction_raw:.3f}")

        is_favorable = expected_value > 0 and kelly_fraction_raw > 0

        if not is_favorable:
            return BetRecommendation(
                recommended_amount=0.0,
                fraction_of_capital=0.0,
                expected_value=expected_value,
                is_favorable=False,
                kelly_fraction_raw=kelly_fraction_raw,
                confidence_level="N/A",
                risk_warning="Negative expected value - no bet recommended",
                # Detailed calculation breakdown
                win_probability=p,
                loss_probability=q,
                win_amount_ratio=win_net,
                loss_amount_ratio=loss_net,
                expected_win=p * max(win_net, 0.0),
                expected_loss=q * loss_net,
                kelly_formula_b=b,
                kelly_formula_p=p,
                kelly_formula_q=q,
                available_capital=params.available_capital
            )

        # --- Sizing chain -------------------------------------------------
        # 1. Fractional Kelly (conservative multiplier), then a correlation haircut
        #    for everything already open. Sizing each of 30 correlated longs as though
        #    it were the only position at risk over-levers the whole book.
        haircut = self.correlation_haircut(getattr(self, 'concurrent_positions', 1))
        adjusted_kelly_fraction = (kelly_fraction_raw
                                   * self.kelly_fraction_multiplier
                                   * haircut)

        # 2. Risk budget: cap capital-at-risk per bet. Because capped-loss Kelly is
        #    unbounded, this is normally the binding constraint and is what keeps
        #    position size responsive to edge instead of pinned at the hard cap.
        risk_budget_fraction = (self.max_risk_fraction / loss_net) if loss_net > 0 else 0.0

        # 3. Hard position cap.
        final_kelly_fraction = min(
            adjusted_kelly_fraction,
            risk_budget_fraction,
            self.max_bet_fraction,
        )

        # Calculate dollar amount
        raw_bet_amount = final_kelly_fraction * params.available_capital

        # Apply the upper bet-size limit and available capital. The minimum bet is a
        # SKIP threshold, not a floor -- raising a small Kelly bet up to the minimum
        # would breach max_bet_fraction exactly when capital is lowest.
        recommended_amount = min(raw_bet_amount, self.max_bet_amount, params.available_capital)

        below_minimum = recommended_amount < self.min_bet_amount
        if below_minimum:
            recommended_amount = 0.0

        # Recalculate actual fraction used
        actual_fraction = recommended_amount / params.available_capital if params.available_capital > 0 else 0

        # Determine confidence level
        confidence_level = self._determine_confidence_level(params.probability_win, kelly_fraction_raw)

        # Generate risk warnings
        risk_warning = self._generate_risk_warnings(params, kelly_fraction_raw, final_kelly_fraction)
        if below_minimum:
            skip_note = (f"Kelly size ${raw_bet_amount:.2f} below minimum "
                         f"${self.min_bet_amount:.2f} - no bet")
            risk_warning = f"{risk_warning}; {skip_note}" if risk_warning else skip_note

        recommendation = BetRecommendation(
            recommended_amount=recommended_amount,
            fraction_of_capital=actual_fraction,
            expected_value=expected_value,
            is_favorable=is_favorable and not below_minimum,
            kelly_fraction_raw=kelly_fraction_raw,
            confidence_level=confidence_level,
            risk_warning=risk_warning,
            # Detailed calculation breakdown (net of the round-trip fee)
            win_probability=p,
            loss_probability=q,
            win_amount_ratio=win_net,
            loss_amount_ratio=loss_net,
            expected_win=p * win_net,
            expected_loss=q * loss_net,
            kelly_formula_b=b,
            kelly_formula_p=p,
            kelly_formula_q=q,
            available_capital=params.available_capital
        )
        
        self.logger.info(f"Kelly recommendation: ${recommended_amount:.2f} "
                        f"({actual_fraction:.1%} of capital), EV={expected_value:.3f}")
        
        return recommendation
    
    def _determine_confidence_level(self, probability: float, kelly_fraction: float) -> str:
        """
        Confidence based on how far the estimate clears break-even.

        Measured against the fee-adjusted break-even probability rather than 50%,
        which is not the neutral point for a barrier bet.
        """
        margin = probability - self.break_even_probability

        if margin >= 0.20:
            return "High"
        elif margin >= 0.10:
            return "Medium"
        elif margin >= 0.03:
            return "Low"
        else:
            return "Very Low"

    def _generate_risk_warnings(self, params: BetParameters,
                               raw_kelly: float, final_kelly: float) -> Optional[str]:
        """Generate risk warnings for the bet"""
        warnings = []

        # Check probability threshold
        if params.probability_win < self.min_probability:
            warnings.append(f"Low win probability ({params.probability_win:.1%})")

        # Margin over the real neutral point for this payoff
        margin = params.probability_win - self.break_even_probability
        if margin < 0.05:
            warnings.append(f"Thin margin over break-even "
                            f"({params.probability_win:.1%} vs {self.break_even_probability:.1%})")

        # Which constraint actually set the size
        haircut = self.correlation_haircut(getattr(self, 'concurrent_positions', 1))
        if haircut < 0.999:
            warnings.append(f"Correlation haircut {haircut:.0%} for "
                            f"{self.concurrent_positions} concurrent positions")

        if final_kelly < raw_kelly * self.kelly_fraction_multiplier * haircut * 0.99:
            warnings.append("Size set by risk cap, not by Kelly")

        # Check loss threshold
        if params.loss_percentage > self.max_loss_percentage:
            warnings.append(f"High loss risk ({params.loss_percentage:.1f}%)")

        # Check if betting large fraction of capital
        if final_kelly > 0.05:  # More than 5%
            warnings.append(f"Large bet size ({final_kelly:.1%} of capital)")

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
        
        # The neutral point for a barrier bet is l/(w+l) adjusted for fees, not 50%.
        break_even = self._break_even_probability(win_threshold, loss_threshold) * 100.0
        if probability < break_even:
            return False, (f"Probability too low: {probability:.2f}% "
                           f"(break-even for {win_threshold:.1f}%/{loss_threshold:.1f}% "
                           f"after fees is {break_even:.2f}%)")

        return True, "Valid"
    
    def get_kelly_info(self) -> Dict:
        """Get information about Kelly calculator configuration"""
        return {
            'kelly_fraction_multiplier': self.kelly_fraction_multiplier,
            'min_bet_amount': self.min_bet_amount,
            'max_bet_amount': self.max_bet_amount,
            'max_bet_fraction': self.max_bet_fraction,
            'max_risk_fraction': self.max_risk_fraction,
            'min_probability': self.min_probability,
            'max_loss_percentage': self.max_loss_percentage,
            'fee_rate_per_side': self.fee_rate,
            'break_even_probability': self.break_even_probability,
            'assumed_correlation': self.assumed_correlation,
        }