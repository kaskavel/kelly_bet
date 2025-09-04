#!/usr/bin/env python3
"""
Demo of Enhanced Kelly Criterion Calculator
Shows the detailed breakdown similar to what you'll see in trading.
"""

import yaml
from src.kelly.calculator import KellyCalculator

def demo_kelly_analysis():
    """Demo the enhanced Kelly analysis"""
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize Kelly calculator
    kelly = KellyCalculator(config)
    
    print("="*80)
    print("KELLY CRITERION BET ANALYSIS - MU (Demo)")
    print("="*80)
    
    # Example calculation similar to your MU bet
    result = kelly.calculate_bet_size(
        probability=78.68,  # Your actual probability
        current_price=118.71,
        available_capital=9000.0
    )
    
    print(f"Asset: MU")
    print(f"Current Price: $118.71")
    print(f"Available Capital: ${result.available_capital:,.2f}")
    
    print(f"\nPROBABILITY ANALYSIS:")
    print(f"  Win Probability (p): {result.win_probability:.1%} (78.68%)")
    print(f"  Loss Probability (q): {result.loss_probability:.1%}")
    
    print(f"\nTHRESHOLD SETUP:")
    print(f"  Win Target: +5.0% ($124.65)")
    print(f"  Loss Stop: -3.0% ($115.15)")
    
    print(f"\nKELLY FORMULA CALCULATION:")
    print(f"  Formula: f = (bp - q) / b")
    print(f"  where:")
    print(f"    b (odds ratio) = {result.kelly_formula_b:.3f} (win/loss ratio)")
    print(f"    p (win probability) = {result.kelly_formula_p:.3f}")
    print(f"    q (loss probability) = {result.kelly_formula_q:.3f}")
    print(f"  ")
    print(f"  Calculation: f = ({result.kelly_formula_b:.3f} × {result.kelly_formula_p:.3f} - {result.kelly_formula_q:.3f}) / {result.kelly_formula_b:.3f}")
    print(f"  Raw Kelly Fraction = {result.kelly_fraction_raw:.1%}")
    
    print(f"\nEXPECTED VALUE ANALYSIS:")
    print(f"  Expected Win: {result.win_probability:.1%} × {result.win_amount_ratio:.1%} = {result.expected_win:.3f}")
    print(f"  Expected Loss: {result.loss_probability:.1%} × {result.loss_amount_ratio:.1%} = {result.expected_loss:.3f}")
    print(f"  Net Expected Value: {result.expected_win:.3f} - {result.expected_loss:.3f} = {result.expected_value:.3f}")
    print(f"  ")
    print(f"  This means: For every $1 bet, you expect to gain ${result.expected_value:.3f} on average")
    
    print(f"\nRISK ADJUSTMENTS:")
    if result.kelly_fraction_raw != result.fraction_of_capital:
        kelly_multiplier = config.get('trading', {}).get('kelly_fraction', 0.25)
        print(f"  Conservative Multiplier: {kelly_multiplier:.1%} (reduces risk)")
        print(f"  Max Position Size Cap: {config.get('trading', {}).get('max_bet_fraction', 0.1):.1%}")
        print(f"  After Adjustments: {result.fraction_of_capital:.1%}")
    else:
        print(f"  No adjustments applied")
    
    print(f"\nFINAL RECOMMENDATION:")
    print(f"  Bet Amount: ${result.recommended_amount:,.2f}")
    print(f"  Position Size: {result.fraction_of_capital:.1%} of capital")
    print(f"  Confidence Level: {result.confidence_level}")
    
    if result.risk_warning:
        print(f"\nWARNINGS:")
        for warning in result.risk_warning.split(';'):
            print(f"  WARNING: {warning.strip()}")
    
    print(f"\n" + "="*80)

if __name__ == '__main__':
    demo_kelly_analysis()