# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Trading system that uses ML predictions and Kelly Criterion for optimal bet sizing. Supports manual and automated modes with built-in risk controls.

### Operating Modes

#### Manual Mode
1. Calculate probabilities for all defined assets (S&P 500 stocks + top 50 crypto)
2. Display top 10 assets with highest probability of price increase
3. User selects bet by number input
4. System asks for confirmation (y/n) before placing bet

#### Automated Mode
1. Same probability calculation and ranking
2. If top asset has >z% probability (benchmark during testing, target >50%), automatically place bet
3. If all probabilities <50%, wait 30 minutes and repeat
4. No human intervention required

## Technology Stack

- **Python**: Core implementation language
- **SQLite/PostgreSQL**: Local relational database for data storage
- **APIs**: Market data polling (stocks and crypto)
- **ML Libraries**: scikit-learn, pandas, numpy for predictions
- **Trading APIs**: For order execution
- **Streamlit**: Dashboard UI framework

## UI/Dashboard Design Guidelines

### Layout Principles
- **Vertical Stacking**: All tables and sections should be stacked vertically (underneath each other), NOT side-by-side
- **Trading Dashboard**: Active Bets should appear underneath Market Opportunities
- **All Bets Tab**: Closed Bets should appear underneath Active Bets
- **Consistent Layout**: Apply vertical stacking consistently across all dashboard sections

### Fee Structure
- **Trading Fee**: 0.25% per side, applied on both entry and exit
- **Total Fee per Round Trip**: 0.50% (0.25% x 2)
- **Consequence**: on a +5%/-3% bet this raises the break-even win rate from
  37.5% to 43.75%. Fees must appear in the expected-value calculation, not just
  be deducted from cash afterwards.

## Core Architecture

### Bet Definition
- **Bet**: Buy an asset with win condition of +x% price increase, loss condition of -y% price decrease
- **States**: `alive`, `won`, `lost`
- **Exit Triggers**: Price hits win/loss thresholds -> automatic sell, OR the time
  barrier (`max_hold_days`) expires and the position is closed at market
- **Capital**: X amount allocated per bet using Kelly formula

**Barriers are volatility-scaled by default** (`trading.barrier.mode: volatility`):
win at `+k*sigma_h`, loss at `-m*sigma_h`, where `sigma_h` is the asset's own trailing
volatility scaled to the holding horizon. Fixed percentage barriers are still
available (`mode: fixed`) but were measured to be the wrong shape for an 860-asset
universe: a +5%/-3% bet is a completely different event on RIVN than on EURGBP.

All barrier placement goes through `src/trading/barriers.py::BarrierPolicy`. Labelling,
Kelly sizing, bet placement and monitoring must all derive barriers from it, or they
will silently disagree about what bet is being made. Barriers travel on the prediction
dict (`win_threshold`/`loss_threshold`, as PERCENTAGES) so sizing prices the same bet
the model forecast.

**This is a first-touch double-barrier event.** Whether a bet wins depends on which
price level is touched FIRST, which is path-dependent. Three consequences that govern
the whole design:

1. **The neutral point is around 40%, not 50%.** For a driftless asset the win
   probability is set by the barrier geometry (roughly `m/(k+m)`), independent of
   volatility. Treat that formula as an APPROXIMATION: real bars are discrete and
   prices compound, so the true value is a couple of points away and depends on step
   size relative to barrier width. Where the reference matters, call
   `BarrierPolicy.estimate_geometry_probability()` (which measures it) or read the
   base rate the backtest reports from real data. Never compare against 50%.

1b. **Fees do not scale with volatility, so break-even varies per asset.** A 0.50%
   round trip is noise against an 8% win barrier and a third of a 1.4% one. That
   arithmetic is what makes quiet assets uneconomic, and `BarrierPolicy.is_economic()`
   refuses them rather than trading at a 60% break-even.
2. **Models must be trained on this event.** A fixed-horizon label
   (`Close.shift(-5)/Close - 1 > 0.03`) is path-independent and ignores the stop
   entirely - it predicts something nobody bets on. Use the triple-barrier labels in
   `BasePredictionAlgorithm._barrier_labels()` / `barrier_outcomes()`.

### Key Components

#### 1. Data Management (`data/`)
- Market data polling and storage
- Price history and real-time feeds
- Database schema for assets, bets, portfolio

#### 2. Prediction Engine (`prediction/`)
- ML models for price movement forecasting
- Feature engineering from market data
- Model training and evaluation pipelines

#### 3. Kelly Calculator (`kelly/`)
- Probability estimation from ML predictions
- Optimal bet sizing using Kelly Criterion: f* = (bp - q) / b
- Risk-adjusted position sizing

#### 4. Trading Engine (`trading/`)
- Order placement and execution
- Position monitoring and management
- Automatic sell triggers for win/loss conditions

#### 5. Portfolio Manager (`portfolio/`)
- Capital allocation and tracking
- Bet lifecycle management (alive → won/lost)
- Performance metrics and reporting

#### 6. Risk Management (`risk/`)
- Loss streak detection and circuit breakers
- Capital preservation rules
- Emergency stop conditions

### Database Schema

#### Assets Table
- `asset_id`, `symbol`, `asset_type` (stock/crypto), `current_price`, `last_updated`

#### Price Data Table (OHLCV)
- `price_id`, `asset_id`, `timestamp`, `open`, `high`, `low`, `close`, `volume`, `interval`

#### Bets Table
- `bet_id`, `asset_id`, `entry_price`, `win_threshold`, `loss_threshold`, `amount`, `status`, `entry_time`, `exit_time`, `pnl`

#### Portfolio Table
- `portfolio_id`, `total_capital`, `available_capital`, `active_bets_value`, `timestamp`

## Asset Coverage

- **Stocks**: S&P 500 companies (500 assets)
- **Crypto**: Top 50 cryptocurrencies by market cap
- **Future Expansion**: Other stock indices, additional cryptocurrencies, forex, commodities

## Development Commands

```bash
# Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Database setup
python scripts/init_db.py

# Run manual mode
python main.py --mode manual

# Run automated mode
python main.py --mode automated --threshold 65

# Run tests
pytest tests/

# Walk-forward backtest (build this understanding BEFORE changing any threshold)
python scripts/backtest.py --max-assets 120 --folds 4

# Sweep the barrier geometry - max_hold_days is the dominant parameter
python scripts/backtest.py --max-hold 30
python scripts/backtest.py --win-sigma 1.5 --loss-sigma 1.0 --horizon 10
python scripts/backtest.py --barrier-mode fixed --win 5 --loss 3   # A/B the old shape

# Fit the ensemble calibration from backtest output
python scripts/fit_calibration.py

# Repair legacy BLOB-encoded prediction rows (one-off)
python scripts/repair_prediction_blobs.py --dry-run
```

## Key Configuration Parameters

- **Win Threshold (x%)**: Profit target for bet victory
- **Loss Threshold (y%)**: Stop loss for bet defeat  
- **Kelly Fraction**: Conservative multiplier (0.25 for quarter-Kelly)
- **Auto Threshold (z%)**: Minimum probability for automated betting (target >50%, benchmark during testing)
- **Max Concurrent Bets**: Portfolio diversification limit
- **Circuit Breaker**: Consecutive loss limit or capital drawdown threshold
- **Polling Interval**: Market data refresh frequency (30 minutes when no bets qualify)
- **Top N Display**: Show top 10 assets in manual mode

## Risk Controls

- **Loss Streak Limit**: Pause after N consecutive losses
- **Drawdown Threshold**: Stop if capital drops by X% in Y timeframe
- **Position Size Limits**: Maximum bet size as % of portfolio
- **Correlation Limits**: Avoid concentrated exposure to similar assets

## Important Implementation Notes

- Validate Kelly inputs against the fee-adjusted break-even for the configured
  payoff (43.75% for 5%/3% at 0.25% per side), NOT against 0.5
- Handle edge cases: network failures, API rate limits, market closures
- Implement proper logging for audit trail of all decisions
- Use fractional Kelly. NOTE: for this capped-loss payoff the multiplier is
  ~0.005, not 0.25 - quarter-Kelly here would be 206% of capital.
- Store all bet rationale and ML model outputs for analysis
- Cast numpy scalars with `float()` before inserting into SQLite. numpy float32 does
  not adapt to REAL and is stored as a 4-byte BLOB, which silently corrupts every
  aggregate over the column.
- Never compare a native-currency quote against a USD barrier. Currency conversion
  belongs in `MarketDataManager.get_latest_data()`, which normalises everything to
  USD and drops assets it cannot convert.
- Barrier outcomes on overlapping forward windows are strongly correlated, so a raw
  observation count badly overstates the effective sample. A single 2,500-bar path
  estimates a barrier win rate to only about +/-5 percentage points. Average across
  INDEPENDENT paths (or assets) before believing a win-rate difference.
- Implement graceful shutdown and position cleanup procedures

## Critical Design Considerations (Open Topics)

### Market Execution Reality
- **Slippage & Gaps**: Markets don't guarantee exact +x%/-y% exits due to price gaps, after-hours movements, and order slippage
- **Solution**: Implement buffer zones (e.g., exit at x-0.1% for wins) and realistic exit mechanisms with market orders vs limit orders

### Kelly Criterion Challenges  
- **Probability Accuracy**: Kelly assumes accurate probability estimation, but ML confidence scores may not reflect true probabilities
- **Model Drift**: Prediction accuracy degrades over time as market conditions change
- **Solution**: Continuous model validation, out-of-sample testing, and dynamic probability calibration

### Portfolio Correlation Risk
- **Simultaneous Losses**: Multiple bets can hit loss thresholds together during market crashes or sector rotations
- **Hidden Correlations**: Assets may appear uncorrelated in normal times but correlate during stress
- **Solution**: Correlation-aware position sizing and stress testing with historical crisis scenarios

### Execution Timing
- **Prediction-to-Trade Lag**: Time between ML prediction, decision, and actual order execution can be significant
- **Stale Predictions**: Asset prices may move substantially during API calls and order processing
- **Solution**: Factor execution delay into prediction windows and implement rapid order execution pipelines

### Regulatory & Practical Constraints
- **Pattern Day Trading**: Rules limiting frequent trading with <$25k accounts
- **API Limits**: Rate limiting on data feeds and trading APIs
- **Transaction Costs**: Fees can erode small gains, especially with frequent trading
- **Solution**: Model transaction costs explicitly and ensure compliance with trading regulations

### Overfitting & Model Robustness
- **Historical Bias**: Models trained on past data may not generalize to future market regimes
- **Data Snooping**: Over-optimization on limited historical data
- **Solution**: Walk-forward analysis, regime detection, and conservative out-of-sample validation periods

**Testing Strategy**: Use paper trading extensively before live deployment to validate these assumptions under real market conditions.

## Windows Console Compatibility

**IMPORTANT**: Avoid using emoji symbols or Unicode characters in output as Windows console has encoding issues.

- Use plain text instead of emojis (✅ ❌ 🎯 etc.)
- Stick to standard ASCII characters
- This applies to all console output, error messages, and UI text

## Data Caching & Storage Strategy

### Local Database Storage
- **Store all polled data** to minimize API calls and respect rate limits
- **Recent data (90 days)**: 30-minute intervals for detailed analysis
- **Historical data**: Aggregated to daily intervals to save space
- **Estimated storage**: ~500MB-1GB per year for 550 assets (very manageable for local PC)

### Caching Benefits
- **API rate limiting**: yfinance (~2000 req/hour), crypto APIs vary
- **Performance**: Local queries much faster than API calls
- **Reliability**: System works offline for analysis and backtesting  
- **Cost efficiency**: Avoid expensive high-frequency API plans

### Data Retention
- Automatic cleanup of old detailed data
- Intelligent caching prevents redundant API calls
- Database size monitoring with alerts