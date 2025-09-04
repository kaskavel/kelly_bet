# Kelly Criterion Trading System

An intelligent trading system that uses machine learning predictions and the Kelly Criterion for optimal bet sizing. The system analyzes S&P 500 stocks and top cryptocurrencies using multiple algorithms, then places bets with mathematically optimal position sizes.

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   pip install -r requirements.txt
   ```

2. **Run the System**
   ```bash
   # Manual mode - you select which bets to place
   python main.py --mode manual
   
   # Automated mode - system places bets automatically
   python main.py --mode automated --threshold 65
   ```

3. **First Run**
   - System initializes with $10,000 paper money
   - Database created automatically at `data/trading.db`
   - All trades are simulated - no real money involved

## 🎯 Features

### **Multi-Algorithm Prediction Engine**
- **5 Algorithms**: Simple Moving Average, RSI, Random Forest, LSTM Neural Network, Linear Regression
- **Ensemble Scoring**: Weighted predictions with adaptive learning
- **Performance Tracking**: System learns which algorithms work best over time

### **Kelly Criterion Position Sizing**
- Mathematically optimal bet sizing using Kelly formula
- Conservative approach (25% Kelly fraction) to reduce volatility
- Risk-adjusted based on probability and win/loss thresholds

### **Risk Management**
- **Circuit Breakers**: Auto-pause on loss streaks or excessive drawdown
- **Drawdown Limits**: Emergency stop at 20% portfolio decline
- **Exposure Controls**: Maximum 10% of capital per bet
- **Daily/Weekly Loss Limits**: Prevent catastrophic losses

### **Paper Trading**
- **Realistic Simulation**: 0.5% trading fees on entry and exit
- **Real Market Data**: Live prices from Yahoo Finance and crypto exchanges
- **Complete Portfolio Tracking**: Cash + active positions with P&L

## 🏗️ System Architecture

```
Market Data → Predictions → Kelly Calculator → Portfolio Manager → Risk Manager
     ↓              ↓              ↓                ↓                 ↓
   550 Assets   5 Algorithms   Optimal Sizing   Bet Tracking   Circuit Breakers
   (S&P500 +    (Ensemble      (Kelly Formula)  (Database)     (Auto-Pause)
    Top 50      Weighted)
    Crypto)
```

## 📊 Operating Modes

### **Manual Mode**
```bash
python main.py --mode manual
```
1. System analyzes all assets and shows top 10 opportunities
2. You select which bets to place by number
3. System asks for confirmation before placing each bet
4. Perfect for learning and oversight

### **Automated Mode**
```bash
python main.py --mode automated --threshold 65
```
1. System automatically places bets when probability > threshold
2. If all probabilities < 50%, waits 30 minutes and repeats
3. Fully autonomous trading with risk controls
4. Great for backtesting and live deployment

## 🎛️ Configuration

Edit `config/config.yaml` to customize:

```yaml
# Trading Parameters
trading:
  win_threshold: 5.0        # Profit target (5%)
  loss_threshold: 3.0       # Stop loss (3%)
  kelly_fraction: 0.25      # Conservative Kelly multiplier
  initial_capital: 10000.0  # Starting paper money
  max_concurrent_bets: 10   # Maximum active positions
  trading_fee_percentage: 0.5  # Realistic trading fees

# Risk Management  
risk:
  max_drawdown: 20.0        # Emergency stop at 20% drawdown
  loss_streak_limit: 5      # Pause after 5 consecutive losses
  daily_loss_limit: 5.0     # Daily loss limit (5%)

# Algorithm Settings
prediction:
  algorithms:
    sma:
      short_window: 5
      long_window: 20
    lstm:
      sequence_length: 60
      epochs: 50
    # ... and more
```

## 📈 Example Usage

### Manual Trading Session
```
$ python main.py --mode manual

=============================================================
TOP INVESTMENT OPPORTUNITIES
=============================================================
 1. AAPL      | Probability: 67.30% | Price: $   145.20
 2. MSFT      | Probability: 65.80% | Price: $   285.40  
 3. GOOGL     | Probability: 63.20% | Price: $   125.80
 4. BTC       | Probability: 62.10% | Price: $ 43250.00
 5. NVDA      | Probability: 61.40% | Price: $   420.15
=============================================================

Select bet (1-5) or 'q' to quit: 1

BET DETAILS:
Asset: AAPL
Current Price: $145.20
Win Probability: 67.30%
Recommended Bet Size: $425.50

Proceed with this bet? (y/N): y
✓ Bet placed: AAPL (ID: a1b2c3d4)
```

## 📊 Database Analysis

All data is stored in SQLite database (`data/trading.db`). You can analyze performance anytime:

### Using DB Browser for SQLite (Recommended)
1. Download **DB Browser for SQLite** (free)
2. Open `data/trading.db`
3. Browse tables visually and run SQL queries

### Python Analysis
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('data/trading.db')

# Analyze bet performance
bets = pd.read_sql('SELECT * FROM bets', conn)
print(f"Total bets: {len(bets)}")
print(f"Win rate: {len(bets[bets.status=='won']) / len(bets):.1%}")

# Algorithm performance comparison
predictions = pd.read_sql('SELECT * FROM predictions', conn)
perf = predictions.groupby('algorithm')['probability'].mean()
print(perf)

conn.close()
```

### Key Database Tables
- **`bets`**: Complete bet history with P&L
- **`predictions`**: All algorithm predictions over time  
- **`portfolio_history`**: Portfolio value snapshots
- **`algorithm_performance`**: Track which algorithms work best
- **`risk_events`**: Risk management alerts and actions

## 🛡️ Safety Features

- **Paper Trading Only**: No real money at risk
- **Circuit Breakers**: Auto-pause on dangerous conditions
- **Conservative Kelly**: Uses 25% Kelly fraction to reduce volatility  
- **Risk Monitoring**: Continuous assessment of portfolio health
- **Data Validation**: Extensive input validation and error handling
- **Graceful Shutdown**: Clean database closure and position cleanup

## 🔧 Technical Details

### **Performance**
- **Execution Time**: ~3-5 minutes per cycle (550 assets)
- **Memory Usage**: ~500MB (all models + data)
- **Database Size**: ~500MB-1GB per year of trading
- **API Limits**: Intelligent caching prevents rate limiting

### **Dependencies**
- **Market Data**: Yahoo Finance (yfinance), CCXT for crypto
- **Machine Learning**: scikit-learn, TensorFlow/Keras for LSTM
- **Database**: SQLite (file-based, no server required)
- **Data Processing**: pandas, numpy, scipy

### **Asset Coverage**
- **Stocks**: S&P 500 companies (500 assets)
- **Crypto**: Top 50 cryptocurrencies by market cap
- **Future Expansion**: Easy to add more asset classes

## 📋 System Requirements

- **Python**: 3.8+ required
- **Memory**: 2GB+ recommended  
- **Storage**: 5GB+ for data and models
- **Internet**: Required for market data (caches locally)
- **OS**: Windows, Linux, macOS

## 🆘 Troubleshooting

### Common Issues

**"TensorFlow not available"**
```bash
pip install tensorflow>=2.13.0
```

**"Insufficient data for prediction"**  
- Wait for initial data collection (first run takes longer)
- Check internet connection for market data

**"Emergency stop triggered"**
- Check risk management logs in database
- Adjust risk thresholds in config if needed
- Use manual mode to investigate

**Database locked error**
- Ensure only one instance of the program is running
- Close any DB browser connections to the database file

### Getting Help

1. Check logs in `logs/trading.log`
2. Examine database tables for debugging
3. Review configuration in `config/config.yaml`
4. Run in manual mode for step-by-step troubleshooting

## 📚 Understanding the Kelly Criterion

The Kelly Criterion determines optimal bet size using:

**Formula**: `f = (bp - q) / b`

Where:
- `f` = fraction of capital to bet
- `b` = odds received (win_return / loss_risk)  
- `p` = probability of winning
- `q` = probability of losing (1 - p)

**Example**: 
- 65% win probability, 5% win target, 3% loss limit
- Kelly suggests betting ~8% of capital
- Our system uses 25% of Kelly suggestion = ~2% of capital for safety

## 🔮 Future Enhancements

- **Live Trading Integration**: Connect to real brokers (Interactive Brokers, Alpaca)
- **More Assets**: Forex, commodities, options
- **Advanced Algorithms**: Transformer models, reinforcement learning
- **Portfolio Optimization**: Modern Portfolio Theory integration
- **Web Interface**: Real-time dashboard and controls
- **Backtesting Engine**: Historical strategy testing
- **Multi-Timeframe Analysis**: Different prediction horizons

## ⚠️ Disclaimer

This software is for educational and research purposes only. Past performance does not guarantee future results. Trading involves substantial risk of loss. The authors are not responsible for any financial losses incurred through the use of this software.

## 📄 License

MIT License - see LICENSE file for details.

---

**Ready to start? Run `python main.py --mode manual` and begin your Kelly Criterion trading journey!** 🚀