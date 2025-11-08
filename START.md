# How to Start the Paper Trading System

This guide explains how to start the paper trading system in different modes.

## Prerequisites

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install TA-Lib** (Optional but recommended)
   - Windows: Download wheel from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib) and install
   - Linux: `sudo apt-get install ta-lib` then `pip install TA-Lib`
   - Mac: `brew install ta-lib` then `pip install TA-Lib`

## Starting the System

### Option 1: Start with Command Line Interface (CLI)

Run the main script to start the trading engine with console output:

```bash
python main.py
```

**What happens:**
- Initializes the trading engine
- Connects to data feed (FakeDataFeed by default)
- Adds 3 example strategies:
  - Moving Average Crossover for RELIANCE
  - RSI Strategy for TCS
  - Moving Average Crossover for INFY
- Starts trading and prints portfolio updates every 30 seconds
- Press `Ctrl+C` to stop

**Output:**
- Real-time trading logs
- Portfolio summaries every 30 seconds
- Position updates
- P&L information

### Option 2: Start with Streamlit Dashboard (Recommended)

For a visual interface with charts and real-time metrics:

```bash
streamlit run dashboard.py
```

Or use the helper script:

```bash
python run_dashboard.py
```

**What happens:**
- Opens a web browser with the dashboard
- Shows real-time portfolio metrics
- Displays LTP (Last Traded Price) for all symbols
- Shows open positions
- Displays strategy performance metrics
- Shows daily performance with charts
- Auto-refreshes every 5 seconds (configurable)

**Dashboard Features:**
1. **Control Panel (Sidebar)**
   - Start/Stop Engine buttons
   - Auto-refresh toggle
   - Refresh interval slider

2. **Portfolio Overview**
   - Total Equity
   - Total P&L
   - Realized/Unrealized P&L
   - Cash, Exposure, Commission

3. **LTP Table**
   - Last Traded Price for all symbols
   - Timestamp of last update

4. **Open Positions**
   - Current positions with P&L
   - Entry price, current price
   - Unrealized P&L

5. **Strategy Performance**
   - Per-strategy metrics
   - Win rate, total trades
   - Strategy P&L chart

6. **Daily Performance**
   - Daily metrics table
   - Equity curve chart
   - Daily P&L chart

7. **Summary Metrics**
   - Total trades, win rate
   - Return percentage
   - Max drawdown
   - Average trade P&L

**To use the dashboard:**
1. Run `streamlit run dashboard.py`
2. Click "Start Engine" in the sidebar
3. Watch the dashboard update in real-time
4. Click "Stop Engine" when done

## Configuration

### Using Fake Data Feed (Default)

The system uses `FakeDataFeed` by default, which generates random market data for testing. No configuration needed.

### Using ICICI Breeze API

To use real market data from ICICI Breeze API:

1. Set environment variables:
   ```bash
   export ICICI_API_KEY="your_api_key"
   export ICICI_API_SECRET="your_api_secret"
   export ICICI_SESSION_TOKEN="your_session_token"
   export ICICI_ENABLED="true"
   ```

2. Or edit `config.py`:
   ```python
   ICICIConfig(
       api_key="your_api_key",
       api_secret="your_api_secret",
       session_token="your_session_token",
       enabled=True
   )
   ```

## Customizing Strategies

### Modify Existing Strategies

Edit `main.py` or `dashboard.py` to modify strategy parameters:

```python
ma_strategy = MovingAverageCrossoverStrategy(
    strategy_id="MA_CROSS_RELIANCE",
    symbol="RELIANCE",
    fast_period=10,      # Fast SMA period
    slow_period=30,      # Slow SMA period
    quantity=10,         # Number of shares per trade
    stop_loss_pct=0.02,  # 2% stop loss
    take_profit_pct=0.04 # 4% take profit
)

# Add with resampling (5-minute candles)
engine.add_strategy(ma_strategy, resample_timeframe='5min')
```

### Add More Symbols

Edit the symbols list:

```python
symbols = ["RELIANCE", "TCS", "INFY", "HDFC", "ICICIBANK"]
engine.initialize_data_feed(symbols)
```

### Create Custom Strategies

1. Create a new strategy class inheriting from `BaseStrategy`
2. Implement `_initialize_indicators()` method
3. Optionally override `_check_entry_conditions()` and `_check_exit_conditions()`
4. Add the strategy to the engine

See `strategy/example_strategies.py` for examples.

## Troubleshooting

### Import Errors

If you see import errors:
1. Make sure all dependencies are installed: `pip install -r requirements.txt`
2. Check that you're in the project directory
3. Verify Python version (3.8+)

### TA-Lib Errors

If TA-Lib is not installed:
- The system will use custom indicators (slower, less accurate)
- Install TA-Lib for better performance (see Prerequisites)

### Dashboard Not Loading

1. Make sure Streamlit is installed: `pip install streamlit`
2. Check if port 8501 is available
3. Try: `streamlit run dashboard.py --server.port 8502`

### No Trades Executing

1. Check that strategies are added
2. Verify data feed is generating data
3. Check logs for entry condition evaluations
4. Ensure sufficient cash for trades
5. Check RMS (Risk Management) settings

## Logs

The system generates two types of logs:

1. **trading.log** - Human-readable logs
   - Strategy actions
   - Order submissions
   - Fills
   - Portfolio updates

2. **trading_structured.jsonl** - Machine-readable logs
   - JSON format for analysis
   - All events with timestamps
   - Suitable for data analysis

## Stopping the System

### CLI Mode
- Press `Ctrl+C` to stop gracefully
- The system will close all connections and save logs

### Dashboard Mode
- Click "Stop Engine" button in sidebar
- Close the browser window
- The system stops automatically

## Next Steps

1. **Monitor Performance**: Use the dashboard to track strategy performance
2. **Analyze Logs**: Review `trading.log` and `trading_structured.jsonl`
3. **Adjust Strategies**: Modify parameters based on performance
4. **Add More Strategies**: Create custom strategies for different symbols
5. **Backtest**: Use historical data to test strategies before live trading

## Example Workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start dashboard
streamlit run dashboard.py

# 3. In the dashboard:
#    - Click "Start Engine"
#    - Monitor portfolio and strategy performance
#    - Analyze metrics and charts
#    - Click "Stop Engine" when done

# 4. Review logs
cat trading.log
# or
python -c "import json; [print(json.loads(line)) for line in open('trading_structured.jsonl')]"
```

## Support

For issues or questions:
1. Check the logs in `trading.log`
2. Review `flow.md` for system architecture
3. Check `README.md` for detailed documentation

