# Paper Trading System

A modular, scalable paper trading system that can run multiple trading strategies simultaneously on multiple symbols in the equity market.

## Features

- **Multi-Strategy Support**: Run multiple trading strategies simultaneously on different symbols
- **Flexible Entry/Exit Conditions**: 
  - Indicator-based conditions (SMA, RSI, EMA, MACD, Bollinger Bands)
  - Conditional exits (price crossovers)
  - SL/TP-based exits (stop-loss, take-profit)
- **TA-Lib Integration**: Industry-standard technical indicators using TA-Lib library
- **Data Resampling**: Resample market data to different timeframes (1min, 5min, 15min, 1H, 1D, etc.)
- **LTP Tracking**: Real-time Last Traded Price (LTP) tracking for all symbols
- **Data Feed Options**:
  - ICICI Breeze API integration for real-time OHLCV data
  - FakeDataFeed generator for testing and paper trading
- **Risk Management**: Built-in RMS (Risk Management System) with position size and exposure limits
- **Comprehensive Logging**: Detailed logging of ticks, orders, fills, RMS checks, and condition evaluations
- **Portfolio Tracking**: Real-time portfolio and position tracking with P&L calculation
- **Advanced Metrics**: 
  - Realized and unrealized P&L
  - Daily performance metrics
  - Strategy-wise performance metrics
  - Win rate, drawdown, and other key metrics
- **Streamlit Dashboard**: Interactive web dashboard for real-time monitoring

## Architecture

The system is built with a modular architecture:

- **`config.py`**: Configuration management
- **`indicators/`**: Technical indicators (SMA, RSI, EMA, MACD)
- **`datafeed/`**: Data feed interfaces (ICICI Breeze API, FakeDataFeed)
- **`order/`**: Order management system
- **`portfolio/`**: Portfolio and position tracking
- **`rms/`**: Risk Management System
- **`strategy/`**: Strategy base class and example strategies
- **`logging/`**: Comprehensive logging system
- **`engine.py`**: Main trading engine that orchestrates all components
- **`main.py`**: Entry point with example usage

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: TA-Lib requires additional installation steps:
- Windows: Download TA-Lib wheel from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib) and install
- Linux/Mac: Install TA-Lib C library first, then pip install TA-Lib

2. Configure the system:
   - For ICICI Breeze API: Set environment variables:
     - `ICICI_API_KEY`
     - `ICICI_API_SECRET`
     - `ICICI_SESSION_TOKEN`
     - `ICICI_ENABLED=true`
   - Or edit `config.py` directly

## Usage

### Basic Usage

```python
from config import Config
from engine import TradingEngine
from strategy import MovingAverageCrossoverStrategy

# Initialize engine
config = Config()
engine = TradingEngine(config)

# Initialize data feed
symbols = ["RELIANCE", "TCS"]
engine.initialize_data_feed(symbols)

# Add strategy
strategy = MovingAverageCrossoverStrategy(
    strategy_id="MA_CROSS_1",
    symbol="RELIANCE",
    fast_period=10,
    slow_period=30,
    quantity=10
)
engine.add_strategy(strategy)

# Start trading
engine.start()
```

### Running the Example

```bash
python main.py
```

### Running the Dashboard

```bash
streamlit run dashboard.py
```

Or use the helper script:
```bash
python run_dashboard.py
```

The dashboard provides:
- Real-time portfolio overview
- LTP tracking for all symbols
- Open positions
- Strategy-wise performance metrics
- Daily performance metrics
- Equity curve visualization
- P&L charts

## Strategy Development

### Creating a Custom Strategy

1. Inherit from `BaseStrategy`:
```python
from strategy.base import BaseStrategy, EntryCondition, ExitCondition
from indicators import SMA, RSI

class MyStrategy(BaseStrategy):
    def _initialize_indicators(self):
        self.indicators["sma"] = SMA(20)
        self.indicators["rsi"] = RSI(14)
    
    def _check_entry_conditions(self) -> bool:
        # Custom entry logic
        return self.indicators["rsi"].current_value < 30
```

### Using Data Resampling

Strategies can use resampled data for different timeframes:

```python
# Add strategy with 5-minute resampling
engine.add_strategy(strategy, resample_timeframe='5min')
```

Supported timeframes: `1min`, `5min`, `15min`, `30min`, `1H`, `4H`, `1D`

2. Define entry and exit conditions:
```python
entry_conditions = [
    EntryCondition(
        indicator_name="rsi",
        operator="<",
        threshold=30.0
    )
]

exit_conditions = [
    ExitCondition(
        condition_type="take_profit",
        take_profit=105.0
    ),
    ExitCondition(
        condition_type="stop_loss",
        stop_loss=95.0
    )
]
```

## Configuration

Edit `config.py` to customize:

- **Initial Capital**: Starting capital for paper trading
- **Commission Rate**: Trading commission (default: 0.1%)
- **Slippage**: Execution slippage (default: 0.01%)
- **Risk Management**: Position size and exposure limits
- **Logging**: Log level and file location

## Logging

The system provides comprehensive logging:

- **Standard Logs**: `trading.log` - Human-readable logs
- **Structured Logs**: `trading_structured.jsonl` - JSON-formatted logs for analysis

Logs include:
- Tick/OHLCV data
- Order submissions and fills
- RMS checks
- Condition evaluations
- Portfolio updates
- Errors

## Portfolio Metrics

The system tracks comprehensive portfolio metrics:

- **Realized P&L**: P&L from closed positions
- **Unrealized P&L**: P&L from open positions
- **Daily Metrics**: Opening/closing equity, high/low equity, daily P&L, win rate
- **Strategy Metrics**: Per-strategy performance, trades, win rate, P&L
- **Equity Curve**: Historical equity tracking
- **Drawdown**: Maximum drawdown calculation
- **Win Rate**: Winning vs losing trades

All metrics are available through the Streamlit dashboard and programmatically via `engine.get_metrics()`.

## Database Persistence

The system uses SQLite database for persistent storage:

- **Orders & Fills**: Complete order and fill history
- **Positions**: Current and historical positions
- **Metrics**: Daily and strategy-wise performance metrics
- **Equity History**: Complete equity curve tracking
- **Portfolio State**: Portfolio snapshots over time
- **Market Data**: Optional OHLCV data storage

### Database Features

- Automatic data persistence on all trades and fills
- Historical data querying
- Data survives program restarts
- Efficient indexing for fast queries
- Optional market data storage (can be disabled)

### Querying Data

```python
# Get database query methods
queries = engine.get_database_queries()

# Get orders for a strategy
orders = queries['get_orders'](strategy_id="MA_CROSS_RELIANCE")

# Get daily metrics
daily_metrics = queries['get_daily_metrics'](
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31)
)

# Get equity history
equity_history = queries['get_equity_history'](limit=1000)
```

See `database/README.md` for more details.

## Risk Management

The RMS (Risk Management System) enforces:

- **Position Size Limits**: Maximum position size as % of capital
- **Total Exposure Limits**: Maximum total exposure as % of capital
- **Cash Availability**: Ensures sufficient cash for buy orders
- **Stop Loss Limits**: Optional max loss per trade

## Example Strategies

### Moving Average Crossover
- Enters when fast MA crosses above slow MA
- Exits when fast MA crosses below slow MA or SL/TP triggers

### RSI Strategy
- Enters when RSI is oversold (< 30)
- Exits when RSI is overbought (> 70) or SL/TP triggers

## API Reference

### BaseStrategy

- `on_data(data)`: Handle incoming market data
- `_check_entry_conditions()`: Check if entry conditions are met
- `_check_exit_conditions()`: Check if exit conditions are met
- `_initialize_indicators()`: Initialize strategy indicators

### TradingEngine

- `add_strategy(strategy)`: Add a trading strategy
- `start()`: Start the trading engine
- `stop()`: Stop the trading engine
- `get_portfolio_summary()`: Get portfolio status
- `get_strategy_status()`: Get strategy status

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

