# Database Module

This module provides SQLite database persistence for the paper trading system.

## Features

- **Persistent Storage**: All trading data is saved to SQLite database
- **Orders & Fills**: Complete order and fill history
- **Positions**: Current and historical positions
- **Metrics**: Daily and strategy-wise performance metrics
- **Equity History**: Complete equity curve tracking
- **Portfolio State**: Portfolio snapshots over time
- **Market Data**: Optional OHLCV data storage

## Database Schema

### Tables

1. **orders** - All orders (submitted, filled, cancelled)
2. **fills** - All order fills
3. **positions** - Current positions
4. **portfolio_state** - Portfolio state snapshots
5. **daily_metrics** - Daily performance metrics
6. **strategy_metrics** - Strategy-wise metrics
7. **equity_history** - Equity curve points
8. **market_data** - OHLCV market data (optional)

## Usage

### Enable Database

Database is enabled by default. To disable:

```python
config = Config()
config.trading.enable_db = False
```

### Custom Database Path

```python
config = Config()
config.trading.db_path = "my_trading.db"
```

### Enable Market Data Storage

```python
config = Config()
config.trading.save_market_data = True  # Can be large!
```

### Query Data

```python
from engine import TradingEngine

engine = TradingEngine(config)
queries = engine.get_database_queries()

# Get all orders
orders = queries['get_orders']()

# Get orders for a strategy
strategy_orders = queries['get_orders'](strategy_id="MA_CROSS_RELIANCE")

# Get fills
fills = queries['get_fills'](strategy_id="MA_CROSS_RELIANCE")

# Get daily metrics
daily_metrics = queries['get_daily_metrics'](
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31)
)

# Get equity history
equity_history = queries['get_equity_history'](limit=1000)
```

## Database File

The database file (`trading.db` by default) contains all persistent data. It's automatically created on first run.

**Note**: Add `*.db` to `.gitignore` to avoid committing database files.

## Benefits

1. **Data Persistence**: Data survives program restarts
2. **Historical Analysis**: Query historical trades and metrics
3. **Reporting**: Generate reports from stored data
4. **Backtesting**: Use historical data for backtesting
5. **Audit Trail**: Complete record of all trading activity

## Performance

- Indexes are created on commonly queried fields
- Efficient queries with proper indexing
- Batch operations for better performance
- Optional market data storage (can be disabled to save space)

