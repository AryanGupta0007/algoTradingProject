# Database Implementation Guide

## Overview

The paper trading system now includes SQLite database persistence for all trading data. This enables data to survive program restarts and provides historical analysis capabilities.

## Database Schema

### Tables

1. **orders** - All orders (submitted, filled, cancelled)
   - Stores order details, status, prices, strategy_id
   - Indexed on strategy_id, symbol, timestamp

2. **fills** - All order fills
   - Stores fill details, prices, commissions
   - Linked to orders via order_id
   - Indexed on order_id, timestamp

3. **positions** - Current positions
   - Stores position details, average price, stop_loss, take_profit
   - Unique on (symbol, strategy_id)
   - Indexed on symbol, strategy_id

4. **portfolio_state** - Portfolio state snapshots
   - Stores cash, equity, P&L, exposure
   - Timestamped for historical tracking
   - Indexed on timestamp

5. **daily_metrics** - Daily performance metrics
   - Stores daily equity, P&L, trades, win rate
   - Unique on date
   - Indexed on date

6. **strategy_metrics** - Strategy-wise metrics
   - Stores per-strategy performance, trades, P&L
   - Unique on (strategy_id, timestamp)
   - Indexed on strategy_id, timestamp

7. **equity_history** - Equity curve points
   - Stores timestamp and equity value
   - Indexed on timestamp

8. **market_data** - OHLCV market data (optional)
   - Stores OHLCV candles
   - Unique on (symbol, timestamp)
   - Indexed on symbol, timestamp

## Configuration

### Enable/Disable Database

```python
from config import Config

config = Config()
config.trading.enable_db = True  # Default: True
config.trading.db_path = "trading.db"  # Default: "trading.db"
```

### Enable Market Data Storage

```python
config.trading.save_market_data = True  # Default: False (can be large)
```

## Usage

### Automatic Persistence

The database automatically saves:
- All orders when submitted
- All fills when executed
- Positions when created/updated/closed
- Portfolio state on each fill
- Daily metrics when updated
- Strategy metrics when updated
- Equity history on each market data update
- Market data (if enabled) on each OHLCV update

### Querying Data

```python
from engine import TradingEngine
from datetime import date

engine = TradingEngine(config)
queries = engine.get_database_queries()

# Get all orders
orders = queries['get_orders']()

# Get orders for a strategy
strategy_orders = queries['get_orders'](strategy_id="MA_CROSS_RELIANCE")

# Get orders for a symbol
symbol_orders = queries['get_orders'](symbol="RELIANCE")

# Get fills
fills = queries['get_fills'](strategy_id="MA_CROSS_RELIANCE")

# Get current positions
positions = queries['get_positions']()

# Get positions for a strategy
strategy_positions = queries['get_positions'](strategy_id="MA_CROSS_RELIANCE")

# Get daily metrics
daily_metrics = queries['get_daily_metrics'](
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31)
)

# Get strategy metrics
strategy_metrics = queries['get_strategy_metrics'](strategy_id="MA_CROSS_RELIANCE")

# Get equity history
equity_history = queries['get_equity_history'](limit=1000)

# Get market data
market_data = queries['get_market_data'](
    symbol="RELIANCE",
    start_time=datetime(2024, 1, 1),
    end_time=datetime(2024, 12, 31),
    limit=1000
)

# Get portfolio state history
portfolio_states = queries['get_portfolio_state'](limit=100)
```

### Loading from Database

```python
# Load positions and portfolio state from database on startup
engine.load_from_database()
```

## Database File

- **Location**: `trading.db` (configurable via `config.trading.db_path`)
- **Format**: SQLite 3
- **Size**: Grows with trading activity (market data can make it large if enabled)

## Benefits

1. **Data Persistence**: All data survives program restarts
2. **Historical Analysis**: Query historical trades and metrics
3. **Reporting**: Generate reports from stored data
4. **Backtesting**: Use historical data for backtesting
5. **Audit Trail**: Complete record of all trading activity
6. **Performance Analysis**: Analyze strategy performance over time

## Performance Considerations

- Indexes are created on commonly queried fields
- Efficient queries with proper indexing
- Batch operations for better performance
- Market data storage is optional (can be disabled to save space)
- Equity history is stored but memory cache is limited to 1000 points

## Example Queries

### Get Total Trades
```python
fills = queries['get_fills']()
total_trades = len(fills)
```

### Get Strategy Performance
```python
strategy_metrics = queries['get_strategy_metrics'](strategy_id="MA_CROSS_RELIANCE")
latest = strategy_metrics[0] if strategy_metrics else None
if latest:
    print(f"Total P&L: {latest['total_pnl']}")
    print(f"Win Rate: {latest['win_rate'] * 100:.2f}%")
```

### Get Equity Curve
```python
equity_history = queries['get_equity_history'](limit=10000)
import pandas as pd
df = pd.DataFrame(equity_history)
df['timestamp'] = pd.to_datetime(df['timestamp'])
# Plot equity curve
```

### Get Daily P&L
```python
daily_metrics = queries['get_daily_metrics']()
df = pd.DataFrame(daily_metrics)
df['date'] = pd.to_datetime(df['date'])
# Analyze daily performance
```

## Troubleshooting

### Database Locked Error
- Ensure only one instance is accessing the database
- Close database connections properly

### Large Database Size
- Disable market data storage if not needed
- Regularly archive old data
- Use database compression

### Performance Issues
- Ensure indexes are created (automatic on first run)
- Limit query results with `limit` parameter
- Use date ranges for historical queries

## Migration

The database is automatically created on first run. No migration needed.

## Backup

Regularly backup the `trading.db` file to preserve trading history.

```bash
# Backup database
cp trading.db trading_backup_$(date +%Y%m%d).db
```

## See Also

- `database/README.md` - Detailed database documentation
- `database/db_manager.py` - Database manager implementation
- `flow.md` - System flow documentation

