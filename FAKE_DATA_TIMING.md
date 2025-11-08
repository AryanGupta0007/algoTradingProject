# Fake Data Feed Timing

## Overview

The fake data feed generates market data (OHLCV candles) at configurable intervals for testing and paper trading.

## Default Timing

- **Interval**: 60 seconds (1 minute) between candle batches
- **Generation Time**: < 1 millisecond per candle (instant)
- **Multiple Symbols**: All symbols generate candles in quick succession, then wait

## How It Works

```
Time 0s:    Generate RELIANCE candle (instant)
Time 0.001s: Generate TCS candle (instant)
Time 0.002s: Generate INFY candle (instant)
Time 60s:   Wait...
Time 60s:   Generate RELIANCE candle (instant)
Time 60.001s: Generate TCS candle (instant)
Time 60.002s: Generate INFY candle (instant)
Time 120s:  Wait...
... (repeats)
```

## Customizing the Interval

### Method 1: Configuration File

Edit `config.py`:

```python
config = Config()
config.trading.fake_data_interval = 10  # 10 seconds
```

### Method 2: Environment Variable

```python
import os
os.environ['FAKE_DATA_INTERVAL'] = '10'  # 10 seconds
```

Then in `config.py`:
```python
fake_data_interval: int = int(os.getenv('FAKE_DATA_INTERVAL', '60'))
```

### Method 3: Direct Code Modification

In `engine.py`:
```python
self.data_feed = FakeDataFeed(symbols=symbols, interval_seconds=10)  # 10 seconds
```

## Common Intervals

| Interval | Use Case | Description |
|----------|----------|-------------|
| 1 second | Fast testing | Very rapid data generation for quick testing |
| 5 seconds | Quick testing | Faster testing, still manageable |
| 10 seconds | Standard testing | Good balance for testing |
| 30 seconds | Normal testing | Moderate speed |
| 60 seconds | Default | Realistic for paper trading (1-minute candles) |
| 300 seconds | Slow testing | 5-minute candles for slower strategies |

## Example: Fast Testing

For faster testing, set interval to 5 seconds:

```python
from config import Config

config = Config()
config.trading.fake_data_interval = 5  # 5 seconds

engine = TradingEngine(config)
engine.initialize_data_feed(symbols)
# ... rest of setup
```

This will generate candles every 5 seconds instead of 60 seconds, allowing you to test strategies much faster.

## Performance Considerations

- **Generation Speed**: Data generation is extremely fast (< 1ms per candle)
- **CPU Usage**: Minimal - just random number generation and object creation
- **Memory Usage**: Minimal - only stores current price per symbol
- **Thread Overhead**: Runs in background thread, doesn't block main execution

## Real-World Comparison

- **Real Market Data**: Varies (tick-by-tick to 1-minute candles)
- **Fake Data Feed**: Configurable (default 60 seconds = 1-minute candles)
- **Faster Testing**: Use shorter intervals (1-10 seconds) for quick strategy testing
- **Realistic Testing**: Use 60 seconds or longer for realistic paper trading

## Monitoring Data Generation

You can monitor data generation in logs:

```
INFO - OHLCV: {'symbol': 'RELIANCE', 'open': 100.0, 'high': 100.5, ...}
INFO - OHLCV: {'symbol': 'TCS', 'open': 200.0, 'high': 200.3, ...}
```

Each log entry indicates a new candle was generated and processed.

## Tips

1. **Fast Testing**: Use 1-5 second intervals for rapid strategy testing
2. **Realistic Testing**: Use 60+ second intervals for realistic paper trading
3. **Multiple Symbols**: All symbols generate in the same cycle, then wait
4. **Resampling**: Strategies with resampling will accumulate data until resample period is complete

