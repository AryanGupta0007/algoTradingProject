# Paper Trading System - Application Flow

This document explains the complete flow of the paper trading system, from initialization to execution and monitoring.

## Table of Contents

1. [System Initialization](#system-initialization)
2. [Data Feed Flow](#data-feed-flow)
3. [Strategy Execution Flow](#strategy-execution-flow)
4. [Order Processing Flow](#order-processing-flow)
5. [Portfolio Management Flow](#portfolio-management-flow)
6. [Metrics Tracking Flow](#metrics-tracking-flow)
7. [Dashboard Flow](#dashboard-flow)
8. [Complete End-to-End Flow](#complete-end-to-end-flow)

---

## System Initialization

### 1. Configuration Loading
```
config.py → Config object
├── ICICI Breeze API settings (if enabled)
├── Trading parameters (capital, commission, slippage)
└── Risk management settings
```

### 2. Trading Engine Initialization
```
TradingEngine.__init__()
├── Create OrderManager (handles order execution)
├── Create Portfolio (tracks positions and cash)
├── Create PortfolioMetrics (tracks performance metrics)
├── Create RiskManagementSystem (RMS) - optional
├── Create TradingLogger (logs all activities)
└── Initialize strategy management structures
    ├── strategies: Dict[strategy_id → Strategy]
    ├── strategy_symbol_map: Dict[symbol → List[strategy_ids]]
    └── strategy_resamplers: Dict[strategy_id → DataResampler]
```

### 3. Data Feed Initialization
```
engine.initialize_data_feed(symbols)
├── Check if ICICI Breeze API is enabled
│   ├── YES → Create ICICIBreezeDataFeed
│   │   ├── Connect to API
│   │   └── Setup WebSocket for real-time data
│   └── NO → Create FakeDataFeed
│       └── Start fake data generation thread
└── Subscribe to symbols
```

### 4. Strategy Addition
```
engine.add_strategy(strategy, resample_timeframe=None)
├── Add strategy to strategies dictionary
├── Map symbol to strategy_id
├── Create DataResampler if resample_timeframe provided
├── Setup strategy callbacks
│   ├── order_callback → engine._on_strategy_order()
│   └── log_callback → engine._on_strategy_log()
└── Initialize strategy indicators
```

---

## Data Feed Flow

### 1. Market Data Reception

#### For ICICI Breeze API:
```
WebSocket → ICICIBreezeDataFeed
├── Receive market data (OHLCV or Tick)
├── Parse data into OHLCVData or TickData object
└── Notify all subscribers
    └── engine._on_market_data()
```

#### For FakeDataFeed:
```
Background Thread → FakeDataFeed
├── Generate random OHLCV candles every interval_seconds
├── Create OHLCVData object
└── Notify all subscribers
    └── engine._on_market_data()
```

### 2. Market Data Processing in Engine

```
engine._on_market_data(data)
├── Update LTP (Last Traded Price)
│   ├── ltp[symbol] = data.close (or data.price for ticks)
│   └── ltp_timestamp[symbol] = data.timestamp
├── Log data (OHLCV or Tick)
├── Update portfolio prices
│   └── portfolio.update_position_price(symbol, price)
├── Update portfolio metrics
│   └── portfolio_metrics.update(timestamp)
├── Sync strategy positions with portfolio
│   └── _sync_strategy_positions(symbol)
├── Process strategies for this symbol
│   ├── For each strategy trading this symbol:
│   │   ├── Check if strategy needs resampling
│   │   │   ├── YES → Add to resampler buffer
│   │   │   │   └── Get resampled data if candle complete
│   │   │   └── NO → Pass data directly
│   │   └── strategy.on_data(resampled_data or original_data)
│   └── Strategy evaluates entry/exit conditions
└── Check position exits (stop loss/take profit)
    └── _check_position_exits(symbol, price)
```

---

## Strategy Execution Flow

### 1. Data Reception in Strategy

```
strategy.on_data(data)
├── Store current OHLCV data
├── Update current_price = data.close
└── Update indicators
    └── _update_indicators(data)
        ├── For each indicator:
        │   ├── If indicator has update_ohlcv() → Call with OHLCV
        │   └── Else if indicator has update() → Call with close price
        └── Indicators calculate new values (using TA-Lib)
```

### 2. Entry Condition Evaluation

```
strategy._check_entry_conditions()
├── If has_position → Skip (don't enter if already in position)
├── For each entry_condition:
│   ├── Get indicator value
│   ├── Evaluate condition based on operator:
│   │   ├── '>' → indicator_value > threshold
│   │   ├── '<' → indicator_value < threshold
│   │   ├── 'crossover' → Check if crossed above threshold
│   │   └── 'crossunder' → Check if crossed below threshold
│   └── All conditions must be True
└── If all conditions met → _enter_position()
```

### 3. Entry Position Creation

```
strategy._enter_position()
├── Create Order object
│   ├── symbol = strategy.symbol
│   ├── side = OrderSide.BUY (default)
│   ├── quantity = strategy.quantity
│   ├── order_type = OrderType.MARKET
│   ├── strategy_id = strategy.strategy_id
│   ├── stop_loss = _get_stop_loss() (if configured)
│   └── take_profit = _get_take_profit() (if configured)
└── Call order_callback(order)
    └── engine._on_strategy_order(order)
```

### 4. Exit Condition Evaluation

```
strategy._check_exit_conditions()
├── If not has_position → Skip
├── For each exit_condition:
│   ├── Check condition_type:
│   │   ├── 'stop_loss' → Check if price <= stop_loss
│   │   ├── 'take_profit' → Check if price >= take_profit
│   │   ├── 'indicator' → Evaluate indicator condition
│   │   └── 'timeout' → Check if time elapsed
│   └── If condition met → Return ExitReason
└── If exit condition met → _exit_position(reason)
```

### 5. Exit Position Creation

```
strategy._exit_position(reason)
├── Create Order object
│   ├── symbol = strategy.symbol
│   ├── side = OrderSide.SELL
│   ├── quantity = strategy.quantity
│   ├── order_type = OrderType.MARKET
│   └── strategy_id = strategy.strategy_id
└── Call order_callback(order)
    └── engine._on_strategy_order(order)
```

---

## Order Processing Flow

### 1. Order Submission

```
engine._on_strategy_order(order)
├── Risk Management Check (if RMS enabled)
│   ├── rms.check_order(order, portfolio, current_price)
│   ├── Check position size limits
│   ├── Check total exposure limits
│   ├── Check cash availability (for buy orders)
│   └── If rejected → Log and return (order not submitted)
├── Submit order to OrderManager
│   └── order_manager.submit_order(order)
│       ├── Set order.status = OrderStatus.SUBMITTED
│       ├── Store order in orders dictionary
│       └── Return True if successful
└── Log order submission
```

### 2. Order Execution Loop

```
Background Thread → engine._order_execution_loop()
├── While engine.is_running:
│   ├── Get all open orders
│   │   └── order_manager.get_open_orders()
│   ├── For each open order:
│   │   ├── Get current market price
│   │   │   └── _get_current_price(order.symbol)
│   │   ├── Execute order
│   │   │   └── order_manager.execute_order(order_id, current_price)
│   │   │       ├── Check order type and execution conditions
│   │   │       ├── Calculate execution price (with slippage)
│   │   │       ├── Create Fill object
│   │   │       ├── Update order status (FILLED or PARTIALLY_FILLED)
│   │   │       ├── Update order average_price
│   │   │       └── Return Fill object
│   │   └── If fill created → Notify fill callbacks
│   │       └── engine._on_fill(fill)
│   └── Sleep for 500ms
```

### 3. Order Execution Logic

```
order_manager.execute_order(order_id, current_price)
├── Get order from orders dictionary
├── Check if order can be executed based on type:
│   ├── MARKET → Execute immediately at current_price
│   ├── LIMIT → Execute if price reaches limit
│   │   ├── BUY limit → Execute if current_price <= limit_price
│   │   └── SELL limit → Execute if current_price >= limit_price
│   ├── STOP → Execute if stop price is reached
│   │   ├── BUY stop → Execute if current_price >= stop_price
│   │   └── SELL stop → Execute if current_price <= stop_price
│   └── STOP_LIMIT → Execute if stop reached AND limit condition met
├── Apply slippage
│   ├── BUY → execution_price *= (1 + slippage)
│   └── SELL → execution_price *= (1 - slippage)
├── Calculate commission
│   └── commission = execution_price * quantity * commission_rate
├── Create Fill object
└── Return Fill
```

---

## Portfolio Management Flow

### 1. Fill Processing

```
engine._on_fill(fill)
├── Log fill
├── Get order associated with fill
├── Calculate P&L before portfolio update
│   └── old_realized = portfolio.total_realized_pnl
├── Process fill in portfolio
│   └── portfolio.process_fill(fill, order)
│       ├── Get or create position for symbol
│       ├── Update position based on fill side:
│       │   ├── BUY → Add quantity, subtract cash
│       │   │   ├── Update average_price (weighted)
│       │   │   ├── Set stop_loss and take_profit from order
│       │   │   └── Set strategy_id
│       │   └── SELL → Reduce quantity, add cash
│       │       ├── Update average_price
│       │       └── If position closed → Clear stop_loss/take_profit
│       ├── Calculate realized P&L if position closed/reduced
│       │   ├── For closed long: (sell_price - avg_price) * quantity
│       │   └── For closed short: (avg_price - buy_price) * quantity
│       └── Update total_realized_pnl and total_commission
├── Calculate trade P&L
│   └── pnl = portfolio.total_realized_pnl - old_realized
├── Record trade in metrics
│   └── portfolio_metrics.record_trade(fill, strategy_id, pnl)
├── Update strategy position state
│   ├── Get position from portfolio
│   ├── If position exists and not flat:
│   │   ├── strategy.has_position = True
│   │   ├── strategy.position_entry_price = position.average_price
│   │   └── strategy.position_entry_time = position.opened_at
│   └── Else:
│       ├── strategy.has_position = False
│       ├── strategy.position_entry_price = None
│       └── strategy.position_entry_time = None
├── Update portfolio metrics
│   └── portfolio_metrics.update()
└── Log portfolio update
```

### 2. Position Price Updates

```
portfolio.update_position_price(symbol, price)
├── Get position for symbol
├── Update position.current_price = price
└── Update position.last_updated = datetime.now()
    └── Position.unrealized_pnl() automatically recalculates
```

### 3. Position Exit Checks

```
engine._check_position_exits(symbol, price)
├── Get position for symbol
├── If position exists and not flat:
│   ├── Check stop loss
│   │   ├── position.should_stop_loss()
│   │   │   ├── For long: current_price <= stop_loss
│   │   │   └── For short: current_price >= stop_loss
│   │   └── If triggered → Create exit order
│   └── Check take profit
│       ├── position.should_take_profit()
│       │   ├── For long: current_price >= take_profit
│       │   └── For short: current_price <= take_profit
│       └── If triggered → Create exit order
└── Submit exit order to order_manager
```

---

## Metrics Tracking Flow

### 1. Daily Metrics Update

```
portfolio_metrics.update(timestamp)
├── Get current date from timestamp
├── Check if new day
│   ├── YES → Finalize previous day metrics
│   │   └── Set closing_equity
│   └── Initialize new day metrics
│       ├── opening_equity = last_equity
│       └── peak_equity = opening_equity
├── Update current day metrics
│   ├── closing_equity = portfolio.total_equity()
│   ├── high_equity = max(high_equity, current_equity)
│   ├── low_equity = min(low_equity, current_equity)
│   ├── unrealized_pnl = portfolio.total_unrealized_pnl()
│   ├── realized_pnl = portfolio.total_realized_pnl
│   ├── total_pnl = portfolio.total_pnl()
│   ├── commissions = portfolio.total_commission
│   ├── Update peak_equity if current > peak
│   └── Update max_drawdown = max(max_drawdown, peak - current)
├── Track equity history
│   └── equity_history.append((timestamp, equity))
└── Update strategy metrics
    └── _update_strategy_metrics()
```

### 2. Trade Recording

```
portfolio_metrics.record_trade(fill, strategy_id, pnl)
├── Get current date
├── Update daily metrics
│   ├── daily.trades_count += 1
│   ├── If pnl > 0 → daily.winning_trades += 1
│   └── If pnl < 0 → daily.losing_trades += 1
└── Update strategy metrics
    ├── Get or create strategy metrics
    ├── strategy.total_trades += 1
    ├── strategy.realized_pnl += pnl
    ├── strategy.commissions += fill.commission
    ├── If pnl > 0 → strategy.winning_trades += 1
    └── If pnl < 0 → strategy.losing_trades += 1
```

### 3. Strategy Metrics Update

```
portfolio_metrics._update_strategy_metrics()
├── For each open position:
│   ├── Get strategy_id from position
│   ├── Get or create strategy metrics
│   ├── Update unrealized_pnl = position.unrealized_pnl()
│   ├── Update current_position_value
│   └── Update total_pnl = realized_pnl + unrealized_pnl
```

---

## Dashboard Flow

### 1. Dashboard Initialization

```
Streamlit App Start → dashboard.py
├── Set page config
├── Initialize session state
│   ├── engine = None
│   ├── config = Config()
│   └── is_running = False
└── Render dashboard
```

### 2. User Interaction

```
User clicks "Start Engine"
├── get_engine() → Create TradingEngine if not exists
├── Initialize data feed
├── Add strategies
│   ├── MovingAverageCrossoverStrategy
│   └── RSIStrategy
├── Start engine
│   └── engine.start()
│       ├── Connect data feed
│       ├── Start all strategies
│       ├── Subscribe to symbols
│       └── Start order execution thread
└── Set is_running = True
```

### 3. Data Refresh Loop

```
Auto-refresh (if enabled)
├── Every refresh_interval seconds:
│   ├── Get portfolio summary
│   │   └── engine.get_portfolio_summary()
│   ├── Get metrics
│   │   └── engine.get_metrics()
│   │       ├── Daily metrics
│   │       ├── Strategy metrics
│   │       ├── Summary
│   │       └── Equity curve
│   ├── Get LTP data
│   │   └── engine.get_ltp()
│   └── Get strategy status
│       └── engine.get_strategy_status()
└── Update dashboard display
    ├── Portfolio Overview
    ├── LTP Table
    ├── Open Positions
    ├── Strategy Performance
    ├── Daily Performance
    └── Summary Metrics
```

### 4. Dashboard Display

```
Dashboard Rendering
├── Portfolio Overview Section
│   ├── Total Equity
│   ├── Total P&L
│   ├── Realized P&L
│   ├── Unrealized P&L
│   ├── Cash
│   ├── Exposure
│   └── Commission
├── LTP Section
│   └── Display LTP table for all symbols
├── Positions Section
│   └── Display open positions table
├── Strategy Performance Section
│   ├── Strategy metrics table
│   └── Strategy P&L chart
├── Daily Performance Section
│   ├── Daily metrics table
│   ├── Equity curve chart
│   └── Daily P&L chart
└── Summary Metrics Section
    ├── Total trades
    ├── Win rate
    ├── Winning/losing trades
    ├── Return %
    ├── Max drawdown
    └── Avg trade P&L
```

---

## Complete End-to-End Flow

### Example: Complete Trade Execution Flow

```
1. System Start
   ├── TradingEngine initialized
   ├── Data feed connected (FakeDataFeed or ICICI)
   ├── Strategies added and started
   └── Dashboard launched

2. Market Data Arrives
   ├── FakeDataFeed generates OHLCV candle
   └── engine._on_market_data(ohlcv_data)

3. Data Processing
   ├── Update LTP: ltp["RELIANCE"] = 2500.0
   ├── Update portfolio prices
   ├── Update metrics
   └── Process strategies

4. Strategy Evaluation (MovingAverageCrossoverStrategy)
   ├── Update indicators (SMA(10), SMA(30))
   ├── Check entry conditions
   │   ├── Fast SMA = 2495.0
   │   ├── Slow SMA = 2490.0
   │   └── Fast crossed above Slow → ENTRY SIGNAL
   └── _enter_position()
       └── Create BUY order (quantity=10, market order)

5. Order Processing
   ├── RMS check → PASSED
   ├── Submit order → OrderManager
   ├── Order status = SUBMITTED
   └── Log order

6. Order Execution
   ├── Order execution loop picks up order
   ├── Get current price = 2500.0
   ├── Execute order (market order)
   │   ├── Apply slippage → 2500.25
   │   ├── Calculate commission → 25.00
   │   └── Create Fill
   └── Notify fill callback

7. Portfolio Update
   ├── Process fill
   │   ├── Create position: RELIANCE, qty=10, avg_price=2500.25
   │   ├── Update cash: 100000 - 25025 = 74975
   │   └── Set stop_loss and take_profit
   ├── Update strategy state
   │   ├── has_position = True
   │   └── position_entry_price = 2500.25
   ├── Record trade in metrics
   └── Update portfolio metrics

8. Continuous Monitoring
   ├── New market data arrives
   ├── Update position price
   ├── Check exit conditions
   │   ├── Stop loss check
   │   ├── Take profit check
   │   └── Indicator-based exit check
   └── If exit condition met → Create SELL order

9. Exit Trade
   ├── SELL order created
   ├── Order executed at 2600.0 (take profit hit)
   ├── Process fill
   │   ├── Close position
   │   ├── Calculate realized P&L = (2600 - 2500.25) * 10 = 997.50
   │   ├── Update cash: 74975 + 25975 = 100950
   │   └── Update total_realized_pnl
   ├── Update strategy state
   │   └── has_position = False
   └── Update metrics
       ├── Record winning trade
       ├── Update daily metrics
       └── Update strategy metrics

10. Dashboard Update
    ├── Refresh data every 5 seconds
    ├── Display updated portfolio
    ├── Show trade in strategy metrics
    ├── Update P&L charts
    └── Update equity curve
```

---

## Key Components Interaction

```
TradingEngine (Orchestrator)
├── DataFeed → Provides market data
├── Strategies → Generate trading signals
├── OrderManager → Executes orders
├── Portfolio → Tracks positions and cash
├── PortfolioMetrics → Tracks performance
├── RiskManagementSystem → Validates orders
└── TradingLogger → Logs all activities

Data Flow:
Market Data → Engine → Strategies → Orders → Portfolio → Metrics → Dashboard
```

---

## Threading Model

```
Main Thread
├── TradingEngine initialization
├── Strategy execution (synchronous)
└── Dashboard (Streamlit)

Background Threads:
├── DataFeed Thread (FakeDataFeed or WebSocket)
│   └── Generates/receives market data
├── Order Execution Thread
│   └── Executes pending orders
└── Dashboard Auto-refresh (if enabled)
    └── Updates dashboard periodically
```

---

## Error Handling

```
Error Handling Points:
├── Data feed connection errors → Log and fallback to FakeDataFeed
├── Order execution errors → Log and continue
├── Strategy errors → Log and continue (don't crash system)
├── Portfolio update errors → Log and rollback if possible
└── Dashboard errors → Display error message, continue operation
```

---

## Performance Considerations

```
Optimizations:
├── Indicator calculations use efficient TA-Lib library
├── Data resampling buffers data to minimize calculations
├── Portfolio updates are incremental (only changed positions)
├── Metrics updates are batched where possible
└── Dashboard refresh can be configured (auto-refresh interval)
```

---

This flow document provides a comprehensive overview of how the paper trading system works from start to finish. Each component has a specific role and interacts with other components in a well-defined manner to create a robust trading system.

