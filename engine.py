"""
Main trading engine that orchestrates all components.
"""
import threading
import time
from typing import Dict, List, Optional
from datetime import datetime
from datetime import timedelta
import random
from app_config import Config
from datafeed import DataFeed, ICICIBreezeDataFeed, FakeDataFeed, OHLCVData, TickData
from datafeed.resampler import DataResampler
from order import OrderManager, Order, OrderSide, OrderType
from portfolio import Portfolio
from portfolio.metrics import PortfolioMetrics
from rms import RiskManagementSystem
from strategy import BaseStrategy
from trading_log.trading_logger import TradingLogger
from database import DatabaseManager
import logging

logger = logging.getLogger(__name__)


class TradingEngine:
    """Main trading engine"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        
        # Initialize components
        self.data_feed: Optional[DataFeed] = None
        self.order_manager = OrderManager(
            commission_rate=self.config.trading.commission_rate,
            slippage=self.config.trading.slippage
        )
        self.portfolio = Portfolio(initial_capital=self.config.trading.initial_capital)
        self.portfolio_metrics = PortfolioMetrics(self.portfolio)
        self.rms = RiskManagementSystem(
            max_position_size=self.config.trading.max_position_size,
            max_total_exposure=self.config.trading.max_total_exposure
        ) if self.config.trading.enable_rms else None
        
        self.trading_logger = TradingLogger(
            log_file=self.config.trading.log_file,
            log_level=self.config.trading.log_level
        )
        
        # Database manager (if enabled)
        self.db_manager: Optional[DatabaseManager] = None
        if self.config.trading.enable_db:
            self.db_manager = DatabaseManager(db_path=self.config.trading.db_path)
            logger.info(f"Database manager initialized: {self.config.trading.db_path}")
            # Load any existing positions/portfolio state from DB at startup
            try:
                # Reconcile trades -> positions first
                try:
                    upserts = self.db_manager.reconcile_open_trades_to_positions()
                    if upserts:
                        logger.info(f"Reconciled {upserts} open trades into positions before load")
                except Exception:
                    logger.exception("Failed to reconcile open trades into positions")

                self.load_from_database()
                logger.info("Loaded existing portfolio state from database on initialization")
            except Exception:
                logger.exception("Failed to load existing portfolio state from database on initialization")
        
        # Strategy management
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_symbol_map: Dict[str, List[str]] = {}  # symbol -> strategy_ids
        self.strategy_resamplers: Dict[str, DataResampler] = {}  # strategy_id -> resampler
        
        # LTP tracking
        self.ltp: Dict[str, float] = {}  # symbol -> last traded price
        self.ltp_timestamp: Dict[str, datetime] = {}  # symbol -> last update time
        
        # State
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        # Guard to prevent trading during historical warm-up
        self.is_bootstrapping: bool = False
        
        # Setup callbacks
        self.order_manager.add_fill_callback(self._on_fill)
        self.data_feed = None
    
    def initialize_data_feed(self, symbols: List[str]):
        """Initialize data feed based on configuration"""
        # Convert to set to remove duplicates
        unique_symbols = list(set(symbols))
        
        if self.config.icici.enabled and self.config.icici.api_key:
            # Use ICICI Breeze API
            self.data_feed = ICICIBreezeDataFeed(
                api_key=self.config.icici.api_key,
                api_secret=self.config.icici.api_secret,
                session_token=self.config.icici.session_token,
                base_url=self.config.icici.base_url,
                websocket_url=self.config.icici.websocket_url
            )
            logger.info("Initialized ICICI Breeze data feed")
        else:
            # Use fake data feed
            interval = self.config.trading.fake_data_interval
            self.data_feed = FakeDataFeed(symbols=unique_symbols, interval_seconds=interval)
            logger.info(f"Initialized Fake data feed with {interval} second interval")
        
        # Subscribe to data feed
        self.data_feed.add_subscriber(self._on_market_data)
    
        # Bootstrap with recent 1-minute historical data so indicators can warm up.
        # If the data feed offers a `fetch_historical` method, use it; otherwise
        # generate synthetic recent 1-min bars (useful for the FakeDataFeed).
        try:
            self.is_bootstrapping = True
            self._bootstrap_historical_data(unique_symbols, minutes=30)
        except Exception as e:
            logger.warning(f"Failed to bootstrap historical data: {e}")
        finally:
            self.is_bootstrapping = False
    def add_strategy(self, strategy: BaseStrategy, resample_timeframe: Optional[str] = None):
        """
        Add a trading strategy.
        
        Args:
            strategy: Strategy to add
            resample_timeframe: Optional timeframe for data resampling (e.g., '5min', '15min', '1H')
        """
        if strategy.strategy_id in self.strategies:
            raise ValueError(f"Strategy {strategy.strategy_id} already exists")
        
        self.strategies[strategy.strategy_id] = strategy
        
        # Map symbol to strategy - support multiple strategies per symbol
        if strategy.symbol not in self.strategy_symbol_map:
            self.strategy_symbol_map[strategy.symbol] = []
        
        # Only append if strategy_id not already in list
        if strategy.strategy_id not in self.strategy_symbol_map[strategy.symbol]:
            self.strategy_symbol_map[strategy.symbol].append(strategy.strategy_id)
            logger.info(f"Mapped strategy {strategy.strategy_id} to symbol {strategy.symbol} ({len(self.strategy_symbol_map[strategy.symbol])} strategies total)")
        
        # Create resampler if needed
        if resample_timeframe:
            self.strategy_resamplers[strategy.strategy_id] = DataResampler(timeframe=resample_timeframe)
            logger.info(f"Created resampler for strategy {strategy.strategy_id} with timeframe {resample_timeframe}")
        
        # Set up strategy callbacks
        strategy.set_order_callback(self._on_strategy_order)
        # Forward strategy logs to the TradingSystem logger via a log callback
        try:
            strategy.set_log_callback(self._on_strategy_log)
        except Exception:
            pass
        # Provide strategy with engine's TradingLogger for structured condition logs
        try:
            strategy.set_trading_logger(self.trading_logger)
        except Exception:
            # If strategy doesn't support structured logger, ignore
            pass
        
        # Subscribe to symbol data feed if it's new
        if self.data_feed and strategy.symbol not in self.data_feed.symbols:
            logger.info(f"Subscribing to new symbol {strategy.symbol} for strategy {strategy.strategy_id}")
            self.data_feed.subscribe([strategy.symbol], self._on_market_data)
        
        logger.info(f"Added strategy: {strategy.strategy_id} for symbol: {strategy.symbol}")

    def _on_strategy_log(self, message: str, level: str = "INFO"):
        """Forward strategy log messages into the TradingSystem logger with proper level."""
        try:
            lvl = getattr(self.trading_logger.logger, level.lower(), None)
            if callable(lvl):
                lvl(message)
            else:
                # Fallback
                self.trading_logger.logger.info(message)
        except Exception:
            # Absolute fallback to module logger if trading_logger not available
            getattr(logger, level.lower())(message)
    
    def remove_strategy(self, strategy_id: str):
        """Remove a trading strategy"""
        if strategy_id not in self.strategies:
            return
        
        strategy = self.strategies[strategy_id]
        symbol = strategy.symbol
        
        # Remove from symbol map
        if symbol in self.strategy_symbol_map:
            if strategy_id in self.strategy_symbol_map[symbol]:
                self.strategy_symbol_map[symbol].remove(strategy_id)
            if not self.strategy_symbol_map[symbol]:
                del self.strategy_symbol_map[symbol]
        
        # Stop strategy
        strategy.stop()
        del self.strategies[strategy_id]
        
        logger.info(f"Removed strategy: {strategy_id}")
    
    def start(self):
        """Start the trading engine"""
        if self.is_running:
            logger.warning("Trading engine is already running")
            return
        
        if not self.data_feed:
            raise ValueError("Data feed not initialized")
        
        # Connect to data feed
        if not self.data_feed.connect():
            raise RuntimeError("Failed to connect to data feed")
        
        # Get unique list of symbols with active strategies
        unique_symbols = list(set(self.strategy_symbol_map.keys()))
        if not unique_symbols:
            logger.warning("No symbols mapped to strategies. Engine will start but no data will be received.")
        
        # Log mapped strategies per symbol
        for symbol in unique_symbols:
            strategy_ids = self.strategy_symbol_map.get(symbol, [])
            logger.info(f"Symbol {symbol} mapped to {len(strategy_ids)} strategies: {', '.join(strategy_ids)}")
        
        # Start all strategies
        for strategy in self.strategies.values():
            strategy.start()
        
        # After starting strategies, sync their position state from loaded portfolio
        try:
            for sym, pos in list(self.portfolio.positions.items()):
                if pos and not pos.is_flat():
                    self._sync_strategy_positions(sym)
        except Exception:
            logger.exception("Failed syncing strategy positions from loaded portfolio")

        # Subscribe to unique symbols only
        if unique_symbols:
            self.data_feed.subscribe(unique_symbols, self._on_market_data)
        
        self.is_running = True
        
        # Start order execution thread
        self.thread = threading.Thread(target=self._order_execution_loop, daemon=True)
        self.thread.start()
        
        logger.info("Trading engine started")
        self.trading_logger.logger.info("Trading engine started")
        # Start periodic portfolio snapshots
        try:
            self.portfolio_metrics.start_periodic_snapshot(self.trading_logger, interval_seconds=self.config.trading.portfolio_snapshot_interval)
            logger.debug(f"Started portfolio snapshot thread (interval={self.config.trading.portfolio_snapshot_interval}s)")
        except Exception:
            logger.exception("Failed to start portfolio snapshot thread")

    def _bootstrap_historical_data(self, symbols: List[str], minutes: int = 150):
        """Fetch or generate recent 1-minute OHLCV history for each symbol.

        This helps indicators warm up before live streaming begins. The method
        prefers a data feed provided `fetch_historical(symbol, timeframe, limit)`
        if available. Otherwise, for the FakeDataFeed we synthesize simple
        candles by performing a small random walk from a base price.
        """
        if not symbols:
            return

        now = datetime.now()

        # If the data feed implements a historical fetcher, prefer that.
        if hasattr(self.data_feed, 'fetch_historical') and callable(getattr(self.data_feed, 'fetch_historical')):
            logger.info("Bootstrapping history using data_feed.fetch_historical()")
            for symbol in symbols:
                try:
                    raw_hist = self.data_feed.fetch_historical(symbol, timeframe='1min', limit=minutes)
                    try:
                        cnt = len(raw_hist) if raw_hist is not None else 0
                    except Exception:
                        cnt = 0
                    logger.info(f"Bootstrapped {cnt} historical bars for {symbol}")
                    if not raw_hist:
                        continue
                    # Assume raw_hist is iterable of OHLCVData or dicts convertible to OHLCVData
                    for item in raw_hist:
                        if isinstance(item, dict):
                            ohlcv = OHLCVData(
                                symbol=symbol,
                                open=float(item.get('open')),
                                high=float(item.get('high')),
                                low=float(item.get('low')),
                                close=float(item.get('close')),
                                volume=int(item.get('volume', 0)),
                                timestamp=datetime.fromisoformat(item.get('timestamp')) if isinstance(item.get('timestamp'), str) else item.get('timestamp')
                            )
                        else:
                            ohlcv = item
                        # Feed historical bar into engine in chronological order
                        self._on_market_data(ohlcv)
                except Exception as e:
                    logger.warning(f"Error fetching historical for {symbol}: {e}")
            return

        # Fallback: synthesize simple history (useful for FakeDataFeed)
        logger.info("Bootstrapping synthetic 1-min history for symbols")
        for symbol in symbols:
            # Determine a base price if available from the feed, else default
            base_price = 100.0
            try:
                if hasattr(self.data_feed, 'price_data') and isinstance(getattr(self.data_feed, 'price_data'), dict):
                    base_price = float(self.data_feed.price_data.get(symbol, base_price))
            except Exception:
                base_price = 100.0

            # Create `minutes` bars ending at `now` (oldest -> newest)
            price = base_price
            bars = []
            for i in range(minutes, 0, -1):
                # Timestamp for this bar
                ts = now - timedelta(minutes=i)
                # Small random walk
                change_pct = random.gauss(0.0, 0.001)
                new_price = price * (1 + change_pct)
                open_price = price
                close_price = new_price
                high_price = max(open_price, close_price) * (1 + abs(random.gauss(0, 0.001)))
                low_price = min(open_price, close_price) * (1 - abs(random.gauss(0, 0.001)))
                volume = random.randint(100, 2000)
                ohlcv = OHLCVData(
                    symbol=symbol,
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                    timestamp=ts
                )
                bars.append(ohlcv)
                price = close_price

            # Inject bars in chronological order so indicators receive them correctly
            for bar in bars:
                try:
                    logger.debug(f"Bootstrapping bar for {symbol}: {bar.timestamp} O={bar.open:.2f} C={bar.close:.2f}")
                    self._on_market_data(bar)
                except Exception as e:
                    logger.warning(f"Failed to inject bootstrap bar for {symbol}: {e}")
    
    def stop(self):
        """Stop the trading engine"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop all strategies
        for strategy in self.strategies.values():
            strategy.stop()
        
        # Disconnect data feed
        if self.data_feed:
            self.data_feed.disconnect()
        
        # Wait for thread to finish
        if self.thread:
            self.thread.join(timeout=5)
        
        # Stop periodic snapshots
        try:
            self.portfolio_metrics.stop_periodic_snapshot()
            logger.debug("Stopped portfolio snapshot thread")
        except Exception:
            logger.exception("Failed to stop portfolio snapshot thread")
        
        # Close logger
        self.trading_logger.close()
        
        # Close database
        if self.db_manager:
            self.db_manager.close()
        
        logger.info("Trading engine stopped")
    
    def _on_market_data(self, data):
        """Handle incoming market data"""
        logger.info(f"[Engine] Received market data: {type(data).__name__} for {data.symbol if hasattr(data, 'symbol') else 'unknown'}")
        
        if isinstance(data, OHLCVData):
            logger.debug(f"[Engine] Processing OHLCV: {data.symbol} @ {data.timestamp}")
            
            # Update LTP
            self.ltp[data.symbol] = data.close
            self.ltp_timestamp[data.symbol] = data.timestamp
            logger.debug(f"[Engine] Updated LTP for {data.symbol}: {data.close:.2f}")
            # Update trades LTP in DB
            if self.db_manager:
                try:
                    self.db_manager.update_trades_ltp(data.symbol, data.close)
                except Exception as e:
                    logger.error(f"Error updating trades LTP: {e}")
            # Structured trade LTP log
            try:
                self.trading_logger.log_trade_ltp_update(data.symbol, float(data.close))
            except Exception:
                logger.debug("Failed to log trade LTP update (OHLCV)")
            
            # Log OHLCV data
            self.trading_logger.log_ohlcv(data.to_dict())
            
            # Update portfolio prices
            logger.debug(f"[Engine] Updating portfolio prices for {data.symbol}")
            self.portfolio.update_position_price(data.symbol, data.close)
            
            # Update portfolio metrics
            logger.debug(f"[Engine] Updating portfolio metrics")
            self.portfolio_metrics.update(data.timestamp)
            
            # Save equity history point
            if self.db_manager:
                try:
                    equity = self.portfolio.total_equity()
                    self.db_manager.save_equity_point(data.timestamp, equity)
                    logger.debug(f"[Engine] Saved equity point to database: {equity:.2f}")
                except Exception as e:
                    logger.error(f"Error saving equity point to database: {e}")
            
            # Save market data if enabled
            if self.db_manager and self.config.trading.save_market_data:
                try:
                    self.db_manager.save_market_data(data.symbol, data.to_dict())
                    logger.debug(f"[Engine] Saved market data to database")
                except Exception as e:
                    logger.error(f"Error saving market data to database: {e}")
            
            # Get all strategies for this symbol
            strategy_ids = self.strategy_symbol_map.get(data.symbol, [])

            # During bootstrap, do not run strategies or exits
            if self.is_bootstrapping:
                logger.debug(f"[Engine] Bootstrapping active; skipping strategy execution for {data.symbol}")
                return

            if strategy_ids:
                logger.info(f"[Engine] Processing {len(strategy_ids)} strategies for {data.symbol}: {strategy_ids}")
                
                # Update strategy positions based on portfolio state
                self._sync_strategy_positions(data.symbol)
                
                # Process each strategy
                for strategy_id in strategy_ids:
                    strategy = self.strategies.get(strategy_id)
                    if not strategy:
                        logger.warning(f"[Engine] Strategy {strategy_id} not found (mapped to {data.symbol})")
                        continue
                        
                    logger.debug(f"[Engine] Processing strategy {strategy_id} for {data.symbol}")
                    
                    # Check if strategy needs resampling
                    if strategy_id in self.strategy_resamplers:
                        resampler = self.strategy_resamplers[strategy_id]
                        logger.debug(f"[Engine] Adding data to resampler for {strategy_id}")
                        resampler.add_data(data.symbol, data)
                        resampled_data = resampler.get_resampled_data(data.symbol)
                        if resampled_data:
                            logger.info(f"[Engine] Resampled data ready for {strategy_id}, sending to strategy")
                            strategy.on_data(resampled_data)
                        else:
                            logger.debug(f"[Engine] Resampled data not ready yet for {strategy_id} (incomplete candle)")
                    else:
                        logger.debug(f"[Engine] Sending data directly to strategy {strategy_id}")
                        strategy.on_data(data)
            else:
                logger.debug(f"[Engine] No strategies registered for symbol {data.symbol}")
            
            # Check position exits for all strategies
            logger.debug(f"[Engine] Checking position exits for {data.symbol}")
            self._check_position_exits(data.symbol, data.close)
        
        elif isinstance(data, TickData):
            logger.debug(f"[Engine] Processing Tick: {data.symbol} @ {data.timestamp}, price={data.price:.2f}")
            
            # Update LTP for tick data
            self.ltp[data.symbol] = data.price
            self.ltp_timestamp[data.symbol] = data.timestamp
            logger.debug(f"[Engine] Updated LTP for {data.symbol}: {data.price:.2f}")
            # Update trades LTP in DB
            if self.db_manager:
                try:
                    self.db_manager.update_trades_ltp(data.symbol, data.price)
                except Exception as e:
                    logger.error(f"Error updating trades LTP: {e}")
            # Structured trade LTP log
            try:
                self.trading_logger.log_trade_ltp_update(data.symbol, float(data.price))
            except Exception:
                logger.debug("Failed to log trade LTP update (Tick)")
            
            # Log tick data
            self.trading_logger.log_tick(data.to_dict())
            
            # Update portfolio prices
            logger.debug(f"[Engine] Updating portfolio prices for {data.symbol}")
            self.portfolio.update_position_price(data.symbol, data.price)
            
            # Check for stop loss / take profit on positions
            logger.debug(f"[Engine] Checking position exits for {data.symbol}")
            self._check_position_exits(data.symbol, data.price)
    
    def _on_strategy_order(self, order: Order):
        """Handle order from strategy"""
        # Ignore any orders during historical bootstrap
        if getattr(self, 'is_bootstrapping', False):
            logger.info(f"[Engine] Ignoring order during bootstrap from {order.strategy_id}: {order.side.value} {order.quantity} {order.symbol} @ {order.order_type.value}")
            return
        logger.info(f"[Engine] Received order from strategy {order.strategy_id}: {order.side.value} {order.quantity} {order.symbol} @ {order.order_type.value}")
        # Enforce one-position-per (symbol, strategy) and forbid stacking; only owner may operate while a position exists
        try:
            position = self.portfolio.get_position(order.symbol)
            if position and not position.is_flat():
                owner_id = getattr(position, 'strategy_id', '') or ''
                # If order is from a non-owner strategy, block any operation (both sides)
                if owner_id and owner_id != (order.strategy_id or ''):
                    logger.warning(f"[Engine] Blocking order from non-owner strategy {order.strategy_id} on {order.symbol} (owned by {owner_id})")
                    try:
                        strat = self.strategies.get(order.strategy_id)
                        if strat is not None:
                            setattr(strat, '_entry_inflight', False)
                    except Exception:
                        pass
                    return
                # From owner strategy: block same-side stacking
                qty = getattr(position, 'quantity', 0) or 0
                if qty > 0 and order.side.name == 'BUY':
                    logger.warning(f"[Engine] Blocking duplicate LONG entry for {order.strategy_id} on {order.symbol} (already LONG qty={qty})")
                    try:
                        strat = self.strategies.get(order.strategy_id)
                        if strat is not None:
                            setattr(strat, '_entry_inflight', False)
                    except Exception:
                        pass
                    return
                if qty < 0 and order.side.name == 'SELL':
                    logger.warning(f"[Engine] Blocking duplicate SHORT entry for {order.strategy_id} on {order.symbol} (already SHORT qty={qty})")
                    try:
                        strat = self.strategies.get(order.strategy_id)
                        if strat is not None:
                            setattr(strat, '_entry_inflight', False)
                    except Exception:
                        pass
                    return
        except Exception as e:
            logger.error(f"[Engine] Error enforcing single-position rule: {e}")
        
        # RMS check
        if self.rms:
            logger.debug(f"[Engine] Performing RMS check for order {order.order_id}")
            current_price = self._get_current_price(order.symbol)
            if current_price:
                logger.debug(f"[Engine] Current price for {order.symbol}: {current_price:.2f}")
                rms_result = self.rms.check_order(order, self.portfolio, current_price)
                self.trading_logger.log_rms_check(rms_result.to_dict(), order.to_dict())
                
                if not rms_result.allowed:
                    logger.warning(f"[Engine] Order {order.order_id} REJECTED by RMS: {rms_result.reason}")
                    return
                else:
                    logger.info(f"[Engine] Order {order.order_id} PASSED RMS check")
            else:
                logger.warning(f"[Engine] Cannot perform RMS check: current price not available for {order.symbol}")
        else:
            logger.debug(f"[Engine] RMS disabled, skipping risk check")
        
        # Submit order
        logger.debug(f"[Engine] Submitting order {order.order_id} to order manager")
        if self.order_manager.submit_order(order):
            logger.info(f"[Engine] Order {order.order_id} submitted successfully")
            self.trading_logger.log_order(order.to_dict())
            
            # Save to database
            if self.db_manager:
                try:
                    self.db_manager.save_order(order.to_dict())
                    logger.debug(f"[Engine] Order {order.order_id} saved to database")
                except Exception as e:
                    logger.error(f"Error saving order to database: {e}")
        else:
            logger.error(f"[Engine] Failed to submit order {order.order_id}")
            # Clear inflight flag so strategy can retry later
            try:
                strat = self.strategies.get(order.strategy_id)
                if strat is not None:
                    setattr(strat, '_entry_inflight', False)
            except Exception:
                pass
    
    def _on_fill(self, fill):
        """Handle order fill"""
        logger.info(f"[Engine] Processing fill: {fill.side.value} {fill.quantity} {fill.symbol} @ {fill.price:.2f} (order_id={fill.order_id})")
        
        # Log fill (defer DB persistence until after portfolio update to keep state consistent)
        self.trading_logger.log_fill(fill.to_dict())
        
        # Get order for portfolio update
        order = self.order_manager.get_order(fill.order_id)
        if order:
            logger.debug(f"[Engine] Found order {fill.order_id} for fill processing")
        else:
            logger.warning(f"[Engine] Order {fill.order_id} not found for fill")
        
        # Note: order DB update will be done together with fill/position in a single transaction below
        
        # Calculate P&L before portfolio update
        old_realized = self.portfolio.total_realized_pnl
        old_cash = self.portfolio.cash
        logger.debug(f"[Engine] Portfolio before fill: cash={old_cash:.2f}, realized_pnl={old_realized:.2f}")
        
        # Capture previous position qty to detect open/close transitions
        prev_position = self.portfolio.get_position(fill.symbol)
        prev_qty = prev_position.quantity if prev_position else 0
        
        # Update portfolio
        logger.debug(f"[Engine] Processing fill in portfolio")
        self.portfolio.process_fill(fill, order)
        
        # Calculate changes
        new_realized = self.portfolio.total_realized_pnl
        new_cash = self.portfolio.cash
        pnl_change = new_realized - old_realized
        cash_change = new_cash - old_cash
        
        logger.info(f"[Engine] Portfolio updated: cash {old_cash:.2f} -> {new_cash:.2f} (change: {cash_change:+.2f}), realized_pnl {old_realized:.2f} -> {new_realized:.2f} (change: {pnl_change:+.2f})")
        
        # Persist fill, order, position and a portfolio snapshot atomically to the database
        if self.db_manager:
            try:
                position = self.portfolio.get_position(fill.symbol)
                position_dict = position.to_dict() if position else None
                order_dict = order.to_dict() if order else None
                self.db_manager.save_fill_order_and_position_transaction(
                    fill.to_dict(), order_dict, position_dict, self.portfolio.to_dict()
                )
                logger.debug(f"[Engine] Fill/order/position persisted transactionally")
            except Exception as e:
                logger.error(f"Error persisting fill/order/position transactionally: {e}")
        
        # After state persistence, create/close trade records
        if self.db_manager and order:
            try:
                # Current position after fill
                position_after = self.portfolio.get_position(fill.symbol)
                new_qty = position_after.quantity if position_after else 0
                # Opening a new position from flat -> non-flat
                if prev_qty == 0 and new_qty != 0:
                    entry_price = position_after.average_price if position_after else fill.price
                    trade_type = 'LONG' if (position_after and position_after.is_long()) else 'SHORT'
                    self.db_manager.create_trade(
                        order_id=order.order_id,
                        symbol=fill.symbol,
                        strategy_id=order.strategy_id or "",
                        trade_type=trade_type,
                        entry_time=fill.timestamp.isoformat() if hasattr(fill.timestamp, 'isoformat') else str(fill.timestamp),
                        entry_price=float(entry_price),
                        stop_loss=position_after.stop_loss if position_after else None,
                        take_profit=position_after.take_profit if position_after else None,
                        ltp=position_after.current_price if position_after else fill.price,
                    )
                    logger.info(f"[Engine] Trade opened recorded for {fill.symbol} (order_id={order.order_id})")
                    # Log trade open
                    try:
                        self.trading_logger.log_trade_open({
                            'order_id': order.order_id,
                            'symbol': fill.symbol,
                            'strategy_id': order.strategy_id or "",
                            'trade_type': trade_type,
                            'entry_time': fill.timestamp,
                            'entry_price': float(entry_price),
                            'stop_loss': position_after.stop_loss if position_after else None,
                            'take_profit': position_after.take_profit if position_after else None,
                            'ltp': position_after.current_price if position_after else float(fill.price),
                        })
                    except Exception:
                        logger.debug("Failed to log trade open")
                # Closing position from non-flat -> flat
                elif prev_qty != 0 and new_qty == 0:
                    self.db_manager.close_latest_trade(
                        symbol=fill.symbol,
                        strategy_id=order.strategy_id or "",
                        exit_time=fill.timestamp.isoformat() if hasattr(fill.timestamp, 'isoformat') else str(fill.timestamp),
                        exit_price=float(fill.price),
                        status='closed'
                    )
                    logger.info(f"[Engine] Trade closed recorded for {fill.symbol} (order_id={order.order_id})")
                    # Log trade close
                    try:
                        self.trading_logger.log_trade_close({
                            'order_id': order.order_id,
                            'symbol': fill.symbol,
                            'strategy_id': order.strategy_id or "",
                            'trade_type': 'LONG' if (prev_position and prev_position.quantity > 0) else 'SHORT',
                            'exit_time': fill.timestamp,
                            'exit_price': float(fill.price),
                        })
                    except Exception:
                        logger.debug("Failed to log trade close")
                # Sign flip within a single fill (e.g., long -> short without passing through zero)
                elif prev_qty != 0 and new_qty != 0 and ((prev_qty > 0 and new_qty < 0) or (prev_qty < 0 and new_qty > 0)):
                    # First close the previous trade at the fill price/time
                    self.db_manager.close_latest_trade(
                        symbol=fill.symbol,
                        strategy_id=order.strategy_id or "",
                        exit_time=fill.timestamp.isoformat() if hasattr(fill.timestamp, 'isoformat') else str(fill.timestamp),
                        exit_price=float(fill.price),
                        status='closed'
                    )
                    try:
                        self.trading_logger.log_trade_close({
                            'order_id': order.order_id,
                            'symbol': fill.symbol,
                            'strategy_id': order.strategy_id or "",
                            'trade_type': 'LONG' if (prev_position and prev_position.quantity > 0) else 'SHORT',
                            'exit_time': fill.timestamp,
                            'exit_price': float(fill.price),
                        })
                    except Exception:
                        logger.debug("Failed to log trade close (sign flip)")

                    # Then open a new trade for the residual in the new direction
                    entry_price = position_after.average_price if position_after else fill.price
                    trade_type = 'LONG' if (position_after and position_after.is_long()) else 'SHORT'
                    self.db_manager.create_trade(
                        order_id=order.order_id,
                        symbol=fill.symbol,
                        strategy_id=order.strategy_id or "",
                        trade_type=trade_type,
                        entry_time=fill.timestamp.isoformat() if hasattr(fill.timestamp, 'isoformat') else str(fill.timestamp),
                        entry_price=float(entry_price),
                        stop_loss=position_after.stop_loss if position_after else None,
                        take_profit=position_after.take_profit if position_after else None,
                        ltp=position_after.current_price if position_after else fill.price,
                    )
                    try:
                        self.trading_logger.log_trade_open({
                            'order_id': order.order_id,
                            'symbol': fill.symbol,
                            'strategy_id': order.strategy_id or "",
                            'trade_type': trade_type,
                            'entry_time': fill.timestamp,
                            'entry_price': float(entry_price),
                            'stop_loss': position_after.stop_loss if position_after else None,
                            'take_profit': position_after.take_profit if position_after else None,
                            'ltp': position_after.current_price if position_after else float(fill.price),
                        })
                    except Exception:
                        logger.debug("Failed to log trade open (sign flip)")
            except Exception as e:
                logger.error(f"Error updating trades table: {e}")
        
        # Calculate trade P&L (change in realized P&L)
        pnl = new_realized - old_realized
        if pnl != 0:
            logger.info(f"[Engine] Trade P&L: {pnl:+.2f}")
        
        # Update metrics
        if order and order.strategy_id:
            logger.debug(f"[Engine] Recording trade in metrics for strategy {order.strategy_id}")
            # Record trade in metrics
            self.portfolio_metrics.record_trade(fill, order.strategy_id, pnl)
            logger.debug(f"[Engine] Trade recorded in metrics")
        
        # Update strategy position state based on actual portfolio position
        if order and order.strategy_id in self.strategies:
            strategy = self.strategies[order.strategy_id]
            position = self.portfolio.get_position(fill.symbol)
            
            logger.debug(f"[Engine] Updating strategy {order.strategy_id} position state")
            if position and not position.is_flat():
                strategy.has_position = True
                strategy.position_entry_price = position.average_price
                if position.opened_at:
                    strategy.position_entry_time = position.opened_at
                try:
                    strategy.position_is_long = position.is_long()
                except Exception:
                    pass
                logger.info(f"[Engine] Strategy {order.strategy_id} position updated: has_position=True, entry_price={position.average_price:.2f}")
            else:
                strategy.has_position = False
                strategy.position_entry_price = None
                strategy.position_entry_time = None
                strategy.position_is_long = None
                logger.info(f"[Engine] Strategy {order.strategy_id} position updated: has_position=False (position closed)")
            # In all cases after processing a fill, clear inflight to allow future entries
            try:
                setattr(strategy, '_entry_inflight', False)
            except Exception:
                pass
        
        # Update portfolio metrics
        logger.debug(f"[Engine] Updating portfolio metrics")
        self.portfolio_metrics.update()
        
        # Save metrics to database
        if self.db_manager:
            try:
                # Save daily metrics
                daily_metrics = self.portfolio_metrics.daily_metrics.get(self.portfolio_metrics.current_date)
                if daily_metrics:
                    self.db_manager.save_daily_metrics(daily_metrics.to_dict())
                    logger.debug(f"[Engine] Daily metrics saved to database")
                
                # Save strategy metrics
                for strategy_metrics in self.portfolio_metrics.strategy_metrics.values():
                    self.db_manager.save_strategy_metrics(strategy_metrics.to_dict())
                logger.debug(f"[Engine] Strategy metrics saved to database")
            except Exception as e:
                logger.error(f"Error saving metrics to database: {e}")
        
        # Save portfolio state snapshot
        if self.db_manager:
            try:
                portfolio_dict = self.portfolio.to_dict()
                self.db_manager.save_portfolio_state(portfolio_dict)
                logger.debug(f"[Engine] Portfolio state saved to database")
            except Exception as e:
                logger.error(f"Error saving portfolio state to database: {e}")
        
        # Log portfolio update
        self.trading_logger.log_portfolio_update(self.portfolio.to_dict())
        logger.debug(f"[Engine] Fill processing completed")
    
    def _order_execution_loop(self):
        """Execute pending orders"""
        loop_count = 0
        while self.is_running:
            try:
                loop_count += 1
                open_orders = self.order_manager.get_open_orders()
                
                if open_orders:
                    logger.debug(f"[Engine] Order execution loop #{loop_count}: {len(open_orders)} open order(s)")
                
                for order in open_orders:
                    logger.debug(f"[Engine] Attempting to execute order {order.order_id}: {order.side.value} {order.quantity} {order.symbol}")
                    current_price = self._get_current_price(order.symbol)
                    if current_price:
                        logger.debug(f"[Engine] Current price for {order.symbol}: {current_price:.2f}")
                        fill = self.order_manager.execute_order(order.order_id, current_price)
                        if fill:
                            logger.info(f"[Engine] Order {order.order_id} executed: fill_id={fill.order_id}, price={fill.price:.2f}")
                            time.sleep(0.1)  # Small delay between executions
                        else:
                            logger.debug(f"[Engine] Order {order.order_id} not executed (conditions not met)")
                    else:
                        logger.debug(f"[Engine] Cannot execute order {order.order_id}: current price not available for {order.symbol}")
                
                time.sleep(0.5)  # Check every 500ms
            except Exception as e:
                logger.error(f"[Engine] Error in order execution loop: {e}", exc_info=True)
                self.trading_logger.log_error(f"Error in order execution loop: {str(e)}")
    
    def _sync_strategy_positions(self, symbol: str):
        """Sync strategy position state with portfolio"""
        # Get position and strategies for symbol
        position = self.portfolio.get_position(symbol)
        strategy_ids = self.strategy_symbol_map.get(symbol, [])
        
        if not strategy_ids:
            return
        
        # For each strategy mapped to this symbol
        for strategy_id in strategy_ids:
            strategy = self.strategies.get(strategy_id)
            if not strategy:
                logger.warning(f"[Engine] Strategy {strategy_id} not found while syncing positions")
                continue
            
            # Reset strategy position state
            old_has_position = strategy.has_position
            strategy.has_position = False
            strategy.position_entry_price = None
            strategy.position_entry_time = None
            
            # If we have an active position
            if position and not position.is_flat():
                # Position either belongs to this strategy or is shared (no strategy_id)
                if not position.strategy_id or position.strategy_id == strategy_id:
                    strategy.has_position = True
                    strategy.position_entry_price = position.average_price
                    if position.opened_at:
                        strategy.position_entry_time = position.opened_at
                    try:
                        strategy.position_is_long = position.is_long()
                    except Exception:
                        pass
                    
                    # Log position state change
                    if not old_has_position:
                        logger.info(f"[Engine] Strategy {strategy_id} now has position in {symbol} @ {position.average_price:.2f}")
            
            # Log position state change
            if old_has_position and not strategy.has_position:
                logger.info(f"[Engine] Strategy {strategy_id} no longer has position in {symbol}")
            
            logger.debug(f"[Engine] Synced {strategy_id} position state: has_position={strategy.has_position}, entry_price={strategy.position_entry_price}")
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        position = self.portfolio.get_position(symbol)
        if position:
            return position.current_price
        
        # Try to get from strategies
        if symbol in self.strategy_symbol_map:
            for strategy_id in self.strategy_symbol_map[symbol]:
                strategy = self.strategies[strategy_id]
                if strategy.current_price:
                    return strategy.current_price
        
        return None
    
    def _check_position_exits(self, symbol: str, price: float):
        """Check if positions should be exited due to stop loss/take profit"""
        position = self.portfolio.get_position(symbol)
        if not position or position.is_flat():
            logger.debug(f"[Engine] No position exit check needed for {symbol} (no position or flat)")
            return
        
        logger.debug(f"[Engine] Checking position exits for {symbol}: price={price:.2f}, stop_loss={position.stop_loss}, take_profit={position.take_profit}")
        
        # Check stop loss
        if position.stop_loss is not None and position.should_stop_loss():
            logger.warning(f"[Engine] STOP LOSS TRIGGERED for {symbol}: price={price:.2f} <= stop_loss={position.stop_loss:.2f}")
            # Create exit order
            order = Order(
                symbol=symbol,
                side=OrderSide.SELL if position.is_long() else OrderSide.BUY,
                quantity=abs(position.quantity),
                order_type=OrderType.MARKET,
                strategy_id=position.strategy_id
            )
            self.order_manager.submit_order(order)
            logger.info(f"[Engine] Stop loss exit order submitted for {symbol}: {order.order_id}")
        else:
            if position.stop_loss is not None:
                logger.debug(f"[Engine] Stop loss check: {symbol} price {price:.2f} > stop_loss {position.stop_loss:.2f} (safe)")
            else:
                logger.debug(f"[Engine] Stop loss not set for {symbol}; skipping SL check")
        
        # Check take profit
        if position.take_profit is not None and position.should_take_profit():
            logger.info(f"[Engine] TAKE PROFIT TRIGGERED for {symbol}: price={price:.2f} >= take_profit={position.take_profit:.2f}")
            # Create exit order
            order = Order(
                symbol=symbol,
                side=OrderSide.SELL if position.is_long() else OrderSide.BUY,
                quantity=abs(position.quantity),
                order_type=OrderType.MARKET,
                strategy_id=position.strategy_id
            )
            self.order_manager.submit_order(order)
            logger.info(f"[Engine] Take profit exit order submitted for {symbol}: {order.order_id}")
        else:
            if position.take_profit is not None:
                logger.debug(f"[Engine] Take profit check: {symbol} price {price:.2f} < take_profit {position.take_profit:.2f}")
            else:
                logger.debug(f"[Engine] Take profit not set for {symbol}; skipping TP check")
    
    def get_portfolio_summary(self) -> dict:
        """Get portfolio summary"""
        summary = self.portfolio.to_dict()
        summary['metrics'] = self.portfolio_metrics.get_summary()
        return summary
    
    def get_metrics(self) -> dict:
        """Get portfolio metrics"""
        return {
            'daily_metrics': self.portfolio_metrics.get_daily_metrics(),
            'strategy_metrics': self.portfolio_metrics.get_strategy_metrics(),
            'summary': self.portfolio_metrics.get_summary(),
            'equity_curve': self.portfolio_metrics.get_equity_curve().to_dict('records') if len(self.portfolio_metrics.get_equity_curve()) > 0 else []
        }
    
    def get_strategy_status(self) -> dict:
        """Get status of all strategies"""
        return {
            strategy_id: {
                'symbol': strategy.symbol,
                'is_active': strategy.is_active,
                'has_position': strategy.has_position,
                'current_price': strategy.current_price
            }
            for strategy_id, strategy in self.strategies.items()
        }
    
    def get_ltp(self, symbol: Optional[str] = None) -> dict:
        """Get LTP (Last Traded Price) for symbol or all symbols"""
        if symbol:
            return {
                'symbol': symbol,
                'ltp': self.ltp.get(symbol),
                'timestamp': self.ltp_timestamp[symbol].isoformat() if symbol in self.ltp_timestamp else None
            }
        return {
            sym: {
                'ltp': price,
                'timestamp': self.ltp_timestamp[sym].isoformat() if sym in self.ltp_timestamp else None
            }
            for sym, price in self.ltp.items()
        }
    
    def load_from_database(self):
        """Load portfolio state and positions from database"""
        if not self.db_manager:
            logger.warning("Database not enabled, cannot load from database")
            return
        
        try:
            # Load positions
            positions = self.db_manager.get_positions()
            for pos_data in positions:
                from portfolio.position import Position
                position = Position(
                    symbol=pos_data['symbol'],
                    quantity=pos_data['quantity'],
                    average_price=pos_data['average_price'],
                    current_price=pos_data['current_price'],
                    stop_loss=pos_data.get('stop_loss'),
                    take_profit=pos_data.get('take_profit'),
                    strategy_id=pos_data.get('strategy_id', '')
                )
                if pos_data.get('opened_at'):
                    position.opened_at = datetime.fromisoformat(pos_data['opened_at'])
                if pos_data.get('last_updated'):
                    position.last_updated = datetime.fromisoformat(pos_data['last_updated'])
                
                self.portfolio.positions[pos_data['symbol']] = position
            
            # Load latest portfolio state
            portfolio_states = self.db_manager.get_portfolio_state(limit=1)
            if portfolio_states:
                latest_state = portfolio_states[0]
                self.portfolio.cash = latest_state['cash']
                self.portfolio.total_realized_pnl = latest_state['total_realized_pnl']
                self.portfolio.total_commission = latest_state['total_commission']
            
            logger.info(f"Loaded {len(positions)} positions from database")
        except Exception as e:
            logger.error(f"Error loading from database: {e}")
    
    def get_database_queries(self):
        """Get database query methods"""
        if not self.db_manager:
            return None
        return {
            'get_orders': self.db_manager.get_orders,
            'get_fills': self.db_manager.get_fills,
            'get_positions': self.db_manager.get_positions,
            'get_daily_metrics': self.db_manager.get_daily_metrics,
            'get_strategy_metrics': self.db_manager.get_strategy_metrics,
            'get_equity_history': self.db_manager.get_equity_history,
            'get_market_data': self.db_manager.get_market_data,
            'get_trades': self.db_manager.get_trades,
            'get_portfolio_state': self.db_manager.get_portfolio_state
        }

