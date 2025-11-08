"""
Fake data feed generator for testing and paper trading.
"""
import random
import time
import threading
from datetime import datetime, timedelta
from typing import List, Callable, Dict, Optional
from .base import DataFeed, OHLCVData
import logging

logger = logging.getLogger(__name__)


class FakeDataFeed(DataFeed):
    """Fake data feed that generates random OHLCV candles"""
    
    def __init__(self, symbols: List[str], interval_seconds: int = 60, base_price: float = 100.0):
        super().__init__()
        self.symbols = symbols
        self.interval_seconds = interval_seconds
        self.subscribed_symbols: List[str] = []
        self.callbacks: Dict[str, Callable] = {}
        self.price_data: Dict[str, float] = {symbol: base_price for symbol in symbols}
        self.thread: Optional[threading.Thread] = None
        self.volatility = 0.02  # 2% volatility
        self.drift = 0.0001  # Slight upward drift
    
    def connect(self) -> bool:
        """Start the fake data feed"""
        if self.is_running:
            return True
        
        self.is_running = True
        self.thread = threading.Thread(target=self._generate_data, daemon=True)
        self.thread.start()
        logger.info("Fake data feed connected and started")
        return True
    
    def disconnect(self):
        """Stop the fake data feed"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2)
        logger.info("Fake data feed disconnected")
    
    def subscribe(self, symbols: List[str], callback: Callable):
        """Subscribe to symbols"""
        for symbol in symbols:
            if symbol not in self.subscribed_symbols:
                self.subscribed_symbols.append(symbol)
            self.callbacks[symbol] = callback
        logger.info(f"Subscribed to symbols: {symbols}")
    
    def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from symbols"""
        for symbol in symbols:
            if symbol in self.subscribed_symbols:
                self.subscribed_symbols.remove(symbol)
            if symbol in self.callbacks:
                del self.callbacks[symbol]
        logger.info(f"Unsubscribed from symbols: {symbols}")
    
    def _generate_data(self):
        """Generate fake OHLCV data"""
        cycle_count = 0
        while self.is_running:
            cycle_count += 1
            logger.info(f"[FakeDataFeed] Starting data generation cycle #{cycle_count}")
            
            for symbol in self.subscribed_symbols:
                if not self.is_running:
                    break
                
                # Generate OHLCV candle
                current_price = self.price_data[symbol]
                logger.debug(f"[FakeDataFeed] Generating candle for {symbol}, current price: {current_price:.2f}")
                
                # Random walk with drift
                change_pct = random.gauss(self.drift, self.volatility)
                new_price = current_price * (1 + change_pct)
                logger.debug(f"[FakeDataFeed] {symbol} price change: {change_pct*100:.4f}%, new price: {new_price:.2f}")
                
                # Generate OHLC from price movement
                open_price = current_price
                close_price = new_price
                high_price = max(open_price, close_price) * (1 + abs(random.gauss(0, 0.005)))
                low_price = min(open_price, close_price) * (1 - abs(random.gauss(0, 0.005)))
                volume = random.randint(1000, 10000)
                
                logger.info(f"[FakeDataFeed] Generated {symbol} candle: O={open_price:.2f}, H={high_price:.2f}, L={low_price:.2f}, C={close_price:.2f}, V={volume}")
                
                # Create OHLCV data
                ohlcv = OHLCVData(
                    symbol=symbol,
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                    timestamp=datetime.now()
                )
                
                # Update price
                self.price_data[symbol] = close_price
                
                # Notify callback
                if symbol in self.callbacks:
                    try:
                        logger.debug(f"[FakeDataFeed] Notifying callback for {symbol}")
                        self.callbacks[symbol](ohlcv)
                        self._notify_subscribers(ohlcv)
                        logger.debug(f"[FakeDataFeed] Callback completed for {symbol}")
                    except Exception as e:
                        logger.error(f"Error in callback for {symbol}: {e}", exc_info=True)
            
            logger.info(f"[FakeDataFeed] Completed cycle #{cycle_count}, waiting {self.interval_seconds} seconds for next cycle")
            time.sleep(self.interval_seconds)

