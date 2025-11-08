"""
Base data feed interface.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Callable, List
from enum import Enum


class DataType(Enum):
    """Type of market data"""
    TICK = "tick"
    OHLCV = "ohlcv"


@dataclass
class TickData:
    """Tick data structure"""
    symbol: str
    price: float
    volume: int
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    
    def to_dict(self):
        return {
            'symbol': self.symbol,
            'price': self.price,
            'volume': self.volume,
            'timestamp': self.timestamp.isoformat(),
            'bid': self.bid,
            'ask': self.ask
        }


@dataclass
class OHLCVData:
    """OHLCV candle data structure"""
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    timestamp: datetime
    
    def to_dict(self):
        return {
            'symbol': self.symbol,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'timestamp': self.timestamp.isoformat()
        }


class DataFeed(ABC):
    """Base class for data feeds"""
    
    def __init__(self):
        self.subscribers: List[Callable] = []
        self.is_running = False
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to data feed"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Disconnect from data feed"""
        pass
    
    @abstractmethod
    def subscribe(self, symbols: List[str], callback: Callable):
        """Subscribe to symbols"""
        pass
    
    @abstractmethod
    def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from symbols"""
        pass
    
    def add_subscriber(self, callback: Callable):
        """Add a subscriber callback"""
        if callback not in self.subscribers:
            self.subscribers.append(callback)
    
    def remove_subscriber(self, callback: Callable):
        """Remove a subscriber callback"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    def _notify_subscribers(self, data):
        """Notify all subscribers of new data"""
        for callback in self.subscribers:
            try:
                callback(data)
            except Exception as e:
                print(f"Error notifying subscriber: {e}")

