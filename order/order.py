"""
Order data structures.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Order representation"""
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    strategy_id: str
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    price: Optional[float] = None  # For limit orders
    stop_price: Optional[float] = None  # For stop orders
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    average_price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    def to_dict(self):
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'order_type': self.order_type.value,
            'price': self.price,
            'stop_price': self.stop_price,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'average_price': self.average_price,
            'timestamp': self.timestamp.isoformat(),
            'strategy_id': self.strategy_id,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit
        }
    
    def is_open(self) -> bool:
        """Check if order is still open"""
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]
    
    def is_filled(self) -> bool:
        """Check if order is filled"""
        return self.status == OrderStatus.FILLED
    
    def remaining_quantity(self) -> int:
        """Get remaining quantity to fill"""
        return self.quantity - self.filled_quantity


@dataclass
class Fill:
    """Order fill representation"""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    timestamp: datetime = field(default_factory=datetime.now)
    commission: float = 0.0
    
    def to_dict(self):
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'price': self.price,
            'commission': self.commission,
            'timestamp': self.timestamp.isoformat()
        }

