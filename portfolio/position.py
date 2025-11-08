"""
Position tracking.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from order.order import OrderSide


@dataclass
class Position:
    """Position representation"""
    symbol: str
    quantity: int = 0
    average_price: float = 0.0
    current_price: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy_id: str = ""
    opened_at: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update_price(self, price: float):
        """Update current price"""
        self.current_price = price
        self.last_updated = datetime.now()
    
    def add_quantity(self, quantity: int, price: float):
        """Add to position"""
        if self.quantity == 0:
            self.average_price = price
            self.quantity = quantity
            self.opened_at = datetime.now()
        else:
            total_cost = (self.average_price * self.quantity) + (price * quantity)
            self.quantity += quantity
            self.average_price = total_cost / self.quantity
        self.last_updated = datetime.now()
    
    def reduce_quantity(self, quantity: int, price: float):
        """Reduce position"""
        if quantity > self.quantity:
            quantity = self.quantity
        self.quantity -= quantity
        if self.quantity == 0:
            self.average_price = 0.0
            self.opened_at = None
        self.last_updated = datetime.now()
    
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L"""
        if self.quantity == 0:
            return 0.0
        return (self.current_price - self.average_price) * self.quantity
    
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized P&L percentage"""
        if self.quantity == 0 or self.average_price == 0:
            return 0.0
        return ((self.current_price - self.average_price) / self.average_price) * 100
    
    def is_long(self) -> bool:
        """Check if position is long"""
        return self.quantity > 0
    
    def is_short(self) -> bool:
        """Check if position is short"""
        return self.quantity < 0
    
    def is_flat(self) -> bool:
        """Check if position is flat"""
        return self.quantity == 0
    
    def should_stop_loss(self) -> bool:
        """Check if stop loss should trigger"""
        if self.stop_loss is None or self.quantity == 0:
            return False
        if self.is_long():
            return self.current_price <= self.stop_loss
        else:
            return self.current_price >= self.stop_loss
    
    def should_take_profit(self) -> bool:
        """Check if take profit should trigger"""
        if self.take_profit is None or self.quantity == 0:
            return False
        if self.is_long():
            return self.current_price >= self.take_profit
        else:
            return self.current_price <= self.take_profit
    
    def to_dict(self):
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'average_price': self.average_price,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl(),
            'unrealized_pnl_pct': self.unrealized_pnl_pct(),
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'strategy_id': self.strategy_id,
            'opened_at': self.opened_at.isoformat() if self.opened_at else None,
            'last_updated': self.last_updated.isoformat()
        }

