"""
Portfolio management.
"""
from typing import Dict, Optional, List, TYPE_CHECKING
from datetime import datetime
from .position import Position
from order.order import Fill, OrderSide
import logging

if TYPE_CHECKING:
    from order.order import Order

logger = logging.getLogger(__name__)


class Portfolio:
    """Portfolio management and tracking"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.total_realized_pnl = 0.0
        self.total_commission = 0.0
    
    def update_position_price(self, symbol: str, price: float):
        """Update position price"""
        if symbol in self.positions:
            self.positions[symbol].update_price(price)
    
    def process_fill(self, fill: Fill, order: Optional['Order'] = None):
        """Process a fill and update portfolio"""
        symbol = fill.symbol
        
        # Get or create position
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        
        position = self.positions[symbol]
        old_quantity = position.quantity
        old_avg_price = position.average_price
        
        # Update position
        if fill.side == OrderSide.BUY:
            position.add_quantity(fill.quantity, fill.price)
            self.cash -= (fill.price * fill.quantity + fill.commission)
            # Update stop loss and take profit from order
            if order:
                if order.stop_loss:
                    position.stop_loss = order.stop_loss
                if order.take_profit:
                    position.take_profit = order.take_profit
                if order.strategy_id:
                    position.strategy_id = order.strategy_id
        else:
            position.reduce_quantity(fill.quantity, fill.price)
            self.cash += (fill.price * fill.quantity - fill.commission)
            # Clear stop loss and take profit if position is closed
            if position.is_flat():
                position.stop_loss = None
                position.take_profit = None
        
        # Update realized P&L if position was closed or reversed
        if old_quantity != 0:
            if old_quantity > 0 and fill.side == OrderSide.SELL:
                # Closing or reducing long position
                closed_quantity = min(fill.quantity, old_quantity)
                realized_pnl = (fill.price - old_avg_price) * closed_quantity
                self.total_realized_pnl += realized_pnl
            elif old_quantity < 0 and fill.side == OrderSide.BUY:
                # Closing or reducing short position
                closed_quantity = min(fill.quantity, abs(old_quantity))
                realized_pnl = (old_avg_price - fill.price) * closed_quantity
                self.total_realized_pnl += realized_pnl
        
        # Update commission
        self.total_commission += fill.commission
        
        logger.info(f"Fill processed: {fill.to_dict()}, Cash: {self.cash:.2f}")
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol"""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> List[Position]:
        """Get all positions"""
        return list(self.positions.values())
    
    def get_open_positions(self) -> List[Position]:
        """Get all open positions"""
        return [pos for pos in self.positions.values() if not pos.is_flat()]
    
    def total_unrealized_pnl(self) -> float:
        """Calculate total unrealized P&L"""
        return sum(pos.unrealized_pnl() for pos in self.positions.values())
    
    def total_pnl(self) -> float:
        """Calculate total P&L (realized + unrealized)"""
        return self.total_realized_pnl + self.total_unrealized_pnl()
    
    def total_value(self) -> float:
        """Calculate total portfolio value"""
        return self.cash + sum(pos.current_price * pos.quantity for pos in self.positions.values())
    
    def total_equity(self) -> float:
        """Calculate total equity"""
        return self.initial_capital + self.total_pnl()
    
    def get_exposure(self) -> float:
        """Calculate total exposure (absolute value of all positions)"""
        return sum(abs(pos.current_price * pos.quantity) for pos in self.positions.values())
    
    def to_dict(self):
        return {
            'initial_capital': self.initial_capital,
            'cash': self.cash,
            'total_equity': self.total_equity(),
            'total_value': self.total_value(),
            'total_realized_pnl': self.total_realized_pnl,
            'total_unrealized_pnl': self.total_unrealized_pnl(),
            'total_pnl': self.total_pnl(),
            'total_commission': self.total_commission,
            'exposure': self.get_exposure(),
            'positions': [pos.to_dict() for pos in self.get_open_positions()]
        }

