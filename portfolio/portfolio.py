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
        
        # Update position with full long/short support
        if fill.side == OrderSide.BUY:
            # Cash move (buy -> cash out)
            self.cash -= (fill.price * fill.quantity + fill.commission)

            if position.quantity >= 0:
                # Increase/create long
                new_qty = position.quantity + fill.quantity
                if new_qty > 0:
                    total_cost = (position.average_price * position.quantity) + (fill.price * fill.quantity)
                    position.average_price = total_cost / new_qty
                else:
                    position.average_price = 0.0
                position.quantity = new_qty
            else:
                # Cover short
                short_qty = abs(position.quantity)
                cover = min(fill.quantity, short_qty)
                # Realized PnL from covering short
                realized_pnl = (position.average_price - fill.price) * cover
                self.total_realized_pnl += realized_pnl
                position.quantity += cover  # moves toward zero

                remaining = fill.quantity - cover
                if remaining > 0:
                    # Flip to long with remaining buy
                    position.quantity = remaining
                    position.average_price = fill.price
                    # On sign flip, reset stale stops if order doesn't provide new ones
                    if order:
                        position.strategy_id = order.strategy_id or position.strategy_id
                        position.stop_loss = order.stop_loss if order.stop_loss is not None else None
                        position.take_profit = order.take_profit if order.take_profit is not None else None
                    else:
                        position.stop_loss = None
                        position.take_profit = None
                if position.quantity == 0:
                    position.average_price = 0.0

            # Update stops/strategy tag from order (same-side increase or fresh long)
            if order:
                if order.strategy_id:
                    position.strategy_id = order.strategy_id
                # Only override when provided; otherwise keep prior values for same-side increases
                if order.stop_loss is not None:
                    position.stop_loss = order.stop_loss
                if order.take_profit is not None:
                    position.take_profit = order.take_profit
        else:  # SELL
            # Cash move (sell -> cash in)
            self.cash += (fill.price * fill.quantity - fill.commission)

            if position.quantity <= 0:
                # Increase/create short
                if position.quantity < 0:
                    # Increasing existing short
                    total_cost = (position.average_price * abs(position.quantity)) + (fill.price * fill.quantity)
                    new_abs_qty = abs(position.quantity) + fill.quantity
                    position.average_price = total_cost / new_abs_qty
                    position.quantity = -new_abs_qty
                else:
                    # Open fresh short
                    position.quantity = -fill.quantity
                    position.average_price = fill.price
                # Update stops/strategy tag from order when resulting position is short
                if order:
                    if order.strategy_id:
                        position.strategy_id = order.strategy_id
                    if order.stop_loss is not None:
                        position.stop_loss = order.stop_loss
                    if order.take_profit is not None:
                        position.take_profit = order.take_profit
            else:
                # Reduce/close long
                close = min(fill.quantity, position.quantity)
                realized_pnl = (fill.price - position.average_price) * close
                self.total_realized_pnl += realized_pnl
                position.quantity -= close
                remaining = fill.quantity - close
                if remaining > 0:
                    # Flip to short with remaining sell
                    position.quantity = -remaining
                    position.average_price = fill.price
                    # On sign flip, reset stale stops if order doesn't provide new ones
                    if order:
                        if order.strategy_id:
                            position.strategy_id = order.strategy_id
                        position.stop_loss = order.stop_loss if order.stop_loss is not None else None
                        position.take_profit = order.take_profit if order.take_profit is not None else None
                    else:
                        position.stop_loss = None
                        position.take_profit = None
                elif position.quantity == 0:
                    position.average_price = 0.0

            # Clear stops if flat
            if position.is_flat():
                position.stop_loss = None
                position.take_profit = None
        
        # Update realized P&L if position was closed or reversed
        # (Handled inline during adjustment above to support flips accurately)
        
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

