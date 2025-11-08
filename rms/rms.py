"""
Risk Management System for order validation and risk checks.
"""
from dataclasses import dataclass
from typing import Optional
from order.order import Order, OrderSide
from portfolio.portfolio import Portfolio
import logging

logger = logging.getLogger(__name__)


@dataclass
class RMSResult:
    """RMS check result"""
    allowed: bool
    reason: Optional[str] = None
    
    def to_dict(self):
        return {
            'allowed': self.allowed,
            'reason': self.reason
        }


class RiskManagementSystem:
    """Risk Management System for validating orders"""
    
    def __init__(
        self,
        max_position_size: float = 0.1,  # 10% of capital
        max_total_exposure: float = 0.5,  # 50% of capital
        max_loss_per_trade: Optional[float] = None,  # Absolute max loss per trade
        max_daily_loss: Optional[float] = None  # Max daily loss
    ):
        self.max_position_size = max_position_size
        self.max_total_exposure = max_total_exposure
        self.max_loss_per_trade = max_loss_per_trade
        self.max_daily_loss = max_daily_loss
        self.daily_pnl = 0.0
    
    def check_order(self, order: Order, portfolio: Portfolio, current_price: float) -> RMSResult:
        """
        Check if order passes risk management rules.
        
        Args:
            order: Order to check
            portfolio: Current portfolio state
            current_price: Current market price
            
        Returns:
            RMSResult indicating if order is allowed
        """
        # Check position size
        position_value = abs(order.quantity * current_price)
        max_allowed_position = portfolio.initial_capital * self.max_position_size
        
        if position_value > max_allowed_position:
            reason = f"Position size {position_value:.2f} exceeds max {max_allowed_position:.2f}"
            logger.warning(f"RMS check failed: {reason}")
            return RMSResult(allowed=False, reason=reason)
        
        # Check total exposure
        current_exposure = portfolio.get_exposure()
        new_exposure = current_exposure
        position = portfolio.get_position(order.symbol)
        
        if position:
            # Adjust for existing position
            current_pos_value = abs(position.current_price * position.quantity)
            if order.side == OrderSide.BUY:
                new_pos_value = abs((position.quantity + order.quantity) * current_price)
            else:
                new_pos_value = abs((position.quantity - order.quantity) * current_price)
            new_exposure = current_exposure - current_pos_value + new_pos_value
        else:
            new_exposure = current_exposure + position_value
        
        max_allowed_exposure = portfolio.initial_capital * self.max_total_exposure
        if new_exposure > max_allowed_exposure:
            reason = f"Total exposure {new_exposure:.2f} would exceed max {max_allowed_exposure:.2f}"
            logger.warning(f"RMS check failed: {reason}")
            return RMSResult(allowed=False, reason=reason)
        
        # Check cash availability for buy orders
        if order.side == OrderSide.BUY:
            required_cash = order.quantity * current_price
            if required_cash > portfolio.cash:
                reason = f"Insufficient cash. Required: {required_cash:.2f}, Available: {portfolio.cash:.2f}"
                logger.warning(f"RMS check failed: {reason}")
                return RMSResult(allowed=False, reason=reason)
        
        # Check max loss per trade (if stop loss is set)
        if order.stop_loss and self.max_loss_per_trade:
            if order.side == OrderSide.BUY:
                loss_per_share = order.price - order.stop_loss if order.price else current_price - order.stop_loss
            else:
                loss_per_share = order.stop_loss - order.price if order.price else order.stop_loss - current_price
            
            max_loss = abs(loss_per_share * order.quantity)
            if max_loss > self.max_loss_per_trade:
                reason = f"Max loss per trade {max_loss:.2f} exceeds limit {self.max_loss_per_trade:.2f}"
                logger.warning(f"RMS check failed: {reason}")
                return RMSResult(allowed=False, reason=reason)
        
        # Check daily loss limit
        if self.max_daily_loss:
            if portfolio.total_pnl() < -self.max_daily_loss:
                reason = f"Daily loss {abs(portfolio.total_pnl()):.2f} exceeds limit {self.max_daily_loss:.2f}"
                logger.warning(f"RMS check failed: {reason}")
                return RMSResult(allowed=False, reason=reason)
        
        logger.info(f"RMS check passed for order {order.order_id}")
        return RMSResult(allowed=True)
    
    def reset_daily_pnl(self):
        """Reset daily P&L tracking"""
        self.daily_pnl = 0.0

