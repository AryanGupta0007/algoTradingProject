"""
Order management and execution system.
"""
from typing import Dict, List, Optional, Callable
from datetime import datetime
from .order import Order, OrderType, OrderSide, OrderStatus, Fill
import logging
import threading

logger = logging.getLogger(__name__)


class OrderManager:
    """Manages order submission, execution, and tracking"""
    
    def __init__(self, commission_rate: float = 0.001, slippage: float = 0.0001):
        self.orders: Dict[str, Order] = {}
        self.fills: List[Fill] = []
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.fill_callbacks: List[Callable] = []
        # Lock to protect concurrent execute_order calls and in-flight tracking
        self._lock = threading.Lock()
        # Track orders currently being executed to prevent double execution
        self._executing_orders: set = set()
    
    def submit_order(self, order: Order) -> bool:
        """Submit an order"""
        with self._lock:
            if order.order_id in self.orders:
                logger.warning(f"Order {order.order_id} already exists")
                return False

            # Enforce max one open order per (symbol, strategy_id)
            for existing in self.orders.values():
                try:
                    same_symbol = existing.symbol == order.symbol
                    same_strategy = getattr(existing, 'strategy_id', None) == getattr(order, 'strategy_id', None)
                except Exception:
                    same_symbol = False
                    same_strategy = False

                if same_symbol and same_strategy and existing.is_open():
                    logger.warning(f"Cannot submit order {order.order_id}: an open order already exists for symbol={order.symbol} strategy={order.strategy_id} (existing_order={existing.order_id})")
                    return False

            order.status = OrderStatus.SUBMITTED
            order.timestamp = datetime.now()
            self.orders[order.order_id] = order
            logger.info(f"Order submitted: {order.to_dict()}")
            return True
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        with self._lock:
            if order_id not in self.orders:
                logger.warning(f"Order {order_id} not found")
                return False

            order = self.orders[order_id]
            if not order.is_open():
                logger.warning(f"Order {order_id} cannot be cancelled (status: {order.status})")
                return False

            order.status = OrderStatus.CANCELLED
            logger.info(f"Order cancelled: {order_id}")
            return True
    
    def execute_order(self, order_id: str, current_price: float) -> Optional[Fill]:
        """
        Execute an order at current market price.
        
        Args:
            order_id: Order ID to execute
            current_price: Current market price
            
        Returns:
            Fill object if order was executed, None otherwise
        """
        # Ensure the order exists
        if order_id not in self.orders:
            logger.debug(f"[OrderManager] Order {order_id} not found")
            return None

        order = self.orders[order_id]
        logger.debug(f"[OrderManager] Evaluating order {order_id}: type={order.order_type.value}, side={order.side.value}, status={order.status.value}")

        # Prevent concurrent execution of the same order_id
        with self._lock:
            if order_id in self._executing_orders:
                logger.debug(f"[OrderManager] Order {order_id} is already being executed by another thread")
                return None
            # Mark as executing
            self._executing_orders.add(order_id)

        try:
            if not order.is_open():
                logger.debug(f"[OrderManager] Order {order_id} is not open (status: {order.status.value})")
                return None

            # Check if order can be executed based on type
            execution_price = self._get_execution_price(order, current_price)
            if execution_price is None:
                logger.debug(f"[OrderManager] Order {order_id} execution conditions not met (type={order.order_type.value}, price={current_price:.2f})")
                return None

            logger.info(f"[OrderManager] Order {order_id} execution conditions met: execution_price={execution_price:.2f} (market_price={current_price:.2f})")

            # Apply slippage
            before_slippage = execution_price
            if order.side == OrderSide.BUY:
                execution_price *= (1 + self.slippage)
            else:
                execution_price *= (1 - self.slippage)

            if execution_price != before_slippage:
                logger.debug(f"[OrderManager] Applied slippage: {before_slippage:.2f} -> {execution_price:.2f} (slippage={self.slippage*100:.4f}%)")

            # Calculate fill quantity (full fill for market orders)
            fill_quantity = order.remaining_quantity()
            logger.debug(f"[OrderManager] Fill quantity: {fill_quantity} (remaining from {order.quantity})")

            # If nothing to fill, bail out
            if fill_quantity <= 0:
                logger.debug(f"[OrderManager] No remaining quantity for order {order_id}; skipping execution")
                return None

            # Create fill
            commission = execution_price * fill_quantity * self.commission_rate
            fill = Fill(
                order_id=order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=fill_quantity,
                price=execution_price,
                commission=commission
            )
            logger.debug(f"[OrderManager] Created fill: price={execution_price:.2f}, quantity={fill_quantity}, commission={commission:.2f}")

            # Update order
            old_status = order.status
            order.filled_quantity += fill_quantity
            if order.filled_quantity == order.quantity:
                order.status = OrderStatus.FILLED
                logger.info(f"[OrderManager] Order {order_id} FILLED completely")
            else:
                order.status = OrderStatus.PARTIALLY_FILLED
                logger.info(f"[OrderManager] Order {order_id} PARTIALLY FILLED: {order.filled_quantity}/{order.quantity}")

            # Update average price
            if order.average_price is None:
                order.average_price = execution_price
            else:
                total_cost = (order.average_price * (order.filled_quantity - fill_quantity)) + (execution_price * fill_quantity)
                order.average_price = total_cost / order.filled_quantity
                logger.debug(f"[OrderManager] Updated average price: {order.average_price:.2f}")

            # Store fill
            self.fills.append(fill)
            logger.debug(f"[OrderManager] Fill stored (total fills: {len(self.fills)})")

            # Notify callbacks
            logger.debug(f"[OrderManager] Notifying {len(self.fill_callbacks)} fill callback(s)")
            for callback in self.fill_callbacks:
                try:
                    callback(fill)
                except Exception as e:
                    logger.error(f"Error in fill callback: {e}", exc_info=True)

            logger.info(f"[OrderManager] Order executed: {fill.to_dict()}")
            return fill
        finally:
            # Ensure we always clear the executing flag even on exceptions/returns
            with self._lock:
                if order_id in self._executing_orders:
                    self._executing_orders.remove(order_id)
    
    def _get_execution_price(self, order: Order, current_price: float) -> Optional[float]:
        """Determine execution price based on order type"""
        if order.order_type == OrderType.MARKET:
            logger.debug(f"[OrderManager] Market order: executing at current price {current_price:.2f}")
            return current_price
        
        elif order.order_type == OrderType.LIMIT:
            if order.price is None:
                logger.debug(f"[OrderManager] Limit order: no limit price set")
                return None
            if order.side == OrderSide.BUY:
                # Buy limit: execute if price <= limit
                if current_price <= order.price:
                    logger.debug(f"[OrderManager] Buy limit order: current_price {current_price:.2f} <= limit {order.price:.2f} -> EXECUTE")
                    return order.price
                else:
                    logger.debug(f"[OrderManager] Buy limit order: current_price {current_price:.2f} > limit {order.price:.2f} -> WAIT")
                    return None
            else:
                # Sell limit: execute if price >= limit
                if current_price >= order.price:
                    logger.debug(f"[OrderManager] Sell limit order: current_price {current_price:.2f} >= limit {order.price:.2f} -> EXECUTE")
                    return order.price
                else:
                    logger.debug(f"[OrderManager] Sell limit order: current_price {current_price:.2f} < limit {order.price:.2f} -> WAIT")
                    return None
        
        elif order.order_type == OrderType.STOP:
            if order.stop_price is None:
                logger.debug(f"[OrderManager] Stop order: no stop price set")
                return None
            if order.side == OrderSide.BUY:
                # Buy stop: execute if price >= stop
                if current_price >= order.stop_price:
                    logger.debug(f"[OrderManager] Buy stop order: current_price {current_price:.2f} >= stop {order.stop_price:.2f} -> EXECUTE")
                    return current_price
                else:
                    logger.debug(f"[OrderManager] Buy stop order: current_price {current_price:.2f} < stop {order.stop_price:.2f} -> WAIT")
                    return None
            else:
                # Sell stop: execute if price <= stop
                if current_price <= order.stop_price:
                    logger.debug(f"[OrderManager] Sell stop order: current_price {current_price:.2f} <= stop {order.stop_price:.2f} -> EXECUTE")
                    return current_price
                else:
                    logger.debug(f"[OrderManager] Sell stop order: current_price {current_price:.2f} > stop {order.stop_price:.2f} -> WAIT")
                    return None
        
        elif order.order_type == OrderType.STOP_LIMIT:
            if order.stop_price is None or order.price is None:
                logger.debug(f"[OrderManager] Stop-limit order: missing stop_price or limit price")
                return None
            if order.side == OrderSide.BUY:
                # Buy stop-limit: trigger when price >= stop, execute at limit
                if current_price >= order.stop_price:
                    if current_price <= order.price:
                        logger.debug(f"[OrderManager] Buy stop-limit: triggered and limit met -> EXECUTE at {order.price:.2f}")
                        return order.price
                    else:
                        logger.debug(f"[OrderManager] Buy stop-limit: triggered but limit not met (current {current_price:.2f} > limit {order.price:.2f})")
                        return None
                else:
                    logger.debug(f"[OrderManager] Buy stop-limit: not triggered (current {current_price:.2f} < stop {order.stop_price:.2f})")
                    return None
            else:
                # Sell stop-limit: trigger when price <= stop, execute at limit
                if current_price <= order.stop_price:
                    if current_price >= order.price:
                        logger.debug(f"[OrderManager] Sell stop-limit: triggered and limit met -> EXECUTE at {order.price:.2f}")
                        return order.price
                    else:
                        logger.debug(f"[OrderManager] Sell stop-limit: triggered but limit not met (current {current_price:.2f} < limit {order.price:.2f})")
                        return None
                else:
                    logger.debug(f"[OrderManager] Sell stop-limit: not triggered (current {current_price:.2f} > stop {order.stop_price:.2f})")
                    return None
        
        logger.debug(f"[OrderManager] Unknown order type: {order.order_type}")
        return None
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        with self._lock:
            return self.orders.get(order_id)
    
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Get all orders for a symbol"""
        with self._lock:
            return [order for order in self.orders.values() if order.symbol == symbol]
    
    def get_open_orders(self) -> List[Order]:
        """Get all open orders"""
        with self._lock:
            return [order for order in self.orders.values() if order.is_open()]
    
    def add_fill_callback(self, callback: Callable):
        """Add callback for fill events"""
        if callback not in self.fill_callbacks:
            self.fill_callbacks.append(callback)
    
    def remove_fill_callback(self, callback: Callable):
        """Remove fill callback"""
        if callback in self.fill_callbacks:
            self.fill_callbacks.remove(callback)

