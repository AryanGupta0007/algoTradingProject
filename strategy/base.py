"""
Base strategy class for trading strategies.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Callable
from enum import Enum
from order.order import Order, OrderSide, OrderType
from datafeed.base import OHLCVData, TickData
import logging
import uuid

logger = logging.getLogger(__name__)


class ExitReason(Enum):
    """Exit reasons"""
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    SIGNAL = "signal"
    TIMEOUT = "timeout"
    MANUAL = "manual"


@dataclass
class EntryCondition:
    """Entry condition configuration"""
    indicator_name: str
    operator: str  # '>', '<', '>=', '<=', '==', 'crossover', 'crossunder'
    threshold: float
    additional_params: Optional[Dict] = None


@dataclass
class ExitCondition:
    """Exit condition configuration"""
    condition_type: str  # 'indicator', 'stop_loss', 'take_profit', 'timeout'
    indicator_name: Optional[str] = None
    operator: Optional[str] = None
    threshold: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timeout_seconds: Optional[int] = None
    additional_params: Optional[Dict] = None


class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(
        self,
        strategy_id: str,
        symbol: str,
        entry_conditions: List[EntryCondition],
        exit_conditions: List[ExitCondition],
        quantity: int = 1,
        initial_capital: float = 100000.0
    ):
        self.strategy_id = strategy_id
        self.symbol = symbol
        self.entry_conditions = entry_conditions
        self.exit_conditions = exit_conditions
        self.quantity = quantity
        self.initial_capital = initial_capital
        
        # State
        self.is_active = False
        self.has_position = False
        self.indicators: Dict[str, any] = {}
        self.current_price: Optional[float] = None
        self.current_ohlcv: Optional[OHLCVData] = None
        self.position_entry_price: Optional[float] = None
        self.position_entry_time: Optional[datetime] = None
        
        # Callbacks
        self.order_callback: Optional[Callable] = None
        self.log_callback: Optional[Callable] = None
        # Optional structured trading logger (TradingLogger instance)
        self.trading_logger = None
        
        # Initialize indicators
        self._initialize_indicators()
    
    @abstractmethod
    def _initialize_indicators(self):
        """Initialize strategy-specific indicators"""
        pass
    
    def set_order_callback(self, callback: Callable):
        """Set callback for order generation"""
        self.order_callback = callback
    
    def set_log_callback(self, callback: Callable):
        """Set callback for logging"""
        self.log_callback = callback

    def set_trading_logger(self, trading_logger):
        """Set the engine's TradingLogger instance so the strategy can emit structured events.

        The strategy will call `trading_logger.log_condition_evaluation(...)` when it
        evaluates entry/exit conditions so those appear in the structured JSONL log.
        """
        self.trading_logger = trading_logger
    
    def _log(self, message: str, level: str = "INFO"):
        """Log message"""
        log_msg = f"[{self.strategy_id}] {message}"
        if self.log_callback:
            self.log_callback(log_msg, level)
        else:
            getattr(logger, level.lower())(log_msg)

    def _fmt(self, v, precision: int = 4) -> str:
        """Safely format a numeric value; return 'None' when value is None.

        This avoids formatting errors when a value is missing (None) but a
        format specifier like :.4f is used in logging.
        """
        if v is None:
            return "None"
        try:
            return f"{v:.{precision}f}"
        except Exception:
            try:
                return str(v)
            except Exception:
                return "None"
    
    def on_data(self, data: OHLCVData):
        """
        Handle incoming market data.
        
        Args:
            data: OHLCV data or Tick data
        """
        if isinstance(data, OHLCVData):
            self._log(f"Received OHLCV data: {data.symbol} O={data.open:.2f} H={data.high:.2f} L={data.low:.2f} C={data.close:.2f} V={data.volume}", "DEBUG")
            self.current_ohlcv = data
            self.current_price = data.close
            
            # Update indicators
            self._log(f"Updating indicators with new OHLCV data", "DEBUG")
            self._update_indicators(data)
            
            # Log indicator values
            for name, indicator in self.indicators.items():
                if hasattr(indicator, 'current_value') and indicator.current_value is not None:
                    self._log(f"Indicator {name} value: {self._fmt(indicator.current_value,4)}", "DEBUG")
            
            # Check exit conditions first (if in position)
            if self.has_position:
                self._log(f"Checking exit conditions (has_position=True)", "DEBUG")
                exit_reason = self._check_exit_conditions()
                if exit_reason:
                    self._log(f"Exit condition met: {exit_reason.value}", "INFO")
                    self._exit_position(exit_reason)
                else:
                    self._log(f"No exit conditions met, holding position", "DEBUG")
            else:
                self._log(f"Skipping exit check (has_position=False)", "DEBUG")
            
            # Check entry conditions (if not in position)
            if not self.has_position:
                self._log(f"Checking entry conditions (has_position=False)", "DEBUG")
                if self._check_entry_conditions():
                    self._log(f"Entry conditions met, generating entry order", "INFO")
                    self._enter_position()
                else:
                    self._log(f"Entry conditions not met, waiting", "DEBUG")
            else:
                self._log(f"Skipping entry check (has_position=True)", "DEBUG")
        
        elif isinstance(data, TickData):
            self._log(f"Received Tick data: {data.symbol} price={data.price:.2f}", "DEBUG")
            self.current_price = data.price
            # Update indicators if needed for tick data
    
    def _update_indicators(self, ohlcv: OHLCVData):
        """Update all indicators with new data"""
        # Update indicators with OHLCV data (for TA-Lib indicators)
        updated_count = 0
        for name, indicator in self.indicators.items():
            old_value = indicator.current_value if hasattr(indicator, 'current_value') else None
            
            if hasattr(indicator, 'update_ohlcv'):
                new_value = indicator.update_ohlcv(ohlcv.open, ohlcv.high, ohlcv.low, ohlcv.close, ohlcv.volume)
                updated_count += 1
                if new_value is not None and new_value != old_value:
                    self._log(f"Indicator {name} updated: {self._fmt(old_value,4)} -> {self._fmt(new_value,4)}", "DEBUG")
            elif hasattr(indicator, 'update'):
                new_value = indicator.update(ohlcv.close)
                updated_count += 1
                if new_value is not None and new_value != old_value:
                    self._log(f"Indicator {name} updated: {self._fmt(old_value,4)} -> {self._fmt(new_value,4)}", "DEBUG")
        
        if updated_count > 0:
            self._log(f"Updated {updated_count} indicator(s)", "DEBUG")
    
    def _check_entry_conditions(self) -> bool:
        """
        Check if entry conditions are met.
        
        Returns:
            True if all entry conditions are met
        """
        if not self.current_ohlcv or self.current_price is None:
            self._log(f"Entry check skipped: no OHLCV data or price (ohlcv={self.current_ohlcv is not None}, price={self.current_price})", "DEBUG")
            return False
        
        self._log(f"Checking entry conditions (has_position={self.has_position}, price={self.current_price:.2f})", "DEBUG")
        
        for idx, condition in enumerate(self.entry_conditions):
            result = self._evaluate_condition(condition)
            # Gather richer details for logging
            indicator = None
            indicator_value = None
            prev_val = None
            curr_val = None
            indicator_values = None
            # Support direct OHLCV field comparisons (e.g., open > close)
            ohlcv_lhs = None
            ohlcv_rhs = None
            if condition.indicator_name in self.indicators:
                indicator = self.indicators[condition.indicator_name]
                indicator_value = getattr(indicator, 'current_value', None)
                indicator_values = getattr(indicator, 'values', None)
                if indicator_values and len(indicator_values) >= 1:
                    curr_val = indicator_values[-1]
                if indicator_values and len(indicator_values) >= 2:
                    prev_val = indicator_values[-2]
            else:
                # If the condition references an OHLCV field (open/high/low/close/volume)
                # capture current OHLCV values for richer logging
                if self.current_ohlcv and condition.indicator_name in ('open','high','low','close','volume'):
                    try:
                        ohlcv_lhs = getattr(self.current_ohlcv, condition.indicator_name)
                        # RHS can be another OHLCV field specified in additional_params
                        if condition.additional_params and isinstance(condition.additional_params, dict) and 'compare_to' in condition.additional_params:
                            rhs_field = condition.additional_params.get('compare_to')
                            if rhs_field and isinstance(rhs_field, str):
                                ohlcv_rhs = getattr(self.current_ohlcv, rhs_field, None)
                            else:
                                ohlcv_rhs = None
                        else:
                            ohlcv_rhs = condition.threshold
                    except Exception:
                        ohlcv_lhs = None
                        ohlcv_rhs = None

            condition_details = {
                'condition_index': idx,
                'indicator': condition.indicator_name,
                'operator': condition.operator,
                'threshold': condition.threshold,
                'result': result,
                'indicator_value': indicator_value,
                'indicator_prev': prev_val,
                'indicator_curr': curr_val,
                'indicator_values_last': list(indicator_values[-5:]) if indicator_values else None,
                'ohlcv_lhs': ohlcv_lhs,
                'ohlcv_rhs': ohlcv_rhs,
                'current_price': self.current_price,
                'ohlcv_timestamp': self.current_ohlcv.timestamp.isoformat() if self.current_ohlcv and hasattr(self.current_ohlcv, 'timestamp') else None
            }
            
            if result:
                self._log(f"Entry condition {idx+1}/{len(self.entry_conditions)} PASSED: {condition.indicator_name} {condition.operator} {condition.threshold}", "DEBUG")
            else:
                self._log(f"Entry condition {idx+1}/{len(self.entry_conditions)} FAILED: {condition.indicator_name} {condition.operator} {condition.threshold}", "DEBUG")
            
            # Log condition evaluation (if trading logger supports it)
            # Note: Condition evaluation is already logged via _log() method above
            
            if not result:
                self._log(f"Entry conditions NOT met (failed at condition {idx+1})", "DEBUG")
                # Emit structured condition evaluation if trading_logger is available
                if self.trading_logger:
                    try:
                        self.trading_logger.log_condition_evaluation(self.strategy_id, 'entry', result, condition_details)
                    except Exception:
                        # Do not raise from logging
                        pass
                return False
        
        self._log(f"All entry conditions PASSED - Entry signal generated!", "INFO")
        # Emit structured condition evaluation for the final pass - include summary of evaluated conditions
        if self.trading_logger:
            try:
                # Build per-condition summaries
                conds = []
                for idx, condition in enumerate(self.entry_conditions):
                    ind = condition.indicator_name
                    ind_obj = None
                    if isinstance(ind, str):
                        ind_obj = self.indicators.get(ind)
                    # Since we're in the branch where all entry conditions passed,
                    # mark each condition's result as True for clarity in logs.
                    conds.append({
                        'condition_index': idx,
                        'indicator': ind,
                        'operator': condition.operator,
                        'threshold': condition.threshold,
                        'indicator_value': getattr(ind_obj, 'current_value', None) if ind_obj else None,
                        'result': True
                    })
                self.trading_logger.log_condition_evaluation(self.strategy_id, 'entry', True, {'conditions': conds, 'current_price': self.current_price})
            except Exception:
                pass
        return True
    
    def _check_exit_conditions(self) -> Optional[ExitReason]:
        """
        Check if any exit condition is met.
        
        Returns:
            ExitReason if exit condition met, None otherwise
        """
        if not self.current_price:
            self._log(f"Exit check skipped: no current price", "DEBUG")
            return None
        
        if not self.has_position:
            self._log(f"Exit check skipped: no position", "DEBUG")
            return None
        
        self._log(f"Checking exit conditions (price={self.current_price:.2f}, entry_price={self.position_entry_price})", "DEBUG")
        
        for idx, condition in enumerate(self.exit_conditions):
            exit_reason = self._evaluate_exit_condition(condition)

            # Rich details for exit condition
            condition_details = {
                'condition_index': idx,
                'condition_type': condition.condition_type,
                    'condition_index': idx,
                'result': exit_reason is not None,
                'exit_reason': exit_reason.value if exit_reason else None,
                'current_price': self.current_price,
                'position_entry_price': self.position_entry_price,
                'position_has': self.has_position,
                'ohlcv_timestamp': self.current_ohlcv.timestamp.isoformat() if self.current_ohlcv and hasattr(self.current_ohlcv, 'timestamp') else None
            }
            
            if condition.condition_type == 'stop_loss':
                condition_details['stop_loss'] = condition.stop_loss
            elif condition.condition_type == 'take_profit':
                condition_details['take_profit'] = condition.take_profit
            elif condition.condition_type == 'indicator':
                condition_details['indicator'] = condition.indicator_name
                condition_details['operator'] = condition.operator
                condition_details['threshold'] = condition.threshold
            
            if exit_reason:
                # Enrich with condition-specific params
                if condition.condition_type == 'stop_loss':
                    condition_details['stop_loss'] = condition.stop_loss
                if condition.condition_type == 'take_profit':
                    condition_details['take_profit'] = condition.take_profit
                if condition.condition_type == 'indicator':
                    ind = condition.indicator_name
                    ind_obj = None
                    if ind is not None:
                        ind_obj = self.indicators.get(ind)
                    condition_details.update({
                        'indicator': ind,
                        'indicator_value': getattr(ind_obj, 'current_value', None) if ind_obj else None,
                        'indicator_prev': getattr(ind_obj, 'values', [])[-2] if ind_obj and getattr(ind_obj, 'values', None) and len(ind_obj.values) >= 2 else None,
                        'indicator_curr': getattr(ind_obj, 'values', [])[-1] if ind_obj and getattr(ind_obj, 'values', None) and len(ind_obj.values) >= 1 else None,
                    })

                self._log(f"Exit condition {idx+1}/{len(self.exit_conditions)} TRIGGERED: {condition.condition_type} -> {exit_reason.value}", "INFO")
                # Emit structured condition evaluation (and a human-readable eval log)
                if self.trading_logger:
                    try:
                        details = condition_details.copy()
                        details['exit_reason'] = exit_reason.value
                        self.trading_logger.log_condition_evaluation(self.strategy_id, 'exit', True, details)
                    except Exception:
                        pass

                # Also emit a per-condition human-friendly debug line so exit checks appear
                # in the same style as entry evaluations in the main log stream.
                try:
                    human_eval = f"Condition evaluation [{self.strategy_id}]: exit = True — condition[{idx}] {condition.condition_type} -> exit_reason={exit_reason.value}"
                    self._log(human_eval, "DEBUG")
                except Exception:
                    pass

                return exit_reason
            else:
                # Log not-triggered exit condition with details
                self._log(f"Exit condition {idx+1}/{len(self.exit_conditions)} NOT triggered: {condition.condition_type}", "DEBUG")
                # Emit a human-readable evaluation similar to entry logs
                try:
                    human_eval = f"Condition evaluation [{self.strategy_id}]: exit = False — condition[{idx}] {condition.condition_type} -> result=False"
                    # For indicator-type exit conditions, include indicator/threshold/value when available
                    if condition.condition_type == 'indicator' and condition.indicator_name in self.indicators:
                        ind_obj = self.indicators.get(condition.indicator_name)
                        ind_val = getattr(ind_obj, 'current_value', None)
                        human_eval = f"Condition evaluation [{self.strategy_id}]: exit = False — condition[{idx}] {condition.indicator_name} {condition.operator} {condition.threshold} -> value={self._fmt(ind_val,4)}, result=False"
                    self._log(human_eval, "DEBUG")
                except Exception:
                    pass

                if self.trading_logger:
                    try:
                        # Add small enrichments for non-triggered conditions
                        if condition.condition_type == 'stop_loss':
                            condition_details['stop_loss'] = condition.stop_loss
                        if condition.condition_type == 'take_profit':
                            condition_details['take_profit'] = condition.take_profit
                        self.trading_logger.log_condition_evaluation(self.strategy_id, 'exit', False, condition_details)
                    except Exception:
                        pass
        
        self._log(f"All exit conditions checked - No exit signal", "DEBUG")
        return None
    
    def _evaluate_condition(self, condition: EntryCondition) -> bool:
        """Evaluate a single condition"""
        # Support direct OHLCV field comparisons (e.g., open > close)
        if condition.indicator_name in ('open', 'high', 'low', 'close', 'volume'):
            if not self.current_ohlcv:
                self._log(f"Condition evaluation FAILED: no OHLCV data for field '{condition.indicator_name}'", "DEBUG")
                return False
            lhs = getattr(self.current_ohlcv, condition.indicator_name, None)
            # RHS may be specified as another OHLCV field via additional_params['compare_to'] or as a numeric threshold
            rhs = None
            rhs_field = None
            if condition.additional_params and isinstance(condition.additional_params, dict) and 'compare_to' in condition.additional_params:
                rhs_field = condition.additional_params.get('compare_to')
                if isinstance(rhs_field, str):
                    rhs = getattr(self.current_ohlcv, rhs_field, None)
            else:
                rhs = condition.threshold

            if lhs is None or rhs is None:
                self._log(f"Condition evaluation FAILED: missing lhs/rhs for OHLCV comparison ({condition.indicator_name} vs {rhs_field or 'threshold'})", "DEBUG")
                return False

            op = condition.operator
            result = False
            if op == '>':
                result = lhs > rhs
            elif op == '<':
                result = lhs < rhs
            elif op == '>=':
                result = lhs >= rhs
            elif op == '<=':
                result = lhs <= rhs
            elif op == '==':
                result = lhs == rhs
            else:
                self._log(f"Condition: Unknown operator '{op}' for OHLCV comparison", "WARNING")

            rhs_label = rhs_field if rhs_field else str(condition.threshold)
            self._log(f"Condition: {condition.indicator_name} ({self._fmt(lhs,4)}) {op} {rhs_label} ({self._fmt(rhs,4)}) = {result}", "DEBUG")
            return result

        # Indicator-based conditions
        if condition.indicator_name not in self.indicators:
            self._log(f"Condition evaluation FAILED: Indicator '{condition.indicator_name}' not found", "DEBUG")
            return False
        
        indicator = self.indicators[condition.indicator_name]
        indicator_value = indicator.current_value if hasattr(indicator, 'current_value') else None
        
        if indicator_value is None:
            self._log(f"Condition evaluation FAILED: Indicator '{condition.indicator_name}' has no value yet", "DEBUG")
            return False
        
        # Evaluate based on operator
        result = False
        if condition.operator == '>':
            result = indicator_value > condition.threshold
            self._log(f"Condition: {condition.indicator_name} ({self._fmt(indicator_value,4)}) > {self._fmt(condition.threshold,4)} = {result}", "DEBUG")
        elif condition.operator == '<':
            result = indicator_value < condition.threshold
            self._log(f"Condition: {condition.indicator_name} ({self._fmt(indicator_value,4)}) < {self._fmt(condition.threshold,4)} = {result}", "DEBUG")
        elif condition.operator == '>=':
            result = indicator_value >= condition.threshold
            self._log(f"Condition: {condition.indicator_name} ({self._fmt(indicator_value,4)}) >= {self._fmt(condition.threshold,4)} = {result}", "DEBUG")
        elif condition.operator == '<=':
            result = indicator_value <= condition.threshold
            self._log(f"Condition: {condition.indicator_name} ({self._fmt(indicator_value,4)}) <= {self._fmt(condition.threshold,4)} = {result}", "DEBUG")
        elif condition.operator == '==':
            result = abs(indicator_value - condition.threshold) < 0.0001
            self._log(f"Condition: {condition.indicator_name} ({self._fmt(indicator_value,4)}) == {self._fmt(condition.threshold,4)} = {result}", "DEBUG")
        elif condition.operator == 'crossover':
            # Check if indicator crossed above threshold
            if len(indicator.values) >= 2:
                prev_val = indicator.values[-2]
                curr_val = indicator.values[-1]
                result = prev_val <= condition.threshold and curr_val > condition.threshold
                self._log(f"Condition: {condition.indicator_name} crossover ({self._fmt(prev_val,4)} -> {self._fmt(curr_val,4)}) above {self._fmt(condition.threshold,4)} = {result}", "DEBUG")
            else:
                self._log(f"Condition: {condition.indicator_name} crossover - not enough data ({len(indicator.values)} values)", "DEBUG")
        elif condition.operator == 'crossunder':
            # Check if indicator crossed below threshold
            if len(indicator.values) >= 2:
                prev_val = indicator.values[-2]
                curr_val = indicator.values[-1]
                result = prev_val >= condition.threshold and curr_val < condition.threshold
                self._log(f"Condition: {condition.indicator_name} crossunder ({self._fmt(prev_val,4)} -> {self._fmt(curr_val,4)}) below {self._fmt(condition.threshold,4)} = {result}", "DEBUG")
            else:
                self._log(f"Condition: {condition.indicator_name} crossunder - not enough data ({len(indicator.values)} values)", "DEBUG")
        else:
            self._log(f"Condition: Unknown operator '{condition.operator}'", "WARNING")
        
        return result
    
    def _evaluate_exit_condition(self, condition: ExitCondition) -> Optional[ExitReason]:
        """Evaluate exit condition"""
        if condition.condition_type == 'stop_loss':
            if condition.stop_loss is not None and self.position_entry_price is not None:
                if self.has_position:
                    # Assuming long position for now
                    if self.current_price is not None and condition.stop_loss is not None and self.current_price <= condition.stop_loss:
                        self._log(f"Stop loss triggered: price {self.current_price:.2f} <= stop_loss {condition.stop_loss:.2f}", "INFO")
                        return ExitReason.STOP_LOSS
                    else:
                        self._log(f"Stop loss check: price {self.current_price:.2f} > stop_loss {condition.stop_loss:.2f} (safe)", "DEBUG")
            else:
                self._log(f"Stop loss check skipped: stop_loss={condition.stop_loss}, entry_price={self.position_entry_price}", "DEBUG")
        
        elif condition.condition_type == 'take_profit':
            if condition.take_profit is not None and self.position_entry_price is not None:
                if self.has_position:
                    if self.current_price is not None and condition.take_profit is not None and self.current_price >= condition.take_profit:
                        self._log(f"Take profit triggered: price {self.current_price:.2f} >= take_profit {condition.take_profit:.2f}", "INFO")
                        return ExitReason.TAKE_PROFIT
                    else:
                        self._log(f"Take profit check: price {self.current_price:.2f} < take_profit {condition.take_profit:.2f}", "DEBUG")
            else:
                self._log(f"Take profit check skipped: take_profit={condition.take_profit}, entry_price={self.position_entry_price}", "DEBUG")
        
        elif condition.condition_type == 'indicator':
            if condition.indicator_name and condition.indicator_name in self.indicators:
                indicator = self.indicators[condition.indicator_name]
                indicator_value = indicator.current_value if hasattr(indicator, 'current_value') else None
                
                if indicator_value is not None and condition.operator and condition.threshold:
                    if condition.operator == '>':
                        result = indicator_value > condition.threshold
                        self._log(f"Exit indicator condition: {condition.indicator_name} ({self._fmt(indicator_value,4)}) > {self._fmt(condition.threshold,4)} = {result}", "DEBUG")
                        if result:
                            return ExitReason.SIGNAL
                    elif condition.operator == '<':
                        result = indicator_value < condition.threshold
                        self._log(f"Exit indicator condition: {condition.indicator_name} ({self._fmt(indicator_value,4)}) < {self._fmt(condition.threshold,4)} = {result}", "DEBUG")
                        if result:
                            return ExitReason.SIGNAL
                else:
                    self._log(f"Exit indicator condition: {condition.indicator_name} value=None or missing operator/threshold", "DEBUG")
            else:
                self._log(f"Exit indicator condition: Indicator '{condition.indicator_name}' not found", "DEBUG")
        
        elif condition.condition_type == 'timeout':
            if condition.timeout_seconds and self.position_entry_time:
                elapsed = (datetime.now() - self.position_entry_time).total_seconds()
                if elapsed >= condition.timeout_seconds:
                    self._log(f"Timeout triggered: elapsed {elapsed:.1f}s >= timeout {condition.timeout_seconds}s", "INFO")
                    return ExitReason.TIMEOUT
                else:
                    self._log(f"Timeout check: elapsed {elapsed:.1f}s < timeout {condition.timeout_seconds}s", "DEBUG")
            else:
                self._log(f"Timeout check skipped: timeout_seconds={condition.timeout_seconds}, entry_time={self.position_entry_time}", "DEBUG")
        
        return None
    
    def _enter_position(self):
        """Enter a position"""
        if self.current_price is None:
            self._log("Cannot enter position: current_price is None", "WARNING")
            return
        
        stop_loss = self._get_stop_loss()
        take_profit = self._get_take_profit()
        
        self._log(f"Generating entry order: symbol={self.symbol}, quantity={self.quantity}, price={self.current_price:.2f}, stop_loss={stop_loss}, take_profit={take_profit}", "INFO")
        
        order = Order(
            symbol=self.symbol,
            side=OrderSide.BUY,  # Default to long, can be overridden
            quantity=self.quantity,
            order_type=OrderType.MARKET,
            strategy_id=self.strategy_id,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        if self.order_callback:
            self._log(f"Calling order callback to submit entry order", "DEBUG")
            self.order_callback(order)
            self._log(f"Entry order submitted: order_id={order.order_id}", "INFO")
        else:
            self._log("No order callback set, cannot submit order", "ERROR")
        
        self._log(f"Entry signal generated: {order.to_dict()}", "INFO")
    
    def _exit_position(self, reason: ExitReason):
        """Exit current position"""
        if not self.has_position:
            self._log("Cannot exit position: has_position is False", "WARNING")
            return
        
        if self.current_price is None:
            self._log("Cannot exit position: current_price is None", "WARNING")
            return
        
        entry_price_str = f"{self.position_entry_price:.2f}" if self.position_entry_price else "N/A"
        pnl_str = ""
        if self.position_entry_price:
            pnl = (self.current_price - self.position_entry_price) * self.quantity
            pnl_pct = ((self.current_price - self.position_entry_price) / self.position_entry_price) * 100
            pnl_str = f", P&L={pnl:.2f} ({pnl_pct:+.2f}%)"
        
        self._log(f"Generating exit order: reason={reason.value}, entry_price={entry_price_str}, exit_price={self.current_price:.2f}, quantity={self.quantity}{pnl_str}", "INFO")
        
        # Get current position quantity (this should come from portfolio)
        # For now, assume we exit the full quantity
        order = Order(
            symbol=self.symbol,
            side=OrderSide.SELL,
            quantity=self.quantity,
            order_type=OrderType.MARKET,
            strategy_id=self.strategy_id
        )
        
        if self.order_callback:
            self._log(f"Calling order callback to submit exit order", "DEBUG")
            self.order_callback(order)
            self._log(f"Exit order submitted: order_id={order.order_id}", "INFO")
        else:
            self._log("No order callback set, cannot submit order", "ERROR")
        
        self._log(f"Exit signal generated: {reason.value}, Order: {order.to_dict()}", "INFO")
        # Note: has_position will be updated by engine after fill is processed
    
    def _get_stop_loss(self) -> Optional[float]:
        """Get stop loss price from exit conditions"""
        for condition in self.exit_conditions:
            if condition.condition_type == 'stop_loss' and condition.stop_loss:
                return condition.stop_loss
        return None
    
    def _get_take_profit(self) -> Optional[float]:
        """Get take profit price from exit conditions"""
        for condition in self.exit_conditions:
            if condition.condition_type == 'take_profit' and condition.take_profit:
                return condition.take_profit
        return None
    
    def start(self):
        """Start the strategy"""
        self.is_active = True
        self._log("Strategy started")
    
    def stop(self):
        """Stop the strategy"""
        self.is_active = False
        self._log("Strategy stopped")
    
    def reset(self):
        """Reset strategy state"""
        self.has_position = False
        self.position_entry_price = None
        self.position_entry_time = None
        for indicator in self.indicators.values():
            if hasattr(indicator, 'reset'):
                indicator.reset()
        self._log("Strategy reset")

