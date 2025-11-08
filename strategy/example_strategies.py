"""
Example trading strategies.
"""
from typing import List, Optional
from .base import BaseStrategy, EntryCondition, ExitCondition
from indicators import SMA, RSI, EMA
import logging

logger = logging.getLogger(__name__)


from typing import Optional

class MovingAverageCrossoverStrategy(BaseStrategy):
    """Moving average crossover strategy"""
    
    def __init__(
        self,
        strategy_id: str,
        symbol: str,
        fast_period: int = 10,
        slow_period: int = 30,
        quantity: int = 1,
        stop_loss_pct: float = 0.02,  # 2% stop loss
        take_profit_pct: float = 0.04,  # 4% take profit
        initial_capital: float = 100000.0
    ):
        # Store period attributes BEFORE calling super().__init__()
        # because _initialize_indicators() is called in base class __init__
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Create entry/exit conditions
        entry_conditions = [
            EntryCondition(
                indicator_name="fast_sma",
                operator=">",
                threshold=0.0,  # Dummy threshold, will use indicator comparison
                additional_params={"compare_to": "slow_sma"}
            )
        ]
        
        exit_conditions = [
            ExitCondition(
                condition_type="indicator",
                indicator_name="fast_sma",
                operator="crossunder",
                threshold=0.0  # Will be evaluated against slow_sma
            ),
            ExitCondition(
                condition_type="stop_loss",
                stop_loss=None  # Will be calculated dynamically
            ),
            ExitCondition(
                condition_type="take_profit",
                take_profit=None  # Will be calculated dynamically
            )
        ]
        
        super().__init__(
            strategy_id=strategy_id,
            symbol=symbol,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            quantity=quantity,
            initial_capital=initial_capital
        )
    
    def _initialize_indicators(self):
        """Initialize SMA indicators"""
        self.indicators["fast_sma"] = SMA(self.fast_period)
        self.indicators["slow_sma"] = SMA(self.slow_period)
    
    def _check_entry_conditions(self) -> bool:
        """Check for moving average crossover"""
        fast_sma = self.indicators.get("fast_sma")
        slow_sma = self.indicators.get("slow_sma")
        
        if not fast_sma or not slow_sma:
            self._log("Missing SMA indicators", "DEBUG")
            if self.trading_logger:
                try:
                    self.trading_logger.log_condition_evaluation(
                        self.strategy_id,
                        'entry',
                        False,
                        {
                            'indicator': 'sma_crossover',
                            'reason': 'missing_indicators',
                            'has_fast': bool(fast_sma),
                            'has_slow': bool(slow_sma),
                            'fast_period': getattr(self, 'fast_period', None),
                            'slow_period': getattr(self, 'slow_period', None),
                        }
                    )
                except Exception as e:
                    self._log(f"Error logging entry condition: {str(e)}", "ERROR")
            return False
        
        if fast_sma.current_value is None or slow_sma.current_value is None:
            self._log("SMA indicators not initialized", "DEBUG")
            if self.trading_logger:
                try:
                    self.trading_logger.log_condition_evaluation(
                        self.strategy_id,
                        'entry',
                        False,
                        {
                            'indicator': 'sma_crossover',
                            'reason': 'indicator_not_initialized',
                            'fast_current': getattr(fast_sma, 'current_value', None),
                            'slow_current': getattr(slow_sma, 'current_value', None),
                            'fast_period': self.fast_period,
                            'slow_period': self.slow_period,
                        }
                    )
                except Exception as e:
                    self._log(f"Error logging entry condition: {str(e)}", "ERROR")
            return False
        
        # Check for MA comparison
        result = False
        if len(fast_sma.values) >= 2 and len(slow_sma.values) >= 2:
            prev_fast = fast_sma.values[-2]
            prev_slow = slow_sma.values[-2]
            curr_fast = fast_sma.values[-1]
            curr_slow = slow_sma.values[-1]
            
            # Log the current values of both indicators
            self._log(f"Checking Fast SMA {self.fast_period} vs Slow SMA {self.slow_period}", "DEBUG")
            self._log(f"Fast SMA={curr_fast:.2f} (prev={prev_fast:.2f})", "DEBUG")
            self._log(f"Slow SMA={curr_slow:.2f} (prev={prev_slow:.2f})", "DEBUG")
            
            # Check if fast crosses above slow
            if prev_fast <= prev_slow and curr_fast > curr_slow:
                result = True
                self._log(f"Entry signal: Fast SMA crossed above Slow SMA", "INFO")
            else:
                if prev_fast <= prev_slow:
                    self._log("No crossover: Fast SMA still below Slow SMA", "DEBUG")
                else:
                    self._log("No crossover: Fast SMA already above Slow SMA", "DEBUG")
            
            # Log condition evaluation to structured log
            if self.trading_logger:
                condition_details = {
                    'fast_sma': curr_fast,
                    'slow_sma': curr_slow,
                    'prev_fast_sma': prev_fast,
                    'prev_slow_sma': prev_slow,
                    'fast_period': self.fast_period,
                    'slow_period': self.slow_period,
                    'result': result
                }
                try:
                    self.trading_logger.log_condition_evaluation(
                        self.strategy_id,
                        'entry',
                        result,
                        condition_details
                    )
                except Exception as e:
                    self._log(f"Error logging condition: {str(e)}", "ERROR")
        
        else:
            # Not enough values yet to detect a crossover
            self._log(
                f"Not enough SMA values to evaluate crossover (fast_len={len(fast_sma.values)}, slow_len={len(slow_sma.values)})",
                "DEBUG",
            )
            if self.trading_logger:
                try:
                    self.trading_logger.log_condition_evaluation(
                        self.strategy_id,
                        'entry',
                        False,
                        {
                            'indicator': 'sma_crossover',
                            'reason': 'insufficient_data',
                            'fast_len': len(fast_sma.values) if hasattr(fast_sma, 'values') else None,
                            'slow_len': len(slow_sma.values) if hasattr(slow_sma, 'values') else None,
                            'fast_period': self.fast_period,
                            'slow_period': self.slow_period,
                        }
                    )
                except Exception as e:
                    self._log(f"Error logging entry condition: {str(e)}", "ERROR")
        return result
    
    def _check_exit_conditions(self):
        """Check exit conditions including crossover"""
        # Check for crossunder
        fast_sma = self.indicators.get("fast_sma")
        slow_sma = self.indicators.get("slow_sma")
        
        if fast_sma and slow_sma:
            if fast_sma.current_value is not None and slow_sma.current_value is not None:
                if len(fast_sma.values) >= 2 and len(slow_sma.values) >= 2:
                    prev_fast = fast_sma.values[-2]
                    prev_slow = slow_sma.values[-2]
                    curr_fast = fast_sma.values[-1]
                    curr_slow = slow_sma.values[-1]
                    
                    # Log current SMA state
                    self._log(f"Exit check (SMA crossunder): Fast SMA={curr_fast:.2f} (prev={prev_fast:.2f}), Slow SMA={curr_slow:.2f} (prev={prev_slow:.2f})", "DEBUG")
                    # Crossunder condition
                    crossunder = prev_fast >= prev_slow and curr_fast < curr_slow
                    if self.trading_logger:
                        try:
                            self.trading_logger.log_condition_evaluation(
                                self.strategy_id,
                                'exit',
                                crossunder,
                                {
                                    'indicator': 'sma_crossunder',
                                    'fast_period': self.fast_period,
                                    'slow_period': self.slow_period,
                                    'prev_fast_sma': prev_fast,
                                    'prev_slow_sma': prev_slow,
                                    'curr_fast_sma': curr_fast,
                                    'curr_slow_sma': curr_slow,
                                    'operator': 'crossunder',
                                    'threshold': 0.0,
                                }
                            )
                        except Exception as e:
                            self._log(f"Error logging exit condition: {str(e)}", "ERROR")
                    if crossunder:
                        self._log("Exit signal: Fast SMA crossed below Slow SMA", "INFO")
                        from .base import ExitReason
                        return ExitReason.SIGNAL
        
        # Check stop loss and take profit
        if self.position_entry_price and self.current_price:
            # Stop loss
            stop_loss_price = self.position_entry_price * (1 - self.stop_loss_pct)
            sl_hit = self.current_price <= stop_loss_price
            self._log(f"SL check: price={self.current_price:.2f} <= SL={stop_loss_price:.2f} -> {sl_hit}", "DEBUG")
            if self.trading_logger:
                try:
                    self.trading_logger.log_condition_evaluation(
                        self.strategy_id,
                        'exit',
                        sl_hit,
                        {
                            'condition_type': 'stop_loss',
                            'stop_loss': stop_loss_price,
                            'current_price': self.current_price,
                            'entry_price': self.position_entry_price,
                            'stop_loss_pct': self.stop_loss_pct,
                        }
                    )
                except Exception as e:
                    self._log(f"Error logging SL condition: {str(e)}", "ERROR")
            if sl_hit:
                from .base import ExitReason
                return ExitReason.STOP_LOSS
            
            # Take profit
            take_profit_price = self.position_entry_price * (1 + self.take_profit_pct)
            tp_hit = self.current_price >= take_profit_price
            self._log(f"TP check: price={self.current_price:.2f} >= TP={take_profit_price:.2f} -> {tp_hit}", "DEBUG")
            if self.trading_logger:
                try:
                    self.trading_logger.log_condition_evaluation(
                        self.strategy_id,
                        'exit',
                        tp_hit,
                        {
                            'condition_type': 'take_profit',
                            'take_profit': take_profit_price,
                            'current_price': self.current_price,
                            'entry_price': self.position_entry_price,
                            'take_profit_pct': self.take_profit_pct,
                        }
                    )
                except Exception as e:
                    self._log(f"Error logging TP condition: {str(e)}", "ERROR")
            if tp_hit:
                from .base import ExitReason
                return ExitReason.TAKE_PROFIT
        
        return None
    
    def _get_stop_loss(self) -> Optional[float]:
        """Calculate stop loss price"""
        if self.current_price:
            return self.current_price * (1 - self.stop_loss_pct)
        return None
    
    def _get_take_profit(self) -> Optional[float]:
        """Calculate take profit price"""
        if self.current_price:
            return self.current_price * (1 + self.take_profit_pct)
        return None


class RSIStrategy(BaseStrategy):
    """RSI-based mean reversion strategy"""
    
    def __init__(
        self,
        strategy_id: str,
        symbol: str,
        rsi_period: int = 14,
        oversold_threshold: float = 30.0,
        overbought_threshold: float = 70.0,
        quantity: int = 1,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.03,
        initial_capital: float = 100000.0
    ):
        # Store attributes BEFORE calling super().__init__()
        self.rsi_period = rsi_period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        entry_conditions = [
            EntryCondition(
                indicator_name="rsi",
                operator="<",
                threshold=oversold_threshold
            )
        ]
        
        exit_conditions = [
            ExitCondition(
                condition_type="indicator",
                indicator_name="rsi",
                operator=">",
                threshold=overbought_threshold
            ),
            ExitCondition(
                condition_type="stop_loss",
                stop_loss=None
            ),
            ExitCondition(
                condition_type="take_profit",
                take_profit=None
            )
        ]
        
        super().__init__(
            strategy_id=strategy_id,
            symbol=symbol,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            quantity=quantity,
            initial_capital=initial_capital
        )
    
    def _initialize_indicators(self):
        """Initialize RSI indicator"""
        self.indicators["rsi"] = RSI(self.rsi_period)
    
    def _get_stop_loss(self) -> Optional[float]:
        """Calculate stop loss price"""
        if self.current_price:
            return self.current_price * (1 - self.stop_loss_pct)
        return None
    
    def _get_take_profit(self) -> Optional[float]:
        """Calculate take profit price"""
        if self.current_price:
            return self.current_price * (1 + self.take_profit_pct)
        return None

