"""
Moving Average Convergence Divergence (MACD) indicator.
"""
import pandas as pd
from typing import Optional
from .ema import EMA
from .base import Indicator


class MACD(Indicator):
    """MACD indicator (difference between two EMAs)"""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__(slow_period, f"MACD({fast_period},{slow_period},{signal_period})")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.fast_ema = EMA(fast_period)
        self.slow_ema = EMA(slow_period)
        self.signal_ema = EMA(signal_period)
        self.macd_line: Optional[float] = None
        self.signal_line: Optional[float] = None
        self.histogram: Optional[float] = None
    
    def update(self, price: float) -> Optional[float]:
        """Update MACD with new price"""
        fast_val = self.fast_ema.update(price)
        slow_val = self.slow_ema.update(price)
        
        if fast_val is not None and slow_val is not None:
            self.macd_line = fast_val - slow_val
            
            signal_val = self.signal_ema.update(self.macd_line)
            if signal_val is not None:
                self.signal_line = signal_val
                self.histogram = self.macd_line - self.signal_line
                self.current_value = self.histogram
                self.values.append(self.current_value)
                return self.current_value
        
        return None
    
    def calculate(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD for price series"""
        fast_ema = prices.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = prices.ewm(span=self.slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        return histogram
    
    def reset(self):
        """Reset MACD state"""
        super().reset()
        self.fast_ema.reset()
        self.slow_ema.reset()
        self.signal_ema.reset()
        self.macd_line = None
        self.signal_line = None
        self.histogram = None

