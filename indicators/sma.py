"""
Simple Moving Average (SMA) indicator.
"""
import pandas as pd
from typing import List, Optional
from .base import Indicator


class SMA(Indicator):
    """Simple Moving Average indicator"""
    
    def __init__(self, period: int):
        super().__init__(period, f"SMA({period})")
        self.prices: List[float] = []
    
    def update(self, price: float) -> Optional[float]:
        """Update SMA with new price"""
        self.prices.append(price)
        if len(self.prices) > self.period:
            self.prices.pop(0)
        
        if len(self.prices) == self.period:
            self.current_value = sum(self.prices) / self.period
            self.values.append(self.current_value)
            return self.current_value
        return None
    
    def calculate(self, prices: pd.Series) -> pd.Series:
        """Calculate SMA for price series"""
        sma = prices.rolling(window=self.period, min_periods=self.period).mean()
        return sma
    
    def reset(self):
        """Reset SMA state"""
        super().reset()
        self.prices = []

