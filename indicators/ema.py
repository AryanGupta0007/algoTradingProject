"""
Exponential Moving Average (EMA) indicator.
"""
import pandas as pd
from typing import Optional
from .base import Indicator


class EMA(Indicator):
    """Exponential Moving Average indicator"""
    
    def __init__(self, period: int):
        super().__init__(period, f"EMA({period})")
        self.multiplier = 2.0 / (period + 1)
        self.current_ema: Optional[float] = None
    
    def update(self, price: float) -> Optional[float]:
        """Update EMA with new price"""
        if self.current_ema is None:
            # Initialize with first price
            self.current_ema = price
            self.values.append(self.current_ema)
            return self.current_ema
        
        # EMA calculation
        self.current_ema = (price * self.multiplier) + (self.current_ema * (1 - self.multiplier))
        self.values.append(self.current_ema)
        return self.current_ema
    
    def calculate(self, prices: pd.Series) -> pd.Series:
        """Calculate EMA for price series"""
        ema = prices.ewm(span=self.period, adjust=False).mean()
        return ema
    
    def reset(self):
        """Reset EMA state"""
        super().reset()
        self.current_ema = None

