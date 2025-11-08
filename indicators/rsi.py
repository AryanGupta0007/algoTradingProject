"""
Relative Strength Index (RSI) indicator.
"""
import pandas as pd
from typing import List, Optional
from .base import Indicator


class RSI(Indicator):
    """Relative Strength Index indicator"""
    
    def __init__(self, period: int = 14):
        super().__init__(period, f"RSI({period})")
        self.gains: List[float] = []
        self.losses: List[float] = []
        self.prev_price: Optional[float] = None
        self.avg_gain: Optional[float] = None
        self.avg_loss: Optional[float] = None
    
    def update(self, price: float) -> Optional[float]:
        """Update RSI with new price"""
        if self.prev_price is None:
            self.prev_price = price
            return None
        
        change = price - self.prev_price
        gain = change if change > 0 else 0.0
        loss = -change if change < 0 else 0.0
        
        self.gains.append(gain)
        self.losses.append(loss)
        
        if len(self.gains) > self.period:
            self.gains.pop(0)
            self.losses.pop(0)
        
        if len(self.gains) == self.period:
            if self.avg_gain is None:
                # Initial average
                self.avg_gain = sum(self.gains) / self.period
                self.avg_loss = sum(self.losses) / self.period
            else:
                # Smoothed average
                self.avg_gain = ((self.avg_gain * (self.period - 1)) + gain) / self.period
                self.avg_loss = ((self.avg_loss * (self.period - 1)) + loss) / self.period
            
            if self.avg_loss == 0:
                self.current_value = 100.0
            else:
                rs = self.avg_gain / self.avg_loss
                self.current_value = 100 - (100 / (1 + rs))
            
            self.values.append(self.current_value)
            self.prev_price = price
            return self.current_value
        
        self.prev_price = price
        return None
    
    def calculate(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI for price series"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def reset(self):
        """Reset RSI state"""
        super().reset()
        self.gains = []
        self.losses = []
        self.prev_price = None
        self.avg_gain = None
        self.avg_loss = None

