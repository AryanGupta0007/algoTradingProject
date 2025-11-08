"""
Base class for technical indicators.
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Optional


class Indicator(ABC):
    """Base class for all technical indicators"""
    
    def __init__(self, period: int, name: str):
        self.period = period
        self.name = name
        self.values: List[float] = []
        self.current_value: Optional[float] = None
    
    @abstractmethod
    def update(self, price: float) -> Optional[float]:
        """
        Update indicator with new price value.
        
        Args:
            price: Current price value
            
        Returns:
            Current indicator value or None if not enough data
        """
        pass
    
    @abstractmethod
    def calculate(self, prices: pd.Series) -> pd.Series:
        """
        Calculate indicator for a series of prices.
        
        Args:
            prices: Series of price values
            
        Returns:
            Series of indicator values
        """
        pass
    
    def reset(self):
        """Reset indicator state"""
        self.values = []
        self.current_value = None

