"""
TA-Lib based technical indicators.
"""
import talib
import numpy as np
import pandas as pd
from typing import List, Optional, Dict
from .base import Indicator


class TALibIndicator(Indicator):
    """Base class for TA-Lib indicators"""
    
    def __init__(self, period: int, name: str):
        super().__init__(period, name)
        self.prices: List[float] = []
        self.highs: List[float] = []
        self.lows: List[float] = []
        self.volumes: List[int] = []
        self.opens: List[float] = []
    
    def update_ohlcv(self, open: float, high: float, low: float, close: float, volume: int = 0):
        """Update with OHLCV data"""
        self.opens.append(open)
        self.highs.append(high)
        self.lows.append(low)
        self.prices.append(close)
        self.volumes.append(volume)
        
        # Keep only recent data (2x period for safety)
        max_length = self.period * 2
        if len(self.prices) > max_length:
            self.opens = self.opens[-max_length:]
            self.highs = self.highs[-max_length:]
            self.lows = self.lows[-max_length:]
            self.prices = self.prices[-max_length:]
            self.volumes = self.volumes[-max_length:]
        
        return self._calculate()
    
    def update(self, price: float) -> Optional[float]:
        """Update with price only (uses price for OHLC)"""
        return self.update_ohlcv(price, price, price, price, 0)
    
    def _calculate(self) -> Optional[float]:
        """Calculate indicator value - to be overridden"""
        return None
    
    def calculate(self, prices: pd.Series) -> pd.Series:
        """Calculate indicator for a series of prices"""
        # For backward compatibility, create OHLC from prices
        df = pd.DataFrame({
            'close': prices,
            'open': prices,
            'high': prices,
            'low': prices
        })
        return self.calculate_from_dataframe(df)
    
    def calculate_from_dataframe(self, df: pd.DataFrame) -> pd.Series:
        """Calculate indicator from OHLCV DataFrame - to be overridden"""
        return pd.Series()
    
    def reset(self):
        """Reset indicator state"""
        super().reset()
        self.prices = []
        self.highs = []
        self.lows = []
        self.volumes = []
        self.opens = []


class SMA(TALibIndicator):
    """Simple Moving Average using TA-Lib"""
    
    def __init__(self, period: int):
        super().__init__(period, f"SMA({period})")
    
    def _calculate(self) -> Optional[float]:
        if len(self.prices) < self.period:
            return None
        
        np_prices = np.array(self.prices, dtype=float)
        result = talib.SMA(np_prices, timeperiod=self.period)
        self.current_value = result[-1] if not np.isnan(result[-1]) else None
        if self.current_value is not None:
            self.values.append(self.current_value)
        return self.current_value
    
    def calculate_from_dataframe(self, df: pd.DataFrame) -> pd.Series:
        """Calculate SMA from DataFrame"""
        np_close = df['close'].values.astype(float)
        result = talib.SMA(np_close, timeperiod=self.period)
        return pd.Series(result, index=df.index)


class EMA(TALibIndicator):
    """Exponential Moving Average using TA-Lib"""
    
    def __init__(self, period: int):
        super().__init__(period, f"EMA({period})")
    
    def _calculate(self) -> Optional[float]:
        if len(self.prices) < self.period:
            return None
        
        np_prices = np.array(self.prices, dtype=float)
        result = talib.EMA(np_prices, timeperiod=self.period)
        self.current_value = result[-1] if not np.isnan(result[-1]) else None
        if self.current_value is not None:
            self.values.append(self.current_value)
        return self.current_value
    
    def calculate_from_dataframe(self, df: pd.DataFrame) -> pd.Series:
        """Calculate EMA from DataFrame"""
        np_close = df['close'].values.astype(float)
        result = talib.EMA(np_close, timeperiod=self.period)
        return pd.Series(result, index=df.index)


class RSI(TALibIndicator):
    """Relative Strength Index using TA-Lib"""
    
    def __init__(self, period: int = 14):
        super().__init__(period, f"RSI({period})")
    
    def _calculate(self) -> Optional[float]:
        if len(self.prices) < self.period + 1:
            return None
        
        np_prices = np.array(self.prices, dtype=float)
        result = talib.RSI(np_prices, timeperiod=self.period)
        self.current_value = result[-1] if not np.isnan(result[-1]) else None
        if self.current_value is not None:
            self.values.append(self.current_value)
        return self.current_value
    
    def calculate_from_dataframe(self, df: pd.DataFrame) -> pd.Series:
        """Calculate RSI from DataFrame"""
        np_close = df['close'].values.astype(float)
        result = talib.RSI(np_close, timeperiod=self.period)
        return pd.Series(result, index=df.index)


class MACD(TALibIndicator):
    """MACD using TA-Lib"""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__(slow_period, f"MACD({fast_period},{slow_period},{signal_period})")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.macd_line: Optional[float] = None
        self.signal_line: Optional[float] = None
        self.histogram: Optional[float] = None
    
    def _calculate(self) -> Optional[float]:
        if len(self.prices) < self.slow_period:
            return None
        
        np_prices = np.array(self.prices, dtype=float)
        macd, signal, histogram = talib.MACD(
            np_prices,
            fastperiod=self.fast_period,
            slowperiod=self.slow_period,
            signalperiod=self.signal_period
        )
        
        if not np.isnan(histogram[-1]):
            self.macd_line = macd[-1] if not np.isnan(macd[-1]) else None
            self.signal_line = signal[-1] if not np.isnan(signal[-1]) else None
            self.histogram = histogram[-1]
            self.current_value = self.histogram
            self.values.append(self.current_value)
            return self.current_value
        return None
    
    def calculate_from_dataframe(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate MACD from DataFrame"""
        np_close = df['close'].values.astype(float)
        macd, signal, histogram = talib.MACD(
            np_close,
            fastperiod=self.fast_period,
            slowperiod=self.slow_period,
            signalperiod=self.signal_period
        )
        return {
            'macd': pd.Series(macd, index=df.index),
            'signal': pd.Series(signal, index=df.index),
            'histogram': pd.Series(histogram, index=df.index)
        }
    
    def calculate(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD histogram for compatibility"""
        df = pd.DataFrame({
            'close': prices,
            'open': prices,
            'high': prices,
            'low': prices
        })
        result = self.calculate_from_dataframe(df)
        return result['histogram']


class BB(TALibIndicator):
    """Bollinger Bands using TA-Lib"""
    
    def __init__(self, period: int = 20, nbdevup: float = 2.0, nbdevdn: float = 2.0):
        super().__init__(period, f"BB({period},{nbdevup},{nbdevdn})")
        self.nbdevup = nbdevup
        self.nbdevdn = nbdevdn
        self.upper_band: Optional[float] = None
        self.middle_band: Optional[float] = None
        self.lower_band: Optional[float] = None
    
    def _calculate(self) -> Optional[float]:
        if len(self.prices) < self.period:
            return None
        
        np_prices = np.array(self.prices, dtype=float)
        upper, middle, lower = talib.BBANDS(
            np_prices,
            timeperiod=self.period,
            nbdevup=self.nbdevup,
            nbdevdn=self.nbdevdn
        )
        
        if not np.isnan(middle[-1]):
            self.upper_band = upper[-1] if not np.isnan(upper[-1]) else None
            self.middle_band = middle[-1]
            self.lower_band = lower[-1] if not np.isnan(lower[-1]) else None
            self.current_value = self.middle_band
            self.values.append(self.current_value)
            return self.current_value
        return None
    
    def calculate_from_dataframe(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands from DataFrame"""
        np_close = df['close'].values.astype(float)
        upper, middle, lower = talib.BBANDS(
            np_close,
            timeperiod=self.period,
            nbdevup=self.nbdevup,
            nbdevdn=self.nbdevdn
        )
        return {
            'upper': pd.Series(upper, index=df.index),
            'middle': pd.Series(middle, index=df.index),
            'lower': pd.Series(lower, index=df.index)
        }

