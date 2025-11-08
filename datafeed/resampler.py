"""
Data resampling functionality for strategies.
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from .base import OHLCVData
import logging

logger = logging.getLogger(__name__)


class DataResampler:
    """Resample OHLCV data to different timeframes"""
    
    def __init__(self, timeframe: str = '1min'):
        """
        Initialize resampler.
        
        Args:
            timeframe: Target timeframe (e.g., '1min', '5min', '15min', '1H', '1D')
        """
        self.timeframe = timeframe
        self.buffer: Dict[str, List[OHLCVData]] = {}  # symbol -> list of OHLCV data
    
    def add_data(self, symbol: str, data: OHLCVData):
        """Add new data point to buffer"""
        if symbol not in self.buffer:
            self.buffer[symbol] = []
        self.buffer[symbol].append(data)
    
    def get_resampled_data(self, symbol: str) -> Optional[OHLCVData]:
        """
        Get resampled OHLCV data if a complete candle is formed.
        
        Returns:
            Resampled OHLCVData or None if candle is not complete
        """
        if symbol not in self.buffer or len(self.buffer[symbol]) == 0:
            return None
        
        # Convert to DataFrame for resampling
        df = self._to_dataframe(self.buffer[symbol])
        if df is None or len(df) == 0:
            return None
        
        # Resample based on timeframe
        resampled = self._resample_dataframe(df, self.timeframe)
        
        if resampled is None or len(resampled) == 0:
            return None
        
        # Get the latest resampled candle
        latest = resampled.iloc[-1]
        
        # Create OHLCVData
        resampled_data = OHLCVData(
            symbol=symbol,
            open=latest['open'],
            high=latest['high'],
            low=latest['low'],
            close=latest['close'],
            volume=int(latest['volume']),
            timestamp=latest.name if hasattr(latest, 'name') else datetime.now()
        )
        
        # Clear processed data from buffer (keep only the last incomplete candle)
        self._clean_buffer(symbol, resampled_data.timestamp)
        
        return resampled_data
    
    def _to_dataframe(self, data_list: List[OHLCVData]) -> Optional[pd.DataFrame]:
        """Convert list of OHLCVData to DataFrame"""
        if not data_list:
            return None
        
        records = []
        for data in data_list:
            records.append({
                'timestamp': data.timestamp,
                'open': data.open,
                'high': data.high,
                'low': data.low,
                'close': data.close,
                'volume': data.volume
            })
        
        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df
    
    def _resample_dataframe(self, df: pd.DataFrame, timeframe: str) -> Optional[pd.DataFrame]:
        """Resample DataFrame to target timeframe"""
        try:
            # Map timeframe string to pandas offset
            timeframe_map = {
                '1min': '1min',
                '5min': '5min',
                '15min': '15min',
                '30min': '30min',
                '1H': '1H',
                '4H': '4H',
                '1D': '1D'
            }
            
            offset = timeframe_map.get(timeframe, '1min')
            
            # Resample OHLCV data
            resampled = df.resample(offset).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            
            # Drop incomplete candles (last row might be incomplete)
            # For now, return all candles - caller should handle incomplete ones
            return resampled.dropna()
        except Exception as e:
            logger.error(f"Error resampling data: {e}")
            return None
    
    def _clean_buffer(self, symbol: str, last_complete_timestamp: datetime):
        """Clean buffer, keeping only data after last complete candle"""
        if symbol not in self.buffer:
            return
        
        # Keep only data after the last complete timestamp
        self.buffer[symbol] = [
            data for data in self.buffer[symbol]
            if data.timestamp > last_complete_timestamp
        ]
    
    def reset(self, symbol: Optional[str] = None):
        """Reset buffer for symbol or all symbols"""
        if symbol:
            if symbol in self.buffer:
                self.buffer[symbol] = []
        else:
            self.buffer = {}

