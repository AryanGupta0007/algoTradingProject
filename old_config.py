"""
Configuration settings for the paper trading system.
"""
from dataclasses import dataclass
from typing import Optional
import os
from dotenv import load_dotenv
load_dotenv()


@dataclass
class ICICIConfig:
    """ICICI Breeze API configuration"""
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    session_token: Optional[str] = None
    base_url: str = "https://api.icicidirect.com/breezeapi/v1"
    websocket_url: str = "wss://api.icicidirect.com/breezeapi/v1"
    enabled: bool = False  # Set to True to use real API


@dataclass
class TradingConfig:
    """Trading system configuration"""
    initial_capital: float = 100000.0
    max_position_size: float = 0.1  # 10% of capital per position
    max_total_exposure: float = 0.5  # 50% of capital
    commission_rate: float = 0.001  # 0.1% commission
    slippage: float = 0.0001  # 0.01% slippage
    enable_rms: bool = True
    log_level: str = "DEBUG"  # Changed to DEBUG for detailed logging
    log_file: str = "trading.log"
    db_path: str = "trading.db"  # SQLite database path
    enable_db: bool = True  # Enable database persistence
    save_market_data: bool = True  # Save market data to database (can be large)
    fake_data_interval: int = 0  # Fake data feed interval in seconds (default: 60)
    portfolio_snapshot_interval: int = 5  # Portfolio snapshot interval in seconds


@dataclass
class Config:
    """Main configuration class"""
    icici: ICICIConfig = None
    trading: TradingConfig = None
    
    def __post_init__(self):
        if self.icici is None:
            self.icici = ICICIConfig(
                api_key=os.getenv("BREEZE_API_KEY"),
                api_secret=os.getenv("BREEZE_SECRET"),
                session_token=os.getenv("MY_API_SESSION"),
                enabled=os.getenv("ICICI_ENABLED", "False").lower() == "true"
            )
        if self.trading is None:
            self.trading = TradingConfig()

             