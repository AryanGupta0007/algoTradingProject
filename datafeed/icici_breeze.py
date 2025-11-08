"""
ICICI Breeze API data feed integration.
"""
import requests
import websocket
import json
import threading
from datetime import datetime
from typing import List, Callable, Optional
from .base import DataFeed, OHLCVData, TickData
import logging

logger = logging.getLogger(__name__)


class ICICIBreezeDataFeed(DataFeed):
    """ICICI Breeze API data feed"""
    
    def __init__(self, api_key: str, api_secret: str, session_token: str, base_url: str, websocket_url: str):
        super().__init__()
        self.api_key = api_key
        self.api_secret = api_secret
        self.session_token = session_token
        self.base_url = base_url
        self.websocket_url = websocket_url
        self.subscribed_symbols: List[str] = []
        self.callbacks: dict = {}
        self.ws: Optional[websocket.WebSocketApp] = None
        self.ws_thread: Optional[threading.Thread] = None
    
    def connect(self) -> bool:
        """Connect to ICICI Breeze API"""
        try:
            # Verify session token
            headers = {
                'Content-Type': 'application/json',
                'X-SessionToken': self.session_token,
                'X-APIKey': self.api_key
            }
            
            # Test connection
            response = requests.get(
                f"{self.base_url}/customer/v1",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                self.is_running = True
                logger.info("Connected to ICICI Breeze API")
                return True
            else:
                logger.error(f"Failed to connect to ICICI Breeze API: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error connecting to ICICI Breeze API: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from ICICI Breeze API"""
        self.is_running = False
        if self.ws:
            self.ws.close()
        logger.info("Disconnected from ICICI Breeze API")
    
    def subscribe(self, symbols: List[str], callback: Callable):
        """Subscribe to symbols via WebSocket"""
        for symbol in symbols:
            if symbol not in self.subscribed_symbols:
                self.subscribed_symbols.append(symbol)
            self.callbacks[symbol] = callback
        
        # Start WebSocket connection if not already started
        if not self.ws:
            self._start_websocket()
        
        logger.info(f"Subscribed to symbols: {symbols}")
    
    def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from symbols"""
        for symbol in symbols:
            if symbol in self.subscribed_symbols:
                self.subscribed_symbols.remove(symbol)
            if symbol in self.callbacks:
                del self.callbacks[symbol]
        logger.info(f"Unsubscribed from symbols: {symbols}")
    
    def _start_websocket(self):
        """Start WebSocket connection for real-time data"""
        def on_message(ws, message):
            try:
                data = json.loads(message)
                # Process ICICI Breeze data format
                # Adjust based on actual API response structure
                symbol = data.get('symbol')
                if symbol and symbol in self.callbacks:
                    # Convert to OHLCVData or TickData based on data type
                    if 'open' in data:
                        ohlcv = OHLCVData(
                            symbol=symbol,
                            open=float(data['open']),
                            high=float(data['high']),
                            low=float(data['low']),
                            close=float(data['close']),
                            volume=int(data.get('volume', 0)),
                            timestamp=datetime.fromisoformat(data['timestamp'])
                        )
                        self.callbacks[symbol](ohlcv)
                        self._notify_subscribers(ohlcv)
                    elif 'price' in data:
                        tick = TickData(
                            symbol=symbol,
                            price=float(data['price']),
                            volume=int(data.get('volume', 0)),
                            timestamp=datetime.fromisoformat(data['timestamp']),
                            bid=data.get('bid'),
                            ask=data.get('ask')
                        )
                        self.callbacks[symbol](tick)
                        self._notify_subscribers(tick)
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
        
        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            logger.info("WebSocket connection closed")
            if self.is_running:
                # Attempt to reconnect
                threading.Timer(5.0, self._start_websocket).start()
        
        def on_open(ws):
            logger.info("WebSocket connection opened")
            # Subscribe to all symbols
            for symbol in self.subscribed_symbols:
                subscribe_msg = {
                    "action": "subscribe",
                    "symbol": symbol,
                    "sessionToken": self.session_token,
                    "apiKey": self.api_key
                }
                ws.send(json.dumps(subscribe_msg))
        
        # Create WebSocket connection
        ws_url = f"{self.websocket_url}/realtime?sessionToken={self.session_token}&apiKey={self.api_key}"
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        # Run WebSocket in separate thread
        self.ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        self.ws_thread.start()

