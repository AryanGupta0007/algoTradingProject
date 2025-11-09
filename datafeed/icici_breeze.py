"""
ICICI Breeze API data feed integration.

Refactored to use Breeze Socket.IO streaming as per working reference:
- REST auth: https://api.icicidirect.com/breezeapi/api/v1/customerdetails
- Socket.IO server: https://breezeapi.icicidirect.com/ with path 'ohlcvstream'
- Subscribe by emitting 'join' with topic strings like '4.1!<TOKEN>'
"""
import base64
import csv
import json
import logging
import threading
from datetime import datetime
from typing import Dict, List, Callable, Optional

import requests
import socketio
import pandas as pd

from .base import DataFeed, OHLCVData, TickData
 

logger = logging.getLogger(__name__)


class ICICIBreezeDataFeed(DataFeed):
    """ICICI Breeze API data feed via Socket.IO"""
    
    def __init__(self, api_key: str, api_secret: str, session_token: str, base_url: str, websocket_url: str):
        super().__init__()
        self.api_key = api_key
        self.api_secret = api_secret
        self.session_token = session_token
        # Correct endpoints (override potentially incorrect provided URLs)
        self.base_url = "https://api.icicidirect.com/breezeapi/api/v1"
        self.socket_base_url = "https://breezeapi.icicidirect.com/"
        self.socket_path = "ohlcvstream"

        # Subscriptions and callbacks
        self.subscribed_symbols: List[str] = []
        self.symbols: List[str] = []
        self.callbacks: Dict[str, Callable] = {}

        # Token mapping
        self.symbol_to_token: Dict[str, str] = {}
        self.token_to_symbol: Dict[str, str] = {}
        self.code_to_symbol: Dict[str, str] = {}
        # CSVs
        # ICICIFULL.csv: maps trading symbol (ExchangeCode) -> ShortName (code)
        # ICICIFULL_socket.csv: maps code (SM) -> token (TK) [optionally EC/SR filters]
        self.full_mapping_csv: str = "ICICIFULL.csv"
        self.socket_mapping_csv: str = "ICICIFULL_socket.csv"

        # Socket.IO client
        self.sio: Optional[socketio.Client] = None
        self.sio_user: Optional[str] = None
        self.sio_token: Optional[str] = None
        self.sio_connected = False

    def fetch_historical(self, symbol: str, timeframe: str = '1min', limit: int = 30):
        """Fetch real historical 1-minute bars directly via Breeze SDK if available.

        - Maps symbol -> code using ICICIFULL.csv ("ExchangeCode" -> "ShortName").
        - Uses BreezeConnect.get_historical_data_v2 to fetch the last trading day's
          intraday data (9:15 to 15:30), falling back up to 5 previous business days.
        - Returns at most `limit` OHLCVData items, ordered oldest->newest.
        """
        logger.info(f"[Hist] Start fetch for {symbol} (limit={limit}, tf={timeframe})")
        try:
            from breeze_connect import BreezeConnect  # type: ignore
        except Exception as e1:
            try:
                # Some installs expose class under package submodule
                from breeze_connect.breeze_connect import BreezeConnect  # type: ignore
                logger.info("[Hist] Imported BreezeConnect via fallback path (breeze_connect.breeze_connect)")
            except Exception as e2:
                logger.info(f"[Hist] BreezeConnect import failed; SDK not usable. primary={repr(e1)}, fallback={repr(e2)}")
                return []

        # Find code for symbol from ICICIFULL.csv
        code = None
        try:
            with open(self.full_mapping_csv, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    # Handle quoted/space-padded headers
                    ec = r.get(' "ExchangeCode"') or r.get('ExchangeCode') or r.get('ExchangeCode ') or r.get(' "ExchangeCode" ')
                    sn = r.get(' "ShortName"') or r.get('ShortName') or r.get('ShortName ') or r.get(' "ShortName" ')
                    if ec and ec.strip().strip('"').upper() == symbol.upper():
                        code = (sn or '').strip().strip('"')
                        print('codeedd ', code)
                        break
        except Exception as e:
            logger.info(f"[Hist] Failed reading {self.full_mapping_csv} for symbol mapping: {e}")
        if not code:
            logger.info(f"[Hist] No code found for {symbol}; skipping")
            return []
        logger.info(f"[Hist] Mapped {symbol} -> code {code}")

        # Init Breeze client
        try:
            client = BreezeConnect(api_key=self.api_key)
            # Some SDKs expect method names slightly different; try common variant
            try:
                client.generate_session(api_secret=self.api_secret, session_token=self.session_token)
            except Exception:
                # Older signature
                client.generate_session(api_secret=self.api_secret, session_token=self.session_token)
            logger.info("[Hist] BreezeConnect session initialized")
        except Exception as e:
            logger.info(f"[Hist] Failed to initialize BreezeConnect: {e}")
            return []

        import datetime as _dt
        def is_bday(d):
            return d.weekday() < 5

        # Try up to 5 previous business days
        bars: List[OHLCVData] = []
        lookback = 0
        tries = 0
        while tries < 5 and len(bars) < limit:
            tries += 1
            target = _dt.datetime.now() - _dt.timedelta(days=lookback + 1)
            lookback += 1
            if not is_bday(target):
                continue
            start_dt = _dt.datetime(target.year, target.month, target.day, 9, 15)
            end_dt = _dt.datetime(target.year, target.month, target.day, 15, 30)
            from_date = start_dt.isoformat()[:19] + '.000Z'
            to_date = end_dt.isoformat()[:19] + '.000Z'
            logger.debug(f"[Hist] Try {tries}: {symbol}/{code} {from_date} -> {to_date}")
            try:
                resp = client.get_historical_data_v2(
                    interval="1minute",
                    from_date=from_date,
                    to_date=to_date,
                    stock_code=code,
                    exchange_code="NSE",
                    product_type="cash",
                )
            except Exception as e:
                logger.info(f"[Hist] SDK call failed for {symbol}/{code}: {e}")
                continue

            if not resp or not isinstance(resp, dict):
                logger.debug(f"[Hist] Empty/non-dict response for {symbol}/{code} on {target.date()}")
                continue
            data = resp.get('Success') or resp.get('success')
            if not data:
                logger.debug(f"[Hist] No 'Success' key in response for {symbol}/{code} on {target.date()}: {str(resp)[:200]}")
                continue
            try:
                df = pd.DataFrame(data)
                if df.empty:
                    logger.debug(f"[Hist] DataFrame empty for {symbol}/{code} on {target.date()}")
                    continue
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                df = df.dropna(subset=['datetime']).sort_values('datetime')
                logger.info(f"[Hist] Retrieved {len(df)} rows for {symbol} on {target.date()}")
                # Keep only needed columns
                needed = ['open', 'high', 'low', 'close', 'volume', 'datetime']
                for col in needed:
                    if col not in df.columns:
                        df[col] = None
                # Convert last `limit - len(bars)` rows
                take = max(0, limit - len(bars))
                for _, row in df.tail(take).iterrows():
                    try:
                        bars.append(OHLCVData(
                            symbol=symbol,
                            open=float(row['open']),
                            high=float(row['high']),
                            low=float(row['low']),
                            close=float(row['close']),
                            volume=int(float(row['volume'])) if pd.notna(row['volume']) else 0,
                            timestamp=row['datetime'].to_pydatetime() if hasattr(row['datetime'], 'to_pydatetime') else row['datetime'],
                        ))
                    except Exception:
                        continue
            except Exception as e:
                logger.info(f"[Hist] Failed processing Breeze historical for {symbol}: {e}")
                continue

        # Ensure oldest->newest and trim to limit
        if bars:
            bars = sorted(bars, key=lambda x: x.timestamp)[:limit]
        logger.info(f"[Hist] Built {len(bars)} bars for {symbol}")
        return bars
    
    def connect(self) -> bool:
        """Authenticate and prepare Socket.IO client. Actual network connect occurs on first subscribe."""
        try:
            payload = json.dumps({"SessionToken": self.session_token, "AppKey": self.api_key})
            headers = {"Content-Type": "application/json"}
            url = f"{self.base_url}/customerdetails"
            resp = requests.get(url, headers=headers, data=payload, timeout=10)
            logger.debug(f"[BREEZE] customerdetails request sent to {url}, status={resp.status_code}")
            # Log short headers/body for debugging (avoid secrets)
            try:
                logger.debug(f"[BREEZE] customerdetails headers: { {k: v for k, v in resp.headers.items()} }")
            except Exception:
                pass
            try:
                logger.debug(f"[BREEZE] customerdetails body preview: {resp.text[:400] if resp.text else ''}")
            except Exception:
                pass
            resp.raise_for_status()
            try:
                auth_data = resp.json() or {}
            except Exception:
                logger.error(f"customerdetails non-JSON response: {resp.text[:300]}")
                return False

            # Try common casings/paths
            session_token_b64 = None
            if isinstance(auth_data, dict):
                container = auth_data.get("Success") or auth_data.get("success") or auth_data
                if isinstance(container, dict):
                    session_token_b64 = container.get("session_token") or container.get("SessionToken")
            try:
                logger.debug(f"[BREEZE] Parsed auth_data keys: {list(auth_data.keys()) if isinstance(auth_data, dict) else type(auth_data)}")
            except Exception:
                pass

            if not session_token_b64:
                logger.error(f"Missing session_token in auth response: {str(auth_data)[:300]}")
                return False

            try:
                decoded = base64.b64decode(session_token_b64.encode()).decode()
                user_id, sio_token = decoded.split(":", 1)
            except Exception as de:
                logger.error(f"Failed to decode session_token: {de}")
                return False

            self.sio_user = user_id
            self.sio_token = sio_token

            # Prepare Socket.IO client with handlers
            self._init_socketio_client()
            self.is_running = True
            logger.info("Authenticated with Breeze and initialized Socket.IO client")
            return True
        except Exception as e:
            logger.error(f"Error connecting to ICICI Breeze API: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from ICICI Breeze API"""
        self.is_running = False
        try:
            if self.sio and self.sio_connected:
                self.sio.disconnect()
        except Exception:
            pass
        finally:
            self.sio_connected = False
            logger.info("Disconnected from ICICI Breeze API")
    
    def subscribe(self, symbols: List[str], callback: Callable):
        """Subscribe to symbols via Socket.IO (1-minute OHLCV)."""
        # Track callbacks and symbol list
        for symbol in symbols:
            if symbol not in self.subscribed_symbols:
                self.subscribed_symbols.append(symbol)
            if symbol not in self.symbols:
                self.symbols.append(symbol)
            self.callbacks[symbol] = callback

        # Ensure tokens for these symbols
        self._ensure_tokens(symbols)

        # Connect Socket.IO if needed
        if self.sio and not self.sio_connected:
            try:
                self.sio.connect(
                    self.socket_base_url,
                    socketio_path=self.socket_path,
                    headers={"User-Agent": "python-socketio[client]/socket"},
                    auth={"user": self.sio_user, "token": self.sio_token},
                    transports=["websocket"],
                )
                # Connection event handler will emit joins for all known tokens
            except Exception as e:
                logger.error(f"Socket.IO connection failed: {e}")
                return

        # If already connected, emit join for any new tokens immediately
        if self.sio_connected:
            for symbol in symbols:
                tk = self.symbol_to_token.get(symbol)
                if tk:
                    try:
                        self.sio.emit("join", [f"4.1!{tk}"])
                        logger.info(f"[BREEZE] Subscribed token {tk} for {symbol}")
                    except Exception as e:
                        logger.error(f"[BREEZE] Failed joining {symbol}/{tk}: {e}")
        logger.info(f"Subscribed to symbols: {symbols}")
    
    def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from symbols (client does not provide explicit part/remove; we stop forwarding)."""
        for symbol in symbols:
            if symbol in self.subscribed_symbols:
                self.subscribed_symbols.remove(symbol)
            if symbol in self.symbols:
                self.symbols.remove(symbol)
            if symbol in self.callbacks:
                del self.callbacks[symbol]
        logger.info(f"Unsubscribed from symbols: {symbols}")
    
    def _init_socketio_client(self):
        """Initialize Socket.IO client and register event handlers."""

        self.sio = socketio.Client(reconnection=True, reconnection_attempts=5, reconnection_delay=3)

        @self.sio.event
        def connect():
            logger.info("Socket.IO connected to Breeze")
            self.sio_connected = True
            # Subscribe to all known tokens
            for sym, tk in list(self.symbol_to_token.items()):
                try:
                    self.sio.emit("join", [f"4.1!{tk}"])
                    logger.info(f"[BREEZE] Subscribed token {tk} for {sym}")
                except Exception as e:
                    logger.error(f"[BREEZE] Failed joining {sym}/{tk}: {e}")

        @self.sio.event
        def disconnect():
            self.sio_connected = False
            logger.info("Socket.IO disconnected from Breeze")

        @self.sio.event
        def connect_error(data):
            logger.error(f"[BREEZE] Connection error: {data}")

        # 1-minute candle updates event
        @self.sio.on("1MIN")
        def on_1min(payload):
            try:
                # Expected CSV payload; parse parts
                parts = payload.split(',') if isinstance(payload, str) else []
                if len(parts) < 9:
                    logger.debug(f"[BREEZE] Incomplete payload: {str(payload)[:100]}")
                    return

                ident = parts[1]
                dt_str = parts[7]
                try:
                    ts = datetime.fromisoformat(dt_str)
                except Exception:
                    # Fallback: ignore if timestamp invalid
                    return

                # Prices order based on sample: o=parts[4], h=parts[3], l=parts[2], c=parts[5], v=parts[6]
                o = float(parts[4])
                h = float(parts[3])
                l = float(parts[2])
                c = float(parts[5])
                try:
                    v = int(float(parts[6]))
                except Exception:
                    v = 0

                # Resolve symbol by token or by code
                symbol = self.token_to_symbol.get(ident) or self.code_to_symbol.get(ident)
                if not symbol:
                    # Unknown token; skip
                    return
                logger.debug(f"[BREEZE] Ident '{ident}' mapped to symbol '{symbol}'")

                ohlcv = OHLCVData(
                    symbol=symbol,
                    open=o,
                    high=h,
                    low=l,
                    close=c,
                    volume=v,
                    timestamp=ts,
                )

                # Deliver to per-symbol callback if present
                cb = self.callbacks.get(symbol)
                if cb:
                    try:
                        cb(ohlcv)
                    except Exception as e:
                        logger.error(f"Error delivering OHLCV to callback: {e}")
                # Notify generic subscribers
                self._notify_subscribers(ohlcv)
            except Exception as e:
                logger.error(f"[BREEZE] Tick processing error: {e}")

    def _ensure_tokens(self, symbols: List[str]):
        """Ensure tokens exist: symbol -> code (ICICIFULL.csv) -> token (ICICIFULL_socket.csv)."""
        # Load FULL mapping (symbol -> code)
        full_rows = []
        try:
            with open(self.full_mapping_csv, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    full_rows.append(r)
        except Exception as e:
            logger.warning(f"Could not read full mapping CSV {self.full_mapping_csv}: {e}")

        # Load socket mapping (code -> token)
        socket_rows = []
        try:
            with open(self.socket_mapping_csv, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    socket_rows.append(r)
        except Exception as e:
            logger.warning(f"Could not read socket mapping CSV {self.socket_mapping_csv}: {e}")

        # Helper to normalize header names seen in user's CSVs (with quotes/spaces)
        def get_full_symbol(row):
            # ' "ExchangeCode"' may include leading space and quotes; try multiple keys
            for k in ("ExchangeCode", ' "ExchangeCode"', 'ExchangeCode ', ' "ExchangeCode" '):
                if k in row:
                    return (row.get(k) or '').strip().strip('"')
            return ''

        def get_full_code(row):
            # ' "ShortName"' may include leading space and quotes
            for k in ("ShortName", ' "ShortName"', 'ShortName ', ' "ShortName" '):
                if k in row:
                    return (row.get(k) or '').strip().strip('"')
            return ''

        def get_socket_code(row):
            for k in ("SM", "Symbol", "SYMBOL", "symbol"):
                if k in row:
                    return (row.get(k) or '').strip().strip('"')
            return ''

        def get_socket_token(row):
            for k in ("TK", "Token", "token"):
                if k in row:
                    return (row.get(k) or '').strip().strip('"')
            return ''

        # Build lookups
        symbol_to_code: Dict[str, str] = {}
        for r in full_rows:
            sym = get_full_symbol(r).upper()
            code = get_full_code(r)
            if sym and code:
                symbol_to_code[sym] = code

        code_to_token: Dict[str, str] = {}
        for r in socket_rows:
            code = get_socket_code(r)
            tk = get_socket_token(r)
            if code and tk:
                code_to_token[code] = tk

        for sym in symbols:
            if sym in self.symbol_to_token:
                continue
            code = symbol_to_code.get(sym.upper())
            if not code:
                logger.warning(f"No code found for symbol {sym} in {self.full_mapping_csv}")
                continue
            tk = code_to_token.get(code)
            if not tk:
                logger.warning(f"No token found for code {code} (symbol {sym}) in {self.socket_mapping_csv}")
                continue
            self.symbol_to_token[sym] = tk
            self.token_to_symbol[tk] = sym
            self.code_to_symbol[code] = sym
            logger.info(f"[TOKEN] {sym} -> code {code} -> token {tk}")

