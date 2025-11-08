"""
SQLite database manager for trading system.
"""
import sqlite3
import json
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging
import threading
import uuid

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite database for persistent storage"""
    
    def __init__(self, db_path: str = "trading.db"):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        # Lock to serialize access to sqlite connection across threads
        self._lock = threading.Lock()
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database and create tables"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        self._create_tables()
        logger.info(f"Database initialized: {self.db_path}")
    
    def _create_tables(self):
        """Create all necessary tables"""
        cursor = self.conn.cursor()
        
        # Orders table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                order_type TEXT NOT NULL,
                price REAL,
                stop_price REAL,
                status TEXT NOT NULL,
                filled_quantity INTEGER DEFAULT 0,
                average_price REAL,
                timestamp TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                stop_loss REAL,
                take_profit REAL,
                created_at TEXT NOT NULL
            )
        """)
        
        # Fills table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fills (
                fill_id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                commission REAL NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (order_id) REFERENCES orders(order_id)
            )
        """)
        
        # Positions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                position_id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                average_price REAL NOT NULL,
                current_price REAL NOT NULL,
                stop_loss REAL,
                take_profit REAL,
                strategy_id TEXT NOT NULL,
                opened_at TEXT,
                last_updated TEXT NOT NULL,
                UNIQUE(symbol, strategy_id)
            )
        """)
        
        # Portfolio state table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cash REAL NOT NULL,
                total_equity REAL NOT NULL,
                total_value REAL NOT NULL,
                total_realized_pnl REAL NOT NULL,
                total_unrealized_pnl REAL NOT NULL,
                total_pnl REAL NOT NULL,
                total_commission REAL NOT NULL,
                exposure REAL NOT NULL,
                initial_capital REAL NOT NULL
            )
        """)
        
        # Daily metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL UNIQUE,
                opening_equity REAL NOT NULL,
                closing_equity REAL NOT NULL,
                high_equity REAL NOT NULL,
                low_equity REAL NOT NULL,
                realized_pnl REAL NOT NULL,
                unrealized_pnl REAL NOT NULL,
                total_pnl REAL NOT NULL,
                commissions REAL NOT NULL,
                trades_count INTEGER NOT NULL,
                winning_trades INTEGER NOT NULL,
                losing_trades INTEGER NOT NULL,
                max_drawdown REAL NOT NULL,
                peak_equity REAL NOT NULL,
                win_rate REAL NOT NULL
            )
        """)
        
        # Strategy metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                total_trades INTEGER NOT NULL,
                winning_trades INTEGER NOT NULL,
                losing_trades INTEGER NOT NULL,
                realized_pnl REAL NOT NULL,
                unrealized_pnl REAL NOT NULL,
                total_pnl REAL NOT NULL,
                commissions REAL NOT NULL,
                max_drawdown REAL NOT NULL,
                peak_equity REAL NOT NULL,
                current_position_value REAL NOT NULL,
                win_rate REAL NOT NULL,
                avg_trade_pnl REAL NOT NULL,
                UNIQUE(strategy_id, timestamp)
            )
        """)
        
        # Equity history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS equity_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                equity REAL NOT NULL
            )
        """)
        
        # Market data table (optional, for OHLCV storage)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                UNIQUE(symbol, timestamp)
            )
        """)
        
        # Trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                trade_type TEXT NOT NULL,
                status TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_time TEXT,
                exit_price REAL,
                stop_loss REAL,
                take_profit REAL,
                ltp REAL,
                UNIQUE(order_id)
            )
        """)
        # Backward-compat: attempt to add trade_type if table existed without it
        try:
            cursor.execute("ALTER TABLE trades ADD COLUMN trade_type TEXT")
        except Exception:
            pass
        
        # Create indexes for better query performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_strategy ON orders(strategy_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_timestamp ON orders(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fills_order_id ON fills(order_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fills_timestamp ON fills(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_strategy ON positions(strategy_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_timestamp ON portfolio_state(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_metrics_date ON daily_metrics(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_strategy_metrics_strategy ON strategy_metrics(strategy_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_strategy_metrics_timestamp ON strategy_metrics(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_equity_history_timestamp ON equity_history(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol_status ON trades(symbol, status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_strategy_status ON trades(strategy_id, status)")
        
        self.conn.commit()
        logger.info("Database tables created successfully")
    
    def save_order(self, order: Dict[str, Any]):
        """Save order to database"""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO orders (
                order_id, symbol, side, quantity, order_type, price, stop_price,
                status, filled_quantity, average_price, timestamp, strategy_id,
                stop_loss, take_profit, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            order['order_id'],
            order['symbol'],
            order['side'],
            order['quantity'],
            order['order_type'],
            order.get('price'),
            order.get('stop_price'),
            order['status'],
            order.get('filled_quantity', 0),
            order.get('average_price'),
            order['timestamp'],
            order['strategy_id'],
            order.get('stop_loss'),
            order.get('take_profit'),
            datetime.now().isoformat()
        ))
            self.conn.commit()
    
    def save_fill(self, fill: Dict[str, Any]):
        """Save fill to database"""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
            INSERT INTO fills (
                order_id, symbol, side, quantity, price, commission, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            fill['order_id'],
            fill['symbol'],
            fill['side'],
            fill['quantity'],
            fill['price'],
            fill.get('commission', 0.0),
            fill['timestamp']
        ))
            self.conn.commit()
            return cursor.lastrowid

    def save_fill_order_and_position_transaction(self, fill: Dict[str, Any], order: Optional[Dict[str, Any]],
                                                position: Optional[Dict[str, Any]], portfolio: Optional[Dict[str, Any]]):
        """Save fill, update order and upsert/delete position and save portfolio state in a single transaction.

        This helps keep in-memory portfolio state and database state consistent by avoiding partial writes
        when a crash or interleaved thread interrupts the sequence.
        """
        transaction_id = uuid.uuid4().hex
        logger.debug("DB tx start %s fill=%s order=%s", transaction_id, fill.get('order_id'), order.get('order_id') if order else None)

        with self._lock:
            cursor = self.conn.cursor()
            try:
                # Begin transaction
                cursor.execute('BEGIN')

                # Insert fill
                cursor.execute(
                    """
                    INSERT INTO fills (
                        order_id, symbol, side, quantity, price, commission, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        fill['order_id'],
                        fill['symbol'],
                        fill['side'],
                        fill['quantity'],
                        fill['price'],
                        fill.get('commission', 0.0),
                        fill['timestamp'],
                    ),
                )

                # Upsert order if provided
                if order:
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO orders (
                            order_id, symbol, side, quantity, order_type, price, stop_price,
                            status, filled_quantity, average_price, timestamp, strategy_id,
                            stop_loss, take_profit, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            order['order_id'],
                            order['symbol'],
                            order['side'],
                            order['quantity'],
                            order['order_type'],
                            order.get('price'),
                            order.get('stop_price'),
                            order['status'],
                            order.get('filled_quantity', 0),
                            order.get('average_price'),
                            order.get('timestamp'),
                            order.get('strategy_id', ''),
                            order.get('stop_loss'),
                            order.get('take_profit'),
                            datetime.now().isoformat(),
                        ),
                    )

                # Upsert or delete position depending on whether it's flat
                if position:
                    qty = position.get('quantity', 0)
                    if qty == 0:
                        cursor.execute(
                            """
                            DELETE FROM positions WHERE symbol = ? AND strategy_id = ?
                            """,
                            (position.get('symbol'), position.get('strategy_id', '')),
                        )
                    else:
                        cursor.execute(
                            """
                            INSERT OR REPLACE INTO positions (
                                symbol, quantity, average_price, current_price, stop_loss,
                                take_profit, strategy_id, opened_at, last_updated
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                position['symbol'],
                                position['quantity'],
                                position['average_price'],
                                position['current_price'],
                                position.get('stop_loss'),
                                position.get('take_profit'),
                                position.get('strategy_id', ''),
                                position.get('opened_at'),
                                position.get('last_updated', datetime.now().isoformat()),
                            ),
                        )

                # Save a portfolio state snapshot if provided
                if portfolio:
                    cursor.execute(
                        """
                        INSERT INTO portfolio_state (
                            timestamp, cash, total_equity, total_value, total_realized_pnl,
                            total_unrealized_pnl, total_pnl, total_commission, exposure, initial_capital
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            datetime.now().isoformat(),
                            portfolio['cash'],
                            portfolio['total_equity'],
                            portfolio['total_value'],
                            portfolio['total_realized_pnl'],
                            portfolio['total_unrealized_pnl'],
                            portfolio['total_pnl'],
                            portfolio['total_commission'],
                            portfolio['exposure'],
                            portfolio['initial_capital'],
                        ),
                    )

                # Commit transaction
                self.conn.commit()
                logger.debug("DB tx commit %s fill=%s", transaction_id, fill.get('order_id'))
            except Exception:
                # Rollback on error
                try:
                    self.conn.rollback()
                except Exception:
                    logger.exception("Failed rollback for tx %s", transaction_id)
                logger.exception("DB tx failed %s", transaction_id)
                raise
    
    def save_position(self, position: Dict[str, Any]):
        """Save or update position"""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO positions (
                symbol, quantity, average_price, current_price, stop_loss,
                take_profit, strategy_id, opened_at, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            position['symbol'],
            position['quantity'],
            position['average_price'],
            position['current_price'],
            position.get('stop_loss'),
            position.get('take_profit'),
            position.get('strategy_id', ''),
            position.get('opened_at'),
            position.get('last_updated', datetime.now().isoformat())
        ))
            self.conn.commit()
    
    def delete_position(self, symbol: str, strategy_id: str):
        """Delete position (when closed)"""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                DELETE FROM positions WHERE symbol = ? AND strategy_id = ?
            """, (symbol, strategy_id))
            self.conn.commit()
    
    def save_portfolio_state(self, portfolio: Dict[str, Any]):
        """Save portfolio state snapshot"""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
            INSERT INTO portfolio_state (
                timestamp, cash, total_equity, total_value, total_realized_pnl,
                total_unrealized_pnl, total_pnl, total_commission, exposure, initial_capital
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            portfolio['cash'],
            portfolio['total_equity'],
            portfolio['total_value'],
            portfolio['total_realized_pnl'],
            portfolio['total_unrealized_pnl'],
            portfolio['total_pnl'],
            portfolio['total_commission'],
            portfolio['exposure'],
            portfolio['initial_capital']
        ))
            self.conn.commit()
    
    def save_daily_metrics(self, metrics: Dict[str, Any]):
        """Save or update daily metrics"""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO daily_metrics (
                date, opening_equity, closing_equity, high_equity, low_equity,
                realized_pnl, unrealized_pnl, total_pnl, commissions,
                trades_count, winning_trades, losing_trades, max_drawdown,
                peak_equity, win_rate
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metrics['date'],
            metrics['opening_equity'],
            metrics['closing_equity'],
            metrics['high_equity'],
            metrics['low_equity'],
            metrics['realized_pnl'],
            metrics['unrealized_pnl'],
            metrics['total_pnl'],
            metrics['commissions'],
            metrics['trades_count'],
            metrics['winning_trades'],
            metrics['losing_trades'],
            metrics['max_drawdown'],
            metrics['peak_equity'],
            metrics.get('win_rate', 0.0)
        ))
            self.conn.commit()
    
    def save_strategy_metrics(self, metrics: Dict[str, Any]):
        """Save strategy metrics snapshot"""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO strategy_metrics (
                strategy_id, symbol, timestamp, total_trades, winning_trades,
                losing_trades, realized_pnl, unrealized_pnl, total_pnl,
                commissions, max_drawdown, peak_equity, current_position_value,
                win_rate, avg_trade_pnl
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metrics['strategy_id'],
            metrics['symbol'],
            datetime.now().isoformat(),
            metrics['total_trades'],
            metrics['winning_trades'],
            metrics['losing_trades'],
            metrics['realized_pnl'],
            metrics['unrealized_pnl'],
            metrics['total_pnl'],
            metrics['commissions'],
            metrics.get('max_drawdown', 0.0),
            metrics.get('peak_equity', 0.0),
            metrics.get('current_position_value', 0.0),
            metrics.get('win_rate', 0.0),
            metrics.get('avg_trade_pnl', 0.0)
        ))
            self.conn.commit()
    
    def save_equity_point(self, timestamp: datetime, equity: float):
        """Save equity history point"""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO equity_history (timestamp, equity) VALUES (?, ?)
            """, (timestamp.isoformat(), equity))
            self.conn.commit()
    
    def save_market_data(self, symbol: str, ohlcv: Dict[str, Any]):
        """Save market data (OHLCV)"""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO market_data (
                symbol, timestamp, open, high, low, close, volume
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol,
            ohlcv['timestamp'],
            ohlcv['open'],
            ohlcv['high'],
            ohlcv['low'],
            ohlcv['close'],
            ohlcv['volume']
        ))
            self.conn.commit()
    
    # Query methods
    def get_orders(self, strategy_id: Optional[str] = None, 
                   symbol: Optional[str] = None,
                   limit: int = 100) -> List[Dict[str, Any]]:
        """Get orders from database"""
        query = "SELECT * FROM orders WHERE 1=1"
        params = []

        if strategy_id:
            query += " AND strategy_id = ?"
            params.append(strategy_id)
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    # Trades API
    def create_trade(self, *, order_id: str, symbol: str, strategy_id: str,
                     trade_type: str,
                     entry_time: str, entry_price: float,
                     stop_loss: Optional[float] = None, take_profit: Optional[float] = None,
                     ltp: Optional[float] = None):
        """Insert a new trade row with status 'open'."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT OR IGNORE INTO trades (
                    order_id, symbol, strategy_id, trade_type, status, entry_time, entry_price,
                    exit_time, exit_price, stop_loss, take_profit, ltp
                ) VALUES (?, ?, ?, ?, 'open', ?, ?, NULL, NULL, ?, ?, ?)
                """,
                (order_id, symbol, strategy_id, trade_type, entry_time, entry_price, stop_loss, take_profit, ltp),
            )
            self.conn.commit()
    
    def update_trades_ltp(self, symbol: str, ltp: float):
        """Update LTP for all open trades of a symbol."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                UPDATE trades
                SET ltp = ?
                WHERE symbol = ? AND status = 'open'
                """,
                (ltp, symbol),
            )
            self.conn.commit()
    
    def close_latest_trade(self, *, symbol: str, strategy_id: str, exit_time: str,
                           exit_price: float, status: str = 'closed'):
        """Close the most recent open trade for a given symbol+strategy."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                UPDATE trades
                SET status = ?, exit_time = ?, exit_price = ?
                WHERE trade_id = (
                    SELECT trade_id FROM trades
                    WHERE symbol = ? AND strategy_id = ? AND status = 'open'
                    ORDER BY entry_time DESC
                    LIMIT 1
                )
                """,
                (status, exit_time, exit_price, symbol, strategy_id),
            )
            self.conn.commit()
    
    def get_trades(self, *, symbol: Optional[str] = None, strategy_id: Optional[str] = None,
                   status: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
        """Fetch trades with optional filters."""
        query = "SELECT * FROM trades WHERE 1=1"
        params: List[Any] = []
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if strategy_id:
            query += " AND strategy_id = ?"
            params.append(strategy_id)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY entry_time DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_fills(self, order_id: Optional[str] = None,
                  strategy_id: Optional[str] = None,
                  limit: int = 100) -> List[Dict[str, Any]]:
        """Get fills from database"""
        query = "SELECT * FROM fills WHERE 1=1"
        params = []

        if order_id:
            query += " AND order_id = ?"
            params.append(order_id)
        elif strategy_id:
            # Join with orders table
            query = """
                SELECT f.* FROM fills f
                JOIN orders o ON f.order_id = o.order_id
                WHERE o.strategy_id = ?
            """
            params.append(strategy_id)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_positions(self, strategy_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get positions from database"""
        with self._lock:
            cursor = self.conn.cursor()
            if strategy_id:
                cursor.execute("SELECT * FROM positions WHERE strategy_id = ?", (strategy_id,))
            else:
                cursor.execute("SELECT * FROM positions")
            return [dict(row) for row in cursor.fetchall()]
    
    def get_daily_metrics(self, start_date: Optional[date] = None,
                          end_date: Optional[date] = None) -> List[Dict[str, Any]]:
        """Get daily metrics from database"""
        query = "SELECT * FROM daily_metrics WHERE 1=1"
        params = []

        if start_date:
            query += " AND date >= ?"
            params.append(start_date.isoformat())
        if end_date:
            query += " AND date <= ?"
            params.append(end_date.isoformat())

        query += " ORDER BY date DESC"
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_strategy_metrics(self, strategy_id: Optional[str] = None,
                             limit: int = 100) -> List[Dict[str, Any]]:
        """Get strategy metrics from database"""
        with self._lock:
            cursor = self.conn.cursor()
            if strategy_id:
                cursor.execute("""
                    SELECT * FROM strategy_metrics 
                    WHERE strategy_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (strategy_id, limit))
            else:
                cursor.execute("""
                    SELECT * FROM strategy_metrics
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_equity_history(self, start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          limit: int = 1000) -> List[Dict[str, Any]]:
        """Get equity history from database"""
        query = "SELECT * FROM equity_history WHERE 1=1"
        params = []

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_market_data(self, symbol: str, start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None,
                       limit: int = 1000) -> List[Dict[str, Any]]:
        """Get market data from database"""
        query = "SELECT * FROM market_data WHERE symbol = ?"
        params = [symbol]

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_portfolio_state(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get portfolio state history"""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT * FROM portfolio_state
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    def close(self):
        """Close database connection"""
        if self.conn:
            with self._lock:
                try:
                    self.conn.close()
                    logger.info("Database connection closed")
                except Exception:
                    logger.exception("Error closing database connection")

