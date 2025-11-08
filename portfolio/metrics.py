"""
Portfolio metrics and performance tracking.
"""
from typing import Dict, List, Optional
from datetime import datetime, date
from dataclasses import dataclass, field
from collections import defaultdict
import pandas as pd
import threading
import time
from .portfolio import Portfolio
from order.order import Fill, OrderSide


@dataclass
class DailyMetrics:
    """Daily performance metrics"""
    date: date
    opening_equity: float = 0.0
    closing_equity: float = 0.0
    high_equity: float = 0.0
    low_equity: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_pnl: float = 0.0
    commissions: float = 0.0
    trades_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    max_drawdown: float = 0.0
    peak_equity: float = 0.0
    
    def to_dict(self):
        return {
            'date': self.date.isoformat(),
            'opening_equity': self.opening_equity,
            'closing_equity': self.closing_equity,
            'high_equity': self.high_equity,
            'low_equity': self.low_equity,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_pnl': self.total_pnl,
            'commissions': self.commissions,
            'trades_count': self.trades_count,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'max_drawdown': self.max_drawdown,
            'peak_equity': self.peak_equity,
            'win_rate': self.winning_trades / self.trades_count if self.trades_count > 0 else 0.0
        }


@dataclass
class StrategyMetrics:
    """Strategy-wise performance metrics"""
    strategy_id: str
    symbol: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_pnl: float = 0.0
    commissions: float = 0.0
    max_drawdown: float = 0.0
    peak_equity: float = 0.0
    current_position_value: float = 0.0
    
    def to_dict(self):
        return {
            'strategy_id': self.strategy_id,
            'symbol': self.symbol,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_pnl': self.total_pnl,
            'commissions': self.commissions,
            'max_drawdown': self.max_drawdown,
            'peak_equity': self.peak_equity,
            'current_position_value': self.current_position_value,
            'win_rate': self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0,
            'avg_trade_pnl': self.realized_pnl / self.total_trades if self.total_trades > 0 else 0.0
        }


class PortfolioMetrics:
    """Track and calculate portfolio metrics"""
    
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        self.daily_metrics: Dict[date, DailyMetrics] = {}
        self.strategy_metrics: Dict[str, StrategyMetrics] = {}
        self.equity_history: List[tuple] = []  # (timestamp, equity)
        self.current_date: Optional[date] = None
        self.last_equity: float = portfolio.initial_capital
        
    def update(self, timestamp: Optional[datetime] = None):
        """Update metrics"""
        if timestamp is None:
            timestamp = datetime.now()
        
        current_date = timestamp.date()
        
        # Initialize daily metrics if new day
        if current_date != self.current_date:
            if self.current_date is not None:
                # Finalize previous day
                self._finalize_daily_metrics(self.current_date)
            
            # Initialize new day
            self.current_date = current_date
            if current_date not in self.daily_metrics:
                opening_equity = self.portfolio.total_equity() if not self.daily_metrics else self.last_equity
                self.daily_metrics[current_date] = DailyMetrics(
                    date=current_date,
                    opening_equity=opening_equity,
                    peak_equity=opening_equity
                )
        
        # Update current day metrics
        daily = self.daily_metrics[current_date]
        current_equity = self.portfolio.total_equity()
        
        daily.closing_equity = current_equity
        daily.high_equity = max(daily.high_equity, current_equity)
        daily.low_equity = min(daily.low_equity, current_equity) if daily.low_equity > 0 else current_equity
        daily.unrealized_pnl = self.portfolio.total_unrealized_pnl()
        daily.realized_pnl = self.portfolio.total_realized_pnl
        daily.total_pnl = self.portfolio.total_pnl()
        daily.commissions = self.portfolio.total_commission
        
        # Update peak and drawdown
        if current_equity > daily.peak_equity:
            daily.peak_equity = current_equity
        daily.max_drawdown = max(daily.max_drawdown, daily.peak_equity - current_equity)
        
        # Track equity history (keep in memory for quick access)
        # Note: Also saved to database by engine
        self.equity_history.append((timestamp, current_equity))
        # Keep only recent history in memory (last 1000 points)
        if len(self.equity_history) > 1000:
            self.equity_history = self.equity_history[-1000:]
        
        self.last_equity = current_equity
        
        # Update strategy metrics
        self._update_strategy_metrics()
    
    def _finalize_daily_metrics(self, date: date):
        """Finalize daily metrics for a date"""
        if date in self.daily_metrics:
            daily = self.daily_metrics[date]
            daily.closing_equity = self.portfolio.total_equity()
    
    def record_trade(self, fill: Fill, strategy_id: str, pnl: float):
        """Record a trade for metrics"""
        current_date = datetime.now().date()
        
        # Update daily metrics
        if current_date not in self.daily_metrics:
            self.update()
        
        daily = self.daily_metrics[current_date]
        daily.trades_count += 1
        if pnl > 0:
            daily.winning_trades += 1
        elif pnl < 0:
            daily.losing_trades += 1
        
        # Update strategy metrics
        if strategy_id not in self.strategy_metrics:
            # Get symbol from portfolio positions or fills
            symbol = fill.symbol
            self.strategy_metrics[strategy_id] = StrategyMetrics(
                strategy_id=strategy_id,
                symbol=symbol
            )
        
        strategy = self.strategy_metrics[strategy_id]
        strategy.total_trades += 1
        strategy.realized_pnl += pnl
        strategy.commissions += fill.commission
        
        if pnl > 0:
            strategy.winning_trades += 1
        elif pnl < 0:
            strategy.losing_trades += 1
    
    def _update_strategy_metrics(self):
        """Update strategy-wise metrics"""
        # Get strategy P&L from positions
        for position in self.portfolio.get_open_positions():
            if position.strategy_id and position.strategy_id in self.strategy_metrics:
                strategy = self.strategy_metrics[position.strategy_id]
                strategy.unrealized_pnl = position.unrealized_pnl()
                strategy.current_position_value = position.current_price * position.quantity
                strategy.total_pnl = strategy.realized_pnl + strategy.unrealized_pnl
    
    def get_daily_metrics(self, start_date: Optional[date] = None, end_date: Optional[date] = None) -> List[Dict]:
        """Get daily metrics for date range"""
        metrics_list = []
        for date_key, metrics in sorted(self.daily_metrics.items()):
            if start_date and date_key < start_date:
                continue
            if end_date and date_key > end_date:
                continue
            metrics_list.append(metrics.to_dict())
        return metrics_list
    
    def get_strategy_metrics(self) -> List[Dict]:
        """Get all strategy metrics"""
        # Update before returning
        self._update_strategy_metrics()
        return [metrics.to_dict() for metrics in self.strategy_metrics.values()]
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame"""
        if not self.equity_history:
            return pd.DataFrame(columns=['timestamp', 'equity'])
        
        df = pd.DataFrame(self.equity_history, columns=['timestamp', 'equity'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def get_summary(self) -> Dict:
        """Get overall portfolio summary"""
        if not self.daily_metrics:
            return {}
        
        total_days = len(self.daily_metrics)
        total_trades = sum(d.trades_count for d in self.daily_metrics.values())
        total_winning = sum(d.winning_trades for d in self.daily_metrics.values())
        total_losing = sum(d.losing_trades for d in self.daily_metrics.values())
        
        latest_daily = max(self.daily_metrics.values(), key=lambda x: x.date)
        
        return {
            'total_days': total_days,
            'total_trades': total_trades,
            'winning_trades': total_winning,
            'losing_trades': total_losing,
            'win_rate': total_winning / total_trades if total_trades > 0 else 0.0,
            'current_equity': self.portfolio.total_equity(),
            'realized_pnl': self.portfolio.total_realized_pnl,
            'unrealized_pnl': self.portfolio.total_unrealized_pnl(),
            'total_pnl': self.portfolio.total_pnl(),
            'total_commission': self.portfolio.total_commission,
            'max_drawdown': latest_daily.max_drawdown,
            'return_pct': ((self.portfolio.total_equity() - self.portfolio.initial_capital) / self.portfolio.initial_capital) * 100
        }

    # ---- Periodic snapshot logging support ----
    def start_periodic_snapshot(self, trading_logger, interval_seconds: int = 5):
        """Start a background thread that writes portfolio snapshots every interval_seconds.

        The trading_logger should be an instance of TradingLogger (or provide a compatible
        log_portfolio_update(dict) method).
        """
        if hasattr(self, '_snapshot_thread') and getattr(self, '_snapshot_thread', None) is not None:
            # Already running
            return

        self._snapshot_interval = interval_seconds
        self._snapshot_stop_event = threading.Event()
        self._snapshot_logger = trading_logger
        self._snapshot_thread = threading.Thread(target=self._snapshot_loop, daemon=True)
        self._snapshot_thread.start()

    def _snapshot_loop(self):
        """Internal loop which emits snapshots periodically."""
        try:
            while not getattr(self, '_snapshot_stop_event').is_set():
                now = datetime.now()
                try:
                    # Refresh metrics
                    self.update(now)

                    total_trades = sum(d.trades_count for d in self.daily_metrics.values())
                    total_winning = sum(d.winning_trades for d in self.daily_metrics.values())
                    total_losing = sum(d.losing_trades for d in self.daily_metrics.values())
                    latest_daily = max(self.daily_metrics.values(), key=lambda x: x.date) if self.daily_metrics else None

                    snapshot = {
                        'timestamp': now.isoformat(),
                        'summary': {
                            'total_trades': total_trades,
                            'winning_trades': total_winning,
                            'losing_trades': total_losing,
                            'win_rate': (total_winning / total_trades) if total_trades > 0 else 0.0,
                            'current_equity': self.portfolio.total_equity(),
                            'realized_pnl': self.portfolio.total_realized_pnl,
                            'unrealized_pnl': self.portfolio.total_unrealized_pnl(),
                            'total_pnl': self.portfolio.total_pnl(),
                            'total_commission': self.portfolio.total_commission,
                            'max_drawdown': latest_daily.max_drawdown if latest_daily else 0.0,
                            'return_pct': ((self.portfolio.total_equity() - self.portfolio.initial_capital) / self.portfolio.initial_capital) * 100 if self.portfolio.initial_capital else 0.0
                        },
                        'open_positions_count': len(self.portfolio.get_open_positions()),
                        'closed_positions_count': len([p for p in self.portfolio.positions.values() if p.is_flat()]),
                        'positions': [p.to_dict() for p in self.portfolio.get_all_positions()],
                        'strategy_metrics': [s.to_dict() for s in self.strategy_metrics.values()]
                    }

                    # Emit structured portfolio snapshot
                    try:
                        if hasattr(self, '_snapshot_logger') and self._snapshot_logger is not None:
                            # Use same structured log type as portfolio_update
                            self._snapshot_logger.log_portfolio_update(snapshot)
                        else:
                            # Fallback to printing
                            print(f"Portfolio snapshot: {snapshot}")
                    except Exception:
                        # Avoid letting logging errors kill the snapshot loop
                        pass

                except Exception:
                    # Swallow exceptions from building snapshot (do not stop the loop)
                    pass

                # Sleep for the configured interval (allow early wake via event)
                stop_event = getattr(self, '_snapshot_stop_event')
                if stop_event.wait(timeout=self._snapshot_interval):
                    break
        finally:
            # Clean-up
            try:
                self._snapshot_thread = None
                self._snapshot_stop_event = None
                self._snapshot_logger = None
            except Exception:
                pass

    def stop_periodic_snapshot(self):
        """Stop the periodic snapshot thread if running."""
        if not hasattr(self, '_snapshot_stop_event') or getattr(self, '_snapshot_stop_event') is None:
            return
        try:
            self._snapshot_stop_event.set()
            if hasattr(self, '_snapshot_thread') and self._snapshot_thread is not None:
                self._snapshot_thread.join(timeout=5)
        except Exception:
            pass
        finally:
            self._snapshot_thread = None
            self._snapshot_stop_event = None
            self._snapshot_logger = None

