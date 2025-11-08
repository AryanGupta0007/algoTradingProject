"""
Streamlit dashboard for paper trading system.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date, timedelta
import json
import sys
import os
import time
from collections import deque, defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from database.db_manager import DatabaseManager

# Page config
st.set_page_config(
    page_title="Paper Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state and DB
if 'config' not in st.session_state:
    st.session_state.config = Config()
if 'db' not in st.session_state:
    # Use configured DB path
    db_path = st.session_state.config.trading.db_path
    st.session_state.db = DatabaseManager(db_path=db_path)


def get_db():
    """Return the DatabaseManager instance from session state"""
    return st.session_state.db


def format_currency(value):
    """Format value as currency"""
    return f"₹{value:,.2f}"


def format_percentage(value):
    """Format value as percentage"""
    return f"{value:.2f}%"


def get_closed_positions(db: DatabaseManager, limit: int = 10000):
    """Reconstruct closed round-trip positions from fills.

    This walks fills in chronological order, groups by (symbol, strategy_id) and
    identifies sequences where the running position returns to zero. Each such
    sequence is a closed position with opened_at (first fill) and closed_at (last fill).
    """
    conn = db.conn
    cur = conn.cursor()
    # Join fills with orders to get strategy_id (if available)
    cur.execute("""
        SELECT f.fill_id, f.order_id, f.symbol, f.side, f.quantity, f.price, f.commission, f.timestamp,
               o.strategy_id
        FROM fills f
        LEFT JOIN orders o ON f.order_id = o.order_id
        ORDER BY f.timestamp ASC
        LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in cur.fetchall()]

    # Group by (symbol, strategy_id)
    groups = defaultdict(list)
    for r in rows:
        key = (r.get('symbol'), r.get('strategy_id') or '')
        groups[key].append(r)

    closed_positions = []

    for (symbol, strategy_id), fills in groups.items():
        # FIFO lots for current open run: list of (qty_signed, price, timestamp)
        lots = deque()
        run_opened_at = None
        run_realized = 0.0
        run_entry_qty = 0
        run_entry_cost = 0.0
        run_exit_qty = 0
        run_exit_cost = 0.0

        for f in fills:
            ts = f.get('timestamp')
            # parse timestamp if string
            try:
                f_ts = datetime.fromisoformat(ts) if isinstance(ts, str) else ts
            except Exception:
                f_ts = ts

            side = f.get('side')
            qty = int(f.get('quantity', 0))
            signed_qty = qty if side and side.upper() == 'BUY' else -qty
            price = float(f.get('price', 0.0))

            # If starting a new run (position was flat), mark opened_at
            if not lots and signed_qty != 0:
                run_opened_at = f_ts
                run_realized = 0.0
                run_entry_qty = 0
                run_entry_cost = 0.0
                run_exit_qty = 0
                run_exit_cost = 0.0

            # If same sign as existing lots or empty, just append
            if not lots or (lots and (lots[0][0] > 0) == (signed_qty > 0)):
                lots.append((signed_qty, price, f_ts))
                # update entry/exit aggregates
                if signed_qty > 0:
                    run_entry_qty += signed_qty
                    run_entry_cost += signed_qty * price
                else:
                    run_exit_qty += abs(signed_qty)
                    run_exit_cost += abs(signed_qty) * price
            else:
                # Opposite sign: consume FIFO lots
                remaining = abs(signed_qty)
                # If signed_qty is negative, we're selling; else buying to cover
                while remaining > 0 and lots:
                    lot_qty, lot_price, lot_ts = lots[0]
                    lot_abs = abs(lot_qty)
                    if lot_abs > remaining:
                        # Partial consume
                        # Compute realized pnl for matched amount
                        matched = remaining
                        # If closing a long (lot_qty>0 and signed_qty<0): realized = (sell_price - buy_price)*matched
                        if lot_qty > 0 and signed_qty < 0:
                            run_realized += matched * (price - lot_price)
                        elif lot_qty < 0 and signed_qty > 0:
                            run_realized += matched * (lot_price - price)
                        # Reduce lot
                        new_qty = lot_qty - (matched if lot_qty > 0 else -matched)
                        lots[0] = (new_qty, lot_price, lot_ts)
                        remaining = 0
                        # update exit aggregates
                        if signed_qty < 0:
                            run_exit_qty += matched
                            run_exit_cost += matched * price
                        else:
                            run_entry_qty += matched
                            run_entry_cost += matched * price
                    else:
                        # Consume whole lot
                        lots.popleft()
                        matched = lot_abs
                        if lot_qty > 0 and signed_qty < 0:
                            run_realized += matched * (price - lot_price)
                        elif lot_qty < 0 and signed_qty > 0:
                            run_realized += matched * (lot_price - price)
                        remaining -= matched
                        if signed_qty < 0:
                            run_exit_qty += matched
                            run_exit_cost += matched * price
                        else:
                            run_entry_qty += matched
                            run_entry_cost += matched * price

                # If there's residual in the opposite direction (e.g., oversized close that opens reverse), append remaining
                if remaining > 0:
                    # remaining will carry sign of signed_qty
                    remaining_signed = remaining if signed_qty > 0 else -remaining
                    lots.append((remaining_signed, price, f_ts))
                    if remaining_signed > 0:
                        run_entry_qty += remaining_signed
                        run_entry_cost += remaining_signed * price
                    else:
                        run_exit_qty += abs(remaining_signed)
                        run_exit_cost += abs(remaining_signed) * price

            # If after processing, lots is empty -> a closed run finished
            if not lots and run_opened_at:
                opened_at = run_opened_at
                closed_at = f_ts
                entry_avg = (run_entry_cost / run_entry_qty) if run_entry_qty else None
                exit_avg = (run_exit_cost / run_exit_qty) if run_exit_qty else None
                closed_positions.append({
                    'symbol': symbol,
                    'strategy_id': strategy_id,
                    'opened_at': opened_at.isoformat() if opened_at else None,
                    'closed_at': closed_at.isoformat() if closed_at else None,
                    'quantity': run_entry_qty if run_entry_qty else run_exit_qty,
                    'entry_avg_price': entry_avg,
                    'exit_avg_price': exit_avg,
                    'realized_pnl': run_realized
                })
                run_opened_at = None

    # Return most recent closed positions (by closed_at desc)
    closed_positions.sort(key=lambda x: x.get('closed_at') or '', reverse=True)
    return closed_positions


def main():
    st.title("📈 Paper Trading Dashboard")
    
    # Sidebar (read-only dashboard)
    with st.sidebar:
        st.header("Dashboard (read-only)")
        st.markdown("This Streamlit app displays metrics from the database only.\nRun the trading engine from the command line (python main.py).")
        st.divider()
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto Refresh", value=True)
        if auto_refresh:
            # Default to 10 seconds as requested
            refresh_interval = st.slider("Refresh Interval (seconds)", 1, 60, 10)
        # Database info
        db = get_db()
        st.markdown(f"**DB:** {db.db_path}")

        # Quick DB diagnostics - counts per table and sample rows (useful when dashboard appears empty)
        if st.button("Show DB diagnostics"):
            try:
                conn = db.conn
                cur = conn.cursor()
                cur.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name NOT LIKE 'sqlite_%'
                    ORDER BY name
                """)
                tables = [r[0] for r in cur.fetchall()]
                diag = {}
                for t in tables:
                    cur.execute(f"SELECT COUNT(*) as c FROM {t}")
                    count = cur.fetchone()[0]
                    cur.execute(f"SELECT * FROM {t} ORDER BY rowid DESC LIMIT 5")
                    rows = [dict(row) for row in cur.fetchall()]
                    diag[t] = {"count": count, "sample": rows}

                st.subheader("DB diagnostics")
                for t, info in diag.items():
                    st.markdown(f"**{t}** — rows: {info['count']}")
                    if info['sample']:
                        st.dataframe(pd.DataFrame(info['sample']))
            except Exception as e:
                st.error(f"DB diagnostics failed: {e}")
    
    # Main content
    # Auto-refresh (non-blocking)
    # Use session_state to schedule reruns instead of sleeping which blocks Streamlit
    if '_last_refresh' not in st.session_state:
        st.session_state['_last_refresh'] = time.time()

    if auto_refresh:
        now = time.time()
        # Only trigger rerun when the interval has passed
        if now - st.session_state.get('_last_refresh', 0) >= refresh_interval:
            st.session_state['_last_refresh'] = now
            # Use experimental_rerun to refresh without blocking UI
            try:
                st.experimental_rerun()
            except Exception:
                # In some environments (tests) experimental_rerun may be unavailable; ignore
                pass

    db = get_db()

    # Get data from DB using only orders and fills
    try:
        # Calculate portfolio metrics from fills
        cursor = db.conn.cursor()
        
        # Get total realized P&L and commission
        cursor.execute("""
            SELECT 
                SUM(
                    CASE 
                        WHEN o.side = 'BUY' THEN -f.quantity * f.price
                        ELSE f.quantity * f.price
                    END
                ) as total_realized_pnl,
                SUM(f.commission) as total_commission
            FROM fills f
            JOIN orders o ON f.order_id = o.order_id
        """)
        pnl_row = dict(cursor.fetchone())
        
        # Get current positions from positions table
        cursor.execute("""
            SELECT 
                symbol,
                strategy_id,
                quantity as current_quantity,
                average_price,
                current_price,
                stop_loss,
                take_profit,
                opened_at,
                last_updated
            FROM positions
            WHERE quantity != 0
        """)
        positions = [dict(row) for row in cursor.fetchall()]
        
        # Calculate exposure from current positions
        exposure = sum(abs(pos['current_quantity'] * pos['current_price']) for pos in positions)
        
        # Use config for initial capital
        initial_capital = st.session_state.config.trading.initial_capital
        
        portfolio_summary = {
            'total_equity': initial_capital + (pnl_row.get('total_realized_pnl') or 0.0),
            'initial_capital': initial_capital,
            'total_pnl': pnl_row.get('total_realized_pnl') or 0.0,
            'total_realized_pnl': pnl_row.get('total_realized_pnl') or 0.0,
            'total_unrealized_pnl': 0.0,  # Will be calculated from current positions
            'cash': initial_capital + (pnl_row.get('total_realized_pnl') or 0.0) - exposure,
            'exposure': exposure,
            'total_commission': pnl_row.get('total_commission') or 0.0,
            'positions': positions
        }

        # Calculate metrics from orders and fills
        cursor = db.conn.cursor()
        
        # Daily metrics
        cursor.execute("""
            SELECT 
                DATE(f.timestamp) as date,
                SUM(
                    CASE 
                        WHEN o.side = 'BUY' THEN -f.quantity * f.price
                        ELSE f.quantity * f.price
                    END
                ) as daily_pnl,
                SUM(f.commission) as commissions,
                COUNT(DISTINCT o.order_id) as trades_count
            FROM fills f
            JOIN orders o ON f.order_id = o.order_id
            GROUP BY DATE(f.timestamp)
            ORDER BY date DESC
        """)
        daily_metrics = [dict(row) for row in cursor.fetchall()]
        
        # Strategy metrics
        cursor.execute("""
            SELECT 
                o.strategy_id,
                o.symbol,
                COUNT(DISTINCT o.order_id) as total_trades,
                SUM(
                    CASE 
                        WHEN o.side = 'BUY' THEN -f.quantity * f.price
                        ELSE f.quantity * f.price
                    END
                ) as total_pnl,
                SUM(f.commission) as commissions
            FROM fills f
            JOIN orders o ON f.order_id = o.order_id
            GROUP BY o.strategy_id, o.symbol
        """)
        strategy_metrics = [dict(row) for row in cursor.fetchall()]
        
        # Equity curve points
        cursor.execute("""
            SELECT 
                f.timestamp,
                ? + SUM(
                    CASE 
                        WHEN o.side = 'BUY' THEN -f.quantity * f.price
                        ELSE f.quantity * f.price
                    END
                ) OVER (ORDER BY f.timestamp) - 
                SUM(f.commission) OVER (ORDER BY f.timestamp) as equity
            FROM fills f
            JOIN orders o ON f.order_id = o.order_id
            ORDER BY f.timestamp DESC
            LIMIT 1000
        """, (initial_capital,))
        equity_curve = [dict(row) for row in cursor.fetchall()]
        
        # Calculate winning/losing trades
        cursor.execute("""
            WITH trade_pnl AS (
                SELECT 
                    o.order_id,
                    SUM(
                        CASE 
                            WHEN o.side = 'BUY' THEN -f.quantity * f.price
                            ELSE f.quantity * f.price
                        END
                    ) - SUM(f.commission) as trade_pnl
                FROM fills f
                JOIN orders o ON f.order_id = o.order_id
                GROUP BY o.order_id
            )
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN trade_pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN trade_pnl <= 0 THEN 1 ELSE 0 END) as losing_trades
            FROM trade_pnl
        """)
        trade_stats = dict(cursor.fetchone())
        
        metrics = {
            'daily_metrics': daily_metrics,
            'strategy_metrics': strategy_metrics,
            'equity_curve': equity_curve,
            'summary': {}
        }

        # Build summary metrics
        total_trades = trade_stats.get('total_trades', 0)
        winning_trades = trade_stats.get('winning_trades', 0)
        losing_trades = trade_stats.get('losing_trades', 0)
        total_pnl = portfolio_summary['total_pnl']
        win_rate = (winning_trades / total_trades) if total_trades > 0 else 0.0

        metrics['summary'] = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'total_days': len(daily_metrics)
        }

        # LTP data derived from positions or market_data
        ltp_data = {}
        positions = portfolio_summary.get('positions', [])
        for pos in positions:
            ltp_data[pos['symbol']] = {'ltp': pos.get('current_price'), 'timestamp': pos.get('last_updated')}

        # Strategy status from positions and recent strategy metrics
        strategy_status = {}
        # Use latest metric per strategy if available
        latest_metrics_by_strategy = {}
        for m in strategy_metrics:
            sid = m.get('strategy_id')
            if sid and sid not in latest_metrics_by_strategy:
                latest_metrics_by_strategy[sid] = m

        for sid, m in latest_metrics_by_strategy.items():
            strategy_status[sid] = {
                'symbol': m.get('symbol'),
                'is_active': False,
                'has_position': any(p.get('strategy_id') == sid for p in positions),
                'current_price': None
            }
            # set current_price from positions if present
            for p in positions:
                if p.get('strategy_id') == sid:
                    strategy_status[sid]['current_price'] = p.get('current_price')

    except Exception as e:
        st.error(f"Error getting data from DB: {e}")
        return
    
    # Portfolio Overview
    st.header("Portfolio Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Equity", format_currency(portfolio_summary['total_equity']))
        st.metric("Initial Capital", format_currency(portfolio_summary['initial_capital']))
    
    with col2:
        st.metric("Total P&L", format_currency(portfolio_summary['total_pnl']),
                 delta=format_percentage((portfolio_summary['total_pnl'] / portfolio_summary['initial_capital']) * 100) if portfolio_summary['initial_capital'] != 0 else None)
        st.metric("Realized P&L", format_currency(portfolio_summary['total_realized_pnl']))
    
    with col3:
        st.metric("Unrealized P&L", format_currency(portfolio_summary['total_unrealized_pnl']))
        st.metric("Cash", format_currency(portfolio_summary['cash']))
    
    with col4:
        st.metric("Exposure", format_currency(portfolio_summary['exposure']))
        st.metric("Commission", format_currency(portfolio_summary['total_commission']))
    
    # LTP Section
    st.header("Last Traded Prices (LTP)")
    if ltp_data:
        ltp_df = pd.DataFrame([
            {'Symbol': sym, 'LTP': data['ltp'], 'Timestamp': data['timestamp']}
            for sym, data in ltp_data.items()
        ])
        st.dataframe(ltp_df, use_container_width=True)
    
    # Positions
    st.header("Open Positions")
    if portfolio_summary['positions']:
        positions_df = pd.DataFrame(portfolio_summary['positions'])
        st.dataframe(positions_df, use_container_width=True)
    else:
        st.info("No open positions")

    # Closed Positions (reconstructed from fills)
    st.header("Closed Positions")
    try:
        # Allow user to control how many fills are scanned and paging so
        # large histories don't overwhelm the UI. Defaults are generous but
        # adjustable for debugging.
        with st.expander("Closed positions options", expanded=False):
            closed_limit = st.number_input(
                "Max fills to consider for reconstruction (higher = slower)",
                min_value=100,
                max_value=200000,
                value=10000,
                step=100
            )
            page_size = st.slider("Closed positions per page", 5, 500, 25)
            page = st.number_input("Page number", min_value=1, value=1)
            show_all = st.checkbox("Show all closed positions (ignore paging)", value=False)

        closed_positions = get_closed_positions(db, limit=int(closed_limit))
        total_closed = len(closed_positions)

        if not closed_positions:
            st.info("No closed positions found in fills/history")
        else:
            st.markdown(f"**Total closed runs reconstructed:** {total_closed}")

            # Apply paging unless user asked to show all
            if not show_all:
                start = (int(page) - 1) * int(page_size)
                end = start + int(page_size)
                page_positions = closed_positions[start:end]
                st.markdown(f"Showing rows {start+1} to {min(end, total_closed)}")
            else:
                page_positions = closed_positions

            closed_df = pd.DataFrame(page_positions)
            # Convert timestamps to datetime for nicer display
            if 'opened_at' in closed_df.columns:
                closed_df['opened_at'] = pd.to_datetime(closed_df['opened_at'])
            if 'closed_at' in closed_df.columns:
                closed_df['closed_at'] = pd.to_datetime(closed_df['closed_at'])

            st.dataframe(closed_df, use_container_width=True)
    except Exception as e:
        st.error(f"Error reconstructing closed positions: {e}")
    
    # Strategy Performance
    st.header("Strategy Performance")
    if metrics['strategy_metrics']:
        # Add win rate calculation for each strategy
        cursor = db.conn.cursor()
        strategy_win_rates = {}
        
        for strategy in metrics['strategy_metrics']:
            cursor.execute("""
                WITH trade_pnl AS (
                    SELECT 
                        o.order_id,
                        SUM(
                            CASE 
                                WHEN o.side = 'BUY' THEN -f.quantity * f.price
                                ELSE f.quantity * f.price
                            END
                        ) - SUM(f.commission) as trade_pnl
                    FROM fills f
                    JOIN orders o ON f.order_id = o.order_id
                    WHERE o.strategy_id = ?
                    GROUP BY o.order_id
                )
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN trade_pnl > 0 THEN 1 ELSE 0 END) as winning_trades
                FROM trade_pnl
            """, (strategy['strategy_id'],))
            stats = dict(cursor.fetchone())
            strategy_win_rates[strategy['strategy_id']] = (
                stats['winning_trades'] / stats['total_trades']
                if stats['total_trades'] > 0 else 0.0
            )
        
        strategy_df = pd.DataFrame(metrics['strategy_metrics'])
        # Add win rate to dataframe
        strategy_df['win_rate'] = strategy_df['strategy_id'].map(strategy_win_rates)
        
        # Display metrics
        st.dataframe(strategy_df, use_container_width=True)
        
        # Strategy P&L Chart
        if len(strategy_df) > 0:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=strategy_df['strategy_id'],
                y=strategy_df['total_pnl'],
                name='Total P&L',
                marker_color=['green' if x > 0 else 'red' for x in strategy_df['total_pnl']]
            ))
            fig.update_layout(
                title="Strategy P&L",
                xaxis_title="Strategy",
                yaxis_title="P&L (₹)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No strategy metrics available")
    
    # Daily Performance
    st.header("Daily Performance")
    if metrics['daily_metrics']:
        daily_df = pd.DataFrame(metrics['daily_metrics'])
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        
        # Display daily metrics
        st.dataframe(daily_df, use_container_width=True)
        
        # Equity Curve
        if metrics['equity_curve']:
            equity_df = pd.DataFrame(metrics['equity_curve'])
            equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity_df['timestamp'],
                y=equity_df['equity'],
                mode='lines',
                name='Equity',
                line=dict(color='blue', width=2)
            ))
            fig.update_layout(
                title="Equity Curve",
                xaxis_title="Time",
                yaxis_title="Equity (₹)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Daily P&L Chart
        if len(daily_df) > 0:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=daily_df['date'],
                y=daily_df['daily_pnl'],  # Changed from total_pnl to daily_pnl to match our SQL query
                name='Daily P&L',
                marker_color=['green' if x > 0 else 'red' for x in daily_df['daily_pnl']]
            ))
            fig.update_layout(
                title="Daily P&L",
                xaxis_title="Date",
                yaxis_title="P&L (₹)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No daily metrics available")
    
    # Strategy Status
    st.header("Strategy Status")
    if strategy_status:
        status_df = pd.DataFrame([
            {
                'Strategy ID': sid,
                'Symbol': data['symbol'],
                'Active': data['is_active'],
                'Has Position': data['has_position'],
                'Current Price': data['current_price']
            }
            for sid, data in strategy_status.items()
        ])
        st.dataframe(status_df, use_container_width=True)

    # Open Orders by Strategy
    st.header("Open Orders by Strategy")
    try:
        # Fetch recent orders (joined with strategy_id in DB)
        all_orders = db.get_orders(limit=1000)
        # Filter open orders (status not filled/cancelled/rejected)
        open_orders = [o for o in all_orders if o.get('status') not in ('filled', 'cancelled', 'rejected')]

        if open_orders:
            # Group by strategy_id
            orders_by_strategy = defaultdict(list)
            for o in open_orders:
                sid = o.get('strategy_id') or 'unknown'
                orders_by_strategy[sid].append(o)

            for sid, orders_list in orders_by_strategy.items():
                st.subheader(f"Strategy: {sid} — {len(orders_list)} open order(s)")
                df = pd.DataFrame(orders_list)
                # Convert timestamps if present
                if 'timestamp' in df.columns:
                    try:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                    except Exception:
                        pass
                st.dataframe(df, use_container_width=True)
        else:
            st.info("No open orders found in DB")
    except Exception as e:
        st.error(f"Error fetching open orders: {e}")
    
    # Summary Metrics
    st.header("Summary Metrics")
    if metrics['summary']:
        summary = metrics['summary']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", summary.get('total_trades', 0))
            st.metric("Win Rate", format_percentage(summary.get('win_rate', 0) * 100))
        
        with col2:
            st.metric("Winning Trades", summary.get('winning_trades', 0))
            st.metric("Losing Trades", summary.get('losing_trades', 0))
        
        with col3:
            st.metric("Return %", format_percentage(summary.get('return_pct', 0)))
            st.metric("Max Drawdown", format_currency(summary.get('max_drawdown', 0)))
        
        with col4:
            st.metric("Total Days", summary.get('total_days', 0))
            st.metric("Avg Trade P&L", format_currency(
                summary.get('total_pnl', 0) / summary.get('total_trades', 1) if summary.get('total_trades', 0) > 0 else 0
            ))


if __name__ == "__main__":
    main()

