import streamlit as st
import math
import sqlite3
import pandas as pd

# Connect to the SQLite database
conn = sqlite3.connect("trading.db")

# ---- PAGE CONFIG ----
st.set_page_config(page_title="Trading Dashboard", layout="wide")

orders_df = pd.read_sql("SELECT * FROM orders", conn)
positions_df = pd.read_sql("SELECT * FROM positions", conn)
fills_df = pd.read_sql("SELECT * FROM fills", conn)
portfolio_df = pd.read_sql("SELECT * FROM portfolio_state", conn)
daily_df = pd.read_sql("SELECT * FROM daily_metrics", conn)
metrics_df = pd.read_sql("SELECT * FROM strategy_metrics", conn)
trade_logs_df = pd.read_sql("SELECT * FROM trades", conn)


# ---- SAMPLE DATA ----
open_positions_df = positions_df 
portfolio_df_cols = portfolio_df.columns
trade_history_df = trade_logs_df 
portfolio_summary_df = portfolio_df.iloc[-1]
print(portfolio_summary_df)
metrics = ['Cash', 'Holdings value', 'Total value', 'Total Realized PnL', 'Total Unrealized PnL', 'Total PnL', 'Total commission', 'Exposure', 'Initial Capital']
portfolio_metrics = pd.DataFrame({
    'metrics': metrics[:-1],
    'values': portfolio_summary_df.values.tolist()[2:-1]
})
trade_history_df = trade_history_df.drop(columns=["trade_id"])
trade_history_df = trade_history_df.set_index("order_id")
trade_history_df = trade_history_df.sort_values(by="entry_time", ascending=False)
trade_history_df.reset_index(inplace=True)
trade_history_df = trade_history_df.rename(
    columns={
    "order_id": "Order ID",
    "ltp": "LTP",
    "symbol": "Symbol",
    "strategy_id": "Strategy ID",
    "trade_type": "Direction",
    "status": "Status",
    "entry_time": "Entry Time",
    "entry_price": "Entry Price",
    "exit_time": "Exit Time",
    "exit_price": "Exit Price",
    "stop_loss":  "SL",
    "take_profit": "TP",
})
# ---- HEADER ----
trade_history_df = trade_history_df.set_index('Order ID')
st.title("📊 Trading Dashboard")

# ---- SECTION 1: OPEN POSITIONS ----
st.subheader("💼 Open Positions")
positions_df = positions_df.rename(
    columns={
    "average_price": "Average Price",
    "current_price": "LTP",
    "symbol": "Symbol",
    "strategy_id": "Strategy ID",
    "stop_loss":  "SL",
    "take_profit": "TP",
    "last_updated": "Last Updated",
    "quantity": "Quantity",
})
positions_df['Last Updated'] = pd.to_datetime(positions_df['Last Updated'])
positions_df.drop(columns=['position_id', 'opened_at'], inplace=True)
positions_df = positions_df.set_index("Last Updated")

st.dataframe(positions_df, use_container_width=True)

st.divider()

# ---- SECTION 2: TRADE HISTORY WITH PAGINATION ----
st.subheader("🧾 Trade History")

trade_history_df['Entry Time'] = pd.to_datetime(trade_history_df['Entry Time'])
trade_history_df['Exit Time'] = pd.to_datetime(trade_history_df['Exit Time'])
# Pagination controls
page_size = st.selectbox("Rows per page:", [5, 10, 15, 20], index=1)
total_pages = math.ceil(len(trade_history_df) / page_size)
page_number = st.number_input("Page:", min_value=1, max_value=total_pages, value=1, step=1)

# Paginated data
start_idx = (page_number - 1) * page_size
end_idx = start_idx + page_size
paginated_df = trade_history_df.iloc[start_idx:end_idx]

st.dataframe(paginated_df, use_container_width=True)
st.caption(f"Page {page_number} of {total_pages}")

st.divider()

# ---- SECTION 3: PORTFOLIO SUMMARY ----
st.subheader("📈 Portfolio Summary")
portfolio_metrics = portfolio_metrics.set_index('metrics') 
st.dataframe(portfolio_metrics, use_container_width=True)

# ---- FOOTER ----
st.markdown("---")
st.caption("Built with ❤️ using Streamlit")

