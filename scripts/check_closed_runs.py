"""
Script to analyze fills and reconstructed closed positions directly.
"""
import sys
import os
from datetime import datetime
import json
from collections import defaultdict

# Add parent directory to path so we can import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from config import Config
from database.db_manager import DatabaseManager

def get_closed_positions_with_debug(db: DatabaseManager, limit: int = 10000, debug=True):
    """Debug version that shows how many fills were considered and what matched."""
    conn = db.conn
    cur = conn.cursor()
    
    # Get fills joined with orders to get strategy_id
    cur.execute("""
        SELECT f.fill_id, f.order_id, f.symbol, f.side, f.quantity, f.price, f.commission, f.timestamp,
               o.strategy_id
        FROM fills f
        LEFT JOIN orders o ON f.order_id = o.order_id
        ORDER BY f.timestamp ASC
        LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    
    if debug:
        print(f"\nFound {len(rows)} fills to analyze")
        fills_by_symbol = defaultdict(int)
        for r in rows:
            fills_by_symbol[r['symbol']] += 1
        print("\nFills per symbol:")
        for sym, count in fills_by_symbol.items():
            print(f"  {sym}: {count} fills")

    # Group by (symbol, strategy_id)
    groups = defaultdict(list)
    for r in rows:
        key = (r.get('symbol'), r.get('strategy_id') or '')
        groups[key].append(r)

    if debug:
        print(f"\nFound {len(groups)} symbol+strategy combinations:")
        for (sym, strat), fills in groups.items():
            print(f"  {sym} + {strat or 'no-strat'}: {len(fills)} fills")

    closed_positions = []

    for (symbol, strategy_id), fills in groups.items():
        # Current position state
        lots = []
        run_opened_at = None
        run_realized = 0.0
        run_entry_qty = 0
        run_entry_cost = 0.0
        run_exit_qty = 0
        run_exit_cost = 0.0

        for f in fills:
            ts = f.get('timestamp')
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

            if debug:
                print(f"\nProcessing fill {f.get('fill_id')}: {side} {qty} @ {price}")
                print(f"Current lots: {lots}")

            # If same sign as existing lots or empty, just append
            if not lots or (lots and (lots[0][0] > 0) == (signed_qty > 0)):
                lots.append((signed_qty, price, f_ts))
                # Update entry/exit aggregates
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
                        matched = remaining
                        if lot_qty > 0 and signed_qty < 0:
                            run_realized += matched * (price - lot_price)
                        elif lot_qty < 0 and signed_qty > 0:
                            run_realized += matched * (lot_price - price)
                        new_qty = lot_qty - (matched if lot_qty > 0 else -matched)
                        lots[0] = (new_qty, lot_price, lot_ts)
                        remaining = 0
                        if signed_qty < 0:
                            run_exit_qty += matched
                            run_exit_cost += matched * price
                        else:
                            run_entry_qty += matched
                            run_entry_cost += matched * price
                    else:
                        # Consume whole lot
                        lots.pop(0)
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

                # If residual in opposite direction, append
                if remaining > 0:
                    remaining_signed = remaining if signed_qty > 0 else -remaining
                    lots.append((remaining_signed, price, f_ts))
                    if remaining_signed > 0:
                        run_entry_qty += remaining_signed
                        run_entry_cost += remaining_signed * price
                    else:
                        run_exit_qty += abs(remaining_signed)
                        run_exit_cost += abs(remaining_signed) * price

            if debug:
                print(f"After processing: lots={lots}")

            # If after processing, lots is empty -> closed run
            if not lots and run_opened_at:
                opened_at = run_opened_at
                closed_at = f_ts
                entry_avg = (run_entry_cost / run_entry_qty) if run_entry_qty else None
                exit_avg = (run_exit_cost / run_exit_qty) if run_exit_qty else None
                
                closed_run = {
                    'symbol': symbol,
                    'strategy_id': strategy_id,
                    'opened_at': opened_at.isoformat() if opened_at else None,
                    'closed_at': closed_at.isoformat() if closed_at else None,
                    'quantity': run_entry_qty if run_entry_qty else run_exit_qty,
                    'entry_avg_price': entry_avg,
                    'exit_avg_price': exit_avg,
                    'realized_pnl': run_realized
                }
                closed_positions.append(closed_run)
                if debug:
                    print(f"\nFound closed run: {json.dumps(closed_run, indent=2)}")
                run_opened_at = None

    # Sort by closed_at desc
    closed_positions.sort(key=lambda x: x.get('closed_at') or '', reverse=True)
    
    if debug:
        print(f"\nTotal closed runs found: {len(closed_positions)}")
        if closed_positions:
            print("\nFirst 3 closed runs:")
            for run in closed_positions[:3]:
                print(json.dumps(run, indent=2))

    return closed_positions

def analyze_db():
    """Analyze fills and reconstructed positions in the DB"""
    config = Config()
    db = DatabaseManager(db_path=config.trading.db_path)
    
    print(f"\nAnalyzing database: {db.db_path}")
    
    try:
        # Get raw fill count first
        cur = db.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM fills")
        fill_count = cur.fetchone()[0]
        print(f"\nTotal fills in DB: {fill_count}")
        
        # Get open positions
        open_positions = db.get_positions()
        print(f"\nOpen positions: {len(open_positions)}")
        for pos in open_positions:
            print(json.dumps(pos, indent=2))
        
        print("\nReconstructing closed runs (with debug)...")
        closed_runs = get_closed_positions_with_debug(db, limit=50000, debug=True)
        
        # Save to JSON for inspection
        output_path = 'closed_runs_debug.json'
        with open(output_path, 'w') as f:
            json.dump({
                'total_fills': fill_count,
                'open_positions': open_positions,
                'closed_runs': closed_runs
            }, f, indent=2)
        print(f"\nSaved complete debug output to {output_path}")
        
    finally:
        db.close()

if __name__ == '__main__':
    analyze_db()