import logging
import json
import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from database.db_manager import DatabaseManager

# Silence noisy loggers
for name in ['datafeed', 'engine', 'TradingSystem', 'order.order_manager']:
    logging.getLogger(name).setLevel(logging.WARNING)

c = Config()
db = DatabaseManager(db_path=c.trading.db_path)

out = {
    'db_path': db.db_path,
    'positions': db.get_positions(),
    'recent_fills': db.get_fills(limit=200),
    'recent_orders': db.get_orders(limit=200)
}

with open('db_debug_dump.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, default=str, indent=2)

print('Wrote db_debug_dump.json')
