"""
Small smoke test: start the engine with FakeDataFeed, add an RSI strategy, run for a short time, then stop.
This helps verify structured condition_evaluation logs are written.
"""
import time
from config import Config
from engine import TradingEngine
from strategy.example_strategies import RSIStrategy


def run_smoke():
    cfg = Config()
    cfg.trading.enable_db = False
    cfg.trading.fake_data_interval = 1

    engine = TradingEngine(cfg)
    symbols = ["TCS"]
    engine.initialize_data_feed(symbols)

    rsi = RSIStrategy(
        strategy_id="RSI_TEST",
        symbol="TCS",
        rsi_period=5,
        oversold_threshold=30.0,
        overbought_threshold=70.0,
        quantity=1
    )
    engine.add_strategy(rsi)

    print("Starting engine (smoke test) — will run for 8 seconds...")
    engine.start()
    try:
        time.sleep(8)
    finally:
        engine.stop()
        print("Engine stopped — smoke test complete")


if __name__ == '__main__':
    run_smoke()
