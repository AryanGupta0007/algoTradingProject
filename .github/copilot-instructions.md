## Purpose
This file gives concise, repo-specific guidance to an AI coding assistant so it can be immediately productive in the Paper_Trader codebase.

Keep answers actionable and reference concrete files and symbols below. Avoid generic advice unless it directly maps to the code here.

## Big picture (one-line)
Data flows: DataFeed -> TradingEngine -> Strategy -> OrderManager -> Portfolio -> Database/Logs/Telemetry.

## Key modules & where to look
- `engine.py`: central orchestrator. Important methods: `initialize_data_feed`, `add_strategy`, `start`, `_on_market_data`, `_on_strategy_order`, `_on_fill`.
- `datafeed/` (see `datafeed/base.py`, `fake_feed.py`, `icici_breeze.py`): implement `DataFeed` subclasses. Use `add_subscriber(callback)` and `subscribe(symbols, callback)` to receive `OHLCVData` or `TickData`.
- `strategy/` (see `strategy/base.py`, `example_strategies.py`): strategies inherit `BaseStrategy`. Key lifecycle: `_initialize_indicators`, `on_data`, `start`, `stop`. Strategies call `set_order_callback()` to submit orders.
- `order/` (see `order/order_manager.py`, `order/order.py`): order lifecycle is handled by `OrderManager`. Use `submit_order`, `execute_order`, and `add_fill_callback`.
- `portfolio/` (see `portfolio/portfolio.py`, `portfolio/position.py`, `portfolio/metrics.py`): portfolio state, P&L, and position handling. `process_fill(fill, order)` mutates cash/positions.
- `trading_log/trading_logger.py`: structured JSONL logs (file: `*_structured.jsonl`) and human-readable `trading.log`. Use `trading_logger.log_*` helpers for consistent structured events.
- `config.py`: default feature flags (DB, RMS), timing (`fake_data_interval`), commission/slippage. Use env vars for ICICI API credentials.
- `main.py` and `README.md`: canonical usage examples (e.g., `python main.py`, `streamlit run dashboard.py`).

## Concrete conventions and patterns to follow
- Strategy indicators: stored in `strategy.indicators` and expose either `.current_value` or `.values`. Update methods are either `update_ohlcv(open, high, low, close, volume)` or `update(close)` — check indicator implementation in `indicators/`.
- Entry/exit rules: strategies declare `EntryCondition` and `ExitCondition` dataclasses (see `strategy/base.py`). Operators include `>, <, >=, <=, ==, crossover, crossunder`.
- Order submission: strategies create `Order` objects and call the engine-provided order callback. Orders may set `stop_loss`/`take_profit`; `Portfolio.process_fill` will pick these up when applying fills.
- Resampling: `TradingEngine` creates `DataResampler` per `strategy_id` when `add_strategy(..., resample_timeframe=...)` is used. Resampler keys live in `engine.strategy_resamplers`.
- RMS: Risk checks happen in `_on_strategy_order` using `rms.RiskManagementSystem` when enabled via `config.trading.enable_rms`.
- DB & persistence: Database manager (`database/`) is optional. Enabled by `config.trading.enable_db`; default DB path `trading.db`. Engine uses DB for orders, fills, positions, metrics, and equity points.

## Developer workflows (commands & quick checks)
- Install deps: `pip install -r requirements.txt` (TA-Lib has platform-specific install steps; see README).
- Run example/trading: `python main.py` — loads `Config()` and uses the fake feed unless ICICI is enabled.
- Run dashboard: `streamlit run dashboard.py` or `python run_dashboard.py`.
- Debugging: set `Config.trading.log_level='DEBUG'` or export `ICICI_ENABLED` and other vars for live feeds. Structured logs are appended to `<log_file>_structured.jsonl`.

## Useful code snippets (for the agent to reference)
- Add a strategy (canonical):
  engine.initialize_data_feed(["RELIANCE","TCS"])
  engine.add_strategy(my_strategy, resample_timeframe='5min')

- Strategy order callback pattern:
  strategy.set_order_callback(engine._on_strategy_order)

- Inspect current price path used by engine: `engine.ltp[symbol]` and `engine.get_ltp(symbol)`.

## Edge-cases and gotchas (observed in code)
- Engine requires `data_feed` before `start()`; `initialize_data_feed()` must be called first or `start()` will raise.
- RMS, DB, and market-data-saving are feature flags in `config.trading`; tests or runs that assume DB/logging must enable them.
- Indicator updates can be either incremental (`update`) or OHLCV-based (`update_ohlcv`) — the agent should inspect the indicator class before calling.
- Order execution loop sleeps 0.5s and uses `order_manager.execute_order` logic — market vs limit/stop semantics are implemented in code; mimic them in tests.
- Fake feed timing: `config.trading.fake_data_interval` defaults to 3 seconds (used by `FakeDataFeed`).

## Files to reference when making edits
- `engine.py` (orchestrator)
- `strategy/base.py` (strategy lifecycle & condition evaluation)
- `datafeed/base.py` and `datafeed/fake_feed.py`
- `order/order_manager.py` and `order/order.py`
- `portfolio/portfolio.py` and `portfolio/position.py`
- `trading_log/trading_logger.py`
- `config.py` and `README.md` (usage examples and feature flags)

## Style for changes the agent should follow
- Keep public APIs stable (do not rename `TradingEngine.add_strategy`, `BaseStrategy.on_data`, or `OrderManager.submit_order` without updating all callers).
- When adding logs, use `TradingLogger` structured helpers for machine-readable events (e.g., `log_order`, `log_fill`, `log_condition_evaluation`).
- Small, focused tests: create short run scripts using `FakeDataFeed` and `Config(trading=TradingConfig(enable_db=False))` to validate changes locally.

## When to ask the user
- If a change requires enabling ICICI or adding secrets (API keys), request explicit confirmation — do not add secrets to the repo.
- If a design change affects persistence schema (database tables), propose a migration and ask before applying.

---
If anything above is unclear or you'd like more examples (unit tests, small harness, or changes merged into `main.py`), tell me which area to expand first.
