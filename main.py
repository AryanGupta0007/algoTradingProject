"""
Main entry point for the paper trading system.
"""
import time
from config import Config
from engine import TradingEngine
from strategy import (
    MovingAverageCrossoverStrategy,
    RSIStrategy,
    ADXSignalStrategy,
    EMACrossoverStrategy,
    ADXDMISupertrendSignalStrategy,
    OpenRangeBreakoutSignalStrategy,
)
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Ensure the engine/logger named 'TradingSystem' prints DEBUG to console so
# strategy-level DEBUG messages (e.g. MA_CROSS condition evaluations) are
# visible during interactive runs without forcing the whole app to DEBUG.
trade_logger = logging.getLogger('TradingSystem')
trade_logger.setLevel(logging.DEBUG)
has_stream = any(isinstance(h, logging.StreamHandler) for h in trade_logger.handlers)
if not has_stream:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    trade_logger.addHandler(ch)


def main():
    """Main function to run the paper trading system"""
    # Load configuration
    config = Config()
    
    # Initialize trading engine
    engine = TradingEngine(config)
    
    # Define symbols to trade
    symbols = ["RELIANCE", "ADANIENT", "ADANIPORTS", "ICICIBANK", "ASIANPAINT"]
    
    # Initialize data feed
    engine.initialize_data_feed(symbols)
    
    # Create and add strategies for each symbol
    for symbol in symbols:
        # Add Moving Average Crossover strategy
        ma_strategy = MovingAverageCrossoverStrategy(
            strategy_id=f"MA_CROSS_{symbol}",
            symbol=symbol,
            fast_period=10,
            slow_period=30,
            quantity=10,
            stop_loss_pct=0.02,
            take_profit_pct=0.04
        )
        engine.add_strategy(ma_strategy, resample_timeframe='5min')
        logger.info(f"Added MA Crossover strategy for {symbol}")
        
        # Add RSI strategy 
        rsi_strategy = RSIStrategy(
            strategy_id=f"RSI_{symbol}",
            symbol=symbol,
            rsi_period=14,
            oversold_threshold=30.0,
            overbought_threshold=70.0,
            quantity=5,
            stop_loss_pct=0.02,
            take_profit_pct=0.03
        )
        engine.add_strategy(rsi_strategy)
        logger.info(f"Added RSI strategy for {symbol}")

        # Add ADX momentum strategy (LONG/SHORT with ATR-based SL/TP)
        adx_strategy = ADXSignalStrategy(
            strategy_id=f"ADX_SIG_{symbol}",
            symbol=symbol,
            adx_period=14,
            atr_mult_tp=4.0,
            atr_mult_sl=2.0,
            quantity=5,
        )
        engine.add_strategy(adx_strategy)
        logger.info(f"Added ADX Signal strategy for {symbol}")

        # Add EMA crossover strategy (LONG/SHORT)
        ema_strategy = EMACrossoverStrategy(
            strategy_id=f"EMA_X_{symbol}",
            symbol=symbol,
            short=5,
            long=20,
            quantity=5,
        )
        engine.add_strategy(ema_strategy)
        logger.info(f"Added EMA Crossover strategy for {symbol}")

        # Add ADX/DMI + Supertrend hybrid strategy
        adx_st_strategy = ADXDMISupertrendSignalStrategy(
            strategy_id=f"ADX_ST_{symbol}",
            symbol=symbol,
            adx_period=7,
            supertrend_period=7,
            supertrend_multiplier=2.0,
            atr_period=7,
            atr_lookback=30,
            risk_reward=2.0,
            quantity=5,
        )
        engine.add_strategy(adx_st_strategy)
        logger.info(f"Added ADX+DMI Supertrend strategy for {symbol}")

        # Add Open Range Breakout strategy
        orb_strategy = OpenRangeBreakoutSignalStrategy(
            strategy_id=f"ORB_{symbol}",
            symbol=symbol,
            tp_mult=4.0,
            sl_lookback=6,
            quantity=5,
        )
        engine.add_strategy(orb_strategy)
        logger.info(f"Added Open Range Breakout strategy for {symbol}")
    
    try:
        # Start trading engine
        logger.info("Starting paper trading system...")
        engine.start()
        
        # Run for a period (or indefinitely)
        logger.info("Trading system is running. Press Ctrl+C to stop.")
        
        # Print portfolio updates periodically
        while True:
            time.sleep(30)  # Update every 30 seconds
            
            # Print portfolio summary
            portfolio = engine.get_portfolio_summary()
            logger.info(f"\n{'='*60}")
            logger.info(f"Portfolio Summary:")
            logger.info(f"  Total Equity: ₹{portfolio['total_equity']:,.2f}")
            logger.info(f"  Total P&L: ₹{portfolio['total_pnl']:,.2f}")
            logger.info(f"  Cash: ₹{portfolio['cash']:,.2f}")
            logger.info(f"  Exposure: ₹{portfolio['exposure']:,.2f}")
            logger.info(f"  Open Positions: {len(portfolio['positions'])}")
            
            # Get metrics summary
            metrics = engine.get_metrics()
            summary = metrics.get('summary', {})
            
            # Print position info
            if portfolio['positions']:
                logger.info(f"\n  Open Positions:")
                for pos in portfolio['positions']:
                    logger.info(f"    {pos['symbol']}: {pos['quantity']} @ ₹{pos['average_price']:.2f} "
                              f"(Current: ₹{pos['current_price']:.2f}, "
                              f"P&L: ₹{pos['unrealized_pnl']:.2f} ({pos['unrealized_pnl_pct']:.2f}%))")
            
            # Print trading statistics
            logger.info(f"\n  Trading Statistics:")
            logger.info(f"    Total Trades: {summary.get('total_trades', 0)}")
            logger.info(f"    Winning Trades: {summary.get('winning_trades', 0)}")
            logger.info(f"    Losing Trades: {summary.get('losing_trades', 0)}")
            win_rate = summary.get('win_rate', 0) * 100
            logger.info(f"    Win Rate: {win_rate:.1f}%")
            
            logger.info(f"{'='*60}\n")
    
    except KeyboardInterrupt:
        logger.info("\nStopping trading system...")
    finally:
        engine.stop()
        logger.info("Trading system stopped.")


if __name__ == "__main__":
    main()

