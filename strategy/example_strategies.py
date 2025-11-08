import logging
from typing import List, Optional
import pandas as pd
import numpy as np
import talib
from .base import BaseStrategy, EntryCondition, ExitCondition
from indicators.sma import SMA
from indicators.rsi import RSI

logger = logging.getLogger(__name__)

class _OHLCVBufferMixin:
    def _init_buffer(self, maxlen: int = 1000):
        from collections import deque
        self._buf_maxlen = maxlen
        self._o = deque(maxlen=maxlen)
        self._h = deque(maxlen=maxlen)
        self._l = deque(maxlen=maxlen)
        self._c = deque(maxlen=maxlen)
        self._v = deque(maxlen=maxlen)
        self._ts_last = None

    def _append_from_current(self):
        if not self.current_ohlcv:
            return
        ts = getattr(self.current_ohlcv, 'timestamp', None)
        if self._ts_last == ts:
            return
        self._o.append(self.current_ohlcv.open)
        self._h.append(self.current_ohlcv.high)
        self._l.append(self.current_ohlcv.low)
        self._c.append(self.current_ohlcv.close)
        self._v.append(self.current_ohlcv.volume)
        self._ts_last = ts

    def _as_df(self) -> Optional[pd.DataFrame]:
        if len(self._c) < 5:
            return None
        idx = pd.RangeIndex(len(self._c))
        return pd.DataFrame({
            'open': np.array(self._o, dtype=float),
            'high': np.array(self._h, dtype=float),
            'low': np.array(self._l, dtype=float),
            'close': np.array(self._c, dtype=float),
            'volume': np.array(self._v, dtype=float),
        }, index=idx)


class ADXSignalStrategy(_OHLCVBufferMixin, BaseStrategy):
    """Refactored ADXSignalStrategy supporting LONG/SHORT entries and exits."""
    def __init__(self, strategy_id: str, symbol: str, adx_period: int = 14, atr_mult_tp: float = 4.0, atr_mult_sl: float = 2.0, quantity: int = 1, initial_capital: float = 100000.0):
        self.adx_period = adx_period
        self.atr_mult_tp = atr_mult_tp
        self.atr_mult_sl = atr_mult_sl
        self._next_entry_sl = None
        self._next_entry_tp = None
        entry_conditions = []
        exit_conditions = []
        super().__init__(strategy_id, symbol, entry_conditions, exit_conditions, quantity, initial_capital)
        self._init_buffer()

    def _initialize_indicators(self):
        pass

    def _check_entry_conditions(self) -> bool:
        self._append_from_current()
        df = self._as_df()
        if df is None or len(df) < self.adx_period + 2:
            self._log(f"[ADX] insufficient data: have={len(df) if df is not None else 0}", "DEBUG")
            if self.trading_logger:
                try:
                    self.trading_logger.log_condition_evaluation(self.strategy_id, 'entry', False, {
                        'indicator': 'adx_dmi', 'reason': 'insufficient_data', 'have': int(len(df) if df is not None else 0), 'need': int(self.adx_period + 2)
                    })
                except Exception:
                    pass
            return False
        high, low, close = df['high'], df['low'], df['close']
        adx = talib.ADX(high, low, close, timeperiod=self.adx_period)
        dmp = talib.PLUS_DI(high, low, close, timeperiod=self.adx_period)
        dmn = talib.MINUS_DI(high, low, close, timeperiod=self.adx_period)
        atr = talib.ATR(high, low, close, timeperiod=self.adx_period)
        adx_mean = adx.rolling(20).mean()
        adx_std = adx.rolling(20).std()
        adx_now, adx_prev = adx.iloc[-1], adx.iloc[-2]
        dmp_now, dmn_now = dmp.iloc[-1], dmn.iloc[-1]
        atr_now = atr.iloc[-1]
        cond_long = (adx_prev > (adx_mean.iloc[-1] + adx_std.iloc[-1]) and dmp_now > adx_now and adx_now > dmn_now)
        cond_short = (adx_prev > (adx_mean.iloc[-1] + adx_std.iloc[-1]) and dmn_now > adx_now and adx_now > dmp_now)
        self._log(f"[ADX] entry eval: LONG={cond_long} SHORT={cond_short} | prevADX={adx_prev:.2f} mean={adx_mean.iloc[-1]:.2f} std={adx_std.iloc[-1]:.2f} DMP={dmp_now:.2f} DMN={dmn_now:.2f} ADX={adx_now:.2f}", "DEBUG")
        if self.trading_logger:
            try:
                self.trading_logger.log_condition_evaluation(self.strategy_id, 'entry', cond_long, {
                    'side': 'LONG', 'indicator': 'adx_dmi', 'adx_prev': float(adx_prev), 'adx_now': float(adx_now),
                    'adx_mean': float(adx_mean.iloc[-1]), 'adx_std': float(adx_std.iloc[-1]), 'dmp': float(dmp_now), 'dmn': float(dmn_now)
                })
                self.trading_logger.log_condition_evaluation(self.strategy_id, 'entry', cond_short, {
                    'side': 'SHORT', 'indicator': 'adx_dmi', 'adx_prev': float(adx_prev), 'adx_now': float(adx_now),
                    'adx_mean': float(adx_mean.iloc[-1]), 'adx_std': float(adx_std.iloc[-1]), 'dmp': float(dmp_now), 'dmn': float(dmn_now)
                })
            except Exception:
                pass
        if cond_long:
            entry = close.iloc[-1]
            self._next_entry_sl = float(entry - self.atr_mult_sl * atr_now)
            self._next_entry_tp = float(entry + self.atr_mult_tp * atr_now)
            self._entry_signal_side = 'LONG'
            return True
        if cond_short:
            entry = close.iloc[-1]
            self._next_entry_sl = float(entry + self.atr_mult_sl * atr_now)
            self._next_entry_tp = float(entry - self.atr_mult_tp * atr_now)
            self._entry_signal_side = 'SHORT'
            return True
        return False

    def _check_exit_conditions(self):
        # Pure indicator-based exits are not specified; rely on engine SL/TP and optional reversals
        return None

    def _get_stop_loss_for_side(self, side):
        return self._next_entry_sl

    def _get_take_profit_for_side(self, side):
        return self._next_entry_tp


class OpenRangeBreakoutSignalStrategy(_OHLCVBufferMixin, BaseStrategy):
    """Intraday Open Range Breakout with ADX/DMI/VWAP/EMA filters.

    - Accumulates first 30 bars of the day to define range.
    - LONG: close > range_high and trend filters.
    - SHORT: close < range_low and trend filters.
    - SL/TP: computed from recent swing (lookback) and risk multiple.
    - Exits: via engine SL/TP.
    """
    def __init__(self, strategy_id: str, symbol: str, tp_mult: float = 4.0, sl_lookback: int = 6, quantity: int = 1, initial_capital: float = 100000.0):
        self.tp_mult = tp_mult
        self.sl_lookback = sl_lookback
        self.current_day = None
        self._first_n = 30
        self._count_today = 0
        self._first_highs = []
        self._first_lows = []
        self._next_entry_sl = None
        self._next_entry_tp = None
        entry_conditions = []
        exit_conditions = []
        super().__init__(strategy_id, symbol, entry_conditions, exit_conditions, quantity, initial_capital)
        self._init_buffer()

    def _initialize_indicators(self):
        pass

    def _check_entry_conditions(self) -> bool:
        # Append latest bar
        self._append_from_current()
        if not self.current_ohlcv:
            return False
        ts = getattr(self.current_ohlcv, 'timestamp', None)
        try:
            day = ts.date() if ts is not None else None
        except Exception:
            day = None

        # Day rollover management
        if self.current_day != day:
            self.current_day = day
            self._count_today = 0
            self._first_highs = []
            self._first_lows = []

        # Accumulate first-N bars
        self._count_today += 1
        self._first_highs.append(self.current_ohlcv.high)
        self._first_lows.append(self.current_ohlcv.low)
        if self._count_today <= self._first_n:
            self._log(f"[ORB] Accumulating ORB range ({self._count_today}/{self._first_n}) H={self.current_ohlcv.high:.2f} L={self.current_ohlcv.low:.2f}", "DEBUG")
            if self.trading_logger:
                try:
                    self.trading_logger.log_condition_evaluation(self.strategy_id, 'entry', False, {
                        'indicator': 'orb', 'reason': 'warmup_range', 'count': int(self._count_today), 'required': int(self._first_n)
                    })
                except Exception:
                    pass
            return False

        range_high = max(self._first_highs)
        range_low = min(self._first_lows)

        # Prepare indicators from buffer
        df = self._as_df()
        if df is None or len(df) < max(self.sl_lookback, 10):
            if self.trading_logger:
                try:
                    self.trading_logger.log_condition_evaluation(self.strategy_id, 'entry', False, {
                        'indicator': 'orb', 'reason': 'insufficient_data', 'have': int(len(df) if df is not None else 0), 'need': int(max(self.sl_lookback, 10))
                    })
                except Exception:
                    pass
            return False
        high, low, close = df['high'], df['low'], df['close']
        adx = talib.ADX(high, low, close, timeperiod=5)
        dmp = talib.PLUS_DI(high, low, close, timeperiod=5)
        dmn = talib.MINUS_DI(high, low, close, timeperiod=5)
        atr = talib.ATR(high, low, close, timeperiod=5)
        ema5 = talib.EMA(close, timeperiod=5)
        sma5 = talib.SMA(close, timeperiod=5)
        vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

        adx_now = float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 0.0
        dmp_now = float(dmp.iloc[-1]) if not np.isnan(dmp.iloc[-1]) else 0.0
        dmn_now = float(dmn.iloc[-1]) if not np.isnan(dmn.iloc[-1]) else 0.0
        close_now = float(close.iloc[-1])
        ema5_now = float(ema5.iloc[-1]) if not np.isnan(ema5.iloc[-1]) else close_now
        sma5_now = float(sma5.iloc[-1]) if not np.isnan(sma5.iloc[-1]) else close_now
        vwap_now = float(vwap.iloc[-1]) if not np.isnan(vwap.iloc[-1]) else close_now
        atr_now = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else 0.0

        long_condition = True #(
        #     (close_now > range_high)
        #     and (adx_now > 30)
        #     and (dmp_now > adx_now > dmn_now)
        #     and (close_now > vwap_now or ema5_now > sma5_now)
        # )
        short_condition = (
            (close_now < range_low)
            and (adx_now > 30)
            and (dmp_now < adx_now < dmn_now)
            and (close_now < vwap_now or ema5_now < sma5_now)
        )

        self._log(f"[ORB] LONG={long_condition} SHORT={short_condition} | RH={range_high:.2f} RL={range_low:.2f} ADX={adx_now:.2f} DMP={dmp_now:.2f} DMN={dmn_now:.2f} VWAP={vwap_now:.2f} EMA5={ema5_now:.2f} SMA5={sma5_now:.2f}", "DEBUG")
        if self.trading_logger:
            try:
                self.trading_logger.log_condition_evaluation(self.strategy_id, 'entry', bool(long_condition), {
                    'side': 'LONG', 'indicator': 'orb', 'range_high': float(range_high), 'range_low': float(range_low), 'adx': float(adx_now), 'dmp': float(dmp_now), 'dmn': float(dmn_now), 'vwap': float(vwap_now), 'ema5': float(ema5_now), 'sma5': float(sma5_now)
                })
                self.trading_logger.log_condition_evaluation(self.strategy_id, 'entry', bool(short_condition), {
                    'side': 'SHORT', 'indicator': 'orb', 'range_high': float(range_high), 'range_low': float(range_low), 'adx': float(adx_now), 'dmp': float(dmp_now), 'dmn': float(dmn_now), 'vwap': float(vwap_now), 'ema5': float(ema5_now), 'sma5': float(sma5_now)
                })
            except Exception:
                pass

        if long_condition:
            entry_price = close_now
            sl_price = float(low.iloc[-self.sl_lookback:].min())
            risk = max(entry_price - sl_price, 0.0)
            tp_price = entry_price + self.tp_mult * risk
            self._next_entry_sl = sl_price
            self._next_entry_tp = tp_price
            self._entry_signal_side = 'LONG'
            return True

        if short_condition:
            entry_price = close_now
            sl_price = float(high.iloc[-self.sl_lookback:].max())
            risk = max(sl_price - entry_price, 0.0)
            tp_price = entry_price - self.tp_mult * risk
            self._next_entry_sl = sl_price
            self._next_entry_tp = tp_price
            self._entry_signal_side = 'SHORT'
            return True

        return False

    def _check_exit_conditions(self):
        # Exits enforced by engine SL/TP
        return None

    def _get_stop_loss_for_side(self, side):
        return self._next_entry_sl

    def _get_take_profit_for_side(self, side):
        return self._next_entry_tp


class EMACrossoverStrategy(_OHLCVBufferMixin, BaseStrategy):
    """Two-sided EMA crossover with side-aware SL/TP optional (defaults None)."""
    def __init__(self, strategy_id: str, symbol: str, short: int = 5, long: int = 20, quantity: int = 1, initial_capital: float = 100000.0):
        self.short = short
        self.long = long
        entry_conditions = []
        exit_conditions = []
        super().__init__(strategy_id, symbol, entry_conditions, exit_conditions, quantity, initial_capital)
        self._init_buffer()

    def _initialize_indicators(self):
        pass

    def _check_entry_conditions(self) -> bool:
        self._append_from_current()
        df = self._as_df()
        if df is None or len(df) < self.long + 2:
            if self.trading_logger:
                try:
                    self.trading_logger.log_condition_evaluation(self.strategy_id, 'entry', False, {
                        'indicator': 'ema_cross', 'reason': 'insufficient_data', 'have': int(len(df) if df is not None else 0), 'need': int(self.long + 2)
                    })
                except Exception:
                    pass
            return False
        close = df['close']
        ema_s = close.ewm(span=self.short).mean()
        ema_l = close.ewm(span=self.long).mean()
        prev_s, curr_s = ema_s.iloc[-2], ema_s.iloc[-1]
        prev_l, curr_l = ema_l.iloc[-2], ema_l.iloc[-1]
        # LONG entry
        cross_over = (prev_s <= prev_l and curr_s > curr_l)
        cross_under = (prev_s >= prev_l and curr_s < curr_l)
        self._log(f"[EMA] entry eval: LONG(crossover)={cross_over}, SHORT(crossunder)={cross_under} | prev_s={prev_s:.2f} prev_l={prev_l:.2f} curr_s={curr_s:.2f} curr_l={curr_l:.2f}", "DEBUG")
        if self.trading_logger:
            try:
                self.trading_logger.log_condition_evaluation(self.strategy_id, 'entry', cross_over, {
                    'side': 'LONG', 'indicator': 'ema_cross', 'prev_short': float(prev_s), 'prev_long': float(prev_l), 'curr_short': float(curr_s), 'curr_long': float(curr_l), 'operator': 'crossover'
                })
                self.trading_logger.log_condition_evaluation(self.strategy_id, 'entry', cross_under, {
                    'side': 'SHORT', 'indicator': 'ema_cross', 'prev_short': float(prev_s), 'prev_long': float(prev_l), 'curr_short': float(curr_s), 'curr_long': float(curr_l), 'operator': 'crossunder'
                })
            except Exception:
                pass
        if cross_over:
            self._entry_signal_side = 'LONG'
            return True
        # SHORT entry
        if cross_under:
            self._entry_signal_side = 'SHORT'
            return True
        return False

    def _check_exit_conditions(self):
        self._append_from_current()
        df = self._as_df()
        if df is None or len(df) < self.long + 2:
            return None
        close = df['close']
        ema_s = close.ewm(span=self.short).mean()
        ema_l = close.ewm(span=self.long).mean()
        prev_s, curr_s = ema_s.iloc[-2], ema_s.iloc[-1]
        prev_l, curr_l = ema_l.iloc[-2], ema_l.iloc[-1]
        # Human-readable exit evaluation
        want_long = (self.position_is_long is not False)
        bear_cross = (prev_s >= prev_l and curr_s < curr_l)
        bull_cross = (prev_s <= prev_l and curr_s > curr_l)
        self._log(f"[EMA] exit eval: side={'LONG' if want_long else 'SHORT'} | bear_cross={bear_cross} bull_cross={bull_cross}", "DEBUG")
        if self.trading_logger:
            try:
                self.trading_logger.log_condition_evaluation(self.strategy_id, 'exit', bear_cross if want_long else bull_cross, {
                    'side': 'LONG' if want_long else 'SHORT', 'indicator': 'ema_cross', 'prev_short': float(prev_s), 'prev_long': float(prev_l), 'curr_short': float(curr_s), 'curr_long': float(curr_l), 'operator': 'crossunder' if want_long else 'crossover'
                })
            except Exception:
                pass
        # If LONG, exit on bearish cross
        if want_long and bear_cross:
            from .base import ExitReason
            return ExitReason.SIGNAL
        # If SHORT, exit on bullish cross
        if (self.position_is_long is False) and bull_cross:
            from .base import ExitReason
            return ExitReason.SIGNAL
        return None


class ADXDMISupertrendSignalStrategy(_OHLCVBufferMixin, BaseStrategy):
    """Combined ADX/DMI with Supertrend; computes SL/TP from Supertrend bands."""
    def __init__(self, strategy_id: str, symbol: str, adx_period: int = 7, supertrend_period: int = 7, supertrend_multiplier: float = 2.0, atr_period: int = 7, atr_lookback: int = 30, risk_reward: float = 2.0, quantity: int = 1, initial_capital: float = 100000.0):
        self.adx_period = adx_period
        self.supertrend_period = supertrend_period
        self.supertrend_multiplier = supertrend_multiplier
        self.atr_period = atr_period
        self.atr_lookback = atr_lookback
        self.risk_reward = risk_reward
        self._next_entry_sl = None
        self._next_entry_tp = None
        entry_conditions = []
        exit_conditions = []
        super().__init__(strategy_id, symbol, entry_conditions, exit_conditions, quantity, initial_capital)
        self._init_buffer()

    def _initialize_indicators(self):
        pass

    def _supertrend(self, df: pd.DataFrame) -> pd.DataFrame:
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=self.supertrend_period)
        hl2 = (df['high'] + df['low']) / 2.0
        upperband = hl2 + self.supertrend_multiplier * atr
        lowerband = hl2 - self.supertrend_multiplier * atr
        st = np.zeros(len(df))
        for i in range(1, len(df)):
            if df['close'].iloc[i] > upperband.iloc[i - 1]:
                st[i] = lowerband.iloc[i]
            elif df['close'].iloc[i] < lowerband.iloc[i - 1]:
                st[i] = upperband.iloc[i]
            else:
                st[i] = st[i - 1]
        return pd.DataFrame({'supertrend': st, 'upperband': upperband, 'lowerband': lowerband})

    def _check_entry_conditions(self) -> bool:
        self._append_from_current()
        df = self._as_df()
        if df is None or len(df) < max(self.adx_period, self.supertrend_period, self.atr_lookback) + 2:
            if self.trading_logger:
                try:
                    self.trading_logger.log_condition_evaluation(self.strategy_id, 'entry', False, {
                        'indicator': 'adx_dmi_supertrend', 'reason': 'insufficient_data', 'have': int(len(df) if df is not None else 0), 'need': int(max(self.adx_period, self.supertrend_period, self.atr_lookback) + 2)
                    })
                except Exception:
                    pass
            return False
        high, low, close = df['high'], df['low'], df['close']
        adx = talib.ADX(high, low, close, timeperiod=self.adx_period)
        dmi_plus = talib.PLUS_DI(high, low, close, timeperiod=self.adx_period)
        dmi_minus = talib.MINUS_DI(high, low, close, timeperiod=self.adx_period)
        atr = talib.ATR(high, low, close, timeperiod=self.atr_period)
        atr_mean = atr.rolling(window=self.atr_lookback).mean()
        st = self._supertrend(df)
        st_upper, st_lower = st['upperband'].iloc[-1], st['lowerband'].iloc[-1]
        adx_now = adx.iloc[-1]
        dmi_plus_now = dmi_plus.iloc[-1]
        dmi_minus_now = dmi_minus.iloc[-1]
        atr_now = atr.iloc[-1]
        atr_mean_now = atr_mean.iloc[-1]
        close_now = close.iloc[-1]
        long_condition = (adx_now > 40 and atr_now > atr_mean_now and dmi_plus_now > dmi_minus_now)
        short_condition = (adx_now > 40 and atr_now > atr_mean_now and dmi_minus_now > dmi_plus_now)
        self._log(f"[ADX/ST] entry eval: LONG={long_condition} SHORT={short_condition} | ADX={adx_now:.2f} DMI+={dmi_plus_now:.2f} DMI-={dmi_minus_now:.2f} ATR={atr_now:.4f} ATRmean={atr_mean_now:.4f}", "DEBUG")
        if self.trading_logger:
            try:
                self.trading_logger.log_condition_evaluation(self.strategy_id, 'entry', bool(long_condition), {
                    'side': 'LONG', 'indicator': 'adx_dmi_supertrend', 'adx': float(adx_now), 'dmi_plus': float(dmi_plus_now), 'dmi_minus': float(dmi_minus_now), 'atr': float(atr_now), 'atr_mean': float(atr_mean_now), 'st_lower': float(st_lower), 'st_upper': float(st_upper)
                })
                self.trading_logger.log_condition_evaluation(self.strategy_id, 'entry', bool(short_condition), {
                    'side': 'SHORT', 'indicator': 'adx_dmi_supertrend', 'adx': float(adx_now), 'dmi_plus': float(dmi_plus_now), 'dmi_minus': float(dmi_minus_now), 'atr': float(atr_now), 'atr_mean': float(atr_mean_now), 'st_lower': float(st_lower), 'st_upper': float(st_upper)
                })
            except Exception:
                pass
        if long_condition:
            entry_price = close_now
            stop_loss = st_lower
            risk = entry_price - stop_loss
            take_profit = entry_price + risk * self.risk_reward
            self._next_entry_sl = float(stop_loss)
            self._next_entry_tp = float(take_profit)
            self._entry_signal_side = 'LONG'
            return True
        if short_condition:
            entry_price = close_now
            stop_loss = st_upper
            risk = stop_loss - entry_price
            take_profit = entry_price - risk * self.risk_reward
            self._next_entry_sl = float(stop_loss)
            self._next_entry_tp = float(take_profit)
            self._entry_signal_side = 'SHORT'
            return True
        return False

    def _check_exit_conditions(self):
        # Exits rely on engine SL/TP or user-defined additional signals
        return None

    def _get_stop_loss_for_side(self, side):
        return self._next_entry_sl

    def _get_take_profit_for_side(self, side):
        return self._next_entry_tp

class MovingAverageCrossoverStrategy(BaseStrategy):
    """Moving average crossover strategy"""
    
    def __init__(
        self,
        strategy_id: str,
        symbol: str,
        fast_period: int = 10,
        slow_period: int = 30,
        quantity: int = 1,
        stop_loss_pct: float = 0.02,  # 2% stop loss
        take_profit_pct: float = 0.04,  # 4% take profit
        initial_capital: float = 100000.0
    ):
        # Store period attributes BEFORE calling super().__init__()
        # because _initialize_indicators() is called in base class __init__
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Create entry/exit conditions
        entry_conditions = [
            EntryCondition(
                indicator_name="fast_sma",
                operator=">",
                threshold=0.0,  # Dummy threshold, will use indicator comparison
                additional_params={"compare_to": "slow_sma"}
            )
        ]
        
        exit_conditions = [
            ExitCondition(
                condition_type="indicator",
                indicator_name="fast_sma",
                operator="crossunder",
                threshold=0.0  # Will be evaluated against slow_sma
            ),
            ExitCondition(
                condition_type="stop_loss",
                stop_loss=None  # Will be calculated dynamically
            ),
            ExitCondition(
                condition_type="take_profit",
                take_profit=None  # Will be calculated dynamically
            )
        ]
        
        super().__init__(
            strategy_id=strategy_id,
            symbol=symbol,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            quantity=quantity,
            initial_capital=initial_capital
        )
    
    def _initialize_indicators(self):
        """Initialize SMA indicators"""
        self.indicators["fast_sma"] = SMA(self.fast_period)
        self.indicators["slow_sma"] = SMA(self.slow_period)
    
    def _check_entry_conditions(self) -> bool:
        """Check for moving average crossover"""
        fast_sma = self.indicators.get("fast_sma")
        slow_sma = self.indicators.get("slow_sma")
        
        if not fast_sma or not slow_sma:
            self._log("Missing SMA indicators", "DEBUG")
            if self.trading_logger:
                try:
                    self.trading_logger.log_condition_evaluation(
                        self.strategy_id,
                        'entry',
                        False,
                        {
                            'indicator': 'sma_crossover',
                            'reason': 'missing_indicators',
                            'has_fast': bool(fast_sma),
                            'has_slow': bool(slow_sma),
                            'fast_period': getattr(self, 'fast_period', None),
                            'slow_period': getattr(self, 'slow_period', None),
                        }
                    )
                except Exception as e:
                    self._log(f"Error logging entry condition: {str(e)}", "ERROR")
            return False
        
        if fast_sma.current_value is None or slow_sma.current_value is None:
            self._log("SMA indicators not initialized", "DEBUG")
            if self.trading_logger:
                try:
                    self.trading_logger.log_condition_evaluation(
                        self.strategy_id,
                        'entry',
                        False,
                        {
                            'indicator': 'sma_crossover',
                            'reason': 'indicator_not_initialized',
                            'fast_current': getattr(fast_sma, 'current_value', None),
                            'slow_current': getattr(slow_sma, 'current_value', None),
                            'fast_period': self.fast_period,
                            'slow_period': self.slow_period,
                        }
                    )
                except Exception as e:
                    self._log(f"Error logging entry condition: {str(e)}", "ERROR")
            return False
        
        # Check for MA comparison (support LONG and SHORT)
        result = False
        if len(fast_sma.values) >= 2 and len(slow_sma.values) >= 2:
            prev_fast = fast_sma.values[-2]
            prev_slow = slow_sma.values[-2]
            curr_fast = fast_sma.values[-1]
            curr_slow = slow_sma.values[-1]
            
            # Log the current values of both indicators
            self._log(f"Checking Fast SMA {self.fast_period} vs Slow SMA {self.slow_period}", "DEBUG")
            self._log(f"Fast SMA={curr_fast:.2f} (prev={prev_fast:.2f})", "DEBUG")
            self._log(f"Slow SMA={curr_slow:.2f} (prev={prev_slow:.2f})", "DEBUG")
            
            # Check if fast crosses above slow -> LONG
            cross_over = prev_fast <= prev_slow and curr_fast > curr_slow
            # Check if fast crosses below slow -> SHORT
            cross_under = prev_fast >= prev_slow and curr_fast < curr_slow

            # Human-readable evaluation for both sides
            self._log(f"MA entry eval: LONG(crossover)={cross_over}, SHORT(crossunder)={cross_under}", "DEBUG")

            # Emit structured evaluations for both LONG and SHORT checks
            if self.trading_logger:
                try:
                    self.trading_logger.log_condition_evaluation(
                        self.strategy_id,
                        'entry',
                        cross_over,
                        {
                            'side': 'LONG',
                            'indicator': 'sma_crossover',
                            'prev_fast_sma': prev_fast,
                            'prev_slow_sma': prev_slow,
                            'curr_fast_sma': curr_fast,
                            'curr_slow_sma': curr_slow,
                            'operator': 'crossover',
                            'fast_period': self.fast_period,
                            'slow_period': self.slow_period,
                        }
                    )
                    self.trading_logger.log_condition_evaluation(
                        self.strategy_id,
                        'entry',
                        cross_under,
                        {
                            'side': 'SHORT',
                            'indicator': 'sma_crossunder',
                            'prev_fast_sma': prev_fast,
                            'prev_slow_sma': prev_slow,
                            'curr_fast_sma': curr_fast,
                            'curr_slow_sma': curr_slow,
                            'operator': 'crossunder',
                            'fast_period': self.fast_period,
                            'slow_period': self.slow_period,
                        }
                    )
                except Exception:
                    pass

            if cross_over:
                result = True
                self._entry_signal_side = 'LONG'
                self._log(f"Entry signal: Fast SMA crossed ABOVE Slow SMA (LONG)", "INFO")
            elif cross_under:
                result = True
                self._entry_signal_side = 'SHORT'
                self._log(f"Entry signal: Fast SMA crossed BELOW Slow SMA (SHORT)", "INFO")
            else:
                self._log("No SMA crossover entry signal", "DEBUG")
            
            # Log condition evaluation to structured log
            if self.trading_logger:
                condition_details = {
                    'fast_sma': curr_fast,
                    'slow_sma': curr_slow,
                    'prev_fast_sma': prev_fast,
                    'prev_slow_sma': prev_slow,
                    'fast_period': self.fast_period,
                    'slow_period': self.slow_period,
                    'result': result
                }
                try:
                    self.trading_logger.log_condition_evaluation(
                        self.strategy_id,
                        'entry',
                        result,
                        condition_details
                    )
                except Exception as e:
                    self._log(f"Error logging condition: {str(e)}", "ERROR")
        
        else:
            # Not enough values yet to detect a crossover
            self._log(
                f"Not enough SMA values to evaluate crossover (fast_len={len(fast_sma.values)}, slow_len={len(slow_sma.values)})",
                "DEBUG",
            )
            if self.trading_logger:
                try:
                    self.trading_logger.log_condition_evaluation(
                        self.strategy_id,
                        'entry',
                        False,
                        {
                            'indicator': 'sma_crossover',
                            'reason': 'insufficient_data',
                            'fast_len': len(fast_sma.values) if hasattr(fast_sma, 'values') else None,
                            'slow_len': len(slow_sma.values) if hasattr(slow_sma, 'values') else None,
                            'fast_period': self.fast_period,
                            'slow_period': self.slow_period,
                        }
                    )
                except Exception as e:
                    self._log(f"Error logging entry condition: {str(e)}", "ERROR")
        return result
    
    def _check_exit_conditions(self):
        """Check exit conditions including crossover"""
        # Check for crossunder
        fast_sma = self.indicators.get("fast_sma")
        slow_sma = self.indicators.get("slow_sma")
        
        if fast_sma and slow_sma:
            if fast_sma.current_value is not None and slow_sma.current_value is not None:
                if len(fast_sma.values) >= 2 and len(slow_sma.values) >= 2:
                    prev_fast = fast_sma.values[-2]
                    prev_slow = slow_sma.values[-2]
                    curr_fast = fast_sma.values[-1]
                    curr_slow = slow_sma.values[-1]
                    
                    # Log current SMA state
                    self._log(f"Exit check (SMA crossunder): Fast SMA={curr_fast:.2f} (prev={prev_fast:.2f}), Slow SMA={curr_slow:.2f} (prev={prev_slow:.2f})", "DEBUG")
                    # Decide exit crossover depending on current side
                    # If currently LONG -> exit on crossunder; if SHORT -> exit on crossover
                    want_crossunder = (self.position_is_long is not False)
                    crossunder = prev_fast >= prev_slow and curr_fast < curr_slow
                    crossover = prev_fast <= prev_slow and curr_fast > curr_slow
                    exit_on_signal = crossunder if want_crossunder else crossover
                    # Human-readable evaluation for both sides
                    self._log(f"MA exit eval: want={'LONG' if want_crossunder else 'SHORT'} -> crossunder={crossunder}, crossover={crossover}", "DEBUG")
                    if self.trading_logger:
                        try:
                            self.trading_logger.log_condition_evaluation(
                                self.strategy_id,
                                'exit',
                                exit_on_signal,
                                {
                                    'side': 'LONG' if want_crossunder else 'SHORT',
                                    'indicator': 'sma_cross',
                                    'fast_period': self.fast_period,
                                    'slow_period': self.slow_period,
                                    'prev_fast_sma': prev_fast,
                                    'prev_slow_sma': prev_slow,
                                    'curr_fast_sma': curr_fast,
                                    'curr_slow_sma': curr_slow,
                                    'operator': 'crossunder' if want_crossunder else 'crossover',
                                    'threshold': 0.0,
                                }
                            )
                            # Also emit explicit other side evaluation for completeness
                            self.trading_logger.log_condition_evaluation(
                                self.strategy_id,
                                'exit',
                                crossover if want_crossunder else crossunder,
                                {
                                    'side': 'SHORT' if want_crossunder else 'LONG',
                                    'indicator': 'sma_cross',
                                    'fast_period': self.fast_period,
                                    'slow_period': self.slow_period,
                                    'prev_fast_sma': prev_fast,
                                    'prev_slow_sma': prev_slow,
                                    'curr_fast_sma': curr_fast,
                                    'curr_slow_sma': curr_slow,
                                    'operator': 'crossover' if want_crossunder else 'crossunder',
                                    'threshold': 0.0,
                                }
                            )
                        except Exception as e:
                            self._log(f"Error logging exit condition: {str(e)}", "ERROR")
                    if exit_on_signal:
                        self._log("Exit signal: SMA cross exit triggered", "INFO")
                        from .base import ExitReason
                        return ExitReason.SIGNAL
        
        # Check stop loss and take profit (side-aware)
        if self.position_entry_price and self.current_price:
            is_long = (self.position_is_long is not False)
            # Compute side-aware thresholds
            sl_price = self.position_entry_price * (1 - self.stop_loss_pct) if is_long else self.position_entry_price * (1 + self.stop_loss_pct)
            tp_price = self.position_entry_price * (1 + self.take_profit_pct) if is_long else self.position_entry_price * (1 - self.take_profit_pct)

            # Stop loss condition (LONG: price <= sl; SHORT: price >= sl)
            sl_hit = (self.current_price <= sl_price) if is_long else (self.current_price >= sl_price)
            self._log(
                f"SL check ({'LONG' if is_long else 'SHORT'}): price={self.current_price:.2f} vs SL={sl_price:.2f} -> {sl_hit}",
                "DEBUG",
            )
            if self.trading_logger:
                try:
                    self.trading_logger.log_condition_evaluation(
                        self.strategy_id,
                        'exit',
                        sl_hit,
                        {
                            'side': 'LONG' if is_long else 'SHORT',
                            'condition_type': 'stop_loss',
                            'stop_loss': sl_price,
                            'current_price': self.current_price,
                            'entry_price': self.position_entry_price,
                            'stop_loss_pct': self.stop_loss_pct,
                        }
                    )
                except Exception as e:
                    self._log(f"Error logging SL condition: {str(e)}", "ERROR")
            if sl_hit:
                from .base import ExitReason
                return ExitReason.STOP_LOSS

            # Take profit condition (LONG: price >= tp; SHORT: price <= tp)
            tp_hit = (self.current_price >= tp_price) if is_long else (self.current_price <= tp_price)
            self._log(
                f"TP check ({'LONG' if is_long else 'SHORT'}): price={self.current_price:.2f} vs TP={tp_price:.2f} -> {tp_hit}",
                "DEBUG",
            )
            if self.trading_logger:
                try:
                    self.trading_logger.log_condition_evaluation(
                        self.strategy_id,
                        'exit',
                        tp_hit,
                        {
                            'side': 'LONG' if is_long else 'SHORT',
                            'condition_type': 'take_profit',
                            'take_profit': tp_price,
                            'current_price': self.current_price,
                            'entry_price': self.position_entry_price,
                            'take_profit_pct': self.take_profit_pct,
                        }
                    )
                except Exception as e:
                    self._log(f"Error logging TP condition: {str(e)}", "ERROR")
            if tp_hit:
                from .base import ExitReason
                return ExitReason.TAKE_PROFIT
        
        return None
    
    def _get_stop_loss(self) -> Optional[float]:
        """Calculate stop loss price"""
        if self.current_price:
            return self.current_price * (1 - self.stop_loss_pct)
        return None
    
    def _get_take_profit(self) -> Optional[float]:
        """Calculate take profit price"""
        if self.current_price:
            return self.current_price * (1 + self.take_profit_pct)
        return None

    # Side-aware overrides for MA strategy
    def _get_stop_loss_for_side(self, side):
        if self.current_price is None:
            return None
        if getattr(side, 'name', '') == 'SELL':  # SHORT
            return self.current_price * (1 + self.stop_loss_pct)
        return self.current_price * (1 - self.stop_loss_pct)

    def _get_take_profit_for_side(self, side):
        if self.current_price is None:
            return None
        if getattr(side, 'name', '') == 'SELL':  # SHORT
            return self.current_price * (1 - self.take_profit_pct)
        return self.current_price * (1 + self.take_profit_pct)


class RSIStrategy(BaseStrategy):
    """RSI-based mean reversion strategy"""
    
    def __init__(
        self,
        strategy_id: str,
        symbol: str,
        rsi_period: int = 14,
        oversold_threshold: float = 30.0,
        overbought_threshold: float = 70.0,
        quantity: int = 1,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.03,
        initial_capital: float = 100000.0
    ):
        # Store attributes BEFORE calling super().__init__()
        self.rsi_period = rsi_period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        entry_conditions = [
            EntryCondition(
                indicator_name="rsi",
                operator="<",
                threshold=oversold_threshold
            )
        ]
        
        exit_conditions = [
            ExitCondition(
                condition_type="indicator",
                indicator_name="rsi",
                operator=">",
                threshold=overbought_threshold
            ),
            ExitCondition(
                condition_type="stop_loss",
                stop_loss=None
            ),
            ExitCondition(
                condition_type="take_profit",
                take_profit=None
            )
        ]
        
        super().__init__(
            strategy_id=strategy_id,
            symbol=symbol,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            quantity=quantity,
            initial_capital=initial_capital
        )
    
    def _initialize_indicators(self):
        """Initialize RSI indicator"""
        self.indicators["rsi"] = RSI(self.rsi_period)

    def _check_entry_conditions(self) -> bool:
        """Enter LONG on oversold; enter SHORT on overbought."""
        rsi = self.indicators.get("rsi")
        if not rsi or rsi.current_value is None:
            if self.trading_logger:
                try:
                    self.trading_logger.log_condition_evaluation(self.strategy_id, 'entry', False, {
                        'indicator': 'rsi', 'reason': 'not_ready', 'rsi': getattr(rsi, 'current_value', None)
                    })
                except Exception:
                    pass
            return False
        val = rsi.current_value
        go_long = val < self.oversold_threshold
        go_short = val > self.overbought_threshold
        self._log(f"RSI entry check: rsi={val:.2f}, LONG(val<oversold)={go_long}, SHORT(val>overbought)={go_short}", "DEBUG")
        # Structured logs for both sides
        if self.trading_logger:
            try:
                self.trading_logger.log_condition_evaluation(self.strategy_id, 'entry', go_long, {
                    'side': 'LONG', 'indicator': 'rsi', 'operator': '<', 'threshold': self.oversold_threshold, 'value': val
                })
                self.trading_logger.log_condition_evaluation(self.strategy_id, 'entry', go_short, {
                    'side': 'SHORT', 'indicator': 'rsi', 'operator': '>', 'threshold': self.overbought_threshold, 'value': val
                })
            except Exception:
                pass
        if go_long:
            self._entry_signal_side = 'LONG'
            return True
        if go_short:
            self._entry_signal_side = 'SHORT'
            return True
        return False

    def _check_exit_conditions(self):
        """Exit LONG when RSI > overbought; exit SHORT when RSI < oversold; also apply SL/TP from base."""
        rsi = self.indicators.get("rsi")
        if rsi and rsi.current_value is not None:
            val = rsi.current_value
            exit_long = (self.position_is_long is not False) and (val > self.overbought_threshold)
            exit_short = (self.position_is_long is False) and (val < self.oversold_threshold)
            result = exit_long or exit_short
            self._log(f"RSI exit eval: side={'LONG' if (self.position_is_long is not False) else 'SHORT'}, value={val:.2f}, exit_long={exit_long}, exit_short={exit_short}", "DEBUG")
            if self.trading_logger:
                try:
                    self.trading_logger.log_condition_evaluation(self.strategy_id, 'exit', exit_long, {
                        'side': 'LONG', 'indicator': 'rsi', 'operator': '>', 'threshold': self.overbought_threshold, 'value': val
                    })
                    self.trading_logger.log_condition_evaluation(self.strategy_id, 'exit', exit_short, {
                        'side': 'SHORT', 'indicator': 'rsi', 'operator': '<', 'threshold': self.oversold_threshold, 'value': val
                    })
                except Exception:
                    pass
            if result:
                from .base import ExitReason
                return ExitReason.SIGNAL
        # Fallback to base stop-loss/take-profit checks
        return super()._check_exit_conditions()

    # Side-aware overrides for RSI strategy
    def _get_stop_loss_for_side(self, side):
        if self.current_price is None:
            return None
        if getattr(side, 'name', '') == 'SELL':
            return self.current_price * (1 + self.stop_loss_pct)
        return self.current_price * (1 - self.stop_loss_pct)

    def _get_take_profit_for_side(self, side):
        if self.current_price is None:
            return None
        if getattr(side, 'name', '') == 'SELL':
            return self.current_price * (1 - self.take_profit_pct)
        return self.current_price * (1 + self.take_profit_pct)
    
    def _get_stop_loss(self) -> Optional[float]:
        """Calculate stop loss price"""
        if self.current_price:
            return self.current_price * (1 - self.stop_loss_pct)
        return None
    
    def _get_take_profit(self) -> Optional[float]:
        """Calculate take profit price"""
        if self.current_price:
            return self.current_price * (1 + self.take_profit_pct)
        return None

