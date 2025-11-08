"""
Comprehensive logging system for trading operations.
"""
import logging
import json
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path


class TradingLogger:
    """Logger for all trading operations"""
    
    def __init__(self, log_file: str = "trading.log", log_level: str = "INFO"):
        self.log_file = log_file
        self.log_level = getattr(logging, log_level.upper())
        
        # Create logger
        self.logger = logging.getLogger("TradingSystem")
        self.logger.setLevel(self.log_level)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(self.log_level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Separate log file for structured data
        self.structured_log_file = log_file.replace('.log', '_structured.jsonl')
        self.structured_log = open(self.structured_log_file, 'a')
    
    def log_tick(self, tick_data: Dict[str, Any]):
        """Log tick data"""
        log_entry = {
            'type': 'tick',
            'timestamp': datetime.now().isoformat(),
            'data': tick_data
        }
        self._write_structured_log(log_entry)
        self.logger.debug(f"Tick: {tick_data}")
    
    def log_ohlcv(self, ohlcv_data: Dict[str, Any]):
        """Log OHLCV data"""
        log_entry = {
            'type': 'ohlcv',
            'timestamp': datetime.now().isoformat(),
            'data': ohlcv_data
        }
        self._write_structured_log(log_entry)
        self.logger.debug(f"OHLCV: {ohlcv_data}")
    
    def log_order(self, order_data: Dict[str, Any]):
        """Log order submission"""
        log_entry = {
            'type': 'order',
            'timestamp': datetime.now().isoformat(),
            'data': order_data
        }
        self._write_structured_log(log_entry)
        self.logger.info(f"Order: {order_data}")
    
    def log_fill(self, fill_data: Dict[str, Any]):
        """Log order fill"""
        log_entry = {
            'type': 'fill',
            'timestamp': datetime.now().isoformat(),
            'data': fill_data
        }
        self._write_structured_log(log_entry)
        self.logger.info(f"Fill: {fill_data}")
    
    def log_rms_check(self, rms_result: Dict[str, Any], order_data: Dict[str, Any]):
        """Log RMS check"""
        log_entry = {
            'type': 'rms_check',
            'timestamp': datetime.now().isoformat(),
            'data': {
                'rms_result': rms_result,
                'order': order_data
            }
        }
        self._write_structured_log(log_entry)
        if not rms_result.get('allowed', False):
            self.logger.warning(f"RMS check failed: {rms_result}")
        else:
            self.logger.debug(f"RMS check passed: {rms_result}")
    
    def log_condition_evaluation(self, strategy_id: str, condition_type: str, result: bool, details: Dict[str, Any]):
        """Log condition evaluation"""
        log_entry = {
            'type': 'condition_evaluation',
            'timestamp': datetime.now().isoformat(),
            'data': {
                'strategy_id': strategy_id,
                'condition_type': condition_type,
                'result': result,
                'details': details
            }
        }
        self._write_structured_log(log_entry)
        # Build a richer human-readable message when possible
        try:
            human_msg = f"Condition evaluation [{strategy_id}]: {condition_type} = {result}"

            def _safe_fmt(x):
                if x is None:
                    return 'None'
                try:
                    if isinstance(x, (int, float)):
                        return f"{x:.4f}"
                    return str(x)
                except Exception:
                    return str(x)

            # If details provide a list of conditions, summarize them
            if isinstance(details, dict):
                if 'conditions' in details and isinstance(details['conditions'], list):
                    conds = details['conditions']
                    human_msg += f" — {len(conds)} condition(s): "
                    parts = []
                    for i, c in enumerate(conds):
                        name = c.get('indicator') or c.get('indicator_name') or f"cond_{i}"
                        op = c.get('operator', '')
                        thresh = c.get('threshold', c.get('indicator_value', None))
                        val = c.get('indicator_value', None)
                        passed = c.get('result', None)
                        parts.append(f"{i+1}:{name} {op} {_safe_fmt(thresh)} -> {_safe_fmt(val)} = {passed}")
                    human_msg += ", ".join(parts)
                # If a single condition evaluation was logged (condition_index present), give detail
                elif 'condition_index' in details:
                    idx = details.get('condition_index')
                    name = details.get('indicator') or details.get('indicator_name')
                    passed = details.get('result')

                    # Prefer indicator value info when available
                    ind_val = details.get('indicator_value') if isinstance(details, dict) else None
                    ind_curr = details.get('indicator_curr') if isinstance(details, dict) else None

                    lhs = details.get('ohlcv_lhs') if isinstance(details, dict) else None
                    rhs = details.get('ohlcv_rhs') if isinstance(details, dict) else None

                    # Only show OHLCV lhs/rhs when they are not None
                    if lhs is not None or rhs is not None:
                        human_msg += f" — condition[{idx}] {name} {details.get('operator','?')} {details.get('threshold', '')} -> lhs={_safe_fmt(lhs)}, rhs={_safe_fmt(rhs)}, result={passed}"
                    # Otherwise prefer indicator values if present
                    elif ind_val is not None or ind_curr is not None:
                        # Show current indicator or single indicator_value
                        shown = ind_val if ind_val is not None else ind_curr
                        human_msg += f" — condition[{idx}] {name} {details.get('operator','?')} {details.get('threshold','')} -> value={_safe_fmt(shown)}, result={passed}"
                    else:
                        human_msg += f" — condition[{idx}] {name} result={passed}"

        except Exception:
            human_msg = f"Condition evaluation [{strategy_id}]: {condition_type} = {result}"

        self.logger.debug(human_msg)
    
    def log_portfolio_update(self, portfolio_data: Dict[str, Any]):
        """Log portfolio update"""
        log_entry = {
            'type': 'portfolio_update',
            'timestamp': datetime.now().isoformat(),
            'data': portfolio_data
        }
        self._write_structured_log(log_entry)
        self.logger.info(f"Portfolio update: P&L={portfolio_data.get('total_pnl', 0):.2f}, Equity={portfolio_data.get('total_equity', 0):.2f}")
    
    def log_error(self, error_message: str, error_details: Optional[Dict[str, Any]] = None):
        """Log error"""
        log_entry = {
            'type': 'error',
            'timestamp': datetime.now().isoformat(),
            'data': {
                'message': error_message,
                'details': error_details
            }
        }
        self._write_structured_log(log_entry)
        # logging.error's exc_info expects an exception info tuple or bool; avoid passing arbitrary dicts
        self.logger.error(f"Error: {error_message}")
        if error_details:
            # Provide details at debug level to avoid cluttering errors with non-exception payloads
            self.logger.debug(f"Error details: {error_details}")
    
    def _write_structured_log(self, log_entry: Dict[str, Any]):
        """Write structured log entry"""
        try:
            # Convert common non-serializable types (datetime, numpy scalars, objects)
            def _serialize(obj):
                # Local import to avoid hard dependency if numpy is not installed
                try:
                    import numpy as _np
                except Exception:
                    _np = None

                # datetime -> isoformat
                if isinstance(obj, datetime):
                    return obj.isoformat()

                # numpy scalar types -> native python types
                if _np is not None:
                    if isinstance(obj, _np.integer):
                        return int(obj)
                    if isinstance(obj, _np.floating):
                        return float(obj)
                    if isinstance(obj, _np.bool_):
                        return bool(obj)

                # Objects exposing to_dict -> use that
                if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
                    try:
                        return obj.to_dict()
                    except Exception:
                        pass

                # Fallback to string
                try:
                    return str(obj)
                except Exception:
                    return repr(obj)

            self.structured_log.write(json.dumps(log_entry, default=_serialize) + '\n')
            self.structured_log.flush()
        except Exception as e:
            # As a last resort try dumping with default=str to avoid losing logs
            try:
                self.structured_log.write(json.dumps(log_entry, default=str) + '\n')
                self.structured_log.flush()
            except Exception:
                self.logger.error(f"Failed to write structured log: {e}")
    
    def close(self):
        """Close log files"""
        if hasattr(self, 'structured_log'):
            self.structured_log.close()

