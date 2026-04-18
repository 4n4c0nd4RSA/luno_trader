import json

APP_TITLE = "Luno Trader"
CONFIG_FILE = "luno_trader_config.json"
AUDIT_HISTORY_FILE = "trade_audit_history.json"

SIGNAL_OPTIONS = [
    "EMA9",
    "EMA21",
    "EMA50",
    "EMA200",
    "RSI14",
    "MACD",
    "MACD_SIGNAL",
    "MACD_HIST",
    "CHANNEL_POS",
    "ATR14",
    "ATR_PCT",
    "ADX14",
    "+DI14",
    "-DI14",
    "VOL_SMA20",
    "VOL_RATIO",
    "RET_1",
    "RET_6",
    "RET_24",
    "PRICE_DELTA",
    "PRICE_DELTA_PCT",
    "PRICE_ACCEL",
    "PRICE_ACCEL_PCT",
    "ABS_PRICE_DELTA_PCT",
    "ABS_PRICE_ACCEL_PCT",
    "RED_CANDLE",
    "RED_BODY_DIFF",
    "SELL_RED_OK",
    "MACD_CROSS_UP",
    "MACD_CROSS_DOWN",
    "MACD_HIST_IMPROVING",
    "MACD_HIST_WEAKENING",
    "LIVE_PROFIT_PCT",
    "MAX_PROFIT_PCT",
]

OPERATOR_OPTIONS = [">", "<", ">=", "<=", "==", "!="]
RIGHT_TYPE_OPTIONS = ["signal", "number", "boolean", "config"]


def deep_copy_json(data):
    return json.loads(json.dumps(data))


DEFAULT_RULES = [
    {
        "name": "bullish_entry",
        "active": True,
        "action": "BUY",
        "salience": 1,
        "checks": [
            {
                "left_type": "signal",
                "left_signal": "MACD",
                "operator": ">",
                "right_type": "number",
                "right_value": 10.0,
            },
            {
                "left_type": "signal",
                "left_signal": "MACD",
                "operator": ">",
                "right_type": "signal",
                "right_value": "MACD_SIGNAL",
            },
            {
                "left_type": "signal",
                "left_signal": "CHANNEL_POS",
                "operator": "<",
                "right_type": "config",
                "right_value": "max_buy_channel_pos",
            },
        ],
    },
    {
        "name": "bearish_reversal",
        "active": True,
        "action": "SELL",
        "salience": 2,
        "checks": [
            {
                "left_type": "signal",
                "left_signal": "MACD_CROSS_DOWN",
                "operator": "==",
                "right_type": "boolean",
                "right_value": True,
            },
            {
                "left_type": "signal",
                "left_signal": "RED_CANDLE",
                "operator": "==",
                "right_type": "boolean",
                "right_value": True,
            },
        ],
    },
    {
        "name": "bearish_reversal2",
        "active": True,
        "action": "SELL",
        "salience": 2,
        "checks": [
            {
                "left_type": "signal",
                "left_signal": "MACD_CROSS_DOWN",
                "operator": "==",
                "right_type": "boolean",
                "right_value": True,
            },
            {
                "left_type": "signal",
                "left_signal": "MACD_HIST_WEAKENING",
                "operator": "==",
                "right_type": "boolean",
                "right_value": True,
            },
        ],
    },
    {
        "name": "stop_loss_exit",
        "active": True,
        "action": "SELL",
        "salience": 3,
        "checks": [
            {
                "left_type": "signal",
                "left_signal": "LIVE_PROFIT_PCT",
                "operator": "<=",
                "right_type": "config",
                "right_value": "stop_loss_pct",
            }
        ],
    },
    {
        "name": "bearish_profit_exit",
        "active": True,
        "action": "SELL",
        "salience": 4,
        "checks": [
            {
                "left_type": "rule",
                "left_rule": "bearish_reversal",
                "operator": "==",
                "right_type": "boolean",
                "right_value": True,
            },
            {
                "left_type": "signal",
                "left_signal": "LIVE_PROFIT_PCT",
                "operator": ">=",
                "right_type": "config",
                "right_value": "min_bearish_exit_profit_pct",
            },
        ],
    },
    {
        "name": "bearish_profit_exit2",
        "active": True,
        "action": "SELL",
        "salience": 4,
        "checks": [
            {
                "left_type": "rule",
                "left_rule": "bearish_reversal2",
                "operator": "==",
                "right_type": "boolean",
                "right_value": True,
            },
            {
                "left_type": "signal",
                "left_signal": "LIVE_PROFIT_PCT",
                "operator": ">=",
                "right_type": "config",
                "right_value": "min_bearish_exit_profit_pct",
            },
        ],
    },
]

DEFAULT_CONFIG = {
    "mock_trade": True,
    "audit_history_file": AUDIT_HISTORY_FILE,
    "api_key_id": "",
    "api_key_secret": "",
    "history_hours": 1000,
    "candle_seconds": 3600,
    "trend_window_hours": 48,
    "backtest_start_zar": 10000.0,
    "trade_buy_fee_rate": 0.005,
    "trade_sell_fee_rate": 0.003,
    "current_entry_price": 0.0,
    "raw_score_buy_bias": 10.0,
    "raw_score_sell_bias": -40.0,
    "sell_red_body_min_diff": 10000.0,
    "entry_min_adx": 20.0,
    "entry_min_hist": 0.0,
    "early_fail_max_profit_pct": 1.5,
    "min_bearish_exit_profit_pct": 1.0, #3.5
    "stop_loss_pct": -4.5, #!-2.5
    "min_trail_profit_pct": 1.0,
    "trail_giveback_ratio": 0.60,
    "max_buy_channel_pos": 0.40, #0.25
    "pair": "XBTZAR",
    "rules": DEFAULT_RULES,
}


def ensure_config_shape(data):
    merged = deep_copy_json(DEFAULT_CONFIG)
    merged.update(data or {})
    if not isinstance(merged.get("rules"), list):
        merged["rules"] = deep_copy_json(DEFAULT_RULES)
    return merged
