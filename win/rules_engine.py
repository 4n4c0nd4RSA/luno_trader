import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config_defaults import CONFIG_FILE, ensure_config_shape

try:
    import luno_python.client as luno
except Exception:
    luno = None


CONFIG_PATH = Path(CONFIG_FILE)


@dataclass
class Candle:
    ts_ms: int
    open: float
    close: float
    high: float
    low: float
    volume: float


def load_config() -> Dict[str, Any]:
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            return ensure_config_shape(json.load(f))
    return ensure_config_shape({})


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_macd(series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = ema(series, 12)
    ema26 = ema(series, 26)
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            (df["High"] - df["Low"]).abs(),
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return atr


def compute_adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)


def build_middle_trend(segment: pd.DataFrame) -> np.ndarray:
    if len(segment) < 2:
        return segment["Close"].values.astype(float)

    x = np.arange(len(segment))
    y = segment["Close"].values.astype(float)
    coef = np.polyfit(x, y, 1)
    trend = np.poly1d(coef)(x)
    return trend.astype(float)


def build_lower_touch_line(segment: pd.DataFrame) -> Tuple[np.ndarray, int, Optional[int], float]:
    lows = segment["Low"].values.astype(float)
    n = len(lows)

    if n < 2:
        return lows.copy(), 0, None, 0.0

    anchor_idx = int(np.argmin(lows))
    anchor_y = float(lows[anchor_idx])

    best_line = None
    best_touch_idx = None
    best_slope = None

    for j in range(n):
        if j == anchor_idx:
            continue

        slope = (lows[j] - anchor_y) / (j - anchor_idx)
        line = anchor_y + slope * (np.arange(n) - anchor_idx)

        if np.all(line <= lows + 1e-9):
            if best_line is None or slope > best_slope:
                best_line = line
                best_touch_idx = j
                best_slope = slope

    if best_line is None:
        best_line = np.full(n, anchor_y, dtype=float)
        best_touch_idx = None
        best_slope = 0.0

    return best_line.astype(float), anchor_idx, best_touch_idx, float(best_slope)


def build_upper_touch_line(segment: pd.DataFrame) -> Tuple[np.ndarray, int, Optional[int], float]:
    highs = segment["High"].values.astype(float)
    n = len(highs)

    if n < 2:
        return highs.copy(), 0, None, 0.0

    anchor_idx = int(np.argmax(highs))
    anchor_y = float(highs[anchor_idx])

    best_line = None
    best_touch_idx = None
    best_slope = None

    for j in range(n):
        if j == anchor_idx:
            continue

        slope = (highs[j] - anchor_y) / (j - anchor_idx)
        line = anchor_y + slope * (np.arange(n) - anchor_idx)

        if np.all(line >= highs - 1e-9):
            if best_line is None or slope < best_slope:
                best_line = line
                best_touch_idx = j
                best_slope = slope

    if best_line is None:
        best_line = np.full(n, anchor_y, dtype=float)
        best_touch_idx = None
        best_slope = 0.0

    return best_line.astype(float), anchor_idx, best_touch_idx, float(best_slope)


def nanmean_two(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    out = np.full(len(a), np.nan, dtype=float)

    a_ok = ~np.isnan(a)
    b_ok = ~np.isnan(b)

    both = a_ok & b_ok
    only_a = a_ok & ~b_ok
    only_b = ~a_ok & b_ok

    out[both] = (a[both] + b[both]) / 2.0
    out[only_a] = a[only_a]
    out[only_b] = b[only_b]

    return out


def _safe_pct(a: float, b: float) -> float:
    if a == 0:
        return 0.0
    return ((b - a) / a) * 100.0


def bias_from_raw_score(score: float, cfg: Dict[str, Any]) -> str:
    if score >= float(cfg.get("raw_score_buy_bias", 10.0)):
        return "BUY"
    if score <= float(cfg.get("raw_score_sell_bias", -40.0)):
        return "SELL"
    return "HOLD"


def classify_segment(
    segment: pd.DataFrame,
    trend: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    upper_slope: float,
    lower_slope: float,
) -> Dict[str, Any]:
    if len(segment) < 2:
        return {
            "score": 0.0,
            "confidence": 0.0,
            "slope": 0.0,
            "pct_change": 0.0,
            "channel_pos": 0.5,
            "reasons": ["segment_too_small"],
        }

    x = np.arange(len(segment))
    y = segment["Close"].values.astype(float)
    coef = np.polyfit(x, y, 1)
    slope = float(coef[0])

    start_close = float(segment["Close"].iloc[0])
    end_close = float(segment["Close"].iloc[-1])
    pct_change = _safe_pct(start_close, end_close)

    last = segment.iloc[-1]
    prev = segment.iloc[-2]

    close_now = float(last["Close"])
    upper_now = float(upper[-1]) if len(upper) else close_now
    lower_now = float(lower[-1]) if len(lower) else close_now

    channel_width = max(upper_now - lower_now, 1e-9)
    channel_pos = (close_now - lower_now) / channel_width
    channel_pos = float(np.clip(channel_pos, 0.0, 1.0))

    score = 0.0
    reasons = []

    if last["MACD"] > last["MACD_SIGNAL"]:
        score += 50.0
        reasons.append("macd_above_signal")
    else:
        score -= 50.0
        reasons.append("macd_below_signal")

    if last["MACD_HIST"] > prev["MACD_HIST"]:
        score += 10.0
        reasons.append("macd_hist_improving")
    else:
        score -= 10.0
        reasons.append("macd_hist_weakening")

    if bool(last.get("MACD_CROSS_UP", False)):
        score += 40.0
        reasons.append("macd_cross_up")
    if bool(last.get("MACD_CROSS_DOWN", False)):
        score -= 40.0
        reasons.append("macd_cross_down")

    score = float(np.clip(score, -100.0, 100.0))
    confidence = float(min(abs(score), 100.0))

    return {
        "score": score,
        "confidence": confidence,
        "slope": slope,
        "pct_change": pct_change,
        "channel_pos": channel_pos,
        "reasons": reasons,
    }


def build_segment_signal_arrays(
    df: pd.DataFrame,
    window: int,
    cfg: Dict[str, Any],
    offset: int = 0,
    set_name: str = "SET1",
):
    trend_full = np.full(len(df), np.nan)
    upper_full = np.full(len(df), np.nan)
    lower_full = np.full(len(df), np.nan)
    statuses = []

    if len(df) < 2:
        return trend_full, upper_full, lower_full, statuses

    for start in range(offset, len(df), window):
        end = min(start + window, len(df))
        segment = df.iloc[start:end]

        if len(segment) < 2:
            continue

        trend = build_middle_trend(segment)
        upper, upper_anchor_idx, upper_touch_idx, upper_slope = build_upper_touch_line(segment)
        lower, lower_anchor_idx, lower_touch_idx, lower_slope = build_lower_touch_line(segment)

        analysis = classify_segment(
            segment=segment,
            trend=trend,
            upper=upper,
            lower=lower,
            upper_slope=upper_slope,
            lower_slope=lower_slope,
        )

        trend_full[start:end] = trend
        upper_full[start:end] = upper
        lower_full[start:end] = lower

        seg_start = segment.index[0]
        seg_end = segment.index[-1]
        last_row = segment.iloc[-1]

        action = "BUY" if bool(last_row.get("MACD_CROSS_UP", False)) else (
            "SELL" if bool(last_row.get("MACD_CROSS_DOWN", False)) else "HOLD"
        )
        raw_bias = bias_from_raw_score(float(analysis["score"]), cfg)

        upper_anchor_time = segment.index[upper_anchor_idx]
        upper_touch_time = segment.index[upper_touch_idx] if upper_touch_idx is not None else None
        lower_anchor_time = segment.index[lower_anchor_idx]
        lower_touch_time = segment.index[lower_touch_idx] if lower_touch_idx is not None else None

        status = {
            "set_name": set_name,
            "offset_candles": offset,
            "start_idx": start,
            "end_idx": end - 1,
            "start": seg_start,
            "end": seg_end,
            "action": action,
            "raw_bias": raw_bias,
            "score": float(analysis["score"]),
            "confidence": analysis["confidence"],
            "slope": analysis["slope"],
            "pct_change": analysis["pct_change"],
            "channel_pos": analysis["channel_pos"],
            "reasons": analysis["reasons"],
            "upper_anchor": upper_anchor_time,
            "upper_touch": upper_touch_time,
            "upper_slope": upper_slope,
            "lower_anchor": lower_anchor_time,
            "lower_touch": lower_touch_time,
            "lower_slope": lower_slope,
            "macd": float(last_row["MACD"]),
            "macd_signal": float(last_row["MACD_SIGNAL"]),
            "macd_hist": float(last_row["MACD_HIST"]),
            "macd_cross_up": bool(last_row["MACD_CROSS_UP"]),
            "macd_cross_down": bool(last_row["MACD_CROSS_DOWN"]),
        }
        statuses.append(status)

    return trend_full, upper_full, lower_full, statuses


def summarize_latest_status(statuses: List[dict]) -> str:
    if not statuses:
        return "NO SIGNAL"

    latest = max(statuses, key=lambda s: s["end"])
    return (
        f"LATEST SEGMENT SIGNAL: {latest['action']} | "
        f"{latest['set_name']} | "
        f"offset={latest['offset_candles']} candles | "
        f"{latest['start']} -> {latest['end']} | "
        f"score={latest['score']:.2f} | "
        f"confidence={latest['confidence']:.2f} | "
        f"macd={latest['macd']:.6f} | "
        f"macd_signal={latest['macd_signal']:.6f} | "
        f"macd_hist={latest['macd_hist']:.6f} | "
        f"macd_cross_up={latest['macd_cross_up']} | "
        f"macd_cross_down={latest['macd_cross_down']}"
    )


def _compute_current_market_score(
    df: pd.DataFrame,
    avg_upper: np.ndarray,
    avg_lower: np.ndarray,
) -> Dict[str, Any]:
    if df.empty:
        return {"score": 0.0, "channel_pos": 0.5, "reasons": ["no_data"]}

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last

    close_now = float(last["Close"])
    upper_now = float(avg_upper[-1]) if len(avg_upper) and not np.isnan(avg_upper[-1]) else close_now
    lower_now = float(avg_lower[-1]) if len(avg_lower) and not np.isnan(avg_lower[-1]) else close_now
    width = max(upper_now - lower_now, 1e-9)
    channel_pos = float(np.clip((close_now - lower_now) / width, 0.0, 1.0))

    score = 0.0
    reasons = []

    if last["MACD"] > last["MACD_SIGNAL"]:
        score += 50.0
        reasons.append("macd_above_signal")
    else:
        score -= 50.0
        reasons.append("macd_below_signal")

    if last["MACD_HIST"] > prev["MACD_HIST"]:
        score += 10.0
        reasons.append("macd_hist_improving")
    else:
        score -= 10.0
        reasons.append("macd_hist_weakening")

    if bool(last.get("MACD_CROSS_UP", False)):
        score += 40.0
        reasons.append("macd_cross_up")

    if bool(last.get("MACD_CROSS_DOWN", False)):
        score -= 40.0
        reasons.append("macd_cross_down")

    score = float(np.clip(score, -100.0, 100.0))

    return {"score": score, "channel_pos": channel_pos, "reasons": reasons}


def compute_current_market_signal(
    df: pd.DataFrame,
    avg_upper: np.ndarray,
    avg_lower: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    if df.empty:
        return {
            "action": "HOLD",
            "score": 0.0,
            "channel_pos": 0.5,
            "live_profit_pct": None,
            "reasons": ["no_data"],
        }

    current_entry_price = float(cfg.get("current_entry_price", 0.0))
    current_peak_profit_pct = float(cfg.get("current_peak_profit_pct", 0.0))
    buy_fee_rate = float(cfg.get("trade_buy_fee_rate", 0.005))
    sell_fee_rate = float(cfg.get("trade_sell_fee_rate", 0.003))

    current_base = _compute_current_market_score(df, avg_upper, avg_lower)
    last = df.iloc[-1]

    sell_red_ok = bool(last.get("SELL_RED_OK", False))
    red_body_diff = float(last.get("RED_BODY_DIFF", 0.0))
    macd_cross_up = bool(last["MACD_CROSS_UP"])
    macd_cross_down = bool(last["MACD_CROSS_DOWN"])

    live_profit_pct = None
    max_profit_signal_pct = 0.0
    if current_entry_price > 0:
        buy_cost_per_unit = current_entry_price * (1.0 + buy_fee_rate)
        sell_value_per_unit = float(last["Close"]) * (1.0 - sell_fee_rate)
        live_profit_pct = ((sell_value_per_unit / buy_cost_per_unit) - 1.0) * 100.0
        current_peak_profit_pct = max(current_peak_profit_pct, live_profit_pct)
        if current_peak_profit_pct > 0:
            max_profit_signal_pct = (live_profit_pct / current_peak_profit_pct) * 100.0

    signal_values = _build_signal_values(last, live_profit_pct or 0.0, max_profit_signal_pct)
    rule_state = _evaluate_active_rules(signal_values, cfg)
    buy_hits = rule_state["buy_hits"]
    sell_hits = rule_state["sell_hits"]
    in_position = current_entry_price > 0
    if in_position and sell_hits:
        action = "SELL"
    elif (not in_position) and buy_hits:
        action = "BUY"
    else:
        action = "HOLD"

    reasons = list(current_base["reasons"])
    reasons.extend([f"matched_buy_rule={name}" for name in buy_hits])
    reasons.extend([f"matched_sell_rule={name}" for name in sell_hits])
    reasons.append(f"macd={float(last['MACD']):.6f}")
    reasons.append(f"macd_signal={float(last['MACD_SIGNAL']):.6f}")
    reasons.append(f"macd_hist={float(last['MACD_HIST']):.6f}")
    reasons.append(f"macd_cross_up={macd_cross_up}")
    reasons.append(f"macd_cross_down={macd_cross_down}")
    reasons.append(f"hist_improving={bool(last['MACD_HIST_IMPROVING'])}")
    reasons.append(f"hist_weakening={bool(last['MACD_HIST_WEAKENING'])}")
    reasons.append(f"red_candle={bool(last.get('RED_CANDLE', False))}")
    reasons.append(f"red_body_diff={red_body_diff:.2f}")
    reasons.append(f"sell_red_ok={sell_red_ok}")
    reasons.append(f"close_gt_ema21={bool(last['Close'] > last['EMA21'])}")
    reasons.append(f"ema21_gt_ema50={bool(last['EMA21'] > last['EMA50'])}")
    reasons.append(f"adx14={float(last['ADX14']):.2f}")
    reasons.append(f"channel_pos={float(current_base['channel_pos']):.2f}")
    if live_profit_pct is not None:
        reasons.append(f"live_profit_pct={live_profit_pct:.4f}")

    return {
        "action": action,
        "score": current_base["score"],
        "channel_pos": current_base["channel_pos"],
        "live_profit_pct": live_profit_pct,
        "max_profit_pct": max_profit_signal_pct,
        "matched_rules": buy_hits if action == "BUY" else sell_hits if action == "SELL" else [],
        "reasons": reasons,
    }


def _resolve_config_value(cfg: Dict[str, Any], key: Any) -> Any:
    if isinstance(key, str):
        if key in cfg:
            return cfg[key]
        lowered = key.lower()
        if lowered in cfg:
            return cfg[lowered]
    return key


def _resolve_left_value(
    check: Dict[str, Any],
    signal_values: Dict[str, Any],
    rule_results: Dict[str, bool],
) -> Any:
    left_type = check.get("left_type", "signal")
    if left_type == "rule":
        return bool(rule_results.get(str(check.get("left_rule", "")), False))
    return signal_values.get(check.get("left_signal"))


def _resolve_right_value(
    check: Dict[str, Any],
    signal_values: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Any:
    right_type = check.get("right_type")
    right_value = check.get("right_value")

    if right_type == "signal":
        return signal_values.get(right_value)
    if right_type == "number":
        return float(right_value)
    if right_type == "boolean":
        return bool(right_value)
    if right_type == "config":
        return _resolve_config_value(cfg, right_value)
    return right_value


def _compare(left: Any, operator: str, right: Any) -> bool:
    if operator == ">":
        return left > right
    if operator == "<":
        return left < right
    if operator == ">=":
        return left >= right
    if operator == "<=":
        return left <= right
    if operator == "==":
        return left == right
    if operator == "!=":
        return left != right
    raise ValueError(f"Unsupported operator: {operator}")


def _evaluate_rule_checks(
    rule: Dict[str, Any],
    signal_values: Dict[str, Any],
    rule_results: Dict[str, bool],
    cfg: Dict[str, Any],
) -> bool:
    return _evaluate_rule_block(rule, signal_values, rule_results, cfg)


def _evaluate_rule_block(
    block: Dict[str, Any],
    signal_values: Dict[str, Any],
    rule_results: Dict[str, bool],
    cfg: Dict[str, Any],
) -> bool:
    checks = block.get("checks", []) or []
    if not checks:
        return False

    mode = str(block.get("match", "all")).lower()
    results: List[bool] = []

    for check in checks:
        if not isinstance(check, dict):
            results.append(False)
            continue

        if "checks" in check:
            results.append(_evaluate_rule_block(check, signal_values, rule_results, cfg))
            continue

        left = _resolve_left_value(check, signal_values, rule_results)
        right = _resolve_right_value(check, signal_values, cfg)
        op = str(check.get("operator", "=="))
        results.append(_compare(left, op, right))

    if mode == "any":
        return any(results)
    return all(results)


def _get_active_rules(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    rules = [r for r in cfg.get("rules", []) if bool(r.get("active", False))]
    return sorted(rules, key=lambda r: (int(r.get("salience", 999999)), str(r.get("name", ""))))


def _build_signal_values(prev: pd.Series, live_profit_pct: float, max_profit_signal_pct: float) -> Dict[str, Any]:
    return {
        "EMA9": float(prev.get("EMA9", 0.0)),
        "EMA21": float(prev.get("EMA21", 0.0)),
        "EMA50": float(prev.get("EMA50", 0.0)),
        "EMA200": float(prev.get("EMA200", 0.0)),
        "RSI14": float(prev.get("RSI14", 0.0)),
        "MACD": float(prev.get("MACD", 0.0)),
        "MACD_SIGNAL": float(prev.get("MACD_SIGNAL", 0.0)),
        "MACD_HIST": float(prev.get("MACD_HIST", 0.0)),
        "CHANNEL_POS": float(prev.get("CHANNEL_POS", 0.0)),
        "ATR14": float(prev.get("ATR14", 0.0)),
        "ATR_PCT": float(prev.get("ATR_PCT", 0.0)),
        "ADX14": float(prev.get("ADX14", 0.0)),
        "+DI14": float(prev.get("+DI14", 0.0)),
        "-DI14": float(prev.get("-DI14", 0.0)),
        "VOL_SMA20": float(prev.get("VOL_SMA20", 0.0)),
        "VOL_RATIO": float(prev.get("VOL_RATIO", 0.0)),
        "RET_1": float(prev.get("RET_1", 0.0)),
        "RET_6": float(prev.get("RET_6", 0.0)),
        "RET_24": float(prev.get("RET_24", 0.0)),
        "PRICE_DELTA": float(prev.get("PRICE_DELTA", 0.0)),
        "PRICE_DELTA_PCT": float(prev.get("PRICE_DELTA_PCT", 0.0)),
        "PRICE_ACCEL": float(prev.get("PRICE_ACCEL", 0.0)),
        "PRICE_ACCEL_PCT": float(prev.get("PRICE_ACCEL_PCT", 0.0)),
        "ABS_PRICE_DELTA_PCT": float(prev.get("ABS_PRICE_DELTA_PCT", 0.0)),
        "ABS_PRICE_ACCEL_PCT": float(prev.get("ABS_PRICE_ACCEL_PCT", 0.0)),
        "RED_CANDLE": bool(prev.get("RED_CANDLE", False)),
        "RED_BODY_DIFF": float(prev.get("RED_BODY_DIFF", 0.0)),
        "SELL_RED_OK": bool(prev.get("SELL_RED_OK", False)),
        "MACD_CROSS_UP": bool(prev.get("MACD_CROSS_UP", False)),
        "MACD_CROSS_DOWN": bool(prev.get("MACD_CROSS_DOWN", False)),
        "MACD_HIST_IMPROVING": bool(prev.get("MACD_HIST_IMPROVING", False)),
        "MACD_HIST_WEAKENING": bool(prev.get("MACD_HIST_WEAKENING", False)),
        "LIVE_PROFIT_PCT": float(live_profit_pct),
        "MAX_PROFIT_PCT": float(max_profit_signal_pct),
    }


def _evaluate_active_rules(signal_values: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    rule_results: Dict[str, bool] = {}
    buy_hits: List[str] = []
    sell_hits: List[str] = []

    for rule in _get_active_rules(cfg):
        rule_name = str(rule.get("name", ""))
        result = _evaluate_rule_checks(rule, signal_values, rule_results, cfg)
        rule_results[rule_name] = result
        if not result:
            continue
        action = str(rule.get("action", "")).upper()
        if action == "BUY":
            buy_hits.append(rule_name)
        elif action == "SELL":
            sell_hits.append(rule_name)

    return {"rule_results": rule_results, "buy_hits": buy_hits, "sell_hits": sell_hits}

def _get_rule_reason(rule_name: str, cfg: Dict[str, Any]) -> str:
    for rule in cfg.get("rules", []) or []:
        if str(rule.get("name", "")) == str(rule_name):
            custom_reason = str(rule.get("reason", "")).strip()
            return custom_reason or str(rule_name)
    return str(rule_name)


def build_macd_signals(df: pd.DataFrame) -> List[dict]:
    cfg = load_config()

    signals: List[dict] = []

    peak_profit_pct = 0.0
    last_buy_price: Optional[float] = None

    for i in range(2, len(df)):
        prev = df.iloc[i - 1]

        exec_time = df.index[i]
        exec_price = float(df["Open"].iloc[i])
        if exec_price <= 0:
            continue

        prev_red_body_diff = float(prev.get("RED_BODY_DIFF", 0.0))
        channel_pos = float(prev.get("CHANNEL_POS", 0.0))

        in_position = last_buy_price is not None

        if in_position:
            profit = exec_price - last_buy_price
            live_profit_pct = (profit / last_buy_price) * 100.0 if last_buy_price > 0 else 0.0
            peak_profit_pct = max(peak_profit_pct, live_profit_pct)
            if peak_profit_pct > 0:
                max_profit_signal_pct = (live_profit_pct / peak_profit_pct) * 100.0
            else:
                max_profit_signal_pct = 0.0
        else:
            profit = 0.0
            live_profit_pct = 0.0
            max_profit_signal_pct = 0.0

        signal_values = _build_signal_values(prev, live_profit_pct, max_profit_signal_pct)
        rule_state = _evaluate_active_rules(signal_values, cfg)
        buy_hits = rule_state["buy_hits"]
        sell_hits = rule_state["sell_hits"]

        if (not in_position) and buy_hits:
            signals.append({
                "exec_idx": i,
                "exec_time": exec_time,
                "action": "BUY",
                "macd": float(prev["MACD"]),
                "macd_signal": float(prev["MACD_SIGNAL"]),
                "macd_hist": float(prev["MACD_HIST"]),
                "channel_pos": channel_pos,
                "red_body_diff": prev_red_body_diff,
                "live_profit_pct": 0.0,
                "max_profit_pct": 0.0,
                "peak_profit_pct": 0.0,
                "profit": 0.0,
                "reason": _get_rule_reason(buy_hits[0], cfg),
                "matched_rules": buy_hits,
            })
            last_buy_price = exec_price
            peak_profit_pct = 0.0

        elif in_position and sell_hits:
            signals.append({
                "exec_idx": i,
                "exec_time": exec_time,
                "action": "SELL",
                "macd": float(prev["MACD"]),
                "macd_signal": float(prev["MACD_SIGNAL"]),
                "macd_hist": float(prev["MACD_HIST"]),
                "channel_pos": channel_pos,
                "red_body_diff": prev_red_body_diff,
                "live_profit_pct": live_profit_pct,
                "max_profit_pct": max_profit_signal_pct,
                "peak_profit_pct": peak_profit_pct,
                "profit": profit,
                "reason": _get_rule_reason(sell_hits[0], cfg),
                "matched_rules": sell_hits,
            })
            peak_profit_pct = 0.0
            last_buy_price = None

    return signals


def run_mock_backtrade(
    df: pd.DataFrame,
    macd_signals: List[dict],
    cfg: Dict[str, Any],
) -> dict:
    start_zar = float(cfg.get("backtest_start_zar", 10000.0))
    buy_fee_rate = float(cfg.get("trade_buy_fee_rate", 0.005))
    sell_fee_rate = float(cfg.get("trade_sell_fee_rate", 0.003))

    if df.empty:
        return {
            "start_zar": start_zar,
            "final_zar": start_zar,
            "final_btc": 0.0,
            "mark_to_market_zar": start_zar,
            "trades": [],
            "history": {"time": [], "zar": [], "btc": [], "bank": []},
        }

    zar_balance = float(start_zar)
    btc_balance = 0.0
    position = "ZAR"
    trades = []

    zar_history = [zar_balance]
    btc_history = [btc_balance]
    bank_history = [zar_balance]
    time_index = [df.index[0]]

    signal_map: Dict[int, dict] = {
        int(s["exec_idx"]): s
        for s in macd_signals
        if int(s["exec_idx"]) < len(df)
    }

    for i in range(1, len(df)):
        exec_time = df.index[i]
        exec_price = float(df["Open"].iloc[i])

        if exec_price <= 0:
            last_close = float(df["Close"].iloc[i])
            bank_value = zar_balance + (btc_balance * last_close)
            zar_history.append(zar_balance)
            btc_history.append(btc_balance)
            bank_history.append(bank_value)
            time_index.append(exec_time)
            continue

        signal = signal_map.get(i)
        if signal:
            action = str(signal["action"]).upper()

            if action == "BUY" and position == "ZAR" and zar_balance > 0:
                prev_zar = zar_balance
                fee_zar = prev_zar * buy_fee_rate
                net_zar_to_spend = prev_zar - fee_zar
                btc_bought = net_zar_to_spend / exec_price

                btc_balance = btc_bought
                zar_balance = 0.0
                position = "BTC"

                trades.append({
                    "type": "BUY",
                    "exec_time": exec_time,
                    "exec_price": exec_price,
                    "macd": float(signal["macd"]),
                    "macd_signal": float(signal["macd_signal"]),
                    "macd_hist": float(signal["macd_hist"]),
                    "fee_zar": fee_zar,
                    "btc_after": btc_balance,
                    "reason": signal.get("reason", ""),
                })

            elif action == "SELL" and position == "BTC" and btc_balance > 0:
                prev_btc = btc_balance
                gross_zar = prev_btc * exec_price
                fee_zar = gross_zar * sell_fee_rate
                zar_received = gross_zar - fee_zar

                zar_balance = zar_received
                btc_balance = 0.0
                position = "ZAR"

                trades.append({
                    "type": "SELL",
                    "exec_time": exec_time,
                    "exec_price": exec_price,
                    "macd": float(signal["macd"]),
                    "macd_signal": float(signal["macd_signal"]),
                    "macd_hist": float(signal["macd_hist"]),
                    "red_body_diff": float(signal.get("red_body_diff", 0.0)),
                    "fee_zar": fee_zar,
                    "zar_after": zar_balance,
                    "reason": signal.get("reason", ""),
                })

        last_close = float(df["Close"].iloc[i])
        bank_value = zar_balance + (btc_balance * last_close)

        zar_history.append(zar_balance)
        btc_history.append(btc_balance)
        bank_history.append(bank_value)
        time_index.append(exec_time)

    last_close = float(df["Close"].iloc[-1])
    mark_to_market_zar = zar_balance + (btc_balance * last_close * (1.0 - sell_fee_rate))
    pnl_zar = mark_to_market_zar - start_zar
    pnl_pct = (pnl_zar / start_zar) * 100.0 if start_zar else 0.0
    total_fees_zar = sum(float(t.get("fee_zar", 0.0)) for t in trades)
    price_change_pct = _safe_pct(float(df["Close"].iloc[0]), float(df["Close"].iloc[-1]))

    return {
        "start_zar": start_zar,
        "final_zar": zar_balance,
        "final_btc": btc_balance,
        "mark_to_market_zar": mark_to_market_zar,
        "last_close": last_close,
        "pnl_zar": pnl_zar,
        "pnl_pct": pnl_pct,
        "total_fees_zar": total_fees_zar,
        "price_change_pct": price_change_pct,
        "position": position,
        "trades": trades,
        "history": {
            "time": time_index,
            "zar": zar_history,
            "btc": btc_history,
            "bank": bank_history,
        },
    }


def add_indicators(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()

    candle_seconds = int(cfg.get("candle_seconds", 3600))
    trend_window_hours = int(cfg.get("trend_window_hours", 48))
    sell_red_body_min_diff = float(cfg.get("sell_red_body_min_diff", 10000.0))

    out["EMA9"] = ema(out["Close"], 9)
    out["EMA21"] = ema(out["Close"], 21)
    out["EMA50"] = ema(out["Close"], 50)
    out["EMA200"] = ema(out["Close"], 200)

    out["RSI14"] = compute_rsi(out["Close"], 14)

    macd_line, macd_signal, macd_hist = compute_macd(out["Close"])
    out["MACD"] = macd_line
    out["MACD_SIGNAL"] = macd_signal
    out["MACD_HIST"] = macd_hist

    window = max(2, int((trend_window_hours * 3600) / candle_seconds))
    upper1 = np.full(len(df), np.nan)
    lower1 = np.full(len(df), np.nan)
    upper2 = np.full(len(df), np.nan)
    lower2 = np.full(len(df), np.nan)
    for start in range(0, len(df), window):
        end = min(start + window, len(df))
        segment = df.iloc[start:end]
        upper, _, _, _ = build_upper_touch_line(segment)
        lower, _, _, _ = build_lower_touch_line(segment)
        upper1[start:end] = upper
        lower1[start:end] = lower
        upper2[start:end] = upper
        lower2[start:end] = lower

    avg_upper = nanmean_two(upper1, upper2)
    avg_lower = nanmean_two(lower1, lower2)

    channel_pos_price = (out["Close"] - avg_lower) / (avg_upper - avg_lower)
    channel_pos_price = channel_pos_price.clip(0, 1)
    out["CHANNEL_POS"] = channel_pos_price

    out["ATR14"] = compute_atr(out, 14)
    out["ATR_PCT"] = np.where(out["Close"] > 0, out["ATR14"] / out["Close"] * 100.0, 0.0)

    adx, plus_di, minus_di = compute_adx(out, 14)
    out["ADX14"] = adx
    out["+DI14"] = plus_di
    out["-DI14"] = minus_di

    out["VOL_SMA20"] = out["Volume"].rolling(20, min_periods=1).mean()
    out["VOL_RATIO"] = np.where(out["VOL_SMA20"] > 0, out["Volume"] / out["VOL_SMA20"], 1.0)

    out["RET_1"] = out["Close"].pct_change(1).fillna(0.0)
    out["RET_6"] = out["Close"].pct_change(6).fillna(0.0)
    out["RET_24"] = out["Close"].pct_change(24).fillna(0.0)

    out["PRICE_DELTA"] = out["Close"].diff().fillna(0.0)
    out["PRICE_DELTA_PCT"] = out["Close"].pct_change().fillna(0.0) * 100.0
    out["PRICE_ACCEL"] = out["PRICE_DELTA"].diff().fillna(0.0)
    out["PRICE_ACCEL_PCT"] = out["PRICE_DELTA_PCT"].diff().fillna(0.0)
    out["ABS_PRICE_DELTA_PCT"] = out["PRICE_DELTA_PCT"].abs()
    out["ABS_PRICE_ACCEL_PCT"] = out["PRICE_ACCEL_PCT"].abs()

    out["RED_CANDLE"] = out["Close"] < out["Open"]
    out["RED_BODY_DIFF"] = out["Open"] - out["Close"]
    out["SELL_RED_OK"] = out["RED_BODY_DIFF"] > sell_red_body_min_diff

    out["MACD_CROSS_UP"] = (
        (out["MACD"] > out["MACD_SIGNAL"]) &
        (out["MACD"].shift(1) <= out["MACD_SIGNAL"].shift(1))
    ).fillna(False)

    out["MACD_CROSS_DOWN"] = (
        (out["MACD"] < out["MACD_SIGNAL"]) &
        (out["MACD"].shift(1) >= out["MACD_SIGNAL"].shift(1))
    ).fillna(False)

    out["MACD_HIST_IMPROVING"] = (out["MACD_HIST"] > out["MACD_HIST"].shift(1)).fillna(False)
    out["MACD_HIST_WEAKENING"] = (out["MACD_HIST"] < out["MACD_HIST"].shift(1)).fillna(False)

    return out


def candles_to_dataframe(candles: List[Candle]) -> pd.DataFrame:
    if not candles:
        raise RuntimeError("No candles returned from Luno")

    df = pd.DataFrame(
        [
            {
                "Date": pd.to_datetime(c.ts_ms, unit="ms"),
                "Open": c.open,
                "High": c.high,
                "Low": c.low,
                "Close": c.close,
                "Volume": c.volume,
            }
            for c in candles
        ]
    )

    df.set_index("Date", inplace=True)
    return df


def get_candles(cfg: Dict[str, Any]) -> List[Candle]:
    if luno is None:
        raise RuntimeError("luno_python is not installed.")

    api_key = cfg.get("api_key_id") or None
    api_secret = cfg.get("api_key_secret") or None

    if not api_key or not api_secret:
        raise RuntimeError("Missing api_key_id / api_key_secret in config.")

    pair = cfg.get("pair", "XBTZAR")
    history_hours = int(cfg.get("history_hours", 1000))
    candle_seconds = int(cfg.get("candle_seconds", 3600))

    now_ms = int(time.time() * 1000)
    since_ms = now_ms - (history_hours * 60 * 60 * 1000)

    client = luno.Client(api_key_id=api_key, api_key_secret=api_secret)
    res = client.get_candles(pair=pair, since=since_ms, duration=candle_seconds)
    raw = res.get("candles", []) or []

    candles: List[Candle] = []
    for c in raw:
        candles.append(
            Candle(
                ts_ms=int(c["timestamp"]),
                open=float(c["open"]),
                close=float(c["close"]),
                high=float(c["high"]),
                low=float(c["low"]),
                volume=float(c["volume"]),
            )
        )

    candles.sort(key=lambda x: x.ts_ms)
    return candles


def build_analysis(df: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = cfg or load_config()

    if df.empty:
        raise RuntimeError("No candle data available")

    candles_per_segment = max(2, int((int(cfg.get("trend_window_hours", 48)) * 3600) / int(cfg.get("candle_seconds", 3600))))
    half_offset = max(1, candles_per_segment // 2)

    trend1, upper1, lower1, statuses_1 = build_segment_signal_arrays(
        df=df,
        window=candles_per_segment,
        cfg=cfg,
        offset=0,
        set_name="SET1",
    )

    trend2, upper2, lower2, statuses_2 = build_segment_signal_arrays(
        df=df,
        window=candles_per_segment,
        cfg=cfg,
        offset=half_offset,
        set_name="SET2",
    )

    avg_upper = nanmean_two(upper1, upper2)
    avg_lower = nanmean_two(lower1, lower2)

    statuses = statuses_1 + statuses_2
    latest_status = summarize_latest_status(statuses)

    macd_signals = build_macd_signals(df)

    current_signal = compute_current_market_signal(
        df=df,
        avg_upper=avg_upper,
        avg_lower=avg_lower,
        cfg=cfg,
    )

    backtest = run_mock_backtrade(df=df, macd_signals=macd_signals, cfg=cfg)

    hist = backtest.get("history", {})
    bank_series = pd.Series(hist.get("bank", []), index=hist.get("time", []), dtype=float)
    bank_series = bank_series.reindex(df.index)

    return {
        "df": df,
        "avg_upper": avg_upper,
        "avg_lower": avg_lower,
        "statuses": statuses,
        "latest_status": latest_status,
        "macd_signals": macd_signals,
        "current_signal": current_signal,
        "backtest": backtest,
        "bank_series": bank_series,
    }


def build_analysis_from_luno(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = cfg or load_config()
    candles = get_candles(cfg)
    df = candles_to_dataframe(candles)
    df = add_indicators(df, cfg)
    return build_analysis(df, cfg)
