import os
import time
import logging
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import luno_python.client as luno
from rules_engine import build_macd_signals

# =========
# Settings
# =========
PAIR = os.getenv("PAIR", "XBTZAR")
HISTORY_HOURS = int(os.getenv("HISTORY_HOURS", random.randint(1000, 128000)))
HISTORY_HOURS = 1000
print(HISTORY_HOURS)

CANDLE_SECONDS = int(os.getenv("CANDLE_SECONDS", "3600"))
TREND_WINDOW_HOURS = int(os.getenv("TREND_WINDOW_HOURS", "48"))
BACKTEST_START_ZAR = float(os.getenv("BACKTEST_START_ZAR", "10000"))

TRADE_BUY_FEE_RATE = float(os.getenv("TRADE_BUY_FEE_RATE", "0.005"))
TRADE_SELL_FEE_RATE = float(os.getenv("TRADE_SELL_FEE_RATE", "0.003"))

CURRENT_ENTRY_PRICE = float(os.getenv("CURRENT_ENTRY_PRICE", "0"))

RAW_SCORE_BUY_BIAS = float(os.getenv("RAW_SCORE_BUY_BIAS", "10"))
RAW_SCORE_SELL_BIAS = float(os.getenv("RAW_SCORE_SELL_BIAS", "-40"))

SELL_RED_BODY_MIN_DIFF = float(os.getenv("SELL_RED_BODY_MIN_DIFF", "10000"))

ENTRY_MIN_ADX = float(os.getenv("ENTRY_MIN_ADX", "20"))
ENTRY_MIN_HIST = float(os.getenv("ENTRY_MIN_HIST", "0"))
EARLY_FAIL_MAX_PROFIT_PCT = float(os.getenv("EARLY_FAIL_MAX_PROFIT_PCT", "1.5"))
MIN_BEARISH_EXIT_PROFIT_PCT = float(os.getenv("MIN_BEARISH_EXIT_PROFIT_PCT", "2.0"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "-2.5"))
MIN_TRAIL_PROFIT_PCT = float(os.getenv("MIN_TRAIL_PROFIT_PCT", "1.0"))
TRAIL_GIVEBACK_RATIO = float(os.getenv("TRAIL_GIVEBACK_RATIO", "0.60"))
MAX_BUY_CHANNEL_POS = float(os.getenv("MAX_BUY_CHANNEL_POS", "0.2"))

# =========
# Logging
# =========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("luno_candle_graph.log"),
        logging.StreamHandler()
    ],
)

# =========
# Luno auth
# =========
LUNO_API_KEY = os.getenv("LUNO_API_KEY_ID")
LUNO_API_SECRET = os.getenv("LUNO_API_KEY_SECRET")

if not LUNO_API_KEY or not LUNO_API_SECRET:
    raise RuntimeError("Missing LUNO_API_KEY_ID / LUNO_API_KEY_SECRET env vars")

luno_client = luno.Client(
    api_key_id=LUNO_API_KEY,
    api_key_secret=LUNO_API_SECRET
)

peak_profit_pct = 0.0
last_buy_price = None

# =========
# Data model
# =========
@dataclass
class Candle:
    ts_ms: int
    open: float
    close: float
    high: float
    low: float
    volume: float


# =========
# Luno helpers
# =========
def get_candles(
    pair: str = PAIR,
    history_hours: int = HISTORY_HOURS,
    candle_seconds: int = CANDLE_SECONDS
) -> List[Candle]:
    now_ms = int(time.time() * 1000)
    since_ms = now_ms - (history_hours * 60 * 60 * 1000)

    res = luno_client.get_candles(pair=pair, since=since_ms, duration=candle_seconds)
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


# =========
# Indicator helpers
# =========
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


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["EMA9"] = ema(out["Close"], 9)
    out["EMA21"] = ema(out["Close"], 21)
    out["EMA50"] = ema(out["Close"], 50)
    out["EMA200"] = ema(out["Close"], 200)

    out["RSI14"] = compute_rsi(out["Close"], 14)

    macd_line, macd_signal, macd_hist = compute_macd(out["Close"])
    out["MACD"] = macd_line
    out["MACD_SIGNAL"] = macd_signal
    out["MACD_HIST"] = macd_hist

    window = max(2, int((TREND_WINDOW_HOURS * 3600) / CANDLE_SECONDS))
    upper1 = np.full(len(df), np.nan)
    lower1 = np.full(len(df), np.nan)
    upper2 = np.full(len(df), np.nan)
    lower2 = np.full(len(df), np.nan)
    for start in range(0, len(df), window):
        end = min(start + window, len(df))
        segment = df.iloc[start:end]
        upper, upper_anchor_idx, upper_touch_idx, upper_slope = build_upper_touch_line(segment)
        lower, lower_anchor_idx, lower_touch_idx, lower_slope = build_lower_touch_line(segment)
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
    out["SELL_RED_OK"] = out["RED_BODY_DIFF"] > SELL_RED_BODY_MIN_DIFF

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


# =========
# Trend / channel helpers
# =========
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


def bias_from_raw_score(score: float) -> str:
    if score >= RAW_SCORE_BUY_BIAS:
        return "BUY"
    if score <= RAW_SCORE_SELL_BIAS:
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
        raw_bias = bias_from_raw_score(float(analysis["score"]))

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


# =========
# Current market signal
# =========
def _compute_current_market_score(
    df: pd.DataFrame,
    avg_upper: np.ndarray,
    avg_lower: np.ndarray,
) -> Dict[str, Any]:
    if df.empty:
        return {
            "score": 0.0,
            "channel_pos": 0.5,
            "reasons": ["no_data"],
        }

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

    return {
        "score": score,
        "channel_pos": channel_pos,
        "reasons": reasons,
    }


def compute_current_market_signal(
    df: pd.DataFrame,
    avg_upper: np.ndarray,
    avg_lower: np.ndarray,
    current_entry_price: float = CURRENT_ENTRY_PRICE,
    buy_fee_rate: float = TRADE_BUY_FEE_RATE,
    sell_fee_rate: float = TRADE_SELL_FEE_RATE,
) -> Dict[str, Any]:
    if df.empty:
        return {
            "action": "HOLD",
            "score": 0.0,
            "channel_pos": 0.5,
            "live_profit_pct": None,
            "reasons": ["no_data"],
        }

    current_base = _compute_current_market_score(df, avg_upper, avg_lower)
    last = df.iloc[-1]

    sell_red_ok = bool(last.get("SELL_RED_OK", False))
    red_body_diff = float(last.get("RED_BODY_DIFF", 0.0))
    macd_cross_up = bool(last["MACD_CROSS_UP"])
    macd_cross_down = bool(last["MACD_CROSS_DOWN"])

    buy_ready = bool(
        macd_cross_up and
        last["MACD"] > last["MACD_SIGNAL"] and
        last["MACD_HIST_IMPROVING"] and
        last["MACD_HIST"] > ENTRY_MIN_HIST and
        last["MACD"] > 0 and
        last["Close"] > last["EMA21"] and
        last["EMA21"] > last["EMA50"] and
        last["ADX14"] >= ENTRY_MIN_ADX and
        current_base["channel_pos"] < MAX_BUY_CHANNEL_POS
    )

    sell_ready = bool(
        macd_cross_down and
        (sell_red_ok or bool(last["MACD_HIST_WEAKENING"]))
    )

    live_profit_pct = None
    if current_entry_price > 0:
        buy_cost_per_unit = current_entry_price * (1.0 + buy_fee_rate)
        sell_value_per_unit = float(last["Close"]) * (1.0 - sell_fee_rate)
        live_profit_pct = ((sell_value_per_unit / buy_cost_per_unit) - 1.0) * 100.0

    action = "BUY" if buy_ready else ("SELL" if sell_ready else "HOLD")

    reasons = list(current_base["reasons"])
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
        "reasons": reasons,
    }


# =========
# Signal builder
# =========
def build_macd_signals_old(df: pd.DataFrame) -> List[dict]:
    global peak_profit_pct
    global last_buy_price

    signals: List[dict] = []

    peak_profit_pct = 0.0
    last_buy_price = None

    for i in range(2, len(df)):
        prev = df.iloc[i - 1]
        prev2 = df.iloc[i - 2]

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

        bullish_entry = bool(
            # prev["MACD_CROSS_UP"] and
            prev["MACD"] > 10 and
            prev["MACD"] > prev["MACD_SIGNAL"] and
            # prev["MACD_HIST"] > prev2["MACD_HIST"] and
            # prev2["MACD_HIST"] > df.iloc[i - 3]["MACD_HIST"] and
            # prev["Close"] > prev["EMA21"] and
            # prev["EMA21"] > prev["EMA50"] and
            # prev["ADX14"] >= ENTRY_MIN_ADX and
            # prev["+DI14"] > prev["-DI14"] and
            channel_pos < MAX_BUY_CHANNEL_POS
        )

        bearish_reversal = bool(
            prev["MACD_CROSS_DOWN"] and
            (
                bool(prev["RED_CANDLE"]) or
                bool(prev["MACD_HIST_WEAKENING"])
            )
        )

        stop_loss_exit = bool(live_profit_pct <= STOP_LOSS_PCT)

        early_fail_exit = False
        # early_fail_exit = bool(
        #     bearish_reversal and
        #     max_profit_signal_pct < EARLY_FAIL_MAX_PROFIT_PCT
        # )

        trailing_exit = False
        # trailing_exit = bool(
        #     max_profit_signal_pct >= MIN_TRAIL_PROFIT_PCT and
        #     live_profit_pct > 0 and
        #     max_profit_signal_pct <= (100.0 * TRAIL_GIVEBACK_RATIO)
        # )

        bearish_profit_exit = bool(
            bearish_reversal and
            live_profit_pct >= MIN_BEARISH_EXIT_PROFIT_PCT
        )

        if (not in_position) and bullish_entry:
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
                "reason": "strong_bullish_entry",
            })
            last_buy_price = exec_price
            peak_profit_pct = 0.0

        elif in_position and (stop_loss_exit or early_fail_exit or trailing_exit or bearish_profit_exit):
            sell_reason = (
                "stop_loss" if stop_loss_exit else
                "early_fail_exit" if early_fail_exit else
                "trail_take_profit" if trailing_exit else
                "bearish_exit_in_profit"
            )

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
                "reason": sell_reason,
            })
            peak_profit_pct = 0.0
            last_buy_price = None

    return signals


# =========
# Backtest
# =========
def run_mock_backtrade(
    df: pd.DataFrame,
    macd_signals: List[dict],
    start_zar: float = BACKTEST_START_ZAR,
    buy_fee_rate: float = TRADE_BUY_FEE_RATE,
    sell_fee_rate: float = TRADE_SELL_FEE_RATE,
) -> dict:
    if df.empty:
        return {
            "start_zar": start_zar,
            "final_zar": start_zar,
            "final_btc": 0.0,
            "mark_to_market_zar": start_zar,
            "trades": [],
            "history": {
                "time": [],
                "zar": [],
                "btc": [],
                "bank": [],
            },
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

    logging.info(
        "BACKTEST | START | start_zar=%.2f | buy_fee_rate=%.4f | sell_fee_rate=%.4f | candles=%d | signals=%d",
        zar_balance,
        buy_fee_rate,
        sell_fee_rate,
        len(df),
        len(signal_map),
    )

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

                logging.info(
                    "BACKTEST | BUY  | exec_time=%s | exec_price=%.2f | channel_pos=%.2f | live_profit_pct=%.2f | profit=%.2f | reason=%s",
                    exec_time,
                    exec_price,
                    float(signal["channel_pos"]),
                    float(signal["live_profit_pct"]),
                    float(signal["profit"]),
                    signal.get("reason", ""),
                )

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

                logging.info(
                    "BACKTEST | SELL | exec_time=%s | exec_price=%.2f | channel_pos=%.2f | live_profit_pct=%.2f | profit=%.2f | max_profit_pct=%.2f | reason=%s",
                    exec_time,
                    exec_price,
                    float(signal["channel_pos"]),
                    float(signal["live_profit_pct"]),
                    float(signal["profit"]),
                    float(signal.get("max_profit_pct", 0.0)),
                    signal.get("reason", ""),
                )

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
    # add the % of the change in price from very first to very last data point as a benchmark
    price_change_pct = _safe_pct(float(df["Close"].iloc[0]), float(df["Close"].iloc[-1]))
    logging.info(
        "BACKTEST | END | pnl_pct=%.2f%% | price_change_pct=%.2f%% | trades=%d | final_position=%s | final_zar=%.2f | final_btc=%.8f | last_close=%.2f | mark_to_market_zar=%.2f | total_fees_zar=%.2f | pnl_zar=%.2f",
        pnl_pct,
        price_change_pct,
        len(trades),
        position,
        zar_balance,
        btc_balance,
        last_close,
        mark_to_market_zar,
        total_fees_zar,
        pnl_zar,
    )

    return {
        "start_zar": start_zar,
        "final_zar": zar_balance,
        "final_btc": btc_balance,
        "mark_to_market_zar": mark_to_market_zar,
        "last_close": last_close,
        "pnl_zar": pnl_zar,
        "pnl_pct": pnl_pct,
        "total_fees_zar": total_fees_zar,
        "position": position,
        "trades": trades,
        "history": {
            "time": time_index,
            "zar": zar_history,
            "btc": btc_history,
            "bank": bank_history,
        },
    }


# =========
# Main
# =========
def main():
    logging.info("Starting candle graph for %s", PAIR)

    candles = get_candles()
    df = candles_to_dataframe(candles)
    df = add_indicators(df)

    if df.empty:
        raise RuntimeError("No candle data available")

    latest = df.iloc[-1]
    logging.info(
        "Loaded %d candles | latest close=%.2f | latest high=%.2f | latest low=%.2f",
        len(df),
        latest["Close"],
        latest["High"],
        latest["Low"],
    )

    candles_per_segment = max(2, int((TREND_WINDOW_HOURS * 3600) / CANDLE_SECONDS))
    half_offset = max(1, candles_per_segment // 2)

    logging.info(
        "Trend segment size: %dh = %d candles",
        TREND_WINDOW_HOURS,
        candles_per_segment,
    )
    logging.info(
        "Second line set offset: %d candles (half-window)",
        half_offset,
    )
    logging.info(
        "Trading rules | BUY on fresh bullish MACD cross with MACD>0, trend alignment, ADX strength, and lower-half channel position | SELL on stop loss, early failure, trailing giveback, or bearish reversal while in profit"
    )

    trend1, upper1, lower1, statuses_1 = build_segment_signal_arrays(
        df=df,
        window=candles_per_segment,
        offset=0,
        set_name="SET1",
    )

    trend2, upper2, lower2, statuses_2 = build_segment_signal_arrays(
        df=df,
        window=candles_per_segment,
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
        current_entry_price=CURRENT_ENTRY_PRICE,
        buy_fee_rate=TRADE_BUY_FEE_RATE,
        sell_fee_rate=TRADE_SELL_FEE_RATE,
    )

    backtest = run_mock_backtrade(
        df=df,
        macd_signals=macd_signals,
        start_zar=BACKTEST_START_ZAR,
        buy_fee_rate=TRADE_BUY_FEE_RATE,
        sell_fee_rate=TRADE_SELL_FEE_RATE,
    )

    hist = backtest.get("history", {})
    bank_series = pd.Series(hist.get("bank", []), index=hist.get("time", []), dtype=float)
    bank_series = bank_series.reindex(df.index)

    buy_marker = np.full(len(df), np.nan, dtype=float)
    sell_marker = np.full(len(df), np.nan, dtype=float)

    for trade in backtest["trades"]:
        ts = trade["exec_time"]
        idx = df.index.get_loc(ts)
        if trade["type"] == "BUY":
            buy_marker[idx] = float(df["Low"].iloc[idx]) * 0.997
        elif trade["type"] == "SELL":
            sell_marker[idx] = float(df["High"].iloc[idx]) * 1.003

    addplots = [
        mpf.make_addplot(avg_upper, width=1),
        mpf.make_addplot(avg_lower, width=1),

        # MACD panel
        mpf.make_addplot(df["MACD"], panel=2, ylabel="MACD"),
        mpf.make_addplot(df["MACD_SIGNAL"], panel=2),
        mpf.make_addplot(df["MACD_HIST"], type="bar", panel=2),

        # CHANNEL_POS panel
        mpf.make_addplot(df["CHANNEL_POS"], panel=3, ylabel="Channel Pos"),

        # BANK VALUE panel
        mpf.make_addplot(bank_series, panel=4, ylabel="Bank (ZAR)"),
    ]

    if np.isfinite(buy_marker).any():
        addplots.append(
            mpf.make_addplot(buy_marker, type="scatter", markersize=80, marker="^")
        )

    if np.isfinite(sell_marker).any():
        addplots.append(
            mpf.make_addplot(sell_marker, type="scatter", markersize=80, marker="v")
        )

    print(
        "BACKTEST: "
        f"start_zar={backtest['start_zar']:.2f} | "
        f"mark_to_market_zar={backtest['mark_to_market_zar']:.2f} | "
        f"fees_zar={backtest['total_fees_zar']:.2f} | "
        f"pnl_zar={backtest['pnl_zar']:.2f} | "
        f"pnl_pct={backtest['pnl_pct']:.2f}% | "
        f"final_position={backtest['position']} | "
        f"trades={len(backtest['trades'])}"
    )

    print("CURRENT SIGNAL REASONS:", ", ".join(current_signal["reasons"][:14]))

    logging.info("MACD signal count: %d", len(macd_signals))
    logging.info(latest_status)
    logging.info(
        "CURRENT SIGNAL | action=%s | score=%.2f | channel_pos=%.2f | macd=%.6f | macd_signal=%.6f | macd_hist=%.6f | live_profit_pct=%s | reasons=%s",
        current_signal["action"],
        current_signal["score"],
        current_signal["channel_pos"],
        float(df["MACD"].iloc[-1]),
        float(df["MACD_SIGNAL"].iloc[-1]),
        float(df["MACD_HIST"].iloc[-1]),
        "n/a" if current_signal["live_profit_pct"] is None else f"{current_signal['live_profit_pct']:.4f}",
        ",".join(current_signal["reasons"][:14]),
    )
    logging.info("Averaged upper/lower lines plotted from SET1 and SET2")

    mpf.plot(
        df,
        type="candle",
        volume=True,
        mav=(7, 21),
        addplot=addplots,
        title=(
            f"{PAIR} Luno Candles "
            f"({TREND_WINDOW_HOURS}h Segments + MACD Momentum Trading)"
        ),
        ylabel="Price (ZAR)",
        ylabel_lower="Volume",
        style="yahoo",
        tight_layout=True,
        figratio=(16, 9),
        figscale=1.2,
        panel_ratios=(6, 2, 2, 2, 2),
    )

    plt.show()


if __name__ == "__main__":
    main()
