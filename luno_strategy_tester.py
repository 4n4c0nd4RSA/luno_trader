import os
import time
import logging
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import luno_python.client as luno


# =========
# Settings
# =========
PAIR = os.getenv("PAIR", "XBTZAR")

RUN_COUNT = int(os.getenv("RUN_COUNT", "10000"))
MIN_HISTORY_HOURS = int(os.getenv("MIN_HISTORY_HOURS", "1000"))
MAX_HISTORY_HOURS = int(os.getenv("MAX_HISTORY_HOURS", "114000"))

CANDLE_SECONDS = int(os.getenv("CANDLE_SECONDS", "3600"))
TREND_WINDOW_HOURS = int(os.getenv("TREND_WINDOW_HOURS", "48"))
BACKTEST_START_ZAR = float(os.getenv("BACKTEST_START_ZAR", "10000"))

TRADE_BUY_FEE_RATE = float(os.getenv("TRADE_BUY_FEE_RATE", "0.005"))
TRADE_SELL_FEE_RATE = float(os.getenv("TRADE_SELL_FEE_RATE", "0.003"))

RAW_SCORE_BUY_BIAS = float(os.getenv("RAW_SCORE_BUY_BIAS", "10"))
RAW_SCORE_SELL_BIAS = float(os.getenv("RAW_SCORE_SELL_BIAS", "-40"))

SELL_RED_BODY_MIN_DIFF = float(os.getenv("SELL_RED_BODY_MIN_DIFF", "10000"))

ENTRY_MIN_ADX = float(os.getenv("ENTRY_MIN_ADX", "20"))
ENTRY_MIN_HIST = float(os.getenv("ENTRY_MIN_HIST", "0"))
EARLY_FAIL_MAX_PROFIT_PCT = float(os.getenv("EARLY_FAIL_MAX_PROFIT_PCT", "1.5"))
MIN_BEARISH_EXIT_PROFIT_PCT = float(os.getenv("MIN_BEARISH_EXIT_PROFIT_PCT", "1.0")) #1.8
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "-4.5"))
MIN_TRAIL_PROFIT_PCT = float(os.getenv("MIN_TRAIL_PROFIT_PCT", "1.0"))
TRAIL_GIVEBACK_RATIO = float(os.getenv("TRAIL_GIVEBACK_RATIO", "0.60"))
MAX_BUY_CHANNEL_POS = float(os.getenv("MAX_BUY_CHANNEL_POS", "0.40"))

RANDOM_SEED = os.getenv("RANDOM_SEED")
if RANDOM_SEED not in (None, ""):
    random.seed(int(RANDOM_SEED))


# =========
# Logging
# =========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("strategy_tester.log"),
        logging.StreamHandler(),
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
    api_key_secret=LUNO_API_SECRET,
)

max_profit_pct = 0.0
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
    pair: str,
    history_hours: int,
    candle_seconds: int,
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


# =========
# Signal builder
# =========
def build_macd_signals(df: pd.DataFrame) -> List[dict]:
    global max_profit_pct
    global last_buy_price

    signals: List[dict] = []

    max_profit_pct = 0.0
    last_buy_price = None

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
            max_profit_pct = max(max_profit_pct, live_profit_pct)
        else:
            profit = 0.0
            live_profit_pct = 0.0

        bullish_entry = bool(
            prev["MACD"] > 10 and
            prev["MACD"] > prev["MACD_SIGNAL"] and
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
        trailing_exit = False

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
                "profit": 0.0,
                "reason": "strong_bullish_entry",
            })
            last_buy_price = exec_price
            max_profit_pct = 0.0

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
                "max_profit_pct": max_profit_pct,
                "profit": profit,
                "reason": sell_reason,
            })
            max_profit_pct = 0.0
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

    signal_map: Dict[int, dict] = {
        int(s["exec_idx"]): s
        for s in macd_signals
        if int(s["exec_idx"]) < len(df)
    }

    for i in range(1, len(df)):
        exec_time = df.index[i]
        exec_price = float(df["Open"].iloc[i])

        if exec_price <= 0:
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
                    "fee_zar": fee_zar,
                    "zar_after": zar_balance,
                    "reason": signal.get("reason", ""),
                })

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
        "price_change_pct": price_change_pct,
        "total_fees_zar": total_fees_zar,
        "position": position,
        "trades": trades,
    }


def run_single_test(run_number: int, history_hours: int) -> Dict[str, Any]:
    candles = get_candles(
        pair=PAIR,
        history_hours=history_hours,
        candle_seconds=CANDLE_SECONDS,
    )
    df = candles_to_dataframe(candles)
    df = add_indicators(df)

    if df.empty:
        raise RuntimeError(f"Run {run_number}: no candle data available")

    candles_per_segment = max(2, int((TREND_WINDOW_HOURS * 3600) / CANDLE_SECONDS))
    half_offset = max(1, candles_per_segment // 2)

    _, upper1, lower1, _ = build_segment_signal_arrays(
        df=df,
        window=candles_per_segment,
        offset=0,
        set_name="SET1",
    )

    _, upper2, lower2, _ = build_segment_signal_arrays(
        df=df,
        window=candles_per_segment,
        offset=half_offset,
        set_name="SET2",
    )

    avg_upper = nanmean_two(upper1, upper2)
    avg_lower = nanmean_two(lower1, lower2)

    _ = avg_upper, avg_lower  # kept because the original script builds them before signalling

    macd_signals = build_macd_signals(df)

    backtest = run_mock_backtrade(
        df=df,
        macd_signals=macd_signals,
        start_zar=BACKTEST_START_ZAR,
        buy_fee_rate=TRADE_BUY_FEE_RATE,
        sell_fee_rate=TRADE_SELL_FEE_RATE,
    )

    return {
        "run": run_number,
        "history_hours": history_hours,
        "candles": len(df),
        "start_time": df.index[0],
        "end_time": df.index[-1],
        "start_close": float(df["Close"].iloc[0]),
        "end_close": float(df["Close"].iloc[-1]),
        "pnl_pct": float(backtest["pnl_pct"]),
        "price_change_pct": float(backtest["price_change_pct"]),
        "trade_count": len(backtest["trades"]),
        "position": backtest["position"],
        "mark_to_market_zar": float(backtest["mark_to_market_zar"]),
        "fees_zar": float(backtest["total_fees_zar"]),
    }


def print_results_table(results: List[Dict[str, Any]]) -> None:
    print()
    print("=" * 160)
    print("STRATEGY TEST RESULTS")
    print("=" * 160)
    header = (
        f"{'Run':>3} | {'History Hours':>13} | {'Candles':>7} | "
        f"{'PnL %':>10} | {'Price Change %':>15} | {'Trades':>6} | {'Final Pos':>9} | "
        f"{'Mark-to-Market ZAR':>18} | {'Fees ZAR':>10} | {'Range'}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['run']:>3} | "
            f"{r['history_hours']:>13} | "
            f"{r['candles']:>7} | "
            f"{r['pnl_pct']:>9.2f}% | "
            f"{r['price_change_pct']:>14.2f}% | "
            f"{r['trade_count']:>6} | "
            f"{r['position']:>9} | "
            f"{r['mark_to_market_zar']:>18.2f} | "
            f"{r['fees_zar']:>10.2f} | "
            f"{r['start_time']} -> {r['end_time']}"
        )

    print("=" * 160)
    print()


def print_summary(results: List[Dict[str, Any]]) -> None:
    pnl_values = [r["pnl_pct"] for r in results]
    price_values = [r["price_change_pct"] for r in results]
    outperform_count = sum(1 for r in results if r["pnl_pct"] > r["price_change_pct"])
    profitable_count = sum(1 for r in results if r["pnl_pct"] > 0)

    best = max(results, key=lambda r: r["pnl_pct"])
    worst = min(results, key=lambda r: r["pnl_pct"])

    print("SUMMARY")
    print("-" * 80)
    print(f"runs                 : {len(results)}")
    print(f"avg pnl_pct          : {np.mean(pnl_values):.2f}%")
    print(f"median pnl_pct       : {np.median(pnl_values):.2f}%")
    print(f"min pnl_pct          : {np.min(pnl_values):.2f}%")
    print(f"max pnl_pct          : {np.max(pnl_values):.2f}%")
    print(f"avg price_change_pct : {np.mean(price_values):.2f}%")
    print(f"profitable runs      : {profitable_count}/{len(results)}")
    print(f"beat market runs     : {outperform_count}/{len(results)}")
    print(f"best run             : #{best['run']} ({best['pnl_pct']:.2f}% pnl, {best['price_change_pct']:.2f}% market, {best['history_hours']}h)")
    print(f"worst run            : #{worst['run']} ({worst['pnl_pct']:.2f}% pnl, {worst['price_change_pct']:.2f}% market, {worst['history_hours']}h)")
    print("-" * 80)


def main() -> None:
    if RUN_COUNT <= 0:
        raise RuntimeError("RUN_COUNT must be > 0")

    if MIN_HISTORY_HOURS <= 0 or MAX_HISTORY_HOURS <= 0:
        raise RuntimeError("MIN_HISTORY_HOURS and MAX_HISTORY_HOURS must be > 0")

    if MIN_HISTORY_HOURS > MAX_HISTORY_HOURS:
        raise RuntimeError("MIN_HISTORY_HOURS cannot be greater than MAX_HISTORY_HOURS")

    logging.info(
        "Starting strategy tester | pair=%s | runs=%d | history_range=%d..%d hours | candle_seconds=%d",
        PAIR,
        RUN_COUNT,
        MIN_HISTORY_HOURS,
        MAX_HISTORY_HOURS,
        CANDLE_SECONDS,
    )

    results: List[Dict[str, Any]] = []

    for run_number in range(1, RUN_COUNT + 1):
        history_hours = random.randint(MIN_HISTORY_HOURS, MAX_HISTORY_HOURS)
        logging.info("RUN %d/%d | history_hours=%d", run_number, RUN_COUNT, history_hours)

        try:
            result = run_single_test(run_number=run_number, history_hours=history_hours)
            results.append(result)

            print(
                f"RUN {result['run']:>2} | "
                f"history_hours={result['history_hours']:>6} | "
                f"candles={result['candles']:>5} | "
                f"pnl_pct={result['pnl_pct']:>8.2f}% | "
                f"price_change_pct={result['price_change_pct']:>8.2f}% | "
                f"trades={result['trade_count']:>3}"
            )
        except Exception as exc:
            logging.exception("RUN %d FAILED", run_number)
            print(
                f"RUN {run_number:>2} | "
                f"history_hours={history_hours:>6} | "
                f"FAILED | {exc}"
            )

    if not results:
        raise RuntimeError("All runs failed")

    print_results_table(results)
    print_summary(results)


if __name__ == "__main__":
    main()
