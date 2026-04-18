# luno_trader.py
import os
import time
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import luno_python.client as luno

# OpenAI (Responses API + optional Structured Outputs)
from openai import OpenAI
from pydantic import BaseModel
from typing_extensions import Literal


# =========
# Settings
# =========
PAIR = "XBTZAR"
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "3600"))  # how often to re-check and decide
HISTORY_HOURS = int(os.getenv("HISTORY_HOURS", "168"))  # 168 = last 7 days in candles
CANDLE_SECONDS = int(os.getenv("CANDLE_SECONDS", "3600"))  # 1h candles
BUY_ZAR_AMOUNT = float(os.getenv("BUY_ZAR_AMOUNT", "1000"))  # buy R1000 per BUY

# Start wallet (mock)
START_ZAR = float(os.getenv("START_ZAR", "5000"))
START_BTC = float(os.getenv("START_BTC", "0"))

# OpenAI model
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")


# =========
# Logging
# =========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("mock_trading_bot.log"), logging.StreamHandler()],
)


# =========
# Clients
# =========
LUNO_API_KEY = os.getenv("LUNO_API_KEY_ID")
LUNO_API_SECRET = os.getenv("LUNO_API_KEY_SECRET")
if not LUNO_API_KEY or not LUNO_API_SECRET:
    raise RuntimeError("Missing LUNO_API_KEY_ID / LUNO_API_KEY_SECRET env vars")

luno_client = luno.Client(api_key_id=LUNO_API_KEY, api_key_secret=LUNO_API_SECRET)

openai_client = OpenAI()  # uses OPENAI_API_KEY from env


# =========
# Data models
# =========
@dataclass
class Candle:
    ts_ms: int
    open: float
    close: float
    high: float
    low: float
    volume: float


class TradeDecision(BaseModel):
    action: Literal["BUY", "SELL"]


# =========
# Luno helpers
# =========
def get_ticker(pair: str = PAIR) -> dict:
    return luno_client.get_ticker(pair=pair)


def get_candles(pair: str = PAIR, history_hours: int = HISTORY_HOURS, candle_seconds: int = CANDLE_SECONDS) -> List[Candle]:
    """
    Luno candles endpoint takes: since (ms) and duration (seconds).
    We fetch the full history window in one call.
    """
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


# =========
# Stats for prompt-building
# =========
def pct_change(a: float, b: float) -> float:
    # change from a -> b, in %
    if a == 0:
        return 0.0
    return ((b - a) / a) * 100.0


def summarize_market(ticker: dict, candles: List[Candle]) -> str:
    """
    Build a compact, human-readable market snapshot for the LLM.
    """
    bid = float(ticker["bid"])
    ask = float(ticker["ask"])
    last_trade = float(ticker.get("last_trade", bid))

    if len(candles) < 2:
        return (
            f"PAIR={PAIR}\n"
            f"bid={bid:.2f} ZAR, ask={ask:.2f} ZAR, last_trade={last_trade:.2f} ZAR\n"
            f"Not enough candle history."
        )

    first = candles[0].close
    last = candles[-1].close

    # Approx windows
    # 24h change: use last candle vs candle ~24h ago (24 candles if 1h candles)
    idx_24h = max(0, len(candles) - 1 - 24)
    close_24h = candles[idx_24h].close

    # 7d change: use earliest available (history_hours default 168)
    close_7d = first

    hi_24h = max(c.high for c in candles[idx_24h:])
    lo_24h = min(c.low for c in candles[idx_24h:])

    vol_24h = sum(c.volume for c in candles[idx_24h:])

    # simple momentum proxy: last 6 candles average vs prior 6 candles average
    def avg_close(slice_: List[Candle]) -> float:
        return sum(c.close for c in slice_) / max(1, len(slice_))

    last6 = candles[-6:] if len(candles) >= 6 else candles
    prev6 = candles[-12:-6] if len(candles) >= 12 else candles[: max(1, len(candles) - len(last6))]

    mom = avg_close(last6) - avg_close(prev6) if prev6 else 0.0

    return (
        f"PAIR={PAIR}\n"
        f"bid={bid:.2f} ZAR, ask={ask:.2f} ZAR, last_trade={last_trade:.2f} ZAR\n"
        f"24h_close_change={pct_change(close_24h, last):.2f}% (from {close_24h:.2f} to {last:.2f})\n"
        f"7d_close_change={pct_change(close_7d, last):.2f}% (from {close_7d:.2f} to {last:.2f})\n"
        f"24h_range_high={hi_24h:.2f}, 24h_range_low={lo_24h:.2f}\n"
        f"24h_volume(sum candles)={vol_24h:.6f} BTC\n"
        f"momentum_proxy(last6_avg_minus_prev6_avg)={mom:.2f} ZAR\n"
        f"NOTE: This is mock trading, not financial advice."
    )


# =========
# OpenAI decision
# =========
FINAL_USER_QUESTION = 'Would you buy or sell btc at this very moment. Do a internet search then ONLY RESPOND WITH "BUY" OR "SELL"'


def llm_decide(market_summary: str) -> str:
    """
    Returns ONLY: "BUY" or "SELL"
    """
    prompt = f"""
You are deciding a single action for BTC spot trading.

Use ONLY the market data below.
Do NOT browse the internet.
Reply with exactly one word: BUY or SELL.

Rules:
- If short-term momentum and recent trend are positive, prefer BUY.
- If momentum and trend are negative, prefer SELL.
- No extra words.

MARKET DATA:
{market_summary}
""".strip()

    try:
        logging.info("Attempting structured output parsing for decision...")
        resp = openai_client.responses.parse(
            model=OPENAI_MODEL,
            input=prompt,
            text_format=TradeDecision,
            temperature=0,
        )
        logging.info(f"Structured output response: {resp}")
        parsed = resp.output_parsed
        if parsed and parsed.action in ("BUY", "SELL"):
            return parsed.action
    except Exception as e:
        logging.warning(f"Structured decision parse failed, falling back. Error={e}")

    resp = openai_client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
        temperature=0,
    )
    text = (resp.output_text or "").strip().upper()

    logging.info(f"Plain text response: {text!r}")

    if text == "BUY":
        return "BUY"
    if text == "SELL":
        return "SELL"

    logging.warning(f"Model returned non-compliant output: {text!r}. Defaulting to SELL.")
    return "SELL"


# =========
# Mock trading engine
# =========
@dataclass
class Wallet:
    zar: float
    btc: float


def mock_buy(wallet: Wallet, zar_amount: float, ask_price: float) -> None:
    """
    Spend zar_amount (or whatever is left) at ask_price to get BTC.
    """
    spend = min(wallet.zar, zar_amount)
    if spend <= 0:
        return
    btc_bought = spend / ask_price
    wallet.zar -= spend
    wallet.btc += btc_bought


def mock_sell_all(wallet: Wallet, bid_price: float) -> None:
    """
    Sell all BTC at bid_price.
    """
    if wallet.btc <= 0:
        return
    wallet.zar += wallet.btc * bid_price
    wallet.btc = 0.0


def wallet_value(wallet: Wallet, bid_price: float) -> float:
    return wallet.zar + (wallet.btc * bid_price)


# =========
# Main loop
# =========
def run():
    wallet = Wallet(zar=START_ZAR, btc=START_BTC)

    logging.info(f"Starting MOCK trader for {PAIR}")
    logging.info(f"Start wallet: ZAR={wallet.zar:.2f}, BTC={wallet.btc:.8f}")

    while True:
        try:
            ticker = get_ticker(PAIR)
            candles = get_candles(PAIR, HISTORY_HOURS, CANDLE_SECONDS)

            bid = float(ticker["bid"])
            ask = float(ticker["ask"])

            summary = summarize_market(ticker, candles)
            logging.info("Market summary for LLM:\n" + summary.replace("\n", "\n  "))
            decision = llm_decide(summary)  # "BUY" or "SELL" only
            logging.info(f"Decision: {decision}")

            before_val = wallet_value(wallet, bid)
            before = (wallet.zar, wallet.btc)

            if decision == "BUY":
                before_btc = wallet.btc
                before_zar = wallet.zar
                mock_buy(wallet, BUY_ZAR_AMOUNT, ask_price=ask)
                spent = before_zar - wallet.zar
                bought = wallet.btc - before_btc
                action_line = f"BUY -> spent R{spent:.2f}, bought {bought:.8f} BTC at ask={ask:.2f}"
            else:
                if wallet.btc > 0:
                    sold_btc = wallet.btc
                    mock_sell_all(wallet, bid_price=bid)
                    action_line = f"SELL -> sold {sold_btc:.8f} BTC at bid={bid:.2f}"
                else:
                    action_line = "SELL -> no BTC to sell"

            after_val = wallet_value(wallet, bid)
            after = (wallet.zar, wallet.btc)

            logging.info("====================================================")
            logging.info(f"Price: bid={bid:.2f} ask={ask:.2f} | Decision={decision}")
            logging.info(action_line)
            logging.info(f"Wallet before: ZAR={before[0]:.2f}, BTC={before[1]:.8f}, value={before_val:.2f}")
            logging.info(f"Wallet after : ZAR={after[0]:.2f}, BTC={after[1]:.8f}, value={after_val:.2f}")
            logging.info("====================================================")

        except Exception as e:
            logging.exception(f"Loop error: {e}")

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    run()