import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from config_defaults import AUDIT_HISTORY_FILE, CONFIG_FILE, ensure_config_shape
from license_util import check_license_file
from rules_engine import build_analysis_from_luno, compute_current_market_signal

try:
    import luno_python.client as luno
except Exception:
    luno = None


CONFIG_PATH = Path(CONFIG_FILE)


def ensure_live_trading_allowed(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = cfg or load_config()
    if bool(cfg.get("mock_trade", True)):
        return cfg

    api_key_id = str(cfg.get("api_key_id", "")).strip()
    if not api_key_id:
        raise RuntimeError("A licensed Luno API key id is required for non-mock trading.")

    valid, _, message = check_license_file(
        license_file_path="license.key",
        public_key_path="public_key.pem",
        expected_key_id=api_key_id,
    )
    if not valid:
        raise RuntimeError(
            "A valid license key is required for non-mock trading.\n\n"
            f"{message}"
        )
    return cfg


def load_config() -> Dict[str, Any]:
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            return ensure_config_shape(json.load(f))
    return ensure_config_shape({})


def get_luno_client(cfg: Optional[Dict[str, Any]] = None):
    cfg = cfg or load_config()

    if luno is None:
        raise RuntimeError("luno_python is not installed.")

    api_key = str(cfg.get("api_key_id", "")).strip()
    api_secret = str(cfg.get("api_key_secret", "")).strip()

    if not api_key or not api_secret:
        raise RuntimeError("Missing api_key_id / api_key_secret in config.")

    return luno.Client(api_key_id=api_key, api_key_secret=api_secret)


def get_luno_user_ids(cfg: Optional[Dict[str, Any]] = None) -> list[str]:
    return get_luno_user_identity(cfg)["user_ids"]


def get_luno_user_identity(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = cfg or load_config()
    api_key = str(cfg.get("api_key_id", "")).strip()
    client = get_luno_client(cfg)
    user_ids = []
    seen = set()
    diagnostics = []

    for endpoint in ("/api/1/users/list", "/api/1/users/linked"):
        try:
            response = client.session.request(
                "GET",
                client.make_url(endpoint, None),
                timeout=client.timeout,
                headers={"User-Agent": client.make_user_agent()},
                auth=(client.api_key_id, client.api_key_secret),
            )
        except Exception as exc:
            diagnostics.append(f"{endpoint}: {exc}")
            continue

        try:
            payload = response.json()
        except Exception:
            diagnostics.append(f"{endpoint}: HTTP {response.status_code} non-JSON response")
            continue

        if response.status_code != 200:
            message = payload.get("error") or payload.get("message") or str(payload)
            diagnostics.append(f"{endpoint}: HTTP {response.status_code} {message}")
            continue

        users = payload.get("users", []) or []
        diagnostics.append(f"{endpoint}: {len(users)} user(s)")

        for user in users:
            user_id = str(user.get("user_id", "")).strip()
            if user_id and user_id not in seen:
                seen.add(user_id)
                user_ids.append(user_id)

    return {
        "user_ids": user_ids,
        "api_key_id": api_key,
        "diagnostics": diagnostics,
    }


def buy(
    amount_zar: float,
    pair: Optional[str] = None,
    cfg: Optional[Dict[str, Any]] = None,
    reason: str = "",
    record_audit: bool = True,
    market_price: Optional[float] = None,
) -> Dict[str, Any]:
    cfg = cfg or load_config()
    ensure_live_trading_allowed(cfg)
    pair = pair or str(cfg.get("pair", "XBTZAR")).strip() or "XBTZAR"

    amount_zar = float(amount_zar)
    if amount_zar <= 0:
        raise ValueError("amount_zar must be greater than 0.")

    if bool(cfg.get("mock_trade", True)):
        mock_price = _coerce_positive_float(market_price)
        if mock_price is None:
            mock_price = _get_latest_market_price(cfg)

        buy_fee_rate = float(cfg.get("trade_buy_fee_rate", 0.005))
        fee_zar = amount_zar * buy_fee_rate
        net_zar_to_spend = max(0.0, amount_zar - fee_zar)
        btc_bought = (net_zar_to_spend / mock_price) if mock_price and mock_price > 0 else None
        bank_value = (btc_bought * mock_price) if btc_bought is not None and mock_price else net_zar_to_spend

        response = _build_mock_order_response(
            action="BUY",
            pair=pair,
            amount_zar=amount_zar,
            btc_amount=btc_bought,
            price=mock_price,
        )
    else:
        client = get_luno_client(cfg)
        response = _submit_market_order(
            client=client,
            pair=pair,
            order_type="BUY",
            counter_volume=str(amount_zar),
        )

    event = _build_audit_event(
        action="BUY",
        pair=pair,
        reason=reason,
        raw_response=response,
        amount_zar=amount_zar,
        btc_amount=_extract_float(response, "base", "base_volume", "volume"),
        cash_after=0.0 if bool(cfg.get("mock_trade", True)) else None,
        price=_extract_float(response, "price", "avg_price", "average_price"),
        bank_value=bank_value if bool(cfg.get("mock_trade", True)) else None,
        mode="MOCK" if bool(cfg.get("mock_trade", True)) else "LIVE",
    )
    if record_audit:
        append_audit_event(event, cfg)
    return event


def sell(
    btc_amount: float,
    pair: Optional[str] = None,
    cfg: Optional[Dict[str, Any]] = None,
    reason: str = "",
    record_audit: bool = True,
    market_price: Optional[float] = None,
) -> Dict[str, Any]:
    cfg = cfg or load_config()
    ensure_live_trading_allowed(cfg)
    pair = pair or str(cfg.get("pair", "XBTZAR")).strip() or "XBTZAR"

    btc_amount = float(btc_amount)
    if btc_amount <= 0:
        raise ValueError("btc_amount must be greater than 0.")

    if bool(cfg.get("mock_trade", True)):
        mock_price = _coerce_positive_float(market_price)
        if mock_price is None:
            mock_price = _get_latest_market_price(cfg)

        sell_fee_rate = float(cfg.get("trade_sell_fee_rate", 0.003))
        gross_zar = (btc_amount * mock_price) if mock_price and mock_price > 0 else None
        fee_zar = (gross_zar * sell_fee_rate) if gross_zar is not None else None
        zar_received = (gross_zar - fee_zar) if gross_zar is not None and fee_zar is not None else None

        response = _build_mock_order_response(
            action="SELL",
            pair=pair,
            btc_amount=btc_amount,
            amount_zar=zar_received,
            price=mock_price,
        )
    else:
        client = get_luno_client(cfg)
        response = _submit_market_order(
            client=client,
            pair=pair,
            order_type="SELL",
            base_volume=str(btc_amount),
        )

    event = _build_audit_event(
        action="SELL",
        pair=pair,
        reason=reason,
        raw_response=response,
        amount_zar=_extract_float(response, "counter", "counter_volume", "proceeds"),
        btc_amount=0.0 if bool(cfg.get("mock_trade", True)) else btc_amount,
        cash_after=_extract_float(response, "counter", "counter_volume", "proceeds"),
        price=_extract_float(response, "price", "avg_price", "average_price"),
        bank_value=_extract_float(response, "counter", "counter_volume", "proceeds") if bool(cfg.get("mock_trade", True)) else None,
        mode="MOCK" if bool(cfg.get("mock_trade", True)) else "LIVE",
    )
    if record_audit:
        append_audit_event(event, cfg)
    return event


def append_audit_event(event: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> Path:
    cfg = cfg or load_config()
    audit_path = _get_audit_path(cfg)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    if audit_path.exists():
        try:
            with audit_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                records = loaded
        except Exception:
            records = []

    records.append(event)
    with audit_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    return audit_path


class TraderRunner:
    def __init__(self, check_interval_seconds: int = 3600):
        self.check_interval_seconds = max(60, int(check_interval_seconds))
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._status: Dict[str, Any] = {
            "running": False,
            "mode": "IDLE",
            "last_run_at": None,
            "last_action": "NONE",
            "last_signal": "UNKNOWN",
            "last_reason": "",
            "last_error": "",
            "last_order": None,
            "next_run_in_seconds": None,
            "thread_name": None,
        }

    def start(self) -> bool:
        cfg = load_config()
        ensure_live_trading_allowed(cfg)

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False

            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                name="LunoTraderRunner",
                daemon=True,
            )
            self._status.update(
                {
                    "running": True,
                    "mode": "STARTING",
                    "last_error": "",
                    "thread_name": "LunoTraderRunner",
                }
            )
            self._thread.start()
            return True

    def stop(self) -> bool:
        thread = None
        with self._lock:
            thread = self._thread
            if thread is None or not thread.is_alive():
                self._status["running"] = False
                self._status["mode"] = "STOPPED"
                self._status["next_run_in_seconds"] = None
                return False
            self._stop_event.set()
            self._status["mode"] = "STOPPING"

        thread.join(timeout=5)
        with self._lock:
            self._status["running"] = False
            self._status["mode"] = "STOPPED"
            self._status["next_run_in_seconds"] = None
            self._thread = None
        return True

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def run_once(self) -> Dict[str, Any]:
        cfg = load_config()
        ensure_live_trading_allowed(cfg)
        engine_result = build_analysis_from_luno(cfg)
        position_state = self._get_position_state(cfg)
        signal_cfg = dict(cfg)
        signal_cfg["current_entry_price"] = position_state.get("entry_price", 0.0) or 0.0
        signal_cfg["current_peak_profit_pct"] = self._estimate_peak_profit_pct(
            df=engine_result.get("df"),
            position_state=position_state,
            cfg=cfg,
        )
        current_signal = compute_current_market_signal(
            df=engine_result["df"],
            avg_upper=engine_result["avg_upper"],
            avg_lower=engine_result["avg_lower"],
            cfg=signal_cfg,
        )
        engine_result["current_signal"] = current_signal
        df = engine_result.get("df")
        latest_price = None
        if df is not None and not df.empty and "Close" in df:
            try:
                latest_price = float(df["Close"].iloc[-1])
            except Exception:
                latest_price = None
        action = str(current_signal.get("action", "HOLD")).upper()
        matched_rules = current_signal.get("matched_rules", []) or []
        reasons = current_signal.get("reasons", []) or []
        reason_label = str(matched_rules[0]) if matched_rules else _infer_signal_rule_name(action, reasons, cfg)
        reason_text = reason_label or ", ".join(str(reason) for reason in reasons[:8])
        order_event = None

        if action == "BUY" and not position_state["in_position"]:
            amount_zar = float(cfg.get("backtest_start_zar", 0.0))
            if amount_zar > 0:
                order_event = buy(
                    amount_zar=amount_zar,
                    cfg=cfg,
                    reason=reason_text,
                    record_audit=True,
                    market_price=latest_price,
                )
        elif action == "SELL" and position_state["in_position"]:
            btc_amount = position_state["btc_amount"]
            if btc_amount > 0:
                order_event = sell(
                    btc_amount=btc_amount,
                    cfg=cfg,
                    reason=reason_text,
                    record_audit=True,
                    market_price=latest_price,
                )

        outcome = {
            "engine_result": engine_result,
            "signal_action": action,
            "signal_reason": reason_text,
            "order_event": order_event,
            "in_position": position_state["in_position"],
            "executed_action": order_event["action"] if order_event else "HOLD",
        }

        with self._lock:
            self._status["last_run_at"] = datetime.now(timezone.utc).isoformat()
            self._status["last_signal"] = action
            self._status["last_reason"] = reason_text
            self._status["last_action"] = outcome["executed_action"]
            self._status["last_order"] = order_event
            self._status["last_error"] = ""

        return outcome

    def _run_loop(self):
        while not self._stop_event.is_set():
            with self._lock:
                self._status["mode"] = "CHECKING"
            try:
                self.run_once()
            except Exception as exc:
                with self._lock:
                    self._status["last_run_at"] = datetime.now(timezone.utc).isoformat()
                    self._status["last_error"] = str(exc)
                    self._status["mode"] = "ERROR"

            for remaining in range(self.check_interval_seconds, 0, -1):
                if self._stop_event.wait(1):
                    break
                with self._lock:
                    self._status["running"] = True
                    self._status["mode"] = "SLEEPING"
                    self._status["next_run_in_seconds"] = remaining - 1

        with self._lock:
            self._status["running"] = False
            self._status["mode"] = "STOPPED"
            self._status["next_run_in_seconds"] = None

    def _get_position_state(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        audit_path = _get_audit_path(cfg)
        if not audit_path.exists():
            return {"in_position": False, "btc_amount": 0.0, "entry_price": 0.0, "buy_time": None}

        try:
            with audit_path.open("r", encoding="utf-8") as f:
                events = json.load(f)
        except Exception:
            return {"in_position": False, "btc_amount": 0.0, "entry_price": 0.0, "buy_time": None}

        if not isinstance(events, list):
            return {"in_position": False, "btc_amount": 0.0, "entry_price": 0.0, "buy_time": None}

        in_position = False
        btc_amount = 0.0
        entry_price = 0.0
        buy_time = None
        for item in reversed(events):
            if not isinstance(item, dict):
                continue
            action = str(item.get("action", "")).upper()
            if action == "SELL":
                return {"in_position": False, "btc_amount": 0.0, "entry_price": 0.0, "buy_time": None}
            if action == "BUY":
                in_position = True
                btc_value = item.get("btc")
                if btc_value not in (None, ""):
                    try:
                        btc_amount = max(0.0, float(btc_value))
                    except Exception:
                        btc_amount = 0.0
                price_value = item.get("price")
                if price_value not in (None, ""):
                    try:
                        entry_price = max(0.0, float(price_value))
                    except Exception:
                        entry_price = 0.0
                buy_time = item.get("time")
                return {
                    "in_position": in_position,
                    "btc_amount": btc_amount,
                    "entry_price": entry_price,
                    "buy_time": buy_time,
                }
        return {"in_position": False, "btc_amount": 0.0, "entry_price": 0.0, "buy_time": None}

    def _estimate_peak_profit_pct(self, df, position_state: Dict[str, Any], cfg: Dict[str, Any]) -> float:
        if df is None or df.empty or not position_state.get("in_position"):
            return 0.0

        entry_price = float(position_state.get("entry_price") or 0.0)
        if entry_price <= 0:
            return 0.0

        buy_fee_rate = float(cfg.get("trade_buy_fee_rate", 0.005))
        sell_fee_rate = float(cfg.get("trade_sell_fee_rate", 0.003))
        buy_cost_per_unit = entry_price * (1.0 + buy_fee_rate)
        if buy_cost_per_unit <= 0:
            return 0.0

        trade_slice = df
        buy_time = _parse_audit_time(position_state.get("buy_time"))
        if buy_time is not None:
            try:
                trade_slice = df.loc[df.index >= buy_time]
            except Exception:
                trade_slice = df
            if trade_slice.empty:
                trade_slice = df

        try:
            best_close = float(trade_slice["Close"].astype(float).max())
        except Exception:
            return 0.0

        sell_value_per_unit = best_close * (1.0 - sell_fee_rate)
        return max(0.0, ((sell_value_per_unit / buy_cost_per_unit) - 1.0) * 100.0)


def _submit_market_order(client, pair: str, order_type: str, **kwargs) -> Dict[str, Any]:
    payload = {"pair": pair, "type": order_type}
    payload.update({key: value for key, value in kwargs.items() if value not in (None, "")})

    for method_name in ["post_market_order", "create_market_order"]:
        method = getattr(client, method_name, None)
        if callable(method):
            response = method(**payload)
            return response if isinstance(response, dict) else {"result": response}

    raise RuntimeError(
        "The installed luno client does not expose a supported market order method. "
        "Expected post_market_order or create_market_order."
    )


def _build_audit_event(
    action: str,
    pair: str,
    raw_response: Dict[str, Any],
    reason: str = "",
    amount_zar: Optional[float] = None,
    btc_amount: Optional[float] = None,
    cash_after: Optional[float] = None,
    price: Optional[float] = None,
    bank_value: Optional[float] = None,
    mode: str = "LIVE",
) -> Dict[str, Any]:
    order_id = _extract_text(raw_response, "order_id", "id")
    completed = raw_response.get("complete")
    event_time = _format_audit_time(_extract_text(raw_response, "timestamp", "time", "created_at"))

    return {
        "time": event_time,
        "action": action,
        "mode": mode,
        "pair": pair,
        "price": price,
        "amount_zar": amount_zar,
        "btc": btc_amount,
        "cash_after": cash_after,
        "bank_value": bank_value,
        "reason": reason,
        "order_id": order_id,
        "complete": completed,
        "response": raw_response,
    }


def _build_mock_order_response(
    action: str,
    pair: str,
    amount_zar: Optional[float] = None,
    btc_amount: Optional[float] = None,
    price: Optional[float] = None,
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    order_id = f"mock-{action.lower()}-{now.strftime('%Y%m%d%H%M%S%f')}"

    response: Dict[str, Any] = {
        "order_id": order_id,
        "id": order_id,
        "pair": pair,
        "type": action,
        "complete": True,
        "mock_trade": True,
        "timestamp": now.isoformat(),
    }

    if amount_zar is not None:
        response["counter_volume"] = float(amount_zar)
        response["counter"] = float(amount_zar)
    if btc_amount is not None:
        response["base_volume"] = float(btc_amount)
        response["base"] = float(btc_amount)
    if price is not None:
        response["price"] = float(price)
        response["avg_price"] = float(price)

    return response


def _get_audit_path(cfg: Dict[str, Any]) -> Path:
    configured = str(cfg.get("audit_history_file", AUDIT_HISTORY_FILE)).strip() or AUDIT_HISTORY_FILE
    path = Path(configured)
    if path.is_absolute():
        return path
    return CONFIG_PATH.resolve().parent / path


def _extract_float(data: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        if key in data and data[key] not in (None, ""):
            try:
                return float(data[key])
            except Exception:
                continue
    return None


def _extract_text(data: Dict[str, Any], *keys: str) -> Optional[str]:
    for key in keys:
        if key in data and data[key] not in (None, ""):
            return str(data[key])
    return None


def _coerce_positive_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except Exception:
        return None
    return number if number > 0 else None


def _get_latest_market_price(cfg: Dict[str, Any]) -> Optional[float]:
    try:
        engine_result = build_analysis_from_luno(cfg)
        df = engine_result.get("df")
        if df is None or df.empty or "Close" not in df:
            return None
        return _coerce_positive_float(df["Close"].iloc[-1])
    except Exception:
        return None


def _format_audit_time(value: Optional[str]) -> str:
    if value not in (None, ""):
        for candidate in [str(value).strip(), str(value).strip().replace("Z", "+00:00")]:
            try:
                dt_value = datetime.fromisoformat(candidate)
                if dt_value.tzinfo is not None:
                    dt_value = dt_value.astimezone(timezone.utc).replace(tzinfo=None)
                return dt_value.strftime("%Y-%m-%d %H:%M")
            except Exception:
                continue
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")


def _parse_audit_time(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    text = str(value).strip()
    for candidate in [text, text.replace("Z", "+00:00")]:
        try:
            dt_value = datetime.fromisoformat(candidate)
            if dt_value.tzinfo is not None:
                dt_value = dt_value.astimezone(timezone.utc).replace(tzinfo=None)
            return dt_value
        except Exception:
            continue
    return None


def _infer_signal_rule_name(action: str, reasons: Any, cfg: Dict[str, Any]) -> str:
    active_rules = {
        str(rule.get("name", "")): rule
        for rule in (cfg.get("rules", []) or [])
        if bool(rule.get("active", False))
    }

    reason_flags = set()
    for item in reasons or []:
        text = str(item)
        if "=" in text:
            key, value = text.split("=", 1)
            if value.strip().lower() == "true":
                reason_flags.add(key.strip())
        else:
            reason_flags.add(text.strip())

    if action == "BUY" and "bullish_entry" in active_rules:
        bullish_markers = {
            "macd_cross_up",
            "macd_above_signal",
            "hist_improving",
            "close_gt_ema21",
            "ema21_gt_ema50",
        }
        if "macd_cross_up" in reason_flags and len(bullish_markers & reason_flags) >= 2:
            return "bullish_entry"

    if action == "SELL":
        if "stop_loss_exit" in active_rules and "live_profit_pct" in "".join(str(item) for item in reasons or []):
            for item in reasons or []:
                text = str(item)
                if text.startswith("live_profit_pct="):
                    try:
                        live_profit_pct = float(text.split("=", 1)[1])
                        stop_loss_pct = float(cfg.get("stop_loss_pct", -2.5))
                        if live_profit_pct <= stop_loss_pct:
                            return "stop_loss_exit"
                    except Exception:
                        break

        if "bearish_reversal2" in active_rules and "macd_cross_down" in reason_flags and "hist_weakening" in reason_flags:
            return "bearish_reversal2"

        if "bearish_reversal" in active_rules and "macd_cross_down" in reason_flags and "red_candle" in reason_flags:
            return "bearish_reversal"

    return ""
