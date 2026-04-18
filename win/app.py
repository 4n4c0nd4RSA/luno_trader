import json
import tkinter as tk
import webbrowser
from tkinter import messagebox

from config_defaults import APP_TITLE, CONFIG_FILE, ensure_config_shape
from license_util import check_license_file
from welcome_frame import WelcomeFrame
from config_frame import ConfigFrame
from backtest_frame import BacktestFrame
from dashboard_frame import DashboardFrame
from rules_engine import build_analysis_from_luno
from trade_engine import TraderRunner, ensure_live_trading_allowed


class LunoTraderUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1640x940")
        self.minsize(1320, 760)
        self.configure(bg="#f4f6f8")

        self.config_file = CONFIG_FILE
        self.config_data = self.load_config()
        self.backtest_result = None
        self.latest_engine_result = None
        self.trader_runner = TraderRunner(check_interval_seconds=3600)

        self.sidebar = None
        self.content = None
        self.frames = {}

        self._build_layout()
        self._build_frames()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.show_frame("welcome")
        self.after(200, self._auto_start_trader_if_enabled)

    def load_config(self):
        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                return ensure_config_shape(json.load(f))
        except Exception:
            return ensure_config_shape({})

    def save_config(self):
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(self.config_data, f, indent=2)

    def _enforce_runtime_license_guard(self):
        try:
            ensure_live_trading_allowed(self.config_data)
        except Exception:
            if not bool(self.config_data.get("mock_trade", True)):
                self.config_data["mock_trade"] = True
            raise

    def _has_valid_certificate(self):
        return check_license_file(
            license_file_path="license.key",
            public_key_path="public_key.pem",
        )

    def _auto_start_trader_if_enabled(self):
        self.config_data = self.load_config()
        if not bool(self.config_data.get("auto_start_trader", False)):
            return

        valid, _, message = self._has_valid_certificate()
        if not valid:
            self.config_data["auto_start_trader"] = False
            self.save_config()
            self.frames["config"].refresh()
            messagebox.showwarning(
                "Auto-Start Disabled",
                "Auto-Start Trader was turned off because a valid certificate/license was not found.\n\n"
                f"{message}"
            )
            return

        try:
            self.start_trader()
        except Exception as exc:
            messagebox.showerror("Auto-Start Trader Error", str(exc))

    def _build_layout(self):
        self.sidebar = tk.Frame(self, bg="#1f2937", width=250)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        self.content = tk.Frame(self, bg="#f4f6f8")
        self.content.pack(side="right", fill="both", expand=True)

        tk.Label(
            self.sidebar,
            text="Luno Trader",
            bg="#1f2937",
            fg="white",
            font=("Segoe UI", 18, "bold"),
            pady=20
        ).pack(fill="x")

        nav_items = [
            ("Dashboard", "dashboard"),
            ("Config", "config"),
            ("Backtest Results", "backtest"),
        ]

        for label, frame_name in nav_items:
            tk.Button(
                self.sidebar,
                text=label,
                anchor="w",
                relief="flat",
                bd=0,
                bg="#1f2937",
                fg="white",
                activebackground="#374151",
                activeforeground="white",
                padx=20,
                pady=14,
                font=("Segoe UI", 11, "bold"),
                command=lambda name=frame_name: self.show_frame(name)
            ).pack(fill="x", pady=2)

        tk.Frame(self.sidebar, bg="#1f2937").pack(fill="both", expand=True)

        footer = tk.Frame(self.sidebar, bg="#1f2937")
        footer.pack(fill="x", pady=20)

        tk.Label(
            footer,
            text="Version 2.0.0",
            bg="#1f2937",
            fg="#cbd5e1",
            font=("Segoe UI", 9),
            justify="left",
        ).pack(fill="x")

        tk.Label(
            footer,
            text="(c) 2026",
            bg="#1f2937",
            fg="#cbd5e1",
            font=("Segoe UI", 9),
            justify="left",
        ).pack(fill="x")

        company_link = tk.Label(
            footer,
            text="QuantumMind Software",
            bg="#1f2937",
            fg="#93c5fd",
            font=("Segoe UI", 9, "underline"),
            cursor="hand2",
            justify="left",
        )
        company_link.pack(fill="x")
        company_link.bind(
            "<Button-1>",
            lambda _event: webbrowser.open("https://www.quantummindsoftware.com/"),
        )

    def _build_frames(self):
        for cls, name in [
            (WelcomeFrame, "welcome"),
            (ConfigFrame, "config"),
            (BacktestFrame, "backtest"),
            (DashboardFrame, "dashboard"),
        ]:
            frame = cls(self.content, self)
            self.frames[name] = frame
            frame.place(relx=0, rely=0, relwidth=1, relheight=1)

    def show_frame(self, name):
        frame = self.frames[name]
        frame.tkraise()
        if hasattr(frame, "refresh"):
            frame.refresh()

    def run_backtest(self):
        try:
            self.save_config()
            engine_result = build_analysis_from_luno(self.config_data)
            self.latest_engine_result = engine_result

            df = engine_result["df"]
            backtest = engine_result["backtest"]
            current_signal = engine_result["current_signal"]

            signal_map = {}
            for signal in engine_result["macd_signals"]:
                signal_map[int(signal["exec_idx"])] = signal

            audit_rows = []
            for trade in backtest["trades"]:
                exec_time = trade["exec_time"]
                idx = df.index.get_loc(exec_time)
                bank_value = float(backtest["history"]["bank"][idx]) if idx < len(backtest["history"]["bank"]) else 0.0

                if trade["type"] == "BUY":
                    amount_zar = float(backtest["history"]["zar"][max(0, idx - 1)]) if idx > 0 else float(backtest["start_zar"])
                    btc_val = float(trade.get("btc_after", 0.0))
                    cash_after = 0.0
                else:
                    amount_zar = float(trade.get("zar_after", 0.0))
                    btc_val = 0.0
                    cash_after = float(trade.get("zar_after", 0.0))

                audit_rows.append({
                    "time": exec_time.strftime("%Y-%m-%d %H:%M"),
                    "action": trade["type"],
                    "price": round(float(trade["exec_price"]), 2),
                    "amount_zar": round(amount_zar, 2),
                    "btc": round(btc_val, 8),
                    "cash_after": round(cash_after, 2),
                    "bank_value": round(bank_value, 2),
                    "mode": "MOCK" if self.config_data.get("mock_trade", True) else "LIVE",
                    "reason": trade.get("rule_name") or trade.get("reason", ""),
                    "display_reason": trade.get("reason", ""),
                })

            events = []
            for trade in backtest["trades"]:
                events.append({
                    "type": trade["type"],
                    "time": trade["exec_time"],
                    "price": float(trade["exec_price"]),
                })

            self.backtest_result = {
                "times": list(df.index),
                "prices": df["Close"].astype(float).tolist(),
                "events": events,
                "audit_rows": audit_rows,
                "bank_values": [float(x) for x in backtest["history"]["bank"]],
                "bank_times": list(backtest["history"]["time"]),
                "summary": {
                    "start_bank": float(backtest["start_zar"]),
                    "end_bank": float(backtest["mark_to_market_zar"]),
                    "pnl": round(float(backtest["pnl_zar"]), 2),
                    "trade_count": len(backtest["trades"]),
                    "rule_count": len([r for r in self.config_data.get("rules", []) if r.get("active")]),
                    "mock_trade": bool(self.config_data.get("mock_trade", True)),
                    "current_signal_action": current_signal.get("action", "HOLD"),
                },
            }

            self.frames["backtest"].refresh()
            self.frames["dashboard"].refresh()
            self.show_frame("backtest")

        except Exception as e:
            messagebox.showerror("Backtest Error", str(e))

    def start_trader(self):
        self.save_config()
        self.config_data = self.load_config()
        self._enforce_runtime_license_guard()
        return self.trader_runner.start()

    def stop_trader(self):
        return self.trader_runner.stop()

    def get_trader_status(self):
        return self.trader_runner.get_status()

    def run_trader_once(self):
        self.save_config()
        self.config_data = self.load_config()
        self._enforce_runtime_license_guard()
        result = self.trader_runner.run_once()
        engine_result = result.get("engine_result")
        if isinstance(engine_result, dict):
            self.latest_engine_result = engine_result
        return result

    def _on_close(self):
        try:
            self.trader_runner.stop()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = LunoTraderUI()
    app.mainloop()
