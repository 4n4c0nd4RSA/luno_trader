import json
import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path
from tkinter import messagebox, ttk

from base_page import BasePage
from config_defaults import AUDIT_HISTORY_FILE, SIGNAL_OPTIONS
from rules_engine import build_analysis_from_luno

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except Exception:
    FigureCanvasTkAgg = None
    Figure = None


class DashboardFrame(BasePage):
    DASHBOARD_HISTORY_HOURS = 975

    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.summary_labels = {}
        self.history_file_label = None
        self.trader_status_labels = {}
        self.signal_var = None
        self.signal_combo = None
        self.signal_chart_host = None
        self._available_signal_options = []
        self.trade_table = None
        self.price_chart_host = None
        self.bank_chart_host = None
        self._status_job = None
        self._latest_analysis_error = None
        self._current_analysis = None
        self.start_trader_button = None
        self.stop_trader_button = None
        self.run_once_button = None

        self.build_header("Dashboard", "See your trade audit history and control the live trader loop.")
        self._build_ui()

    def _build_ui(self):
        wrapper = tk.Frame(self, bg="#f4f6f8")
        wrapper.pack(fill="both", expand=True, padx=24, pady=(0, 20))

        top = tk.Frame(wrapper, bg="#f4f6f8")
        top.pack(fill="x", pady=(0, 10))

        self.start_trader_button = tk.Button(
            top,
            text="Start Trader",
            bg="#2563eb",
            fg="white",
            activebackground="#1d4ed8",
            activeforeground="white",
            font=("Segoe UI", 10, "bold"),
            padx=18,
            pady=10,
            command=self._start_trader,
        )
        self.start_trader_button.pack(side="left", padx=(0, 10))

        self.stop_trader_button = tk.Button(
            top,
            text="Stop Trader",
            bg="#ef4444",
            fg="white",
            activebackground="#dc2626",
            activeforeground="white",
            font=("Segoe UI", 10, "bold"),
            padx=18,
            pady=10,
            command=self._stop_trader,
        )

        self.run_once_button = tk.Button(
            top,
            text="Run Once Now",
            bg="#0f766e",
            fg="white",
            activebackground="#115e59",
            activeforeground="white",
            font=("Segoe UI", 10, "bold"),
            padx=18,
            pady=10,
            command=self._run_trader_once,
        )
        self.run_once_button.pack(side="left", padx=(0, 10))

        tk.Button(
            top,
            text="Refresh History",
            bg="#6b7280",
            fg="white",
            activebackground="#4b5563",
            activeforeground="white",
            font=("Segoe UI", 10, "bold"),
            padx=18,
            pady=10,
            command=self.refresh,
        ).pack(side="left", padx=(0, 10))

        tk.Button(
            top,
            text="Open Config",
            bg="#10b981",
            fg="white",
            activebackground="#059669",
            activeforeground="white",
            font=("Segoe UI", 10, "bold"),
            padx=18,
            pady=10,
            command=lambda: self.app.show_frame("config"),
        ).pack(side="left")

        self.history_file_label = tk.Label(
            top,
            text="",
            bg="#f4f6f8",
            fg="#6b7280",
            font=("Segoe UI", 10),
        )
        self.history_file_label.pack(side="right")

        status_strip = tk.Frame(wrapper, bg="#f4f6f8")
        status_strip.pack(fill="x", pady=(0, 10))

        for key in ["Trader", "Last Signal", "Next Check", "Last Run", "Error"]:
            card = tk.Frame(status_strip, bg="white", bd=1, relief="solid")
            card.pack(side="left", fill="x", expand=True, padx=6)

            tk.Label(
                card,
                text=key,
                bg="white",
                fg="#6b7280",
                font=("Segoe UI", 9, "bold"),
            ).pack(anchor="w", padx=12, pady=(10, 4))

            value = tk.Label(
                card,
                text="-",
                bg="white",
                fg="#111827",
                font=("Segoe UI", 12, "bold"),
                wraplength=220,
                justify="left",
            )
            value.pack(anchor="w", padx=12, pady=(0, 12))
            self.trader_status_labels[key] = value

        summary = tk.Frame(wrapper, bg="#f4f6f8")
        summary.pack(fill="x", pady=(0, 10))

        for key in ["Trades", "Buys", "Sells", "Last Action", "Last Price", "Last Bank"]:
            card = tk.Frame(summary, bg="white", bd=1, relief="solid")
            card.pack(side="left", fill="x", expand=True, padx=6)

            tk.Label(
                card,
                text=key,
                bg="white",
                fg="#6b7280",
                font=("Segoe UI", 9, "bold"),
            ).pack(anchor="w", padx=12, pady=(10, 4))

            value = tk.Label(
                card,
                text="-",
                bg="white",
                fg="#111827",
                font=("Segoe UI", 15, "bold"),
            )
            value.pack(anchor="w", padx=12, pady=(0, 12))
            self.summary_labels[key] = value

        body = ttk.PanedWindow(wrapper, orient="horizontal")
        body.pack(fill="both", expand=True)

        left = ttk.PanedWindow(body, orient="vertical")
        right = ttk.PanedWindow(body, orient="vertical")
        body.add(left, weight=3)
        body.add(right, weight=2)

        latest_card = tk.Frame(left, bg="white", bd=1, relief="solid")
        price_card = tk.Frame(left, bg="white", bd=1, relief="solid")
        left.add(latest_card, weight=1)
        left.add(price_card, weight=3)

        table_card = tk.Frame(right, bg="white", bd=1, relief="solid")
        bank_card = tk.Frame(right, bg="white", bd=1, relief="solid")
        right.add(table_card, weight=1)
        right.add(bank_card, weight=1)

        tk.Label(
            latest_card,
            text="Signal Explorer",
            bg="white",
            fg="#111827",
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w", padx=12, pady=(12, 2))

        controls = tk.Frame(latest_card, bg="white")
        controls.pack(fill="x", padx=12, pady=(4, 10))

        tk.Label(
            controls,
            text="Signal",
            bg="white",
            fg="#4b5563",
            font=("Segoe UI", 10, "bold"),
        ).pack(side="left", padx=(0, 8))

        self.signal_var = tk.StringVar(value="MACD")
        self.signal_combo = ttk.Combobox(
            controls,
            textvariable=self.signal_var,
            state="readonly",
            width=28,
            values=["MACD"],
        )
        self.signal_combo.pack(side="left", fill="x", expand=True)
        self.signal_combo.bind("<<ComboboxSelected>>", self._on_signal_selected)

        self.signal_chart_host = tk.Frame(latest_card, bg="white")
        self.signal_chart_host.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        tk.Label(
            table_card,
            text="Trade Audit Events",
            bg="white",
            fg="#111827",
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w", padx=12, pady=(12, 10))

        table_wrap = tk.Frame(table_card, bg="white")
        table_wrap.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        cols = ("time", "action", "mode", "price", "amount_zar", "btc", "cash_after", "bank_value", "reason")
        self.trade_table = ttk.Treeview(table_wrap, columns=cols, show="headings", height=18)
        for col, width in [
            ("time", 150),
            ("action", 80),
            ("mode", 70),
            ("price", 90),
            ("amount_zar", 100),
            ("btc", 110),
            ("cash_after", 110),
            ("bank_value", 110),
            ("reason", 220),
        ]:
            self.trade_table.heading(col, text=col.replace("_", " ").title())
            self.trade_table.column(col, width=width, anchor="center")

        vsb = ttk.Scrollbar(table_wrap, orient="vertical", command=self.trade_table.yview)
        self.trade_table.configure(yscrollcommand=vsb.set)
        self.trade_table.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        tk.Label(
            price_card,
            text="Market Price + Indicators",
            bg="white",
            fg="#111827",
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w", padx=12, pady=(12, 0))

        self.price_chart_host = tk.Frame(price_card, bg="white")
        self.price_chart_host.pack(fill="both", expand=True, padx=10, pady=10)

        tk.Label(
            bank_card,
            text="Bank Value History",
            bg="white",
            fg="#111827",
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w", padx=12, pady=(12, 0))

        self.bank_chart_host = tk.Frame(bank_card, bg="white")
        self.bank_chart_host.pack(fill="both", expand=True, padx=10, pady=10)

    def refresh(self):
        self._clear_table()
        self._clear_frame(self.signal_chart_host)
        self._clear_frame(self.price_chart_host)
        self._clear_frame(self.bank_chart_host)

        history_path = self._get_history_file_path()
        self.history_file_label.config(text=f"History File: {history_path.name}")
        self._refresh_trader_status()
        self._schedule_status_refresh()

        records, error_text = self._load_history_records(history_path)
        if error_text:
            records = []
        analysis = self._load_market_analysis()
        self._current_analysis = analysis

        if error_text and not analysis:
            self._set_empty_state(history_path, error_text)
            return

        normalized = [self._normalize_trade_event(item, index) for index, item in enumerate(records or [])]
        normalized = [item for item in normalized if item is not None]
        if not normalized and not analysis:
            self._set_empty_state(history_path, "The history file contains no usable trade events.")
            return
        if normalized:
            normalized.sort(key=lambda item: item["time_order"])

        if normalized:
            self._populate_summary(normalized)
            self._populate_trade_table(normalized)
        else:
            self._clear_summary()

        self._update_signal_selector(analysis)
        self._draw_signal_chart(analysis)

        self._draw_price_chart(analysis, normalized)
        self._draw_bank_chart(normalized)

    def _start_trader(self):
        try:
            started = self.app.start_trader()
            self._refresh_trader_status()
            if not started:
                messagebox.showinfo("Trader", "Trader loop is already running.")
        except Exception as exc:
            messagebox.showerror("Trader Start Error", str(exc))

    def _stop_trader(self):
        try:
            stopped = self.app.stop_trader()
            self._refresh_trader_status()
            if not stopped:
                messagebox.showinfo("Trader", "Trader loop is already stopped.")
        except Exception as exc:
            messagebox.showerror("Trader Stop Error", str(exc))

    def _run_trader_once(self):
        try:
            result = self.app.run_trader_once()
            self.refresh()
            messagebox.showinfo(
                "Trader Run",
                f"Signal: {result.get('signal_action', 'HOLD')}\nExecuted: {result.get('executed_action', 'HOLD')}",
            )
        except Exception as exc:
            messagebox.showerror("Trader Run Error", str(exc))

    def _refresh_trader_status(self):
        status = self.app.get_trader_status()
        self._sync_trader_action_buttons(status)
        trader_text = "RUNNING" if status.get("running") else status.get("mode", "STOPPED")
        self.trader_status_labels["Trader"].config(text=trader_text)
        self.trader_status_labels["Last Signal"].config(text=str(status.get("last_signal", "-")))
        self.trader_status_labels["Next Check"].config(text=self._format_countdown(status.get("next_run_in_seconds")))
        self.trader_status_labels["Last Run"].config(text=self._format_status_time(status.get("last_run_at")))
        self.trader_status_labels["Error"].config(text=str(status.get("last_error", "")) or "-")

    def _sync_trader_action_buttons(self, status):
        if self.start_trader_button is None or self.stop_trader_button is None:
            return

        is_running = bool(status.get("running"))
        if is_running:
            if self.start_trader_button.winfo_manager():
                self.start_trader_button.pack_forget()
            if not self.stop_trader_button.winfo_manager():
                pack_kwargs = {"side": "left", "padx": (0, 10)}
                if self.run_once_button is not None:
                    pack_kwargs["before"] = self.run_once_button
                self.stop_trader_button.pack(**pack_kwargs)
        else:
            if self.stop_trader_button.winfo_manager():
                self.stop_trader_button.pack_forget()
            if not self.start_trader_button.winfo_manager():
                pack_kwargs = {"side": "left", "padx": (0, 10)}
                if self.run_once_button is not None:
                    pack_kwargs["before"] = self.run_once_button
                self.start_trader_button.pack(**pack_kwargs)

    def _schedule_status_refresh(self):
        if self._status_job is not None:
            self.after_cancel(self._status_job)
        self._status_job = self.after(1000, self._status_tick)

    def _status_tick(self):
        self._refresh_trader_status()
        self._status_job = self.after(1000, self._status_tick)

    def _get_history_file_path(self):
        configured = str(self.app.config_data.get("audit_history_file", AUDIT_HISTORY_FILE)).strip()
        if not configured:
            configured = AUDIT_HISTORY_FILE
        path = Path(configured)
        if not path.is_absolute():
            path = Path(self.app.config_file).resolve().parent / path
        return path

    def _load_history_records(self, history_path):
        if not history_path.exists():
            return None, f"History file not found: {history_path}"

        try:
            with history_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            return None, f"Could not read history file: {exc}"

        if not isinstance(data, list):
            return None, "History file must contain a JSON list of trade events."

        return data, None

    def _normalize_trade_event(self, item, index):
        if not isinstance(item, dict):
            return None

        time_text = self._coalesce(item, "time", "timestamp", "date", "exec_time", "executed_at", "created_at")
        action = str(self._coalesce(item, "action", "type", "side", default="UNKNOWN")).upper()
        mode = str(self._coalesce(item, "mode", default="LIVE")).upper()
        price = self._to_float(self._coalesce(item, "price", "exec_price", "close", "trade_price"))
        amount_zar = self._to_float(self._coalesce(item, "amount_zar", "zar_amount", "amount", "notional_zar"))
        btc = self._to_float(self._coalesce(item, "btc", "btc_after", "btc_amount", "amount_btc"))
        cash_after = self._to_float(self._coalesce(item, "cash_after", "zar_after", "cash_balance", "zar_balance"))
        bank_value = self._to_float(self._coalesce(item, "bank_value", "portfolio_value", "equity", "account_value"))
        reason = str(self._coalesce(item, "reason", "note", "message", default=""))

        return {
            "index": index,
            "time": self._format_time_text(time_text, index),
            "raw_time": time_text,
            "time_order": self._parse_time_for_sort(time_text, index),
            "action": action,
            "mode": mode,
            "price": price,
            "amount_zar": amount_zar,
            "btc": btc,
            "cash_after": cash_after,
            "bank_value": bank_value,
            "reason": reason,
        }

    def _populate_summary(self, records):
        buys = sum(1 for row in records if row["action"] == "BUY")
        sells = sum(1 for row in records if row["action"] == "SELL")
        last = records[-1]

        self.summary_labels["Trades"].config(text=str(len(records)))
        self.summary_labels["Buys"].config(text=str(buys))
        self.summary_labels["Sells"].config(text=str(sells))
        self.summary_labels["Last Action"].config(text=last["action"])
        self.summary_labels["Last Price"].config(text=self._format_currency(last["price"]))
        self.summary_labels["Last Bank"].config(text=self._format_currency(last["bank_value"]))

    def _populate_trade_table(self, records):
        for row in reversed(records[-200:]):
            self.trade_table.insert(
                "",
                "end",
                values=(
                    row["time"],
                    row["action"],
                    row["mode"],
                    self._format_number(row["price"]),
                    self._format_number(row["amount_zar"]),
                    self._format_number(row["btc"], digits=8),
                    self._format_number(row["cash_after"]),
                    self._format_number(row["bank_value"]),
                    row["reason"],
                ),
            )

    def _load_market_analysis(self):
        self._latest_analysis_error = None

        try:
            self.app.save_config()
            self.app.config_data = self.app.load_config()
            dashboard_cfg = dict(self.app.config_data)
            dashboard_cfg["history_hours"] = self.DASHBOARD_HISTORY_HOURS
            analysis = build_analysis_from_luno(dashboard_cfg)
            self.app.latest_engine_result = analysis
            return analysis
        except Exception as exc:
            self._latest_analysis_error = str(exc)
            cached = getattr(self.app, "latest_engine_result", None)
            if isinstance(cached, dict) and cached.get("df") is not None:
                self._latest_analysis_error = f"Showing cached market analysis. Refresh failed: {exc}"
                return cached
            return None

    def _draw_price_chart(self, analysis, records):
        if not FigureCanvasTkAgg or not Figure:
            self._render_no_data(self.price_chart_host, "matplotlib not installed. Run: pip install matplotlib")
            return

        if not analysis:
            message = "Could not load market history for the dashboard chart."
            if self._latest_analysis_error:
                message = f"{message}\n{self._latest_analysis_error}"
            self._render_no_data(self.price_chart_host, message)
            return

        df = analysis["df"]
        if df is None or df.empty:
            self._render_no_data(self.price_chart_host, "No candle history returned for the dashboard chart.")
            return

        fig = Figure(figsize=(7, 4.8), dpi=100)
        price_ax = fig.add_subplot(211)
        indicator_ax = fig.add_subplot(212, sharex=price_ax)

        x_values = list(df.index)
        close_values = df["Close"].astype(float)
        price_ax.plot(x_values, close_values, linewidth=1.6, color="#0f172a", label="Close")

        for column, color, width in [
            ("EMA9", "#2563eb", 1.1),
            ("EMA21", "#059669", 1.1),
            ("EMA50", "#f59e0b", 1.0),
            ("EMA200", "#7c3aed", 1.0),
        ]:
            if column in df:
                price_ax.plot(x_values, df[column].astype(float), linewidth=width, color=color, alpha=0.95, label=column)

        avg_upper = analysis.get("avg_upper")
        avg_lower = analysis.get("avg_lower")
        if avg_upper is not None and avg_lower is not None:
            price_ax.plot(x_values, avg_upper, linewidth=0.9, color="#94a3b8", alpha=0.9, linestyle="--", label="Trend Upper")
            price_ax.plot(x_values, avg_lower, linewidth=0.9, color="#cbd5e1", alpha=0.9, linestyle="--", label="Trend Lower")

        buy_points = []
        sell_points = []
        trade_time_lookup = {}
        for ts in x_values:
            trade_time_lookup[str(ts)] = ts
            trade_time_lookup[ts.strftime("%Y-%m-%d %H:%M")] = ts
            trade_time_lookup[ts.isoformat()] = ts
        for row in records:
            if row["price"] is None:
                continue
            raw_time = row.get("raw_time")
            if raw_time in (None, ""):
                continue
            matched_time = self._match_trade_time_to_index(raw_time, trade_time_lookup)
            if matched_time is None:
                continue
            target_list = buy_points if row["action"] == "BUY" else sell_points if row["action"] == "SELL" else None
            if target_list is not None:
                target_list.append((matched_time, row["price"]))

        if buy_points:
            price_ax.scatter([point[0] for point in buy_points], [point[1] for point in buy_points], marker="^", s=54, color="#10b981", label="BUY")
        if sell_points:
            price_ax.scatter([point[0] for point in sell_points], [point[1] for point in sell_points], marker="v", s=54, color="#ef4444", label="SELL")

        price_ax.set_title(f"Last {self.DASHBOARD_HISTORY_HOURS} Hours Price History")
        price_ax.set_ylabel("Price")
        price_ax.grid(True, alpha=0.25)
        price_ax.legend(loc="upper left", ncol=4, fontsize=8)

        macd_hist = df["MACD_HIST"].astype(float)
        bar_colors = ["#10b981" if value >= 0 else "#ef4444" for value in macd_hist]
        indicator_ax.bar(x_values, macd_hist, color=bar_colors, alpha=0.35, width=0.03)
        indicator_ax.plot(x_values, df["MACD"].astype(float), linewidth=1.1, color="#2563eb", label="MACD")
        indicator_ax.plot(x_values, df["MACD_SIGNAL"].astype(float), linewidth=1.0, color="#f59e0b", label="MACD Signal")
        indicator_ax.plot(x_values, df["RSI14"].astype(float), linewidth=0.9, color="#7c3aed", label="RSI14")
        indicator_ax.axhline(0.0, linewidth=0.8, color="#9ca3af", alpha=0.7)
        indicator_ax.axhline(30.0, linewidth=0.8, color="#d1d5db", alpha=0.8, linestyle=":")
        indicator_ax.axhline(70.0, linewidth=0.8, color="#d1d5db", alpha=0.8, linestyle=":")
        indicator_ax.set_ylabel("Indicators")
        indicator_ax.set_xlabel("Time")
        indicator_ax.grid(True, alpha=0.18)
        indicator_ax.legend(loc="upper left", ncol=3, fontsize=8)

        if self._latest_analysis_error:
            fig.suptitle(self._latest_analysis_error, fontsize=9, color="#b45309", y=0.995)

        fig.autofmt_xdate()
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.price_chart_host)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _update_signal_selector(self, analysis):
        options = []
        if analysis and analysis.get("df") is not None:
            df = analysis["df"]
            options = [name for name in SIGNAL_OPTIONS if name in df.columns]

        if not options:
            options = ["MACD"]

        self._available_signal_options = options
        self.signal_combo["values"] = options
        if self.signal_var.get() not in options:
            self.signal_var.set(options[0])

    def _draw_signal_chart(self, analysis):
        if not FigureCanvasTkAgg or not Figure:
            self._render_no_data(self.signal_chart_host, "matplotlib not installed. Run: pip install matplotlib")
            return

        if not analysis:
            message = "Could not load market history for the selected signal."
            if self._latest_analysis_error:
                message = f"{message}\n{self._latest_analysis_error}"
            self._render_no_data(self.signal_chart_host, message)
            return

        df = analysis.get("df")
        selected_signal = self.signal_var.get().strip()
        if df is None or df.empty or not selected_signal or selected_signal not in df.columns:
            self._render_no_data(self.signal_chart_host, "The selected signal is not available in the loaded analysis.")
            return

        series = df[selected_signal]
        fig = Figure(figsize=(7, 2.7), dpi=100)
        ax = fig.add_subplot(111)
        x_values = list(df.index)

        try:
            y_values = series.astype(float)
            ax.plot(x_values, y_values, linewidth=1.7, color="#2563eb")
            if selected_signal == "MACD_HIST":
                bar_colors = ["#10b981" if value >= 0 else "#ef4444" for value in y_values]
                ax.bar(x_values, y_values, color=bar_colors, alpha=0.35, width=0.03)
                ax.axhline(0.0, linewidth=0.8, color="#9ca3af", alpha=0.7)
            elif selected_signal == "RSI14":
                ax.axhline(30.0, linewidth=0.8, color="#d1d5db", alpha=0.8, linestyle=":")
                ax.axhline(70.0, linewidth=0.8, color="#d1d5db", alpha=0.8, linestyle=":")
        except Exception:
            y_values = [str(value) for value in series.tolist()]
            ax.plot(range(len(y_values)), y_values, linewidth=1.4, color="#2563eb")

        ax.set_title(selected_signal)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.2)
        fig.autofmt_xdate()
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.signal_chart_host)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _draw_bank_chart(self, records):
        if not FigureCanvasTkAgg or not Figure:
            self._render_no_data(self.bank_chart_host, "matplotlib not installed. Run: pip install matplotlib")
            return

        points = [(row["time"], row["bank_value"]) for row in records if row["bank_value"] is not None]
        if not points:
            self._render_no_data(self.bank_chart_host, "No bank values found in the history file.")
            return

        fig = Figure(figsize=(7, 3.0), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(
            [time_text for time_text, _ in points],
            [bank for _, bank in points],
            linewidth=1.8,
            color="#2563eb",
        )
        ax.set_title("Bank Value From Audit History")
        ax.set_xlabel("Event Time")
        ax.set_ylabel("Bank")
        ax.grid(True, alpha=0.25)

        canvas = FigureCanvasTkAgg(fig, master=self.bank_chart_host)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _set_empty_state(self, history_path, message):
        self._clear_summary()
        self._current_analysis = None
        self.history_file_label.config(text=f"History File: {history_path.name}")
        self._render_no_data(self.signal_chart_host, message)
        self._render_no_data(self.price_chart_host, message)
        self._render_no_data(self.bank_chart_host, message)

    def _clear_summary(self):
        for key in self.summary_labels:
            self.summary_labels[key].config(text="-")

    def _render_no_data(self, host, text):
        tk.Label(host, text=text, bg="white", fg="#6b7280", font=("Segoe UI", 11), wraplength=500).pack(expand=True)

    def _on_signal_selected(self, _event=None):
        self._clear_frame(self.signal_chart_host)
        self._draw_signal_chart(self._current_analysis)

    def _coalesce(self, item, *keys, default=None):
        for key in keys:
            if key in item and item[key] not in (None, ""):
                return item[key]
        return default

    def _to_float(self, value):
        if value in (None, ""):
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _format_time_text(self, value, index):
        if value in (None, ""):
            return f"Event {index + 1}"
        return str(value)

    def _parse_time_for_sort(self, value, index):
        if value in (None, ""):
            return (1, index)

        text = str(value).strip()
        dt_value = self._parse_trade_datetime(value)
        if dt_value is not None:
            return (0, dt_value)
        return (1, text, index)

    def _match_trade_time_to_index(self, value, trade_time_lookup):
        if value in (None, ""):
            return None

        text = str(value).strip()
        dt_value = self._parse_trade_datetime(value)
        if dt_value is not None:
            iso_key = dt_value.isoformat()
            if iso_key in trade_time_lookup:
                return trade_time_lookup[iso_key]
            minute_key = dt_value.strftime("%Y-%m-%d %H:%M")
            if minute_key in trade_time_lookup:
                return trade_time_lookup[minute_key]

        if text in trade_time_lookup:
            return trade_time_lookup[text]
        return None

    def _parse_trade_datetime(self, value):
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

    def _format_number(self, value, digits=2):
        if value is None:
            return "-"
        return f"{float(value):.{digits}f}"

    def _format_currency(self, value):
        if value is None:
            return "-"
        return f"R {float(value):.2f}"

    def _format_status_time(self, value):
        if not value:
            return "-"
        return str(value)

    def _format_countdown(self, seconds):
        if seconds in (None, ""):
            return "-"
        seconds = int(seconds)
        minutes, secs = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _clear_table(self):
        for item in self.trade_table.get_children():
            self.trade_table.delete(item)

    def _clear_frame(self, frame):
        for child in frame.winfo_children():
            child.destroy()
