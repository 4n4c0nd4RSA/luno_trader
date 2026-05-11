import json
import tkinter as tk
from datetime import datetime, timezone
from tkinter import ttk, messagebox

from base_page import BasePage
from widgets import ScrollableFrame
from rule_editor_dialog import RuleEditorDialog
from config_defaults import deep_copy_json
from license_util import check_license_file


class ConfigFrame(BasePage):
    def __init__(self, parent, app):
        super().__init__(parent, app)

        self.mock_trade_var = tk.BooleanVar()
        self.auto_start_trader_var = tk.BooleanVar()
        self.license_key_id_var = tk.StringVar()
        self.api_key_id_var = tk.StringVar()
        self.api_key_secret_var = tk.StringVar()
        self.license_key_id_entry = None
        self.api_key_id_entry = None

        self.pair_var = tk.StringVar()
        self.history_hours_var = tk.StringVar()
        self.candle_seconds_var = tk.StringVar()
        self.trend_window_hours_var = tk.StringVar()
        self.backtest_start_zar_var = tk.StringVar()
        self.trade_buy_fee_rate_var = tk.StringVar()
        self.trade_sell_fee_rate_var = tk.StringVar()
        self.current_entry_price_var = tk.StringVar()
        self.raw_score_buy_bias_var = tk.StringVar()
        self.raw_score_sell_bias_var = tk.StringVar()
        self.sell_red_body_min_diff_var = tk.StringVar()
        self.entry_min_adx_var = tk.StringVar()
        self.entry_min_hist_var = tk.StringVar()
        self.early_fail_max_profit_pct_var = tk.StringVar()
        self.min_bearish_exit_profit_pct_var = tk.StringVar()
        self.stop_loss_pct_var = tk.StringVar()
        self.min_trail_profit_pct_var = tk.StringVar()
        self.trail_giveback_ratio_var = tk.StringVar()
        self.max_buy_channel_pos_var = tk.StringVar()
        self.license_status_frame = None
        self.license_status_title = None
        self.license_status_body = None

        self.rules = []
        self.rule_table = None

        self.build_header(
            "Config Screen",
            "Edit config values and strategy rules."
        )
        self._build_ui()

    def _build_ui(self):
        main = ttk.PanedWindow(self, orient="horizontal")
        main.pack(fill="both", expand=True, padx=24, pady=10)

        left = tk.Frame(main, bg="white", bd=1, relief="solid")
        right = tk.Frame(main, bg="white", bd=1, relief="solid")
        main.add(left, weight=2)
        main.add(right, weight=3)

        self._build_settings_panel(left)
        self._build_rules_panel(right)

    def _build_settings_panel(self, parent):
        scroll = ScrollableFrame(parent)
        scroll.pack(fill="both", expand=True)
        form = scroll.inner

        tk.Label(
            form,
            text="Settings",
            bg="white",
            fg="#111827",
            font=("Segoe UI", 16, "bold")
        ).pack(anchor="w", padx=20, pady=(20, 14))

        self.license_status_frame = tk.Frame(form, bg="#eff6ff", bd=1, relief="solid")
        self.license_status_frame.pack(fill="x", padx=20, pady=(0, 14))

        self.license_status_title = tk.Label(
            self.license_status_frame,
            text="License Status",
            bg="#eff6ff",
            fg="#1d4ed8",
            font=("Segoe UI", 10, "bold"),
            justify="left",
            anchor="w",
        )
        self.license_status_title.pack(fill="x", padx=12, pady=(10, 2))

        self.license_status_body = tk.Label(
            self.license_status_frame,
            text="Checking license key...",
            bg="#eff6ff",
            fg="#1e3a8a",
            font=("Segoe UI", 10),
            justify="left",
            anchor="w",
        )
        self.license_status_body.pack(fill="x", padx=12, pady=(0, 10))

        tk.Checkbutton(
            form,
            text="Enable Mock Trade",
            variable=self.mock_trade_var,
            bg="white",
            fg="#111827",
            activebackground="white",
            font=("Segoe UI", 10),
            command=self._on_toggle_mock_trade
        ).pack(anchor="w", padx=20, pady=(0, 14))

        self.auto_start_trader_check = tk.Checkbutton(
            form,
            text="Auto-Start Trader",
            variable=self.auto_start_trader_var,
            bg="white",
            fg="#111827",
            activebackground="white",
            font=("Segoe UI", 10),
            command=self._on_toggle_auto_start_trader,
        )
        self.auto_start_trader_check.pack(anchor="w", padx=20, pady=(0, 14))

        self.license_key_id_entry = self._field(form, "License Key ID", self.license_key_id_var, state="readonly")
        self.api_key_id_entry = self._field(form, "API Key ID", self.api_key_id_var)
        self._field(form, "API Key Secret", self.api_key_secret_var, show="*")
        self._field(form, "PAIR", self.pair_var)
        self._field(form, "HISTORY_HOURS", self.history_hours_var)
        self._field(form, "CANDLE_SECONDS", self.candle_seconds_var)
        self._field(form, "TREND_WINDOW_HOURS", self.trend_window_hours_var)
        self._field(form, "BACKTEST_START_ZAR", self.backtest_start_zar_var)
        self._field(form, "TRADE_BUY_FEE_RATE", self.trade_buy_fee_rate_var)
        self._field(form, "TRADE_SELL_FEE_RATE", self.trade_sell_fee_rate_var)
        self._field(form, "CURRENT_ENTRY_PRICE", self.current_entry_price_var)
        self._field(form, "RAW_SCORE_BUY_BIAS", self.raw_score_buy_bias_var)
        self._field(form, "RAW_SCORE_SELL_BIAS", self.raw_score_sell_bias_var)
        self._field(form, "SELL_RED_BODY_MIN_DIFF", self.sell_red_body_min_diff_var)
        self._field(form, "ENTRY_MIN_ADX", self.entry_min_adx_var)
        self._field(form, "ENTRY_MIN_HIST", self.entry_min_hist_var)
        self._field(form, "EARLY_FAIL_MAX_PROFIT_PCT", self.early_fail_max_profit_pct_var)
        self._field(form, "MIN_BEARISH_EXIT_PROFIT_PCT", self.min_bearish_exit_profit_pct_var)
        self._field(form, "STOP_LOSS_PCT", self.stop_loss_pct_var)
        self._field(form, "MIN_TRAIL_PROFIT_PCT", self.min_trail_profit_pct_var)
        self._field(form, "TRAIL_GIVEBACK_RATIO", self.trail_giveback_ratio_var)
        self._field(form, "MAX_BUY_CHANNEL_POS", self.max_buy_channel_pos_var)

        buttons = tk.Frame(form, bg="white")
        buttons.pack(fill="x", padx=20, pady=(10, 20))

        tk.Button(
            buttons,
            text="Save Config",
            bg="#10b981",
            fg="white",
            activebackground="#059669",
            activeforeground="white",
            font=("Segoe UI", 10, "bold"),
            padx=18,
            pady=10,
            command=self.on_save
        ).pack(side="left", padx=(0, 10))

        tk.Button(
            buttons,
            text="Run Backtest",
            bg="#2563eb",
            fg="white",
            activebackground="#1d4ed8",
            activeforeground="white",
            font=("Segoe UI", 10, "bold"),
            padx=18,
            pady=10,
            command=self.on_run_backtest
        ).pack(side="left")

    def _field(self, parent, label, variable, show=None, state="normal"):
        tk.Label(
            parent,
            text=label,
            bg="white",
            fg="#111827",
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", padx=20, pady=(0, 6))
        entry = ttk.Entry(parent, textvariable=variable, show=show, state=state)
        entry.pack(fill="x", padx=20, pady=(0, 12))
        return entry

    def _build_rules_panel(self, parent):
        wrapper = tk.Frame(parent, bg="white")
        wrapper.pack(fill="both", expand=True, padx=16, pady=16)

        header = tk.Frame(wrapper, bg="white")
        header.pack(fill="x", pady=(0, 12))

        tk.Label(
            header,
            text="BUY / SELL Rules",
            bg="white",
            fg="#111827",
            font=("Segoe UI", 16, "bold")
        ).pack(side="left")

        actions = tk.Frame(header, bg="white")
        actions.pack(side="right")

        tk.Button(actions, text="Add Rule", bg="#2563eb", fg="white", font=("Segoe UI", 10, "bold"), command=self.add_rule).pack(side="left", padx=(0, 8))
        tk.Button(actions, text="Edit Rule", bg="#f59e0b", fg="white", font=("Segoe UI", 10, "bold"), command=self.edit_rule).pack(side="left", padx=(0, 8))
        tk.Button(actions, text="Duplicate", bg="#10b981", fg="white", font=("Segoe UI", 10, "bold"), command=self.duplicate_rule).pack(side="left", padx=(0, 8))
        tk.Button(actions, text="Delete Rule", bg="#ef4444", fg="white", font=("Segoe UI", 10, "bold"), command=self.delete_rule).pack(side="left")

        table_wrap = tk.Frame(wrapper, bg="white")
        table_wrap.pack(fill="both", expand=True)

        cols = ("active", "action", "salience", "name", "checks_summary")
        self.rule_table = ttk.Treeview(table_wrap, columns=cols, show="headings")

        self.rule_table.heading("active", text="Active", anchor="center")
        self.rule_table.heading("action", text="Action", anchor="center")
        self.rule_table.heading("salience", text="Salience", anchor="center")
        self.rule_table.heading("name", text="Name", anchor="w")
        self.rule_table.heading("checks_summary", text="Checks", anchor="w")

        self.rule_table.column("active", width=70, anchor="center", stretch=False)
        self.rule_table.column("action", width=80, anchor="center", stretch=False)
        self.rule_table.column("salience", width=80, anchor="center", stretch=False)
        self.rule_table.column("name", width=180, anchor="w", stretch=False)
        self.rule_table.column("checks_summary", width=420, anchor="w", stretch=True)

        yscroll = ttk.Scrollbar(table_wrap, orient="vertical", command=self.rule_table.yview)
        self.rule_table.configure(yscrollcommand=yscroll.set)

        self.rule_table.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")

        self.rule_table.bind("<Double-1>", lambda e: self.edit_rule())

    def _has_valid_live_trading_license(self):
        expected_key_id = self.api_key_id_var.get().strip()
        if not expected_key_id:
            return False, None, "Configured Luno API key id is required."

        valid, payload, message = check_license_file(
            license_file_path="license.key",
            public_key_path="public_key.pem",
            expected_key_id=expected_key_id,
        )
        return valid, payload, message

    def _get_license_key_id(self):
        valid, payload, _ = check_license_file(
            license_file_path="license.key",
            public_key_path="public_key.pem",
        )
        if not valid or not payload:
            return ""
        return str(payload.get("key_id", "")).strip()

    def _get_license_status_display(self):
        valid, payload, message = self._has_valid_live_trading_license()

        if not valid:
            return {
                "bg": "#fef2f2",
                "title_fg": "#b91c1c",
                "body_fg": "#7f1d1d",
                "title": "License Status: Invalid",
                "body": message,
            }

        payload = payload or {}
        customer = str(payload.get("customer", "")).strip() or "Unknown company"
        expiry_dt = self._extract_license_expiry(payload)
        expiry_text = expiry_dt.strftime("%Y-%m-%d") if expiry_dt else str(payload.get("valid_until", "Unknown"))

        body_lines = [
            f"Company: {customer}",
            f"License Key ID: {payload.get('key_id', '')}",
            f"Expires: {expiry_text}",
        ]

        if expiry_dt is not None:
            now_utc = datetime.now(timezone.utc)
            days_left = max(0, int((expiry_dt - now_utc).total_seconds() // 86400))
            if days_left <= 30:
                return {
                    "bg": "#fff7ed",
                    "title_fg": "#c2410c",
                    "body_fg": "#9a3412",
                    "title": "License Status: Warning",
                    "body": "\n".join(
                        body_lines
                        + [f"Warning: Only {days_left} day(s) left on this license."]
                    ),
                }

        return {
            "bg": "#ecfdf5",
            "title_fg": "#047857",
            "body_fg": "#065f46",
            "title": "License Status: Valid",
            "body": "\n".join(body_lines),
        }

    def _extract_license_expiry(self, payload):
        exp_value = payload.get("exp")
        if exp_value not in (None, ""):
            try:
                return datetime.fromtimestamp(float(exp_value), tz=timezone.utc)
            except Exception:
                pass

        valid_until = str(payload.get("valid_until", "")).strip()
        if valid_until:
            for candidate in [valid_until, f"{valid_until}T23:59:59+00:00"]:
                try:
                    dt_value = datetime.fromisoformat(candidate)
                    if dt_value.tzinfo is None:
                        dt_value = dt_value.replace(tzinfo=timezone.utc)
                    return dt_value.astimezone(timezone.utc)
                except Exception:
                    continue
        return None

    def _refresh_license_status(self):
        if not self.license_status_frame:
            return

        status = self._get_license_status_display()
        self.license_status_frame.config(bg=status["bg"], highlightbackground=status["bg"])
        self.license_status_title.config(
            text=status["title"],
            bg=status["bg"],
            fg=status["title_fg"],
        )
        self.license_status_body.config(
            text=status["body"],
            bg=status["bg"],
            fg=status["body_fg"],
        )

    def _on_toggle_mock_trade(self):
        if not self.mock_trade_var.get():
            valid, _, message = self._has_valid_live_trading_license()
            if not valid:
                messagebox.showerror(
                    "License Required",
                    f"A valid license is required to disable mock trading.\n\n{message}"
                )
                self.mock_trade_var.set(True)

    def _on_toggle_auto_start_trader(self):
        if not self.auto_start_trader_var.get():
            return

        valid, _, message = self._has_valid_live_trading_license()
        if not valid:
            messagebox.showerror(
                "Valid Certificate Required",
                f"A valid certificate/license is required to enable Auto-Start Trader.\n\n{message}"
            )
            self.auto_start_trader_var.set(False)

    def _sync_auto_start_trader_state(self):
        if not hasattr(self, "auto_start_trader_check"):
            return

        valid, _, _ = self._has_valid_live_trading_license()
        state = "normal" if valid else "disabled"
        self.auto_start_trader_check.config(state=state)
        if not valid:
            self.auto_start_trader_var.set(False)

    def refresh(self):
        cfg = self.app.config_data
        license_key_id = self._get_license_key_id()
        self.license_key_id_var.set(license_key_id)
        self.api_key_id_var.set(cfg.get("api_key_id", ""))

        self._refresh_license_status()
        self._sync_auto_start_trader_state()

        mock_trade = cfg.get("mock_trade", True)
        if not mock_trade:
            valid, _, _ = self._has_valid_live_trading_license()
            if not valid:
                mock_trade = True

        auto_start_trader = bool(cfg.get("auto_start_trader", False))
        valid, _, _ = self._has_valid_live_trading_license()
        if not valid:
            auto_start_trader = False

        self.mock_trade_var.set(mock_trade)
        self.auto_start_trader_var.set(auto_start_trader)
        self.api_key_secret_var.set(cfg.get("api_key_secret", ""))
        self.pair_var.set(cfg.get("pair", "XBTZAR"))

        self.history_hours_var.set(str(cfg.get("history_hours", 1000)))
        self.candle_seconds_var.set(str(cfg.get("candle_seconds", 3600)))
        self.trend_window_hours_var.set(str(cfg.get("trend_window_hours", 48)))
        self.backtest_start_zar_var.set(str(cfg.get("backtest_start_zar", 10000.0)))
        self.trade_buy_fee_rate_var.set(str(cfg.get("trade_buy_fee_rate", 0.005)))
        self.trade_sell_fee_rate_var.set(str(cfg.get("trade_sell_fee_rate", 0.003)))
        self.current_entry_price_var.set(str(cfg.get("current_entry_price", 0.0)))
        self.raw_score_buy_bias_var.set(str(cfg.get("raw_score_buy_bias", 10.0)))
        self.raw_score_sell_bias_var.set(str(cfg.get("raw_score_sell_bias", -40.0)))
        self.sell_red_body_min_diff_var.set(str(cfg.get("sell_red_body_min_diff", 10000.0)))
        self.entry_min_adx_var.set(str(cfg.get("entry_min_adx", 20.0)))
        self.entry_min_hist_var.set(str(cfg.get("entry_min_hist", 0.0)))
        self.early_fail_max_profit_pct_var.set(str(cfg.get("early_fail_max_profit_pct", 1.5)))
        self.min_bearish_exit_profit_pct_var.set(str(cfg.get("min_bearish_exit_profit_pct", 1.0)))
        self.stop_loss_pct_var.set(str(cfg.get("stop_loss_pct", -4.5)))
        self.min_trail_profit_pct_var.set(str(cfg.get("min_trail_profit_pct", 1.0)))
        self.trail_giveback_ratio_var.set(str(cfg.get("trail_giveback_ratio", 0.60)))
        self.max_buy_channel_pos_var.set(str(cfg.get("max_buy_channel_pos", 0.40)))

        self.rules = deep_copy_json(cfg.get("rules", []))
        self.refresh_rule_table()

    def refresh_rule_table(self):
        for item in self.rule_table.get_children():
            self.rule_table.delete(item)

        for idx, rule in enumerate(self.rules):
            checks = rule.get("checks", [])
            parts = []
            for c in checks:
                left = c.get("left_signal", "")
                if c.get("left_type") == "rule":
                    left = f"RULE:{c.get('left_rule', '')}"
                parts.append(f"{left} {c.get('operator', '')} {c.get('right_value', '')}")
            checks_summary = " AND ".join(parts)

            self.rule_table.insert(
                "",
                "end",
                iid=str(idx),
                values=(
                    "Yes" if rule.get("active") else "No",
                    rule.get("action", ""),
                    rule.get("salience", ""),
                    rule.get("name", ""),
                    checks_summary,
                )
            )

    def add_rule(self):
        dialog = RuleEditorDialog(self, None)
        self.wait_window(dialog)
        if dialog.result:
            self.rules.append(dialog.result)
            self.refresh_rule_table()

    def edit_rule(self):
        selected = self.rule_table.selection()
        if not selected:
            messagebox.showwarning("Select Rule", "Select a rule to edit.")
            return
        idx = int(selected[0])
        dialog = RuleEditorDialog(self, self.rules[idx])
        self.wait_window(dialog)
        if dialog.result:
            self.rules[idx] = dialog.result
            self.refresh_rule_table()

    def duplicate_rule(self):
        selected = self.rule_table.selection()
        if not selected:
            messagebox.showwarning("Select Rule", "Select a rule to duplicate.")
            return
        idx = int(selected[0])
        rule = deep_copy_json(self.rules[idx])
        rule["name"] = f"{rule.get('name', 'rule')}_copy"
        self.rules.append(rule)
        self.refresh_rule_table()

    def delete_rule(self):
        selected = self.rule_table.selection()
        if not selected:
            messagebox.showwarning("Select Rule", "Select a rule to delete.")
            return
        idx = int(selected[0])
        name = self.rules[idx].get("name", "rule")
        if messagebox.askyesno("Delete Rule", f"Delete '{name}'?"):
            self.rules.pop(idx)
            self.refresh_rule_table()

    def save_into_config(self):
        cfg = self.app.config_data

        mock_trade = bool(self.mock_trade_var.get())
        if not mock_trade:
            valid, _, message = self._has_valid_live_trading_license()
            if not valid:
                raise Exception(f"Valid license required to disable mock trading.\n\n{message}")

        auto_start_trader = bool(self.auto_start_trader_var.get())
        if auto_start_trader:
            valid, _, message = self._has_valid_live_trading_license()
            if not valid:
                raise Exception(
                    f"Valid certificate/license required to enable Auto-Start Trader.\n\n{message}"
                )

        cfg["mock_trade"] = mock_trade
        cfg["auto_start_trader"] = auto_start_trader
        cfg["api_key_id"] = self.api_key_id_var.get().strip()
        cfg["api_key_secret"] = self.api_key_secret_var.get().strip()
        cfg["pair"] = self.pair_var.get().strip() or "XBTZAR"
        cfg["history_hours"] = int(self.history_hours_var.get())
        cfg["candle_seconds"] = int(self.candle_seconds_var.get())
        cfg["trend_window_hours"] = int(self.trend_window_hours_var.get())
        cfg["backtest_start_zar"] = float(self.backtest_start_zar_var.get())
        cfg["trade_buy_fee_rate"] = float(self.trade_buy_fee_rate_var.get())
        cfg["trade_sell_fee_rate"] = float(self.trade_sell_fee_rate_var.get())
        cfg["current_entry_price"] = float(self.current_entry_price_var.get())
        cfg["raw_score_buy_bias"] = float(self.raw_score_buy_bias_var.get())
        cfg["raw_score_sell_bias"] = float(self.raw_score_sell_bias_var.get())
        cfg["sell_red_body_min_diff"] = float(self.sell_red_body_min_diff_var.get())
        cfg["entry_min_adx"] = float(self.entry_min_adx_var.get())
        cfg["entry_min_hist"] = float(self.entry_min_hist_var.get())
        cfg["early_fail_max_profit_pct"] = float(self.early_fail_max_profit_pct_var.get())
        cfg["min_bearish_exit_profit_pct"] = float(self.min_bearish_exit_profit_pct_var.get())
        cfg["stop_loss_pct"] = float(self.stop_loss_pct_var.get())
        cfg["min_trail_profit_pct"] = float(self.min_trail_profit_pct_var.get())
        cfg["trail_giveback_ratio"] = float(self.trail_giveback_ratio_var.get())
        cfg["max_buy_channel_pos"] = float(self.max_buy_channel_pos_var.get())
        cfg["rules"] = deep_copy_json(self.rules)

    def on_save(self):
        try:
            self.save_into_config()
            self.app.save_config()
            messagebox.showinfo("Saved", f"Config saved to {self.app.config_file}")
        except Exception as e:
            messagebox.showerror("Config Error", f"Invalid config values.\n\n{e}")

    def on_run_backtest(self):
        try:
            self.save_into_config()
            with open(self.app.config_file, "w", encoding="utf-8") as f:
                json.dump(self.app.config_data, f, indent=2)
            self.app.run_backtest()
        except Exception as e:
            messagebox.showerror("Config Error", f"Invalid config values.\n\n{e}")
