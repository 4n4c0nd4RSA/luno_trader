import tkinter as tk
from tkinter import ttk, messagebox

from config_defaults import SIGNAL_OPTIONS, OPERATOR_OPTIONS, RIGHT_TYPE_OPTIONS


class RuleEditorDialog(tk.Toplevel):
    def __init__(self, parent, rule=None):
        super().__init__(parent)
        self.title("Rule Editor")
        self.geometry("760x1140")
        self.configure(bg="white")
        self.result = None

        rule = rule or {
            "name": "",
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
                }
            ],
        }

        self.name_var = tk.StringVar(value=str(rule.get("name", "bullish_entry")))
        self.active_var = tk.BooleanVar(value=bool(rule.get("active", True)))
        self.action_var = tk.StringVar(value=str(rule.get("action", "BUY")))
        self.salience_var = tk.StringVar(value=str(rule.get("salience", 1)))

        self.checks = rule.get("checks", [])
        if not self.checks:
            self.checks = [
                {
                    "left_type": "signal",
                    "left_signal": "MACD",
                    "operator": ">",
                    "right_type": "number",
                    "right_value": 10.0,
                }
            ]

        self.left_type_var = tk.StringVar()
        self.left_signal_var = tk.StringVar()
        self.left_rule_var = tk.StringVar()
        self.operator_var = tk.StringVar()
        self.right_type_var = tk.StringVar()
        self.right_signal_var = tk.StringVar()
        self.right_number_var = tk.StringVar()
        self.right_bool_var = tk.StringVar()
        self.right_config_var = tk.StringVar()

        self.check_table = None
        self.left_host = None
        self.right_host = None

        self._build_ui()
        self._load_default_editor_values()
        self._refresh_left_input()
        self._refresh_right_input()
        self.refresh_check_table()

        self.transient(parent)
        self.grab_set()
        self.focus_set()

    def _build_ui(self):
        wrapper = tk.Frame(self, bg="white")
        wrapper.pack(fill="both", expand=True, padx=20, pady=20)

        tk.Label(
            wrapper,
            text="Define Rule",
            bg="white",
            fg="#111827",
            font=("Segoe UI", 16, "bold")
        ).pack(anchor="w", pady=(0, 14))

        def add_label(text):
            tk.Label(
                wrapper,
                text=text,
                bg="white",
                fg="#111827",
                font=("Segoe UI", 10, "bold")
            ).pack(anchor="w", pady=(0, 6))

        add_label("Rule Name")
        ttk.Entry(wrapper, textvariable=self.name_var).pack(fill="x", pady=(0, 12))

        tk.Checkbutton(
            wrapper,
            text="Active",
            variable=self.active_var,
            bg="white",
            fg="#111827",
            activebackground="white",
            font=("Segoe UI", 10)
        ).pack(anchor="w", pady=(0, 12))

        add_label("Action")
        ttk.Combobox(
            wrapper,
            textvariable=self.action_var,
            state="readonly",
            values=["BUY", "SELL"]
        ).pack(fill="x", pady=(0, 12))

        add_label("Salience")
        ttk.Entry(wrapper, textvariable=self.salience_var).pack(fill="x", pady=(0, 16))

        divider = tk.Frame(wrapper, bg="#e5e7eb", height=1)
        divider.pack(fill="x", pady=(0, 16))

        tk.Label(
            wrapper,
            text="Checks (all AND together)",
            bg="white",
            fg="#111827",
            font=("Segoe UI", 12, "bold")
        ).pack(anchor="w", pady=(0, 10))

        editor_card = tk.Frame(wrapper, bg="#f9fafb", bd=1, relief="solid")
        editor_card.pack(fill="x", pady=(0, 12))

        editor_inner = tk.Frame(editor_card, bg="#f9fafb")
        editor_inner.pack(fill="x", padx=12, pady=12)

        add_label_editor = lambda text: tk.Label(
            editor_inner,
            text=text,
            bg="#f9fafb",
            fg="#111827",
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", pady=(0, 6))

        add_label_editor("Left Type")
        left_type_combo = ttk.Combobox(
            editor_inner,
            textvariable=self.left_type_var,
            state="readonly",
            values=["signal", "rule"]
        )
        left_type_combo.pack(fill="x", pady=(0, 10))
        left_type_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_left_input())

        add_label_editor("Left Value")
        self.left_host = tk.Frame(editor_inner, bg="#f9fafb")
        self.left_host.pack(fill="x", pady=(0, 10))

        add_label_editor("Operator")
        ttk.Combobox(
            editor_inner,
            textvariable=self.operator_var,
            state="readonly",
            values=OPERATOR_OPTIONS
        ).pack(fill="x", pady=(0, 10))

        add_label_editor("Right Type")
        right_type_combo = ttk.Combobox(
            editor_inner,
            textvariable=self.right_type_var,
            state="readonly",
            values=RIGHT_TYPE_OPTIONS
        )
        right_type_combo.pack(fill="x", pady=(0, 10))
        right_type_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_right_input())

        add_label_editor("Right Value")
        self.right_host = tk.Frame(editor_inner, bg="#f9fafb")
        self.right_host.pack(fill="x", pady=(0, 12))

        editor_buttons = tk.Frame(editor_inner, bg="#f9fafb")
        editor_buttons.pack(fill="x")

        tk.Button(
            editor_buttons,
            text="Add Check",
            bg="#2563eb",
            fg="white",
            activebackground="#1d4ed8",
            activeforeground="white",
            font=("Segoe UI", 10, "bold"),
            padx=14,
            pady=8,
            command=self.add_check
        ).pack(side="left", padx=(0, 8))

        tk.Button(
            editor_buttons,
            text="Update Selected Check",
            bg="#10b981",
            fg="white",
            activebackground="#059669",
            activeforeground="white",
            font=("Segoe UI", 10, "bold"),
            padx=14,
            pady=8,
            command=self.update_selected_check
        ).pack(side="left")

        table_wrap = tk.Frame(wrapper, bg="white")
        table_wrap.pack(fill="both", expand=True, pady=(0, 16))

        cols = ("idx", "left_value", "operator", "right_type", "right_value")
        self.check_table = ttk.Treeview(table_wrap, columns=cols, show="headings", height=10)

        self.check_table.heading("idx", text="#", anchor="center")
        self.check_table.heading("left_value", text="Left", anchor="w")
        self.check_table.heading("operator", text="Operator", anchor="center")
        self.check_table.heading("right_type", text="Right Type", anchor="center")
        self.check_table.heading("right_value", text="Right Value", anchor="w")

        self.check_table.column("idx", width=50, anchor="center", stretch=False)
        self.check_table.column("left_value", width=240, anchor="w", stretch=False)
        self.check_table.column("operator", width=90, anchor="center", stretch=False)
        self.check_table.column("right_type", width=110, anchor="center", stretch=False)
        self.check_table.column("right_value", width=260, anchor="w", stretch=True)

        yscroll = ttk.Scrollbar(table_wrap, orient="vertical", command=self.check_table.yview)
        self.check_table.configure(yscrollcommand=yscroll.set)

        self.check_table.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")

        self.check_table.bind("<<TreeviewSelect>>", self.on_check_select)

        check_buttons = tk.Frame(wrapper, bg="white")
        check_buttons.pack(fill="x", pady=(0, 18))

        tk.Button(
            check_buttons,
            text="Remove Selected Check",
            bg="#ef4444",
            fg="white",
            activebackground="#dc2626",
            activeforeground="white",
            font=("Segoe UI", 10, "bold"),
            padx=14,
            pady=8,
            command=self.remove_selected_check
        ).pack(side="left", padx=(0, 8))

        tk.Button(
            check_buttons,
            text="Move Up",
            bg="#6b7280",
            fg="white",
            activebackground="#4b5563",
            activeforeground="white",
            font=("Segoe UI", 10, "bold"),
            padx=14,
            pady=8,
            command=lambda: self.move_selected_check(-1)
        ).pack(side="left", padx=(0, 8))

        tk.Button(
            check_buttons,
            text="Move Down",
            bg="#6b7280",
            fg="white",
            activebackground="#4b5563",
            activeforeground="white",
            font=("Segoe UI", 10, "bold"),
            padx=14,
            pady=8,
            command=lambda: self.move_selected_check(1)
        ).pack(side="left")

        bottom = tk.Frame(wrapper, bg="white")
        bottom.pack(fill="x")

        tk.Button(
            bottom,
            text="Save",
            bg="#10b981",
            fg="white",
            activebackground="#059669",
            activeforeground="white",
            font=("Segoe UI", 10, "bold"),
            padx=16,
            pady=10,
            command=self._save
        ).pack(side="left", padx=(0, 10))

        tk.Button(
            bottom,
            text="Cancel",
            bg="#6b7280",
            fg="white",
            activebackground="#4b5563",
            activeforeground="white",
            font=("Segoe UI", 10, "bold"),
            padx=16,
            pady=10,
            command=self.destroy
        ).pack(side="left")

    def _get_rule_name_options(self):
        rule_names = []
        try:
            parent_widget = self.master
            app = getattr(parent_widget, "app", None)
            if app and hasattr(app, "config_data"):
                for rule in app.config_data.get("rules", []):
                    name = rule.get("name")
                    if name:
                        rule_names.append(name)
        except Exception:
            pass
        return rule_names

    def _load_default_editor_values(self):
        first = self.checks[0]

        left_type = first.get("left_type", "signal")
        self.left_type_var.set(left_type)

        if left_type == "rule":
            self.left_rule_var.set(str(first.get("left_rule", "")))
            self.left_signal_var.set(SIGNAL_OPTIONS[0])
        else:
            self.left_signal_var.set(str(first.get("left_signal", SIGNAL_OPTIONS[0])))
            self.left_rule_var.set("")

        self.operator_var.set(str(first.get("operator", OPERATOR_OPTIONS[0])))
        self.right_type_var.set(str(first.get("right_type", "number")))

        if first.get("right_type") == "signal":
            self.right_signal_var.set(str(first.get("right_value", SIGNAL_OPTIONS[0])))
            self.right_number_var.set("0")
            self.right_bool_var.set("True")
            self.right_config_var.set("max_buy_channel_pos")
        elif first.get("right_type") == "boolean":
            self.right_signal_var.set(SIGNAL_OPTIONS[0])
            self.right_number_var.set("0")
            self.right_bool_var.set("True" if bool(first.get("right_value", True)) else "False")
            self.right_config_var.set("max_buy_channel_pos")
        elif first.get("right_type") == "config":
            self.right_signal_var.set(SIGNAL_OPTIONS[0])
            self.right_number_var.set("0")
            self.right_bool_var.set("True")
            self.right_config_var.set(str(first.get("right_value", "max_buy_channel_pos")))
        else:
            self.right_signal_var.set(SIGNAL_OPTIONS[0])
            self.right_number_var.set(str(first.get("right_value", 0.0)))
            self.right_bool_var.set("True")
            self.right_config_var.set("max_buy_channel_pos")

    def _refresh_left_input(self):
        for child in self.left_host.winfo_children():
            child.destroy()

        lt = self.left_type_var.get()

        if lt == "rule":
            ttk.Combobox(
                self.left_host,
                textvariable=self.left_rule_var,
                state="readonly",
                values=self._get_rule_name_options()
            ).pack(fill="x")
        else:
            ttk.Combobox(
                self.left_host,
                textvariable=self.left_signal_var,
                state="readonly",
                values=SIGNAL_OPTIONS
            ).pack(fill="x")

    def _refresh_right_input(self):
        for child in self.right_host.winfo_children():
            child.destroy()

        rt = self.right_type_var.get()
        if rt == "signal":
            ttk.Combobox(
                self.right_host,
                textvariable=self.right_signal_var,
                state="readonly",
                values=SIGNAL_OPTIONS
            ).pack(fill="x")
        elif rt == "number":
            ttk.Entry(self.right_host, textvariable=self.right_number_var).pack(fill="x")
        elif rt == "boolean":
            ttk.Combobox(
                self.right_host,
                textvariable=self.right_bool_var,
                state="readonly",
                values=["True", "False"]
            ).pack(fill="x")
        elif rt == "config":
            ttk.Entry(self.right_host, textvariable=self.right_config_var).pack(fill="x")

    def build_check_from_editor(self):
        left_type = self.left_type_var.get().strip()
        right_type = self.right_type_var.get().strip()

        if left_type == "rule":
            left_rule = self.left_rule_var.get().strip()
            if not left_rule:
                raise ValueError("Select a left rule name.")
        else:
            left_signal = self.left_signal_var.get().strip()
            if not left_signal:
                raise ValueError("Select a left signal.")

        if right_type == "signal":
            right_value = self.right_signal_var.get()
        elif right_type == "boolean":
            right_value = self.right_bool_var.get() == "True"
        elif right_type == "config":
            right_value = self.right_config_var.get().strip()
        else:
            right_value = float(self.right_number_var.get())

        check = {
            "left_type": left_type,
            "operator": self.operator_var.get().strip(),
            "right_type": right_type,
            "right_value": right_value,
        }

        if left_type == "rule":
            check["left_rule"] = self.left_rule_var.get().strip()
        else:
            check["left_signal"] = self.left_signal_var.get().strip()

        return check

    def refresh_check_table(self):
        for item in self.check_table.get_children():
            self.check_table.delete(item)

        for idx, check in enumerate(self.checks):
            left_display = check.get("left_signal", "")
            if check.get("left_type") == "rule":
                left_display = f"RULE:{check.get('left_rule', '')}"

            self.check_table.insert(
                "",
                "end",
                iid=str(idx),
                values=(
                    idx + 1,
                    left_display,
                    check.get("operator", ""),
                    check.get("right_type", ""),
                    str(check.get("right_value", "")),
                )
            )

    def add_check(self):
        try:
            check = self.build_check_from_editor()
            self.checks.append(check)
            self.refresh_check_table()
        except Exception as e:
            messagebox.showerror("Check Error", f"Could not add check.\n\n{e}")

    def update_selected_check(self):
        selected = self.check_table.selection()
        if not selected:
            messagebox.showwarning("Select Check", "Select a check to update.")
            return

        try:
            idx = int(selected[0])
            self.checks[idx] = self.build_check_from_editor()
            self.refresh_check_table()
            self.check_table.selection_set(str(idx))
        except Exception as e:
            messagebox.showerror("Check Error", f"Could not update check.\n\n{e}")

    def remove_selected_check(self):
        selected = self.check_table.selection()
        if not selected:
            messagebox.showwarning("Select Check", "Select a check to remove.")
            return

        idx = int(selected[0])
        if len(self.checks) <= 1:
            messagebox.showwarning("Cannot Remove", "A rule must have at least one check.")
            return

        self.checks.pop(idx)
        self.refresh_check_table()

    def move_selected_check(self, direction):
        selected = self.check_table.selection()
        if not selected:
            messagebox.showwarning("Select Check", "Select a check to move.")
            return

        idx = int(selected[0])
        new_idx = idx + direction
        if new_idx < 0 or new_idx >= len(self.checks):
            return

        self.checks[idx], self.checks[new_idx] = self.checks[new_idx], self.checks[idx]
        self.refresh_check_table()
        self.check_table.selection_set(str(new_idx))

    def on_check_select(self, event=None):
        selected = self.check_table.selection()
        if not selected:
            return

        idx = int(selected[0])
        check = self.checks[idx]

        left_type = check.get("left_type", "signal")
        self.left_type_var.set(left_type)

        if left_type == "rule":
            self.left_rule_var.set(check.get("left_rule", ""))
            self.left_signal_var.set(SIGNAL_OPTIONS[0])
        else:
            self.left_signal_var.set(str(check.get("left_signal", SIGNAL_OPTIONS[0])))
            self.left_rule_var.set("")

        self.operator_var.set(str(check.get("operator", OPERATOR_OPTIONS[0])))
        self.right_type_var.set(str(check.get("right_type", "number")))

        if check.get("right_type") == "signal":
            self.right_signal_var.set(str(check.get("right_value", SIGNAL_OPTIONS[0])))
        elif check.get("right_type") == "boolean":
            self.right_bool_var.set("True" if bool(check.get("right_value", True)) else "False")
        elif check.get("right_type") == "config":
            self.right_config_var.set(str(check.get("right_value", "max_buy_channel_pos")))
        else:
            self.right_number_var.set(str(check.get("right_value", 0.0)))

        self._refresh_left_input()
        self._refresh_right_input()

    def _save(self):
        try:
            if not self.checks:
                raise ValueError("A rule must have at least one check.")

            self.result = {
                "name": self.name_var.get().strip() or "new_rule",
                "active": bool(self.active_var.get()),
                "action": self.action_var.get().strip(),
                "salience": int(float(self.salience_var.get())),
                "checks": self.checks,
            }
            self.destroy()
        except Exception as e:
            messagebox.showerror("Rule Error", f"Could not save rule.\n\n{e}")