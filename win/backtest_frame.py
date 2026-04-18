import tkinter as tk
from tkinter import ttk

from base_page import BasePage

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except Exception:
    FigureCanvasTkAgg = None
    Figure = None


class BacktestFrame(BasePage):
    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.summary_labels = {}
        self.table = None
        self.price_chart_host = None
        self.bank_chart_host = None

        self.build_header("Backtest Results", "Audit log, price graph, and bank graph.")
        self._build_ui()

    def _build_ui(self):
        actions = tk.Frame(self, bg="#f4f6f8")
        actions.pack(fill="x", padx=24, pady=(0, 10))

        tk.Button(
            actions,
            text="Run Backtest",
            bg="#2563eb",
            fg="white",
            activebackground="#1d4ed8",
            activeforeground="white",
            font=("Segoe UI", 10, "bold"),
            padx=18,
            pady=10,
            command=self.app.run_backtest
        ).pack(side="left")

        summary = tk.Frame(self, bg="#f4f6f8")
        summary.pack(fill="x", padx=24, pady=(0, 10))

        for key in ["Start Bank", "End Bank", "PnL", "Trade Count", "Active Rules", "Mode", "Current Signal"]:
            card = tk.Frame(summary, bg="white", bd=1, relief="solid")
            card.pack(side="left", fill="x", expand=True, padx=6)

            tk.Label(
                card,
                text=key,
                bg="white",
                fg="#6b7280",
                font=("Segoe UI", 9, "bold")
            ).pack(anchor="w", padx=12, pady=(10, 4))

            value = tk.Label(
                card,
                text="-",
                bg="white",
                fg="#111827",
                font=("Segoe UI", 14, "bold")
            )
            value.pack(anchor="w", padx=12, pady=(0, 12))
            self.summary_labels[key] = value

        body = ttk.PanedWindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=24, pady=(0, 20))

        left = tk.Frame(body, bg="white", bd=1, relief="solid")
        right = ttk.PanedWindow(body, orient="vertical")

        body.add(left, weight=2)
        body.add(right, weight=3)

        table_wrap = tk.Frame(left, bg="white")
        table_wrap.pack(fill="both", expand=True, padx=12, pady=12)

        tk.Label(
            table_wrap,
            text="Audit Log Table",
            bg="white",
            fg="#111827",
            font=("Segoe UI", 12, "bold")
        ).pack(anchor="w", pady=(0, 10))

        cols = ("time", "action", "price", "amount_zar", "btc", "cash_after", "bank_value", "mode", "reason")
        self.table = ttk.Treeview(table_wrap, columns=cols, show="headings", height=18)

        for c in cols:
            self.table.heading(c, text=c)
            self.table.column(c, width=120, anchor="center")

        vsb = ttk.Scrollbar(table_wrap, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=vsb.set)

        self.table.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        price_panel = tk.Frame(right, bg="white", bd=1, relief="solid")
        bank_panel = tk.Frame(right, bg="white", bd=1, relief="solid")
        right.add(price_panel, weight=1)
        right.add(bank_panel, weight=1)

        tk.Label(
            price_panel,
            text="Price Over Time With Buy/Sell Events",
            bg="white",
            fg="#111827",
            font=("Segoe UI", 12, "bold")
        ).pack(anchor="w", padx=12, pady=(12, 0))

        self.price_chart_host = tk.Frame(price_panel, bg="white")
        self.price_chart_host.pack(fill="both", expand=True, padx=10, pady=10)

        tk.Label(
            bank_panel,
            text="Bank Value Over Time",
            bg="white",
            fg="#111827",
            font=("Segoe UI", 12, "bold")
        ).pack(anchor="w", padx=12, pady=(12, 0))

        self.bank_chart_host = tk.Frame(bank_panel, bg="white")
        self.bank_chart_host.pack(fill="both", expand=True, padx=10, pady=10)

    def refresh(self):
        result = self.app.backtest_result
        self._clear_table()
        self._clear_frame(self.price_chart_host)
        self._clear_frame(self.bank_chart_host)

        if not result:
            for key in self.summary_labels:
                self.summary_labels[key].config(text="-")
            self._render_no_data(self.price_chart_host, "Run a backtest to see the price graph.")
            self._render_no_data(self.bank_chart_host, "Run a backtest to see the bank graph.")
            return

        summary = result["summary"]
        self.summary_labels["Start Bank"].config(text=f"R {summary['start_bank']:.2f}")
        self.summary_labels["End Bank"].config(text=f"R {summary['end_bank']:.2f}")
        self.summary_labels["PnL"].config(text=f"R {summary['pnl']:.2f}")
        self.summary_labels["Trade Count"].config(text=str(summary["trade_count"]))
        self.summary_labels["Active Rules"].config(text=str(summary["rule_count"]))
        self.summary_labels["Mode"].config(text="MOCK" if summary["mock_trade"] else "LIVE")
        self.summary_labels["Current Signal"].config(text=str(summary.get("current_signal_action", "-")))

        for row in result["audit_rows"]:
            self.table.insert(
                "",
                "end",
                values=(
                    row["time"],
                    row["action"],
                    row["price"],
                    row["amount_zar"],
                    row["btc"],
                    row["cash_after"],
                    row["bank_value"],
                    row["mode"],
                    row.get("reason", "")
                )
            )

        self._draw_price_chart(result)
        self._draw_bank_chart(result)

    def _draw_price_chart(self, result):
        if not FigureCanvasTkAgg or not Figure:
            self._render_no_data(self.price_chart_host, "matplotlib not installed. Run: pip install matplotlib")
            return

        fig = Figure(figsize=(7, 3.5), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(result["times"], result["prices"], linewidth=1.6, color="#a7a7a7")

        buy_x = [e["time"] for e in result["events"] if e["type"] == "BUY"]
        buy_y = [e["price"] for e in result["events"] if e["type"] == "BUY"]
        sell_x = [e["time"] for e in result["events"] if e["type"] == "SELL"]
        sell_y = [e["price"] for e in result["events"] if e["type"] == "SELL"]

        if buy_x:
            ax.scatter(buy_x, buy_y, marker="^", s=70, label="BUY")
        if sell_x:
            ax.scatter(sell_x, sell_y, marker="v", s=70, label="SELL")

        ax.set_title("Price")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        ax.grid(True, alpha=0.3)
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=self.price_chart_host)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _draw_bank_chart(self, result):
        if not FigureCanvasTkAgg or not Figure:
            self._render_no_data(self.bank_chart_host, "matplotlib not installed. Run: pip install matplotlib")
            return

        fig = Figure(figsize=(7, 3.5), dpi=100)
        ax = fig.add_subplot(111)
        bank_times = result.get("bank_times", result["times"])
        ax.plot(bank_times, result["bank_values"], linewidth=1.6)
        ax.set_title("Bank Value")
        ax.set_xlabel("Time")
        ax.set_ylabel("Bank")
        ax.grid(True, alpha=0.3)

        canvas = FigureCanvasTkAgg(fig, master=self.bank_chart_host)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _render_no_data(self, host, text):
        tk.Label(host, text=text, bg="white", fg="#6b7280", font=("Segoe UI", 11)).pack(expand=True)

    def _clear_table(self):
        for item in self.table.get_children():
            self.table.delete(item)

    def _clear_frame(self, frame):
        for child in frame.winfo_children():
            child.destroy()
