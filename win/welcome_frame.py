import tkinter as tk
from base_page import BasePage


class WelcomeFrame(BasePage):
    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.build_header(
            "StartUp / Welcome Help",
            "Guide the user through API setup, config, rules, and backtesting."
        )

        outer = tk.Frame(self, bg="#f4f6f8")
        outer.pack(fill="both", expand=True, padx=24, pady=10)

        card = tk.Frame(outer, bg="white", bd=1, relief="solid")
        card.pack(fill="both", expand=True)

        text = tk.Text(
            card,
            wrap="word",
            bg="white",
            fg="#111827",
            relief="flat",
            font=("Segoe UI", 11),
            padx=20,
            pady=20
        )
        text.pack(fill="both", expand=True)

        guide = """
Welcome to the Luno Trader.

Screens:
1. Welcome
2. Config
3. Backtest Results
4. Dashboard

What is supported:
- JSON-backed config
- flat rules with multiple checks ANDed together
- signal-to-signal, signal-to-number, signal-to-config, and rule-to-boolean checks
- real indicator pipeline
- current signal computation
- mock backtest from real Luno candle data
- price graph with buy/sell markers
- bank graph
        """.strip()

        text.insert("1.0", guide)
        text.config(state="disabled")

        buttons = tk.Frame(outer, bg="#f4f6f8")
        buttons.pack(fill="x", pady=16)

        tk.Button(
            buttons,
            text="Go to Config",
            bg="#2563eb",
            fg="white",
            activebackground="#1d4ed8",
            activeforeground="white",
            font=("Segoe UI", 10, "bold"),
            padx=18,
            pady=10,
            command=lambda: self.app.show_frame("config")
        ).pack(side="left", padx=(0, 10))

        tk.Button(
            buttons,
            text="Run Backtest",
            bg="#f59e0b",
            fg="white",
            activebackground="#d97706",
            activeforeground="white",
            font=("Segoe UI", 10, "bold"),
            padx=18,
            pady=10,
            command=self.app.run_backtest
        ).pack(side="left", padx=(0, 10))

        tk.Button(
            buttons,
            text="Go to Dashboard",
            bg="#1dd845",
            fg="white",
            activebackground="#19b33a",
            activeforeground="white",
            font=("Segoe UI", 10, "bold"),
            padx=18,
            pady=10,
            command=lambda: self.app.show_frame("dashboard")
        ).pack(side="left")