import tkinter as tk
from base_page import BasePage


class WelcomeFrame(BasePage):
    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.build_header(
            "Welcome",
            "Set up your trader, test your strategy safely, and monitor everything in one place."
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
Welcome to Trader for Luno.

This app helps you configure your trading setup, test your rules on historical market data, and monitor trader activity from a single workspace.

A simple way to get started:

1. Open Config
   Add your API keys, choose your market pair, and review the strategy settings.

2. Review your rules
   Create or adjust BUY and SELL rules using signals, numbers, config values, and rule-based checks.

3. Run a backtest
   Test your setup against recent Luno candle data before using it in live trading.

4. Open the Dashboard
   Start or stop the trader, review audit history, and track signals, prices, and bank value.

Helpful notes:
- Mock Trade is the safest way to test behavior before placing real trades.
- Live trading requires a valid license/certificate.
- Backtest Results help you understand how your rules would have behaved.
- The Dashboard gives you a quick operational view once you are ready to run.
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
