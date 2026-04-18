import tkinter as tk


class BasePage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg="#f4f6f8")
        self.app = app

    def build_header(self, title, subtitle=""):
        header = tk.Frame(self, bg="#f4f6f8")
        header.pack(fill="x", padx=24, pady=(20, 10))

        tk.Label(
            header,
            text=title,
            bg="#f4f6f8",
            fg="#111827",
            font=("Segoe UI", 22, "bold")
        ).pack(anchor="w")

        if subtitle:
            tk.Label(
                header,
                text=subtitle,
                bg="#f4f6f8",
                fg="#6b7280",
                font=("Segoe UI", 10)
            ).pack(anchor="w", pady=(6, 0))