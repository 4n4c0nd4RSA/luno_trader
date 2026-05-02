import tkinter as tk
from tkinter import ttk


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        canvas = tk.Canvas(self, bg="white", highlightthickness=0)
        vscroll = tk.Scrollbar(
            self,
            orient="vertical",
            command=canvas.yview,
            width=16,
            bg="#e5e7eb",
            troughcolor="#f3f4f6",
            activebackground="#9ca3af",
            relief="flat",
            bd=0,
        )

        self.inner = tk.Frame(canvas, bg="white")
        self.inner.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        window_id = canvas.create_window((0, 0), window=self.inner, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window_id, width=e.width))
        canvas.configure(yscrollcommand=vscroll.set)

        canvas.pack(side="left", fill="both", expand=True)
        vscroll.pack(side="right", fill="y")

        self.canvas = canvas
        self.vscroll = vscroll

        canvas.bind("<Enter>", self._bind_mousewheel)
        canvas.bind("<Leave>", self._unbind_mousewheel)

    def _bind_mousewheel(self, _event):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, _event):
        self.canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
