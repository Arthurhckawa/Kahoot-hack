"""Always-on-top tkinter overlay that highlights the correct Kahoot answer."""
from __future__ import annotations

import queue
import threading
import tkinter as tk


COLOR_HEX = {
    "red":     "#e21b3c",
    "blue":    "#1368ce",
    "yellow":  "#d89e00",
    "green":   "#26890c",
    "unknown": "#444444",
}


class AnswerOverlay:
    """Small floating window. Tk runs on the main thread; updates arrive via a queue."""

    def __init__(self):
        self._q: "queue.Queue[dict]" = queue.Queue()
        self._stop = threading.Event()
        self.root: tk.Tk | None = None

    def _build(self):
        self.root = tk.Tk()
        self.root.title("Kahoot Solver")
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.92)
        self.root.geometry("420x140+40+40")
        self.root.configure(bg="#111111")

        self.color_box = tk.Frame(self.root, bg="#444444", width=400, height=14)
        self.color_box.pack(fill="x", padx=10, pady=(10, 6))

        self.answer_label = tk.Label(
            self.root, text="Waiting for Kahoot...",
            font=("Helvetica", 16, "bold"),
            fg="#ffffff", bg="#111111", wraplength=400, justify="left",
        )
        self.answer_label.pack(fill="x", padx=10)

        self.meta_label = tk.Label(
            self.root, text="", font=("Helvetica", 10),
            fg="#aaaaaa", bg="#111111", wraplength=400, justify="left",
        )
        self.meta_label.pack(fill="x", padx=10, pady=(4, 10))

        self.root.after(100, self._poll)

    def _poll(self):
        try:
            while True:
                payload = self._q.get_nowait()
                self._render(payload)
        except queue.Empty:
            pass
        if self._stop.is_set():
            self.root.destroy()
            return
        self.root.after(120, self._poll)

    def _render(self, payload: dict):
        color = payload.get("answer_color", "unknown")
        hex_ = COLOR_HEX.get(color, "#444444")
        self.color_box.configure(bg=hex_)
        text = payload.get("answer_text") or "?"
        self.answer_label.configure(text=f"{color.upper()}  ->  {text}")
        meta = f"conf {payload.get('confidence', 0):.0%}  |  {payload.get('reasoning', '')}"
        self.meta_label.configure(text=meta[:160])

    def update(self, payload: dict):
        self._q.put(payload)

    def run(self):
        self._build()
        self.root.mainloop()

    def stop(self):
        self._stop.set()
