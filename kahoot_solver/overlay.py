"""Loading overlay with animated progress bar + answer overlay."""
from __future__ import annotations

import queue
import threading
import time
import tkinter as tk


COLOR_HEX = {
    "red":     "#e21b3c",
    "blue":    "#1368ce",
    "yellow":  "#d89e00",
    "green":   "#26890c",
    "unknown": "#444444",
    "loading": "#7c3aed",
    "ready":   "#16a34a",
}

WIDTH = 440
BAR_WIDTH = 416


class AnswerOverlay:
    def __init__(self):
        self._q = queue.Queue()
        self._stop = threading.Event()
        self.root = None

        self._phase_name = "Initializing"
        self._phase_start = time.time()
        self._phase_duration = 1.0
        self._phase_active = True
        self._phase_percent = 0.0
        self._mode = "loading"

    def _build(self):
        self.root = tk.Tk()
        self.root.title("Kahoot Solver")
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.94)
        self.root.geometry(f"{WIDTH}x190+40+40")
        self.root.configure(bg="#111111")

        self.title_label = tk.Label(
            self.root, text="LOADING",
            font=("Helvetica", 11, "bold"),
            fg="#7c3aed", bg="#111111",
        )
        self.title_label.pack(anchor="w", padx=12, pady=(10, 0))

        self.phase_label = tk.Label(
            self.root, text=self._phase_name,
            font=("Helvetica", 16, "bold"),
            fg="#ffffff", bg="#111111", wraplength=WIDTH - 20, justify="left",
        )
        self.phase_label.pack(fill="x", padx=12, pady=(2, 8))

        # Progress bar: outer dark frame + inner colored frame
        self.bar_bg = tk.Frame(self.root, bg="#222222", width=BAR_WIDTH, height=14)
        self.bar_bg.pack(padx=12, pady=(0, 4))
        self.bar_bg.pack_propagate(False)

        self.bar_fill = tk.Frame(self.bar_bg, bg=COLOR_HEX["loading"], width=0, height=14)
        self.bar_fill.place(x=0, y=0)

        self.percent_label = tk.Label(
            self.root, text="0 %", font=("Helvetica", 10, "bold"),
            fg="#cccccc", bg="#111111",
        )
        self.percent_label.pack(anchor="w", padx=12)

        self.detail_label = tk.Label(
            self.root, text="", font=("Helvetica", 9),
            fg="#888888", bg="#111111", wraplength=WIDTH - 20, justify="left",
        )
        self.detail_label.pack(fill="x", padx=12, pady=(2, 10))

        self.root.after(100, self._poll)
        self.root.after(50, self._tick)

    def _tick(self):
        """Smoothly animate progress towards 95% over the phase duration."""
        if self._mode == "loading" and self._phase_active:
            elapsed = time.time() - self._phase_start
            target = min(95.0, (elapsed / max(0.1, self._phase_duration)) * 95.0)
            if target > self._phase_percent:
                # ease towards target
                self._phase_percent += (target - self._phase_percent) * 0.25
            self._render_progress(self._phase_percent)
        if self._stop.is_set():
            return
        self.root.after(50, self._tick)

    def _render_progress(self, percent, color=None):
        percent = max(0.0, min(100.0, percent))
        w = int(BAR_WIDTH * percent / 100.0)
        self.bar_fill.configure(width=max(1, w))
        if color:
            self.bar_fill.configure(bg=color)
        self.percent_label.configure(text=f"{int(percent)} %")

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

    def _render(self, payload):
        kind = payload.get("kind", "answer")
        if kind == "phase":
            self._mode = "loading"
            self._phase_name = payload.get("name", "Loading")
            self._phase_duration = float(payload.get("duration", 5.0))
            self._phase_start = time.time()
            self._phase_percent = 0.0
            self._phase_active = True
            self.title_label.configure(text="LOADING", fg="#7c3aed")
            self.phase_label.configure(text=self._phase_name)
            self.detail_label.configure(text=payload.get("detail", ""))
            self._render_progress(0, COLOR_HEX["loading"])
        elif kind == "phase_done":
            self._phase_active = False
            self._phase_percent = 100.0
            self._render_progress(100, COLOR_HEX["loading"])
        elif kind == "ready":
            self._mode = "ready"
            self._phase_active = False
            self.title_label.configure(text="READY", fg=COLOR_HEX["ready"])
            self.phase_label.configure(text="Waiting for Kahoot question...")
            self._render_progress(100, COLOR_HEX["ready"])
            self.detail_label.configure(text="Open a Kahoot quiz on this screen.")
        else:
            # answer
            self._mode = "answer"
            color = payload.get("answer_color", "unknown")
            hex_ = COLOR_HEX.get(color, "#444444")
            self.title_label.configure(text=color.upper(), fg=hex_)
            text = payload.get("answer_text") or "?"
            self.phase_label.configure(text=text)
            self._render_progress(100, hex_)
            conf = payload.get("confidence", 0)
            self.percent_label.configure(text=f"conf {int(conf*100)} %")
            meta = (f"{payload.get('latency_s', 0)}s  |  "
                    f"{payload.get('reasoning', '')}")
            self.detail_label.configure(text=meta[:200])

    def update(self, payload):
        payload = dict(payload)
        payload.setdefault("kind", "answer")
        self._q.put(payload)

    def start_phase(self, name, estimated_seconds, detail=""):
        self._q.put({"kind": "phase", "name": name,
                     "duration": estimated_seconds, "detail": detail})

    def end_phase(self):
        self._q.put({"kind": "phase_done"})

    def set_ready(self):
        self._q.put({"kind": "ready"})

    def update_status(self, message, detail=""):
        # legacy compatibility
        self._q.put({"kind": "phase", "name": message,
                     "duration": 3.0, "detail": detail})

    def run(self):
        self._build()
        self.root.mainloop()

    def stop(self):
        self._stop.set()