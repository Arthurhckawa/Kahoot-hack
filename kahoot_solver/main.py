"""Real-time Kahoot solver - entry point.

Run:  python main.py
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

from capture import ScreenGrabber
from color_detect import is_kahoot_question
from ocr_engine import OCREngine
from overlay import AnswerOverlay
from pipeline import build_payload
from solver import Solver


load_dotenv(Path(__file__).parent / ".env")

EMERGENT_LLM_KEY = os.environ.get("EMERGENT_LLM_KEY", "")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic")
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-5-20250929")
OCR_LANGUAGES = [s.strip() for s in os.environ.get("OCR_LANGUAGES", "en,da").split(",") if s.strip()]
SCAN_INTERVAL = float(os.environ.get("SCAN_INTERVAL", "0.4"))


class SolverWorker(threading.Thread):
    """Pulls payloads from a queue, calls the LLM, pushes results to overlay."""

    def __init__(self, solver: Solver, overlay: AnswerOverlay):
        super().__init__(daemon=True)
        self.solver = solver
        self.overlay = overlay
        self.in_q: "queue.Queue[dict]" = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._last_signature = ""

    def submit(self, payload: dict):
        sig = (payload.get("question", "") + "|" +
               "|".join(payload.get("options", {}).values())).strip()
        if sig == self._last_signature or len(sig) < 8:
            return
        self._last_signature = sig
        try:
            self.in_q.put_nowait(payload)
        except queue.Full:
            pass

    def stop(self):
        self._stop.set()

    def run(self):
        while not self._stop.is_set():
            try:
                payload = self.in_q.get(timeout=0.3)
            except queue.Empty:
                continue
            t0 = time.time()
            try:
                answer = self.solver.ask(payload["question"], payload["options"])
            except Exception as e:
                answer = {"answer_color": "unknown", "answer_text": f"error: {e}",
                          "confidence": 0.0, "reasoning": ""}
            answer["latency_s"] = round(time.time() - t0, 2)
            self.overlay.update(answer)

            print(json.dumps(
                {"language_detected": payload["language_detected"],
                 "detected_text": payload["detected_text"],
                 "elements": payload["elements"],
                 "answer": answer},
                ensure_ascii=False)[:1500])


def capture_loop(worker: SolverWorker, ocr: OCREngine, overlay: AnswerOverlay):
    grabber = ScreenGrabber()
    print("[capture] running - press Ctrl+C in terminal to quit")
    try:
        while True:
            frame = grabber.grab()
            if is_kahoot_question(frame):
                payload = build_payload(frame, ocr)
                if payload["question"]:
                    worker.submit(payload)
            time.sleep(SCAN_INTERVAL)
    except KeyboardInterrupt:
        pass
    finally:
        grabber.close()
        worker.stop()
        overlay.stop()


def main():
    if not EMERGENT_LLM_KEY:
        print("ERROR: EMERGENT_LLM_KEY missing. Copy .env.example -> .env and add your key.")
        return

    print("[init] loading EasyOCR (this may take a minute the first time)...")
    ocr = OCREngine(OCR_LANGUAGES, gpu=False)

    solver = Solver(EMERGENT_LLM_KEY, LLM_PROVIDER, LLM_MODEL)
    overlay = AnswerOverlay()
    worker = SolverWorker(solver, overlay)
    worker.start()

    t = threading.Thread(target=capture_loop, args=(worker, ocr, overlay), daemon=True)
    t.start()

    overlay.run()  # tk must run on the main thread


if __name__ == "__main__":
    main()
