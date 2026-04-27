"""Real-time Kahoot solver - vision only."""
from __future__ import annotations

import hashlib, json, os, queue, threading, time
from pathlib import Path
from dotenv import load_dotenv
from overlay import AnswerOverlay


load_dotenv(Path(__file__).parent / ".env")
EMERGENT_LLM_KEY = os.environ.get("EMERGENT_LLM_KEY", "")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic")
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-haiku-4-5-20251001")
SCAN_INTERVAL = float(os.environ.get("SCAN_INTERVAL", "0.2"))


class SolverWorker(threading.Thread):
    def __init__(self, solver, overlay):
        super().__init__(daemon=True)
        self.solver = solver
        self.overlay = overlay
        self.in_q = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._last_hash = ""

    def submit(self, frame_hash, image):
        if frame_hash == self._last_hash:
            return
        self._last_hash = frame_hash
        try:
            self.in_q.put_nowait(image)
        except queue.Full:
            pass

    def stop(self):
        self._stop.set()

    def run(self):
        while not self._stop.is_set():
            try:
                image = self.in_q.get(timeout=0.3)
            except queue.Empty:
                continue
            t0 = time.time()
            try:
                answer = self.solver.ask(image)
            except Exception as e:
                answer = {"question": "", "answer_color": "unknown",
                          "answer_text": f"error: {e}", "confidence": 0.0,
                          "reasoning": ""}
            answer["latency_s"] = round(time.time() - t0, 2)
            self.overlay.update(answer)
            print(json.dumps(answer, ensure_ascii=False)[:500])


def hash_tiles(frame, tiles, cv2):
    top = min(b[1] for b in tiles.values())
    bottom = max(b[3] for b in tiles.values())
    left = min(b[0] for b in tiles.values())
    right = max(b[2] for b in tiles.values())
    crop = frame[top:bottom, left:right]
    small = cv2.resize(crop, (64, 32))
    return hashlib.md5(small.tobytes()).hexdigest()


def capture_loop(worker, overlay, find_tiles, ScreenGrabber, cv2):
    grabber = ScreenGrabber()
    print("[capture] running - press Ctrl+C to quit")
    try:
        while True:
            frame = grabber.grab()
            tiles = find_tiles(frame)
            if len(tiles) >= 2 and "red" in tiles:
                bottom = max(b[3] for b in tiles.values())
                top = max(0, min(b[1] for b in tiles.values()) - 600)
                left = max(0, min(b[0] for b in tiles.values()) - 40)
                right = min(frame.shape[1], max(b[2] for b in tiles.values()) + 40)
                crop = frame[top:bottom, left:right]
                h = hash_tiles(frame, tiles, cv2)
                worker.submit(h, crop)
            time.sleep(SCAN_INTERVAL)
    except KeyboardInterrupt:
        pass
    finally:
        grabber.close()
        worker.stop()
        overlay.stop()


def init_and_run(overlay):
    if not EMERGENT_LLM_KEY:
        overlay.update_status("ERROR: EMERGENT_LLM_KEY missing in .env")
        return

    overlay.start_phase("Loading OpenCV", 2)
    import cv2
    import numpy  # noqa
    overlay.end_phase()

    overlay.start_phase("Loading LiteLLM", 4, "Connecting to Claude...")
    from solver import Solver
    solver = Solver(EMERGENT_LLM_KEY, LLM_PROVIDER, LLM_MODEL)
    overlay.end_phase()

    overlay.start_phase("Starting screen capture", 1, f"Model: {LLM_MODEL}")
    from capture import ScreenGrabber
    from color_detect import find_tiles
    worker = SolverWorker(solver, overlay)
    worker.start()
    overlay.end_phase()

    overlay.set_ready()
    capture_loop(worker, overlay, find_tiles, ScreenGrabber, cv2)


def main():
    overlay = AnswerOverlay()
    threading.Thread(target=init_and_run, args=(overlay,), daemon=True).start()
    overlay.run()


if __name__ == "__main__":
    main()