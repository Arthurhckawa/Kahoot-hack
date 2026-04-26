"""Offline smoke test - runs the pipeline against a synthetic Kahoot frame.

Run:  python test_offline.py            # offline pipeline only
      python test_offline.py --llm      # also call the LLM
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

from color_detect import is_kahoot_question
from pipeline import build_payload, detect_language
from ocr_engine import OCREngine


def synth_frame() -> np.ndarray:
    h, w = 720, 1280
    frame = np.full((h, w, 3), 30, dtype=np.uint8)

    cv2.putText(frame, "Hvilken farve har himlen?", (60, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)

    y_top = int(h * 0.33)
    y_mid = (y_top + h) // 2
    x_mid = w // 2

    cv2.rectangle(frame, (0, y_top),     (x_mid, y_mid), (60, 27, 226), -1)   # red
    cv2.rectangle(frame, (x_mid, y_top), (w, y_mid),     (206, 104, 19), -1)  # blue
    cv2.rectangle(frame, (0, y_mid),     (x_mid, h),     (0, 158, 216), -1)   # yellow
    cv2.rectangle(frame, (x_mid, y_mid), (w, h),         (12, 137, 38), -1)   # green

    labels = {
        (60,         y_top + 80): "Roed",
        (x_mid + 60, y_top + 80): "Blaa",
        (60,         y_mid + 80): "Gul",
        (x_mid + 60, y_mid + 80): "Groen",
    }
    for (x, y), txt in labels.items():
        cv2.putText(frame, txt, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="also call the LLM solver")
    args = ap.parse_args()

    load_dotenv(Path(__file__).parent / ".env")
    frame = synth_frame()
    print("[test] is_kahoot_question:", is_kahoot_question(frame))

    ocr = OCREngine(["en", "da"], gpu=False)
    payload = build_payload(frame, ocr)
    print("[test] language:", payload["language_detected"])
    print("[test] question:", payload["question"])
    print("[test] options:", payload["options"])

    out = {
        "language_detected": payload["language_detected"],
        "detected_text": payload["detected_text"],
        "elements": payload["elements"],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2)[:1500])

    assert detect_language("Hvilken farve har himlen og hvad er på den") == "da"

    if args.llm:
        from solver import Solver
        key = os.environ.get("EMERGENT_LLM_KEY", "")
        if not key:
            print("[test] skipping LLM: EMERGENT_LLM_KEY not set")
            return
        solver = Solver(key,
                        os.environ.get("LLM_PROVIDER", "anthropic"),
                        os.environ.get("LLM_MODEL", "claude-sonnet-4-5-20250929"))
        ans = solver.ask(payload["question"] or "Hvilken farve har himlen?",
                         payload["options"])
        print("[test] llm answer:", ans)


if __name__ == "__main__":
    main()
