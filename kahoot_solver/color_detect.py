"""Color & UI element detection for Kahoot screens.

Kahoot uses 4 colored answer tiles in a 2x2 grid:
    Top-left:    RED      (triangle)
    Top-right:   BLUE     (diamond)
    Bottom-left: YELLOW   (circle)
    Bottom-right:GREEN    (square)
"""
from __future__ import annotations

import cv2
import numpy as np


KAHOOT_COLORS = [
    ("red",    np.array([0,   120, 80]),  np.array([10, 255, 255])),
    ("red2",   np.array([170, 120, 80]),  np.array([180, 255, 255])),
    ("blue",   np.array([100, 120, 80]),  np.array([130, 255, 255])),
    ("yellow", np.array([20,  120, 120]), np.array([35,  255, 255])),
    ("green",  np.array([40,  100, 60]),  np.array([85,  255, 255])),
]


def _mask(hsv: np.ndarray, name: str) -> np.ndarray:
    masks = [
        cv2.inRange(hsv, lo, hi)
        for n, lo, hi in KAHOOT_COLORS
        if n == name or (name == "red" and n == "red2")
    ]
    out = masks[0]
    for m in masks[1:]:
        out = cv2.bitwise_or(out, m)
    return out


def quadrant_colors(frame: np.ndarray) -> dict[str, float]:
    h, w = frame.shape[:2]
    answer_zone = frame[int(h * 0.33):, :]
    hsv = cv2.cvtColor(answer_zone, cv2.COLOR_BGR2HSV)

    qh, qw = hsv.shape[0] // 2, hsv.shape[1] // 2
    quads = {
        "top_left":     hsv[:qh, :qw],
        "top_right":    hsv[:qh, qw:],
        "bottom_left":  hsv[qh:, :qw],
        "bottom_right": hsv[qh:, qw:],
    }
    expected = {
        "top_left": "red", "top_right": "blue",
        "bottom_left": "yellow", "bottom_right": "green",
    }
    coverage: dict[str, float] = {}
    for q, region in quads.items():
        m = _mask(region, expected[q])
        coverage[q] = float(m.mean()) / 255.0
    return coverage


def is_kahoot_question(frame: np.ndarray, threshold: float = 0.18) -> bool:
    cov = quadrant_colors(frame)
    hits = sum(1 for v in cov.values() if v >= threshold)
    return hits >= 3


def split_answer_regions(frame: np.ndarray) -> dict[str, tuple[int, int, int, int]]:
    h, w = frame.shape[:2]
    y_top = int(h * 0.33)
    y_mid = (y_top + h) // 2
    x_mid = w // 2
    return {
        "red":    (0,     y_top, x_mid, y_mid),
        "blue":   (x_mid, y_top, w,     y_mid),
        "yellow": (0,     y_mid, x_mid, h),
        "green":  (x_mid, y_mid, w,     h),
    }


def question_region(frame: np.ndarray) -> tuple[int, int, int, int]:
    h, w = frame.shape[:2]
    return (0, 0, w, int(h * 0.30))


def crop(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return frame[y1:y2, x1:x2]


def dominant_text_color(image: np.ndarray) -> str:
    if image is None or image.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    best, best_score = "unknown", 0.0
    for name in ["red", "blue", "yellow", "green"]:
        m = _mask(hsv, name)
        score = float(m.mean()) / 255.0
        if score > best_score:
            best, best_score = name, score
    return best if best_score > 0.05 else "unknown"
