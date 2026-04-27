"""Dynamic detection of Kahoot's 4 colored answer tiles."""
from __future__ import annotations

import cv2
import numpy as np


COLOR_RANGES = {
    "red":    [(np.array([0,   100, 70]),  np.array([10,  255, 255])),
               (np.array([170, 100, 70]),  np.array([180, 255, 255]))],
    "blue":   [(np.array([95,  120, 70]),  np.array([130, 255, 255]))],
    "yellow": [(np.array([18,  120, 120]), np.array([35,  255, 255]))],
    "green":  [(np.array([40,  90,  50]),  np.array([85,  255, 255]))],
}


def color_mask(hsv, color):
    ranges = COLOR_RANGES[color]
    mask = cv2.inRange(hsv, *ranges[0])
    for lo, hi in ranges[1:]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    return mask


def largest_blob(mask, min_area_frac=0.01):
    h, w = mask.shape
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    area = stats[idx, cv2.CC_STAT_AREA]
    if area < min_area_frac * h * w:
        return None
    x = stats[idx, cv2.CC_STAT_LEFT]
    y = stats[idx, cv2.CC_STAT_TOP]
    bw = stats[idx, cv2.CC_STAT_WIDTH]
    bh = stats[idx, cv2.CC_STAT_HEIGHT]
    return (int(x), int(y), int(x + bw), int(y + bh))


def find_tiles(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    found = {}
    for color in ("red", "blue", "yellow", "green"):
        bbox = largest_blob(color_mask(hsv, color))
        if bbox is not None:
            found[color] = bbox
    return found


def is_kahoot_question(frame, threshold=4):
    return len(find_tiles(frame)) >= threshold


def split_answer_regions(frame):
    return find_tiles(frame)


def question_region(frame, tiles=None):
    h, w = frame.shape[:2]
    if tiles is None:
        tiles = find_tiles(frame)
    if not tiles:
        return (0, 0, w, int(h * 0.30))
    top_of_tiles = min(b[1] for b in tiles.values())
    bottom = max(0, top_of_tiles - 10)
    top = max(0, bottom - 280)
    return (0, top, w, bottom)


def crop(frame, box):
    x1, y1, x2, y2 = box
    return frame[y1:y2, x1:x2]


def quadrant_colors(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return {c: float(color_mask(hsv, c).mean()) / 255.0
            for c in ("red", "blue", "yellow", "green")}


def dominant_text_color(image):
    if image is None or image.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    best, best_score = "unknown", 0.0
    for name in ("red", "blue", "yellow", "green"):
        score = float(color_mask(hsv, name).mean()) / 255.0
        if score > best_score:
            best, best_score = name, score
    return best if best_score > 0.05 else "unknown"