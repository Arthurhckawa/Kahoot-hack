"""Multilingual OCR using EasyOCR.

EasyOCR supports many languages out of the box (en, da, no, sv, de, fr, es, ...)
which makes it a natural fit for the multilingual requirement.
"""
from __future__ import annotations

import easyocr
import numpy as np


class OCREngine:
    def __init__(self, languages: list[str] | None = None, gpu: bool = False):
        self.languages = languages or ["en", "da"]
        self.reader = easyocr.Reader(self.languages, gpu=gpu, verbose=False)

    def read(self, image: np.ndarray) -> list[dict]:
        """Returns a list of {bbox, text, confidence} dicts."""
        if image is None or image.size == 0:
            return []
        try:
            raw = self.reader.readtext(image, detail=1, paragraph=False)
        except Exception:
            return []
        results: list[dict] = []
        for bbox, text, conf in raw:
            text = (text or "").strip()
            if not text:
                continue
            xs = [int(p[0]) for p in bbox]
            ys = [int(p[1]) for p in bbox]
            results.append(
                {
                    "text": text,
                    "confidence": float(conf),
                    "bbox": (min(xs), min(ys), max(xs), max(ys)),
                }
            )
        return results

    @staticmethod
    def join_text(blocks: list[dict]) -> str:
        return " ".join(b["text"] for b in blocks).strip()
