"""Pipeline helpers shared by main.py and tests.

Kept separate from main.py so they can be imported in headless environments
(no tkinter required).
"""
from __future__ import annotations

import numpy as np

from color_detect import crop, question_region, split_answer_regions
from ocr_engine import OCREngine


LANG_HINTS = {
    "da": [" og ", " ikke ", " hvad ", " hvilken ", " hvor ", " på ", " være ", "æ", "ø", "å"],
    "no": [" ikke ", " hva ", " hvilken ", " på ", " være ", "æ", "ø", "å"],
    "sv": [" och ", " inte ", " vad ", " vilken ", " på ", "ä", "ö"],
    "de": [" und ", " nicht ", " welche ", " ist ", "ä", "ö", "ü", "ß"],
    "fr": [" est ", " quel ", " quelle ", " pour ", " avec ", " à"],
    "es": [" qué ", " cuál ", " es ", " para ", "ñ"],
    "en": [" the ", " is ", " what ", " which ", " who ", " how "],
}


def detect_language(text: str) -> str:
    if not text:
        return "unknown"
    t = f" {text.lower()} "
    scores = {lang: sum(1 for h in hints if h in t) for lang, hints in LANG_HINTS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unknown"


def build_payload(frame: np.ndarray, ocr: OCREngine) -> dict:
    """Run OCR on every region and assemble the spec-compliant JSON payload."""
    qbox = question_region(frame)
    q_img = crop(frame, qbox)
    q_blocks = ocr.read(q_img)
    question_text = ocr.join_text(q_blocks)

    elements: list[dict] = []
    options: dict = {}

    for block in q_blocks:
        elements.append({"type": "text", "value": block["text"], "color": "white"})

    for color, box in split_answer_regions(frame).items():
        sub = crop(frame, box)
        blocks = ocr.read(sub)
        text = ocr.join_text(blocks)
        options[color] = text
        elements.append({"type": "button", "value": text, "color": color})
        for b in blocks:
            elements.append({"type": "label", "value": b["text"], "color": color})

    full_text = " | ".join(filter(None, [question_text, *options.values()]))
    return {
        "language_detected": detect_language(full_text),
        "detected_text": full_text,
        "question": question_text,
        "options": options,
        "elements": elements,
    }
