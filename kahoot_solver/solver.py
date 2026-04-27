"""Vision-only Kahoot solver."""
from __future__ import annotations
import base64, json, re
import cv2, litellm


SYSTEM_PROMPT = (
    "Kahoot question. The screen can show 2, 3, or 4 colored answer tiles. "
    "Layouts: "
    "(a) 4 tiles = multi-choice: RED top-left, BLUE top-right, YELLOW bottom-left, GREEN bottom-right. "
    "(b) 3 tiles = multi-choice with 3 options: usually RED, BLUE, YELLOW. "
    "(c) 2 tiles = either True/False (RED=True/Sandt, BLUE=False/Falsk) or 2-option multi-choice. "
    "Look at the screenshot to count how many tiles are actually visible and "
    "ONLY answer with a color that has a tile. Never pick a color that isn't shown. "
    "Some questions allow MULTIPLE correct answers (multi-select) - if the question "
    "says 'choose all that apply' or similar, list all correct colors comma-separated "
    "in answer_color (e.g. 'red,yellow'). "
    "Read the question and the 4 options DIRECTLY from the image (ignore "
    "browser tabs, terminals, taskbar, or anything outside the centered Kahoot "
    "card). If there is a picture inside the Kahoot card, use it as a hint. "
    "Pick the correct answer. IMPORTANT: If 2+ options seem equally valid AND "
    "one option is 'All of the answers are valid' / 'All of the above' / "
    "similar - that meta-option is almost always correct. Likewise prefer "
    "'None of the above' when no concrete option fits. "
    "Respond with ONLY a JSON object on one line, no markdown:\n"
    '{"question": "...", "answer_color": "red|blue|yellow|green", '
    '"answer_text": "...", "confidence": 0.0-1.0, "reasoning": "..."}'
)


def _encode_image(image_np):
    _, buf = cv2.imencode(".png", image_np)
    return base64.b64encode(buf).decode("utf-8")


class Solver:
    def __init__(self, api_key, provider="anthropic", model="claude-haiku-4-5-20251001"):
        if not api_key:
            raise RuntimeError("EMERGENT_LLM_KEY missing")
        self.api_key = api_key
        self.provider = provider
        self.model = model

    def ask(self, image):
        b64 = _encode_image(image)
        content = [
            {"type": "text", "text": "Read the Kahoot question and pick the correct answer."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]
        response = litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            api_key=self.api_key,
            api_base="https://integrations.emergentagent.com/llm",
            custom_llm_provider="openai",
        )
        return self._parse(response.choices[0].message.content)

    @staticmethod
    def _parse(reply):
        m = re.search(r"\{.*\}", reply, re.DOTALL)
        if not m:
            return {"question": "", "answer_color": "unknown",
                    "answer_text": reply.strip()[:120], "confidence": 0.0,
                    "reasoning": "could not parse"}
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {"question": "", "answer_color": "unknown",
                    "answer_text": reply.strip()[:120], "confidence": 0.0,
                    "reasoning": "invalid json"}
        data.setdefault("question", "")
        data.setdefault("answer_color", "unknown")
        data.setdefault("answer_text", "")
        data.setdefault("confidence", 0.0)
        data.setdefault("reasoning", "")
        return data