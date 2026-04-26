"""Send the parsed question + answers to Claude (or any Emergent-supported model)."""
from __future__ import annotations

import asyncio
import json
import re
import uuid

from emergentintegrations.llm.chat import LlmChat, UserMessage


SYSTEM_PROMPT = (
    "You are a Kahoot quiz assistant. You will be given a question and the four "
    "answer options taken straight from a Kahoot screen via OCR. The OCR can be "
    "imperfect - infer the most likely original text. "
    "Respond with ONLY a JSON object on a single line, no markdown, no commentary:\n"
    '{"answer_color": "red|blue|yellow|green", '
    '"answer_text": "<the exact option text>", '
    '"confidence": 0.0-1.0, '
    '"reasoning": "<one short sentence>"}'
)


class Solver:
    def __init__(self, api_key: str, provider: str = "anthropic",
                 model: str = "claude-sonnet-4-5-20250929"):
        if not api_key:
            raise RuntimeError("EMERGENT_LLM_KEY is missing - set it in .env")
        self.api_key = api_key
        self.provider = provider
        self.model = model

    async def _ask_async(self, question: str, options: dict) -> dict:
        chat = LlmChat(
            api_key=self.api_key,
            session_id=f"kahoot-{uuid.uuid4().hex[:8]}",
            system_message=SYSTEM_PROMPT,
        ).with_model(self.provider, self.model)

        prompt = (
            f"Question: {question}\n\n"
            f"Options:\n"
            f"- RED: {options.get('red', '')}\n"
            f"- BLUE: {options.get('blue', '')}\n"
            f"- YELLOW: {options.get('yellow', '')}\n"
            f"- GREEN: {options.get('green', '')}\n\n"
            "Pick the correct option."
        )
        reply = await chat.send_message(UserMessage(text=prompt))
        return self._parse(reply)

    def ask(self, question: str, options: dict) -> dict:
        return asyncio.run(self._ask_async(question, options))

    @staticmethod
    def _parse(reply: str) -> dict:
        match = re.search(r"\{.*\}", reply, re.DOTALL)
        if not match:
            return {"answer_color": "unknown", "answer_text": reply.strip(),
                    "confidence": 0.0, "reasoning": "could not parse"}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"answer_color": "unknown", "answer_text": reply.strip(),
                    "confidence": 0.0, "reasoning": "invalid json"}
        data.setdefault("answer_color", "unknown")
        data.setdefault("answer_text", "")
        data.setdefault("confidence", 0.0)
        data.setdefault("reasoning", "")
        return data
