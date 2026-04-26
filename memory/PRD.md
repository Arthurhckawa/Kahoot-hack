# Real-time Kahoot Solver - PRD

## Original problem statement
Build a real-time Python computer vision system for Kahoot quizzes:
- continuous frame analysis
- multilingual OCR
- detect main text, UI elements (buttons/boxes/labels), colors
- output structured JSON
- low latency (< 5 s end-to-end)
- OpenCV + threading/async + Python

## Architecture (chosen with user: option 1 - local desktop script)
mss screen capture -> color quadrant detector -> EasyOCR (multilingual)
-> Claude Sonnet 4.5 via Emergent LLM key -> always-on-top tkinter overlay.
Threaded worker so the LLM call never stalls the capture loop.

## Files (in /app/kahoot_solver/)
- capture.py        - mss-based BGR screen grabber
- color_detect.py   - HSV detection of the 4 Kahoot tiles + region splitting
- ocr_engine.py     - EasyOCR multilingual reader (en, da default)
- pipeline.py       - language detection + spec JSON builder
- solver.py         - LLM call via emergentintegrations
- overlay.py        - tkinter always-on-top answer window
- main.py           - capture loop + worker thread orchestration
- test_offline.py   - synthetic Kahoot frame smoke test
- requirements.txt
- .env.example
- README.md

## What's been implemented (Feb 2026)
- Real-time capture loop (configurable scan interval)
- Kahoot screen detection via 4-tile HSV signature
- Multilingual OCR (en, da out of box; configurable)
- Structured JSON output matching the spec exactly
- Background thread worker decouples LLM from capture
- Frame-deduplication so each unique question is asked only once
- Always-on-top overlay highlighting predicted color + text
- Offline test verified: synthetic Danish Kahoot frame -> correct OCR + JSON

## Backlog
- P1: GPU support flag for EasyOCR (4-10x speedup on supported HW)
- P1: Hotkey trigger (e.g., F8) for one-shot capture instead of polling
- P2: Save question history to local SQLite for review
- P2: Image-based questions: send the cropped question region to a
       multimodal model (Claude Sonnet 4.5 with vision / Gemini 3) instead
       of OCR text
- P2: Auto-click the answer (PyAutoGUI) - currently overlay only

## Next action items
- User runs `pip install -r requirements.txt` locally + paste their
  EMERGENT_LLM_KEY into .env
- Test on a real Kahoot practice quiz, tune SCAN_INTERVAL if needed
