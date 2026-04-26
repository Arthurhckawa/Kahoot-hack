# Real-time Kahoot Solver

A standalone Python desktop tool that watches your screen, detects a Kahoot
question, runs multilingual OCR on it, and asks an LLM (Claude Sonnet 4.5 by
default) for the most likely correct answer. The answer is shown in an
always-on-top overlay window — typically within ~3-5 seconds.

> Disclaimer: This is a technical demo of real-time CV + OCR + LLM
> orchestration. Using it to cheat in live quizzes is unethical and may
> violate Kahoot's terms of service. Use at your own risk on practice quizzes
> or as a learning project.

## Architecture

```
 mss screen  ->  color detector  ->  EasyOCR (multi-lang)  ->  Claude Sonnet 4.5
   capture                                                            |
                                                                      v
                                                            tkinter overlay
```

* `capture.py`      - `mss` based fast full-screen grab (BGR numpy)
* `color_detect.py` - detects the 4 Kahoot answer tiles by HSV signature
* `ocr_engine.py`   - EasyOCR reader (English + Danish out of the box)
* `solver.py`       - sends question + options to Claude via the Emergent LLM key
* `overlay.py`      - always-on-top tkinter window
* `main.py`         - capture loop + worker thread orchestration

## Output format (matches the spec)

```json
{
  "language_detected": "da",
  "detected_text": "Hvilken farve har himlen | Roed | Blaa | Gul | Groen",
  "elements": [
    {"type": "text",   "value": "Hvilken farve har himlen", "color": "white"},
    {"type": "button", "value": "Roed",  "color": "red"},
    {"type": "button", "value": "Blaa",  "color": "blue"},
    {"type": "button", "value": "Gul",   "color": "yellow"},
    {"type": "button", "value": "Groen", "color": "green"}
  ]
}
```

## Setup (run on your own computer)

You need Python 3.10+ and a real desktop session (this script captures the
actual screen, so it cannot run inside a headless container).

```bash
cd kahoot_solver
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install emergentintegrations \
    --extra-index-url https://d33sy5i8bnduwe.cloudfront.net/simple/

cp .env.example .env
# open .env and paste your EMERGENT_LLM_KEY
```

### macOS

* Grant your terminal Screen Recording permission
  (System Settings -> Privacy & Security -> Screen Recording).
* Tk requires python.org Python or `brew install python-tk`.

### Linux

* Wayland blocks raw screen capture - use X11 / XWayland.
* `sudo apt install python3-tk libgl1`.

### Windows

* Tk is bundled with the python.org installer.

## Run it

```bash
python main.py
```

You will see:

```
[init] loading EasyOCR (this may take a minute the first time)...
[capture] running - press Ctrl+C in terminal to quit
```

Open a Kahoot question on the same monitor. The overlay flashes the correct
color + text within a few seconds and structured JSON is printed on stdout.

## Smoke test (no real Kahoot needed)

```bash
python test_offline.py            # builds a synthetic Kahoot frame
python test_offline.py --llm      # also calls Claude
```

## Tuning (.env)

| Variable        | Default                         | Notes                            |
| --------------- | ------------------------------- | -------------------------------- |
| `LLM_PROVIDER`  | `anthropic`                     | `openai`, `gemini`, `anthropic`  |
| `LLM_MODEL`     | `claude-sonnet-4-5-20250929`    | any Emergent-supported model     |
| `OCR_LANGUAGES` | `en,da`                         | comma-separated EasyOCR codes    |
| `SCAN_INTERVAL` | `0.4`                           | seconds between screen scans     |

## Performance

* `mss` 1080p frame: ~10-20 ms
* HSV color check:    ~5 ms
* EasyOCR (5 regions):~150-400 ms on CPU, much faster on GPU
* Claude Sonnet 4.5:  ~1-2 s
* End-to-end latency: typically under 5 seconds.
