# StudyMate

A study assistant for students, built from scratch on a free Groq or Gemini API key.

## Features

1. **Ask (hints first)**: the question is categorized by domain and theme, then you get three progressive hints and a thinking timer. Open the full answer when you are ready. Choose "I'm in a hurry" (or write words like "urgent" or "asap") to skip the hints.
2. **Notes summarizer**: paste text or upload .txt, .md, .csv or .pdf for a summary, key points, definitions and revision notes.
3. **Quiz generator**: multiple choice, true or false and short answer, with explanations after you answer.
4. **Flashcards**: terms, formulas and definitions as a flip-and-sort deck.
5. **Study plan**: quiz results are tracked by subject and topic, weak topics are ranked, and exam dates become a day-by-day revision plan.
6. **Explain any topic**: Explain simply, Detailed, or Exam-ready, with follow-up questions.

The agent answers any general question like a chat assistant. For arithmetic and unit conversion it calls two tools instead of guessing: `calculator` (safe parser, no `eval`) and `unit_converter`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then paste a free key
streamlit run app.py
```

Free keys: https://console.groq.com/keys or https://aistudio.google.com/app/apikey

## Tests

```bash
pytest -q
```

The tests use a fake model, so they need no key and no internet.

## Files

| File | Purpose |
| --- | --- |
| `llm_client.py` | Groq and Gemini over plain HTTP, JSON parsing, friendly errors |
| `tools.py` | `calculator` and `unit_converter` |
| `agent.py` | Hints, urgency detection, tool-calling answer loop, memory |
| `features.py` | Notes, quiz, grading, flashcards, explainer |
| `planner.py` | Weakness detector, revision plan, progress file |
| `app.py` | Streamlit interface |
| `tests/` | Unit tests |

`progress.json` is created next to `app.py` and holds quiz results and exam dates for this machine only.
