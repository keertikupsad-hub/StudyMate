"""Notes summarizer, quiz generator, flashcards and the three-style explainer."""
import json

from llm_client import LLMClient, LLMError

STYLES = {
    "simple": "Explain like you are teaching a first-year student who is new to the topic: plain words, one everyday analogy, define any jargon, under 200 words.",
    "detailed": "Give a thorough explanation with simple markdown headings and bullets: background, how it works step by step, one worked example, and common mistakes.",
    "exam": "Write an exam-ready answer for a 5 to 8 mark question: a one-line definition, numbered key points, a formula or labelled-diagram description if relevant, one short example, and a two-line conclusion.",
}
SIZES = {"short": ("2-3", "4-6", "5-7"), "standard": ("4-5", "6-9", "8-11"), "detailed": ("6-8", "10-14", "12-16")}


def summarize_notes(client: LLMClient, text: str, detail: str = "standard") -> dict:
    s, k, r = SIZES.get(detail, SIZES["standard"])
    system = (
        "You are a study-notes assistant. Reply with ONLY a JSON object:\n"
        f'{{"title":"short title","summary":"{s} sentences","key_points":["{k} short points"],'
        '"definitions":[{"term":"...","meaning":"one sentence"}],'
        f'"revision_notes":["{r} compact last-minute revision bullets, include formulas and dates if present"],'
        '"likely_questions":["3-5 questions an examiner could ask"]}\n'
        "Use only information found in the notes. Do not add outside facts."
    )
    return client.chat_json([{"role": "system", "content": system}, {"role": "user", "content": text[:12000]}])


QUIZ_SYSTEM = """You write exam-style quizzes. Reply with ONLY a JSON object:
{"questions":[{"type":"mcq or tf or short","question":"...","options":["...","...","...","..."],"answer":"mcq: index 0-3 | tf: true or false | short: model answer in 1-3 sentences","explanation":"why, 1-2 sentences","subtopic":"2-4 word topic label"}]}
Use only the requested question types, spread evenly. An mcq has exactly 4 plausible options and one correct. Omit options for tf and short. If the input looks like notes, base the questions on them; if it is a topic name, use well-established knowledge. Questions must be unambiguous."""


def normalize_quiz(raw: dict) -> list:
    out = []
    for x in raw.get("questions", []):
        t = str(x.get("type", "")).lower()
        t = "tf" if ("tf" in t or "true" in t) else "mcq" if t.startswith("m") else "short"
        q = {"type": t, "question": str(x.get("question", "")).strip(),
             "explanation": str(x.get("explanation", "")), "subtopic": str(x.get("subtopic", "")).strip()}
        if not q["question"]:
            continue
        if t == "mcq":
            opts = []
            for o in (x.get("options") or [])[:4]:
                o = str(o)
                opts.append(o[3:].strip() if len(o) > 3 and o[0] in "ABCD" and o[1] in ").'" and o[2] == " " else o)
            a = x.get("answer")
            if isinstance(a, str) and len(a.strip()) == 1 and a.strip().upper() in "ABCD":
                a = ord(a.strip().upper()) - 65
            elif isinstance(a, str) and not a.strip().isdigit():
                a = next((i for i, o in enumerate(opts) if o.strip().lower() == a.strip().lower()), -1)
            try:
                a = int(a)
            except (TypeError, ValueError):
                continue
            if len(opts) < 2 or not 0 <= a < len(opts):
                continue
            q["options"], q["answer"] = opts, a
        elif t == "tf":
            q["answer"] = x.get("answer") is True or str(x.get("answer")).lower() == "true"
        else:
            q["answer"] = str(x.get("answer") or x.get("model_answer") or "")
            if not q["answer"]:
                continue
        out.append(q)
    return out


def generate_quiz(client: LLMClient, subject: str, source: str, count: int, types, difficulty: str) -> list:
    payload = {"subject": subject, "difficulty": difficulty, "count": count,
               "types": list(types) or ["mcq"], "topic_or_notes": source[:8000]}
    raw = client.chat_json([{"role": "system", "content": QUIZ_SYSTEM},
                            {"role": "user", "content": json.dumps(payload)}])
    qs = normalize_quiz(raw)
    if not qs:
        raise LLMError("No usable questions came back. Try again or simplify the topic.")
    return qs


def grade_short_answers(client: LLMClient, items: list) -> dict:
    """items: [{"i": int, "question": str, "reference_answer": str, "student_answer": str}] -> {i: (points, feedback)}"""
    if not items:
        return {}
    system = ('You grade short answers by meaning, not wording. Reply with ONLY JSON: '
              '{"grades":[{"i":0,"verdict":"correct or partial or incorrect","feedback":"one helpful sentence"}]}')
    data = client.chat_json([{"role": "system", "content": system}, {"role": "user", "content": json.dumps(items)}])
    points = {"correct": 1.0, "partial": 0.5}
    return {int(g["i"]): (points.get(str(g.get("verdict", "")).lower(), 0.0), str(g.get("feedback", "")))
            for g in data.get("grades", []) if "i" in g}


def generate_flashcards(client: LLMClient, source: str, count: int = 12) -> list:
    system = ('You make flashcards. Reply with ONLY a JSON object: {"cards":[{"front":"term, question, formula name or date prompt",'
              '"back":"concise answer under 30 words","tag":"1-2 word category"}]}. Cover definitions, formulas, dates, '
              f"key terms and concepts found in the input. Make exactly {count} cards.")
    data = client.chat_json([{"role": "system", "content": system}, {"role": "user", "content": source[:8000]}])
    cards = [{"front": str(c["front"]), "back": str(c["back"]), "tag": str(c.get("tag", ""))}
             for c in data.get("cards", []) if isinstance(c, dict) and c.get("front") and c.get("back")]
    if not cards:
        raise LLMError("No cards came back. Try again.")
    return cards


def explain(client: LLMClient, thread: list, style: str = "simple") -> str:
    """thread = [{"role": "user"|"assistant", "content": str}] ending on a user message."""
    system = "You are StudyMate, a patient tutor. " + STYLES[style] + " Use simple markdown. Never invent facts."
    return client.chat([{"role": "system", "content": system}] + thread[-8:], temperature=0.4)
