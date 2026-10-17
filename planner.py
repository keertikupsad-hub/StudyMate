"""Weakness detector and revision planner. Pure functions plus a small JSON progress file."""
import json
from datetime import date, timedelta
from pathlib import Path

PROGRESS_FILE = Path(__file__).with_name("progress.json")


def load_progress(path: Path = PROGRESS_FILE) -> dict:
    try:
        data = json.loads(path.read_text())
        return {"results": list(data.get("results", [])), "exams": list(data.get("exams", []))}
    except (OSError, ValueError):
        return {"results": [], "exams": []}


def save_progress(data: dict, path: Path = PROGRESS_FILE) -> None:
    path.write_text(json.dumps(data, indent=2))


def aggregate(results: list) -> list:
    """Group quiz results by subject then topic. Weakest subjects and topics come first."""
    subjects = {}
    for r in results:
        s_name = (r.get("subject") or "General").strip()
        s = subjects.setdefault(s_name.lower(), {"name": s_name, "topics": {}, "c": 0.0, "t": 0})
        t_name = (r.get("topic") or "General").strip()
        t = s["topics"].setdefault(t_name.lower(), {"topic": t_name, "c": 0.0, "t": 0})
        t["c"] += r["correct"]
        t["t"] += r["total"]
        s["c"] += r["correct"]
        s["t"] += r["total"]
    out = []
    for s in subjects.values():
        topics = sorted(({"topic": t["topic"], "total": t["t"], "pct": round(100 * t["c"] / t["t"])}
                         for t in s["topics"].values()), key=lambda x: x["pct"])
        out.append({"name": s["name"], "total": s["t"], "pct": round(100 * s["c"] / s["t"]), "topics": topics})
    return sorted(out, key=lambda x: x["pct"])


def weak_topics(aggregated: list, below: int = 75) -> dict:
    return {s["name"].lower(): [t for t in s["topics"] if t["pct"] < below] for s in aggregated}


def build_plan(exams: list, weak: dict, today: date, horizon: int = 20) -> list:
    """exams: [{"subject", "date": "YYYY-MM-DD"}]. Returns [{"date": date, "items": [{"subject", "task"}]}]."""
    upcoming = []
    for e in exams:
        d = (date.fromisoformat(e["date"]) - today).days
        if d >= 0:
            upcoming.append({**e, "d": d})
    upcoming.sort(key=lambda e: e["d"])
    if not upcoming:
        return []
    last = min(upcoming[-1]["d"], horizon)
    cursor, days = {}, []
    for i in range(last + 1):
        items = []
        for e in [x for x in upcoming if x["d"] >= i][:2]:  # focus on the two nearest exams
            left = e["d"] - i
            if left == 0:
                items.append({"subject": e["subject"], "task": "Exam day. Skim formulas and definitions only."})
            elif left == 1:
                items.append({"subject": e["subject"], "task": "Final revision: flashcards, then one mixed quiz."})
            else:
                topics = weak.get(e["subject"].strip().lower(), [])
                if not topics:
                    items.append({"subject": e["subject"], "task": "Take a quiz to find your weak topics, then revise the lowest scoring one."})
                else:
                    key = e["subject"].lower()
                    t = topics[cursor.get(key, 0) % len(topics)]
                    cursor[key] = cursor.get(key, 0) + 1
                    items.append({"subject": e["subject"],
                                  "task": f'Study "{t["topic"]}" ({t["pct"]}% so far), then answer 5 practice questions.'})
        days.append({"date": today + timedelta(days=i), "items": items})
    return days
