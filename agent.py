"""StudyMate agent: hints-first coaching plus a tool-using answer loop."""
import time
from dataclasses import dataclass, field

from llm_client import LLMClient, LLMError, extract_json
from tools import TOOLS, run_tool

MAX_STEPS = 5
FORMAT_RETRIES = 2
MEM_TURNS = 6
URGENT_WORDS = (
    "urgent", "urgently", "asap", "immediately", "right now", "in a hurry", "hurry", "no time",
    "quickly", "last minute", "last-minute", "deadline", "due today", "due tonight",
    "exam tomorrow", "exam today", "exam in",
)
DOMAINS = ("Mathematics, Physics, Chemistry, Biology, Computer Science, Electronics & ECE, Engineering, "
           "Languages & Literature, History & Social Science, Commerce & Economics, General Knowledge, Other")

SYSTEM = """You are StudyMate, a friendly and accurate study assistant for college students. You can answer questions on any subject (maths, science, engineering, programming, languages, humanities, commerce, general knowledge) the way a strong tutor would: clear, structured and correct.
TOOLS:
{tools}
Reply with ONLY one JSON object, no markdown fences, exactly this shape:
{{"thought":"short reasoning","tool":"<tool name or null>","args":{{}},"final_answer":"<markdown text or null>"}}
RULES:
1. Arithmetic, evaluating expressions and unit conversions must go through a tool, even easy ones. One tool per reply.
2. Everything else (explanations, definitions, code, essays, advice) is answered directly in final_answer with tool set to null.
3. After a TOOL RESULT, call another tool or give the final answer.
4. If a tool returns an ERROR, fix the arguments or try another approach.
5. final_answer uses simple markdown: short paragraphs, "- " bullets, **bold** for key terms, and fenced code blocks for code. Show the steps of any calculation.
6. Use earlier turns to resolve follow-ups like "convert that to km".
7. Never invent facts. If you are not sure, say so."""

HINT_SYSTEM = f"""You are StudyMate, a study coach. The student asked a question. Do NOT answer it. Reply with ONLY a JSON object:
{{"domain":"<one of: {DOMAINS}>","theme":"<specific topic, 2-5 words>","kind":"<one of: Calculation, Concept, Programming, Definition, Reasoning, Factual, Other>","difficulty":"Easy or Medium or Hard","think_minutes":<integer 1-3>,"hints":["hint 1: nudge toward the right concept, no formulas","hint 2: name the method, formula or idea to use","hint 3: describe the first concrete step without finishing it"]}}
Hints must never contain the final answer or the final number. Keep each hint under 35 words."""


def is_urgent(question: str) -> bool:
    q = question.lower()
    return any(w in q for w in URGENT_WORDS)


def parse_decision(raw: str) -> dict:
    obj = extract_json(raw)
    if not isinstance(obj, dict):
        raise ValueError("JSON must be an object")
    tool, final = obj.get("tool"), obj.get("final_answer")
    if tool in (None, "", "null", "none"):
        if not isinstance(final, str) or not final.strip():
            raise ValueError('need either a "tool" or a non-empty "final_answer"')
        return {"thought": str(obj.get("thought", "")), "tool": None, "args": {}, "final": final.strip()}
    if tool not in TOOLS:
        raise ValueError(f'unknown tool "{tool}"; choose from {list(TOOLS)}')
    args = obj.get("args") or {}
    if not isinstance(args, dict):
        raise ValueError('"args" must be an object')
    return {"thought": str(obj.get("thought", "")), "tool": tool, "args": args, "final": None}


@dataclass
class AgentResult:
    question: str
    answer: str = ""
    ok: bool = True
    steps: int = 0
    tool_calls: int = 0
    seconds: float = 0.0
    tokens: int = 0
    trace: list = field(default_factory=list)


class StudyAgent:
    def __init__(self, client: LLMClient):
        self.client = client
        self.memory = []  # [(question, answer)]

    def clear_memory(self):
        self.memory = []

    def hints(self, question: str) -> dict:
        data = self.client.chat_json([{"role": "system", "content": HINT_SYSTEM},
                                      {"role": "user", "content": question}])
        hints = [h for h in (data.get("hints") or []) if isinstance(h, str) and h.strip()][:3]
        if not hints:
            raise LLMError("Could not build hints for that question. Try rephrasing it.")
        try:
            minutes = min(3, max(1, int(data.get("think_minutes", 2))))
        except (TypeError, ValueError):
            minutes = 2
        return {"domain": data.get("domain", "General Knowledge"), "theme": data.get("theme", ""),
                "kind": data.get("kind", ""), "difficulty": data.get("difficulty", ""),
                "think_minutes": minutes, "hints": hints}

    def answer(self, question: str, urgent: bool = False, why: str = "") -> AgentResult:
        t0, tok0 = time.time(), self.client.tokens
        res = AgentResult(question=question)
        tools_doc = "\n".join(f"- {n}: {desc}" for n, (_, desc) in TOOLS.items())
        system = SYSTEM.format(tools=tools_doc)
        if urgent:
            system += (f"\nThe student is in a hurry{' (' + why + ')' if why else ''}. "
                       "Lead with the direct answer in 1-3 lines, then give only the essential steps. Keep it short.")
        base = [{"role": "system", "content": system}]
        for q, a in self.memory[-MEM_TURNS:]:
            base.append({"role": "user", "content": q})
            base.append({"role": "assistant", "content": '{"thought":"","tool":null,"args":{},"final_answer":%s}' % _jstr(a[:700])})
        base.append({"role": "user", "content": question})
        scratch, last_result = [], ""
        for step in range(1, MAX_STEPS + 1):
            res.steps = step
            decision, fix = None, ""
            for _ in range(FORMAT_RETRIES + 1):
                msgs = base + scratch
                if step == MAX_STEPS:
                    msgs = msgs + [{"role": "user", "content": "Give your final_answer now. Do not call another tool."}]
                if fix:
                    msgs = msgs + [{"role": "user", "content": fix}]
                raw = self.client.chat(msgs, json_mode=True)
                try:
                    decision = parse_decision(raw)
                    break
                except (ValueError, LLMError) as exc:
                    res.trace.append({"type": "retry", "step": step, "error": str(exc)})
                    fix = f"Your previous reply was invalid ({exc}). Reply with ONLY the JSON object."
            if decision is None:
                raise LLMError("The model kept returning invalid JSON. Try again.")
            if decision["thought"]:
                res.trace.append({"type": "thought", "step": step, "text": decision["thought"]})
            if decision["tool"] is None:
                res.answer = decision["final"]
                break
            res.trace.append({"type": "call", "step": step, "tool": decision["tool"], "args": decision["args"]})
            ok, text = run_tool(decision["tool"], decision["args"])
            res.tool_calls += 1
            last_result = text
            res.trace.append({"type": "result", "step": step, "ok": ok, "text": text})
            scratch.append({"role": "assistant", "content": _dump({"thought": decision["thought"], "tool": decision["tool"],
                                                                  "args": decision["args"], "final_answer": None})})
            scratch.append({"role": "user", "content": f"TOOL RESULT ({decision['tool']}): {text}" if ok else
                            f"TOOL RESULT ({decision['tool']}) ERROR: {text}. Fix the arguments or try another approach."})
            if step == MAX_STEPS:
                res.ok = False
                res.answer = f"I could not finish within {MAX_STEPS} steps. Last tool result: {last_result}"
        self.memory.append((question, res.answer))
        res.seconds = time.time() - t0
        res.tokens = self.client.tokens - tok0
        return res


def _dump(obj) -> str:
    import json
    return json.dumps(obj)


def _jstr(s: str) -> str:
    import json
    return json.dumps(s)
