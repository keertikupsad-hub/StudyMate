"""StudyMate Streamlit app. Run with:  streamlit run app.py"""
import os
from datetime import date

import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

import features
import planner
from agent import StudyAgent, is_urgent
from llm_client import DEFAULT_MODELS, LLMClient, LLMError

st.set_page_config(page_title="StudyMate", page_icon="📘", layout="wide")
ss = st.session_state


def init():
    ss.setdefault("agent", None)
    ss.setdefault("agent_key", None)
    ss.setdefault("ask", None)
    ss.setdefault("notes", None)
    ss.setdefault("quiz", None)
    ss.setdefault("quiz_done", None)
    ss.setdefault("deck", None)
    ss.setdefault("thread", [])
    ss.setdefault("progress", planner.load_progress())


def get_client():
    """Sidebar connection settings. Returns an LLMClient or None."""
    with st.sidebar:
        st.header("Connection")
        provider = st.selectbox("AI provider", ["groq", "gemini"],
                                index=0 if os.getenv("LLM_PROVIDER", "groq").lower() == "groq" else 1)
        env_key = os.getenv("GROQ_API_KEY" if provider == "groq" else "GEMINI_API_KEY", "")
        key = st.text_input("API key", value=env_key, type="password",
                            help="Free key: console.groq.com/keys or aistudio.google.com/app/apikey")
        model = st.text_input("Model", value=os.getenv(f"{provider.upper()}_MODEL", DEFAULT_MODELS[provider]))
        st.caption("Keys stay on this machine and go only to the provider you pick.")
    if not key.strip():
        st.warning("Add a free Groq or Gemini API key in the sidebar (or in .env) to start.")
        return None
    sig = (provider, key, model)
    if ss.agent is None or ss.agent_key != sig:
        ss.agent = StudyAgent(LLMClient(provider, key, model))
        ss.agent_key = sig
    return ss.agent.client


def show_answer(res):
    st.markdown(res.answer)
    st.caption(f"{res.steps} step(s) · {res.tool_calls} tool call(s) · {res.seconds:.1f} s · ~{res.tokens} tokens")
    with st.expander("How StudyMate worked it out"):
        for ev in res.trace:
            if ev["type"] == "thought":
                st.write(f"Step {ev['step']} thought: {ev['text']}")
            elif ev["type"] == "call":
                st.code(f"{ev['tool']}({ev['args']})", language="python")
            elif ev["type"] == "result":
                (st.success if ev["ok"] else st.error)(ev["text"])
            elif ev["type"] == "retry":
                st.warning(f"Step {ev['step']}: bad reply format, retried.")


# ---------------------------------------------------------------- Ask
def tab_ask(agent):
    q = st.text_area("Your question", key="ask_q", height=90,
                     placeholder="Ask anything: a maths problem, a concept, code, a unit conversion...")
    mode = st.radio("How do you want help?", ["Hints first", "I'm in a hurry"], horizontal=True)
    why = st.text_input("What is the rush? (optional)", placeholder="exam in 20 minutes") if mode == "I'm in a hurry" else ""
    if st.button("Ask StudyMate", type="primary") and q.strip():
        urgent = mode == "I'm in a hurry" or is_urgent(q)
        try:
            if urgent:
                with st.spinner("Thinking..."):
                    ss.ask = {"q": q, "hints": None, "answer": agent.answer(q, urgent=True, why=why)}
            else:
                with st.spinner("Working out the subject area and hints..."):
                    ss.ask = {"q": q, "hints": agent.hints(q), "shown": 1, "answer": None}
        except LLMError as exc:
            st.error(str(exc))
            return
    a = ss.ask
    if not a:
        return
    if a["hints"]:
        h = a["hints"]
        st.caption(f"Domain: **{h['domain']}** · Theme: **{h['theme']}** · Type: {h['kind']} · Level: {h['difficulty']}")
        st.info(f"Try it yourself first. Suggested thinking time: {h['think_minutes']} min.")
        for i, text in enumerate(h["hints"][: a["shown"]], 1):
            st.warning(f"**Hint {i} of {len(h['hints'])}**  \n{text}")
        c1, c2 = st.columns(2)
        if a["shown"] < len(h["hints"]) and c1.button(f"Another hint ({len(h['hints']) - a['shown']} left)"):
            a["shown"] += 1
            st.rerun()
        if c2.button("Show full answer", type="primary"):
            try:
                with st.spinner("Thinking..."):
                    a["answer"] = agent.answer(a["q"])
            except LLMError as exc:
                st.error(str(exc))
    if a["answer"]:
        st.subheader("Answer")
        show_answer(a["answer"])
        fu = st.text_input("Still unclear? Ask a follow-up", key="fu")
        if st.button("Ask follow-up") and fu.strip():
            try:
                with st.spinner("Thinking..."):
                    ss.ask = {"q": fu, "hints": None, "answer": agent.answer(fu)}
                st.rerun()
            except LLMError as exc:
                st.error(str(exc))


# ---------------------------------------------------------------- Notes
def tab_notes(client):
    text = st.text_area("Paste your notes", height=200, key="notes_text")
    up = st.file_uploader("Or upload a file", type=["txt", "md", "csv", "pdf"])
    detail = st.selectbox("Detail", ["short", "standard", "detailed"], index=1)
    if st.button("Summarize notes", type="primary"):
        if up is not None:
            if up.name.lower().endswith(".pdf"):
                try:
                    from pypdf import PdfReader
                    text = "\n\n".join((p.extract_text() or "") for p in PdfReader(up).pages[:40])
                except Exception:  # unreadable or scanned PDF
                    st.error("Could not read that PDF. Paste the text instead.")
                    return
            else:
                text = up.read().decode("utf-8", errors="ignore")
        if len(text.strip()) < 30:
            st.error("Paste at least a few sentences of notes first.")
            return
        try:
            with st.spinner("Summarizing..."):
                ss.notes = {"data": features.summarize_notes(client, text, detail), "text": text}
        except LLMError as exc:
            st.error(str(exc))
    n = ss.notes
    if n:
        d = n["data"]
        st.subheader(d.get("title", "Summary"))
        st.write(d.get("summary", ""))
        st.markdown("**Key points**\n" + "\n".join(f"- {x}" for x in d.get("key_points", [])))
        if d.get("definitions"):
            st.markdown("**Important definitions**\n" + "\n".join(f"- **{x['term']}**: {x['meaning']}" for x in d["definitions"]))
        st.markdown("**Revision notes**\n" + "\n".join(f"- {x}" for x in d.get("revision_notes", [])))
        if d.get("likely_questions"):
            st.markdown("**Questions to expect**\n" + "\n".join(f"- {x}" for x in d["likely_questions"]))


# ---------------------------------------------------------------- Quiz
def tab_quiz(client):
    c1, c2, c3 = st.columns(3)
    subject = c1.text_input("Subject", placeholder="e.g. Data Structures")
    diff = c2.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)
    n = c3.selectbox("Questions", [5, 8, 10, 15], index=1)
    src = st.text_area("Topic, or paste notes to be quizzed on", height=90)
    types = st.multiselect("Question types", ["mcq", "tf", "short"], default=["mcq", "tf", "short"],
                           format_func={"mcq": "Multiple choice", "tf": "True or false", "short": "Short answer"}.get)
    if st.button("Generate quiz", type="primary") and src.strip():
        try:
            with st.spinner("Writing questions..."):
                qs = features.generate_quiz(client, subject or "General", src, n, types, diff)
            ss.quiz = {"subject": subject or "General", "topic": src.splitlines()[0][:40], "qs": qs}
            ss.quiz_done = None
        except LLMError as exc:
            st.error(str(exc))
    quiz = ss.quiz
    if not quiz:
        return
    answers = {}
    with st.form("quiz_form"):
        for i, q in enumerate(quiz["qs"]):
            st.markdown(f"**{i + 1}. {q['question']}**")
            if q["type"] == "mcq":
                answers[i] = st.radio("Answer", range(len(q["options"])), index=None,
                                      format_func=lambda j, q=q: q["options"][j], key=f"a{i}", label_visibility="collapsed")
            elif q["type"] == "tf":
                answers[i] = st.radio("Answer", [True, False], index=None, key=f"a{i}", horizontal=True,
                                      format_func=lambda v: "True" if v else "False", label_visibility="collapsed")
            else:
                answers[i] = st.text_area("Answer", key=f"a{i}", height=70, label_visibility="collapsed")
        submitted = st.form_submit_button("Check my answers", type="primary")
    if submitted and ss.quiz_done is None:
        check_quiz(client, quiz, answers)
    if ss.quiz_done:
        render_results(quiz, ss.quiz_done)


def check_quiz(client, quiz, answers):
    qs, res = quiz["qs"], {}
    shorts = []
    for i, q in enumerate(qs):
        given = answers.get(i)
        res[i] = {"pts": 0.0, "given": given, "feedback": ""}
        if q["type"] in ("mcq", "tf"):
            res[i]["pts"] = 1.0 if given is not None and given == q["answer"] else 0.0
        elif given and str(given).strip():
            shorts.append({"i": i, "question": q["question"], "reference_answer": q["answer"], "student_answer": given})
    try:
        for i, (pts, fb) in features.grade_short_answers(client, shorts).items():
            if i in res:
                res[i]["pts"], res[i]["feedback"] = pts, fb
    except LLMError as exc:
        st.error(str(exc))
        return
    by = {}
    for i, q in enumerate(qs):
        topic = q["subtopic"] or quiz["topic"]
        by.setdefault(topic, [0.0, 0])
        by[topic][0] += res[i]["pts"]
        by[topic][1] += 1
    for topic, (c, t) in by.items():
        ss.progress["results"].append({"subject": quiz["subject"], "topic": topic, "correct": c, "total": t})
    planner.save_progress(ss.progress)
    ss.quiz_done = res


def render_results(quiz, res):
    total = sum(r["pts"] for r in res.values())
    st.success(f"Score {total:g} / {len(quiz['qs'])}")
    for i, q in enumerate(quiz["qs"]):
        r = res[i]
        ans = q["options"][q["answer"]] if q["type"] == "mcq" else ("True" if q["answer"] else "False") if q["type"] == "tf" else q["answer"]
        head = "Correct." if r["pts"] == 1 else "Partly correct." if r["pts"] > 0 else "Not quite."
        st.markdown(f"**{i + 1}. {head}** {r['feedback']} Answer: {ans}. {q['explanation']}")


# ---------------------------------------------------------------- Flashcards
def tab_cards(client):
    src = st.text_area("Topic or notes to turn into flashcards", height=100)
    n = st.selectbox("Cards", [8, 12, 20], index=1)
    if st.button("Make flashcards", type="primary") and src.strip():
        try:
            with st.spinner("Making cards..."):
                cards = features.generate_flashcards(client, src, n)
            ss.deck = {"all": cards, "queue": list(cards), "known": 0, "flip": False}
        except LLMError as exc:
            st.error(str(exc))
    d = ss.deck
    if not d:
        return
    total = len(d["all"])
    if not d["queue"]:
        st.success(f"Deck complete. You knew all {total} cards.")
        if st.button("Study the deck again"):
            ss.deck = {"all": d["all"], "queue": list(d["all"]), "known": 0, "flip": False}
            st.rerun()
        return
    card = d["queue"][0]
    st.progress(d["known"] / total, text=f"{d['known']} of {total} known · {len(d['queue'])} left")
    st.markdown(f"### {card['back'] if d['flip'] else card['front']}")
    if not d["flip"]:
        if st.button("Show answer"):
            d["flip"] = True
            st.rerun()
    else:
        c1, c2 = st.columns(2)
        if c1.button("Missed it"):
            d["queue"].append(d["queue"].pop(0))
            d["flip"] = False
            st.rerun()
        if c2.button("Got it", type="primary"):
            d["queue"].pop(0)
            d["known"] += 1
            d["flip"] = False
            st.rerun()


# ---------------------------------------------------------------- Planner
def tab_plan(client):
    prog = ss.progress
    agg = planner.aggregate(prog["results"])
    if not agg:
        st.info("Take a quiz and your weak topics will show up here.")
    for s in agg:
        st.markdown(f"**{s['name']}** · {s['pct']}% overall")
        for t in s["topics"]:
            st.progress(t["pct"] / 100, text=f"{t['topic']} · {t['pct']}%")
    st.subheader("Exam dates")
    c1, c2, c3 = st.columns([2, 2, 1])
    subj = c1.text_input("Subject", key="ex_subj")
    when = c2.date_input("Exam date", min_value=date.today(), key="ex_date")
    if c3.button("Add exam") and subj.strip():
        prog["exams"].append({"subject": subj.strip(), "date": when.isoformat()})
        planner.save_progress(prog)
        st.rerun()
    for i, e in enumerate(prog["exams"]):
        c1, c2 = st.columns([5, 1])
        c1.write(f"{e['subject']}: {e['date']}")
        if c2.button("Remove", key=f"rm{i}"):
            prog["exams"].pop(i)
            planner.save_progress(prog)
            st.rerun()
    plan = planner.build_plan(prog["exams"], planner.weak_topics(agg), date.today())
    if plan:
        st.subheader("Revision plan")
        for day in plan:
            st.markdown(f"**{day['date']:%a %d %b}**")
            for it in day["items"]:
                st.write(f"- {it['subject']}: {it['task']}")
    if agg and st.button("Get study advice"):
        try:
            import json
            msg = [{"role": "system", "content": "You are a supportive study coach. Using the quiz statistics, write under 150 words in simple markdown: three bullets on what to study first and why, then one practical habit tip. Do not invent scores."},
                   {"role": "user", "content": json.dumps(agg)}]
            st.markdown(client.chat(msg))
        except LLMError as exc:
            st.error(str(exc))
    if prog["results"] and st.button("Clear my results"):
        prog["results"] = []
        planner.save_progress(prog)
        st.rerun()


# ---------------------------------------------------------------- Explain
def tab_explain(client):
    style = st.radio("Style", ["simple", "detailed", "exam"], horizontal=True,
                     format_func={"simple": "Explain simply", "detailed": "Detailed", "exam": "Exam-ready"}.get)
    q = st.text_area("What do you want explained?", height=80, key="ex_q")
    if st.button("Explain", type="primary") and q.strip():
        ss.thread = [{"role": "user", "content": q}]
        run_explain(client, style)
    for m in ss.thread:
        st.markdown(("**You asked:** " + m["content"]) if m["role"] == "user" else m["content"])
    if ss.thread:
        fu = st.text_input("Did not get it? Ask a follow-up", key="ex_fu")
        c1, c2 = st.columns(2)
        if c1.button("Ask") and fu.strip():
            ss.thread.append({"role": "user", "content": fu})
            run_explain(client, style)
            st.rerun()
        if c2.button("Explain again in the selected style"):
            ss.thread.append({"role": "user", "content": "Please explain that again in the style you were just given, using different words and a different example."})
            run_explain(client, style)
            st.rerun()


def run_explain(client, style):
    try:
        with st.spinner("Explaining..."):
            ss.thread.append({"role": "assistant", "content": features.explain(client, ss.thread, style)})
    except LLMError as exc:
        ss.thread.pop()
        st.error(str(exc))


def main():
    init()
    st.title("📘 StudyMate")
    st.caption("Hints before answers. Quizzes that remember what you miss.")
    client = get_client()
    if client is None:
        st.stop()
    t_ask, t_notes, t_quiz, t_cards, t_plan, t_explain = st.tabs(
        ["Ask", "Notes", "Quiz", "Flashcards", "Study plan", "Explain"])
    with t_ask:
        tab_ask(ss.agent)
    with t_notes:
        tab_notes(client)
    with t_quiz:
        tab_quiz(client)
    with t_cards:
        tab_cards(client)
    with t_plan:
        tab_plan(client)
    with t_explain:
        tab_explain(client)


main()
