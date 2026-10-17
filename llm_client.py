"""Thin client for the free Groq and Gemini APIs (plain HTTP, no vendor SDK)."""
import json
import os
import re

import requests

DEFAULT_MODELS = {"groq": "llama-3.3-70b-versatile", "gemini": "gemini-2.5-flash"}
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class LLMError(Exception):
    """Raised with a message that is safe to show to the student."""


def extract_json(text: str):
    """Parse a JSON value from a model reply, tolerating code fences and extra prose."""
    t = re.sub(r"^```(?:json)?|```$", "", str(text).strip(), flags=re.M).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (t.find("{"), t.find("[")) if i >= 0]
    end = max(t.rfind("}"), t.rfind("]"))
    if not starts or end <= min(starts):
        raise LLMError("The AI reply was not valid JSON.")
    try:
        return json.loads(t[min(starts): end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError("The AI reply was not valid JSON.") from exc


class LLMClient:
    """chat(messages) where messages = [{"role": "system"|"user"|"assistant", "content": str}]."""

    def __init__(self, provider=None, api_key=None, model=None, timeout=60):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "groq")).lower()
        if self.provider not in DEFAULT_MODELS:
            raise LLMError("LLM_PROVIDER must be 'groq' or 'gemini'.")
        env_key = os.getenv("GROQ_API_KEY" if self.provider == "groq" else "GEMINI_API_KEY", "")
        self.api_key = (api_key or env_key).strip()
        self.model = (model or os.getenv(f"{self.provider.upper()}_MODEL") or DEFAULT_MODELS[self.provider]).strip()
        self.timeout = timeout
        self.tokens = 0

    def chat(self, messages, json_mode=False, temperature=0.3) -> str:
        if not self.api_key:
            raise LLMError(f"Add your free {self.provider.title()} API key first (see README).")
        try:
            if self.provider == "groq":
                text = self._groq(messages, json_mode, temperature)
            else:
                text = self._gemini(messages, json_mode, temperature)
        except requests.RequestException as exc:
            raise LLMError(f"Could not reach {self.provider.title()}: {exc}") from exc
        if not text.strip():
            raise LLMError("The AI returned an empty reply. Try again.")
        self.tokens += (len(json.dumps(messages)) + len(text)) // 4
        return text

    def chat_json(self, messages, temperature=0.3):
        last = ""
        for attempt in range(2):
            msgs = messages if not attempt else messages + [
                {"role": "assistant", "content": last},
                {"role": "user", "content": "That was not valid JSON. Reply again with ONLY the JSON object."},
            ]
            last = self.chat(msgs, json_mode=True, temperature=temperature)
            try:
                return extract_json(last)
            except LLMError:
                continue
        raise LLMError("The AI reply was not valid JSON. Please try again.")

    # -- providers ---------------------------------------------------------
    def _groq(self, messages, json_mode, temperature):
        body = {"model": self.model, "messages": messages, "temperature": temperature}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        r = requests.post(GROQ_URL, json=body, timeout=self.timeout,
                          headers={"Authorization": f"Bearer {self.api_key}"})
        self._check(r)
        return r.json()["choices"][0]["message"]["content"] or ""

    def _gemini(self, messages, json_mode, temperature):
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [
            {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
            for m in messages if m["role"] != "system"
        ]
        config = {"temperature": temperature}
        if json_mode:
            config["responseMimeType"] = "application/json"
        r = requests.post(GEMINI_URL.format(model=self.model), timeout=self.timeout,
                          headers={"x-goog-api-key": self.api_key},
                          json={"systemInstruction": {"parts": [{"text": system}]},
                                "contents": contents, "generationConfig": config})
        self._check(r)
        data = r.json()
        parts = (data.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts)

    def _check(self, r):
        if r.ok:
            return
        name = self.provider.title()
        try:
            msg = r.json().get("error", {}).get("message", "")
        except ValueError:
            msg = ""
        if r.status_code in (401, 403) or "api key" in msg.lower():
            raise LLMError(f"{name} rejected the API key. Check that you pasted the whole key.")
        if r.status_code == 429:
            raise LLMError(f"{name} free-tier limit reached. Wait about a minute and try again.")
        if r.status_code == 404 or (r.status_code == 400 and "model" in msg.lower()):
            raise LLMError(f"{name} could not use the model '{self.model}'. Change the model name in .env.")
        raise LLMError(f"{name} error {r.status_code}: {msg[:200]}")
