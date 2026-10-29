"""Generate deep_dive/rater_groq_appendix.py.txt from rater_gemini_appendix.py.txt (same safety design, Groq endpoint)."""
from pathlib import Path

D = Path(__file__).resolve().parent
t = (D / "rater_gemini_appendix.py.txt").read_text(encoding="utf-8")


def rep(old, new):
    global t
    assert t.count(old) >= 1, old[:80]
    t = t.replace(old, new)


rep("# ============================================================================= Gemini item-rating offset (deep dive 2026-10-29)",
    "# ============================================================================= Groq item-rating offset (deep dive 2026-10-29)")
rep("by deep_dive/build_rater_gemini.py, ONLY if the pre-registered Gemini gate passed\n# (PREREG_items_AMENDMENT_4.md).",
    "by deep_dive/build_rater_groq.py, ONLY if the pre-registered Groq gate passed\n# (PREREG_items_AMENDMENT_5.md).")
rep("and p is GEMINI_MODEL's stated probability", "and p is GROQ_MODEL's stated probability")
rep('({"gemini_api_key": "..."})', '({"groq_api_key": "..."})')
rep("The key goes in the x-goog-api-key header,\n# never in a URL", "The key goes in the Authorization header,\n# never in a URL")
rep('''GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent"
GEMINI_GEN_CONFIG = __GEN_CONFIG__''', '''GROQ_MODEL = __GROQ_MODEL__
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_PARAMS = __GEN_CONFIG__''')
rep('''            k = json.load(fh).get("gemini_api_key", "")''', '''            k = json.load(fh).get("groq_api_key", "")''')
rep('''def _gemini_http(model, body, timeout):
    """One generateContent call; returns the parsed JSON (raises on HTTP / network errors)."""
    import urllib.request
    req = urllib.request.Request(GEMINI_URL % model, data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"x-goog-api-key": _R["key"], "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))''', '''def _groq_http(model, body, timeout):
    """One chat-completions call; returns the parsed JSON (raises on HTTP / network errors)."""
    import urllib.request
    req = urllib.request.Request(GROQ_URL, data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"Authorization": "Bearer " + _R["key"], "Content-Type": "application/json",
                                          "User-Agent": "paiec-submission/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))''')
rep('''    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": GEMINI_GEN_CONFIG}''',
    '''    body = dict(GROQ_PARAMS, model=GROQ_MODEL, messages=[{"role": "user", "content": prompt}])''')
rep('''            out = _gemini_http(GEMINI_MODEL, body, RATER_TIMEOUT)
            cand = (out.get("candidates") or [{}])[0]
            text = "".join(p.get("text", "") for p in (cand.get("content", {}).get("parts") or []) if not p.get("thought"))''',
    '''            out = _groq_http(GROQ_MODEL, body, RATER_TIMEOUT)
            text = (((out.get("choices") or [{}])[0].get("message") or {}).get("content") or "")''')
rep('''            if e.code in (429, 500, 502, 503, 504) and attempt < RATER_RETRIES:
                _t.sleep(delay)''', '''            if e.code in (429, 500, 502, 503, 504) and attempt < RATER_RETRIES:
                try:
                    ra = float(e.headers.get("retry-after")) if e.headers and e.headers.get("retry-after") else delay
                except Exception:
                    ra = delay
                _t.sleep(min(ra, 10.0))''')
rep("RATER_RPM_PER_PROCESS = 40", "RATER_RPM_PER_PROCESS = __RPM__")
assert not [l for l in t.splitlines() if "gemini" in l.lower() and "Gemini 2.5 Pro" not in l], "leftover Gemini reference"
(D / "rater_groq_appendix.py.txt").write_text(t, encoding="utf-8")
print(len(t), "chars written")
