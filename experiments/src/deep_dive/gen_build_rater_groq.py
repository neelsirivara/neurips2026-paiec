"""Generate deep_dive/build_rater_groq.py from build_rater_gemini.py (same checks, Groq constants and mocks)."""
from pathlib import Path

D = Path(__file__).resolve().parent
s = (D / "build_rater_gemini.py").read_text(encoding="utf-8")


def rep(old, new):
    global s
    assert s.count(old) >= 1, old[:80]
    s = s.replace(old, new)


rep('"""Build the Gemini item-rating candidate ZIP (xp_level_m050 + Gemini item offsets): ONLY after the Gemini gate passed.',
    '"""Build the Groq item-rating candidate ZIP (xp_level_m050 + Groq item offsets): ONLY after the Groq gate passed.')
rep("python -I build_rater_gemini.py EVAL_JSON PILOT_JSON [--name xp_m050_gemini]",
    "python -I build_rater_groq.py EVAL_JSON PILOT_JSON [--name xp_m050_groq]")
rep("python -I build_rater_gemini.py --plumbing-test", "python -I build_rater_groq.py --plumbing-test")
rep('({"gemini_api_key": ""})', '({"groq_api_key": ""})')
rep('json.dump({"gemini_api_key": "FAKE-KEY-FOR-MOCK"}', 'json.dump({"groq_api_key": "FAKE-KEY-FOR-MOCK"}')
rep('calls.append(1); return {"candidates": [{"content": {"parts": [{"text": "80"}]}}]}',
    'calls.append(1); return {"choices": [{"message": {"content": "80"}}]}')
rep("m._gemini_http = ok", "m._groq_http = ok")
rep("m._gemini_http = bad", "m._groq_http = bad")
rep('''        name = "test_gemini_DO_NOT_UPLOAD"
        c = {"mu": 0.0, "sd": 2.0, "w": 0.3, "b": 0.1}
        gen = {"temperature": 0.0, "maxOutputTokens": 16, "thinkingConfig": {"thinkingBudget": 0}}''',
    '''        name = "test_groq_DO_NOT_UPLOAD"
        c = {"mu": 0.0, "sd": 2.0, "w": 0.3, "b": 0.1}
        model_id = "llama-3.3-70b-versatile"
        gen = {"temperature": 0.0, "max_tokens": 8}
        rpm = 40''')
rep('''if "--name" in args else "xp_m050_gemini"''', '''if "--name" in args else "xp_m050_groq"''')
rep('''            raise SystemExit("Gemini gate (or the 0.003 bar) did not pass: not building")
        c = ev["deploy_fit"]
        gen = pilot["generation_config"]''', '''            raise SystemExit("Groq gate (or the 0.003 bar) did not pass: not building")
        c = ev["deploy_fit"]
        model_id = pilot["chosen_model"]
        sys.path.insert(0, str(HERE))
        import rate_items_groq as RG
        gen = RG.PARAMS[model_id]
        feas = pilot["candidates"][model_id]["feasibility"]
        rpm = max(1, int(feas["prompts_per_minute"] / 16))   # org-level limit shared by up to 16 workers''')
rep('''(HERE / "rater_gemini_appendix.py.txt")''', '''(HERE / "rater_groq_appendix.py.txt")''')
rep('''    app = app.replace("__GEN_CONFIG__", json.dumps(gen))''', '''    app = app.replace("__GEN_CONFIG__", json.dumps(gen)).replace("__GROQ_MODEL__", json.dumps(model_id))
    app = app.replace("__RPM__", str(int(rpm)))''')
rep("## Gemini item-rating offsets", "## Groq item-rating offsets")
rep('item-difficulty offset from gemini-2.5-flash "', 'item-difficulty offset from " + model_id + " on Groq "')
rep('''            "generation_config": gen, "test": test}''', '''            "generation_config": gen, "model": model_id, "rpm_per_process": rpm, "test": test}''')
rep("(_gemini_http replaced, no network)", "(_groq_http replaced, no network)")
rep('json.dumps({"gemini_api_key": ""}, indent=1)', 'json.dumps({"groq_api_key": ""}, indent=1)')
left = [l for l in s.splitlines() if "emini" in l]
assert not left, left[:5]
(D / "build_rater_groq.py").write_text(s, encoding="utf-8")
print("written")
