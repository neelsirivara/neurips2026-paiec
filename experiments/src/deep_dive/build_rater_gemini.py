"""Build the Gemini item-rating candidate ZIP (xp_level_m050 + Gemini item offsets): ONLY after the Gemini gate passed.

    python -I build_rater_gemini.py EVAL_JSON PILOT_JSON [--name xp_m050_gemini]
    python -I build_rater_gemini.py --plumbing-test     # made-up constants + mocked API; archive deleted after the test

The ZIP carries submission_config.json with an EMPTY key ({"gemini_api_key": ""}). With an empty key the rater stays off
and the archive predicts exactly like xp_level_m050. The user puts the key into that file and uploads that archive
himself. No key ever passes through this script.

Checks:
  * the organisers' validator;
  * smoke_test.py;
  * official_loop_check.py (empty key -> must equal m050 exactly);
  * mocked-API functional tests in a temporary copy with a FAKE key (_gemini_http replaced, no network):
      (1) budget 0: the offset equals w (s1 - mu)/sd + b for the mocked p;
      (2) a labelled list: the cross-pair prediction runs, the background leader rates the entries and the fit picks
          them up;
      (3) every call fails (HTTP 500 / timeout): predictions equal m050 exactly;
      (4) the wall budget: after RATER_WALL_BUDGET no call is made;
      (5) the rate limit: 10 sequential calls take at least 9 * 60 / RPM seconds.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
ROOT = SRC.parents[1]
CAND = ROOT / "submission" / "candidates"
RES = SRC.parent / "results" / "deep_dive" / "candidates"
BASE_ZIP_SHA = "cda9fe9cae49b68997d831f1de8dfe3d53a14ad3d1ac15281b22bd8326c7a715"
M050_LOOP_ALC = 0.217909212732859

FUNC_TEST = r'''
import importlib.util, json, os, sys, time, urllib.error
tmp = sys.argv[1]
json.dump({"gemini_api_key": "FAKE-KEY-FOR-MOCK"}, open(os.path.join(tmp, "submission_config.json"), "w"))
def load():
    spec = importlib.util.spec_from_file_location("m%d" % time.time_ns(), os.path.join(tmp, "model.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
s = {k: "" for k in ("normalized_name","provider","release_date","access_date","harness","harness_version","reasoning_effort","subject_features_extra")}
it = {"item_content": "What is 2+2?", "item_features": "", "interactors": "", "benchmark_id": "b"}
res = {}
# (1) budget 0 with a mocked answer "80"
m = load(); m.RATER_RPM_PER_PROCESS = 100000
calls = []
def ok(model, body, timeout):
    calls.append(1); return {"candidates": [{"content": {"parts": [{"text": "80"}]}}]}
m._gemini_http = ok
p0 = m._core_predict([s, it], [])
p1 = m.predict([s, it], [])
import math
q = 0.8; s1 = math.log(q / (1 - q))
z = min(max(m._subject_logit(s) + m._item_logit(it), -m.PRIOR_LOGIT_CAP), m.PRIOR_LOGIT_CAP) + m.RATER_W * (s1 - m.RATER_MU) / m.RATER_SD + m.RATER_B
res["b0_offset_ok"] = abs(p1 - m._clip(m._sigmoid(z))) < 1e-12
res["b0_calls"] = len(calls)
# (2) labelled list: leader rates entries in the background, the fit picks them up
lab = [[[s, dict(it, item_content="Q%d" % i)], i % 2] for i in range(12)]
pa = m.predict([s, it], lab)
time.sleep(1.0)
m._R["reload_t"] = 0.0
pb = m.predict([s, it], lab)
res["xp_runs"] = 0.0 < pa < 1.0 and 0.0 < pb < 1.0
res["entries_rated"] = sum(1 for e in lab if m._rater_key(m._rater_prompt(e[0][1])) in m._R["cache"])
# (3) every call fails -> plain m050 (fresh module, fresh cache dir)
import shutil
d = os.path.join(tmp, ".paiec_rater")
shutil.rmtree(d, ignore_errors=True)
m = load(); m.RATER_RPM_PER_PROCESS = 100000
m.time_sleep = None
def bad(model, body, timeout):
    raise urllib.error.HTTPError("u", 500, "err", {}, None)
m._gemini_http = bad
import time as _t
orig_sleep = _t.sleep
_t.sleep = lambda x: None
try:
    res["fail_b0_equals_m050"] = m.predict([s, it], []) == m._core_predict([s, it], [])
    res["fail_xp_equals_m050"] = m.predict([s, it], lab) == m._core_predict([s, it], lab)
finally:
    _t.sleep = orig_sleep
# (4) wall budget: no call after the budget
shutil.rmtree(d, ignore_errors=True)
m = load(); m.RATER_RPM_PER_PROCESS = 100000; m.RATER_WALL_BUDGET = -1.0
calls.clear(); m._gemini_http = ok
m.predict([s, it], [])
res["no_calls_after_budget"] = len(calls) == 0
# (5) rate limit: 10 scores at 600 RPM take >= 0.9 s
shutil.rmtree(d, ignore_errors=True)
m = load(); m.RATER_RPM_PER_PROCESS = 600; m._gemini_http = ok
m._rater_start()
t0 = time.time()
for i in range(10):
    m._rater_score("prompt %d" % i)
res["rate_limit_ok"] = time.time() - t0 >= 0.85
print(json.dumps(res))
'''


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    args = sys.argv[1:]
    test = "--plumbing-test" in args
    if test:
        name = "test_gemini_DO_NOT_UPLOAD"
        c = {"mu": 0.0, "sd": 2.0, "w": 0.3, "b": 0.1}
        gen = {"temperature": 0.0, "maxOutputTokens": 16, "thinkingConfig": {"thinkingBudget": 0}}
    else:
        ev = json.loads(Path(args[0]).read_text(encoding="utf-8"))
        pilot = json.loads(Path(args[1]).read_text(encoding="utf-8"))
        name = args[args.index("--name") + 1] if "--name" in args else "xp_m050_gemini"
        ok = (ev.get("gate", {}).get("pass") and ev.get("gate_vs_b_only", {}).get("pass")
              and ev.get("gate_c_h7inv_not_sig_worse", False)
              and -ev["scenarios"]["INV"]["ctrl_diff"]["mean"] >= 0.003)
        if not ok:
            raise SystemExit("Gemini gate (or the 0.003 bar) did not pass: not building")
        c = ev["deploy_fit"]
        gen = pilot["generation_config"]
    assert sha((CAND / "xp_level_m050.zip").read_bytes()) == BASE_ZIP_SHA
    with ZipFile(CAND / "xp_level_m050.zip") as zf:
        base_model = zf.read("model.py").decode("utf-8")
        labeling = zf.read("labeling.py")
        readme = zf.read("README.md").decode("utf-8")
    app = (HERE / "rater_gemini_appendix.py.txt").read_text(encoding="utf-8")
    for k, v in (("__RATER_W__", c["w"]), ("__RATER_B__", c["b"]), ("__RATER_MU__", c["mu"]), ("__RATER_SD__", c["sd"])):
        app = app.replace(k, repr(float(v)))
    app = app.replace("__GEN_CONFIG__", json.dumps(gen))
    model_src = base_model.rstrip("\n") + "\n" + app
    note = ("\n\n## Gemini item-rating offsets\n\n`xp_level_m050` plus an item-difficulty offset from gemini-2.5-flash "
            "(verbalised probability; pre-registered gate in the code repository). The API key is read only from "
            "`submission_config.json`. This archive ships it EMPTY, which keeps the rater off, so the archive then "
            "predicts exactly like xp_level_m050. Any API failure also falls back to xp_level_m050.\n")
    files = {"model.py": model_src.encode("utf-8"), "labeling.py": labeling, "README.md": (readme + note).encode("utf-8"),
             "submission_config.json": json.dumps({"gemini_api_key": ""}, indent=1).encode("utf-8")}
    d = CAND / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for k, v in files.items():
        (d / k).write_bytes(v)
    zpath = CAND / f"{name}.zip"
    with ZipFile(zpath, "w", compression=ZIP_DEFLATED) as archive:
        for k, v in files.items():
            e = ZipInfo(k)
            e.compress_type = ZIP_DEFLATED
            e.external_attr = 0o100644 << 16
            archive.writestr(e, v)
    rdir = RES / name
    rdir.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    info = {"zip": str(zpath), "bytes": zpath.stat().st_size, "sha256": sha(zpath.read_bytes()), "constants": c,
            "generation_config": gen, "test": test}
    p = subprocess.run([py, "-I", str(SRC / "validator" / "check_submission_zip.py"), str(zpath)], capture_output=True, text=True)
    (rdir / "validator_output.txt").write_text(p.stdout + p.stderr + f"\n[exit code {p.returncode}]\n", encoding="utf-8")
    info["validator_exit"] = p.returncode
    env = {k: v for k, v in os.environ.items() if k not in ("PREDICTIVE_EVAL_LOCAL_SMOKE_TEST", "PAIEC_RATER_DISABLE")}
    p = subprocess.run([py, "-I", str(SRC / "smoke_test.py"), "--dir", str(d), "--out", str(rdir / "smoke_test.json")],
                       capture_output=True, text=True, cwd=str(SRC), env=env)
    info["smoke_exit"] = p.returncode
    (rdir / "smoke_test_stdout.txt").write_text(p.stdout + p.stderr, encoding="utf-8")
    rec = SRC.parent / "results" / "round4" / "official_loop_check.json"
    backup = rec.read_bytes()
    try:
        p = subprocess.run([py, "-I", str(SRC / "official_loop_check.py"), "--dir", str(d)], capture_output=True, text=True,
                           cwd=str(SRC), env=env)
        shutil.copyfile(rec, rdir / "official_loop_check.json")
    finally:
        rec.write_bytes(backup)
    info["official_loop_exit"] = p.returncode
    info["official_loop_alc"] = json.loads((rdir / "official_loop_check.json").read_text())["alc_mean"]
    info["official_loop_equals_m050"] = abs(info["official_loop_alc"] - M050_LOOP_ALC) < 1e-12
    with tempfile.TemporaryDirectory() as tmp:
        for k, v in files.items():
            Path(tmp, k).write_bytes(v)
        p = subprocess.run([py, "-I", "-c", FUNC_TEST, tmp], capture_output=True, text=True, env=env, timeout=300)
        out = (p.stdout.strip().splitlines() or [""])[-1]
        try:
            info["mock_tests"] = json.loads(out)
        except Exception:
            info["mock_tests"] = {"error": (p.stdout + p.stderr)[-600:]}
    (rdir / "candidate_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    assert sha((CAND / "xp_level_m050.zip").read_bytes()) == BASE_ZIP_SHA
    if test:
        shutil.rmtree(d)
        zpath.unlink()
        info["test_archive_deleted"] = True
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
