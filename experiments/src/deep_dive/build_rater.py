"""Build the item-rating candidate ZIP (xp_level_m050 + Qwen3-VL-4B item offsets), ONLY after the gate passed.

    python -I build_rater.py EVAL_JSON [--name xp_m050_rater] [--force-test-constants]

Reads the deployment constants (all-fold fit: mu, sd, w, b) from the eval_items JSON. Refuses to build unless
gate["pass"] and gate_vs_b_only["pass"] are both true, except with --force-test-constants, which writes the archive under
the name test_rater_DO_NOT_UPLOAD with made-up constants, for the plumbing tests only. Writes
submission/candidates/<name>/{model.py, labeling.py, models.txt, README.md} + <name>.zip and runs the same checks as
the other candidates:
  * the organisers' validator;
  * smoke_test.py;
  * official_loop_check.py (no GPU here, so the rater turns itself off and the predictions must equal m050);
  * a functional test of the offset path with a fake rating cache: ratings are injected into the cache directory, and
    the predictions must move by exactly the expected offset at budget 0.
"""
from __future__ import annotations

import hashlib
import json
import math
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
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    args = sys.argv[1:]
    test = "--force-test-constants" in args
    ev = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    name = args[args.index("--name") + 1] if "--name" in args else "xp_m050_rater"
    if test:
        name = "test_rater_DO_NOT_UPLOAD"
        c = {"mu": 7.0, "sd": 6.6, "w": 0.3, "b": 0.1}
    else:
        if not (ev.get("gate", {}).get("pass") and ev.get("gate_vs_b_only", {}).get("pass")):
            raise SystemExit("gate did not pass: not building")
        c = ev["deploy_fit"]
    assert sha((CAND / "xp_level_m050.zip").read_bytes()) == BASE_ZIP_SHA
    with ZipFile(CAND / "xp_level_m050.zip") as zf:
        base_model = zf.read("model.py").decode("utf-8")
        labeling = zf.read("labeling.py")
        readme = zf.read("README.md").decode("utf-8")
    app = (HERE / "rater_appendix.py.txt").read_text(encoding="utf-8")
    for k, v in (("__RATER_W__", c["w"]), ("__RATER_B__", c["b"]), ("__RATER_MU__", c["mu"]), ("__RATER_SD__", c["sd"])):
        app = app.replace(k, repr(float(v)))
    model_src = base_model.rstrip("\n") + "\n" + app
    note = (f"\n\n## Item-rating offsets ({name})\n\n`xp_level_m050` plus an LLM item-difficulty offset: `{MODEL_ID}` "
            "(models.txt) rates each item once (one forward pass, cached on disk across workers), and the item's prior "
            f"logit gets w*(s1-mu)/sd + b with w={c['w']:.4f}, b={c['b']:.4f} (fitted leave-benchmark-out on the training "
            "release; pre-registered gate in the code repository). Any failure falls back to plain xp_level_m050.\n")
    files = {"model.py": model_src.encode("utf-8"), "labeling.py": labeling, "models.txt": (MODEL_ID + "\n").encode(),
             "README.md": (readme + note).encode("utf-8")}
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
            "models_txt": MODEL_ID, "test_constants": test}
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
    info["official_loop_equals_m050"] = abs(info["official_loop_alc"] - 0.217909212732859) < 1e-12
    # functional test of the offset path: fake cache with one rating for the test item, no GPU
    with tempfile.TemporaryDirectory() as tmp:
        for k, v in files.items():
            Path(tmp, k).write_bytes(v)
        code = r'''
import importlib.util, json, os, sys, math
tmp = sys.argv[1]
spec = importlib.util.spec_from_file_location("m", os.path.join(tmp, "model.py")); m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s = {k: "" for k in ("normalized_name","provider","release_date","access_date","harness","harness_version","reasoning_effort","subject_features_extra")}
it = {"item_content": "What is 2+2?", "item_features": "", "interactors": "", "benchmark_id": "b"}
p_before = m._core_predict([s, it], [])
d = m._rater_dir(); key = m._rater_key(m._rater_prompt(it))
open(os.path.join(d, "ratings_fake.jsonl"), "w").write("%s\t%r\n" % (key, 10.0))
p_after = m.predict([s, it], [])
z = min(max(m._subject_logit(s) + m._item_logit(it), -m.PRIOR_LOGIT_CAP), m.PRIOR_LOGIT_CAP) + m.RATER_W * (10.0 - m.RATER_MU) / m.RATER_SD + m.RATER_B
lab = [[[s, dict(it, item_content="Q%d" % i)], i % 2] for i in range(6)]
p_xp = m.predict([s, it], lab)
print(json.dumps({"p_before": p_before, "p_after": p_after, "expected": m._clip(m._sigmoid(z)), "dir": d, "p_xp": p_xp,
                  "state": m._R["state"]}))
'''
        p = subprocess.run([py, "-I", "-c", code, tmp], capture_output=True, text=True, env=env)
        out = (p.stdout.strip().splitlines() or [""])[-1]
        info["functional_test"] = {"exit": p.returncode, "stdout": out, "stderr_tail": p.stderr[-400:]}
        try:
            r = json.loads(out)
            info["functional_test"]["offset_applied_correctly"] = abs(r["p_after"] - r["expected"]) < 1e-12
        except Exception:
            info["functional_test"]["offset_applied_correctly"] = False
    (rdir / "candidate_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    assert sha((CAND / "xp_level_m050.zip").read_bytes()) == BASE_ZIP_SHA
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
