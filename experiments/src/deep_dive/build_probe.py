"""Build the two deployment-probe ZIPs (A: diagnostics only; B: failure-code signal) on top of xp_level_m050.

    python -I build_probe.py

Writes submission/candidates/probe_{A,B}/ and probe_{A,B}.zip: model.py = xp_level_m050's model.py + probe_appendix.py.txt
(mode substituted), labeling.py and README from xp_level_m050, models.txt = the probe model id.  Then runs, with
``python -I``: the organisers' validator; smoke_test.py and official_loop_check.py with the probe disabled
(PREDICTIVE_EVAL_LOCAL_SMOKE_TEST=1, as the organisers' own local check sets it); and a functional test of the probe
itself in a temporary copy (this laptop has no torch, so mode A must print class "import" and still return a valid
probability, and mode B must return NaN).  Results: experiments/results/deep_dive/candidates/probe_{A,B}/.
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
BASE = ROOT / "submission" / "candidates" / "xp_level_m050"
CAND = ROOT / "submission" / "candidates"
RES = SRC.parent / "results" / "deep_dive" / "candidates"
BASE_ZIP_SHA = "cda9fe9cae49b68997d831f1de8dfe3d53a14ad3d1ac15281b22bd8326c7a715"
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def build(mode: str) -> dict:
    assert sha((CAND / "xp_level_m050.zip").read_bytes()) == BASE_ZIP_SHA
    with ZipFile(CAND / "xp_level_m050.zip") as zf:
        base_model = zf.read("model.py").decode("utf-8")
        labeling = zf.read("labeling.py")
        readme = zf.read("README.md").decode("utf-8")
    appendix = (HERE / "probe_appendix.py.txt").read_text(encoding="utf-8").replace("__PROBE_MODE__", mode)
    model_src = base_model.rstrip("\n") + "\n" + appendix
    note = (f"\n\n## Deployment probe {mode} (2026-10-26)\n\nThis archive is `xp_level_m050` plus a one-off deployment "
            f"probe appended to `model.py` and `models.txt` = `{MODEL_ID}`.  One worker per run tries to load that model "
            "(bfloat16, CUDA, local files only, 300 s cap) and runs one forward pass on a fixed dummy prompt; no test data "
            "is used or printed.  " + {
                "A": "Mode A only prints a diagnostic line and always returns the m050 prediction.",
                "B": "Mode B signals failure through the platform's failure codes (42: import / no CUDA; 40: load or forward "
                     "failure or time cap) in that one call.",
                "C": "Mode C fires only after a successful load: code 42 if the GPU has >= 70 GB, code 40 otherwise; a valid "
                     "score means the probe did not run or the load failed."}[mode] + "\n")
    files = {"model.py": model_src.encode("utf-8"), "labeling.py": labeling, "models.txt": (MODEL_ID + "\n").encode(),
             "README.md": (readme + note).encode("utf-8")}
    d = CAND / f"probe_{mode}"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for k, v in files.items():
        (d / k).write_bytes(v)
    zpath = CAND / f"probe_{mode}.zip"
    with ZipFile(zpath, "w", compression=ZIP_DEFLATED) as archive:
        for k, v in files.items():
            e = ZipInfo(k)
            e.compress_type = ZIP_DEFLATED
            e.external_attr = 0o100644 << 16
            archive.writestr(e, v)
    rdir = RES / f"probe_{mode}"
    rdir.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    env_off = dict(os.environ, PREDICTIVE_EVAL_LOCAL_SMOKE_TEST="1", PAIEC_PROBE_DISABLE="1")
    info = {"zip": str(zpath), "bytes": zpath.stat().st_size, "sha256": sha(zpath.read_bytes()), "members": sorted(files),
            "base": "xp_level_m050", "base_zip_sha256": BASE_ZIP_SHA, "models_txt": MODEL_ID, "mode": mode}
    p = subprocess.run([py, "-I", str(SRC / "validator" / "check_submission_zip.py"), str(zpath)], capture_output=True, text=True)
    (rdir / "validator_output.txt").write_text(p.stdout + p.stderr + f"\n[exit code {p.returncode}]\n", encoding="utf-8")
    info["validator_exit"] = p.returncode
    p = subprocess.run([py, "-I", str(SRC / "smoke_test.py"), "--dir", str(d), "--out", str(rdir / "smoke_test.json")],
                       capture_output=True, text=True, cwd=str(SRC), env=env_off)
    info["smoke_exit"] = p.returncode
    (rdir / "smoke_test_stdout.txt").write_text(p.stdout + p.stderr, encoding="utf-8")
    rec = SRC.parent / "results" / "round4" / "official_loop_check.json"
    backup = rec.read_bytes()
    try:
        p = subprocess.run([py, "-I", str(SRC / "official_loop_check.py"), "--dir", str(d)], capture_output=True, text=True,
                           cwd=str(SRC), env=env_off)
        shutil.copyfile(rec, rdir / "official_loop_check.json")
    finally:
        rec.write_bytes(backup)
    info["official_loop_exit"] = p.returncode
    info["official_loop_alc"] = json.loads((rdir / "official_loop_check.json").read_text())["alc_mean"]
    # functional test of the probe in a temporary copy (no torch on this machine -> class "import")
    with tempfile.TemporaryDirectory() as tmp:
        for k, v in files.items():
            Path(tmp, k).write_bytes(v)
        code = ("import importlib.util,json,math,sys;"
                f"spec=importlib.util.spec_from_file_location('m',r'{tmp}/model.py');m=importlib.util.module_from_spec(spec);"
                "spec.loader.exec_module(m);"
                "s={k:'' for k in ('normalized_name','provider','release_date','access_date','harness','harness_version','reasoning_effort','subject_features_extra')};"
                "it={'item_content':'What is 2+2?','item_features':'','interactors':'','benchmark_id':'b'};"
                "v1=m.predict([s,it],[]);v2=m.predict([s,it],[]);"
                "print(json.dumps({'first':None if v1!=v1 else v1,'first_is_nan':v1!=v1,'second':v2}))")
        env = {k: v for k, v in os.environ.items() if k not in ("PREDICTIVE_EVAL_LOCAL_SMOKE_TEST", "PAIEC_PROBE_DISABLE")}
        p = subprocess.run([py, "-I", "-c", code], capture_output=True, text=True, env=env)
        out = p.stdout.strip().splitlines()
        info["functional_test"] = {"exit": p.returncode, "stdout_tail": out[-2:], "lock_file_created": Path(tmp, ".paiec_probe_lock").exists()}
    (rdir / "probe_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    assert sha((CAND / "xp_level_m050.zip").read_bytes()) == BASE_ZIP_SHA
    return info


if __name__ == "__main__":
    modes = sys.argv[1:] or ["A", "B"]
    res = {m: build(m) for m in modes}
    print(json.dumps(res, indent=1))
