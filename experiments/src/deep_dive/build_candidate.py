"""Build a candidate PAIEC ZIP that differs from the shipped one ONLY in budget-0 level constants, then validate it.

    python build_candidate.py NAME --intercept D [--subject-shrink S] [--tau-mu M]

Starts from the model.py / labeling.py members of the SHIPPED zip (submission/paiec_submission.zip, never
modified), replaces exactly these constant lines in model.py (each must occur exactly once):
    SUBJECT_SHRINK = 0.7   -> S          (multiplier on the hand-set subject terms of the prior logit)
    PRIOR_INTERCEPT = 0.0  -> D          (global shift of the prior logit)
    ("mu", 0.5) in XP_TAUS -> ("mu", M)  (prior sd of the cross-pair global level)
and writes submission/candidates/NAME/{model.py, labeling.py, README.md} and submission/candidates/NAME.zip.
Then runs, all with ``python -I``:
  1. the organisers' validator (experiments/src/validator/check_submission_zip.py, verbatim copy) on the ZIP;
  2. experiments/src/smoke_test.py --dir <candidate dir> --out <candidate results>/smoke_test.json;
  3. experiments/src/official_loop_check.py --dir <candidate dir> (the organisers' streaming client with a mock
     coordinator on the 17-pair real testbed); that script always writes results/round4/official_loop_check.json,
     so the shipped zip's record is backed up first and restored afterwards, and the candidate's copy is kept
     in the candidate results folder.
Outputs: experiments/results/deep_dive/candidates/NAME/*.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
ROOT = SRC.parents[1]
SHIPPED = ROOT / "submission" / "paiec_submission.zip"
CAND = ROOT / "submission" / "candidates"
RES = SRC.parent / "results" / "deep_dive" / "candidates"
SHIPPED_SHA = "c76331d084f175fbf8ce272b00d0a2360d80e440dacf6ede380f3f9eba5172e1"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def replace_once(text: str, old: str, new: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"expected exactly one occurrence of {old!r}, found {n}")
    return text.replace(old, new)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--intercept", type=float, required=True)
    ap.add_argument("--subject-shrink", type=float, default=0.0)
    ap.add_argument("--tau-mu", type=float, default=2.0)
    a = ap.parse_args()
    zbytes = SHIPPED.read_bytes()
    if sha(zbytes) != SHIPPED_SHA:
        raise SystemExit("the shipped zip is not the validated round-4 zip (sha mismatch); stopping")
    with ZipFile(SHIPPED) as zf:
        model_src = zf.read("model.py").decode("utf-8")
        labeling_src = zf.read("labeling.py")
        readme_src = zf.read("README.md").decode("utf-8")
    m = model_src
    m = replace_once(m, "SUBJECT_SHRINK = 0.7   #", f"SUBJECT_SHRINK = {a.subject_shrink!r}   #")
    m = replace_once(m, "PRIOR_INTERCEPT = 0.0        #", f"PRIOR_INTERCEPT = {a.intercept!r}        #")
    m = replace_once(m, 'XP_TAUS = (("mu", 0.5),', f'XP_TAUS = (("mu", {a.tau_mu!r}),')
    note = (f"\n\n## Candidate `{a.name}` (deep dive 2026-10-26)\n\nIdentical to the shipped round-4 submission "
            f"(sha256 {SHIPPED_SHA[:16]}...) except three constants in `model.py`: `SUBJECT_SHRINK` 0.7 -> {a.subject_shrink:g} "
            f"(hand-set subject terms of the budget-0 prior), `PRIOR_INTERCEPT` 0.0 -> {a.intercept:g} (global level of the "
            f"prior logit) and the prior sd of the cross-pair global level 0.5 -> {a.tau_mu:g}.  Rationale and simulations: "
            f"`DEEP_DIVE.md` of the code repository.\n")
    out_dir = CAND / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {"model.py": m.encode("utf-8"), "labeling.py": labeling_src, "README.md": (readme_src + note).encode("utf-8")}
    for k, v in files.items():
        (out_dir / k).write_bytes(v)
    zpath = CAND / f"{a.name}.zip"
    with ZipFile(zpath, "w", compression=ZIP_DEFLATED) as archive:
        for k, v in files.items():
            e = ZipInfo(k)
            e.compress_type = ZIP_DEFLATED
            e.external_attr = 0o100644 << 16
            archive.writestr(e, v)
    rdir = RES / a.name
    rdir.mkdir(parents=True, exist_ok=True)
    info = {"zip": str(zpath), "bytes": zpath.stat().st_size, "sha256": sha(zpath.read_bytes()),
            "members": sorted(files), "model_sha256": sha(files["model.py"]),
            "labeling_identical_to_shipped": True, "constants": {"SUBJECT_SHRINK": a.subject_shrink,
                                                                  "PRIOR_INTERCEPT": a.intercept, "XP_TAUS.mu": a.tau_mu}}
    py = sys.executable
    log = []
    # 1. official validator
    p = subprocess.run([py, "-I", str(SRC / "validator" / "check_submission_zip.py"), str(zpath)], capture_output=True, text=True)
    (rdir / "validator_output.txt").write_text(p.stdout + p.stderr + f"\n[exit code {p.returncode}]\n", encoding="utf-8")
    info["validator_exit"] = p.returncode
    log.append(("validator", p.returncode, (p.stdout + p.stderr).strip()[-300:]))
    # 2. smoke test
    p = subprocess.run([py, "-I", str(SRC / "smoke_test.py"), "--dir", str(out_dir), "--out", str(rdir / "smoke_test.json")],
                       capture_output=True, text=True, cwd=str(SRC))
    (rdir / "smoke_test_stdout.txt").write_text(p.stdout + p.stderr, encoding="utf-8")
    info["smoke_exit"] = p.returncode
    log.append(("smoke_test", p.returncode, (p.stdout + p.stderr).strip()[-300:]))
    # 3. official streaming client on the real testbed (back up / restore the shipped zip's record)
    rec = SRC.parent / "results" / "round4" / "official_loop_check.json"
    backup = rec.read_bytes() if rec.exists() else None
    try:
        p = subprocess.run([py, "-I", str(SRC / "official_loop_check.py"), "--dir", str(out_dir)],
                           capture_output=True, text=True, cwd=str(SRC))
        if rec.exists():
            shutil.copyfile(rec, rdir / "official_loop_check.json")
    finally:
        if backup is not None:
            rec.write_bytes(backup)
    (rdir / "official_loop_check_stdout.txt").write_text(p.stdout + p.stderr, encoding="utf-8")
    info["official_loop_exit"] = p.returncode
    log.append(("official_loop_check", p.returncode, (p.stdout + p.stderr).strip()[-300:]))
    info["shipped_zip_sha256_after"] = sha(SHIPPED.read_bytes())
    (rdir / "candidate_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    for name, code, tail in log:
        print(f"--- {name}: exit {code}\n{tail}\n")
    print(json.dumps(info, indent=1))
    return 0 if all(c == 0 for _, c, _ in log) else 1


if __name__ == "__main__":
    raise SystemExit(main())
