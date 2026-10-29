"""Build a W2-sem candidate = the W2 candidate's model.py (W2 constants unchanged, the char kernel stays as fallback)
+ the semantic-kernel block (w2s_block.py.txt) + models.txt, then run the same checks as W2.

    python -I build_w2s.py NAME --kind sem|both --q-s Q --tau-sem T [--topk K] [--model minilm] [--checks] [--outdir D]

W2 part: tau_f 0.5, share 1, char kernel tau 1, q 2 (the shipped W2 candidate w2_f05s1_k1q2).  Writes
<root>/NAME/{model.py, labeling.py, README.md, models.txt} and <root>/NAME.zip (root = submission/candidates unless
--outdir).  Checks as build_w2.py; the smoke test and the loop check load the embedding model from the local
Hugging Face cache (CPU).
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
sys.path.insert(0, str(HERE))
import build_w2 as B  # noqa: E402

MODELS = {"minilm": ("sentence-transformers/all-MiniLM-L6-v2", "mean")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--kind", choices=("sem", "both"), required=True)
    ap.add_argument("--q-s", type=float, required=True)
    ap.add_argument("--tau-sem", type=float, required=True)
    ap.add_argument("--topk", type=int, default=0)
    ap.add_argument("--m-low", type=float, default=1.0)
    ap.add_argument("--tau-feat", type=float, default=0.5)
    ap.add_argument("--model", default="minilm")
    ap.add_argument("--checks", action="store_true")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    zbytes = B.BASE_ZIP.read_bytes()
    if B.sha(zbytes) != B.BASE_SHA:
        raise SystemExit("xp_level_m050.zip is not the validated candidate (sha mismatch); stopping")
    shipped_before = B.sha(B.SHIPPED.read_bytes())
    with ZipFile(B.BASE_ZIP) as zf:
        model_src = zf.read("model.py").decode("utf-8")
        labeling_src = zf.read("labeling.py")
        readme_src = zf.read("README.md").decode("utf-8")
    m = B.patch_model(model_src, a.tau_feat, 1.0, 1.0, 2.0)
    crlf = "\r\n" in m
    repo, pooling = MODELS[a.model]
    block = (HERE / "w2s_block.py.txt").read_text(encoding="utf-8")
    block = (block.replace("__W2S_MODEL_ID__", repo).replace("__W2S_POOLING__", pooling).replace("__W2S_KIND__", a.kind)
             .replace("__W2S_Q_S__", repr(float(a.q_s))).replace("__W2S_TAU_KERN__", repr(float(a.tau_sem)))
             .replace("__W2S_TOPK__", repr(int(a.topk))).replace("__W2S_M_LOW__", repr(float(a.m_low))))
    assert "__W2S_" not in block
    if crlf:
        block = block.replace("\n", "\r\n")
    m = m + block
    note = (f"\n\n## Candidate `{a.name}` (W2-sem, 2026-10-29)\n\nThe W2 candidate (`w2_f05s1_k1q2`) with the text kernel's "
            f"similarity {'replaced by' if a.kind == 'sem' else 'mixed 50/50 with'} the cosine of `{repo}` sentence embeddings "
            f"(exponent {a.q_s:g}, prior sd {a.tau_sem:g}{', top ' + str(a.topk) if a.topk else ''}).  The model is listed in "
            f"`models.txt` and loaded from local files only; any failure falls back to the W2 character kernel, then to the "
            f"plain cross-pair model.  Simulations and pre-registration: `experiments/src/w2_items/`, "
            f"`experiments/results/w2_items/` of the code repository.\n")
    root = Path(a.outdir).resolve() if a.outdir else B.CAND
    out_dir = root / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {"model.py": m.encode("utf-8"), "labeling.py": labeling_src, "README.md": (readme_src + note).encode("utf-8"),
             "models.txt": (repo + "\n").encode("utf-8")}
    for k, v in files.items():
        (out_dir / k).write_bytes(v)
    zpath = root / f"{a.name}.zip"
    with ZipFile(zpath, "w", compression=ZIP_DEFLATED) as archive:
        for k, v in files.items():
            e = ZipInfo(k)
            e.compress_type = ZIP_DEFLATED
            e.external_attr = 0o100644 << 16
            archive.writestr(e, v)
    rdir = (B.RES if not a.outdir else root / "records") / a.name
    rdir.mkdir(parents=True, exist_ok=True)
    info = {"zip": str(zpath), "bytes": zpath.stat().st_size, "sha256": B.sha(zpath.read_bytes()), "members": sorted(files),
            "model_sha256": B.sha(files["model.py"]), "base_zip_sha256": B.BASE_SHA, "models_txt": [repo],
            "constants": {"W2_TAU_FEAT": a.tau_feat, "W2S_M_LOW": a.m_low, "W2_FEAT_SHARE": 1.0, "W2_TAU_KERN": 1.0, "W2_KERN_Q": 2.0,
                          "W2S_KIND": a.kind, "W2S_Q_S": a.q_s, "W2S_TAU_KERN": a.tau_sem, "W2S_TOPK": a.topk,
                          "W2S_MODEL_ID": repo, "W2S_POOLING": pooling}}
    log = []
    if a.checks:
        py = sys.executable
        p = subprocess.run([py, "-I", str(B.SRC / "validator" / "check_submission_zip.py"), str(zpath)], capture_output=True, text=True)
        (rdir / "validator_output.txt").write_text(p.stdout + p.stderr + f"\n[exit code {p.returncode}]\n", encoding="utf-8")
        info["validator_exit"] = p.returncode
        log.append(("validator", p.returncode, (p.stdout + p.stderr).strip()[-300:]))
        p = subprocess.run([py, "-I", str(B.SRC / "smoke_test.py"), "--dir", str(out_dir), "--out", str(rdir / "smoke_test.json")],
                           capture_output=True, text=True, cwd=str(B.SRC))
        (rdir / "smoke_test_stdout.txt").write_text(p.stdout + p.stderr, encoding="utf-8")
        info["smoke_exit"] = p.returncode
        log.append(("smoke_test", p.returncode, (p.stdout + p.stderr).strip()[-300:]))
        rec = B.SRC.parent / "results" / "round4" / "official_loop_check.json"
        backup = rec.read_bytes() if rec.exists() else None
        try:
            p = subprocess.run([py, "-I", str(B.SRC / "official_loop_check.py"), "--dir", str(out_dir)],
                               capture_output=True, text=True, cwd=str(B.SRC))
            if rec.exists():
                shutil.copyfile(rec, rdir / "official_loop_check.json")
        finally:
            if backup is not None:
                rec.write_bytes(backup)
        info["official_loop_record_restored"] = backup is None or rec.read_bytes() == backup
        (rdir / "official_loop_check_stdout.txt").write_text(p.stdout + p.stderr, encoding="utf-8")
        info["official_loop_exit"] = p.returncode
        log.append(("official_loop_check", p.returncode, (p.stdout + p.stderr).strip()[-300:]))
    info["base_zip_sha256_after"] = B.sha(B.BASE_ZIP.read_bytes())
    info["shipped_zip_sha256_before"] = shipped_before
    info["shipped_zip_sha256_after"] = B.sha(B.SHIPPED.read_bytes())
    (rdir / "candidate_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    for name, code, tail in log:
        print(f"--- {name}: exit {code}\n{tail}\n")
    print(json.dumps(info, indent=1))
    return 0 if all(c == 0 for _, c, _ in log) else 1


if __name__ == "__main__":
    raise SystemExit(main())
