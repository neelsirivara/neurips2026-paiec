"""Build a W2 candidate ZIP = xp_level_m050 + the W2 item layer, then run the same checks as the m050 candidate.

    python -I build_w2.py NAME --tau-feat F [--share S] [--tau-kern K] [--q Q] [--checks]

Starts from the members of submission/candidates/xp_level_m050.zip (sha256 cda9fe9c..., never modified).
model.py changes:
  1. xp_build gets an optional ``extra_types`` argument (effect types that cover only their ``members``);
  2. _xp_predict_keys gets optional ``extra_m`` / ``extra_var`` and skips such types in its own loop;
  3. the block in w2_block.py.txt is appended (constants filled in), which rebinds predict_cross_pair and
     xp_clear_cache.  With the defaults of 1 and 2 the m050 code paths compute exactly what they computed before.
labeling.py is byte-identical.  Writes submission/candidates/NAME/{model.py, labeling.py, README.md} and
submission/candidates/NAME.zip.  With --checks, runs with ``python -I``: the organisers' validator, smoke_test.py,
official_loop_check.py (its fixed output record results/round4/official_loop_check.json is backed up and restored,
the candidate's copy kept), records in experiments/results/w2_items/candidates/NAME/.
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
CAND = ROOT / "submission" / "candidates"
BASE_ZIP = CAND / "xp_level_m050.zip"
BASE_SHA = "cda9fe9cae49b68997d831f1de8dfe3d53a14ad3d1ac15281b22bd8326c7a715"
SHIPPED = ROOT / "submission" / "paiec_submission.zip"
RES = SRC.parent / "results" / "w2_items" / "candidates"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def replace_once(text: str, old: str, new: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"expected exactly one occurrence of {old!r}, found {n}")
    return text.replace(old, new)


def patch_model(src: str, tau_feat: float, share: float, tau_kern: float, q: float) -> str:
    crlf = "\r\n" in src          # the m050 member uses CRLF line endings; patch on LF, restore at the end
    m = src.replace("\r\n", "\n")
    m = replace_once(m, "def xp_build(keys: list, ys: list, offsets: list, item_keys: list | None = None, taus=None,\n"
                        "             sweeps: int | None = None) -> dict:",
                     "def xp_build(keys: list, ys: list, offsets: list, item_keys: list | None = None, taus=None,\n"
                     "             sweeps: int | None = None, extra_types: list | None = None) -> dict:")
    m = replace_once(m, "    eta = [float(o) for o in offsets]\n    sig = _sigmoid\n",
                     "    if extra_types:\n        types.extend(extra_types)  # W2: effect types over a subset (\"members\") of the entries\n"
                     "    eta = [float(o) for o in offsets]\n    sig = _sigmoid\n")
    m = replace_once(m, "            gidx = t[\"gidx\"]\n            for e in range(n):\n                p = sig(eta[e])",
                     "            gidx = t[\"gidx\"]\n            for e in (t[\"members\"] if \"members\" in t else range(n)):\n"
                     "                p = sig(eta[e])")
    m = replace_once(m, "            for e in range(n):\n                eta[e] += step[gidx[e]]",
                     "            for e in (t[\"members\"] if \"members\" in t else range(n)):\n                eta[e] += step[gidx[e]]")
    m = replace_once(m, "def _xp_predict_keys(state: dict, bench, name, skey, ikey, offset: float, tau_item: float | None = None,\n"
                        "                     probit: bool | None = None) -> float:",
                     "def _xp_predict_keys(state: dict, bench, name, skey, ikey, offset: float, tau_item: float | None = None,\n"
                     "                     probit: bool | None = None, extra_m: float = 0.0, extra_var: float = 0.0) -> float:")
    m = replace_once(m, "    m = float(offset)\n    var = 0.0\n    for t in state[\"types\"]:\n        j = t[\"index\"].get(groups[t[\"name\"]])",
                     "    m = float(offset) + extra_m\n    var = extra_var\n    for t in state[\"types\"]:\n"
                     "        if \"members\" in t:\n            continue  # W2 category effects: added by the caller (extra_m, extra_var)\n"
                     "        j = t[\"index\"].get(groups[t[\"name\"]])")
    block = (HERE / "w2_block.py.txt").read_text(encoding="utf-8")
    block = (block.replace("__W2_TAU_FEAT__", repr(float(tau_feat))).replace("__W2_FEAT_SHARE__", repr(float(share)))
             .replace("__W2_TAU_KERN__", repr(float(tau_kern))).replace("__W2_KERN_Q__", repr(float(q))))
    assert "__W2_" not in block
    out = m.rstrip("\n") + "\n" + block
    return out.replace("\n", "\r\n") if crlf else out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--tau-feat", type=float, required=True)
    ap.add_argument("--share", type=float, default=0.5)
    ap.add_argument("--tau-kern", type=float, default=0.0)
    ap.add_argument("--q", type=float, default=8.0)
    ap.add_argument("--checks", action="store_true")
    ap.add_argument("--note", default="")
    ap.add_argument("--outdir", default=None, help="write the candidate here instead of submission/candidates (dev builds)")
    a = ap.parse_args()
    cand_root = Path(a.outdir).resolve() if a.outdir else CAND
    zbytes = BASE_ZIP.read_bytes()
    if sha(zbytes) != BASE_SHA:
        raise SystemExit("xp_level_m050.zip is not the validated candidate (sha mismatch); stopping")
    shipped_before = sha(SHIPPED.read_bytes())
    with ZipFile(BASE_ZIP) as zf:
        model_src = zf.read("model.py").decode("utf-8")
        labeling_src = zf.read("labeling.py")
        readme_src = zf.read("README.md").decode("utf-8")
    m = patch_model(model_src, a.tau_feat, a.share, a.tau_kern, a.q)
    note = (f"\n\n## Candidate `{a.name}` (W2 item layer, 2026-10-29)\n\nIdentical to `xp_level_m050` (sha256 {BASE_SHA[:16]}...) "
            f"plus an item layer learned online from the evaluator's labeled list (end of `model.py`): category effects of "
            f"`item_features` keys per benchmark in the cross-pair fit (prior sd {a.tau_feat:g}, active-key share {a.share:g})"
            + (f" and a character-4-gram text kernel on the residuals (prior sd {a.tau_kern:g}, exponent {a.q:g})" if a.tau_kern > 0 else "")
            + ".  Any failure falls back to the plain cross-pair model.  Simulations and pre-registration: "
              "`experiments/src/w2_items/` and `experiments/results/w2_items/` of the code repository." + (f" {a.note}" if a.note else "") + "\n")
    out_dir = cand_root / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {"model.py": m.encode("utf-8"), "labeling.py": labeling_src, "README.md": (readme_src + note).encode("utf-8")}
    for k, v in files.items():
        (out_dir / k).write_bytes(v)
    zpath = cand_root / f"{a.name}.zip"
    with ZipFile(zpath, "w", compression=ZIP_DEFLATED) as archive:
        for k, v in files.items():
            e = ZipInfo(k)
            e.compress_type = ZIP_DEFLATED
            e.external_attr = 0o100644 << 16
            archive.writestr(e, v)
    rdir = (RES if not a.outdir else cand_root / "records") / a.name
    rdir.mkdir(parents=True, exist_ok=True)
    info = {"zip": str(zpath), "bytes": zpath.stat().st_size, "sha256": sha(zpath.read_bytes()), "members": sorted(files),
            "model_sha256": sha(files["model.py"]), "base_zip_sha256": BASE_SHA, "labeling_identical_to_base": True,
            "constants": {"W2_TAU_FEAT": a.tau_feat, "W2_FEAT_SHARE": a.share, "W2_TAU_KERN": a.tau_kern, "W2_KERN_Q": a.q}}
    log = []
    if a.checks:
        py = sys.executable
        p = subprocess.run([py, "-I", str(SRC / "validator" / "check_submission_zip.py"), str(zpath)], capture_output=True, text=True)
        (rdir / "validator_output.txt").write_text(p.stdout + p.stderr + f"\n[exit code {p.returncode}]\n", encoding="utf-8")
        info["validator_exit"] = p.returncode
        log.append(("validator", p.returncode, (p.stdout + p.stderr).strip()[-300:]))
        p = subprocess.run([py, "-I", str(SRC / "smoke_test.py"), "--dir", str(out_dir), "--out", str(rdir / "smoke_test.json")],
                           capture_output=True, text=True, cwd=str(SRC))
        (rdir / "smoke_test_stdout.txt").write_text(p.stdout + p.stderr, encoding="utf-8")
        info["smoke_exit"] = p.returncode
        log.append(("smoke_test", p.returncode, (p.stdout + p.stderr).strip()[-300:]))
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
        info["official_loop_record_restored"] = backup is None or rec.read_bytes() == backup
        (rdir / "official_loop_check_stdout.txt").write_text(p.stdout + p.stderr, encoding="utf-8")
        info["official_loop_exit"] = p.returncode
        log.append(("official_loop_check", p.returncode, (p.stdout + p.stderr).strip()[-300:]))
    info["base_zip_sha256_after"] = sha(BASE_ZIP.read_bytes())
    info["shipped_zip_sha256_before"] = shipped_before
    info["shipped_zip_sha256_after"] = sha(SHIPPED.read_bytes())
    (rdir / "candidate_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    for name, code, tail in log:
        print(f"--- {name}: exit {code}\n{tail}\n")
    print(json.dumps(info, indent=1))
    return 0 if all(c == 0 for _, c, _ in log) else 1


if __name__ == "__main__":
    raise SystemExit(main())
