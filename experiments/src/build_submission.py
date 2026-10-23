"""Package the PAIEC submission ZIP and run the organisers' validator on it.

    python build_submission.py

Writes submission/paiec_submission.zip (model.py, labeling.py, README.md at the root; no
models.txt and no requirements.txt because the predictor uses only the Python standard
library), then runs experiments/src/validator/check_submission_zip.py (a verbatim copy of the
official validator, commit cc88a959 of aims-foundations/paiec_baseline) with ``python -I`` and
stores its output in experiments/results/validator_output.txt.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SUBMISSION_DIR = ROOT / "submission"
RESULTS = HERE.parent / "results"
ZIP_PATH = SUBMISSION_DIR / "paiec_submission.zip"

ZIP_README = """# PAIEC submission: hierarchical shrinkage with text-aware test-time adaptation

`model.py` implements `predict(input, labeled=None) -> float`, the probability that the
subject (input[0]) answers the item (input[1]) correctly.  It uses only the Python standard
library, loads no data or model files, makes no network calls, and never raises.

Components (all constants are hand-set priors documented in the technical report; nothing is
fitted on competition data):
1. budget-0 prior from visible subject attributes (release date, reasoning effort, parameter
   count in the name, harness) and item cues (tier/points in item_features, item length);
2. beta-binomial shrinkage of the subject-benchmark success rate towards that prior
   (prior strength 2 pseudo-counts), using the acquired labels of the same pair;
3. a within-pair ridge-logistic adjustment on three cheap item features (log length, digit
   density, sentence count), from 7 labels on;
4. an item-similarity correction: TF-IDF (unigram+bigram) cosine similarity between the target
   item and the acquired items, kernel-weighted residual of the acquired labels;
5. leave-one-out gating: from 5 labels on, the kernel scale (including "off") and the use of
   the ridge component are selected by leave-one-out Brier score on the acquired labels, so
   adaptation is applied only when the pair's own labels show that it helps;
6. clipping to [0.01, 0.99].

`labeling.py` implements the streaming `acquisition_function(input, prediction=None,
labeled=None, context=None) -> bool`.  It accepts candidates in stream order (a uniformly
random sample, as the default policy) but skips near-duplicates of already acquired items
(TF-IDF cosine > 0.6) while enough candidates remain to spend the budget.  It never raises.

Deterministic; about 1-3 ms per predict call (hundreds of calls per second on one CPU core).
"""


def build_zip() -> Path:
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    files = {"model.py": (HERE / "model.py").read_bytes(), "labeling.py": (HERE / "labeling.py").read_bytes(),
             "README.md": ZIP_README.encode("utf-8")}
    with ZipFile(ZIP_PATH, "w", compression=ZIP_DEFLATED) as archive:
        for name, data in files.items():
            entry = ZipInfo(name)
            entry.compress_type = ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, data)
    (SUBMISSION_DIR / "model.py").write_bytes(files["model.py"])
    (SUBMISSION_DIR / "labeling.py").write_bytes(files["labeling.py"])
    (SUBMISSION_DIR / "ZIP_README.md").write_bytes(files["README.md"])
    return ZIP_PATH


def run_validator(zip_path: Path) -> int:
    validator = HERE / "validator" / "check_submission_zip.py"
    proc = subprocess.run([sys.executable, "-I", str(validator), str(zip_path)], capture_output=True, text=True)
    out = (f"$ python -I {validator.name} {zip_path.name}\n" + proc.stdout + proc.stderr
           + f"\n[exit code {proc.returncode}]\n")
    (RESULTS / "validator_output.txt").write_text(out, encoding="utf-8")
    print(out)
    return proc.returncode


def main() -> int:
    zip_path = build_zip()
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    with ZipFile(zip_path) as zf:
        names = sorted(zf.namelist())
    meta = {"zip": str(zip_path), "bytes": zip_path.stat().st_size, "sha256": digest, "members": names}
    (RESULTS / "submission_zip_info.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    return run_validator(zip_path)


if __name__ == "__main__":
    raise SystemExit(main())
