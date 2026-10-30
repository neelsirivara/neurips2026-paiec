# PAIEC submission package

**Selected submission: `candidates/w2_f05s1_k1q2.zip`, Codabench submission 1019682** (23,436 bytes;
members `README.md`, `labeling.py`, `model.py`; SHA-256
`d6d1c6a00e72b8aaf9e6628de6f03e50274a1d8f547883a0c0d70300aab969bf`). It is the base ZIP
`candidates/xp_level_m050.zip` (Codabench 1010717) plus the W2 item layer, and `xp_level_m050.zip` is the round-4
ZIP `paiec_submission.zip` (Codabench 1010370) with three constants changed. Steps 1-3 at the end of this file are
the original submission instructions, kept for the record; they describe the round-4 upload.

## For organisers and reviewers: rebuild and check the selected ZIP

Run from `experiments/src`. Needs Python 3.10 or later and pandas (for `official_loop_check.py`); no gated data,
no network. Wall times are from a CPU-only laptop.

```
cd experiments/src
# 1. Base ZIP from paiec_submission.zip; runs the validator, smoke_test.py and official_loop_check.py (about 25 s).
python -I deep_dive/build_candidate.py xp_level_m050 --intercept -0.5 --subject-shrink 0 --tau-mu 2
# 2. Selected ZIP = base + item layer, same three checks (about 50 s).
python -I w2_items/build_w2.py w2_f05s1_k1q2 --tau-feat 0.5 --share 1 --tau-kern 1 --q 2 --checks
# 3. Validator alone (verbatim copy of the organisers' check_submission_zip.py), about 1 s.
python -I validator/check_submission_zip.py ../../submission/candidates/w2_f05s1_k1q2.zip
```

Both build scripts overwrite `submission/candidates/<name>.zip` and `submission/candidates/<name>/`, and write their
check records to `experiments/results/deep_dive/candidates/xp_level_m050/` and
`experiments/results/w2_items/candidates/w2_f05s1_k1q2/` (`candidate_info.json`, `validator_output.txt`,
`smoke_test.json`, `official_loop_check.json`). Expected:

| Check | `xp_level_m050` | `w2_f05s1_k1q2` (selected) |
|---|---|---|
| ZIP SHA-256 | `cda9fe9cae49b68997d831f1de8dfe3d53a14ad3d1ac15281b22bd8326c7a715` | `d6d1c6a00e72b8aaf9e6628de6f03e50274a1d8f547883a0c0d70300aab969bf` |
| validator | `OK: ... looks like a valid submission ZIP.`, exit 0 | same |
| `official_loop_check.json` `alc_mean` (17 real pairs, organisers' `run_streaming`) | 0.217909 | 0.217387 |

`"matches_xpair_sim": false` in both loop records is expected: that field compares with the stored simulation of the
round-4 model (0.217882). `python official_loop_check.py --dir ../../submission` runs the round-4 files and gives
`true`.

**If your ZIP SHA-256 differs.** The member files are deterministic, but the ZIP bytes depend on the zlib that your
Python uses. The recorded ZIPs were built with CPython 3.14.5 from python.org on Windows, which bundles zlib-ng
(`python -c "import zlib; print(zlib.ZLIB_RUNTIME_VERSION)"` prints `1.3.1.zlib-ng`). With stock zlib, for example
CPython 3.12 (zlib 1.3.2), the members come out identical (same CRC-32), but the compressed bytes differ, so the
ZIP SHA-256 differs. `build_w2.py` then checks the base ZIP's members instead of its bytes: it prints a note and
continues if the members match the validated `xp_level_m050` members, and stops with `member sha mismatch` if they
don't. Compare the members of the result:

```
python -I -c "import zipfile,hashlib,sys; z=zipfile.ZipFile(sys.argv[1]); [print(n, hashlib.sha256(z.read(n)).hexdigest()) for n in sorted(z.namelist())]" ../../submission/candidates/w2_f05s1_k1q2.zip
# README.md   d94d9174c98ecf2b7665ee2b8b80123b2125871d1fcbe95793c7bba13546c406
# labeling.py d786011f7f8a2affed857e9a5084eace0af7bf5a6d2afcdca634a6504743f214
# model.py    b3625ff5d58dc0df661277c9931df91218c12162389172596ce1dea86fb90c44
```

Git line-ending settings do not matter. The members are read from the committed ZIPs, and rebuilds with
`core.autocrlf` true and false gave the same bytes.

**If the smoke test fails on speed.** `smoke_test.py` also checks speed on a stress case (2,000 pairs, 62,000
labels): the first fit must take at most 60 s, and later `predict` calls at most 5 ms each. The selected model
takes 4.9 ms per call in the committed record and 7.3-7.7 ms on the same laptop running on battery. On a
slow machine the run therefore ends with `AssertionError: {... 'xp_big_ms_per_call': 7.4 ...}`, and
`candidate_info.json` records `"smoke_exit": 1`. This is a speed threshold, not a functional failure: with only that
threshold relaxed, all 44 cases pass. Runtime on the platform is not a concern, because the organisers allow 8 h per
run.

**Organisers' starting-kit checks.** These are the checks from the competition page. Run them from the starting-kit
directory that Codabench distributes for competition 17828, pointing at this repository:
`python tools/check_submission_zip.py <repo>/submission/candidates/w2_f05s1_k1q2.zip` and
`python tools/run_smoke_test.py <repo>/submission/candidates/w2_f05s1_k1q2/`. Our copy of the validator comes from
`aims-foundations/paiec_baseline` at commit `cc88a959`. There the validator is `check_submission_zip.py` at the
repository root, and there is no `run_smoke_test.py`. We have run the validator. We have not run the starting kit's
`run_smoke_test.py`. Our own smoke test and `official_loop_check.py` (the organisers' `run_streaming`) are what we
ran instead.

## Rerunning the simulations (gated data, heavy)

The round-4, deep-dive and W2 numbers in the report come from the official training release. The report's section
"Reproducing the submission" lists the commands. Before running them:

1. **Data (needs a Hugging Face login).** Run `hf auth login` and accept the terms of the dataset
   `aims-foundations/measurement-db`. Then download it with
   `python -I fit_prior_from_measurement_db.py plan --payload DIR` and `... download --payload DIR`. Our results use
   revision `54074453da7b9911be0f48e4c560be547bc84d21`, but `plan` lists the current head. The download is 163
   directories, 5.1 GB on disk, and took about 8 minutes.
2. **Location.** Use `DIR = ../_downloads/measurement_db_payload` relative to the repository root, that is, a sibling
   of the repository directory. The `deep_dive/` and `w2_items/` scripts have no `--payload` option and read only
   that location (`xpair_sim.DEFAULT_PAYLOAD`). The default of `fit_prior_from_measurement_db.py` is one directory
   higher, so pass `--payload` explicitly.
3. **Memory.** `python -I xpair_sim.py build --payload DIR`, which writes `DIR/round4_cache/frame.npz` and
   `aux.pkl`, needs **12+ GB of free RAM**: on a 32 GB laptop it was stopped at 12.4 GB, around directory 100 of 163.
   It takes about 3 minutes. `w2_items/extract_all.py` (about 4.5 minutes) reads the same response tables; its peak
   memory has not been measured.
4. **Paths.** All paths in `deep_dive/` and `w2_items/` are relative to the repository (fixed on 2026-10-30; earlier
   commits had the author's absolute paths in six files). The only external file is the inventory CSV (item 5): set
   `PAIEC_INV_CSV` to it, or place it at `../_downloads/paiec_deepdive_2026-10-26/intel/inventory_gid0.csv` next to the
   repository.
5. **Benchmark inventory.** Every script that builds the INV scenario needs the organisers' benchmark inventory:
   `w2_items/run_w2.py`, `w2_items/w2_equivalence.py`, `deep_dive/proto_level3.py` and the other deep-dive scripts.
   It is a Google Sheet (https://docs.google.com/spreadsheets/d/1T69QyIZJRWX3d-Tmuh_lcFwtpqv9ytcLxfLWKn6gopI, tab
   `gid=0`); download it as CSV. Only the column `benchmark_name_slug` is used. Our copy was taken on 2026-10-26.
6. **Run times.** These are taken from the committed logs: `w2_items/run_w2.py tune F` about 2 min, `tune K` about
   42 min, `select` about 8 min, `confirm` about 1.5 h, `w2_equivalence.py` about 3 min.
7. **Formative aggregates.** `deep_dive/formative_summary.py` and `w2_items/w2_draws.py` read the Codabench scoring
   logs from `../_downloads/cb_logs_<id>/scoring_stdout.txt`. These logs are private to the submitter and not in the
   repository. The aggregates parsed from them are committed under `experiments/results/`.

The rounds 1-3 commands, unchanged (no gated data unless marked):

```
cd ../experiments/src
python smoke_test.py --dir ../../submission           # round-4 files: interface, robustness, throughput (seconds)
python run_experiments.py --suite all --workers 4      # all rounds 1-3 result JSONs (about 45 min on 4 cores)
python run_extras.py --workers 4                       # TF-IDF representation ablation (5 min)
bash build_paper.sh                                    # macros, figures, audit, PDF (matplotlib, pdflatex, bibtex, pdftotext, pdfinfo)
python build_submission.py                             # validator on the round-4 ZIP (rebuilds only with --force)
python -I text_prior.py build && python -I text_prior.py cv  # round 3: item-text prior (payload)
python -I xpair_sim.py build && python -I xpair_sim.py grid && python -I xpair_sim.py cv --save-units   # round 4 (payload; build needs 12+ GB RAM)
python -I xpair_sim.py grid --config A --grid ext && python -I xpair_sim.py grid --config E --grid ext && python -I xpair_sim.py addendum
python -I xpair_sim.py acq && python xpair_sim.py realtest && python xpair_sim.py realacq && python -I xpair_sim.py runtime
python -I xpair_sim.py equivalence && python -I xpair_sim.py perpair && python official_loop_check.py --dir ../../submission
```

The real Ollama testbed (`../experiments/results/real_testbed/*.jsonl`, plus the frozen copy of
the earlier chain-of-thought runs in `real_testbed_external/`) is already on disk; to
regenerate it you need Ollama with `qwen2.5:1.5b-instruct`, `qwen2.5:3b-instruct`,
`qwen2.5:7b-instruct`, `llama3.2:3b` and the local copies of GSM8K, SVAMP and BIG-Bench-Hard
listed in `real_testbed_generate.py` (about two hours on an otherwise idle 4-6 thread CPU).
The MiniLM embedding ablation additionally needs `sentence-transformers/all-MiniLM-L6-v2` in the
local Hugging Face cache (`embed_cache.py`, offline).

## Files

| File | Purpose |
|---|---|
| `candidates/w2_f05s1_k1q2.zip` | **The selected submission (Codabench 1019682).** Unzipped copy in `candidates/w2_f05s1_k1q2/`; check records in `../experiments/results/w2_items/candidates/w2_f05s1_k1q2/`. |
| `candidates/xp_level_m050.zip` | Its base (Codabench 1010717); records in `../experiments/results/deep_dive/candidates/xp_level_m050/`. |
| `candidates/xp_level_m100.zip`, `candidates/xp_nosubj_mu2.zip`, `candidates/probe_*.zip` | Other level variants and deployment probes described in the report; not selected. |
| `paiec_submission.zip` | Round-4 submission (Codabench 1010370, SHA-256 `c76331d084f1...`, 19.8 kB); the start of the rebuild chain. Do not upload it as the final submission. |
| `model.py`, `labeling.py`, `ZIP_README.md` | Unzipped copies of the round-4 ZIP's members. |
| `../experiments/results/validator_output.txt` | Organisers' validator on the round-4 ZIP (`OK`, exit code 0). |
| `../paper/main.pdf` | The technical report (NeurIPS 2026 template, non-anonymous). |

## Original submission steps (done; kept for the record)

Deadline for the final codebase and technical report: **2026-10-30, 23:59 AoE** (Anywhere on Earth = UTC-12,
i.e. 2026-10-31 11:59 UTC). Codabench: https://www.codabench.org/competitions/17828/.

### Step 1: register the team (free, one form per team)

The competition page (https://aims.stanford.edu/competition) links a registration form. It asks
for: team name, each member's name, affiliation, email, **Codabench username** and
**Hugging Face username**. Member 1 is the primary contact. Team size is up to six. Use the
e-mail `neelsirivara@outlook.com`. You need a Codabench account (https://www.codabench.org/)
and a Hugging Face account; create both yourself if you do not have them.

### Step 2: upload the ZIP to Codabench

1. Log in to Codabench, open competition 17828, read and accept the terms, and join.
2. In "My Submissions", upload the ZIP as is (do not re-zip the folder: `model.py` must stay at the ZIP root,
   which the validator already confirmed). The final submission is `candidates/w2_f05s1_k1q2.zip` (1019682);
   round 4 uploaded `paiec_submission.zip` (1010370).
3. Wait for the run to finish (the organisers allow 8 h per run; this predictor needs minutes).
   Open *Logs -> Scoring output* for the formative feedback (Brier ALC and ECE on a freshly
   sampled subset capped at 1,000 subject-item pairs; it is private to you and the organisers).
4. Note the **submission ID** shown by Codabench; the OpenReview form asks for it.
5. You may submit several times during development; the final OpenReview form names the one
   submission that counts.

### Step 3: upload the report to OpenReview

1. Rebuild the report with `bash ../experiments/src/build_paper.sh` (about a minute). The script reports the page
   count and checks that `main.pdf` has no `??`.
2. The report must link the released code. `\codeurl` in `main.tex` points to
   https://github.com/neelsirivara/neurips2026-paiec. It is **private** until you flip it to public: do that
   before the OpenReview upload, and make sure the latest `experiments/results`, `paper` and
   `submission` are pushed (the reviewers will follow the link).
3. Go to the OpenReview venue linked from the competition page, create the submission, upload
   `main.pdf`, and enter the Codabench submission ID (1019682).
