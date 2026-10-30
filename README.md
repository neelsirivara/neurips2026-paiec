# PAIEC 2026: Calibrate the Level, Borrow Across Pairs, Learn Items Online

Training-free, test-time-adaptive predictor for the NeurIPS 2026 Predictive AI Evaluation Competition (Codabench 17828): cross-pair random effects fitted to the evaluator's global label list, a budget-0 level recalibrated from formative feedback, and an item layer learned online within each benchmark ("W2"). Pure standard-library `model.py`. The per-pair predictor of rounds 1-3 (attribute prior, beta-binomial shrinkage, gated item adaptation) remains inside `model.py` as the fallback and the budget-0 prior.

**Deadlines.** Code + technical report: 2026-10-30 23:59 AoE (OpenReview, with the Codabench submission ID).

**Selected submission.** `submission/candidates/w2_f05s1_k1q2.zip` = Codabench submission **1019682** (23,436 bytes; members `README.md`, `labeling.py`, `model.py`), SHA-256 `d6d1c6a00e72b8aaf9e6628de6f03e50274a1d8f547883a0c0d70300aab969bf`. It passes the organisers' validator. `submission/paiec_submission.zip` is the older round-4 submission (1010370) and is only the starting point of the rebuild chain.

**Quick local check.** Inside the organisers' streaming client (`run_streaming`, mock coordinator) on the 17-pair real Ollama testbed, the selected ZIP scores Brier ALC 0.2174, against 0.2179 for its base `xp_level_m050` (1010717) and 0.25 for the constant predictor. Both numbers reproduce exactly in about a minute without gated data (see below).

## Layout
```
STATUS.md            what is built, verified numbers, user to-dos, risks
experiments/src/     pipeline code (harness, simulators, deep_dive/, w2_items/, build scripts, make_macros.py)
experiments/results/ result JSON/CSV the paper is generated from (caches are git-ignored)
paper/               NeurIPS 2026 LaTeX source, results_macros.tex (generated), main.pdf, AUDIT.md, review/
submission/          the selected ZIP (candidates/), older ZIPs, rebuild and check instructions (README.md)
```

## Rebuild and check the selected ZIP (about 1-2 minutes, no gated data)
```
cd experiments/src
python -I deep_dive/build_candidate.py xp_level_m050 --intercept -0.5 --subject-shrink 0 --tau-mu 2
python -I w2_items/build_w2.py w2_f05s1_k1q2 --tau-feat 0.5 --share 1 --tau-kern 1 --q 2 --checks
```
Expected outputs, and what to do if your SHA-256 differs (it depends on the zlib inside your Python) or the
smoke test's speed check fails on a slow machine: `submission/README.md`, section "For organisers and reviewers".

## Full rerun
The rounds 1-3 tables: `cd experiments/src && python run_experiments.py --suite all --workers 4 && python run_extras.py --workers 4 && bash build_paper.sh`
(about 45 + 5 minutes on 4 cores, plus about a minute for the paper; `build_paper.sh` needs pdflatex, bibtex, pdftotext and pdfinfo).
The round-4, deep-dive and W2 steps need the gated training release (Hugging Face login), about 12+ GB of free RAM for
the frame build, and three manual set-up steps; they are listed in `submission/README.md` and in the report's section
"Reproducing the submission".

Requirements: Python 3.10 or later. The submission and the validator use only the standard library; the checks
(`official_loop_check.py`) need pandas; the simulators need numpy, pandas, pyarrow, scipy and scikit-learn; the download
needs huggingface_hub; the figures need matplotlib.

Every number in `paper/main.pdf` is a macro emitted by `experiments/src/make_macros.py` from the result files;
`paper/AUDIT.md` re-derives each directional claim from the same files. `paper/bib_verification_log.json`
records how every reference was verified.

## Provenance and disclosure
Built on a CPU-only laptop (8 cores, 34 GB RAM, no GPU) with an LLM coding assistant (Claude) for code and
drafting; see the paper's disclosure section. Competition rules, starter kits and style files belong to their
respective organizers. Repository created 2026-10-22 by `make_repos.py`; owner `neelsirivara`.
