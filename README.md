# PAIEC 2026: Shrink First, Adapt Cautiously

Training-free, test-time-adaptive predictor for the NeurIPS 2026 Predictive AI Evaluation Competition (Codabench 17828): attribute prior at budget 0, beta-binomial shrinkage of the subject-benchmark success rate, gated item-level adaptation (TF-IDF kNN + cheap item features) and a streaming acquisition policy. Pure standard-library `model.py`.

**Deadlines.** Code + technical report: 2026-10-30 23:59 AoE (OpenReview, with the Codabench submission ID).

**Headline (local evaluation).** Brier ALC 0.218 [0.192, 0.241] on a real 15-pair Ollama testbed vs 0.275 for the organizers' empirical-mean baseline and 0.25 for the constant predictor; 0.180 vs 0.221 on a synthetic IRT testbed.

**Submission artifact.** `submission/paiec_submission.zip (passes the organizers' validator)`

## Layout
```
STATUS.md            what is built, verified numbers, user to-dos, risks (start here)
experiments/src/     pipeline code (data, evaluation harness, make_macros.py, make_figures.py, make_audit.py)
experiments/results/ result JSON/CSV the paper is generated from (caches are git-ignored)
paper/               NeurIPS 2026 LaTeX source, results_macros.tex (generated), main.pdf, AUDIT.md, review/
submission/          the bundle and upload instructions (README.md)
```

## Reproduce
`cd experiments/src && python run_experiments.py --suite all --workers 4 && python run_extras.py && bash build_paper.sh && python build_submission.py`

Every number in `paper/main.pdf` is a macro emitted by `experiments/src/make_macros.py` from the result files;
`paper/AUDIT.md` re-derives each directional claim from the same files. `paper/bib_verification_log.json`
records how every reference was verified.

## Provenance and disclosure
Built on a CPU-only laptop (8 cores, 34 GB RAM, no GPU) with an LLM coding assistant (Claude) for code and
drafting; see the paper's disclosure section. Competition rules, starter kits and style files belong to their
respective organizers. Repository created 2026-10-22 by `make_repos.py`; owner `neelsirivara`.
