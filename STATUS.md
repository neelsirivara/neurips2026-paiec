# PAIEC (NeurIPS 2026 Predictive AI Evaluation Competition) -- status

Last updated: 2026-10-21 19:47 PDT. Deadline: **2026-10-30 23:59 AoE** (codebase + report via
OpenReview with the Codabench submission ID). Codabench competition 17828.

## What was built (all under `neurips_challenges/paiec/`)

| Deliverable | Path | State |
|---|---|---|
| Submission ZIP | `submission/paiec_submission.zip` (10.8 kB; `model.py`, `labeling.py`, `README.md` at root) | built; **passes the official validator** (`experiments/results/validator_output.txt`: `OK ... valid submission ZIP`, exit 0); SHA-256 `770f86e7...` in `experiments/results/submission_zip_info.json` |
| Predictor | `experiments/src/model.py` (= ZIP copy, byte-identical) | stdlib only (`math`, `re`); deterministic; never raises; 455 predict calls/s at budget 31 with 300-word items (mean 2.2 ms, max 54 ms; `smoke_test.json`) |
| Acquisition policy | `experiments/src/labeling.py` | streaming `acquisition_function(input, prediction=None, labeled=None, context=None) -> bool`; novelty (near-duplicate) filter, otherwise stream order; never raises |
| Local evaluation harness | `experiments/src/{harness,simulate,real_testbed,run_experiments,run_extras,embed_cache}.py` | reproduces pairs, 50/50 pools, random stream, budgets 0..31, Brier ALC, ECE, bootstrap CIs, paired tests, oracle rows |
| Real testbed | `experiments/results/real_testbed/*.jsonl` (+ frozen `real_testbed_external/`) | 15 pairs, 3,426 labelled items, 4 Ollama models x {GSM8K, SVAMP, BBH-mix} x 2 harnesses |
| Results | `experiments/results/*.json` (20 files) | all generated with the shipped constants |
| Report | `paper/main.pdf` (11 pages; main text ends at 86% of page 8, i.e. about 7.9 pages; refs + 3 appendix sections) | 0 `??`, 0 undefined refs, 0 overfull boxes; every number is a macro (`paper/results_macros.tex` from `make_macros.py`) |
| Audit / review / bib | `paper/AUDIT.md` (50 claims re-derived from JSON, 0 failing), `paper/review/REVIEW_1.md`, `paper/bib_verification_log.json` (13 entries verified) | done |
| User instructions | `submission/README.md` | registration, Codabench upload, OpenReview upload |

Rebuild everything: `cd experiments/src && python run_experiments.py --suite all --workers 4 && python run_extras.py && bash build_paper.sh && python build_submission.py`.

## Method (shipped constants)

Attribute prior at budget 0 (release date, reasoning effort, size/name cues, harness; tier/points/length
item cues; capped at +-1.5 logits) -> beta-binomial shrinkage of the pair rate with alpha0 = 4 ->
item-level components (ridge on log-length/digit-density/sentence-count from 7 labels; TF-IDF
unigram+bigram kNN residual correction) switched on only when a leave-one-out test on the acquired
labels shows an improvement of more than 2 standard errors (from 5 labels; no item-level adaptation
below 5) -> clip to [0.01, 0.99]. Design decisions taken by the builder during this session and why:

- alpha0 = 4: shrinkage ALC is best at alpha0 = 2 on the synthetic testbed and at 8 on the real
  testbed; 4 has the smallest worst-case regret (0.0007 on each). Sensitivity table in the paper.
- kNN off below 5 labels: an always-on kNN from 1-3 labels only added noise at B1/B3 (diagnostic).
- 2-SE LOO margin: with 0 or 1 SE the gate opened on noise (full worse than shrinkage by
  +0.0014/+0.0005 ALC on synthetic); with 2 SE the full model is never significantly worse.
- Bigram-diluted TF-IDF kept: a sharper unigram representation made the kNN worse
  (+0.0009 synthetic, +0.0017 real, always-on). MiniLM embeddings: no gain (-0.0002, ns) -> not shipped.
- Novelty acquisition filter shipped: identical to stream order on both testbeds (never fires),
  not significantly different from random; the BLE uncertainty rule is also not different.

## Headline local numbers (Brier ALC, lower is better; 95% bootstrap CIs over pairs, 1,000 resamples)

| | Synthetic IRT (160 pairs) | Real Ollama (15 pairs) |
|---|---|---|
| constant 0.5 | 0.2500 | 0.2500 |
| empirical mean (organisers' baseline, same code) | 0.2208 [0.2055, 0.2362] | 0.2745 [0.2302, 0.3117] |
| prior only | 0.2148 [0.2040, 0.2251] | 0.2330 [0.2184, 0.2491] |
| prior + shrinkage | 0.1797 [0.1691, 0.1900] | 0.2187 [0.1925, 0.2416] |
| **full (shipped)** | **0.1797 [0.1691, 0.1900]** | **0.2183 [0.1923, 0.2410]** |
| oracle: pair success rate | 0.1598 | 0.1966 |
| oracle: true item probability | 0.1379 | n/a |

Paired differences (full minus baseline): vs empirical mean -0.0411 [-0.0488, -0.0339] (synthetic),
-0.0562 [-0.0771, -0.0351] (real), both significant, full better on 80% of real pairs; vs constant
-0.0703 and -0.0317 (both significant); vs shrinkage alone +0.0000 [-0.0002, +0.0003] and -0.0004
[-0.0013, +0.0004] (not significant: item-level adaptation is marginal). Pooled ECE 0.032 / 0.046
(full) vs 0.114 / 0.152 (empirical mean). Replication: synthetic seeds 1-2 full 0.2266 / 0.1963 vs
empirical mean 0.2737 / 0.2438; four further real acquisition trajectories: full beats the empirical
mean significantly in 4/4. The empirical mean is worse than 0.5 at n = 1 on both testbeds (B1 0.330 /
0.468) and its real-testbed ALC is above 0.25.

## Verified on disk

- `experiments/results/validator_output.txt`: official `check_submission_zip.py` (copied verbatim,
  commit cc88a959, run with `python -I`) -> OK, exit 0.
- `experiments/results/smoke_test.json`: 34 cases (organisers' SAMPLE_SUBJECT/SAMPLE_ITEM and
  SMOKE_* shapes at all budgets, with/without benchmark_id, malformed labels/inputs, determinism,
  acquisition hook shapes) -> ok.
- `paper/main.pdf` builds with `bash experiments/src/build_paper.sh` (pdflatex/bibtex/pdflatex x2);
  `paper/AUDIT.md` 50/50 claims hold; bib entries verified (`paper/bib_verification_log.json`).
- ZIP member hashes equal `experiments/src/model.py` and `labeling.py`.

## What the user must do (the agent cannot create accounts, accept terms or upload)

1. Register the team via the form on https://aims.stanford.edu/competition (needs a Codabench
   username and a Hugging Face username; team name; e-mail neelsirivara@outlook.com).
2. Upload `submission/paiec_submission.zip` on Codabench 17828 (unchanged; do not re-zip), note the
   submission ID, read the formative feedback under Logs -> Scoring output.
3. Put your name/affiliation in `paper/main.tex` (`{[}Author Name{]}`, `{[}Affiliation{]}`), push
   the `paiec/` directory to a public repository, set `\codeurl` in `main.tex`, rebuild with
   `bash experiments/src/build_paper.sh`, and upload `paper/main.pdf` on OpenReview with the
   Codabench submission ID -- all before 2026-10-30 23:59 AoE. Details: `submission/README.md`.
4. Optional: if Codabench's formative score looks off, say so; nothing else is pending.

## Known risks

- **Gated training data.** `aims-foundations/measurement-db` needs an HF login; no token on this
  machine and no account was created, so the predictor has zero fitted parameters. The prior's
  subject coefficients are hand-set and untested on frontier models; alpha0 could be off for the
  official distribution of pair rates (ALC moved by 0.004-0.014 over alpha0 in 1..8 locally).
- **Local-vs-official gap.** Local testbeds: 4 small open models x 3 public benchmarks (15 pairs,
  wide CIs) + a synthetic IRT generator; official: frontier systems on many held-out benchmarks.
  The ordering shrinkage > empirical mean > constant follows from the metric and should transfer;
  the item-level effects are marginal either way.
- **Codabench unreachable** (connection refused) all day on 2026-10-21; the starting kit and any
  public leaderboard could not be checked. Retry early; leave slack before the deadline.
- The 7B direct-answer run was cut short (Ollama saturated by other local jobs): 7B x SVAMP has
  126 items (kept, >= 80), 7B x GSM8K/BBH direct-answer were skipped (7B GSM8K exists via CoT).
- Novelty filter untested in practice (never fired locally); harmless by construction (stream order).

## Session log (2026-10-21)

- Resumed from the previous builder's files; the testbed generator had stopped mid-run (laptop
  shutdown) and was resumed (llama3.2:3b complete; 7B partial, see above).
- Added: oracle rows, LOO 2-SE margin, kNN-off-below-5, alpha0 = 4, TF-IDF options and an
  experiments-only embedding hook, extras suite, macro/figure/audit generators, paper, review.
- Diagnostics that motivated the constants are in the session scratchpad only; the shipped suite
  (`alpha0_sensitivity_*`, `loo_margin_sensitivity_*`, `tfidf_variants_*`, `real_testbed_embedding`)
  re-establishes each of them with the final code.
- Review pass 1 must-fixes applied (see `paper/review/REVIEW_1.md`): gating interpretation,
  non-significant shrink-vs-prior wording, policies table width, oracle label, compute statement,
  two ablation paragraphs moved to the appendix to keep the main text under 8 pages.
