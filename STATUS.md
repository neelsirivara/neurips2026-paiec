# PAIEC (NeurIPS 2026 Predictive AI Evaluation Competition) -- status

Last updated: 2026-10-23 (iteration 2026-10-22, see the last section). Deadline: **2026-10-30 23:59 AoE** (codebase + report via
OpenReview with the Codabench submission ID). Codabench competition 17828.

## What was built (all under `neurips_challenges/paiec/`)

| Deliverable | Path | State |
|---|---|---|
| Submission ZIP | `submission/paiec_submission.zip` (11.7 kB; `model.py`, `labeling.py`, `README.md` at root) | rebuilt 2026-10-23; **passes the official validator** (`experiments/results/validator_output.txt`: `OK ... valid submission ZIP`, exit 0); SHA-256 `3f0d15c8...` in `experiments/results/submission_zip_info.json` |
| Predictor | `experiments/src/model.py` (= ZIP copy, byte-identical) | stdlib only (`math`, `re`); deterministic; never raises; 455 predict calls/s at budget 31 with 300-word items (mean 2.2 ms, max 54 ms; `smoke_test.json`) |
| Acquisition policy | `experiments/src/labeling.py` | streaming `acquisition_function(input, prediction=None, labeled=None, context=None) -> bool`; novelty (near-duplicate) filter, otherwise stream order; never raises |
| Local evaluation harness | `experiments/src/{harness,simulate,real_testbed,run_experiments,run_extras,embed_cache}.py` | reproduces pairs, 50/50 pools, random stream, budgets 0..31, Brier ALC, ECE, bootstrap CIs, paired tests, oracle rows |
| Real testbed | `experiments/results/real_testbed/*.jsonl` (+ frozen `real_testbed_external/`) | 17 pairs, 3,800 labelled items, 4 Ollama models x {GSM8K, SVAMP, BBH-mix} x 2 harnesses (7B completed on all three benchmarks on 2026-10-22) |
| Results | `experiments/results/*.json` (26 files) | all generated with the shipped constants; plus `prior_fit*.json` (training-release fit, not shipped), `prior_sensitivity.json`, `novelty_filter_stats.json`, `measurement_db_listing.json` |
| Report | `paper/main.pdf` (13 pages; main text ends at the bottom of page 8; refs + 5 appendix sections) | 0 `??`, 0 undefined refs, 0 overfull boxes; every number is a macro (`paper/results_macros.tex` from `make_macros.py`, 1,361 macros) |
| Audit / review / bib | `paper/AUDIT.md` (66 claims re-derived from JSON, 0 failing), `paper/review/REVIEW_1.md` (2 passes), `paper/bib_verification_log.json` (13 entries verified) | done |
| User instructions | `submission/README.md` | registration, Codabench upload, OpenReview upload |

Rebuild everything: `cd experiments/src && python run_experiments.py --suite all --workers 4 && python run_extras.py && python prior_sensitivity.py --suite all --workers 4 && python novelty_stats.py && bash build_paper.sh && python build_submission.py` (the training-release fit: `python -I fit_prior_from_measurement_db.py all`, needs an HF login).

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
- Novelty acquisition filter shipped: not significantly different from random on either testbed; the
  BLE uncertainty rule is also not different. (Correction 2026-10-23: the filter does fire on templated
  items -- see the iteration section -- but its ALC equals stream order to four decimals.)

## Headline local numbers (Brier ALC, lower is better; 95% bootstrap CIs over pairs, 1,000 resamples)

*(2026-10-21 build, 15 real pairs; the current 17-pair numbers and the training-release validation are in the
"Iteration 2026-10-22" section at the end.)*

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
3. Put your name/affiliation in `paper/main.tex` (`{[}Author Name{]}`, `{[}Affiliation{]}`), rebuild with
   `bash experiments/src/build_paper.sh`, and upload `paper/main.pdf` on OpenReview with the
   Codabench submission ID -- all before 2026-10-30 23:59 AoE. Details: `submission/README.md`.
4. `\codeurl` is set to https://github.com/neelsirivara/neurips2026-paiec (managed by `make_repos.py`).
   The repository is **private until you release it**: flip it to public before the OpenReview upload,
   after pushing the current `experiments/results`, `paper` and `submission`.
5. Done on 2026-10-22: `hf auth login` + acceptance of the `aims-foundations/measurement-db` terms; the
   payload (5.1 GB, 163 eligible directories) is in `../_downloads/measurement_db_payload/` (outside the
   repository) and can be deleted; `python -I experiments/src/fit_prior_from_measurement_db.py all`
   re-creates it.

## Known risks (updated 2026-10-23)

- **Local-vs-official gap, narrowed.** The shipped rule was validated on the official training release
  (16,303 real pairs, 144 benchmarks; closed-form protocol): ALC 0.2051 vs 0.2396 (empirical mean) and
  0.2470 (constant). The official *test* benchmarks are different benchmarks; the ordering should hold,
  the absolute level will differ. The item-level components (full vs shrinkage, |effect| < 0.001 locally)
  were not part of the closed-form check.
- **Frontier subjects.** The subject prior gives p0 up to 0.82 for 2026/high-effort/400B/agent-harness
  descriptions (appendix table). The analytic cost table says: a prior of 0.7 on a pair whose rate is 0.3
  costs 0.062 ALC and loses to the constant; within |p0 - r| <= 0.3 shrinkage beats the constant whenever
  |r - 0.5| >= 0.2. `PRIOR_INTERCEPT` in `model.py` is the knob (negative = more pessimistic) if the
  Codabench formative score points that way.
- **Codabench unreachable** (connection refused, 129.175.8.21:443) again on 2026-10-22; the leaderboard
  and the formative feedback have never been seen from this machine. Retry early.
- Novelty filter: it DOES fire locally (10/17 real pairs: templated BBH navigate/boolean items and
  same-template SVAMP problems, cosine up to 0.86), but its ALC equals plain stream order to four
  decimals; the 2026-10-21 statement that it never fires was wrong (`novelty_filter_stats.json`).

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

## Iteration 2026-10-22 (overnight, autonomous; 2 CPU threads; Ollama shared)

### What changed

- **`model.py`** (predictor logic unchanged, verified bit-identical on 7,340 snapshot cases: synthetic +
  real pairs at every budget, all components): the budget-0 prior is now a sum of
  `(coefficient name, value)` terms exposed as `subject_terms()` / `item_terms()` so a fitted table can
  replace the constants one-to-one; two new knobs with neutral shipped values, `PRIOR_INTERCEPT = 0.0`
  (global shift of the prior logit) and `PRIOR_PROVIDER = {}` (per-provider logits).
- **`fit_prior_from_measurement_db.py`** (new): `plan` (list the gated repo, resolve eligibility from the
  `benchmarks` tables, size cap 30 GB), `download` (eligible `subjects/items/responses` only; traces and
  assets skipped), `fit` (pair-weighted L2 logistic regression on the prior's own terms + provider
  one-hot + benchmark fixed effects; 5-fold CV over whole benchmarks; penalty/cap by budget-0 Brier,
  alpha0 by closed-form shrinkage ALC; paired bootstrap vs the hand-set prior), `testbeds` (fitted vs
  shipped inside the harness on both local testbeds; ship rule = better under CV AND on the real testbed),
  `dry-run` (synthetic tables with the release schema; recovers 13/13 slope signs, max error 0.05).
- **`prior_sensitivity.py`** (new): frontier-subject p0 grid (160 descriptions + the organisers' two sample
  subjects), exact expected ALC cost of a miscalibrated prior under alpha0 = 4 (binomial sums), and the
  harness re-run with `PRIOR_INTERCEPT` in {-1, -0.5, 0, +0.5, +1} on both testbeds.
- **Real testbed extended** to 17 pairs: qwen2.5:7b-instruct now has SVAMP (200, completed from 126),
  GSM8K (200, acc 23.5%) and BBH-mix (300, acc 51.0%) under the direct-answer harness (2 Ollama threads,
  about 60 min). Loader unchanged.
- **Paper**: `\codeurl` set; Data section rewritten around the release (listing, eligibility, validation,
  the fit that was not shipped); new Results paragraph "Frontier-like subjects"; replication and the
  policies table moved to appendices; new appendices "Fitting pipeline details" and "Prior sensitivity for
  frontier-like subjects" (3 tables); abstract/intro/limitations/disclosure updated to state exactly what
  the release was used for. `make_macros.py` / `make_audit.py` extended (1,300+ macros; 60+ audited claims).
- `build_submission.py`: the ZIP README now reads its constants from `model.py` (the old text said
  "prior strength 2", the shipped value is 4) and states the prior's provenance.
- `measurement_db_listing.json` (public file listing of the release: 281 dirs; items 12.1 GB, responses
  2.7 GB, traces 107.8 GB, assets 154.7 GB) saved as a result file.

### The measurement-db path (ran; nothing shipped from it)

- The user's `hf auth login` appeared at 23:08 PDT; the terms were accepted (gate `auto`); `plan`
  resolved 163 eligible of 281 directories; download 5.08 GB (163 x {subjects, items, responses});
  `fit` built 16,718 (subject, test-condition) pairs with >= 20 items from 144 benchmarks (3.12 M rows,
  200 items per pair cap; 16,303 pairs with >= 80 items), 22 providers with >= 15 pairs.
- **Validation of the shipped rule on the release** (closed-form protocol: 50/50 split, 31 random
  acquisitions, prior + beta-binomial; no item-level components): hand-set prior + shrinkage
  **ALC 0.2051 [0.2042, 0.2060]** (alpha0 = 4; 0.2052 at 2, 0.2086 at 8) vs **0.2396 [0.2380, 0.2411]**
  for the organisers' empirical mean and **0.2470** for the constant; prior alone 0.2385 (budget-0 Brier
  0.2388). So the local ordering and gain transfer to real systems on real benchmarks, and alpha0 = 4 is
  confirmed.
- **The fitted prior lost** under benchmark-held-out CV at every penalty (budget-0 Brier 0.2421 at
  C = 0.001 rising to 0.2455 at C = 1, vs 0.2388 hand-set; paired fit - hand at the selected C = 0.001:
  +0.0033 [+0.0030, +0.0036] budget-0, +0.0009 [+0.0008, +0.0009] ALC; fitted better on 43% of pairs);
  intercept-only prior 0.2519. Fitted signs agree with the hand-set ones where identified (tier=hard
  -0.65 vs -0.60; length +0.08 vs +0.12; year +0.14 vs +0.25 effective; size +0.08 vs +0.18 effective);
  effort/harness shrink to 0 (58% of pairs have no provider field, most no effort/harness); the provider
  table (-0.23 .. +0.55) is what held-out benchmarks punish. Ship rule -> keep the hand-set constants.
  Result files: `experiments/results/prior_fit.json`, `prior_fit_testbeds.json` (harness comparison on
  both local testbeds, see below), `prior_fit_dry_run.json`.
- Data sources used (disclosed in the report): `aims-foundations/measurement-db` revision `54074453da`,
  tables `subjects`, `items`, `responses` of the 163 eligible directories (list in `prior_fit.json`
  `data_sources`); only for the validation and the unshipped fit. No other inventory source.

### Before / after (Brier ALC, 95% bootstrap CIs over pairs)

| | before (2026-10-21) | after (2026-10-23) |
|---|---|---|
| Real Ollama testbed, pairs | 15 | 17 |
| real: empirical mean | 0.2745 [0.2302, 0.3117] | 0.2871 [0.2369, 0.3303] |
| real: prior + shrinkage | 0.2187 [0.1925, 0.2416] | 0.2228 [0.1951, 0.2460] |
| real: **full (shipped)** | 0.2183 [0.1923, 0.2410] | **0.2224 [0.1949, 0.2454]** |
| real: full vs empirical mean (paired) | -0.0562 [-0.0771, -0.0351] | -0.0647 [-0.0872, -0.0415] (full better on 82% of pairs) |
| synthetic IRT (160 pairs): full | 0.1797 [0.1691, 0.1900] | unchanged (predictor bit-identical) |
| training release (16,303 pairs): shrinkage / empirical mean / constant | n/a | 0.2051 / 0.2396 / 0.2470 |

Offset sweep (`PRIOR_INTERCEPT`, full-model ALC; paired vs 0):
- real: -1.0: 0.2229 (+0.0005); -0.5: 0.2190 (-0.0034); +0.0: 0.2224 (+0.0000); +0.5: 0.2358 (+0.0134*); +1.0: 0.2576 (+0.0353*); best -0.5
- synthetic: -1.0: 0.1955 (+0.0158*); -0.5: 0.1834 (+0.0037*); +0.0: 0.1797 (+0.0000); +0.5: 0.1844 (+0.0047*); +1.0: 0.1932 (+0.0134*); best +0.0
  (* = 95% CI excludes 0). Real: positive shifts significantly worse, -0.5 not significant; synthetic: 0 best.
  Release calibration: hand-set p0 mean 0.510 vs pair-rate mean 0.537 (not optimistic) -> intercept stays 0.
- Novelty filter (`novelty_filter_stats.json`): fires on 10/17 real pairs (40 candidates skipped, max cosine 0.985; benches ['bbh_mix', 'svamp']), on 0/160 synthetic pairs; ALC equals stream order to 4 decimals.

Validator: official `check_submission_zip.py` on the rebuilt ZIP -> OK, exit 0 (11.7 kB, SHA-256 `3f0d15c879d9...`, members ['README.md', 'labeling.py', 'model.py']). Smoke test: 34 cases ok, 553 calls/s at budget 31 (mean 1.81 ms, max 46.2 ms).

### Not done / left as is
- Codabench still refused connections (one WebFetch attempt, ECONNREFUSED); the leaderboard was not seen.
- The synthetic suite was not re-run (the predictor is bit-identical; its JSONs are unchanged and valid).
- The fitted coefficients are not shipped (rule failed on its first condition); `PRIOR_INTERCEPT` stays 0.
