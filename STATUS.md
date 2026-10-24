# PAIEC (NeurIPS 2026 Predictive AI Evaluation Competition) -- status

Last updated: 2026-10-23 21:40 PDT (round 4, see the last section). Deadline: **2026-10-30 23:59 AoE** (codebase + report via
OpenReview with the Codabench submission ID). Codabench competition 17828.

## What was built (all under `neurips_challenges/paiec/`)

| Deliverable | Path | State |
|---|---|---|
| Submission ZIP | `submission/paiec_submission.zip` (19.8 kB; `model.py`, `labeling.py`, `README.md` at root) | **REBUILT in round 4** (SHA-256 `c76331d084f1...`; cross-pair model + fresh-first acquisition); **passes the official validator** (`experiments/results/validator_output.txt`: `OK ... valid submission ZIP`, exit 0); ZIP members are byte-identical to `experiments/src/model.py` / `labeling.py` |
| Predictor | `experiments/src/model.py` (= `submission/model.py` = ZIP member) | round 4: cross-pair random-effects model on the evaluator's global `labeled` list (`XP_ENABLED = True`), fallback = rounds 1-3 per-pair predictor (also budget 0); stdlib only; deterministic; never raises; 62,000-label fit 1.7 s, 0.12 ms per cached call |
| Acquisition policy | `experiments/src/labeling.py` | round 4: `acquisition_function(input, labeled=None, context=None) -> bool` (no `prediction` parameter, so the client never calls predict per candidate); POLICY `fresh` (skip items already labelled for another subject while labels_remaining < items_remaining); `acquisition_policy(input, prediction, labeled, context)` keeps the old full call shape for the rounds 1-3 experiments |
| Local evaluation harness | `experiments/src/{harness,simulate,real_testbed,run_experiments,run_extras,embed_cache}.py` | reproduces pairs, 50/50 pools, random stream, budgets 0..31, Brier ALC, ECE, bootstrap CIs, paired tests, oracle rows |
| Real testbed | `experiments/results/real_testbed/*.jsonl` (+ frozen `real_testbed_external/`) | 17 pairs, 3,800 labelled items, 4 Ollama models x {GSM8K, SVAMP, BBH-mix} x 2 harnesses (7B completed on all three benchmarks on 2026-10-22) |
| Results | `experiments/results/*.json` (26 files) | all generated with the shipped constants; plus `prior_fit*.json` (training-release fit, not shipped), `prior_sensitivity.json`, `novelty_filter_stats.json`, `measurement_db_listing.json` |
| Report | `paper/main.pdf` (20 pages; main text ends on page 7; round-4 paper, rounds 1-3 in Appendices B-D) | 0 `??`, 0 undefined refs, 0 overfull boxes; every number is a macro (`paper/results_macros.tex` from `make_macros.py`, 2,566 macros) |
| Audit / review / bib | `paper/AUDIT.md` (133 claims re-derived from JSON, 0 failing), `paper/review/REVIEW_1.md` (4 passes), `paper/bib_verification_log.json` (20 entries verified) | done |
| User instructions | `submission/README.md` | registration, Codabench upload, OpenReview upload |

Rebuild everything: `cd experiments/src && python run_experiments.py --suite all --workers 4 && python run_extras.py && python prior_sensitivity.py --suite all --workers 4 && python novelty_stats.py && bash build_paper.sh && python build_submission.py` (the training-release fit: `python -I fit_prior_from_measurement_db.py all`, needs an HF login; round 3: `python -I text_prior.py build|cv|final|check|testbeds|explore|runtime|equivalence`, needs the downloaded payload).

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

## Round 3 (2026-10-23, autonomous; 4 laptop threads; Mac not used)

**Question.** Does the item text predict difficulty across benchmarks well enough to move the budget-0 /
low-budget predictions for an unseen benchmark? (Benchmark mean success rates have sd 0.223 on the release;
the attribute prior cannot see them.)

**Pre-registration** (`experiments/results/round3/PREREG.md`, written 10:27 PDT before any fit, SHA-256
`bb54af425da8...`, stored in every result JSON and checked by the audit). Design: stdlib featuriser in
`model.py` (`text_prior_features`: JSON contents flattened to `key: value` lines; signed `zlib.crc32` hashing
into 2^18 buckets of word unigrams/bigrams from the first 2,000 chars, char 3-grams from the first 500,
`item_features` tokens; sublinear tf, per-family l2; plus 16 text statistics) -> sklearn ridge on the residual
item logit `logit((n r + .5)/(n + 1)) - mean hand subject logit - hand item logit` (<= 1,000 items per benchmark,
benchmarks weighted equally) -> stacked onto the hand-set prior as `T = W s + B` (W, B by pair-weighted logistic
on inner out-of-fold scores), cap in {1.5, 2.5, 3.5}, within-pair weight lambda in {0, .5, 1}, alpha0 = 4 kept;
ridge penalty in {0.03 ... 10} by 4-fold inner benchmark-grouped CV; all selection inside the training benchmarks
of each of the iteration-2 outer folds. **Ship rule:** ship iff (a) benchmark-held-out ALC difference (text - hand)
has a 95% cluster-bootstrap CI (over benchmarks, 2,000 resamples) entirely below 0, AND (b) the real Ollama testbed
full-model ALC difference is not significantly > 0, AND engineering gates (>= 200 calls/s on one core with a new
target item per call, weights <= 5 MB, validator + smoke test, never raises). Secondary (text x release-year
interaction) only if the primary ships -> not run.

**Data used** (disclosed in the report, Appendix D lists all 144 benchmarks): the iteration-2 frame (16,718 pairs,
3,118,742 rows, 144 benchmarks; hand-set B0 0.238753 / ALC 0.205100 reproduced *exactly*), plus `content` and
`item_features` of 515,067 items (109,233 target items) from `items.parquet` of those benchmarks; features cached
outside the repo in `_downloads/measurement_db_payload/round3_cache/` (about 2 GB; deletable, `text_prior.py build`
re-creates it in ~8 min).

### Results (all in `experiments/results/round3/`)

| | hand-set (shipped) | + item-text prior |
|---|---|---|
| benchmark-held-out CV, B0 (16,718 pairs) | 0.2388 | 0.2391 |
| benchmark-held-out CV, ALC (16,303 pairs) | 0.2051 [0.2042, 0.2060] | 0.2051 [0.2042, 0.2060] |
| paired ALC, text - hand (cluster bootstrap over 135 benchmarks) | | **-0.0000 [-0.0013, +0.0014]**, not significant (pair bootstrap [-0.0001, +0.0001]) |
| paired B0, text - hand | | +0.0003 [-0.0036, +0.0046] |
| gap to the pair-rate oracle closed at n = 0 / 1 / 3 | | -0.1% [-6.7, 7.1] / 0.1% [-5.6, 6.0] / 0.1% [-4.2, 4.3] |
| real Ollama testbed, full-model ALC (17 pairs) | 0.2224 | 0.2243; paired +0.0019 [-0.0002, +0.0037], n.s. (without BBH pairs: +0.0017 [-0.0007, +0.0042]) |
| synthetic IRT testbed, full-model ALC (160 pairs) | 0.1797 | 0.1800; paired +0.0003 [-0.0002, +0.0008], n.s. |

- Why it fails: the inner CV picked the **largest penalty (10) in 5/5 folds** (inner MSE at or above the constant's),
  and the out-of-fold calibration slope **W was negative in 5/5 folds (-0.32 to -0.03)**: what the text model learns
  on some benchmarks points the wrong way on others. Held-out R^2 of the residual target -0.06 .. 0.04.
- Headroom exists: an oracle knowing each held-out benchmark's mean rate gets ALC 0.1990 (-0.0061 [-0.0113, -0.0035];
  closes 30% / 25% / 19% of the gap at n = 0 / 1 / 3). Text gets none of it.
- Descriptive: refitting only the intercept shift (W = 0) gives +0.0003 [-0.0007, +0.0009]; alpha0 = 2 +0.0001,
  alpha0 = 8 +0.0035 (significantly worse); per fold the text prior is better in 1/5 (0 significant), per benchmark
  57/135; exact cross-benchmark duplicate items are 0.02% (no leakage concern).
- **Exploratory** (`PREREG_ADDENDUM_1.md`, written after the folds ran and before the CV summary; cannot change the
  decision): E1 penalties {10, 100, 1e3, 1e4}: +0.0002 [-0.0007, +0.0011]; E2 16 statistics only: +0.0001
  [-0.0010, +0.0014]; W < 0 in 5/5 folds for both. So the negative result is not an artefact of the grid.

**Ship decision: NOT shipped** (rule (a) fails; (b) holds; engineering gates would have passed). The submission ZIP is
unchanged (SHA-256 `3f0d15c879d9...`), re-validated: `OK ... valid submission ZIP`, exit 0.

**Engineering (would-be-shipped variant, `results/round3/text_prior_runtime.json`, `text_prior_check.json`):**
weights `text_prior_candidate.json` 1.40 MB (float32, base64, 262,077 non-zero), load 22 ms; predict 3,056 / 673 / 417
calls/s on one core with a new 15- / 300- / 2,000-word target item per call at budget 31; smoke test with the weights
loaded 44 cases OK (incl. malformed JSON contents and a corrupted spec -> hand-set fallback); stdlib features
bit-identical to the fit cache on 2,000/2,000 release items, stdlib vs numpy score max |diff| 9.0e-9. Final all-benchmark
fit: penalty 10, W -0.20, B +0.14, cap 1.5, lambda 0.

### Code / files changed
- `experiments/src/text_prior.py` (new): `build`, `cv` (+ `--resume`), `final`, `check`, `testbeds`, `explore`,
  `runtime`, `equivalence`.
- `experiments/src/model.py`: optional text prior (`text_prior_features`, `load_text_prior`, `_text_logit`, `_cap`,
  `within` offset weight; hand-only fallback path); without `text_prior.json` bit-identical to the shipped file.
- `fit_prior_from_measurement_db.py`: `load_pairs_from_payload(keep_ids=, item_stats=)` (defaults unchanged).
- `run_experiments.py`: `TEXT_PRIOR_FILE` param; `smoke_test.py`: `--dir`, `--out`, JSON/malformed-spec cases;
  `build_submission.py`: no silent rebuild of a validated ZIP; `make_macros.py` (+ round-3 macros, table,
  duplicate guard), `make_audit.py` (+ 20 round-3 claims).
- Paper: Data paragraph "An item-text difficulty prior (pre-registered; not shipped)", Appendix D (details, Table 2,
  benchmark list), abstract/intro/contributions/limitations/reproducibility/disclosure updated; acquisition-policy and
  frontier-subject discussions moved to the appendix to keep 8 pages. 15 pages total.
- `submission/README.md`: model.py provenance note; Step 4 commands.

### Before / after (round 3)

| | before | after |
|---|---|---|
| shipped predictor / ZIP | hand-set prior + shrinkage + gated item-level; SHA `3f0d15c8...` | unchanged (same bytes) |
| CV ALC on the release (hand-set) | 0.2051 | 0.2051 (text prior: 0.2051, diff -0.0000 [-0.0013, +0.0014]) |
| report | 13 pages, 66 audited claims | 15 pages (main text still 8), 86 audited claims, 0 failing |

### Not done / notes
- The secondary interaction (text x release year) was not run (pre-registered as conditional on the primary shipping).
- Pretrained text encoders (e.g. the organisers' item embeddings) were not tried; they remain the obvious next
  hypothesis and would need their own pre-registration (and `models.txt` / allow-listed model if shipped).
- User actions are unchanged (register, upload the unchanged ZIP, fill the author block, OpenReview).
- Timeline / transparency: PREREG.md 10:27 PDT; feature build 10:36-10:44; a solver timing test (one 60/40 benchmark
  split, penalties 1 and 0.03; scratch only) ran before the CV and already showed held-out MSE above the constant --
  the pre-registration was not changed; CV folds 10:45-10:58 (first run, stuck in a slow duplicate diagnostic, killed
  and rerun identically 11:07-11:20); addendum 11:08 (after the first run's folds, before any CV summary).

## Round 4 (2026-10-23 evening, autonomous; 4 laptop threads; Mac not used)

**Discovery driving the round (verified by the coordinator and re-read here).** The official client
(`tools/streaming_ingestion.py`, `run_streaming`, `evaluate` branch) passes ONE global `labeled` list
(`event["labeled"]`: all labels acquired so far for ALL sampled pairs) to every predict() call of a checkpoint; the
page says `labeled` holds "entries across sampled subject-benchmark pairs" and that a formative evaluation is capped at
1,000 subject-item pairs. Our rounds 1-3 predictor used only same-benchmark + same-`normalized_name` entries (so as
deployed it also pooled other subjects with the same name, e.g. other harnesses / test conditions).

**Pre-registration.** `experiments/results/round4/PREREG.md` written 20:45 PDT before any outcome (SHA-256 in every
round-4 JSON; audit-checked). One amendment before any outcome (20:50: the acquisition skip threshold 0.5 would never
skip with 50-item pools; replaced by `labels_remaining < items_remaining`). `PREREG_ADDENDUM_1.md` (20:58, after the
primary result, before anything else): config E (distinct models) + extended grid (tau_bench to 3, tau_item to 4),
"robust pick" on A+E. It returned the same setting as the pre-registered pick.

**Simulator design** (`experiments/src/xpair_sim.py`; frame outside the repo in `_downloads/measurement_db_payload/round4_cache/`,
built in 3 min): 16,718 (subject, test condition) pairs from 144 benchmarks, 3,118,742 item labels, 165,764 items; each pair keeps
its 200 answered items with the lowest benchmark-level hash rank (so subjects share items); item identity = items-table
`content_hash`. Folds = the 5 benchmark folds of round 2. A test set shares ONE global list. Per pair: fixed 50/50 split,
random acquisition order (official random policy = uniform subset in stream order), budget b = first b labels; at budget b
every prediction in the test set gets the union of all its pairs' first-b labels. Configs: A (primary; 1-3 benchmarks, 10
pairs, 100 items/pair <= 1,000 subject-items; 300 test sets/fold), E (A with distinct model names), B (all pairs of one
benchmark, <= 300, 200 items), C (one fold, <= 2,000 pairs in one list), D (A + up to 2 eval-only pairs with 20-79 items).
Units = (test set, pair); 95% CIs = cluster bootstrap over benchmarks (2,000). Baselines: S0 = rounds 1-3 predictor as the
evaluator runs it (closed form: prior + beta-binomial, name-based filter on the global list, fallback to all entries);
S1 = same with the target pair's own labels (= the local harness).

**Cross-pair predictor (XP, shipped).** `model.py`: logit p = o(s,i) + mu + v_bench + u_subject + w_(subject,bench)
(+ optional name and name-bench effects, switched off by selection), Gaussian priors, MAP by backfitting (one Newton step per
effect type per sweep, <= 25 sweeps); exact-item effect from other subjects' labels on the same benchmark_id + text (conditional
scale, 4 Newton steps); probit integration of the posterior variance; subject identity = full attribute dict; fit cached per
checkpoint by a fingerprint, atomic under a lock; any failure -> rounds 1-3 per-pair path. Grid 864 settings (pre-registered),
nested selection by fold; shipped (all-fold, = addendum robust pick): **tau_mu 0.5, tau_bench 1.5, tau_name 0, tau_subj 0.5,
tau_namebench 0, tau_pair 0.6, tau_item 2.25, probit on** (`xp|bench1.5|name0|subj0.5|namebench0|pair0.6|item2.25|probit1`). Extended grid: tau_item and
tau_bench optima interior (best A ALC per tau_item 0/0.75/1.5/2.25/3/4: 0.2013, 0.1989, 0.1969, 0.1965, 0.1968, 0.1980).

### Results (Brier ALC; lower is better)

| Config | units / benchmarks | S0 | S1 | XP | pair-rate oracle | XP - S0 [95% CI] | XP - S1 [95% CI] |
|---|---|---|---|---|---|---|---|
| A formative-like (primary) | 13,460 / 135 | 0.2095 | 0.2074 | 0.1966 | 0.1731 | -0.0129 [-0.0151, -0.0109] | -0.0108 [-0.0134, -0.0084] |
| E distinct models (addendum) | 8,673 / 135 | 0.2078 | 0.2074 | 0.1953 | 0.1730 | -0.0125 [-0.0148, -0.0100] | -0.0121 [-0.0146, -0.0095] |
| B one benchmark, all pairs | 6,772 / 135 | 0.2131 | 0.2087 | 0.1842 | 0.1793 | -0.0289 [-0.0342, -0.0239] | -0.0245 [-0.0312, -0.0181] |
| C one fold in one list | 9,577 / 125 | 0.2190 | 0.2084 | 0.1797 | 0.1780 | -0.0393 [-0.0521, -0.0280] | -0.0287 [-0.0384, -0.0191] |
| D A + eval-only small pairs | 14,293 / 135 | 0.2097 | 0.2096 | 0.1969 | 0.1706 | -0.0128 [-0.0150, -0.0108] | -0.0127 [-0.0160, -0.0094] |

Per budget (B0, B1, B3, B7, B15, B31, ALC; XP - S0 with cluster CIs):

| Config | Method | B0 | B1 | B3 | B7 | B15 | B31 | ALC |
|---|---|---|---|---|---|---|---|---|
| A | S0 | 0.2475 | 0.2221 | 0.2098 | 0.2000 | 0.1956 | 0.1924 | 0.2095 |
| A | XP | 0.2475 | 0.2126 | 0.1992 | 0.1868 | 0.1776 | 0.1663 | 0.1966 |
| A | XP - S0 | -- | -0.0096 [-0.0131, -0.0063] | -0.0106 [-0.0133, -0.0082] | -0.0132 [-0.0156, -0.0109] | -0.0180 [-0.0211, -0.0151] | -0.0261 [-0.0301, -0.0224] | -0.0129 [-0.0151, -0.0109] |
| E | S0 | 0.2487 | 0.2282 | 0.2104 | 0.1958 | 0.1886 | 0.1831 | 0.2078 |
| E | XP | 0.2487 | 0.2135 | 0.1976 | 0.1846 | 0.1751 | 0.1628 | 0.1953 |
| E | XP - S0 | -- | -0.0147 [-0.0207, -0.0087] | -0.0128 [-0.0161, -0.0094] | -0.0111 [-0.0135, -0.0085] | -0.0135 [-0.0167, -0.0102] | -0.0203 [-0.0250, -0.0154] | -0.0125 [-0.0148, -0.0100] |

Per outer fold (A, XP - S0): -0.0136, -0.0074, -0.0167, -0.0128, -0.0137 (all negative).
Config D eval-only pairs: -0.0120 [-0.0217, -0.0036].
S1 - S0 (name pooling): A -0.0021 [-0.0050, +0.0009] (n.s.), C -0.0106 [-0.0168, -0.0025].

Components (difference to the shipped setting; positive = removing hurts):

| Variant | A | E |
|---|---|---|
| - benchmark effect | +0.0048 [+0.0032, +0.0064] | +0.0050 [+0.0032, +0.0070] |
| - exact-item effect | +0.0048 [+0.0038, +0.0059] | +0.0053 [+0.0038, +0.0068] |
| - subject-benchmark effect | +0.0010 [+0.0004, +0.0017] | +0.0006 [-0.0000, +0.0014] |
| - subject effect | +0.0004 [+0.0001, +0.0008] | +0.0002 [-0.0002, +0.0006] |
| - probit | +0.0005 [+0.0000, +0.0009] | +0.0006 [-0.0000, +0.0011] |
| benchmark + pair only | +0.0053 [+0.0043, +0.0063] | +0.0056 [+0.0041, +0.0071] |
| name/subject + pair only | +0.0130 [+0.0105, +0.0157] | +0.0143 [+0.0114, +0.0173] |
| item + pair only | +0.0135 [+0.0099, +0.0174] | +0.0144 [+0.0100, +0.0189] |
| pair only | +0.0202 [+0.0163, +0.0243] | +0.0216 [+0.0171, +0.0264] |

-> benchmark difficulty and exact-item difficulty contribute about equally (~0.005 each); subject, pair and probit
components small; benchmark + pair alone recover more than half of the gain; name/subject + pair or item + pair alone recover none.

**Real Ollama testbed, ONE global list over 17 pairs** (`xpair_real_testbed.json`; shipped-global = rounds 1-3 predictor
with the deployed name filter, per-pair = local harness numbers, reproduced exactly at seed 0: 0.2224):
- seed 0: XP 0.2222, shipped-global 0.2295, per-pair 0.2224; XP - shipped-global -0.0074 [-0.0203, +0.0032]; XP - per-pair -0.0002 [-0.0072, +0.0074]
- seed 1: XP 0.2223, shipped-global 0.2315, per-pair 0.2240; XP - shipped-global -0.0092 [-0.0164, -0.0019]; XP - per-pair -0.0018 [-0.0053, +0.0020]
- seed 2: XP 0.2094, shipped-global 0.2232, per-pair 0.2141; XP - shipped-global -0.0138 [-0.0224, -0.0068]; XP - per-pair -0.0047 [-0.0111, +0.0013]
- seed 3: XP 0.2094, shipped-global 0.2184, per-pair 0.2127; XP - shipped-global -0.0089 [-0.0158, -0.0018]; XP - per-pair -0.0033 [-0.0078, +0.0014]
- seed 4: XP 0.2175, shipped-global 0.2296, per-pair 0.2221; XP - shipped-global -0.0121 [-0.0238, -0.0023]; XP - per-pair -0.0046 [-0.0075, -0.0016]
- XP with per-pair lists vs per-pair shipped (if the evaluator were per-pair), seed 0: +0.0060 [+0.0022, +0.0099] (worse);
  on the release (configs-A pairs alone): -0.0016 [-0.0028, -0.0005] (better).
- XP is worse than the deployed predictor at n = 1 on this testbed (B1 0.2549 vs 0.2468).

**Acquisition (fresh-first vs random, XP, config A):** -0.00053 [-0.00087, -0.00021] (gain at B15 -0.0017 [-0.0024, -0.0011],
B31 -0.0017 [-0.0025, -0.0010]; 154,939 skips); S0 unaffected +0.00008 [-0.00013, +0.00032].
Real testbed (global list, stage-wise round robin, fresh vs in-order):
- seed 0: +0.0001 [-0.0010, +0.0011] (skips 608)
- seed 1: -0.0012 [-0.0038, +0.0013] (skips 557)
- seed 2: -0.0027 [-0.0058, -0.0002] (skips 519)
- seed 3: +0.0007 [+0.0001, +0.0013] (skips 553)
- seed 4: -0.0065 [-0.0126, -0.0017] (skips 529)
Pre-registered rule (A CI < 0 and real seed 0 not significantly worse) holds -> **fresh-first shipped** (seed 3 is
significantly worse, +0.0007; seeds 2 and 4 significantly better; reported in the paper).

**Runtime (stdlib model.py, one laptop core; synthetic lists of 250-word items):**

| pairs | labels | fit (s) | ms / call | checkpoint (s) |
|---|---|---|---|---|
| 10 | 310 | 0.01 | 0.102 | 0.1 |
| 100 | 3,100 | 0.07 | 0.094 | 0.5 |
| 1,000 | 31,000 | 0.56 | 0.098 | 5.5 |
| 2,000 | 62,000 | 1.68 | 0.116 | 13.3 |
| 4,000 | 124,000 | 3.68 | 0.128 | 29.3 |

**Checks.** stdlib model.py vs numpy simulator: 56,750 predictions on 24 config-A test sets, max |diff|
1.35e-05 (pre-registered < 1e-3). Official client (`official_loop_check.py`, mock coordinator, real testbed, shipped
files): mode `stream`, 0 predict calls during 1,135 acquisition decisions,
ALC 0.2179 == simulator (True). Smoke test on `submission/`
(`results/round4/smoke_test_submission.json`): 44 base + 220 cross-pair cases OK (unseen subject/benchmark, malformed entries,
determinism, 4 threads, forced fit failure -> fallback, 62,000-label list: fit 1.7 s, 0.12 ms/call).

**Ship decision:** rule (a) holds (A: -0.0129 [-0.0151, -0.0109]), (b) holds (real seed 0
-0.0074 [-0.0203, +0.0032], not significantly positive), (c) holds (B, C, D all significantly better),
(d) holds -> **XP shipped** (`XP_ENABLED = True`), **fresh-first shipped** (`POLICY = "fresh"`).
ZIP rebuilt with `build_submission.py --force`: 19,818 bytes, SHA-256 `c76331d084f175fbf8ce272b00d0a2360d80e440dacf6ede380f3f9eba5172e1`.
**Validator** (`python -I validator/check_submission_zip.py`): `OK: ...paiec_submission.zip looks like a valid submission ZIP.` exit 0.

### Literature check (verified, cited in the paper; `paper/bib_verification_log.json`)
Rodriguez et al. 2021 (ACL, IRT leaderboards), Hofmann et al. 2025 (Fluid Benchmarking, COLM; arXiv:2509.11106), Kipnis et al.
2025 (metabench, ICLR; arXiv:2407.12844), Ye et al. 2023 (BIG-bench predictability, Findings EMNLP; arXiv:2305.14947), Zhuang
et al. 2025 (adaptive testing position paper, ICML; arXiv:2306.10512), Breslow & Clayton 1993 (GLMM / PQL, JASA), MacKay 1992
(probit approximation, Neural Computation); plus tinyBenchmarks and observational scaling laws already cited. Used to justify:
Rasch structure with random effects learned at test time (new benchmarks), coverage-only acquisition (stream allows only
accept/skip; uncertainty sampling did not help in round 1), probit integration.

### Code / files changed
- New: `experiments/src/xpair_sim.py` (build, explore, grid, cv, addendum, recheck, acq, realtest, realacq, perpair, runtime,
  equivalence), `experiments/src/official_loop_check.py`, `experiments/results/round4/*` (PREREG, addendum, 11 JSONs, logs,
  smoke test of the shipped files).
- `model.py`: cross-pair section (`xp_build`, `_xp_predict_keys`, `_xp_state`, `predict_cross_pair`, `xp_clear_cache`,
  `XP_*` constants); `predict` tries it first when `labeled` is non-empty. Per-pair code unchanged.
- `labeling.py`: evaluator entry point without `prediction`; `acquisition_policy` for experiments; own-pair novelty filter by
  full subject key (never the whole list); `fresh` policy (shipped).
- `run_experiments.py`, `novelty_stats.py`: use `acquisition_policy`; `run_experiments.py`, `text_prior.py` set
  `XP_ENABLED = False` (rounds 1-3 experiments keep evaluating the per-pair predictor).
- `smoke_test.py` (+ `round4_checks`), `build_submission.py` (ZIP README describes the cross-pair model), `make_macros.py`
  (+ round-4 macros and 4 tables), `make_figures.py` (+ `xp_budget`), `make_audit.py` (+ 47 round-4 claims).
- Paper restructured: new title/abstract/intro/related work; Sections 2-5 are round 4; rounds 1-3 moved to Appendices B-D.
- `submission/README.md`: new ZIP hash, round-4 description and commands.

### Not done / notes
- The Mac was not needed (all fits are vectorised; the grid took 3.5 min on 4 cores).
- The rounds 1-3 experiment suite was not re-run (its JSONs are unchanged; the per-pair predictor code is unchanged).
- Test-set composition of the hidden pool is unknown; configs A-E bracket it. If the formative feedback looks worse than
  expected, the first knobs are `XP_TAU_ITEM` (2.25) and `XP_TAUS["bench"]` (1.5); `XP_ENABLED = False` restores the
  rounds 1-3 predictor exactly.
- User actions unchanged except: **upload the rebuilt ZIP (SHA-256 `c76331d0...`)**, then fill the author block, rebuild the
  paper, OpenReview.
