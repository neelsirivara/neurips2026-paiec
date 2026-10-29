# W2 pre-registration: item effects learned online within each benchmark

Written 2026-10-29 10:58 PDT, before any run on the confirmation folds and before the tuning grid below was run.
Code: `experiments/src/w2_items/` (sha256 of each file at this point is listed at the end).

## What was done before this file

All of it used only the tune folds (0, 1) or the release's item tables, never folds 2–4:

- **Ceiling diagnostics** on the INV + AS items, with infinite labels:
  - Files: `diag_ceiling.json`, `diag_cheap.json`.
  - Target: the within-benchmark R² of the empirical item effect.
  - Results:
    - `item_features` categories: bench mean 0.118, item-weighted 0.086.
    - Character-4-gram TF-IDF kernel smoother: bench mean 0.04–0.05.
    - Cheap text features (length, digits, options, newlines, question marks): bench mean 0.014–0.019, median about 0.
  - Idea (c), cheap text features, is therefore **dropped**.
- **Extraction** of item_features and text for all 95,773 items that a config-A test set can use:
  - Files: `extract_manifest.json`, `extract_all.log`.
  - The replay is verified against the frame for all 135 benchmarks.
- **Development run** on the tune folds, A and INV (`explore_tune.json`). ALC difference vs m050, with 95% CIs:

  | Variant | INV-tune | A-tune |
  |---|---|---|
  | F τ_f 0.5, share 0.5 | −0.0005 [−0.0016, +0.0001] | −0.0008 [−0.0015, −0.0002] |
  | F τ_f 1.0 | +0.0001 | −0.0003 |
  | K τ_k 1, q 8 | −0.0001 | −0.0004 [−0.0007, −0.0002] |
  | K τ_k 1, q 4 | −0.0004 [−0.0011, +0.0000] | −0.0006 [−0.0010, −0.0003] |

- **Implementation check** on a dev build (F 0.5 / share 0.5 + K 1 / q 4) on 12 INV-tune test sets:
  - The stdlib layer reproduces the numpy simulator: max |diff| 1.2e-5 over 29,450 predictions.
  - The featurisation is identical in the two implementations.
  - With both strengths at 0 the build equals xp_level_m050 exactly.
  - Record: scratchpad dev build, to be redone on the candidate.

## Base and layer

**Base.** `xp_level_m050` (zip sha256 `cda9fe9c…`):
- subject terms off, intercept −0.5, τ_mu 2, the other XP constants as shipped;
- in the simulator, `w2_sim.eval_w2` with both strengths at 0. That equals `dd_common.eval_xp` for m050 with max |diff| 0.

**Layer** (`w2_sim.py` docstring; the deployed code is `w2_block.py.txt`, patched in by `build_w2.py`):
- **F (category effects).** A random effect f_(bench, key, value) ~ N(0, τ_f²) per `item_features` key.
  - It is added to the stage-1 cross-pair fit: one effect type per key rank, joint MAP.
  - Only active keys get an effect. Among the distinct labelled items of the benchmark carrying the key, it needs
    ≥ 2 values and ≤ share × (number of those items) values.
  - A target adds the posterior mean and variance of its own group. An unseen group adds 0.
- **K (text kernel).** δ = Σ_j K_j G_j / (Σ_j K_j H_j + 1/τ_k²).
  - The sum runs over the other distinct labelled items j of the target's benchmark.
  - G_j and H_j are the summed stage-1 residuals and weights of item j.
  - K_j = cos^q of hashed char-4-gram TF-IDF, with idf from the benchmark's labelled items.
  - The prediction is σ(logit p + δ).
- Both shrink to m050 as evidence → 0 or strength → 0.

## Scenarios

- **Simulator.** `xpair_sim.py` unchanged: config A, R = 300 test sets per fold, cap 100, random acquisition, 5
  benchmark folds of `prior_fit.json`.
- **Benchmark sets.** As in `deep_dive/proto_level3.py`:
  - A: all benchmarks;
  - INV: the 33 inventory benchmarks;
  - MID: benchmarks with mean pair rate ≤ 0.5;
  - HARD: mean pair rate ≤ 0.4.
- **New: SPARSE.** Formative-like: the four formative runs had about 9 pairs on 7 benchmarks.
  - Benchmarks: the INV benchmarks of the folds.
  - Per test set, K = min(7, available) benchmarks are drawn without replacement.
  - It has 9 pairs: one per benchmark, plus 9 − K extra pairs on uniformly drawn benchmarks.
  - Pairs are drawn without replacement from the eligible pairs (≥ 80 items), cap 100.
  - Seed RandomState(fold·7 + 21) (`w2_sim.make_sparse`).

## Split

- **Tune:** folds 0, 1.
- **Confirm:** folds 2, 3, 4.
- Folds partition the benchmarks, so the tune and confirm benchmarks are disjoint. Each scenario keeps its own
  benchmarks within those folds.

## Tuning (run after this file is written: `run_w2.py tune F`, `tune K`, `select`)

- **Grid F:** τ_f ∈ {0.25, 0.35, 0.5, 0.7} × share ∈ {0.5, 1.0}.
- **Grid K:** τ_k ∈ {0.5, 1, 2} × q ∈ {2, 4, 8}.
- **FK:** the best F combined with the best K.
- **Criterion.** The mean, over the four tune scenarios A, INV, MID and HARD, of the mean ALC difference vs m050.
  Lower is better.
- **Selected variant.** The lowest criterion among the 8 + 9 + 1 = 18 candidates. Nothing else is selected.

## Confirmation (`run_w2.py confirm`, folds 2–4)

**Statistic.** The selected variant's ALC difference vs m050:
- per unit, then averaged;
- 95% percentile cluster bootstrap over benchmarks, 2,000 resamples, seed 0 (`xpair_sim.cluster_boot`).

**Gate (all three must hold):**
- (a) **INV:** the CI is entirely below 0 (a CI-separated gain);
- (b) **A:** the CI lower bound ≤ 0 (not significantly worse);
- (c) **SPARSE:** the CI lower bound ≤ 0 (not significantly worse).

**Reported without gating:**
- MID, HARD and the per-budget differences;
- after the gate is recorded, the other grid settings on INV and A of the confirm folds. These are exploratory and
  cannot be shipped.

**Decision.**
- **If the gate passes:**
  1. Build `submission/candidates/w2_<label>.zip` with the selected constants (`build_w2.py --checks`).
  2. Run the organisers' validator, `smoke_test.py` and `official_loop_check.py`, all with exit 0.
  3. Run the stdlib equivalence check (`w2_equivalence.py`): 24 test sets, max |diff| < 1e-3, and zero strength
     equals m050 exactly.
  4. Message the coordinator with the path, sha256 and simulated per-budget gains.
- **If the gate fails:** no zip. Nothing is re-tuned on the confirm folds, and no other variant is promoted from them.

**Rules.**
- Only the evaluator's inputs are used: `item_features`, `item_content` and `benchmark_id`, the last only to group
  the current list.
- Nothing is keyed on anonymous IDs across calls. No identification of benchmarks or models.

## Code sha256 at registration

- build_w2.py 9094abbf…, diag_ceiling.py c9ab7c55…, diag_cheap.py e0a4be0b…, explore_tune.py e2a8f234…
- extract_all.py c137c36b…, run_w2.py 2d22f103…, w2_common.py 1036c505…, w2_equivalence.py 1a07cda5…
- w2_feat.py 5b82c47e…, w2_sim.py 3aa67873…, w2_block.py.txt c42c99b0…
