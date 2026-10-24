# Round 4 pre-registration: cross-pair evidence in the official evaluator (written 2026-10-23, before any outcome was computed)

## Why

The official streaming client (`paiec_baseline/tools/streaming_ingestion.py`, `run_streaming`, the `evaluate`
branch: `prediction_stream(model_path, unique, labeled, ...)` with `labeled = event["labeled"]`) passes ONE global
`labeled` list to every `predict()` call of a budget checkpoint; the competition page says `labeled` "contains
[[subject, item], label] entries across sampled subject-benchmark pairs". The shipped predictor uses only the
entries whose `benchmark_id` and `normalized_name` equal the target's (`model.pair_evidence`), and falls back to
ALL entries when none match. Iterations 2-3 showed that benchmark difficulty dominates the residual error and is
invisible at budget 0 (knowing each held-out benchmark's mean would gain about -0.006 ALC). From budget 1 on,
other pairs' labels reveal it.

What had been looked at before this file was written: the evaluator code and page text above; the round-4 frame
(below) built and summarised for structure only (no labels or predictions): 16,718 pairs, 3,118,742 rows,
165,764 distinct items, 144 benchmarks, 16,378 distinct full subject keys, 647 model names; 92.7% of pairs share
(name, benchmark) with another pair (test-condition variants of one model); 342 names occur on >= 2 benchmarks,
211 full subject keys do; 98.5% of rows are on an item that >= 2 pairs answered; median 17.5 pairs per benchmark.
No predictor was run on the frame. The SHA-256 of this file is stored in every round-4 result JSON.

## Frame (`xpair_sim.py build`, outside the repo in `_downloads/measurement_db_payload/round4_cache/`)

The iteration-2 pair definition (subject x test condition, >= 20 items, label = mean response over trials, the
test condition appended to `subject_features_extra`), but each pair keeps the <= 200 answered items with the
lowest benchmark-level hash rank (`blake2b("r4|dir|item_id")`), so subjects of a benchmark share items as far as
their answered sets allow (iteration 2 sampled each pair independently). Item identity = (benchmark, items-table
`content_hash`). Hand-set prior logit o = clip(z_subject + z_item, +-1.5) exactly as `model.py`.

## Simulator (`xpair_sim.py`)

Folds: the 5 benchmark folds of iteration 2 (`prior_fit.json` `cv.folds`). A *test set* is a set of pairs from one
fold's benchmarks that is evaluated together, i.e. that shares one global `labeled` list. Per pair (fixed across
test sets, seed = pair index): 50/50 split of its used items into acquisition and evaluation pools; acquisition
order = a uniformly random permutation of the acquisition pool (the evaluator's default policy accepts with
probability labels_remaining / items_remaining, which yields a uniformly random subset in stream order);
budget b = its first b acquired labels; budgets 0, 1, 3, 7, 15, 31. At budget b every prediction in the test set
receives the union of all its pairs' first-b labels. Brier per pair and budget over its evaluation items; ALC
with the official weights; one ALC per (test set, pair).

Configurations:
- **A (primary, "formative-like")**: the page caps a formative evaluation at 1,000 subject-item pairs. Per test set:
  K ~ U{1, 2, 3} benchmarks drawn uniformly from the fold's benchmarks that have a pair with >= 80 items; 10 pairs
  split as evenly as possible over them, drawn uniformly among each benchmark's pairs with >= 80 items (all if
  fewer); each pair uses its first min(n, 100) items (<= 1,000 subject-items per test set). 300 test sets per fold.
- **B ("benchmark-all")**: one test set per held-out benchmark with all its pairs with >= 80 items (a uniform random
  300 if more), <= 200 items per pair.
- **C ("fold-all", scale)**: one test set per fold with all its pairs with >= 80 items (a uniform random 2,000 if
  more), <= 100 items per pair: many benchmarks mixed in one list.
- **D ("with-small")**: config A plus up to 2 pairs with 20-79 items from the test set's benchmarks, evaluated on all
  their items at every budget and never acquired (one reading of the page, which does not say how such pairs are
  handled).

## Predictors

- **S0, shipped as deployed**: the shipped rule in closed form (hand prior; beta-binomial alpha0 = 4 toward
  sigmoid(clip(z_subject + mean z_item of the evidence)); the target's item-logit offset) with the shipped
  `pair_evidence` rule applied to the GLOBAL list (same benchmark_id and normalized_name; all entries if none).
  The item-level components (ridge, TF-IDF kNN, LOO gate) need item text and are not simulated on the release
  (their effect was below 0.001 ALC on both local testbeds); they are included on the real testbed.
- **S1, shipped per pair**: the same with the target pair's own labels only (= the local harness numbers).
- **XP, cross-pair** (`model.predict_cross_pair`; stdlib; deterministic): stage 1, a random-effects logistic model
  fitted by MAP on the global list, logit p = o + mu + v_bench + a_name + u_subject + c_(name,bench) + w_(subject,bench),
  every effect ~ N(0, tau^2), backfitting with one Newton step per effect type per sweep (<= 25 sweeps, tol 1e-4);
  stage 2, if the target item (same benchmark_id and item text) was labelled for other subjects, an item effect
  d ~ N(0, tau_item^2) on the conditional scale (stage-1 logits x sqrt(1 + pi tau_item^2 / 8)), 4 Newton steps;
  optional probit integration of the posterior variance (sum over effect types of 1 / (H + 1/tau^2), tau^2 for an
  unseen group). Budget 0 (empty list) = the shipped prior. Clip [0.01, 0.99].
  Subjects are identified by all visible attributes; `name` = lower-cased `normalized_name`.

Grid (864 settings): tau_mu = 0.5 fixed; tau_bench in {0.5, 1.0, 1.5}; tau_name in {0, 0.5}; tau_subj in {0, 0.5};
tau_namebench in {0, 0.5, 1.0}; tau_pair in {0.3, 0.6, 0.9}; tau_item in {0, 0.75, 1.5, 2.25}; probit in {on, off}
(tau = 0 switches an effect off). **Selection (nested):** for outer fold f, the setting with the lowest mean
config-A ALC over the test sets of the other four folds; evaluated on fold f's test sets. The shipped setting is
the one selected on all five folds. The grid is computed with a vectorised numpy re-implementation of the same
equations; the selected settings are re-run with the stdlib `model.py` code on dict inputs for >= 20 config-A test
sets (pre-registered check: max |p_numpy - p_stdlib| < 1e-3).

## Outcomes

**Primary:** config A, per-(test set, pair) ALC difference XP (fold-selected) minus S0; mean and 95% percentile
cluster bootstrap over benchmarks (cluster = the pair's benchmark; 2,000 resamples, seed 0).
**Secondary (reported, not used for selection):** the same for configs B, C, D; XP minus S1; Brier at each budget
with CIs; components (each effect switched off in turn from the selected setting, and "benchmark only",
"name/subject only", "pair only", "+ item"); pair-rate oracle.

## Acquisition (step 3)

One information-driven policy, "fresh-first": skip a candidate whose item (benchmark_id + item text) is already in
the global list for another subject unless labels_remaining / items_remaining >= 0.5 (so the 31 labels are always
spent); otherwise accept (stream order = random). *Amendment 20:50 PDT, before any outcome was computed:* with 100-item pairs the pool has 50 items and 31/50 = 0.62 >= 0.5 from the first candidate, so the 0.5 threshold would make the policy identical to random; the threshold is replaced by the minimal one that still spends the budget: skip a non-fresh candidate only while labels_remaining < items_remaining. Simulated in config A with stage-wise round-robin acquisition
(for b in 1, 3, 7, 15, 31: each pair in test-set order streams until it holds b labels), predictor XP (shipped
setting). Ship iff config-A ALC difference (fresh-first minus random) has a cluster-bootstrap CI entirely below 0
AND it is not significantly worse on the real testbed. Engineering change pre-registered regardless: the shipped
policy never uses `prediction`, so the parameter is removed from `acquisition_function`'s signature (the evaluator
then does not call predict for every candidate, which with XP would refit the global model per candidate).

## Ship rule (XP)

Ship XP (set `XP_ENABLED = True` with the setting selected on all five folds) iff
- (a) the primary CI lies entirely below 0, **and**
- (b) on the real Ollama testbed (17 pairs evaluated in ONE global list; per-pair splits and random acquisition as
  in `run_experiments.py`, seed 0), the ALC difference XP minus the shipped full predictor (applied to the same
  global list) is not significantly positive (paired bootstrap over pairs, 1,000 resamples), **and**
- (c) no secondary configuration (B, C, D) shows XP significantly worse than S0, **and**
- (d) engineering: predict never raises (fallback to the shipped path), deterministic, smoke test (extended with
  cross-pair and large lists) and official validator OK; one fit + all predictions of a checkpoint with a
  62,000-label global list in <= 60 s on one core, cache-hit calls <= 5 ms on average.
Otherwise the shipped predictor stays unchanged and the result is reported as is.
