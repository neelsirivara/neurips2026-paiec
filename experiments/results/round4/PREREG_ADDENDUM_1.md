# Round 4, addendum 1 (written 2026-10-23 20:58 PDT, after the primary results, before running anything below)

## What was known when this was written

The pre-registered primary result (`xpair_cv.json`): config A, XP minus S0 = -0.0129 [-0.0151, -0.0109] (rule (a)
holds); configs B, C, D all significantly better than S0 (rule (c) holds); the stdlib/numpy equivalence check
passed (max |diff| 1.35e-5); real testbed (rule (b)), seed 0: XP minus shipped-global -0.0074 [-0.0203, +0.0032]
(not significantly positive; holds). The setting selected on all five folds lies on the grid boundary in two
coordinates: tau_bench = 1.5 (largest value) and tau_item = 2.25 (largest value); probit on; tau_name = tau_namebench = 0.

Two concerns follow, and this addendum fixes how they are handled before looking at any further outcome.

1. **Grid boundary.** The pre-registered grid may cut off the optimum.
2. **Structure of the release vs the test set.** 92.7% of the release's pairs share (model name, benchmark) with
   another pair (test-condition variants of one model, e.g. one model under thousands of conditions on one
   benchmark). Exact-item matches between such variants are far more informative than between different models,
   so tau_item (and the size of the gain) may be tuned to a structure the hidden test set does not have.

## Added configuration E ("distinct models")

As config A (same K, benchmarks, per-benchmark share, 100 items per pair, 300 test sets per fold), but within a test
set no two pairs share a normalised model name: for each chosen benchmark, names not yet used in the test set are
drawn uniformly without replacement, and one pair (uniform) of each drawn name with >= 80 items is taken. Cross-pair
evidence then only comes from different models.

## Extended grid

tau_bench in {0.5, 1.0, 1.5, 2.0, 3.0}; tau_item in {0, 0.75, 1.5, 2.25, 3.0, 4.0}; the other coordinates as
pre-registered (tau_mu 0.5; tau_name {0, 0.5}; tau_subj {0, 0.5}; tau_namebench {0, 0.5, 1.0}; tau_pair
{0.3, 0.6, 0.9}; probit {on, off}): 2,160 settings, evaluated on configs A and E.

## Which setting ships (the XP-vs-shipped decision itself is unchanged: it is made by the pre-registered rules)

- **Robust pick** = the extended-grid setting with the lowest average of the config-A and config-E mean ALCs over all
  five folds. Its nested version (per outer fold, picked on the other four folds) is reported on A and E.
- Ship the robust pick iff its nested config-A ALC is not significantly worse than the pre-registered nested XP
  (cluster-bootstrap CI of the difference not entirely above 0) and, on config E, the robust nested pick minus S0
  has a CI entirely below 0. Otherwise ship the pre-registered all-fold pick.
- Whatever ships is re-checked on the real testbed (rule (b)) and configs B, C, D (rule (c)) and reported.

## Reported regardless

XP (pre-registered pick) on config E vs S0 and S1; components on E; the tau_item profile on A and E (best ALC per
tau_item value).
