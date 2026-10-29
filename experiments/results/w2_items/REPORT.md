# W2: online within-benchmark item effects (2026-10-29)

Candidate: `submission/candidates/w2_f05s1_k1q2.zip`, sha256
`d6d1c6a00e72b8aaf9e6628de6f03e50274a1d8f547883a0c0d70300aab969bf` (23,436 B) = `xp_level_m050` plus an
item layer learned online from the evaluator's labelled list. Budget 0 is unchanged.

## What the layer does
Both parts shrink to the m050 prediction when evidence is thin.
- **F, category effects (prior sd 0.5).** A random effect per (benchmark, `item_features` key, value),
  fitted jointly with the existing cross-pair effects, for keys with at least 2 values among the
  benchmark's labelled items. A target adds its group's posterior mean (and variance); an unseen value
  adds nothing.
- **K, text kernel (prior sd 1, q = 2).** For a target, the stage-1 residuals of the other labelled items
  on the same benchmark are pooled with weights equal to the squared character-4-gram TF-IDF cosine, and
  added to the logit.
- Standard library only, deterministic; any error falls back to m050. Inputs used: `item_features`,
  `item_content`, and `benchmark_id` only to group the current list.

## Why these two
Infinite-label share of within-benchmark item-effect variance (`diag_ceiling.json`, `diag_cheap.json`):
categories 0.12 (benchmark mean), text kernel 0.04-0.05, cheap text features (length, digits, options)
about 0.015 (dropped).

## Protocol
- Pre-registration `PREREG.md`, 10:58 PDT, sha256 `192748b7...`, before the tuning grid and any
  confirmation-fold run. Tune on benchmark folds 0-1, confirm on folds 2-4.
- Scenarios A, INV, MID, HARD as in `proto_level3`; SPARSE (new) = 9 pairs on 7 INV benchmarks, the shape
  of the formative runs.
- Item text and features recovered for 95,773 items; the replay verifies against the frame for all 135
  benchmarks (`extract_manifest.json`).

## Tuning (folds 0-1, `tune.json`)
18 settings compared on the mean ALC difference over A, INV, MID and HARD. Selected F0.5 (share 1) + K1
(q 2): -0.0016 (A -0.0015, INV -0.0010, MID -0.0018, HARD -0.0020). Best single parts: F0.5 alone -0.0011,
K1 q2 alone -0.0008. All 18 settings were below zero.

## Confirmation (folds 2-4, `confirm.json`)
ALC difference vs m050 (95% cluster bootstrap over benchmarks) and per-budget Brier differences:

| Scenario | Benchmarks | ALC difference | B1 | B3 | B7 | B15 | B31 |
|---|---|---|---|---|---|---|---|
| INV (gate a) | 20 | -0.0035 [-0.0051, -0.0022] | -.0014 | -.0021 | -.0042 | -.0066 | -.0065 |
| A (gate b) | 83 | -0.0024 [-0.0032, -0.0016] | -.0011 | -.0016 | -.0029 | -.0040 | -.0043 |
| SPARSE (gate c) | 20 | -0.0021 [-0.0035, -0.0009] | -.0003 | -.0012 | -.0021 | -.0044 | -.0053 |
| MID | 39 | -0.0019 [-0.0029, -0.0009] | | | | | |
| HARD | 26 | -0.0019 [-0.0033, -0.0009] | | | | | |

All three gate conditions pass (A and SPARSE are significantly better, not just not worse). On INV 90% of
benchmarks improve, median per-benchmark difference -0.0026, and dropping the best benchmark leaves
-0.0031; on A 75% improve. The confirm-fold gain is larger than the tune-fold gain (INV -0.0035 vs
-0.0010), which is benchmark-to-benchmark variation, so expect about -0.002 ALC on the hidden pool: about
5-10% of the perfect-item-difficulty ceiling and well below one formative draw's noise (sd about 0.017).

## Checks on the zip (`python -I`; records in `candidates/w2_f05s1_k1q2/`)
- Organisers' validator OK (exit 0); smoke test 44 cases OK; official streaming loop: stream mode, 0
  prediction calls during acquisition, 31 labels per pair, real-testbed ALC 0.2174 vs m050 0.2179 (the
  shared round-4 record file restored byte-identically).
- Stdlib vs numpy simulator: max |diff| 1.5e-5 over 57,945 predictions; with both strengths at 0 it equals
  m050 exactly. `labeling.py` byte-identical; the m050 and shipped zips keep their sha256.
- Runtime, extreme smoke case (62k labels, 100k targets): first call 6.5 s, later calls 4.9 ms (m050 1.8 s
  and 0.13 ms), measured with another job running; realistic lists cost milliseconds.

## Deviation and pending work
- One code change after pre-registration: the kernel dot product loops over shared keys only (same
  arithmetic; the equivalence check ran on the final code).
- Exploratory runs of the other 17 settings on the confirmation folds go into `confirm.json` under
  `exploratory_other_settings`; they cannot change the selection.
- Real draws on Codabench: see `SWITCH_RULE.md` (five draws, 1019682-1019688).

## Files
Code in `experiments/src/w2_items/` (`w2_feat.py`, `w2_sim.py`, `run_w2.py`, `build_w2.py`,
`w2_equivalence.py`, `w2_block.py.txt`, `extract_all.py`, `diag_ceiling.py`, `diag_cheap.py`); results in
`experiments/results/w2_items/`. Item cache (derived from the training release, outside the repository):
`../_downloads/measurement_db_payload/round4_cache/w2_items_all.jsonl`.

(Written by the coordinator from the W2 agent's final report; subagents cannot write report files.)
