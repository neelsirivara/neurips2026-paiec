# Self-review pass 2 (PAIEC technical report, revision for the selected submission 1010717)

Reviewed: `paper/main.pdf` built 2026-10-26 from `main.tex` + `results_macros.tex` (`build_paper.sh` plus one
extra pdflatex pass; see "Build" below). 24 pages; main text pages 1-7 (references start at the top of page 8),
appendices A-M on pages 9-24. `make_audit.py`: 169 claims, 0 failing. 0 `??`, 0 undefined references or citations,
0 overfull boxes. The previous version is kept as `main_v1010370.tex` / `main_v1010370.pdf` (20 pp, macros in
`results_macros_v1010370.tex`).

Form: the organisers' criteria (adapted from the NeurIPS 2026 E&D reviewing guidelines): technical soundness,
clarity, reproducibility, quality of experimental analysis; novelty secondary. Competition-page requirements:
NeurIPS template, about 4-8 pp of main text, not anonymous, describes data, submitted approach, implementation and
setup, results, limitations, rerun instructions, and discloses training sources, pretrained models and LLM
coding-assistant use.

## Requirement check

| Requirement (competition page) | Where | Status |
|---|---|---|
| NeurIPS template, non-anonymous | `neurips_2026.sty [final]`; author block Neel Sirivara / University of Washington / nsiriv@cs.washington.edu | met (audited) |
| ~4-8 pp main text | 7 pp | met (about 1 page free for the item-difficulty section) |
| Data | Sec. 2 "Data" (training release, hidden-pool structure, local testbed) | met |
| Submitted approach + implementation | Sec. 3, Table 1 (the three constants, parsed from the ZIPs) | met |
| Experimental setup | Sec. 4 (populations), 5.2 (protocol), App. C | met |
| Results | Sec. 5.1 (real formative runs, Table 3, Fig. 1), 5.2, Table 2 | met |
| Limitations | Sec. 6 | met |
| Rerun instructions | Sec. 7: exact ZIP + SHA-256, commands, caveats | met, with two caveats (below) |
| Training sources, pretrained models, LLM assistant | Sec. 8 | met |
| Selected Codabench ID in the paper | abstract, Sec. 1, 3, 4, 5, 7 (`\SelId` = 1010717) | met (audited) |
| Links to released materials | `\codeurl` | **author action**: repository is private until flipped; deep-dive files must be pushed |

## Scores (1-5)

| Criterion | Score | Justification |
|---|---|---|
| Soundness | 4 | Paired, cluster-bootstrapped simulator results; real formative runs reported as unpaired draws with benchmark-matched differences and their standard errors; the one claim the data support (budget 0 below the constant) is separated from the ones they do not (ranking of variants). Weakness: the case for the level change rests on one budget-0 observation plus simulation; the probability of B0 > 0.283 under a release-like pool was not estimated (stated in Sec. 6). |
| Clarity | 4 | Main text is organised around the selected submission (method, why, results, limitations, rerun). Dense in Sec. 4 "The choice"; Table 2 is compact (CIs only for the selected variant; full CIs in Table 6). |
| Reproducibility | 4 | Exact ZIP and hash, rebuild command, organisers' checks, simulator commands, generated numbers, claim audit. Caveats: deep-dive scripts hard-code absolute paths (`dd_common.py`); the dataset download takes the current head, not the pinned revision; formative logs are private by design. |
| Experimental analysis | 4 | Seven simulated populations bracketing the hidden pool, ceilings for side information, honest noise analysis. |
| Significance | 3 | A careful, well-measured entry; the score is mid-table and the main lesson (hand-set "newer is stronger" priors hurt on the hidden pool; most headroom is item-level) is useful but modest. |

## Must-fix items found in this pass and applied

1. Hand-typed numbers in new text ("about 0.007", "40-120 items", "$-1$", "Table 2", constants in the command line)
   -> replaced by macros (`\DdSelGainBest`/`\DdSelCostMax`, `\FmSelItemsPerPair`, `\MhundIntercept`, `\ref`,
   `\SelInterceptRaw` etc. inside `alltt`). Submission IDs in text and table headers are macros from the run record.
2. Table 5 caption first said "No entry reaches two standard errors"; false at one per-budget entry
   (1010715 - 1010370 at n=1, |t| = 2.42). Rewritten: ALC column none; 1 of 42 entries reaches 2 se; 0 reach the
   5% t critical value with 6 df (2.45, computed in `make_macros.py`). Audited.
3. "The release-year term ... cannot help" was too strong (it still separates 2025 from 2026 releases) -> reworded.
4. "D = B0 - B31 does not depend on draw difficulty as much as the ALC" was unsupported -> removed.
5. "The four runs do not show 1010717 is better ... beyond budget 0" implied a paired budget-0 comparison between
   runs -> rewritten: the runs show only that each recalibrated prior beat the constant at budget 0 on its own draw.
6. Oracle-table labels said "true item effect"; the oracle uses each item's leave-pair-out effect from other
   release subjects -> relabelled.
7. Leaderboard median: the brief quoted 0.1700, which is the median of the first API page (50 entries); the median
   over all 58 entries is 0.1768. The report uses 0.1768 (derived from both pages by `make_macros.py`).
8. Old appendix statements that described the round-4 constants as "shipped"/current (intercept kept at 0, subject
   part active) -> qualified as round-4 values, with the selected values stated (App. D, K, offset-table caption,
   validator appendix now covers the selected ZIP).
9. Disclosure now states formative-feedback use (aggregates only, no inversion), reused organiser code, and the
   AI coding assistant plainly. Limitations now name "choosing the level after formative feedback" as mild tuning
   on the test pool.

## Remaining items (not fixable by the builder)

- Author: make https://github.com/neelsirivara/neurips2026-paiec public and push `submission/candidates/`,
  `experiments/src/deep_dive/`, `experiments/results/deep_dive/`, `experiments/results/leaderboard_context.json`,
  `DEEP_DIVE.md` and this `paper/` before the OpenReview upload.
- `experiments/src/build_paper.sh` runs pdflatex three times; this revision needs a fourth pass to clear a
  "Label(s) may have changed" warning (only page numbers in `main.aux` move; visible numbers are unchanged).
  The committed `main.pdf` is from a clean fourth pass. Adding one more pdflatex line to the script would fix it.
- `deep_dive/dd_common.py` uses absolute Windows paths; worth making relative before release.
- Not verified here: that `build_candidate.py` reproduces `xp_level_m050.zip` byte for byte (not re-run, so the
  submission files stay untouched); the organisers' `tools/run_smoke_test.py` was not run by us (only
  `check_submission_zip.py` and `run_streaming` via our mock coordinator).
- If the Oct 28 item-difficulty experiment passes its gate, about one page of main text is free for it; it would
  need its own disclosure (API/model use) in Sec. 8.

## Deltas vs the previous version (main_v1010370)

- Title, abstract and Sections 1-8 rewritten around submission 1010717; new Sec. 4 (diagnosis and choice),
  Sec. 5.1 (real formative runs), Table 1 (constants), Table 2 (populations), Table 3 + Figure 1 (formative runs).
- Round-4 development results condensed to Sec. 5.2 + Table 4; per-budget, component and runtime tables and the
  per-budget figure moved to App. C. Rounds 1-3 material unchanged in App. D-M apart from the qualifications above.
- New appendices A (formative details, matched differences, calibration, checks, full population table) and B
  (deep-dive diagnosis, what did not help, ceilings).
- Main text 7 pp (was 7); total 24 pp (was 20).

## Pass 3 (2026-10-29): the pre-registered item-difficulty study (negative)

Added Section 5.3 (about half a page), an appendix paragraph and Table 8, and updates to the abstract, contributions,
limitations, reproduction commands and disclosure. All numbers come from
`experiments/results/deep_dive/items/eval_items_qwen3vl_4b.json`, the PREREG files (hashes; the timestamps the files
state: 13:08, 13:18, 13:31, 14:08 PDT on 2026-10-26) and the ratings file, through `make_macros.py`; the deployment
probes from `experiments/results/deep_dive/probes_summary.json` (new, derived from the saved Codabench logs).
`make_audit.py` adds 13 claims (182 total, 0 failing), including a re-derivation of the gate from the CIs and the
order of the pre-registration, amendments and gate evaluation by file time.

Checks made while writing it:
- The gate fails on condition (a) in both comparisons; condition (b) holds in both. The text says exactly that,
  not "neither comparison passes either condition".
- The 0.089 / 0.037 correlations are medians over 26 and 21 benchmarks; the text says "medians over benchmarks".
- The deployment claim is worded as the probes' design allows: B's valid score implies load and forward pass
  unless the probe did not run; C's PAIEC-PREDICT-002 is returned only after a successful load on a GPU of at
  least 70 GB (branch checked in `probe_C/model.py`). Probe mapping 1010835/7/65 -> probe_A/B/C is read from
  each run's `details.json`.
- The Mac is described as "a Mac GPU (MPS backend)", not as a specific chip, which nothing on file records.
- Main text is now 7.6 pages (Disclosure ends about 60% down page 8; references follow on page 8). Total 25 pages.
  `build_paper.sh` (now four pdflatex passes): 0 undefined, 0 overfull, 0 `??`, no rerun warning.

Not added (out of scope of the request, flagged to the coordinator): probes A (1010835) and B (1010837) return the
selected predictor's predictions, so their formative scores are two more draws of 1010717's predictor; they would
illustrate the draw noise of Section 5.1 directly.

## Pass 4 (2026-10-29): three draws of the selected predictor

Probes 1010835 and 1010837 run the selected predictor (their `model.py` is the selected `model.py` byte for byte plus
a wrapper that returns `_core_predict`; `labeling.py` identical; with the probe disabled their local-testbed ALC in
the organisers' client equals the selected ZIP's). Their formative scores are parsed from the saved Scoring output
into `experiments/results/deep_dive/selected_predictor_draws.json` (aggregates only) by `make_macros.py`.

- The selected predictor: 0.1628, 0.1920, 0.2143 on three draws; mean 0.1897, sd 0.026, se 0.015. The mean, not
  the best draw, is now given wherever 1010717's score appears (abstract, Sec. 5.1, leaderboard paragraph, new
  Sec. 7 Conclusion; audited paragraph by paragraph), and Table 3 / Figure 1 show all three draws and their mean.
- The three-draw range (0.052) exceeds the range of the four submission runs (0.031): the text now says plainly that
  the gaps between our runs are draw noise and that the variants have about the same expected score; 1010717 is
  kept because the simulator favours it.
- **Correction forced by the new draws:** one draw of the selected predictor (1010837) has B0 = 0.2568, above the
  constant's 0.25. The earlier sentences "every recalibrated run predicted budget 0 better than the constant" and
  "beat the constant at budget 0 on each of its draws" were removed; the text now reports 1 of 5 recalibrated draws
  above 0.25 and the selected predictor's mean B0 of 0.2432 against 0.2834 for the previous submission.
- At the board, the three-draw mean would rank about 46 (the best draw 22).
- Main text: exactly 8 pages (references start at the top of page 9), after trimming a few sentences in Sec. 3, 4,
  5.1, 5.2, 7 and 9 (no content removed that the report relies on). Total 25 pages. Audit: 190 claims, 0 failing.
