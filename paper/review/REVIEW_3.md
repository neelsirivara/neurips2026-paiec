# Self-review pass 3 (2026-10-29): selected submission changes to 1019682 (W2 item layer)

Reviewed: `paper/main.pdf` built with `experiments/src/build_paper.sh` (four pdflatex passes): 27 pages, main text
pages 1-7 (references start on page 7, after about three quarters of it), appendices A-N. 0 undefined references, 0 overfull boxes,
0 `??`, no rerun warning. `make_audit.py`: 211 claims, 0 failing. Previous versions kept as
`main_v1010370.*` (round 4) and `main_v1010717.*` (the 1010717 report, with `results_macros_v1010717.tex`).

## What changed

- Selected submission: **1019682** = `submission/candidates/w2_f05s1_k1q2.zip` (sha256 `d6d1c6a0…`, verified against
  `selected_submission.json`, `w2_results.json` and the build record). The ID is read only from the new
  `experiments/results/selected_submission.json` (status: provisional until the fifth W2 draw, 1019688, lands).
- Main text rebuilt around three layers: cross-pair model, recalibrated level (1010717 = base), W2 item layer.
  New: Eq. 3 (kernel), "Design decisions" (level + why an item layer: W1 information limit, oracle ceilings,
  infinite-label R² of the signals), Sec. 5.1 simulator gate (Table 3), Sec. 5.2 formative draws (Table 4,
  Figure 1: every draw and the arm means), Sec. 5.3 negative results (Qwen3-VL-4B rating, W2-sem, W1, Qwen3-14B
  placeholder), new Limitations/Conclusion/Reproduction/Disclosure.
- Moved to the appendix: the level variants' formative table and figure, the round-4 configuration table and
  development paragraphs, the long version of the Qwen3-VL study. New appendix sections: "Formative draws"
  (every draw) and "Item layer (W2): details" (grids, checks, W2-sem, W1).
- Numbers: all from `w2_items/{w2_results,confirm,tune,draws,confirm_sem,diag_ceiling,diag_cheap}.json`,
  `PREREG.md`, `PREREG_SEM.md`, `SWITCH_RULE.md` (thresholds parsed from the text), `deep_dive/w1/*.json`,
  `selected_submission.json` and `qwen3_14b_status.json`, through `make_macros.py`.

## Checks made while writing

- Arm means, sds and the pair-matched comparison were re-derived from the saved Codabench logs with
  `parse_feedback.py` and match `draws.json` exactly (14 shared pairs, ALC −0.0033, se 0.0022, 10 of 14 better,
  budget-0 difference exactly 0). Each draw's uploaded file name (from `details.json`) matches its arm.
- The arm difference (−0.0229) is already −0.0205 at budget 0, where W2 is identical to the base: the text calls
  this draw composition and does not credit the arm difference to the layer.
- Switch rule: conditions 2 and 3 hold on the four finished W2 draws; condition 1 is pending. The pending draw
  would need ALC > 0.392 (worse than the constant 0.5) to overturn condition 2; the text says so and calls the
  selection final once that run finishes without an error. **If 1019688 ends with a platform error, condition 1
  fails and the rule as written keeps 1010717** — `make_macros.py` then prints a bold warning instead of the
  selection sentence (`RuleSelectSentence`), and the audit's switch claims would need revisiting.
- W1 wording follows the data: the exact posterior's only gain (H7) comes from budget 0 (0.1·ΔB0 = −0.0020 vs
  ΔALC −0.0019); faster early updates are significantly worse on H7-INV. DEEP_DIVE's "does not beat m050 at B1–B3
  in any scenario" was not repeated literally, because NPEB's H7 B1 is slightly lower than m050's (0.2271 vs
  0.2299, no CI on file).
- The simulated effect range in the text is the full range over the five confirmation scenarios
  (−0.0019 to −0.0035), not just SPARSE to INV.

## Scores (1-5)

| Criterion | Score | Note |
|---|---|---|
| Soundness | 4 | Pre-registered tune/confirm split, CI-based gate, real draws read honestly (composition vs effect). The real-draw effect is not significant on its own; said plainly. |
| Clarity | 4 | One story: three layers, why each, what the simulator and the draws show. Section 5.2 is dense but split into four paragraphs. |
| Reproducibility | 4 | Exact ZIP + hash, commands for every stage; absolute paths in `dd_common.py` / `w2_common.py` flagged. |
| Experimental analysis | 4 | Five scenarios, per-budget effects, pair-matched real-draw check, three negative results. |
| Significance | 3 | A small, honest gain (about 0.002–0.0035 ALC) on a mid-table entry. |

## Open items for the coordinator / author

- Confirm the selected ID when 1019688 lands: update `experiments/results/selected_submission.json` (status) if
  needed and `w2_items/draws.json` (the W2 agent); rerun `bash experiments/src/build_paper.sh`.
- Qwen3-14B (resolved 2026-10-29): status `not_evaluated` (rating stopped at 4,865 of 20,655 prompts, gate never
  ran). `make_macros.py` now supports `pending`, `not_evaluated`, `failed` and `passed`; the generated sentence
  says the rater was pre-registered (Amendment 6, 12:53 PDT) and stopped before its gate, with the partial
  ratings kept, and the audit checks that it says neither "failed" nor "passed".
- Before upload: make the repository public and push `submission/candidates/`, `experiments/src/w2_items/`,
  `experiments/results/w2_items/`, `experiments/results/selected_submission.json`, `qwen3_14b_status.json`, the
  deep-dive folders and `paper/`.
