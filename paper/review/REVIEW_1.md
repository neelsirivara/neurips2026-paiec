# Self-review pass 1 (reviewer form: PAIEC technical report, NeurIPS 2026)

Reviewed artifact: `paper/main.pdf` built 2026-10-21 from `main.tex` + `results_macros.tex`
(all results present; `make_audit.py`: 50/50 claims hold; 0 `??`, 0 undefined references,
0 overfull boxes, 11 pages of which about 7.6 are main text).

## Summary of the submission

A training-free predictor for P(subject answers item) with a hand-set attribute prior,
beta-binomial shrinkage of the pair success rate, and two item-level components (ridge on
cheap text features, TF-IDF kNN) gated by a leave-one-out test with a 2-SE margin; a
near-duplicate-skipping acquisition policy. Evaluated with a local reproduction of the protocol
on a 15-pair real Ollama testbed and a 160-pair synthetic IRT testbed with oracle bounds.

## Scores (1-5)

| Criterion | Score | Justification |
|---|---|---|
| Clarity | 4 | Method is fully specified with every constant; the metric-driven design logic is stated up front. Dense in places (Results has nine paragraphs of numbers). |
| Soundness | 4 | Paired bootstrap CIs over pairs, identical trajectories across methods, oracle bounds, replication seeds, negative results reported as such. Weakness: 15 real pairs from 4 small models; the subject part of the prior is effectively untested on real data (acknowledged). |
| Reproducibility | 4 | Every number is a generated macro; code, seeds, commands and timings are given; the validator output is included. The code URL is still a placeholder that the author must fill before upload. |
| Significance | 3 | The headline finding (shrink first; item-level adaptation from 31 labels is marginal; embeddings and uncertainty sampling do not help) is useful for the competition community but the method is deliberately simple and no training data was used, so the official score will likely trail data-driven entries. |
| Overall | 3.5 | Honest, well-measured, limited in scope by the data constraint. |

## Strengths

1. The design is derived from the metric (60% of weight at n <= 7) and the ablations confirm
   where the gain comes from (prior, then shrinkage) with CIs.
2. Oracle decomposition of the remaining headroom is a genuinely informative analysis.
3. Robustness: validator passes, fallbacks, determinism, 455 calls/s measured.
4. Negative results (MiniLM vs TF-IDF, sharper TF-IDF, uncertainty sampling, always-on ridge)
   are reported with paired CIs rather than hidden.

## Weaknesses / must-fix before upload

- M1 (blocking, author action): `\codeurl` is a placeholder and the author block reads
  `[Author Name]`/`[Affiliation]`. The report cannot be submitted as is. -> flagged in STATUS.md
  and submission/README.md; no change possible by the builder.
- M2: The abstract reports the real-testbed numbers before the synthetic ones but the Results
  section leads with the real testbed too; fine. However the abstract's "two comparable parts"
  statement should be checkable: the macros give 0.0199 (rate) vs 0.0219 (item) on synthetic and
  0.0221 vs n/a on real. OK as worded ("comparable" refers to synthetic). No change.
- M3: In "Where the gain comes from", the real shrink-vs-prior difference is not significant
  (CI [-0.0307, +0.0002]); the text now says so explicitly (fixed in this pass).
- M4: The gating paragraph originally reported numbers without interpretation and carried a
  REVISE marker; fixed (interpretation added, marker removed).
- M5: The policies table overflowed the text width by 44 pt; fixed with `\resizebox`.
- M6: Table 4 labelled the oracle row only "oracle"; fixed ("oracle: pair success rate").
- M7: Compute statement said 1.5 h for testbed generation; the actual wall-clock on the shared
  server was about two hours and the 7B run was truncated; fixed in Data and Disclosure.

## Minor / suggestions (not applied)

- The title is long (24 words); acceptable for a competition report.
- Section 7 could be shortened by moving the TF-IDF-variant and gating paragraphs to an
  appendix if a stricter page limit applied; main text is within the 4-8 page range.
- A figure of per-pair ALC differences (full vs empirical mean) would make the 80%-of-pairs
  claim visual; omitted for space.
- The synthetic generator's attribute-ability relation is chosen by us; a sensitivity sweep over
  `attr_signal` would strengthen the prior's evaluation (not run: time).

## Checks performed

- `grep '??' main_text.txt` -> 0; `grep -i TODO` -> 0; REVISE markers -> 0.
- `make_audit.py` -> 50 claims re-derived from JSON, 0 failing (paper/AUDIT.md).
- Bibliography: 13 entries, each verified against its source URL (paper/bib_verification_log.json).
- Validator: `OK ... valid submission ZIP`, exit 0 (experiments/results/validator_output.txt).
- ZIP members' SHA-256 equal to experiments/src/model.py and labeling.py.

## Review pass 2 (iteration 2026-10-22/23)

Reviewed artifact: `paper/main.pdf` rebuilt 2026-10-23 (13 pages; main text ends at the bottom of
page 8; references + 5 appendix sections; 0 `??`, 0 undefined references, 0 overfull boxes;
`make_audit.py`: 66 claims re-derived from JSON, 0 failing).

### Must-fixes found and applied in this pass

- M8 (factual error inherited from pass 1): the report said the novelty acquisition filter "never
  fires" on either testbed. A replay (`novelty_stats.py`, `novelty_filter_stats.json`) shows it fires on
  10 of 17 real pairs (templated BBH navigate/boolean items, same-template SVAMP problems; cosine up to
  0.985; 40 candidates skipped) and on 0 of 160 synthetic pairs; its ALC equals stream order to four
  decimals. Text in Results and Limitations corrected; two audit claims added.
- M9: with 17 pairs the real-testbed empirical mean no longer beats the constant at n = 3 (B3 = 0.264);
  the sentence "only overtakes it from n = 3" now quotes the overtaking budget per testbed from a macro
  (3 synthetic, 7 real) and the audit claim checks that budget.
- M10: the Data, Introduction, Abstract, Limitations and Disclosure sections stated that the training
  release was never accessed. It was downloaded on 2026-10-22 after the author's login arrived; every
  statement now says exactly what it was used for (validation of the already-fixed rule; a fitted prior
  that lost under benchmark-held-out CV and was not shipped) and that no shipped coefficient derives
  from it. Disclosure lists revision, tables and directory count.
- M11: page budget. The new material pushed the main text to 69% of page 9; the replication paragraph,
  the policies table, the prior-strength and text-signal paragraphs and the embedding details were
  moved to appendices and several paragraphs compressed. Main text now ends at the bottom of page 8.
- M12: the ZIP README (generated by `build_submission.py`) said "prior strength 2 pseudo-counts"
  while the shipped alpha0 is 4; it now reads its constants from `model.py`.

### Checks performed

- `grep` of the PDF text for stale phrases ("never fire", "no credentials", "not run", "cut short",
  "SVAMP only", "15 pairs", "was not accessed", "placeholder"): none left except the author block, which
  the author fills before upload (M1 of pass 1, unchanged).
- `model.py` refactor verified bit-identical on 7,340 snapshot predictions before/after.
- Official validator on the rebuilt ZIP: `OK ... valid submission ZIP`, exit 0.
- `verify_all.py` run after the final build (macros newer than every result JSON).

## Review pass 3 (round 3, 2026-10-23: item-text difficulty prior)

Reviewed artifact: `paper/main.pdf` rebuilt 2026-10-23 (15 pages; main text ends at the bottom of
page 8; references from page 9; 7 appendix sections including the new "Item-text difficulty prior:
details"; 0 `??`, 0 undefined references, 0 overfull boxes; 1,675 macros; `make_audit.py`: 86 claims
re-derived from JSON, 0 failing).

### What changed in the report
- New Data paragraph "An item-text difficulty prior (pre-registered; not shipped)" and Appendix D
  (pre-registration and outcome, features/target, nested selection and per-fold fits, Table 2 with the
  CV results incl. exploratory rows and references, testbeds, engineering gates, shipped-file
  equivalence, the list of the 144 training-release benchmarks used).
- Abstract, introduction, contributions, limitations, reproducibility and disclosure now state the
  third use of the release (item `content`/`item_features` of 515,067 items for a candidate text prior
  that was not shipped).

### Scores (1-5), updated
| Criterion | Score | Change and reason |
|---|---|---|
| Clarity | 4 | unchanged; the round-3 paragraph is dense but every number has a CI and a pointer to Appendix D. |
| Soundness | 4 | up in substance, not in score: the text prior was pre-registered (rule, grids, folds), evaluated with a cluster bootstrap over benchmarks rather than over pairs (the pair-bootstrap CI is more than an order of magnitude narrower and would overstate precision), reproduces the iteration-2 hand-set numbers exactly, and the exploratory follow-ups are registered and labelled as such. Weakness: the pre-registered penalty grid was too short (boundary hit in every fold); addressed by exploratory E1 (penalties to 1e4), which changes nothing. |
| Reproducibility | 4 | `text_prior.py` re-creates every round-3 number from the payload; hashes of PREREG/addendum are checked at build time. Still needs the gated release to re-run. |
| Significance | 3 | A clean negative result (item text in hashed-n-gram/statistics form does not transfer across benchmarks; W < 0 in every fold) plus a quantified ceiling (benchmark-mean oracle: -0.0061 ALC) is useful to other teams. |
| Overall | 3.5 | unchanged. |

### Must-fixes found and applied in this pass
- M13 (build-breaking): `make_macros.py` emitted two macros named `\TpHandBZero` (all-row B0 vs the
  evaluation-half B0 of the budget table) and two `\TpDiffBZero*` families; LaTeX would have failed on
  `\newcommand` redefinition. Renamed the per-budget family to `...Bud...` and added a duplicate-name
  guard to `mac()`.
- M14 (page budget): the new paragraph pushed the main text 14 lines onto page 9. Moved the full
  acquisition-policy discussion and the full frontier-subject discussion to the appendix (main text keeps
  the headline numbers and CIs), shortened the prior-strength, replication, data-listing and
  limitations sentences. Main text ends at the bottom of page 8 again.
- M15 (inaccurate wording, inherited): Limitations said the novelty filter "fires on templated items of
  both testbeds"; it fires on 10/17 real pairs and 0/160 synthetic pairs. Now "real-testbed items".
- M16: internal jargon "Round 3" in section/paragraph titles replaced by descriptive titles.
- M17 (provenance): the smoke test now runs against the shipped files (`smoke_test.py --dir
  ../../submission`, 42 cases); the text-prior variant is smoke-tested separately with its weights
  loaded (44 cases, `round3/text_prior_smoke_test.json`).
- M18 (consistency of code and claims): `experiments/src/model.py` is no longer byte-identical to the
  shipped `submission/model.py` (it carries the dormant text-prior hooks). The report now says so;
  `round3/shipped_equivalence.json` shows 5,544/5,544 bit-identical predictions and the unchanged ZIP
  SHA-256; `build_submission.py` refuses to overwrite the validated ZIP without `--force`.

### Checks performed
- PREREG.md SHA-256 `bb54af42...` equals the value stored in `text_prior_cv.json` (audit claim);
  addendum SHA-256 `debcd2aa...` equals `text_prior_explore.json`.
- Every round-3 directional statement (no significant change; W < 0 in every fold; boundary penalty in
  every fold; < 5% of the oracle gap closed at n = 0, 1, 3; benchmark-mean oracle significantly better;
  testbeds not significant; exploratory variants not better; ship decision consistent with files) is an
  audit claim that HOLDS.
- Official validator on the unchanged ZIP: OK, exit 0.
