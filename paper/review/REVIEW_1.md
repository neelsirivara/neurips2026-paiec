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
