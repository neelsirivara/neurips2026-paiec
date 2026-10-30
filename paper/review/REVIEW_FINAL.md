# Final review: PAIEC technical report (selected submission 1019682)

Reviewer stance: a demanding NeurIPS 2026 Evaluations & Datasets reviewer, using the competition's emphasis
(technical soundness, clarity, reproducibility, quality of experimental analysis; novelty secondary) and the E&D
guidelines ("are the authors careful and honest about evaluating both the strengths and weaknesses of their work?",
"could another team re-run it from this paper?", "statistical soundness, rigor, and validity").

Reviewed: `paper/main.pdf` as built on 2026-10-30 (27 pages; main text pages 1-7, references from about 70% of page
7), `paper/main.tex`, the generated tables, `experiments/results/w2_items/{SWITCH_RULE.md,PREREG.md,draws.json,confirm.json}`,
`experiments/results/deep_dive/{proto_level3.json,items/PREREG_items_AMENDMENT_*.md}`, `DEEP_DIVE.md` §9 and §§15-17, the
submitted `model.py`, and the simulator code (`xpair_sim.py`, `w2_items/w2_sim.py`). Baseline build: 0 undefined
references, 0 overfull boxes, 0 `??`; `make_audit.py` 213 claims, 0 failing.

Page numbers refer to the PDF before the fixes below.

## Summary of the submission

A random-effects logistic predictor that uses the evaluator's global label list (cross-pair model, round 4), a
budget-0 level recalibrated after the organisers' formative feedback (base 1010717), and an online item layer (W2:
`item_features` category effects plus a character-4-gram TF-IDF kernel on residuals). The item layer passed a
pre-registered, benchmark-held-out simulator gate (INV -0.0035 [-0.0051, -0.0022]). Real formative draws: base 7
draws (mean 0.1904), W2 4 draws (mean 0.1705); the arm gap is mostly draw composition (already -0.0208 at budget 0).
A pre-written switch rule selected the W2 ZIP. Three alternatives (LLM item rating, semantic kernel, exact Bayes
pair-rate posterior) gave nothing.

Overall: an unusually careful report (generated numbers, a claim audit, pre-registered gates, negative results,
draws read as composition). The main problems are a few places where the wording runs ahead of the evidence, a
setup that is only explained in the appendix, and a selection paragraph that does not say why this particular
Codabench ID was chosen.

## Must-fix

**M1. The pair-matched comparison is sold as agreement with the simulator** (p. 1 abstract; p. 2 Contributions (2);
p. 5 §5.2 "Pair-matched comparison"; p. 6 §7 Conclusion). The text says the matched comparison "recovers an effect
of the simulated size", is "consistent with the simulator", and that "the real draws agree once pairs are matched".
The evidence does not support "agree":
- it is a post-hoc analysis (not in `SWITCH_RULE.md` or `PREREG.md`), on 11 pairs, t = -1.50, not significant;
- its se treats the 11 pairs as independent, although pairs of one run share a label list;
- its budget profile contradicts the simulator. Of the matched ALC difference (-0.0024), about 95% comes from
  B1 (-0.0114, se 0.0028; 0.2 x -0.0114 = -0.0023). The simulator expects almost nothing at B1 (SPARSE -0.0003;
  at most -0.0014 in any scenario). At B3-B15, where the simulator puts the gain, the matched differences are within
  one se of zero (+0.0013, +0.0009, -0.0012).

The honest reading: the matched ALC difference has the simulated size, but not the simulated shape, so the draws
neither confirm nor contradict the simulator. Fix the abstract, the contributions, §5.2 and the conclusion.

**M2. "Each chosen by pre-registered ... simulation" is not true for the level** (p. 2, Contributions (1)). The
cross-pair model (round-4 `PREREG.md`) and the item layer (`w2_items/PREREG.md`) were pre-registered. The level
(1010717: subject terms off, intercept -0.5) was chosen on 2026-10-26 after the formative feedback, by comparing
three uploaded variants and five simulated intercepts (`DEEP_DIVE.md` §9). No pre-registration exists for it. §4
also never says why 1010717 was kept rather than 1010715 or 1010735. Per §9 the reason is: it gains more than
1010715 on the INV/MID/HARD proxies, and 1010735 pays only in HARD-like pools while costing more on A, Y25 and INV.
State the reason, and say that this step was not pre-registered.

**M3. A reader cannot tell why the ID 1019682 was selected** (p. 5, §5.2 "Selection"). Four counted W2 runs are
uploads of the same ZIP (1019682, 1019683, 1019685, 1019903). The rule says "switch to the first W2 run that
finished", and 1019682 is that run (its scoring log was retrieved at 14:00 PDT, the others at 14:10 and later). It is
also the arm's *worst* draw (ALC 0.1881 against the arm mean 0.1705), so the choice of ID cannot be a best-of-N pick.
The report says neither thing. Without them, an E&D reviewer will suspect cherry-picking.

**M4. The experimental setup of every main-text result is only in the appendices** (pp. 3-5, §§4-5). The main text
uses A, INV, MID, HARD, Y25, INV25, SHIFT, SPARSE, H7, H7-INV and AS, "units" and "95% cluster-bootstrap CI over
benchmarks", but never says:
- what a simulated test set is, or what the unit of analysis is;
- that comparisons are paired on identical labels;
- that the CI is a percentile bootstrap over benchmark clusters;
- what the populations are. "SHIFT (A, prior +1)" in Table 2 is cryptic.

The competition requires the report to "clearly describe ... implementation and experimental setup". Two further
facts are missing from the main text:
- the simulations use the evaluator's random acquisition, while the submitted `labeling.py` uses the fresh-first
  rule;
- the scenarios share benchmarks, so they are overlapping views, not independent replications.

Add a short "Simulator and statistics" paragraph to the main text.

**M5. "The pair-rate part is at its information limit"** (p. 4 §4 "Why an item layer" (a); p. 6 §7 Conclusion). The
evidence is that one exploratory exact-Bayes grid posterior (one development seed, priors from the training folds,
evaluated in the simulator's own H7 structure) does not beat the base, and that a 4-scenario grid of prior sds gains
at most 0.0002. That shows the base is close to that posterior *in simulation*. It does not establish an
information limit. Reword both places, and label W1 as exploratory there.

**M6. The code link is dead** (p. 6 §8, `\codeurl`). `https://github.com/neelsirivara/neurips2026-paiec` returns 404,
because the repository is private. The competition requires links to the released code. This is an author action
before upload: make the repository public and push the tree. It cannot be fixed in the text.

## Should-fix

**S1. Disclosure accuracy: "no item was sent to any API"** (p. 7 §9). Claude Code runs on Anthropic's API and read
the project files, which include training-release items, while it worked. The intended (and true) statement is that
no item was *rated* through any API: the Claude, Gemini and Groq rating routes were never run. The LLM-assistant
paragraph should also use the competition's wording ("LLM-based coding assistant"). It should name Claude Code with
Anthropic's Claude models as the coding and research assistant, say what it did, and say what it could access. The
abstract's "an AI coding assistant (Claude)" should point to §9.

**S2. Disclosure understates the local testbed** (p. 7 §9 "Evaluation data ... only to label the local models"). The
testbeds informed the constants marked † in Appendix E (alpha_0, the LOO multiplier), which are still in the
submitted file's fallback path.

**S3. Pre-registration wording and amendments** (pp. 4-6, §5.1, §5.3). "Pre-registered" here means a dated file in
our own repository whose SHA-256 is recorded in the result files. It was not deposited with a third party, and the
git snapshots do not separate the PREREG files from the results (both are in the same snapshot commit). Say this
once. §5.3 says "three amendments before any rating was analysed", but Amendments 4-6 were written *after* the
Qwen3-VL gate had failed (11:13, 11:24 and 12:53 PDT on Oct 29, against the gate at 09:55). They registered further
raters: Gemini and Groq API routes that were never run, and Qwen3-14B. A reader should see that sequence (forking
paths), even though none of them produced a result.

**S4. Retry footnote** (p. 5, footnote 1). The retry note was written *after* 1019686 failed, when its complete log
was available. Say so, and add the fact that makes this harmless: its log-only ALC (0.1581) is below the W2 mean, so
not counting it can only make W2 look worse. Also say that the contingency clause was added before any W2 draw had
finished (`SWITCH_RULE.md`). Condition 2's "two standard errors" should say "of the difference of arm means,
computed with the base arm's sd" (the rule's own formula).

**S5. Multiplicity and the bootstrap** (p. 4 Tables 2-3).
- Table 2 marks "CI excludes 0" in 21 cells (7 populations x 3 variants), with no adjustment. Say the CIs are
  unadjusted.
- Table 3's five scenarios overlap: INV and SPARSE use the inventory benchmarks, and MID and HARD are subsets of A.
  Say they are not independent replications.
- The percentile cluster bootstrap with 20 clusters (INV, SPARSE) can under-cover. Mention it in the limitations.

**S6. Leaderboard context** (pp. 5-6). "Base arm at place 46, W2 arm at 28" invites the reader to credit 18 places
to the layer, which §5.2 has just attributed to draw composition. Reword.

**S7. Notation and definitions in §3** (p. 3).
- delta is both the prior intercept and the kernel shift of Eq. 3; rename the kernel shift.
- "Stage-1 residuals" is undefined. Stage 1 is the cross-pair fit of Eq. 2 with the category effects, before the
  exact-item effect.
- The share cap (`W2_FEAT_SHARE` = 1, which in effect means no cap) appears in the build command but not in the
  method.

**S8. Self-description** (p. 2 Contributions). "An honest reading" is self-praise; describe what the analysis does
instead. "Audited report" should say what the audit checks. It checks that the text's numbers and directional claims
match the result files. It does not check that the analysis code is correct.

**S9. Missing limitations** (p. 6 §6).
- All formative runs drew the same 7 benchmark IDs, so the level recalibration rests on those benchmarks.
- The simulator uses random acquisition, not the submitted fresh-first rule.
- The pre-registrations are self-hosted.
- The pair-matched analysis is post hoc and its se ignores within-run correlation.
- Cluster-bootstrap CIs over 20 benchmarks.
- The scenarios overlap.

**S10. Reproducibility gaps** (p. 7 §8).
- No Python version is given (3.14.5 is recorded in `round4/xpair_runtime.json`).
- The INV populations need the organisers' benchmark inventory CSV, which is outside the repository (path set in
  `dd_common.py`); the text does not say so.
- The audit's log-based checks (draw aggregates, retry timing) need our private Codabench logs, so a third party
  running `build_paper.sh` will see them fail. The paper says the logs are private but not this consequence.
- Per-unit `*.npz` files are git-ignored. CIs can be recomputed only by rerunning the simulator.
- Absolute paths in `dd_common.py` and `w2_common.py` (already disclosed).

**S11. Table 4 precision** (p. 5). The table prints "se 0.002" and "sd 0.019", but the text says "se 0.0016" and "sd
0.0187". Use the same precision.

**S12. Wording** (p. 3 §4). "Lowering the level by -0.5" is a double negative. Use "setting delta = -0.5".

## Nice-to-have

- **N1.** Figure 1: the thin per-draw curves are not in the legend, and red and blue are the main cue (the markers
  differ only slightly). Add legend entries for single draws.
- **N2.** Tables 1-2 name the variants only by Codabench ID; a short descriptor (delta) in Table 2's header would
  help.
- **N3.** "-0.0000" and "+0.0000" are printed for zero differences (§5.3 W2-sem, Table 4).
- **N4.** Appendix D points to itself ("Configurations B-E ... (Appendix D)", "contribute about equally (Appendix
  D)"). The frontier-subject discussion appears twice (Appendices F and J). The appendix (20 pages) is dominated by
  rounds 1-3 material; it is relevant as the fallback and budget-0 prior, but a reader needs a map.
- **N5.** The abstract is long (about 250 words) and uses the internal name W2.
- **N6.** `draws.json` has a B0-anchored arm difference (+0.0042, +0.0032, -0.0043, +0.0005, +0.0024 at
  B1-B31). It supports the composition reading and could go into Appendix A.
- **N7.** The repository `README.md` is stale: it shows the round 1-3 headline, `submission/paiec_submission.zip`
  as the artifact, and the old deadline. It is owned by the other agent.

## Scores before / after the fixes (1-5)

| Criterion | Before | After (expected) | Note |
|---|---|---|---|
| Soundness | 3 | 4 | M1, M2 and M5 overclaim; the underlying analysis is careful |
| Clarity | 3 | 4 | Setup and scenario names only in appendices (M4); notation clash (S7) |
| Reproducibility | 3 | 4 (5 once the repository is public) | Dead link (M6); inventory CSV; Python version |
| Experimental analysis | 4 | 4 | Pre-registered gate, pair matching, negative results; post-hoc status unstated |
| Disclosure | 4 | 5 | S1 and S2 inaccuracies; amendment sequence (S3) |

## What was changed (2026-10-30)

Every number below comes from macros. New macros are generated by `make_macros.py`; new claims are checked by
`make_audit.py`.

| Item | Change | Where |
|---|---|---|
| M1 | Abstract, Contributions (2), §5.2 ("Pair-matched comparison (post hoc)") and §7 now say: post hoc; the simulated size but not significant; about 94% from B1 (simulated B1 effect at most 0.0014 in magnitude); B3-B15 within one se of zero; the se ignores within-run correlation; "neither confirm nor contradict". Table 4 labels the row "post hoc", its ALC cell is no longer bold, and sd/se use 4 decimals (S11). New macros: `PmBOneSharePct`, `WcDBOneMaxAbs`. | `main.tex` abstract, §1, §5.2, §7; `make_macros.py` (arm table) |
| M2 | Contributions (1) separates the pre-registered parts (cross-pair model, item layer) from the level, which is now described as "not pre-registered". §4 states why 1010717 was kept over 1010715 and 1010735, and that the formative draws could not separate them. Table 2's caption names the variants (delta values). | §1, §4, Table 2 caption |
| M3 | `RuleSelectSentence` now says "the first W2 run that finished, as the rule specifies". §5.2 adds that this run's own draw (0.1881) is the arm's highest (worst) ALC, so the ID is not a best-of-draws pick. New macros: `SubDrawAlc`, `SubDrawRank`. | §5.2, `make_macros.py` |
| M4 | New paragraph "Simulator and statistics" (§2). It covers test sets, the unit of analysis, paired comparisons, the percentile cluster bootstrap with `\WcNBoot` resamples, random acquisition (and the size of the fresh-first effect), definitions of every population name, the fact that the populations overlap, and what "pre-registration" means here. Table 3's caption says the scenarios are not independent. | §2, Table 3 caption |
| M5 | "At its information limit" is replaced by "in simulation, the pair-rate part leaves little to gain". W1 is labelled exploratory (one development seed). The conclusion is reworded the same way. | §4, §7 |
| M6 | Not fixable in the text: the repository is private (404). **Author action**. | — |
| S1 | Disclosure: "no item was rated through any API"; the paragraph is renamed "LLM-based coding assistant" and now says Claude Code with Anthropic's Claude models was the coding and research assistant, what it did, and that it read training-release items and our Codabench logs. The abstract points to §9. | §9, abstract |
| S2 | Disclosure: the local testbeds informed the constants marked † (Appendix E), which remain in the fallback path. | §9 |
| S3 | §2 defines pre-registration as self-hosted. §5.3 says that further amendments (Gemini, Groq, Qwen3-14B) came after the Qwen3-VL gate had failed. The limitations note that the W2 pre-registration and its results entered git in the same snapshot. | §2, §5.3, §6 |
| S4 | Retry footnote: the note was written after the failure with the log at hand, and not counting the run cannot favour the switch (log-only 0.1581 < W2 mean 0.1705). The selection paragraph adds that the contingency clause predates every W2 result, and spells out "two standard errors". | `InfraFootnote` in `make_macros.py`; §5.2 |
| S5 | Table 2 caption: the stars are "not adjusted for the number of cells". §6 covers the bootstrap with few clusters and the overlap. | Table 2, §6 |
| S6 | Leaderboard places are reworded: "the difference between them is mostly the draw composition above". | §5.2 |
| S7 | The kernel shift is renamed eta_t. Stage-1 residuals are defined. The share cap is explained ("does not bind"). | §3 |
| S8 | "An honest reading" is replaced; Contribution (4) says what the audit does; §6 says what it does not check. | §1, §6 |
| S9 | Limitations: post-hoc status, random acquisition, the same 7 benchmark IDs, the statistics, self-hosted pre-registration, and the development process. | §6 |
| S10 | Reproduction: Python 3.14.5 (macro `PyVersion`, from `round4/xpair_runtime.json`); third-party packages; the inventory CSV and its path in `dd_common.py`; `*.npz` not in git; what fails without the private logs. Verified on a copy without `_downloads/`: every macro is identical and the audit fails only its 2 log-based claims. | §8 |
| S12 | "Lowering the level by -0.5" is replaced by "setting delta = -0.5". | §4 |
| N1 | Figure 1: base curves are dashed with triangles and W2 curves solid with squares. The legend gains "one base draw" and "one W2 draw", and the constant line is dotted. | `make_macros.py` (`results_fig_draws_all.tex`), Figure 1 caption |
| N4 | The two self-references in Appendix D now point to the "Configurations" paragraph and to Table 13. | Appendix D |

New audit claims, all holding:
- the selected run is the first counted W2 run that finished (earliest saved log);
- its draw is the arm's worst;
- the rule's contingency predates every W2 result;
- the infrastructure-failed run cannot favour the switch;
- the B1 share of the matched difference is at least 80% and B3-B15 are within one se;
- the simulators use random acquisition;
- the share cap does not bind;
- the level-choice comparisons hold;
- every draw used the same 7 benchmark IDs;
- `\SubDrawAlc` is added to the single-draw check.

Not changed:
- **M6.** Repository visibility.
- **N2.** No short names in the table header (the caption now gives the delta values).
- **N3.** "-0.0000" is left as printed: it is honest, and a generator-wide formatting change would touch hundreds of cells.
- **N4.** The duplicated frontier paragraphs and the appendix map are left as they are.
- **N5.** The abstract length.
- **N6.** The B0-anchored difference is not added.
- **N7.** `README.md` is owned by the other agent.
