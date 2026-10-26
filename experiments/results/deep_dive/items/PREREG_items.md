# Pre-registration: LLM-rated item difficulty as an offset in the cross-pair model (deep dive, 2026-10-26)

Written 2026-10-26 13:08 PDT, before any item has been rated (no rating file exists). Its SHA-256 is stored in
every result JSON of this study.

## Question
Does a local open-weight LLM's judgement of an item ("would a 2025–26 frontier AI system solve this?") improve
Brier-ALC when added as an item offset to the cross-pair model, on benchmarks never used to fit it?

## Signal (fixed now)
- Model: `Qwen/Qwen3.5-4B`, snapshot `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (already in the Mac's hf_cache).
  Run in bf16 on MPS, batch 1, one forward pass per item, no sampling and no generation. Chat template with
  thinking disabled (`enable_thinking=False`).
- Inputs:
  - `item_content`: the items-table `content` field, whitespace-normalised, first 1,500 characters;
  - `item_features`: first 300 characters.
- Prompt (user turn, verbatim, with `{content}` and `{features}` substituted):

  > You are estimating how hard a benchmark task is for state-of-the-art AI systems released in 2025-2026 (for
  > example GPT-5, Claude Opus 4, Gemini 2.5 Pro), possibly running inside an agent harness. Read the task and
  > answer with one word: Yes if such a system would most likely complete it correctly, No otherwise.
  >
  > Task:
  > <<<
  > {content}
  > >>>
  >
  > Item attributes: {features}
  >
  > Would a state-of-the-art 2025-2026 AI system most likely complete this task correctly? Answer Yes or No.

- **Primary signal s1** = log P("Yes") − log P("No") at the first assistant token. "Yes"/"No" are the first tokens of
  " Yes"/"Yes" and " No"/"No", taking the log-sum-exp of the variants that exist in the vocabulary.
- **Exploratory s3** (free from the same pass) = mean token log-probability of the content span.
- Cache key: SHA-1 of the exact prompt string. Output: one JSON line per item.

## Items
All distinct frame items needed by two scenarios, both built with `xpair_sim` config A (R = 300 per fold, the
round-2 benchmark folds, random acquisition):
- **INV**: test sets restricted to the 33 measurement-db benchmarks that are rows of the official benchmark inventory.
  This is the primary proxy for the hidden pool.
- **AS**: release-like check. Test sets restricted to 25 non-inventory benchmarks drawn with
  `numpy.random.RandomState(2026).choice` from the sorted list of non-inventory benchmarks that have eligible pairs.

If rating time runs out, INV is rated first. AS may then be cut to its first benchmarks in that seeded order; the
report will state any cut.

## Offset model (fixed now)
- Base predictor: candidate `xp_level_m050` (subject terms off, intercept −0.5, τ_mu 2; all other XP constants as
  shipped).
- Candidate predictor: base offset + w · z(s1) + b. z standardises s1 with the mean and sd of the training-fold items.
- w and b are fitted per outer fold by logistic regression on the training folds' benchmarks (INV ∪ AS items), using
  the base budget-0 logit as a fixed offset. All eligible pairs' rows are used, every benchmark is weighted equally,
  and there is no penalty.
- The fitted map is applied to the held-out fold, inside the XP model, at every budget.
- Items without a rating get offset 0.

## Gate (decides integration)
Ship the item signal only if BOTH hold:
- (a) on INV, ALC(candidate) − ALC(base) has a 95% cluster-bootstrap CI (2,000 resamples over benchmarks) entirely
  below 0;
- (b) on AS, the same difference is not significantly above 0 (CI lower bound ≤ 0).

Engineering gates, if (a) and (b) hold:
- never raises;
- full fallback to the base model on any load, OOM, timeout or API failure;
- a hard time budget well inside the 8 h run;
- the validator, smoke test and official-loop check pass.

## Reported regardless of the gate (exploratory)
- Within-benchmark correlation of s1 with the leave-pair-out empirical item effect.
- The fitted w and b per fold.
- s3 and s1 + s3 as alternative signals.
- Per-budget differences.

None of these can change the gate decision.
