# Round 3 pre-registration: item-text difficulty prior (written 2026-10-23, before any fitting)

Written before any text model was fitted or any text-feature/outcome relationship was looked at.
What had been inspected at this point: the item-table schemas and one example item per benchmark
(content formats: plain text, JSON with `prompt`/`question`/`messages`/`multimedia_elements`, ...),
and the iteration-2 results already in `prior_fit.json`. The SHA-256 of this file is recorded in
every round-3 result JSON.

## Hypothesis

The text of an item (`item_content`, `item_features`) predicts how hard the item is for a typical
subject *across benchmarks* (long multi-step math/code/agentic items vs short factual multiple
choice), so a text model trained on other benchmarks moves the budget-0 and low-budget predictions
for an unseen benchmark toward its true success rate.

## Primary design (interaction-free)

**Text normalisation** (`model.text_prior_text`, stdlib): if `item_content` parses as JSON, walk it
and emit `key: value` lines (keys and string/number leaves in document order); otherwise use the raw
string. Word features use the first 2,000 characters, character 3-grams the first 500.

**Features** (`model.text_prior_features`, stdlib, used verbatim by the fitting code so that the
hashing is identical by construction): signed feature hashing into 2^18 buckets with
`zlib.crc32(namespace + token)` (index = low 18 bits, sign = bit 31) for four families -- word
unigrams (`[a-z]+`, digit runs mapped to a length-shape token), word bigrams, character 3-grams of the
lower-cased whitespace-collapsed text, and `item_features` tokens; each family sublinear tf
(1 + log c) and l2-normalised separately. Plus 16 dense statistics (log length, log word count,
digit/upper-case/non-ASCII/CJK fractions, LaTeX/code/choice-pattern/newline/question-mark counts,
JSON flag, image-element count, item_features presence and length), standardised with training means
and standard deviations.

**Target** (per item of a pair in the iteration-2 frame: pairs with >= 20 items, 144 benchmarks):
`t_i = logit((n_i r_i + 0.5) / (n_i + 1)) - mean_{pairs evaluating i} z_subject_hand - z_item_hand(i)`,
clipped to [-5, 5], where `r_i` is the mean response over the `n_i` pairs that evaluated item i, and the
hand-set subject/item logits are those of the shipped `model.py`. i.e. the text model learns the
residual difficulty the hand-set prior does not explain.

**Model**: weighted ridge regression (scikit-learn), each benchmark's items weighted to a total of 1
(benchmarks count equally), at most 1,000 target items per benchmark (random, seed 0). Ridge penalty
from {0.03, 0.1, 0.3, 1, 3, 10} by 4-fold benchmark-grouped inner CV (weighted MSE of `t`) inside
each outer training set.

**Combination with the shipped rule**: text logit `T(x) = W * s(x) + B` added to the hand-set prior
logit before the cap, `z0 = clip(z_subject + z_item_hand + T, +-cap)`; the pair-level beta-binomial
shrinkage with alpha0 = 4 is kept (prior rate from the acquired items' mean prior logit, as in
`model.py`); the target item's offset from that mean uses `lambda * (T(target) - mean T(acquired))`
plus the hand-set item offset. `W, B` by pair-weighted logistic likelihood of the training pairs'
rows on inner out-of-fold scores; `cap` from {1.5, 2.5, 3.5} by training-fold budget-0 Brier;
`lambda` from {0, 0.5, 1} by training-fold closed-form ALC. Everything is selected inside the
training benchmarks of each outer fold; the test benchmarks are touched once.

**Secondary (only considered if the primary ships)**: one interaction, `gamma * T(x) * year(subject)`
(year = the hand-set release-year term, clipped to +-2), fitted jointly with W and B. It replaces the
primary only if it improves on the primary under the same CV with a CI excluding zero and is not
significantly worse on the real Ollama testbed.

## Evaluation

1. **Benchmark-held-out CV**: the 5 benchmark folds of iteration 2 (`prior_fit.json` `cv.folds`), the
   same frame (16,718 pairs; reproduction of the hand-set numbers B0 0.2388 / ALC 0.2051 is checked
   first), the same closed-form protocol (50/50 split, 31 random acquisitions, budgets 0..31, official
   ALC weights, pairs with >= 80 items for the ALC). Per pair: B0, ALC, Brier at budgets 0/1/3, and the
   pair-rate oracle. Paired differences text minus hand-set with 95% CIs from a **cluster bootstrap over
   benchmarks** (2,000 resamples, seed 0; primary) and the plain pair bootstrap (1,000; secondary).
   Reported: fraction of the hand-set-to-oracle gap closed at budgets 0, 1 and 3; per-fold results;
   references (intercept-only, held-out benchmark-mean oracle).
2. **Local testbeds** in the full harness (all predictor components, as in iteration 2): the 17-pair
   real Ollama testbed and the 160-pair synthetic IRT testbed, text prior vs hand-set, paired bootstrap
   over pairs. The BBH-mix testbed items may overlap the `fewshot_ttt_bbh` training benchmark; the real
   comparison is also reported without the BBH pairs (descriptive).

## Ship rule

Ship the text prior iff
- (a) the benchmark-held-out ALC difference (text minus hand-set, alpha0 = 4) has a 95% cluster-bootstrap
  CI entirely below 0, **and**
- (b) on the real Ollama testbed the full-model ALC difference (text minus hand-set) is not significantly
  positive (its 95% CI does not lie entirely above 0),
- and the engineering gates hold: `predict` >= 200 calls/s on one core at budget 31 with a new target item
  per call, weights <= 5 MB, official validator OK, smoke test OK, `predict` never raises (any failure of
  the text component falls back to the hand-set prior).
If an engineering gate forces a smaller configuration (fewer buckets / shorter text), the CV is re-run
with that configuration and the rule re-applied. Otherwise the shipped predictor is unchanged and the
result is reported as a negative result.

Fixed seeds throughout; every number in the report comes from the round-3 result JSONs via macros.
