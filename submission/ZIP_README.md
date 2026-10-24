# PAIEC submission: cross-pair random-effects prediction with per-pair shrinkage fallback

`model.py` implements `predict(input, labeled=None) -> float`, the probability that the
subject (input[0]) answers the item (input[1]) correctly.  It uses only the Python standard
library, loads no data or model files, makes no network calls, and never raises.

Cross-pair model (used whenever `labeled` is non-empty): the evaluator passes one global list
with the labels of all sampled subject-benchmark pairs.  Once per checkpoint (cached, thread-safe)
`model.py` fits a random-effects logistic model to that list: logit p = prior(subject, item) +
global + benchmark + subject + subject-benchmark effects, Gaussian priors with sds
0.5 / 1.5 / 0.5 / 0.6 logits (selected by benchmark-held-out
simulation of the official protocol on the public training release), plus an item effect (prior sd
2.25) when the target item was labelled for other subjects, with probit integration
of the posterior sd.  Subjects are identified by all their visible attributes.  One fit of a
62,000-label list takes about 2 s; later calls about 0.1-0.3 ms.

Per-pair fallback (budget 0, and on any failure of the cross-pair model; all constants are hand-set priors documented in the technical report; nothing is fitted on competition data):
1. budget-0 prior from visible subject attributes (release date, reasoning effort, parameter
   count in the name, harness, an intercept and a per-provider table) and item cues
   (tier/points in item_features, item length), capped at +-1.5 logits;
2. beta-binomial shrinkage of the subject-benchmark success rate towards that prior
   (prior strength 4 pseudo-counts), using the acquired labels of the same pair;
3. a within-pair ridge-logistic adjustment on three cheap item features (log length, digit
   density, sentence count), from 7 labels on;
4. an item-similarity correction: TF-IDF (unigram+bigram) cosine similarity between the target
   item and the acquired items, kernel-weighted residual of the acquired labels;
5. leave-one-out gating: from 5 labels on, the kernel scale (including "off") and the use of
   the ridge component are selected by leave-one-out Brier score on the acquired labels, so
   adaptation is applied only when the pair's own labels show that it helps (switch-on requires
   an improvement of 2 standard errors);
6. clipping to [0.01, 0.99].

`labeling.py` implements the streaming `acquisition_function(input, labeled=None,
context=None) -> bool` (no `prediction` parameter, so the evaluator does not call predict for
every candidate).  Policy "fresh": candidates are accepted in stream order (a uniformly
random sample, as the default policy), except that a candidate whose item is already labelled for
another subject in the global list is skipped while enough candidates remain to spend the budget,
so the labels cover more distinct items.  It never raises.

Deterministic.
