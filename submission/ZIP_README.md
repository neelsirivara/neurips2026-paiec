# PAIEC submission: hierarchical shrinkage with text-aware test-time adaptation

`model.py` implements `predict(input, labeled=None) -> float`, the probability that the
subject (input[0]) answers the item (input[1]) correctly.  It uses only the Python standard
library, loads no data or model files, makes no network calls, and never raises.

Components (all constants are hand-set priors documented in the technical report; nothing is
fitted on competition data):
1. budget-0 prior from visible subject attributes (release date, reasoning effort, parameter
   count in the name, harness) and item cues (tier/points in item_features, item length);
2. beta-binomial shrinkage of the subject-benchmark success rate towards that prior
   (prior strength 2 pseudo-counts), using the acquired labels of the same pair;
3. a within-pair ridge-logistic adjustment on three cheap item features (log length, digit
   density, sentence count), from 7 labels on;
4. an item-similarity correction: TF-IDF (unigram+bigram) cosine similarity between the target
   item and the acquired items, kernel-weighted residual of the acquired labels;
5. leave-one-out gating: from 5 labels on, the kernel scale (including "off") and the use of
   the ridge component are selected by leave-one-out Brier score on the acquired labels, so
   adaptation is applied only when the pair's own labels show that it helps;
6. clipping to [0.01, 0.99].

`labeling.py` implements the streaming `acquisition_function(input, prediction=None,
labeled=None, context=None) -> bool`.  It accepts candidates in stream order (a uniformly
random sample, as the default policy) but skips near-duplicates of already acquired items
(TF-IDF cosine > 0.6) while enough candidates remain to spend the budget.  It never raises.

Deterministic; about 1-3 ms per predict call (hundreds of calls per second on one CPU core).
