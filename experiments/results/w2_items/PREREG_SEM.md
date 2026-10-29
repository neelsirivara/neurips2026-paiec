# W2-sem pre-registration: semantic text kernel on top of / instead of W2's char kernel

Written 2026-10-29 13:52 PDT. At this point:
- the semantic variants have not been run on any fold in the simulator;
- folds 2–4 have not been touched by W2-sem;
- the tuning grid below has not been run.

Code sha256 prefixes are listed at the end.

## Done before this file

**Embeddings** (`w2_embed.py`, merged by `merge_emb.py`):
- Model: `sentence-transformers/all-MiniLM-L6-v2`, mean pooling, L2-normalised, stored as float16.
- Coverage: all 95,773 usable items, on the laptop CPU (5 threads, 35 min). Output:
  `round4_cache/w2_emb_minilm_all.npz`.
- Text: `w2_feat.embed_text`, i.e. lower-cased, whitespace-normalised, the text if ≤ 1,000 characters, else head 600 +
  tail 400 characters, at most 256 tokens.
- A bge-small-en-v1.5 diagnostic run was stopped after 25%. It needs about twice MiniLM's compute both offline and at
  scoring time (12 layers vs 6), so it does not fit the cutoff. **MiniLM is the model for this round.**

**Infinite-label diagnostic** (`diag_sem.py`, `diag_sem.json`; INV + AS items, within-benchmark R² of the empirical
item effect):

| Signal | R², bench mean | R², median |
|---|---|---|
| categories | 0.118 | |
| char kernel (q 2) | 0.042 | |
| MiniLM, q 8, top 20 | 0.057 | |
| MiniLM, q 8 / 16 | 0.040 | |
| combined: categories + char | 0.184 | 0.111 |
| combined: categories + char + MiniLM | 0.210 | 0.121 |
| combined: categories + MiniLM | 0.167 | |

- The combinations are in-sample least squares over 2–3 coefficients, so they are slightly optimistic.
- The semantic signal is about as strong as the char kernel on its own and adds about +0.026 R² on top of
  categories + char.

**Implementation** (`w2s_block.py.txt`, built by `build_w2s.py`):
- The model is loaded only when an embedding is missing from the disk cache.
- The disk cache is shared by workers and checkpoints.
- Fallbacks: W2 char kernel, then m050.

Functional test of a dev build on the laptop CPU (`w2s_functional.py`):
- The deployed embedding equals the training code to 6e-5 (float16 rounding).
- A second process reproduces the predictions exactly from the cache.
- With the model disabled, or a missing model id, the predictions equal W2 exactly.

## Variants (grid; `run_w2s.grid()`, 504 settings)

- **Fixed:** τ_f ∈ {0.35, 0.5, 0.7} (category effects, share 1).
- **Kernel strength:** τ_k ∈ {0.7, 1, 1.5, 2}.
- **Neighbour truncation:** topk ∈ {all, 10, 30}.
- **Per-budget shrinkage:** m_low ∈ {1, 0.6}. m_low multiplies τ_k when the list's budget (median labels per pair) is
  ≤ 3.
- **Kernel K** (7 options):
  - char: cos_c² (the W2 kernel);
  - sem: cos_s^q, q ∈ {8, 16, 32};
  - both: 0.5 cos_c² + 0.5 cos_s^q, q ∈ {8, 16, 32}.
- **Same-item exclusion and δ:** as in W2.
- **Semantic candidates:** the 288 settings with kernel sem or both. The 216 char-only settings are reported for
  reference.

## Simulator, scenarios, split

- As in `PREREG.md`: `xpair_sim.py` unchanged, config A, R = 300.
- **Tune:** folds 0–1, scenarios A, INV, MID, HARD.
- **Confirm:** folds 2–4, scenarios INV, A, SPARSE, MID, HARD.
- **Same test sets as W2's confirmation.** Folds 2–4 were used once before, to confirm W2. The new comparison
  (selected − W2) is a new test of the semantic increment on the same benchmarks. W2 is recomputed in the same run, so
  the comparison is paired per unit.
- **Evaluator:** `w2s_sim.eval_grid`. It reproduces `w2_sim.eval_w2` for the W2 setting to 3.7e-9 on INV-tune.

## Selection (`run_w2s.py select`)

- **Criterion:** the mean, over the four tune scenarios, of the mean ALC difference vs m050.
- **Selected variant:** the semantic candidate with the lowest criterion.
- **Also reported:** the best char-only setting and W2's own criterion.

## Gate (`run_w2s.py gate`, folds 2–4)

Same as W2, plus that it must beat W2. All five must hold:
- (a) **INV**, selected − m050: the CI is entirely below 0;
- (b) **A**, selected − m050: the CI lower bound ≤ 0;
- (c) **SPARSE**, selected − m050: the CI lower bound ≤ 0;
- (d) **INV**, selected − W2: the CI is entirely below 0 (beats the current W2 candidate);
- (e) **A**, selected − W2: the CI lower bound ≤ 0 (not significantly worse than W2).

The statistic is a 95% percentile cluster bootstrap over benchmarks, 2,000 resamples, seed 0.

## Shipping (only if the gate passes)

1. Build `submission/candidates/w2s_<label>.zip` (`build_w2s.py --checks`, with `models.txt` =
   `sentence-transformers/all-MiniLM-L6-v2`).
2. Run the organisers' validator, `smoke_test.py` and `official_loop_check.py`, all with exit 0.
3. Run `w2s_equivalence.py`: 24 INV-tune test sets, max |diff| < 1e-3.
4. Run `w2s_functional.py` with the real model: embedding identity, cache reuse, and both fallbacks equal to W2.
5. **Coordinator's rule:** it ships only if at least one real Codabench run of it completes cleanly.

If the gate fails, nothing is built for upload and W2 stays the candidate.

## Code sha256 prefixes at registration

| File | sha256 |
|---|---|
| w2s_sim.py | f2d946b8 |
| run_w2s.py | 84994149 |
| w2s_block.py.txt | 4b7c3793 |
| build_w2s.py | 8536fa3b |
| w2s_equivalence.py | f6cecbb8 |
| w2s_functional.py | d2f835e8 |
| w2_embed.py | 5ce371ad |
| merge_emb.py | 4ededeab |
| diag_sem.py | 4a193a2b |
| w2_feat.py | dea9de46 |
| w2_sim.py | 3aa67873 (unchanged) |
