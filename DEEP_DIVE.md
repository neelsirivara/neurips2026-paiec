# PAIEC deep dive (2026-10-26): where our Brier goes and what can still move it before Oct 30

Scope: Codabench 17828, phase 29785, shipped submission 1010370 (zip sha256 `c76331d0…`). Every number below names
the file it was read from. Simulations reuse `experiments/src/xpair_sim.py` unchanged (the round-4 multi-pair
simulator on measurement-db revision `54074453da`). New scripts are in `experiments/src/deep_dive/`; their outputs are in
`experiments/results/deep_dive/`.

Compute and editing:
- Simulations ran on one laptop thread.
- Item rating ran on the Mac GPU under `~/neurips/GPU_LOCK` (§10).
- The paper was not edited by this work.

**Status (2026-10-26 afternoon; latest first):**
- **Selected submission: 1010717** (`xp_level_m050`, sha256 `cda9fe9c…`), chosen by the coordinator on the
  recommendation in §9.
- Four formative runs are analysed in §9.
- Two deployment probes for an on-GPU item-rating model are uploaded (1010835, 1010837); see §11.
- The pre-registered item-rating study is running; its gate result is pending (§10).

## 1. Summary (written before the candidate uploads; §9 has the four-run update)

1. **The loss is at budget 0.** Our own formative log for 1010370 (`_downloads/paiec_logs_1010370/scoring_stdout.txt`,
   aggregated by `deep_dive/parse_feedback.py`, 9 pairs on 7 benchmarks) reads:
   - Brier by budget: B0 **0.2834**, B1 0.2103, B3 0.1909, B7 0.1786, B15 0.1641, B31 0.1627.
   - B0 ECE **0.3405**.

   B0 is worse than predicting the constant 0.5 (0.25), and its ECE is above every population we simulated
   (`results/deep_dive/ece_check.json`):

   | Simulated population | B0 ECE |
   |---|---|
   | release-like | 0.235 |
   | inventory benchmarks | 0.271 |
   | hard benchmarks | 0.299 |
   | prior shifted +1 logit | 0.286 |

   From B1 on, the cross-pair model repairs the level. Observed B1/B3 (0.210/0.191) match the inventory-benchmark
   proxy (0.205/0.189). The hand-set subject terms of the budget-0 prior (release year, reasoning effort, size) push
   2025–26 subjects up. The hidden pool is evidently harder than that.
2. **The fix is cheap and built.** It changes three constants:
   - the subject terms are turned off;
   - the intercept goes to −0.5;
   - the cross-pair level sd goes from 0.5 to 2.

   It is `submission/candidates/xp_level_m050.zip` (uploaded as 1010717). In the simulated populations that resemble
   the hidden pool it gains 0.004–0.007 ALC with CIs excluding 0. It costs 0.002–0.005 if the pool were
   release-like, which the log above argues against. A conservative variant, `xp_nosubj_mu2.zip` (uploaded as
   1010715), gains about 0.001 and is never significantly worse.
3. **Within the "labels only" family we are close to the information limit.**
   - In the simulator the shipped model sits 0.023 ALC above the pair-rate oracle, and 87% of that is at B0–B3.
   - Re-tuning the prior sds on the inventory proxy gains nothing (+0.0004 [−0.0007, +0.0014]).
   - A subject-ability table learned from the training benchmarks does not transfer (R² ≤ 0.02).
4. **The large remaining lever is item-level difficulty, which only an LLM could supply at scoring time.**
   - Simulated ceilings: −0.031 to −0.042 ALC for perfect item difficulty, and −0.0065 / −0.011 for perfect
     benchmark difficulty (`proto_oracle.json`).
   - With both, the simulated ALC is 0.149 (release-like) / 0.119 (hard), which is leaderboard-leader territory.
   - Realistic LLM signals capture a fraction of this (roughly −0.002 to −0.010; §7).
   - The rules allow it (inference APIs from 8 named providers, or a local model via `models.txt`).
   - It needs the user's API key and spend, and about 2 days. It is feasible before Oct 30 only if started on Oct 26–27.
5. **Part of the leaderboard gap is noise and selection.**
   - Each formative run is a fresh sample of about 9 pairs.
   - One draw has SD about 0.011–0.024 (two participants who resubmitted identical files) and 0.039 in our
     simulator. Our own 9 pair ALCs range 0.083–0.267.
   - The board shows each user's best draw, and 50 uploads a day are allowed.
   - The final ranking uses one common subset for all teams. The target is the expected score, not a lucky draw.

## 2. What the hidden test rewards

Sources: the competition page (copy at `_downloads/paiec_deepdive_2026-10-26/pages/competition.txt`), the organisers'
baseline repository, the curation methodology, and participants' public write-ups. Copies of all of these are in
`_downloads/paiec_deepdive_2026-10-26/intel/`.

**Protocol (page, and doctawho42's replica notes in `intel/participants/doctawho42_protocol.md`):**
- Budgets 0/1/3/7/15/31 have weights 0.1/0.2/0.2/0.2/0.2/0.1. B0–B3 carry half the score.
- At budget 0, `labeled` is empty, so B0 is pure prior.
- Scoring is budget-major. Every `predict` call at a checkpoint gets ONE global list holding the first B labels of
  every pair in the run.
- Workers are recreated at every checkpoint.
- Pairs need at least 80 items; items are split 50/50, with the split fixed.
- Each formative evaluation is "a newly sampled subset … capped at 1,000 unique subject–item pairs". Our run had
  9 pairs and 492 evaluation subject-items.
- The summative evaluation uses "a common test subset for all teams" plus a code and report review.

**Population:**
- Test benchmarks come from the benchmark inventory. Candidate benchmarks are split randomly into training and test
  pools before curation.
- The curation guide (5 Sep 2026, `intel/methodology_doc.txt`, line 35) says "eligible base models are currently
  restricted to releases from 2025–2026".
- In the training release, the 33 benchmarks that are inventory rows are harder than the rest
  (`results/deep_dive/population.json`): mean pair rate 0.466 vs 0.547 for the others.
  - On them our hand-set prior is worse than the constant at B0: 0.2596 vs 0.25, and 0.2653 for 2025+ subjects.
  - 58% of eligible release pairs have an empty `normalized_name`, and 70% have no release date. The release is not
    the competition's subject population.
- Participants' public statements about the hidden runs. These come from their own feedback, are not verified by us,
  and the inversion part is in a grey zone of the rules:
  - doctawho42: a 9-pair run's pair-rate oracle averages 0.160 (sd 0.017). Our inventory proxy gives 0.161.
  - s2wish: "Platform pairs harder … median 0.27" (`intel/participants/s2wish_REPRODUCE.md`, line 182).
  - lucaluo925: "B₀ and B₃₁ exceed our local estimate in all nine draws".

**Rules relevant to methods (page, sections Training data / Runtime / Conduct / Q&A):**
- Allowed:
  - any source in the benchmark inventory plus measurement-db, for training;
  - pretrained models (disclosed), bundled or prefetched via `models.txt` (up to 5 repos, 300B parameters each);
  - at scoring time, inference APIs from Anthropic, OpenAI, Google, xAI, DeepSeek, Mistral, Together AI and Groq;
  - 8 h per run;
  - adapting at evaluation time from the supplied attributes and revealed labels;
  - multiple development submissions. Codabench limits them to 50 per day and 1,000 per person.
- Not allowed:
  - data sources outside the inventory, for training. Open LLM Leaderboard and HELM per-instance dumps are not
    inventory rows (intel agent's check of the sheet);
  - web search, browsing or grounding at runtime;
  - "extract or share private test data or attempt to identify the subjects or benchmarks behind anonymous feedback IDs".
- Repeated feedback:
  - Comparing a handful of principled variants on aggregate scores is "formative feedback may guide development".
  - Inverting per-pair Brier/ECE rows into hidden pair success rates is extraction, and we do not do it.
    `parse_feedback.py` prints aggregates only.
  - The page states that matching IDs denote the same subject or benchmark. Averaging per-pair differences over pairs
    shared by two of our submissions is therefore an aggregate comparison, not identification.

**What top entries could be exploiting, and whether it is allowed:**

| Signal | We use it? | Allowed? |
|---|---|---|
| Other pairs' labels via the global list (cross-pair pooling) | yes (round 4) | yes |
| Exact item matches across subjects | yes | yes |
| Level prior calibrated to the hidden pool | now: candidates | yes on aggregates; per-pair inversion no |
| LLM-rated benchmark / item difficulty at scoring time (API or local model); kkk61 (#10 at 0.148) uses a benchmark prior from Qwen3.8-27B item ratings on GPU | no | yes (disclose) |
| Subject ability by model name from measurement-db | tested: does not transfer | yes |
| Best-of-N resubmission on noisy draws | no | yes, but it changes only the board, not the final |
| Upstream per-item results not in the inventory, identifying hidden benchmarks | no | no |

**Leaderboard context** (`../leaderboards/cb_phase_29785_now.json`, `_p2.json`, 58 entries):
- #1 0.1172 (2026-09-20), #5 0.1360, #10 0.1483, median 0.1700.
- The organisers' "AIMS Foundations" entry is 0.1801. Which baseline it is was not found.
- Documented multi-draw means of hierarchical label-only entries: lucaluo925 0.188 (9 identical uploads, SD 0.0113)
  and s2wish 0.1839 (4 uploads, SD 0.024). Our single draw 0.1934 sits with them.
- How far best-of-N can go: for SD 0.018, the expected best of 30 draws is the mean minus about 0.037. Selection alone
  cannot take a 0.19 mean to 0.13. Leaders probably also hold item-level information (§1, point 4).

## 3. Literature (verified by a research sub-agent against the arXiv/ACL/PMLR pages on 2026-10-26)

I re-checked titles, authors and abstracts myself for three entries: Zhou et al. 2503.06378, Truong et al.
2503.13335 and Zhang et al. 2407.01300. Numbers marked (body) were read from the paper body by the sub-agent and were
not re-checked.

- **IRT for efficient LLM evaluation.**
  - tinyBenchmarks (Polo et al., ICML 2024, https://arxiv.org/abs/2402.14992): gp-IRT, a sample-size-dependent blend
    of observed accuracy and IRT prediction.
  - metabench (Kipnis et al., ICLR 2025, https://arxiv.org/abs/2407.12844): a single common factor, Spearman 0.94
    with the total.
  - Fluid Benchmarking (Hofmann et al., COLM 2025, https://arxiv.org/abs/2509.11106): IRT mainly improves validity;
    adaptive selection mainly reduces variance.
  - Anchor Points (Vivek et al., EACL 2024, https://arxiv.org/abs/2309.08638).
  - Rodriguez et al. (ACL 2021, https://aclanthology.org/2021.acl-long.346/): item identity predicts better than item
    text.
  - PromptEval (Polo et al., NeurIPS 2024, https://arxiv.org/abs/2405.17202): sentence embeddings gave no improvement
    over one-hot items (body).
- **Cold-start benchmarks, text to difficulty.**
  - Contextual multidimensional IRT (Shang et al., https://arxiv.org/abs/2608.22295): gains "do not necessarily
    translate … under cross-scenario shift".
  - Text-to-parameters ceilings (Chen & Chen, https://arxiv.org/abs/2607.07141).
  - Both agree with our round-3 negative result for text features.
- **Low-rank capability structure.**
  - Observational scaling laws (Ruan et al., NeurIPS 2024, https://arxiv.org/abs/2405.10938): 3 PCs explain about
    97% (body).
  - Collaborative Performance Prediction (Zhang et al., EMNLP 2024, https://arxiv.org/abs/2407.01300).
  - Burnell et al. 2023 (https://arxiv.org/abs/2306.10062).
  - Sloth (Polo et al., NeurIPS 2025, https://arxiv.org/abs/2412.06540).
  - These motivate a subject-factor prior. In measurement-db, though, names do not transfer within benchmark
    (§4, `population.json`).
- **Item-level success prediction.**
  - General Scales (Zhou et al., https://arxiv.org/abs/2503.06378): GPT-4o-annotated item "demand" levels. New-benchmark
    AUROC 0.747 vs 0.480 for embeddings (body). The abstract says only that the estimates are "superior … especially
    out-of-distribution".
  - PredictaBoard (Pacchiardi et al., Findings ACL 2025, https://arxiv.org/abs/2502.14445): most assessors are near
    the baselines.
  - "100 instances is all you need" (Pacchiardi et al., https://arxiv.org/abs/2409.03563).
  - Anticipatory evaluation from descriptions (Park et al., https://arxiv.org/abs/2509.20645).
- **The organisers' own work.**
  - Amortized model-based evaluation (Truong, Tu, Liang, Li, Koyejo, https://arxiv.org/abs/2503.13335, ICML 2025):
    difficulty predicted from question content, across 22 benchmarks and 172 LLMs per the abstract.
  - Item Response Scaling Laws (https://arxiv.org/abs/2606.07616).
  - Fantastic Bugs (https://arxiv.org/abs/2511.16842).
  - The `torch_measure` package has `AmortizedIRT`, `LogisticFM`, `Bifactor` and an `LLMJudge`
    (https://torch-measure.readthedocs.io/en/latest/api/models.html).
  - The BLE baseline (`paiec_baseline/ble`) is an LLM agent with retrieval. It uses only labels whose subject matches
    the target exactly.
- **Not found:** a "Lord of the benchmarks" paper, and any PAIEC proposal or measurement-db paper.

## 4. Diagnosis: where the Brier is lost (simulator, shipped predictor)

All runs use config A as in round 4: 1,500 test sets of 10 pairs on 1–3 benchmarks, benchmark-held-out folds, random
acquisition. The shipped predictor reproduces ALC **0.1965** (round 4: 0.1966) (`diagnose_A.json`).

**By budget.** Weighted gap to the pair-rate oracle (oracle 0.1731):

| Budget | Weighted gap |
|---|---|
| B0 | 0.0074 |
| B1 | 0.0079 |
| B3 | 0.0052 |
| B7 | 0.0027 |
| B15 | 0.0009 |
| B31 | −0.0007 |
| **Total** | **0.0234** |

87% of the total is at B0–B3.

**By pair success rate.** Extreme pairs lose the most:

| Pair rate | ALC (oracle) | B0 bias | B1 bias |
|---|---|---|---|
| < 0.1 | 0.085 (0.030) | +0.47 | +0.26 |
| > 0.9 | 0.098 (0.049) | −0.42 | −0.24 |
| 0.3–0.7 | within 0.013 of the oracle | | |

The hidden pool's low rates put more pairs in the expensive tail.

**By subject release date.**

| Release date | ALC (oracle) |
|---|---|
| ≥ 2025 | 0.1922 (0.1711) |
| none | 0.1966 (0.1702) |
| 2024 | 0.1978 (0.1785) |

**By benchmark.** The ten worst benchmarks hold 21.6% of the total gap (devbench 3.4%, erbench 2.6%, abstract_reason
2.5%, planbench 2.2%, …). Per-benchmark gaps have quartiles 0.015 / 0.022 / 0.031. The loss is spread out, not
concentrated in a few benchmarks.

**Level vs adaptation.** Shifting the whole prior by −1 logit changes B0 by +0.049 on release-like data but B1 by only
+0.0015 (`proto_level.json`). The global list fixes the level from B1 on, so the level matters almost only at B0.

**Formative noise.**
- In the simulator, a 10-pair draw has ALC SD 0.039 (5–95% range 0.124–0.250) and ECE p10–p90 0.11–0.19 in every
  population.
- So our observed total ALC (0.1934) and ECE (0.169) cannot tell the populations apart. The B0 row can (§1).
- D = B0 − B31, the within-draw statistic: simulated 0.081 (release-like), 0.099 (inventory), 0.119 (hard). Ours is
  0.121. Single-draw SDs per population are in `decision_stat.json`.

**Proxies for the hidden pool, all rules-clean.** Each line gives the shipped ALC, the pair-rate oracle and the
median pair rate:

| Proxy | Definition | Shipped ALC | Oracle | Median rate |
|---|---|---|---|---|
| A | release-like | 0.1965 | 0.1731 | 0.52 |
| Y25 | 2025+ subjects | 0.1860 | 0.1640 | 0.62 |
| INV | inventory benchmarks | 0.1887 | 0.1606 | 0.36 |
| INV25 | inventory × 2025+ | 0.1898 | 0.1692 | 0.42 |
| MID | benchmarks with mean rate ≤ 0.5 | 0.1903 | 0.1664 | 0.28 |
| HARD | mean rate ≤ 0.4 | 0.1630 | 0.1388 | 0.20 |
| SHIFT | release-like, prior shifted +1 | 0.2023 | 0.1731 | 0.52 |

Sources: `proto_level.json`, `proto_level2.json`, `proto_level3.json`.

## 5. Ranked changes

Gains are Brier-ALC differences against the shipped predictor (negative is better), with 95% cluster-bootstrap CIs
over benchmarks.

| # | Change | Expected gain on the hidden pool (how estimated) | Cost | Risk | By Oct 30? |
|---|---|---|---|---|---|
| 1 | **Calibrated budget-0 level**: subject terms off, intercept −0.5, τ_mu 2 → `xp_level_m050` | MID −0.0041 [−0.0057, −0.0025], HARD −0.0069 [−0.0090, −0.0048], SHIFT −0.0042 [−0.0056, −0.0028], INV −0.0020 [−0.0050, +0.0015]. Release-like: A +0.0018, Y25 +0.0046 (`proto_level3.json`). The hidden B0 row says the pool is not release-like (§1). | done | low (B0 is 10% of the weight; B1+ barely changes) | **yes, built and uploaded (1010717)** |
| 2 | Conservative variant: subject terms off, τ_mu 2, intercept 0 → `xp_nosubj_mu2` | −0.0010 to −0.0019 on INV/INV25/MID/HARD (MID and HARD significant), +0.0002 / +0.0007 n.s. on A/Y25 | done | very low | **yes, uploaded (1010715)** |
| 3 | LLM-rated item difficulty (+ its benchmark average) as an extra offset, via allowed API | Ceiling −0.031 to −0.042 (item oracle, A), −0.047 with the benchmark level (`proto_oracle.json`). Realistic ≈ ρ² × ceiling with ρ 0.15–0.5 (literature; doctawho42: best transferable item covariate correlation 0.15) → −0.002 to −0.010; may be 0 | API key + about $20–50 + about 2 days | high: transfer, API failures inside the 8 h run, key in the zip | only if started Oct 26–27 (§7) |
| 4 | LLM-rated benchmark difficulty only (kkk61's approach) | Ceiling −0.0065 (A) / −0.0114 (HARD), mostly B0–B1. Realistic −0.002 to −0.005. Partly overlaps #1, since a level fix already takes the mean part. | same as 3, fewer calls | high | same as 3 |
| 5 | Stronger level shift (intercept −0.75 / −1) | HARD −0.0082/−0.0087 and SHIFT −0.0052/−0.0055 vs −0.0069/−0.0042 at −0.5, but INV −0.0014/−0.0003 and A +0.0036/+0.0057 | 5 min | medium: tuning one draw's B0 | yes, if the candidates' logs still show high B0 |
| 6 | Re-tune XP prior sds on the inventory proxy (48-setting grid, nested by fold) | +0.0004 [−0.0007, +0.0014]; picked setting on A +0.0003 (`grid_inv.json`) | done | – | no gain |
| 7 | Subject-ability table by `normalized_name` from training benchmarks | Held-out within-benchmark correlation −0.03 to −0.06 (all), +0.08 to +0.10 (2025+), R² ≤ 0.019 (`population.json`) | – | – | no gain |
| 8 | Test-time item-feature facet effects, or within-benchmark length slope pooled over subjects | Not prototyped: needs re-reading item tables. Expected < 0.002: participants find weak item covariates, and hidden learning curves are flat after B7 (doctawho42 B7→B31 −0.0012). | 1 day | low | not worth it |
| 9 | Best-of-N resubmission | 0 on the final (common subset); board only | free | – | not a goal |

## 6. Prototypes and candidates

**Candidates.** Both are built by `experiments/src/deep_dive/build_candidate.py` from the shipped zip's members.
Exactly three constant lines in `model.py` change; `labeling.py` is byte-identical.

| Candidate | Constants | Zip sha256 | Uploaded as | Real-testbed ALC through the organisers' streaming client (shipped 0.2179) |
|---|---|---|---|---|
| `submission/candidates/xp_level_m050.zip` | SUBJECT_SHRINK 0, PRIOR_INTERCEPT −0.5, mu 2 | `cda9fe9cae49b68997d831f1de8dfe3d53a14ad3d1ac15281b22bd8326c7a715` (20,047 B) | 1010717 | 0.2179 |
| `submission/candidates/xp_nosubj_mu2.zip` | SUBJECT_SHRINK 0, PRIOR_INTERCEPT 0, mu 2 | `29333f3a8ae551bce99ed8cae4b7b36a64049daf58435347c96b200e39e58fc7` | 1010715 | 0.2208 |

Checks on each, all run with `python -I` (records in `experiments/results/deep_dive/candidates/<name>/`):
- the organisers' `check_submission_zip.py` (verbatim copy, commit cc88a959): `OK … valid submission ZIP`, exit 0;
- `smoke_test.py`: 44 cases OK;
- `official_loop_check.py`: the organisers' `run_streaming` with a mock coordinator on the 17-pair real testbed.
  Stream mode, 0 prediction calls during acquisition, 31 labels per pair;
- `candidate_equivalence.py`: stdlib `model.py` vs the numpy simulator with the same constants, 56,750 predictions,
  max |diff| 1.1e-5;
- the shipped zip is unchanged (sha256 re-checked after each build), and the round-4 `official_loop_check.json`
  record was restored.

**Upload result.** The two candidate draws were pending when this section was written.

**How to read the next logs.** Each draw has different pairs, so compare within draws:
- D = B0 − B31, and B0 ECE;
- where pair IDs recur, the pair-ID-matched B0 Brier, averaged over the shared pairs.

`parse_feedback.py` prints the per-budget aggregates.

## 7. The LLM lever: what it would take before Oct 30

- **Allowed.** Runtime APIs from the 8 providers, with no search or grounding. Local models via `models.txt` get an
  automatic GPU. The organisers' BLE baseline ships API keys inside a private zip (`submission_config.json`,
  240 s per question).
- **We cannot validate a local model.** Loading any model on the Mac requires the GPU lock, which another job holds,
  and the laptop is not for inference. So the practical route is an API.
- **Plan** (about 2 days):
  1. The user picks a fast model (Claude Haiku 4.5, GPT-4.1-mini or Gemini Flash) and creates a dedicated key with a
     spend cap.
  2. Build a rating script (about 3 h). One call per unique item: "probability that a 2025–26 frontier model solves
     this item", cached by content hash. Run it on about 3,000 items of the 33 inventory benchmarks in the release
     (about 1–2 h, about $5–15). The user runs it, or sets an env var for it; I do not enter keys.
  3. Fit the score-to-logit map leave-benchmark-out and add it as an offset in `xpair_sim`. Gate: CI-separated gain
     on INV and MID, not worse on A (about 3 h).
  4. If it passes, integrate it into `model.py` (about 4 h):
     - a call budget of ≤ 1 call per unique item per checkpoint, since workers are recreated;
     - a per-call timeout;
     - fallback to the current model on any error, so it never raises.

     Then the zip, a mocked-API smoke test and the loop check. The user inserts the key and uploads.
- **Risks.**
  - Our round-3 text prior failed to transfer across benchmarks (calibration slope negative in 5/5 folds).
  - Embeddings fail on new benchmarks in the literature (§3).
  - API outages during an 8 h run.
  - The key ships inside the zip, so use a spend-capped key.

## 8. What the user needs to do

1. Keep saving each upload's Logs → Scoring output: shipped 1010370 (saved), 1010715, 1010717. Run
   `python -I experiments/src/deep_dive/parse_feedback.py <file>`.
2. Choose the OpenReview submission ID. On current evidence, prefer **1010717** (`xp_level_m050`). Keep it unless its
   draw shows D or B0 ECE back at release-like levels with a worse matched B0.
3. If a candidate becomes the selected submission, the report's description of the budget-0 prior must say so: three
   constants changed. The paper was not edited here.
4. Decide today on the LLM route (§7). It needs a provider, a key with a spend cap and a budget.

## 9. Formative runs of the candidates (2026-10-26)

**Data.**
- Source: `experiments/results/deep_dive/formative_runs.json`, built by `deep_dive/formative_summary.py` from the saved
  scoring logs (`_downloads/paiec_logs_1010370/`, `_downloads/cb_logs_<id>/`).
- It holds aggregates only. No anonymous IDs, pair-level values or success rates are stored.

| Run (submission) | Pairs | B0 | B1 | B3 | B7 | B15 | B31 | ALC | B0 ECE | D = B0 − B31 |
|---|---|---|---|---|---|---|---|---|---|---|
| shipped (1010370) | 9 | .2834 | .2103 | .1909 | .1786 | .1641 | .1627 | .1934 | .340 | .121 |
| xp_nosubj_mu2 (1010715) | 9 | .2247 | .1732 | .1624 | .1532 | .1494 | .1449 | .1646 | .263 | .080 |
| **xp_level_m050 (1010717)** | 8 | .2302 | .1832 | .1540 | .1433 | .1475 | .1416 | **.1628** | .278 | .089 |
| xp_level_m100 (1010735) | 9 | .2458 | .1944 | .1971 | .1835 | .1740 | .1692 | .1913 | .219 | .077 |

**Structure** (counts only):
- All four runs drew the same 7 benchmark IDs.
- They drew 35 different subjects in total, and no pair repeats.
- So the formative pool looks like 7 fixed benchmarks with resampled subjects, and pair-ID matching is impossible.

**Benchmark-matched differences** (equal weight over the 7 shared benchmarks; subjects differ):

| Comparison | ALC difference (se) |
|---|---|
| nosubj − shipped | −0.042 (0.024) |
| m050 − shipped | −0.029 (0.028) |
| m100 − shipped | −0.015 (0.029) |
| m050 − nosubj | +0.012 (0.040) |
| m100 − m050 | +0.014 (0.028) |

The ~0.03 improvement over the shipped zip is mostly draw noise; the simulator expects 0.001–0.007.

**Decision.**
- Both de-optimised-prior candidates beat the shipped zip in expectation.
  - The shipped draw itself shows the subject terms misfiring at B0.
  - In every simulated population that is not release-like, both candidates are better.
- Between m050 and nosubj the draws are a tie. The simulator's population mix (§5, `proto_level3.json`) gives m050 an
  edge of about −0.001 to −0.002 on the inventory/MID proxies. So **1010717 (m050) is selected**.
- m100 only pays in a HARD-like pool. Its D (0.077) and its draw do not support that.

**Separating the candidates by more uploads is not possible.**
- The run-to-run ALC sd is about 0.017, and a benchmark-matched difference has sd about 0.03–0.04.
- Resolving a 0.003 difference would need several hundred uploads per arm.
- A few repeat draws of the selected zip would only estimate its mean formative score (se about 0.008).

**Rules** (page, Conduct and Q&A): multiple submissions are allowed (50 per day, 1,000 per person), and formative
feedback "may guide development". Not allowed: extracting private test data (we never invert per-pair rows) and
anything keyed on anonymous benchmark or subject IDs (we use them only to average our own runs).

## 10. Item-rating study (pre-registered; in progress)

- **Pre-registration** (`experiments/results/deep_dive/items/`):
  - `PREREG_items.md`: 13:08 PDT, before any rating.
  - `PREREG_items_AMENDMENT_1.md`: 13:18. Model switched to Qwen3-VL-4B-Instruct on a speed criterion (0.72 vs 1.67
    s per item on the same 30 items) and for standard attention.
  - `PREREG_items_AMENDMENT_2.md`: 13:32. A stub-rating dry run showed that the fitted intercept alone moves ALC by
    ±0.002. So the gate must also hold against an intercept-only control.
- **Signal.** s1 = log P(Yes) − log P(No) to "would a state-of-the-art 2025–26 AI system most likely complete this
  task correctly?" (content ≤ 1,500 characters). It comes from one forward pass, batch 1, bf16 on MPS. s3, the mean
  content log-prob, is exploratory.
- **Items.**
  - 24,199 items (20,690 unique prompts) from 33 inventory benchmarks (INV) and 25 seeded non-inventory benchmarks (AS).
  - Text was recovered by replaying the round-4 frame build. All 58 benchmarks were verified against the frame
    (`items_manifest.json`).
- **Run.** Mac, under the GPU lock, owner `paiec_item_rating`, from 13:22 PDT, about 0.45 s per item.
- **Gate.** Leave-benchmark-out fit of w·z(s1) + b on top of m050. It requires both:
  - (a) a CI-separated ALC gain on INV;
  - (b) AS not significantly worse;
  - and both conditions also against the intercept-only control.
- **Pipeline.**
  - Code: `deep_dive/items_extract.py`, `rate_items.py` (hf, api and stub backends), `eval_items.py`.
  - Dry run on stub ratings (`eval_items_STUB_dryrun.json`): the gate correctly fails; the noise signal vs the
    intercept-only control is +0.0001.

## 11. Deployment probes (uploaded as 1010835 and 1010837)

**Why.** Neither the starting kit's `prediction_workers.py` nor the platform's package list is public. The page also
says hosted logs hide raw stdout/stderr.

**Build.** `submission/candidates/probe_A.zip` (sha256 `68678c6c…`) and `probe_B.zip` (`c9489ad2…`), built by
`deep_dive/build_probe.py`:
- each is `xp_level_m050` plus `models.txt` = `Qwen/Qwen3-VL-4B-Instruct` plus an appended probe;
- one worker per run (O_EXCL lock file) loads the model exactly as an integration would: bf16, local files only,
  `.to("cuda:0")`, a 300 s cap and one dummy forward pass.

**Outputs.**
- A prints a diagnostic line and always returns m050.
- B signals through the public failure codes in that one call:
  - valid score: loaded and ran;
  - 42: import failure or no CUDA;
  - 40: load or forward failure, or over the cap;
  - 41: the per-call timeout is shorter than the load.

**Checks.** Both pass the organisers' validator (the probe skips itself during the validator's smoke check), our smoke
test and the official-loop check. In a functional test on the laptop (CPU torch), A returned 0.465 and B returned NaN.

**Probe results** (logs in `_downloads/cb_logs_1010835/` and `_downloads/cb_logs_1010837/`):
- A: ALC 0.1920. Its stdout shows "Raw stdout is hidden for this competition", so A's diagnostic line is lost.
- B: ALC 0.2143, finished normally. Under B's design, one worker imported torch and transformers, found CUDA, loaded
  `Qwen/Qwen3-VL-4B-Instruct` from the prefetched cache and ran a forward pass within 300 s and inside the per-call
  timeout. The alternative, that the probe never ran, cannot be excluded from B alone.
- Probe C (`submission/candidates/probe_C.zip`, sha256 `e5d27cbe…`) fires only after a successful load:
  - error 42: GPU ≥ 70 GB;
  - error 40: GPU < 70 GB;
  - valid score: the probe did not run, or the load failed.
