# PAIEC submission package: what the user must do

Everything in this directory was produced offline by the automated build in
`../experiments/src/build_submission.py`. The agent cannot create accounts, accept terms or
upload anything; the steps below must be done by you. Deadline for the final codebase and
technical report: **2026-10-30, 23:59 AoE** (Anywhere on Earth = UTC-12, i.e. 2026-10-31
11:59 UTC). Submit at least a day early; Codabench (https://www.codabench.org/competitions/17828/)
was unreachable (connection refused) for the whole of 2026-10-21, so assume it may be flaky.

## Files

| File | Purpose |
|---|---|
| `paiec_submission.zip` | The Codabench upload. Root contains `model.py`, `labeling.py`, `README.md` (19.8 kB). Size and SHA-256 in `../experiments/results/submission_zip_info.json`. **Rebuilt in round 4 (2026-10-23 evening): SHA-256 `c76331d084f1...`** -- upload this one, not an older copy. |
| `model.py`, `labeling.py`, `ZIP_README.md` | Unzipped copies of the same files for inspection; byte-identical to `../experiments/src/model.py` and `labeling.py`. Round 4: `model.py` now fits a cross-pair random-effects model to the evaluator's global label list (falls back to the rounds 1-3 per-pair predictor), and `labeling.py` uses the fresh-first rule and no longer takes a `prediction` argument. The round-3 item-text-prior hooks stay inactive (no `text_prior.json` is shipped). |
| `../experiments/results/validator_output.txt` | Output of the organisers' `check_submission_zip.py` on the ZIP (`OK`, exit code 0). |
| `../paper/main.pdf` | The technical report (NeurIPS 2026 template, non-anonymous). |

## Step 1: register the team (free, one form per team)

The competition page (https://aims.stanford.edu/competition) links a registration form. It asks
for: team name, each member's name, affiliation, email, **Codabench username** and
**Hugging Face username**. Member 1 is the primary contact. Team size is up to six. Use the
e-mail `neelsirivara@outlook.com`. You need a Codabench account (https://www.codabench.org/)
and a Hugging Face account; create both yourself if you do not have them.

## Step 2: upload the ZIP to Codabench

1. Log in to Codabench, open competition 17828, read and accept the terms, and join.
2. In "My Submissions", upload `paiec_submission.zip` as is (do not re-zip the folder: `model.py`
   must stay at the ZIP root, which the validator already confirmed).
3. Wait for the run to finish (the organisers allow 8 h per run; this predictor needs minutes:
   one fit per budget checkpoint (about 2 s for 62,000 labels) and about 0.1 ms per predict call).
   Open *Logs -> Scoring output* for the formative feedback (Brier ALC and ECE on a freshly
   sampled subset capped at 1,000 subject-item pairs; it is private to you and the organisers).
4. Note the **submission ID** shown by Codabench; the OpenReview form asks for it.
5. You may submit several times during development; the final OpenReview form names the one
   submission that counts. If the formative score is clearly worse than the empirical-mean
   baseline's (unlikely: locally we beat it by 0.04-0.06 ALC, and the round-4 cross-pair model improves on our earlier
   predictor by about 0.013 ALC in formative-sized simulations), tell the agent before resubmitting.

## Step 3: fill in the report and upload it to OpenReview

1. Open `../paper/main.tex`, replace `{[}Author Name{]}` and `{[}Affiliation{]}` in the `\author{}`
   block with your name and affiliation (the e-mail is already there), then rebuild:
   `bash ../experiments/src/build_paper.sh` (MiKTeX is installed; about a minute). The script
   reports the page count and checks that `main.pdf` has no `??`.
2. The report must link the released code. `\codeurl` in `main.tex` already points to
   https://github.com/neelsirivara/neurips2026-paiec (the repository that `make_repos.py`
   maintains from the `paiec/` directory). It is **private** until you flip it to public: do that
   before the OpenReview upload, and make sure the latest `experiments/results`, `paper` and
   `submission` are pushed (the reviewers will follow the link).
3. Go to the OpenReview venue linked from the competition page, create the submission, upload
   `main.pdf`, and enter the Codabench submission ID from Step 2.

## Step 4 (optional): re-run everything

```
cd ../experiments/src
python smoke_test.py --dir ../../submission           # shipped files: interface, robustness, throughput (seconds)
python run_experiments.py --suite all --workers 4      # all result JSONs (about 45 min on 4 cores)
python run_extras.py --workers 4                       # TF-IDF representation ablation (5 min)
bash build_paper.sh                                    # macros, figures, audit, PDF
python build_submission.py                             # validator on the ZIP (rebuilds only with --force, see below)
python -I text_prior.py build && python -I text_prior.py cv  # round 3: item-text prior (needs the training payload)
python -I xpair_sim.py build && python -I xpair_sim.py grid && python -I xpair_sim.py cv --save-units   # round 4 (payload)
python -I xpair_sim.py grid --config A --grid ext && python -I xpair_sim.py grid --config E --grid ext && python -I xpair_sim.py addendum
python -I xpair_sim.py acq && python xpair_sim.py realtest && python xpair_sim.py realacq && python -I xpair_sim.py runtime
python -I xpair_sim.py equivalence && python -I xpair_sim.py perpair && python official_loop_check.py --dir ../../submission
```

The real Ollama testbed (`../experiments/results/real_testbed/*.jsonl`, plus the frozen copy of
the earlier chain-of-thought runs in `real_testbed_external/`) is already on disk; to
regenerate it you need Ollama with `qwen2.5:1.5b-instruct`, `qwen2.5:3b-instruct`,
`qwen2.5:7b-instruct`, `llama3.2:3b` and the local copies of GSM8K, SVAMP and BIG-Bench-Hard
listed in `real_testbed_generate.py` (about two hours on an otherwise idle 4-6 thread CPU).
The MiniLM embedding ablation additionally needs `sentence-transformers/all-MiniLM-L6-v2` in the
local Hugging Face cache (`embed_cache.py`, offline).
