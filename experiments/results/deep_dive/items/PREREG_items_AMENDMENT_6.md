# Amendment 6 to PREREG_items.md: Qwen3-14B on the Mac (written 2026-10-29 12:53 PDT, before any Qwen3-14B output exists)

State at writing: the Qwen3-14B model has not been downloaded or run, and no rating exists. The Groq and Gemini routes
are on hold (Amendments 4 and 5 unused).

Model: mlx-community/Qwen3-14B-4bit (MLX, 4-bit), run with mlx-lm in its own venv on the Mac, batch 1, one model, under
~/neurips/GPU_LOCK. The scoring-time counterpart would be Qwen/Qwen3-14B (bf16) through models.txt. Scale differences
between the two builds would be absorbed by the online-fitted slope of the W2 item layer, which is why a pass hands the
ratings to W2 instead of shipping fixed constants. Chat template with enable_thinking=False.

Signals (each its own candidate):
  s1  : log P("Yes") - log P("No") at the first answer token for the ORIGINAL PREREG_items.md prompt, verbatim, with
        the same truncation (content 1,500 / features 300 characters).
  s3  : mean token log-probability of the content span inside that same prompt (same forward pass).
  smc : multiple-choice option confidence, for items whose truncated content matches the MC rule. The rule: at least two
        option markers, matched by the regex (?:^|[\s(])([A-J])[.):]\s with consecutive letters starting at A; the
        letters found are the option set. A second prompt is sent: the truncated content followed by
        "\n\nAnswer with the letter of the correct option only." The probabilities of the option letters at the first
        answer token (log-sum-exp over the "X" / " X" token variants) are normalised over the option set.
        smc = logit(clip(max option probability, 0.005, 0.995)). Non-MC items have no smc.
Items and order: the same 24,199 items (rating_order.txt; inventory benchmarks first, then AS in seeded order). The
partial-run rule of Amendment 5 applies: only fully rated benchmarks, at least 15 inventory and 5 AS, otherwise the
gate fails.

Analysis: as in PREREG_items.md + Amendment 2 (base = xp_level_m050; candidate = base + w z(signal) + b, with (w, b) per
outer fold; comparisons vs base and vs the intercept-only control on INV and AS) + the H7-INV check of Amendment 4.
Unrated items get b, i.e. mean imputation, for EVERY signal and scenario of this amendment, because smc covers only
part of the items by construction.
Multiplicity: three signals are tested. Conditions (a) and (a') use two-sided 98.33% cluster-bootstrap intervals
(Bonferroni, 0.05/3; 2,000 resamples over benchmarks). (b), (b') and (c) keep 95%.
Decision: a signal passes if (a), (b), (a'), (b') and (c) all hold AND its point gain vs the intercept-only control on
INV is at least 0.003 ALC. A passing signal is handed to the W2 item layer as a per-benchmark feature with an
online-fitted slope; no separate zip is built. If none passes, the negative result goes into DEEP_DIVE.md and the study
stops. Go/no-go is due by 2026-10-30 10:00 PDT.
