# Amendment 4 to PREREG_items.md: the Gemini route (written 2026-10-29 11:13 PDT, before any Gemini output exists)

State at writing: no Gemini request has been sent. The user has a free-tier key only, and no competition item may go
through a free-tier key: Google may use free-tier inputs for training, with human review. The rating script
(deep_dive/rate_items_gemini.py) refuses to send items without --paid-tier-confirmed. It was tested only against a
mocked API.

Rater: gemini-2.5-flash through the Gemini API (generateContent, v1beta), with temperature 0. Generation config:
"nothink" = thinkingConfig.thinkingBudget 0, maxOutputTokens 16. If the API rejects that config, or fewer than 2 of 3
probe answers parse, the fallback is "default" (API-default thinking, maxOutputTokens 1024). The config is chosen once
on the first 3 items of the run and recorded in every output line.

Signal: the verbalised probability. The prompt is identical, character for character, to Amendment 3 (rate_items_claude.PROMPT_API):
"estimate the probability, from 0 to 100, that such a system completes it correctly ... Reply with the integer
probability only". s1_gem = logit(clip(p/100, 0.005, 0.995)).
Why verbalised rather than the model's own answer confidence:
(a) for RLHF-tuned models, stated confidences are better calibrated than token probabilities (Tian et al., "Just Ask
    for Calibration", EMNLP 2023, arXiv 2305.14975: verbalised confidences cut ECE by about 50% relative);
(b) many inventory items are agentic or open-ended tasks with no single checkable answer, so "own answer confidence"
    is undefined for them;
(c) the signal stays comparable with the local Qwen route. That route asked a Yes/No question on the same framing,
    failed the gate, and had within-benchmark correlation 0.09 with the empirical item effect.
Unparsed answers, blocked answers and request errors count as unrated (offset 0), as in the original analysis.

Items: the same 24,199 items (20,720 unique prompts under this prompt), the same truncation (1,500 / 300 characters)
and the same order (inventory benchmarks first).

Analysis and gate: identical to PREREG_items.md plus Amendment 2. Base = xp_level_m050; candidate = base + w z(s1_gem)
+ b, with (w, b) fitted per outer fold. Ship only if all of these hold:
  (a)  INV: candidate − base CI entirely below 0;            (b)  AS: candidate − base not significantly above 0;
  (a') INV: candidate − intercept-only CI entirely below 0;  (b') AS: candidate − intercept-only not significantly above 0;
  (c)  extra check, H7-INV (hidden-run structure on the inventory benchmarks; deep_dive/npeb.py): candidate −
       intercept-only not significantly above 0. Items of H7-INV test sets that were not rated get offset b (mean
       imputation), as at scoring time; the coverage is reported.
Practical bar for building a scoring-time zip (agreed with the coordinator before any output): a point gain of at least
0.003 ALC vs the intercept-only control on INV.
The Qwen and Gemini signals are not combined unless one of them passes alone; any combination is exploratory.
