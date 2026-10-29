# Amendment 5 to PREREG_items.md: the Groq route (written 2026-10-29 11:24 PDT, before any Groq output exists)

State at writing: no Groq request has been sent. The Gemini route (Amendment 4) is on hold and will not be run, because
the key is free tier. Scripts: deep_dive/rate_items_groq.py (rating), rater_groq_appendix.py.txt +
build_rater_groq.py (scoring time). Both were tested only against mocked APIs.

Rater and model rule. Candidates in this order: openai/gpt-oss-120b (reasoning_effort "low", include_reasoning false,
temperature 0, max_tokens 512), then llama-3.3-70b-versatile (temperature 0, max_tokens 8).
The pilot sends 50 prompts to the first visible candidate and 10 to the second, spread over the pre-registered order.
It reads the per-model limits from the response headers: x-ratelimit-limit-requests (requests per day) and
x-ratelimit-limit-tokens (tokens per minute), plus the remaining/reset values.
Choice: the first candidate that parses at least 90% of its pilot answers AND whose limits allow rating every prompt
before the deadline; failing that, the candidate whose limits allow the most prompts. The choice is recorded in
api_pilot_groq.json.

Signal: identical to Amendments 3 and 4. Same verbalised 0-100 probability prompt (rate_items_claude.PROMPT_API), same
truncation, s1 = logit(clip(p/100, 0.005, 0.995)). Unparsed answers or errors count as unrated.

Items and order: the same 24,199 items (20,720 unique prompts), rated in the pre-registered order (rating_order.txt:
the 33 inventory benchmarks, then the 25 AS benchmarks in seeded order). If the rate limits stop the run before it ends,
the gate is evaluated ONLY on benchmarks whose needed items are all rated:
  * the INV scenario on the fully rated inventory benchmarks;
  * the AS scenario on the fully rated AS benchmarks, if there are at least 5; otherwise condition (b) is "not
    evaluable" and the gate FAILS.
The gate also FAILS if fewer than 15 inventory benchmarks are fully rated. No partial benchmark enters the gate.

Analysis and gate: identical to Amendment 4. Base = xp_level_m050; candidate = base + w z(s1) + b, with (w, b) fitted per
outer fold. Pass requires all of: (a) INV vs base CI below 0; (b) AS vs base not significantly above 0; (a') INV vs
intercept-only CI below 0; (b') AS vs intercept-only not significantly above 0; (c) H7-INV vs intercept-only not
significantly above 0.
Practical bar for building a scoring-time zip: a point gain of at least 0.003 ALC vs the intercept-only control on INV.
The scoring-time zip also needs the org-level rate limits to cover one evaluation run (about 1,000-1,500 unique items
across all workers). Otherwise the rater would be cut off partway and the shipped gain would shrink. The builder sizes
the per-process request rate to the pilot's measured limit divided by 16 workers.
