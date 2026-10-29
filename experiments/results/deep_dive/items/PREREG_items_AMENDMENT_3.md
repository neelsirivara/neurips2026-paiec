# Amendment 3 to PREREG_items.md: the API route (written 2026-10-26 14:08 PDT, before any API rating exists)

The coordinator may obtain a Claude API key from the user. If so, the same pre-registered items get rated by Claude
through the Message Batches API (script deep_dive/rate_items_claude.py). The model is claude-opus-5-5 at effort "low",
or claude-haiku-5-5 at effort "low" if the user chooses that one for cost.

Signal: the Claude API returns no token log-probabilities, so the probability is verbalised. Same framing as the local
prompt; the only change is the final instruction ("estimate the probability, from 0 to 100, ... Reply with the integer
probability only"). Primary signal s1_api = logit(clip(p/100, 0.005, 0.995)). Content and feature truncation are
unchanged (1,500 and 300 characters). Unparsed answers, refusals and max_tokens stops count as unrated (offset 0, as in
the original analysis).

Analysis and gate: identical to PREREG_items.md plus Amendment 2. Base = xp_level_m050; w and b are fitted per outer
fold; gate vs base AND vs the intercept-only control, on INV and AS. The API signal is evaluated as its own candidate.
The local Qwen3-VL signal and the API signal are not combined unless one of them passes on its own; any combination is
exploratory.
