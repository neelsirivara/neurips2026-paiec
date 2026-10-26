# Amendment 2 to PREREG_items.md (written 2026-10-26 13:31 PDT, before any real rating has been analysed)

State: the Qwen3-VL-4B-Instruct rating run is in progress on the Mac. No real rating has been read into any analysis.
The analysis script was dry-run on STUB ratings (deterministic hash noise; eval_items_STUB_dryrun.json).

Finding from the dry run: the pre-registered candidate (base + w z(s1) + b) contains a free intercept b. b is fitted on
the training folds' INV and AS rows. It moves the budget-0 level, and with that alone it changes ALC by +0.0011 on INV
and -0.0022 on AS, even for a pure-noise s1. A gain could therefore come from re-levelling, not from the rating.

Change (stricter, added): integration additionally requires the SAME two gate conditions to hold against the
intercept-only control (base + b fitted the same way with w = 0), i.e.
  (a') INV: ALC(s1 candidate) - ALC(b-only) has a 95% cluster-bootstrap CI entirely below 0;
  (b') AS: that difference is not significantly above 0.
The pre-registered gate (vs base) is still computed and reported unchanged. Ship only if both gates pass.
