# Addendum 1 to the round-3 pre-registration (written 2026-10-23 ~11:10 PDT, after the primary folds ran)

Status: **exploratory**. Nothing in this addendum can change the ship decision, which is made by the
rule in `PREREG.md` on the primary analysis only.

What had been seen when this was written: in all five outer folds and in the final fit the inner
benchmark-grouped CV chose the largest pre-registered ridge penalty (alpha = 10), the inner MSE at that
penalty was at or above the constant predictor's, and the out-of-fold calibration slope W was negative
(-0.03 to -0.32). The primary CV summary (ALC difference) had not been computed yet.

Question: is the negative primary result an artefact of the penalty grid ending too early, or does item
text not transfer across benchmarks in this form at all?

Exploratory variants, each run with the identical nested procedure, folds, frame, metrics and cluster
bootstrap as the primary analysis (only the feature set and the penalty grid differ):
- E1 "heavier penalty": all features, alpha in {10, 100, 1000, 10000}.
- E2 "statistics only": the 16 dense text statistics without hashed n-grams, alpha in {0.1, 1, 10, 100}.
Reported for each: selected alpha per fold, W and B per fold, CV B0 and ALC, paired differences against
the hand-set prior (cluster bootstrap). Whatever the outcome, the shipped predictor is decided by
`PREREG.md`; a positive exploratory result would be reported as a hypothesis for a separately
pre-registered follow-up, not shipped in this round.
