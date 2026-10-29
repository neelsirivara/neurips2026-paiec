# Rule for switching the selected PAIEC submission to the W2 candidate (written 2026-10-29, before its real draws)

Candidate: `submission/candidates/w2_f05s1_k1q2.zip` (sha256 `d6d1c6a0…`), = `xp_level_m050` + an online
item layer (item_features category effects + char-4-gram TF-IDF kernel on residuals). It passed its
pre-registered simulator gate (`PREREG.md`, `confirm.json`): INV -0.0035 [-0.0051, -0.0022], A -0.0024
[-0.0032, -0.0016], SPARSE -0.0021 [-0.0035, -0.0009]; budget 0 unchanged.

Real draws uploaded: 1019682, 1019683, 1019685, 1019686, 1019688. The baseline has seven draws of
`xp_level_m050` (mean 0.1904, sd 0.0187). A -0.002 effect is far below what five draws can resolve
(se of a difference about 0.011), so the real draws only screen for platform failures and gross
regressions; the selection rests on the pre-registered simulator gate.

**Switch** the selected submission from 1010717 to the first W2 run that finished, if and only if:
1. all five W2 runs finish with a score (no platform error code), and
2. the W2 mean Brier-ALC is not worse than the m050 mean by more than 0.022 (two standard errors of
   the difference: sqrt(0.0187^2/7 + 0.0187^2/5) = 0.011), and
3. no per-budget W2 mean exceeds the m050 mean at the same budget by more than 0.03.

Otherwise 1010717 stays, and the report presents W2 as a simulator result that did not hold up in
deployment.

**Contingency (added 2026-10-29 19:30 UTC, still before any W2 draw has finished: the Codabench queue
had not started them after about an hour):** if by Oct 30 12:00 PDT fewer than five W2 runs have finished
but at least two have, conditions 2 and 3 are applied to those, with the standard error recomputed for
the actual count. If fewer than two have finished, 1010717 stays.
