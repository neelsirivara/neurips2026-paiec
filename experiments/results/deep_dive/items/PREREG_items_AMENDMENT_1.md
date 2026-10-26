# Amendment 1 to PREREG_items.md (written 2026-10-26 13:18 PDT, before any rating has been analysed)

State at the time of writing: PREREG_items.md (SHA-256 06d9e784369b70968d316be99ee476895a8f5d0a4d09483c31f751e71fe33146)
was written at 13:08 PDT. The only ratings that exist are a 30-item speed test of Qwen/Qwen3.5-4B (13:11-13:13 PDT,
file ratings_qwen35_4b.jsonl on the Mac). Their values were not summarised, correlated or used in any fit.

Change: the rating model may be either Qwen/Qwen3.5-4B (snapshot 851bf6e8...) or Qwen/Qwen3-VL-4B-Instruct (snapshot
ebb281ec70b05090aa6165b016eac8ec08e71b17, text-only use, chat template from its processor). The choice is made by ONE
criterion, measured on the same first 30 items in the pre-registered order: the lower mean seconds per item with the
pre-registered prompt and signals. The faster model is used for all items.

Reasons, all engineering and none outcome-based:
(1) Speed. Qwen3.5-4B's linear-attention layers fall back to reference PyTorch kernels on MPS (causal_conv1d and
    flash-linear-attention are absent) and took 1.0-1.8 s per item. The 24,199 items would hold the shared GPU about 8.7 h.
(2) Deployability. A standard-attention model needs no special kernels on the Codabench GPU either.

Also allowed: if the content log-probability s3 (exploratory) roughly doubles the time per item, it may be dropped
(--no-s3), with s1 computed from last-position logits only. The primary signal s1, the prompt, the truncation, the
offset model and the gate are unchanged.
