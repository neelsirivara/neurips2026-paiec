"""Item-rating study, Qwen3-14B (MLX 4-bit) on the Mac (PREREG_items_AMENDMENT_6.md).

    ~/neurips/mlx_venv/bin/python -I rate_items_mlx.py --items deep_dive_items.jsonl --out ratings_qwen3_14b.jsonl \
        [--model mlx-community/Qwen3-14B-4bit] [--order a,b,...] [--limit N]

One line per unique prompt: {"key", "idx", "dir", "s1", "s3", "smc", "mc_n", "n_tokens", "backend": "mlx", "model", "sec"}.
  s1  = log P(Yes) - log P(No) at the first answer token of the ORIGINAL pre-registered prompt (chat template,
        enable_thinking=False);
  s3  = mean token log-probability of the content span inside that prompt;
  smc = for multiple-choice items: logit of the max option probability (normalised over the detected option letters)
        when asked for the letter only; null otherwise.
The key is the same sha1(prompt) as rate_items.py, so eval_items.py maps the ratings back. Resumable. It refuses to load
with < 30% free memory and stops below 20% (macOS memory_pressure, checked every 20 prompts). Batch 1. No downloads: HF_HUB_OFFLINE=1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rate_items as RI  # noqa: E402  (build_prompt, prompt_key: the original pre-registered prompt)

MC_RE = re.compile(r"(?:^|[\s(])([A-J])[.):]\s")
MC_SUFFIX = "\n\nAnswer with the letter of the correct option only."


def mc_options(content):
    seen = []
    for m in MC_RE.finditer(content):
        L = m.group(1)
        if L not in seen:
            seen.append(L)
    if "A" not in seen or "B" not in seen:
        return []
    letters = sorted(seen)
    k = 0
    while k < len(letters) and letters[k] == "ABCDEFGHIJ"[k]:
        k += 1
    return list("ABCDEFGHIJ"[:k]) if k >= 2 else []


def mem_free_pct():
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=30).stdout
        m = re.search(r"free percentage:\s*(\d+)%", out)
        return float(m.group(1)) if m else None
    except Exception:
        return None


class MLXRater:
    def __init__(self, model_id):
        import mlx.core as mx
        from mlx_lm import load
        self.mx = mx
        self.model, self.tok = load(model_id)
        self.hf = getattr(self.tok, "_tokenizer", self.tok)
        enc = lambda s: self.hf.encode(s, add_special_tokens=False)
        self.yes = sorted({enc(v)[0] for v in ("Yes", " Yes") if enc(v)})
        self.no = sorted({enc(v)[0] for v in ("No", " No") if enc(v)})
        self.letter_ids = {L: sorted({enc(v)[0] for v in (L, " " + L) if enc(v)}) for L in "ABCDEFGHIJ"}

    def chat(self, user):
        msgs = [{"role": "user", "content": user}]
        try:
            return self.hf.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return self.hf.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def logits(self, text, offsets=False):
        mx = self.mx
        enc = self.hf(text, add_special_tokens=False, return_offsets_mapping=offsets)
        ids = enc["input_ids"]
        out = self.model(mx.array(ids)[None])
        return out[0], ids, (enc["offset_mapping"] if offsets else None)

    def rate(self, prompt, content_span, content_for_mc):
        mx = self.mx
        full = self.chat(prompt)
        lg, ids, offs = self.logits(full, offsets=True)
        last = lg[-1].astype(mx.float32)
        lse = mx.logsumexp(last)
        lp = lambda toks: mx.logsumexp(last[mx.array(toks)]) - lse
        s1 = (lp(self.yes) - lp(self.no)).item()
        s3 = None
        cs = full.find(content_span) if content_span else -1
        if cs >= 0:
            ce = cs + len(content_span)
            pos = [t for t, (a, b) in enumerate(offs) if a >= cs and b <= ce and t > 0]
            if pos:
                sel = lg[mx.array([p - 1 for p in pos])].astype(mx.float32)
                tgt = mx.array([ids[p] for p in pos])
                lps = mx.take_along_axis(sel, tgt[:, None], axis=1)[:, 0] - mx.logsumexp(sel, axis=1)
                s3 = mx.mean(lps).item()
        smc, n_opt = None, 0
        opts = mc_options(content_for_mc)
        if opts:
            n_opt = len(opts)
            lg2, _, _ = self.logits(self.chat(content_for_mc + MC_SUFFIX))
            last2 = lg2[-1].astype(mx.float32)
            vals = mx.array([mx.logsumexp(last2[mx.array(self.letter_ids[L])]).item() for L in opts])
            probs = mx.softmax(vals)
            q = min(max(mx.max(probs).item(), 0.005), 0.995)
            smc = math.log(q / (1 - q))
        return {"s1": float(s1), "s3": s3, "smc": smc, "mc_n": n_opt, "n_tokens": len(ids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="mlx-community/Qwen3-14B-4bit")
    ap.add_argument("--order", default="")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    items = [json.loads(l) for l in open(a.items, encoding="utf-8")]
    if a.order:
        pri = {d: i for i, d in enumerate(a.order.split(","))}
        items.sort(key=lambda r: (pri.get(r["dir"], len(pri)), r["idx"]))
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            try:
                done.add(json.loads(l)["key"])
            except Exception:
                pass
    todo, seen = [], set(done)
    for r in items:
        p = RI.build_prompt(r["content"], r["features"])
        k = RI.prompt_key(p)
        if k not in seen:
            seen.add(k)
            todo.append((k, p, r))
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(items)} items, {len(todo)} prompts to rate", flush=True)
    if not todo:
        return 0
    free = mem_free_pct()
    print(f"memory free {free}%", flush=True)
    if free is not None and free < 30:
        print("refusing to load: < 30% free", flush=True)
        return 3
    t0 = time.time()
    R = MLXRater(a.model)
    print(f"loaded {a.model} in {time.time() - t0:.0f}s; yes {R.yes} no {R.no}", flush=True)
    t0 = time.time()
    with open(a.out, "a", encoding="utf-8") as fh:
        for n, (k, p, r) in enumerate(todo, 1):
            t1 = time.time()
            span = " ".join(str(r["content"] or "").split())[:RI.CONTENT_CHARS]
            try:
                res = R.rate(p, span, span)
                err = None
            except Exception as e:
                res, err = {"s1": None, "s3": None, "smc": None, "mc_n": 0, "n_tokens": None}, repr(e)[:200]
            rec = {"key": k, "idx": r["idx"], "dir": r["dir"], **res, "backend": "mlx", "model": a.model,
                   "sec": round(time.time() - t1, 3)}
            if err:
                rec["error"] = err
            fh.write(json.dumps(rec) + "\n")
            if n % 50 == 0:
                fh.flush()
                el = time.time() - t0
                print(f"{n}/{len(todo)} rated, {el / n:.3f} s/item, eta {(len(todo) - n) * el / n / 60:.1f} min", flush=True)
            if n % 20 == 0:
                free = mem_free_pct()
                if free is not None and free < 20:
                    print(f"stopping: memory free {free}% < 20%", flush=True)
                    return 4
    print(f"done: {len(todo)} prompts in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
