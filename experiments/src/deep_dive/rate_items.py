"""Item-rating study, step 2: rate items with an LLM (pre-registered prompt; PREREG_items.md).

    python rate_items.py --items deep_dive_items.jsonl --out ratings.jsonl --backend hf  [--model Qwen/Qwen3.5-4B]
    python rate_items.py ... --backend api   # OpenAI-compatible endpoint; env PAIEC_API_KEY, PAIEC_API_BASE, PAIEC_API_MODEL
    python rate_items.py ... --backend stub  # deterministic fake signal, for dry runs of the pipeline

One line per item in --out: {"key": sha1(prompt), "idx", "dir", "s1", "s3", "n_tokens", "backend", "model", "sec"}.
  s1 = log P("Yes") - log P("No") for the first answer token (log-sum-exp over the "Yes"/" Yes" and "No"/" No" first
       tokens);
  s3 = mean token log-probability of the item-content span (hf backend only; null otherwise).
Resumable: keys already in --out are skipped.  The hf backend runs batch 1 with no generation, refuses to start when the
system reports < 30% free memory (macOS ``memory_pressure``) and stops when it falls below 20%.  It never downloads:
it loads from the local Hugging Face cache only.
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

PROMPT = ("You are estimating how hard a benchmark task is for state-of-the-art AI systems released in 2025-2026 (for "
          "example GPT-5, Claude Opus 4, Gemini 2.5 Pro), possibly running inside an agent harness. Read the task and "
          "answer with one word: Yes if such a system would most likely complete it correctly, No otherwise.\n\n"
          "Task:\n<<<\n{content}\n>>>\n\nItem attributes: {features}\n\n"
          "Would a state-of-the-art 2025-2026 AI system most likely complete this task correctly? Answer Yes or No.")
CONTENT_CHARS, FEATURE_CHARS = 1500, 300


def build_prompt(content: str, features: str) -> str:
    c = " ".join(str(content or "").split())[:CONTENT_CHARS]
    f = " ".join(str(features or "").split())[:FEATURE_CHARS] or "none"
    return PROMPT.format(content=c, features=f)


def prompt_key(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()


def mem_free_pct() -> float | None:
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=30).stdout
        m = re.search(r"free percentage:\s*(\d+)%", out)
        return float(m.group(1)) if m else None
    except Exception:
        return None


class StubBackend:
    name, model = "stub", "stub"

    def rate(self, prompt, content_span):
        h = int(hashlib.sha1(prompt.encode()).hexdigest()[:8], 16) / 2 ** 32
        return {"s1": math.log(h + 1e-9) - math.log(1 - h + 1e-9), "s3": None, "n_tokens": len(prompt) // 4}


class HFBackend:
    name = "hf"

    def __init__(self, model_id: str, threads: int):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        import torch
        from transformers import AutoTokenizer
        torch.set_num_threads(threads)
        self.torch = torch
        self.model = model_id
        self.dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        kw = dict(dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True)
        m, errs = None, []
        for loader in ("AutoModelForCausalLM", "Qwen3_5ForConditionalGeneration", "AutoModelForImageTextToText"):
            try:
                import transformers
                m = getattr(transformers, loader).from_pretrained(model_id, **kw)
                self.loader = loader
                break
            except Exception as e:  # pragma: no cover
                errs.append(f"{loader}: {e!r}"[:300])
        if m is None:
            raise RuntimeError("could not load model: " + " | ".join(errs))
        self.m = m.to(self.dev).eval()
        self.proc = None
        self.s3 = True
        ids = lambda s: self.tok.encode(s, add_special_tokens=False)
        self.yes = sorted({ids(v)[0] for v in ("Yes", " Yes") if ids(v)})
        self.no = sorted({ids(v)[0] for v in ("No", " No") if ids(v)})

    def text(self, prompt):
        msgs = [{"role": "user", "content": prompt}]
        if getattr(self.tok, "chat_template", None):
            try:
                return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if self.proc is None:
            from transformers import AutoProcessor
            self.proc = AutoProcessor.from_pretrained(self.model, local_files_only=True)
        pm = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        return self.proc.apply_chat_template(pm, tokenize=False, add_generation_prompt=True)

    def rate(self, prompt, content_span):
        torch = self.torch
        full = self.text(prompt)
        enc = self.tok(full, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
        offs = enc.pop("offset_mapping")[0].tolist()
        inp = enc["input_ids"].to(self.dev)
        kw = {} if self.s3 else {"logits_to_keep": 1}
        with torch.no_grad():
            out = self.m(input_ids=inp, attention_mask=enc["attention_mask"].to(self.dev), **kw)
        logits = out.logits[0]
        last = torch.log_softmax(logits[-1].float(), dim=-1)
        lse = lambda ids: torch.logsumexp(last[ids], dim=0).item()
        s1 = lse(self.yes) - lse(self.no)
        s3 = None
        cs = full.find(content_span) if (content_span and self.s3) else -1
        if cs >= 0:
            ce = cs + len(content_span)
            pos = [t for t, (a, b) in enumerate(offs) if a >= cs and b <= ce and t > 0]
            if pos:
                lp = torch.log_softmax(logits[[p - 1 for p in pos]].float(), dim=-1)
                tgt = inp[0, pos]
                s3 = lp.gather(1, tgt[:, None]).mean().item()
        return {"s1": float(s1), "s3": s3, "n_tokens": int(inp.shape[1])}


class APIBackend:
    """OpenAI-compatible chat completions with logprobs (OpenAI, Together, Groq, DeepSeek, Mistral, xAI, Google's
    OpenAI-compatible endpoint).  Key from PAIEC_API_KEY; never printed or written."""
    name = "api"

    def __init__(self):
        import urllib.request  # noqa: F401
        self.key = os.environ.get("PAIEC_API_KEY", "")
        self.base = os.environ.get("PAIEC_API_BASE", "https://api.openai.com/v1").rstrip("/")
        self.model = os.environ.get("PAIEC_API_MODEL", "")
        if not self.key or not self.model:
            raise SystemExit("api backend needs PAIEC_API_KEY and PAIEC_API_MODEL in the environment")

    def rate(self, prompt, content_span):
        import urllib.request
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1,
                "temperature": 0, "logprobs": True, "top_logprobs": 10}
        req = urllib.request.Request(self.base + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            resp = json.loads(r.read())
        ch = resp["choices"][0]
        tops = ((ch.get("logprobs") or {}).get("content") or [{}])[0].get("top_logprobs") or []
        yes = [t["logprob"] for t in tops if t.get("token", "").strip().lower() == "yes"]
        no = [t["logprob"] for t in tops if t.get("token", "").strip().lower() == "no"]
        lse = lambda v: max(v) + math.log(sum(math.exp(x - max(v)) for x in v))
        if yes and no:
            s1 = lse(yes) - lse(no)
        else:
            txt = (ch.get("message") or {}).get("content", "").strip().lower()
            s1 = 4.0 if txt.startswith("yes") else (-4.0 if txt.startswith("no") else 0.0)
        return {"s1": float(s1), "s3": None, "n_tokens": None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=("hf", "api", "stub"), required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--order", default="", help="comma-separated benchmark dirs to rate first")
    ap.add_argument("--no-s3", action="store_true", help="skip the content log-prob (last-position logits only)")
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
    todo = []
    for r in items:
        p = build_prompt(r["content"], r["features"])
        k = prompt_key(p)
        if k not in done:
            todo.append((k, p, r))
            done.add(k)
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(items)} items, {len(todo)} to rate", flush=True)
    if not todo:
        return 0
    if a.backend == "hf":
        free = mem_free_pct()
        print(f"memory free {free}%", flush=True)
        if free is not None and free < 30:
            print("refusing to load: < 30% free", flush=True)
            return 3
        be = HFBackend(a.model, a.threads)
        be.s3 = not a.no_s3
        print(f"loaded {a.model} via {be.loader} on {be.dev}; yes ids {be.yes} no ids {be.no}", flush=True)
    elif a.backend == "api":
        be = APIBackend()
    else:
        be = StubBackend()
    t0 = time.time()
    with open(a.out, "a", encoding="utf-8") as fh:
        for n, (k, p, r) in enumerate(todo, 1):
            t1 = time.time()
            span = " ".join(str(r["content"] or "").split())[:CONTENT_CHARS]
            try:
                res = be.rate(p, span)
                err = None
            except Exception as e:
                res, err = {"s1": None, "s3": None, "n_tokens": None}, repr(e)[:200]
            rec = {"key": k, "idx": r["idx"], "dir": r["dir"], **res, "backend": be.name, "model": getattr(be, "model", ""),
                   "sec": round(time.time() - t1, 4)}
            if err:
                rec["error"] = err
            fh.write(json.dumps(rec) + "\n")
            if n % 50 == 0:
                fh.flush()
                el = time.time() - t0
                print(f"{n}/{len(todo)} rated, {el / n:.3f} s/item, eta {(len(todo) - n) * el / n / 60:.1f} min", flush=True)
            if a.backend == "hf" and n % 200 == 0:
                free = mem_free_pct()
                if free is not None and free < 20:
                    print(f"stopping: memory free {free}% < 20%", flush=True)
                    return 4
    print(f"done: {len(todo)} items in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
