"""Fit the budget-0 prior of model.py on the official PAIEC training release (measurement-db).

Run with ``python -I`` (it reads downloaded data).  One command does everything once the
Hugging Face login exists (``hf auth login``) and the dataset terms are accepted:

    python -I fit_prior_from_measurement_db.py all --payload ../../../_downloads/measurement_db_payload

Individual steps:

    python -I fit_prior_from_measurement_db.py plan     --payload DIR [--max-gb 30] [--skip-items]
    python -I fit_prior_from_measurement_db.py download --payload DIR [--max-gb 30] [--skip-items]
    python -I fit_prior_from_measurement_db.py fit      --payload DIR [--out ../results/prior_fit.json]
    python -I fit_prior_from_measurement_db.py testbeds [--fit ../results/prior_fit.json] [--workers 2]
    python -I fit_prior_from_measurement_db.py dry-run  [--tmp DIR]   # synthetic tables, release schema

plan      lists the repository (HfApi), downloads the small ``benchmarks.parquet`` of every benchmark
          directory and keeps the directories whose benchmark has ``response_type == "binary"`` and
          ``granularity == "item"`` (the organisers' eligibility rule in tools/prepare_data.py).  It
          writes ``plan.json`` with the byte count of the tables still to download and stops if that
          exceeds ``--max-gb``.
download  ``hf_hub_download`` of ``subjects/items/responses.parquet`` per eligible directory into
          ``DIR/data/<dir>/`` (``responses`` stored as ``response.parquet`` as prepare_data.py does)
          plus ``data/corpus_manifest.json``.  ``--skip-items`` leaves out the large items tables; the
          item part of the prior then keeps its hand-set coefficients.
fit       one row per (subject, test condition, item) with label = mean response over trials; the
          hand-set prior's feature terms (``model.subject_terms`` / ``model.item_terms``) plus a
          provider one-hot; L2 logistic regression with pair-equal weights; benchmark-grouped K-fold
          cross-validation against the hand-set prior on (a) the budget-0 Brier score and (b) the
          closed-form shrinkage Brier ALC of the competition protocol; coefficients as plain Python
          literals in ``prior_fit.json``.
testbeds  evaluates the fitted constants inside the real harness (run_experiments.run_dataset) on the
          two local testbeds, paired against the shipped constants, and records whether the shipping
          rule holds (better under benchmark-held-out CV AND on the real Ollama testbed).
dry-run   builds synthetic tables with the release schema from a known generating model, runs ``fit``
          on them and reports the recovered coefficients (``../results/prior_fit_dry_run.json``).

Only the fitting pipeline uses numpy/pandas/pyarrow/scikit-learn; the shipped model.py stays
standard-library only.  Nothing here is executed at competition time.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

# thread budget (shared machine): BLAS/OpenMP and pyarrow; override with PAIEC_FIT_THREADS
_THREADS = os.environ.get("PAIEC_FIT_THREADS", "2")
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, _THREADS)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np

try:
    import pyarrow as _pa
    _pa.set_cpu_count(int(_THREADS))
    _pa.set_io_thread_count(max(2, int(_THREADS)))
except Exception:
    pass

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import harness  # noqa: E402
import model  # noqa: E402

RESULTS = HERE.parent / "results"
PUBLIC_DATA_REPO = "aims-foundations/measurement-db"
CORPUS_SCOPE = "paec-public-training"
TABLES = ("benchmarks", "subjects", "items", "responses")
CORE_TABLES = ("benchmarks", "subjects", "responses")
SUBJECT_COLS = ("normalized_name", "provider", "release_date", "access_date", "harness", "harness_version",
                "reasoning_effort", "subject_features_extra")
CONTENT_CANDIDATES = ("content", "item_content", "prompt", "question", "input", "text")
FEATURE_CANDIDATES = ("item_features", "features", "metadata")
RESPONSE_CANDIDATES = ("response", "score", "correct", "is_correct", "label")
MIN_PAIR_ITEMS_FIT = 20       # pairs with fewer distinct items do not enter the fit
MIN_PAIR_ITEMS_ALC = 80       # protocol minimum for the ALC evaluation
MAX_ITEMS_PER_PAIR = 200      # subsample per pair (fit and CV); keeps the frame small (>= 80 for the ALC protocol)
MIN_PROVIDER_PAIRS = 15       # providers with fewer pairs are pooled into "other"
C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
CAP_GRID = (1.5, 2.5)
ALPHA_GRID = (2.0, 4.0, 8.0)
N_FOLDS = 5
N_BOOT = 1000
SUBJECT_NAMES = list(model.SUBJECT_TERM_NAMES)
ITEM_NAMES = list(model.ITEM_TERM_NAMES)


def log(msg: str) -> None:
    print(msg, flush=True)


# ============================================================================= HF access
def _api():
    from huggingface_hub import HfApi
    return HfApi()


def _explain_gate(exc: Exception) -> None:
    log(f"Hugging Face access failed: {type(exc).__name__}: {str(exc)[:300]}")
    log("  -> log in with `hf auth login` and accept the terms of "
        f"https://huggingface.co/datasets/{PUBLIC_DATA_REPO} (gated = auto: acceptance is immediate).")


def list_tables(api) -> tuple[str, dict]:
    """Return (revision sha, {dir: {table: bytes}}) for every <dir>/formatted_tables/*.parquet."""
    info = api.dataset_info(PUBLIC_DATA_REPO)
    root = list(api.list_repo_tree(PUBLIC_DATA_REPO, repo_type="dataset", recursive=False))
    dirs = sorted(e.path for e in root if type(e).__name__ == "RepoFolder")

    def one(d):
        out = {}
        try:
            for e in api.list_repo_tree(PUBLIC_DATA_REPO, path_in_repo=f"{d}/formatted_tables",
                                        repo_type="dataset", recursive=False):
                if type(e).__name__ == "RepoFile" and e.path.endswith(".parquet"):
                    out[e.path.rsplit("/", 1)[-1][:-8]] = int(e.size or 0)
        except Exception:
            return d, {}
        return d, out

    tables = {}
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for d, out in ex.map(one, dirs):
            if "benchmarks" in out:
                tables[d] = out
    return info.sha, tables


def _fetch(filename: str, revision: str, cache_dir: Path) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(PUBLIC_DATA_REPO, filename, repo_type="dataset", revision=revision, cache_dir=str(cache_dir))


def _read_parquet(path, columns=None):
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=columns)


def _eligible(bench_row: dict) -> tuple[bool, str]:
    rt = str(bench_row.get("response_type", "")).strip().lower()
    gr = str(bench_row.get("granularity", "")).strip().lower()
    ok = rt == "binary" and gr == "item"
    return ok, f"response_type={rt or '?'}, granularity={gr or '?'}"


def cmd_plan(args) -> dict:
    payload = Path(args.payload).resolve()
    payload.mkdir(parents=True, exist_ok=True)
    cache = payload / "hf_cache"
    api = _api()
    log(f"Listing {PUBLIC_DATA_REPO} ...")
    try:
        revision, tables = list_tables(api)
    except Exception as exc:
        _explain_gate(exc)
        raise SystemExit(2)
    log(f"  revision {revision}; {len(tables)} benchmark directories with formatted tables")
    log("  downloading every benchmarks.parquet (a few MB in total) to resolve eligibility ...")

    def one(d):
        try:
            path = _fetch(f"{d}/formatted_tables/benchmarks.parquet", revision, cache)
            tbl = _read_parquet(path)
            row = {k: v for k, v in zip(tbl.column_names, [c[0].as_py() if len(c) else None for c in tbl.columns])}
            ok, why = _eligible(row)
            return d, ok, why, {k: (v if isinstance(v, (str, int, float, bool)) or v is None else str(v))
                                for k, v in row.items() if k in ("benchmark_id", "name", "release_date", "n_items", "n_subjects",
                                                                 "response_type", "granularity", "n_responses")}
        except Exception as exc:
            return d, None, f"{type(exc).__name__}: {str(exc)[:160]}", {}

    eligible, ineligible, errors, meta = [], {}, {}, {}
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for d, ok, why, row in ex.map(one, sorted(tables)):
            meta[d] = row
            if ok is None:
                errors[d] = why
            elif ok:
                eligible.append(d)
            else:
                ineligible[d] = why
    if errors and len(errors) == len(tables):
        _explain_gate(RuntimeError(next(iter(errors.values()))))
        raise SystemExit(2)
    # schema report: the (small) subjects tables of the eligible directories and the parquet footers of
    # a few items/responses tables (read through range requests, no full download)
    schema = {"subjects_columns": {}, "subjects_values": {}, "items_columns": {}, "responses_columns": {}}

    def subj(d):
        try:
            tbl = _read_parquet(_fetch(f"{d}/formatted_tables/subjects.parquet", revision, cache))
            return d, tbl
        except Exception:
            return d, None

    from collections import Counter
    col_counter, values = Counter(), {}
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for d, tbl in ex.map(subj, eligible):
            if tbl is None:
                continue
            col_counter.update(tbl.column_names)
            for c in ("harness", "reasoning_effort", "provider", "developer", "organization"):
                if c in tbl.column_names:
                    cnt = values.setdefault(c, Counter())
                    cnt.update(str(v) for v in tbl.column(c).to_pylist())
    schema["subjects_columns"] = dict(col_counter)
    schema["subjects_values"] = {c: dict(cnt.most_common(25)) for c, cnt in values.items()}
    try:
        from huggingface_hub import HfFileSystem
        import pyarrow.parquet as pq
        fs = HfFileSystem()
        for d in eligible[:4]:
            for t in ("items", "responses"):
                with fs.open(f"datasets/{PUBLIC_DATA_REPO}/{d}/formatted_tables/{t}.parquet", "rb") as fh:
                    names = pq.ParquetFile(fh).schema_arrow.names
                schema[f"{t}_columns"][d] = names
    except Exception as exc:
        schema["footer_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    want = [t for t in TABLES if t != "benchmarks" and not (args.skip_items and t == "items")]
    total = sum(tables[d].get(t, 0) for d in eligible for t in want)
    items_bytes = sum(tables[d].get("items", 0) for d in eligible)
    plan = {"repo": PUBLIC_DATA_REPO, "revision": revision, "n_dirs": len(tables), "eligible_dirs": eligible,
            "ineligible": ineligible, "errors": errors, "benchmark_meta": meta, "tables_to_download": want,
            "download_gb": round(total / 1e9, 3), "items_gb": round(items_bytes / 1e9, 3), "max_gb": args.max_gb,
            "within_cap": total / 1e9 <= args.max_gb, "sizes": {d: tables[d] for d in eligible}, "schema": schema}
    (payload / "plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
    log(f"  eligible: {len(eligible)}; ineligible: {len(ineligible)}; errors: {len(errors)}")
    log(f"  subjects columns: {schema['subjects_columns']}")
    log(f"  items columns (sample): {schema['items_columns']}")
    log(f"  responses columns (sample): {schema['responses_columns']}")
    log(f"  to download: {plan['download_gb']} GB ({', '.join(want)}); items alone {plan['items_gb']} GB; cap {args.max_gb} GB")
    if not plan["within_cap"]:
        log("  STOP: the download would exceed the cap. Re-run with --skip-items or a larger --max-gb; "
            "what would be needed is listed in plan.json.")
    return plan


def cmd_download(args, plan: dict | None = None) -> None:
    payload = Path(args.payload).resolve()
    if plan is None:
        plan = json.loads((payload / "plan.json").read_text(encoding="utf-8"))
    if not plan["within_cap"]:
        raise SystemExit("plan exceeds the size cap; nothing downloaded")
    cache = payload / "hf_cache"
    data = payload / "data"
    data.mkdir(parents=True, exist_ok=True)
    revision = plan["revision"]
    want = ["benchmarks"] + list(plan["tables_to_download"])
    files = []

    def one(d):
        got = []
        (data / d).mkdir(parents=True, exist_ok=True)
        for t in want:
            if t not in plan["sizes"][d]:
                continue
            src = _fetch(f"{d}/formatted_tables/{t}.parquet", revision, cache)
            local = "response" if t == "responses" else t
            dst = data / d / f"{local}.parquet"
            if not dst.exists() or dst.stat().st_size != os.path.getsize(src):
                shutil.copyfile(src, dst)
            got.append(f"{d}/{local}.parquet")
        return got

    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        for i, got in enumerate(ex.map(one, plan["eligible_dirs"]), 1):
            files.extend(got)
            if i % 20 == 0 or i == len(plan["eligible_dirs"]):
                log(f"  {i}/{len(plan['eligible_dirs'])} dirs ({time.time() - t0:.0f}s)")
    manifest = {"db_repo": PUBLIC_DATA_REPO, "corpus_scope": CORPUS_SCOPE, "revision": revision,
                "generated_unix": int(time.time()), "eligible_dirs": plan["eligible_dirs"], "files": sorted(files),
                "tables": want, "built_by": "fit_prior_from_measurement_db.py"}
    (data / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    total = sum(p.stat().st_size for p in data.rglob("*.parquet"))
    log(f"  payload {data}: {total / 1e9:.2f} GB on disk")
    # the hf cache holds a second copy of everything; it is safe to delete after copying
    try:
        shutil.rmtree(cache)
    except Exception:
        pass


# ============================================================================= frame
def _first_col(names, candidates):
    for c in candidates:
        if c in names:
            return c
    return None


def _to_label(col):
    """Arrow column -> numpy float labels in [0, 1] (NaN for unparsable)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    if pa.types.is_boolean(col.type):
        return pc.cast(col, pa.float64()).to_numpy(zero_copy_only=False)
    if pa.types.is_integer(col.type) or pa.types.is_floating(col.type):
        arr = pc.cast(col, pa.float64()).to_numpy(zero_copy_only=False)
        return np.clip(arr, 0.0, 1.0)
    vals = col.to_pylist()
    out = np.full(len(vals), np.nan)
    for i, v in enumerate(vals):
        y = model._label_value(v)
        if y is not None:
            out[i] = float(y)
    return out


def load_pairs_from_payload(data_dir: Path, max_items_per_pair: int = MAX_ITEMS_PER_PAIR, seed: int = 0,
                            dirs: list | None = None) -> list:
    """Return a list of pair records built from the payload tables.

    Each record: {dir, subject_id, test_condition, subject (dict), n_items, items: [(item_terms, label)]}
    where item_terms are the (name, value) terms of model.item_terms and label is the mean response.
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    manifest = json.loads((data_dir / "corpus_manifest.json").read_text(encoding="utf-8"))
    dirs = sorted(dirs or manifest["eligible_dirs"])
    rng = random.Random(seed)
    pairs = []
    t0 = time.time()
    for k, d in enumerate(dirs, 1):
        base = data_dir / d
        resp_path = base / "response.parquet"
        if not resp_path.exists():
            continue
        subj_tbl = _read_parquet(base / "subjects.parquet").to_pandas()
        subj_tbl["subject_id"] = subj_tbl["subject_id"].astype(str)
        subjects = {}
        for row in subj_tbl.to_dict("records"):
            subjects[row["subject_id"]] = {c: ("" if row.get(c) is None or (isinstance(row.get(c), float) and math.isnan(row.get(c)))
                                               else str(row.get(c))) for c in SUBJECT_COLS}
        names = pq.ParquetFile(resp_path).schema_arrow.names
        rcol = _first_col(names, RESPONSE_CANDIDATES)
        if rcol is None or "subject_id" not in names or "item_id" not in names:
            log(f"  [{d}] response table lacks subject_id/item_id/response columns ({names}); skipped")
            continue
        cols = ["subject_id", "item_id", rcol] + (["test_condition"] if "test_condition" in names else [])
        resp = _read_parquet(resp_path, columns=cols)
        y = _to_label(resp.column(rcol))
        sid = pc.cast(resp.column("subject_id"), pa.string()).to_numpy(zero_copy_only=False)
        iid = pc.cast(resp.column("item_id"), pa.string()).to_numpy(zero_copy_only=False)
        cond = (pc.fill_null(pc.cast(resp.column("test_condition"), pa.string()), "").to_numpy(zero_copy_only=False)
                if "test_condition" in names else np.array([""] * len(y), dtype=object))
        ok = ~np.isnan(y)
        df = pd.DataFrame({"sid": sid[ok], "iid": iid[ok], "cond": cond[ok], "y": y[ok]})
        agg = df.groupby(["sid", "cond", "iid"], sort=False)["y"].mean().reset_index()
        # item features (length and item_features) from the items table, if present
        item_info = {}
        items_path = base / "items.parquet"
        if items_path.exists():
            inames = pq.ParquetFile(items_path).schema_arrow.names
            ccol, fcol = _first_col(inames, CONTENT_CANDIDATES), _first_col(inames, FEATURE_CANDIDATES)
            icols = ["item_id"] + [c for c in (ccol, fcol) if c]
            it = _read_parquet(items_path, columns=icols)
            ids = pc.cast(it.column("item_id"), pa.string()).to_pylist()
            lengths = (pc.fill_null(pc.utf8_length(pc.cast(it.column(ccol), pa.string())), 0).to_pylist() if ccol
                       else [0] * len(ids))
            feats = (pc.fill_null(pc.cast(it.column(fcol), pa.string()), "").to_pylist() if fcol else [""] * len(ids))
            item_info = {i: (L, f) for i, L, f in zip(ids, lengths, feats)}
        for (s, c), grp in agg.groupby(["sid", "cond"], sort=False):
            if len(grp) < MIN_PAIR_ITEMS_FIT:
                continue
            subject = dict(subjects.get(s, {col: "" for col in SUBJECT_COLS}))
            if c and subject.get("subject_features_extra") is not None:
                subject["subject_features_extra"] = (subject["subject_features_extra"] + ";" if subject["subject_features_extra"] else "") + f"test_condition={c}"
            rows = list(zip(grp["iid"].tolist(), grp["y"].tolist()))
            n_items = len(rows)
            if n_items > max_items_per_pair:
                rows = rng.sample(rows, max_items_per_pair)
            items = []
            for iid_, yy in rows:
                L, f = item_info.get(iid_, (0, ""))
                terms = model.item_terms({"item_features": f}, content_length=L) if item_info else []
                items.append((terms, float(yy)))
            pairs.append({"dir": d, "subject_id": s, "test_condition": c, "subject": subject, "n_items": n_items,
                          "items": items, "has_item_info": bool(item_info)})
        if k % 10 == 0 or k == len(dirs):
            log(f"  {k}/{len(dirs)} dirs, {len(pairs)} pairs so far ({time.time() - t0:.0f}s)")
    return pairs


# ============================================================================= design
def _terms_to_vec(terms, names):
    v = np.zeros(len(names))
    idx = {n: i for i, n in enumerate(names)}
    for name, value in terms:
        if name in idx:
            v[idx[name]] += value
    return v


def build_design(pairs: list, providers: list):
    """Return dict with arrays: Xs (subject terms per row), Xi (item terms), prov (provider index or -1),
    bench (benchmark index), y, w (pair-equal weights), pair index, and the benchmark directory list."""
    pidx = {p: i for i, p in enumerate(providers)}
    dirs = sorted({p["dir"] for p in pairs})
    didx = {d: i for i, d in enumerate(dirs)}
    n_rows = sum(len(p["items"]) for p in pairs)
    Xs = np.zeros((n_rows, len(SUBJECT_NAMES)))
    Xi = np.zeros((n_rows, len(ITEM_NAMES)))
    prov = np.empty(n_rows, dtype=np.int64)
    bench = np.empty(n_rows, dtype=np.int64)
    y = np.empty(n_rows)
    w = np.empty(n_rows)
    pid = np.empty(n_rows, dtype=np.int64)
    pair_rows = []
    iidx = {n: i for i, n in enumerate(ITEM_NAMES)}
    k = 0
    for j, p in enumerate(pairs):
        sv = _terms_to_vec(model.subject_terms(p["subject"]), SUBJECT_NAMES)
        pi = pidx.get(model.provider_key(p["subject"]), -1)
        bi = didx[p["dir"]]
        n = len(p["items"])
        start = k
        for terms, yy in p["items"]:
            Xs[k] = sv
            for name, value in terms:
                c = iidx.get(name)
                if c is not None:
                    Xi[k, c] += value
            prov[k] = pi
            bench[k] = bi
            y[k] = yy
            w[k] = 1.0 / n
            pid[k] = j
            k += 1
        pair_rows.append(np.arange(start, k))
    return {"Xs": Xs, "Xi": Xi, "prov": prov, "bench": bench, "y": y, "w": w, "pid": pid, "dirs": dirs, "pair_rows": pair_rows}


def hand_set_coefficients() -> dict:
    coefs = {n: float(getattr(model, n)) for n in SUBJECT_NAMES + ITEM_NAMES}
    coefs["PRIOR_INTERCEPT"] = float(model.PRIOR_INTERCEPT)
    coefs["PRIOR_PROVIDER"] = dict(model.PRIOR_PROVIDER)
    coefs["SUBJECT_SHRINK"] = float(model.SUBJECT_SHRINK)
    coefs["PRIOR_LOGIT_CAP"] = float(model.PRIOR_LOGIT_CAP)
    return coefs


def logits_from_coefs(D: dict, coefs: dict, providers: list):
    """Subject logit per row and item logit per row under a coefficient table (model.py semantics)."""
    cs = np.array([coefs[n] for n in SUBJECT_NAMES])
    ci = np.array([coefs[n] for n in ITEM_NAMES])
    prov_coef = np.array([coefs["PRIOR_PROVIDER"].get(p, 0.0) for p in providers] + [0.0])
    zs = coefs["SUBJECT_SHRINK"] * (D["Xs"] @ cs + prov_coef[D["prov"]]) + coefs["PRIOR_INTERCEPT"]
    zi = D["Xi"] @ ci
    return zs, zi


def fit_logistic(D: dict, rows: np.ndarray, C: float, n_prov: int):
    """Weighted L2 logistic regression on the selected rows with benchmark fixed effects.

    Benchmark effects are nuisance parameters (test benchmarks are disjoint from the training ones):
    the shipped intercept is the fitted intercept plus the equal-weight mean of the training
    benchmarks' effects, i.e. the prior for a benchmark of unknown difficulty.  Returns the
    coefficient table, the provider coefficients and the benchmark effects.
    """
    from scipy import sparse
    from sklearn.linear_model import LogisticRegression
    n = len(rows)
    pr = D["prov"][rows]
    sel = np.where(pr >= 0)[0]
    Xp = sparse.csr_matrix((np.ones(len(sel)), (sel, pr[sel])), shape=(n, max(n_prov, 1)))
    n_bench = len(D["dirs"])
    Xb = sparse.csr_matrix((np.ones(n), (np.arange(n), D["bench"][rows])), shape=(n, n_bench))
    X = sparse.hstack([sparse.csr_matrix(np.hstack([D["Xs"][rows], D["Xi"][rows]])), Xp, Xb], format="csr")
    y, w = D["y"][rows], D["w"][rows]
    # fractional labels (mean over trials) -> two weighted rows
    frac = (y > 0) & (y < 1)
    X2 = sparse.vstack([X, X[np.where(frac)[0]]], format="csr")
    y2 = np.concatenate([np.where(y >= 0.5, 1.0, 0.0), np.where(y[frac] >= 0.5, 0.0, 1.0)])
    w2 = np.concatenate([np.where(frac, np.where(y >= 0.5, y, 1 - y), 1.0) * w,
                         np.where(y[frac] >= 0.5, 1 - y[frac], y[frac]) * w[frac]])
    w2 = w2 * (len(w2) / w2.sum())
    clf = LogisticRegression(C=C, solver="lbfgs", max_iter=3000, fit_intercept=True)
    clf.fit(X2, y2.astype(int), sample_weight=w2)
    beta = clf.coef_[0]
    ns, ni = len(SUBJECT_NAMES), len(ITEM_NAMES)
    coefs = {n: float(b) for n, b in zip(SUBJECT_NAMES, beta[:ns])}
    coefs.update({n: float(b) for n, b in zip(ITEM_NAMES, beta[ns:ns + ni])})
    prov_beta = beta[ns + ni:ns + ni + max(n_prov, 1)][:n_prov]
    bench_beta = beta[ns + ni + max(n_prov, 1):]
    present = np.unique(D["bench"][rows])
    coefs["PRIOR_INTERCEPT"] = float(clf.intercept_[0] + np.mean(bench_beta[present]))
    coefs["PRIOR_PROVIDER"] = {}
    coefs["SUBJECT_SHRINK"] = 1.0
    coefs["_bench_effect_sd"] = float(np.std(bench_beta[present]))
    return coefs, prov_beta, bench_beta


# ============================================================================= evaluation
def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def pair_metrics(D: dict, pair_ids: np.ndarray, zs: np.ndarray, zi: np.ndarray, cap: float, alpha0: float,
                 min_items_alc: int = MIN_PAIR_ITEMS_ALC, seed: int = 0) -> dict:
    """Budget-0 Brier (all pairs) and the closed-form shrinkage ALC (pairs with >= min_items_alc items).

    The ALC follows model.predict_components' pair path exactly: 50/50 split, 31 random acquisitions,
    prior rate from the acquired items' mean item logit, beta-binomial posterior with alpha0, item
    logit shift on the logit scale; constant 0.5 and the empirical mean are computed on the same split.
    """
    b0, alcs, alc_const, alc_emp, alc_prior = [], [], [], [], []
    for j in pair_ids:
        rows = D["pair_rows"][j]
        y = D["y"][rows]
        z = np.clip(zs[rows] + zi[rows], -cap, cap)
        p0 = _sig(z)
        b0.append(float(np.mean((p0 - y) ** 2)))
        if len(rows) < min_items_alc:
            continue
        rng = np.random.RandomState(seed + int(j))
        perm = rng.permutation(len(rows))
        half = len(rows) // 2
        acq, ev = perm[:half], perm[half:]
        order = acq[rng.permutation(len(acq))][:harness.MAX_LABELS]
        zs_pair = zs[rows][0]
        briers, briers_c, briers_e, briers_p = {}, {}, {}, {}
        for n in harness.BUDGETS:
            if n == 0:
                p = _sig(np.clip(zs_pair + zi[rows][ev], -cap, cap))
                pe = np.full(len(ev), 0.5)
            else:
                got = order[:n]
                k = float(np.sum(y[got]))
                mean_zi = float(np.mean(zi[rows][got]))
                prior_rate = _sig(np.clip(zs_pair + mean_zi, -cap, cap))
                rate_logit = _logit((alpha0 * prior_rate + k) / (alpha0 + n))
                p = _sig(rate_logit + zi[rows][ev] - mean_zi)
                pe = np.full(len(ev), k / n)
            p = np.clip(p, model.CLIP_LO, model.CLIP_HI)
            yy = y[ev]
            briers[n] = float(np.mean((p - yy) ** 2))
            briers_c[n] = float(np.mean((0.5 - yy) ** 2))
            briers_e[n] = float(np.mean((pe - yy) ** 2))
            briers_p[n] = float(np.mean((_sig(np.clip(zs_pair + zi[rows][ev], -cap, cap)) - yy) ** 2))
        alcs.append(harness.alc(briers))
        alc_const.append(harness.alc(briers_c))
        alc_emp.append(harness.alc(briers_e))
        alc_prior.append(harness.alc(briers_p))
    return {"b0": b0, "alc": alcs, "alc_const": alc_const, "alc_emp": alc_emp, "alc_prior": alc_prior}


def _boot(values: np.ndarray, seed: int) -> dict:
    """Percentile bootstrap of the mean (same definition as harness.bootstrap_mean_ci, vectorised)."""
    rng = np.random.RandomState(seed)
    n = len(values)
    idx = rng.randint(0, n, size=(N_BOOT, n))
    boots = np.sort(values[idx].mean(axis=1))
    return {"mean": float(values.mean()), "ci_lo": float(boots[int(0.025 * N_BOOT)]), "ci_hi": float(boots[int(0.975 * N_BOOT) - 1]),
            "n": int(n), "n_boot": N_BOOT}


def _ci(values: list) -> dict:
    if not len(values):
        return {"mean": None, "ci_lo": None, "ci_hi": None, "n": 0}
    return _boot(np.asarray(values, dtype=float), seed=1)


def _paired(a: list, b: list) -> dict:
    if not len(a) or len(a) != len(b):
        return {"mean": None, "ci_lo": None, "ci_hi": None, "significant": None, "n": len(a)}
    diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    res = _boot(diffs, seed=2)
    res["frac_a_better"] = float(np.mean(diffs < 0))
    res["significant"] = bool(res["ci_hi"] < 0 or res["ci_lo"] > 0)
    return res


def pair_rate_diagnostics(D: dict, hand: dict, providers: list) -> dict:
    """Calibration in the large: distribution of pair success rates against the hand-set prior's p0."""
    zs_h, zi_h = logits_from_coefs(D, hand, providers)
    rates, p0s, bench_of_pair = [], [], []
    for j, rows in enumerate(D["pair_rows"]):
        rates.append(float(D["y"][rows].mean()))
        p0s.append(float(_sig(np.clip(zs_h[rows] + zi_h[rows], -hand["PRIOR_LOGIT_CAP"], hand["PRIOR_LOGIT_CAP"])).mean()))
        bench_of_pair.append(int(D["bench"][rows[0]]))
    rates, p0s, bench_of_pair = np.array(rates), np.array(p0s), np.array(bench_of_pair)
    bench_means = np.array([rates[bench_of_pair == b].mean() for b in np.unique(bench_of_pair)])
    deciles = np.percentile(rates, [10, 25, 50, 75, 90])
    return {"pair_rate_mean": float(rates.mean()), "pair_rate_sd": float(rates.std()),
            "pair_rate_percentiles_10_25_50_75_90": [round(float(v), 3) for v in deciles],
            "hand_p0_mean": float(p0s.mean()), "hand_p0_sd": float(p0s.std()),
            "benchmark_rate_sd": float(bench_means.std()), "n_benchmarks": int(len(bench_means)),
            "frac_pairs_rate_above_half": float((rates > 0.5).mean())}


def cross_validate(pairs: list, D: dict, providers: list, hand: dict) -> dict:
    """Benchmark-grouped K-fold CV of the fitted prior against the hand-set prior."""
    dirs = sorted({p["dir"] for p in pairs})
    rng = random.Random(0)
    rng.shuffle(dirs)
    folds = [set(dirs[i::N_FOLDS]) for i in range(N_FOLDS)]
    pair_dir = np.array([p["dir"] for p in pairs])
    n_prov = len(providers)
    results = {}
    # hand-set prior and references are independent of C/cap: compute once over all pairs, and once per
    # fold (test pairs in fold order) so that the fitted metrics can be paired with them
    zs_h, zi_h = logits_from_coefs(D, hand, providers)
    all_pairs = np.arange(len(pairs))
    hand_fold_alc = {}
    for a0 in ALPHA_GRID:
        m = pair_metrics(D, all_pairs, zs_h, zi_h, hand["PRIOR_LOGIT_CAP"], a0)
        results[f"hand_alpha{a0:g}"] = {"b0": _ci(m["b0"]), "alc": _ci(m["alc"]), "alc_const": _ci(m["alc_const"]),
                                       "alc_emp": _ci(m["alc_emp"]), "alc_prior_only": _ci(m["alc_prior"]),
                                       "n_pairs_b0": len(m["b0"]), "n_pairs_alc": len(m["alc"]), "per_pair_alc": m["alc"],
                                       "per_pair_b0": m["b0"]}
        for fi, f in enumerate(folds):
            te = np.where(np.isin(pair_dir, list(f)))[0]
            hand_fold_alc[(fi, a0)] = pair_metrics(D, te, zs_h, zi_h, hand["PRIOR_LOGIT_CAP"], a0)["alc"]
    log(f"  hand-set prior: B0 {results[f'hand_alpha{ALPHA_GRID[0]:g}']['b0']['mean']:.4f}; "
        + "; ".join(f"alpha{a0:g} ALC {results[f'hand_alpha{a0:g}']['alc']['mean']:.4f}" for a0 in ALPHA_GRID)
        + f"; constant ALC {results[f'hand_alpha{ALPHA_GRID[0]:g}']['alc_const']['mean']:.4f}; "
        f"empirical mean ALC {results[f'hand_alpha{ALPHA_GRID[0]:g}']['alc_emp']['mean']:.4f}; "
        f"{results[f'hand_alpha{ALPHA_GRID[0]:g}']['n_pairs_alc']} pairs with >= {MIN_PAIR_ITEMS_ALC} items")
    # intercept-only reference (train-fold global mean, pair-weighted)
    b0_int, alc_int = [], []
    for f in folds:
        tr = np.where(~np.isin(pair_dir, list(f)))[0]
        te = np.where(np.isin(pair_dir, list(f)))[0]
        rows_tr = np.isin(D["pid"], tr)
        mean = float(np.average(D["y"][rows_tr], weights=D["w"][rows_tr]))
        z0 = np.full(len(D["y"]), _logit(np.array([mean]))[0])
        m = pair_metrics(D, te, z0, np.zeros(len(D["y"])), 50.0, 4.0)
        b0_int += m["b0"]
        alc_int += m["alc"]
    results["intercept_only_alpha4"] = {"b0": _ci(b0_int), "alc": _ci(alc_int)}
    # fitted prior for each C and cap
    for C in C_GRID:
        t_c = time.time()
        fitted_pairs_b0 = {cap: [None] * len(pairs) for cap in CAP_GRID}
        fitted_alc = {(cap, a0): [] for cap in CAP_GRID for a0 in ALPHA_GRID}
        hand_alc_match = {(cap, a0): [] for cap in CAP_GRID for a0 in ALPHA_GRID}
        for fi, f in enumerate(folds):
            tr = np.where(~np.isin(pair_dir, list(f)))[0]
            te = np.where(np.isin(pair_dir, list(f)))[0]
            rows_tr = np.where(np.isin(D["pid"], tr))[0]
            coefs, prov_beta, _ = fit_logistic(D, rows_tr, C, n_prov)
            coefs["PRIOR_PROVIDER"] = {p: float(b) for p, b in zip(providers, prov_beta)}
            for cap in CAP_GRID:
                coefs["PRIOR_LOGIT_CAP"] = cap
                zs, zi = logits_from_coefs(D, coefs, providers)
                for a0 in ALPHA_GRID:
                    m = pair_metrics(D, te, zs, zi, cap, a0)
                    fitted_alc[(cap, a0)] += m["alc"]
                    hand_alc_match[(cap, a0)] += hand_fold_alc[(fi, a0)]
                    if a0 == ALPHA_GRID[0]:
                        for j, v in zip(te, m["b0"]):
                            fitted_pairs_b0[cap][j] = v
        for cap in CAP_GRID:
            for a0 in ALPHA_GRID:
                key = f"fit_C{C:g}_cap{cap:g}_alpha{a0:g}"
                paired = _paired(fitted_alc[(cap, a0)], hand_alc_match[(cap, a0)])
                b0_vals = [v for v in fitted_pairs_b0[cap] if v is not None]
                results[key] = {"C": C, "cap": cap, "alpha0": a0, "b0": _ci(b0_vals), "alc": _ci(fitted_alc[(cap, a0)]),
                                "paired_alc_fit_minus_hand": paired,
                                "paired_b0_fit_minus_hand": _paired(b0_vals, [results[f'hand_alpha{a0:g}']['per_pair_b0'][j]
                                                                              for j in range(len(pairs)) if fitted_pairs_b0[cap][j] is not None])}
        log(f"  C={C:g} ({time.time() - t_c:.0f}s): " + "; ".join(
            f"cap{cap:g} B0 {results[f'fit_C{C:g}_cap{cap:g}_alpha{ALPHA_GRID[0]:g}']['b0']['mean']:.4f} "
            + "/".join(f"a{a0:g} ALC {results[f'fit_C{C:g}_cap{cap:g}_alpha{a0:g}']['alc']['mean']:.4f}" for a0 in ALPHA_GRID
                       if results[f'fit_C{C:g}_cap{cap:g}_alpha{a0:g}']['alc']['mean'] is not None)
            for cap in CAP_GRID))
    # selection: regularisation and cap by the budget-0 Brier score of the prior itself (the quantity the
    # fit targets and the one that varies with C); alpha0 by the shrinkage ALC at that setting
    cands = [(results[f"fit_C{C:g}_cap{cap:g}_alpha{ALPHA_GRID[0]:g}"]["b0"]["mean"], C, cap) for C in C_GRID for cap in CAP_GRID]
    cands = [c for c in cands if c[0] is not None]
    best = None
    if cands:
        _, C, cap = min(cands)
        a_scores = [(results[f"fit_C{C:g}_cap{cap:g}_alpha{a0:g}"]["alc"]["mean"], a0) for a0 in ALPHA_GRID]
        a_scores = [s for s in a_scores if s[0] is not None]
        alc_best, a0 = min(a_scores) if a_scores else (None, model.ALPHA0)
        best = {"C": C, "cap": cap, "alpha0": a0, "cv_alc": alc_best,
                "cv_b0": results[f"fit_C{C:g}_cap{cap:g}_alpha{ALPHA_GRID[0]:g}"]["b0"]["mean"]}
    for k in list(results):
        results[k].pop("per_pair_alc", None)
        results[k].pop("per_pair_b0", None)
    results["selected"] = best
    results["folds"] = [sorted(f) for f in folds]
    return results


def cmd_fit(args, pairs: list | None = None) -> dict:
    payload = Path(args.payload).resolve()
    data_dir = payload / "data"
    if pairs is None:
        log(f"Loading pairs from {data_dir} ...")
        pairs = load_pairs_from_payload(data_dir)
    if not pairs:
        raise SystemExit("no pairs with enough items; nothing to fit")
    manifest = json.loads((data_dir / "corpus_manifest.json").read_text(encoding="utf-8"))
    n_rows = sum(len(p["items"]) for p in pairs)
    log(f"  {len(pairs)} pairs, {n_rows} rows, {len({p['dir'] for p in pairs})} benchmarks")
    # providers with enough pairs
    from collections import Counter
    prov_counts = Counter(model.provider_key(p["subject"]) for p in pairs)
    providers = sorted(p for p, c in prov_counts.items() if p and c >= MIN_PROVIDER_PAIRS)
    log(f"  providers kept: {providers}")
    D = build_design(pairs, providers)
    hand = hand_set_coefficients()
    diag = pair_rate_diagnostics(D, hand, providers)
    log(f"  pair rates: mean {diag['pair_rate_mean']:.3f}, sd {diag['pair_rate_sd']:.3f}; hand-set p0 mean {diag['hand_p0_mean']:.3f}, "
        f"sd {diag['hand_p0_sd']:.3f}; benchmark-mean-rate sd {diag['benchmark_rate_sd']:.3f}")
    log("Benchmark-grouped cross-validation ...")
    cv = cross_validate(pairs, D, providers, hand)
    sel = cv["selected"]
    out = {"repo": PUBLIC_DATA_REPO, "revision": manifest.get("revision"), "fitted_on": time.strftime("%Y-%m-%d %H:%M"),
           "n_eligible_dirs": len(manifest.get("eligible_dirs") or []), "min_pair_items_fit": MIN_PAIR_ITEMS_FIT,
           "max_items_per_pair": MAX_ITEMS_PER_PAIR, "pair_unit": "(benchmark directory, subject_id, test_condition)",
           "n_benchmarks": len({p["dir"] for p in pairs}), "n_pairs": len(pairs), "n_rows": n_rows,
           "n_pairs_alc_eligible": sum(1 for p in pairs if len(p["items"]) >= MIN_PAIR_ITEMS_ALC),
           "item_tables_used": all(p["has_item_info"] for p in pairs), "providers": providers,
           "provider_pair_counts": dict(prov_counts.most_common(40)), "hand_set": hand, "diagnostics": diag, "cv": cv,
           "data_sources": sorted({p["dir"] for p in pairs})}
    if sel is not None:
        coefs, prov_beta, bench_beta = fit_logistic(D, np.arange(len(D["y"])), sel["C"], len(providers))
        coefs["PRIOR_PROVIDER"] = {p: round(float(b), 4) for p, b in zip(providers, prov_beta)}
        coefs["PRIOR_LOGIT_CAP"] = sel["cap"]
        out["benchmark_effects"] = {d: round(float(b), 4) for d, b in zip(D["dirs"], bench_beta)}
        out["benchmark_effect_sd"] = round(coefs.pop("_bench_effect_sd"), 4)
        coefs = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in coefs.items()}
        if not out["item_tables_used"]:
            for n in ITEM_NAMES:
                coefs[n] = hand[n]
        out["fitted"] = coefs
        key = f"fit_C{sel['C']:g}_cap{sel['cap']:g}_alpha{sel['alpha0']:g}"
        d = cv[key]["paired_alc_fit_minus_hand"]
        d0 = cv[key]["paired_b0_fit_minus_hand"]
        out["cv_better_than_hand"] = bool(d["mean"] is not None and d["mean"] < 0)
        out["cv_significantly_better"] = bool(d["mean"] is not None and d["mean"] < 0 and d["significant"])
        out["recommended_alpha0"] = sel["alpha0"]
        log(f"  selected C={sel['C']:g} cap={sel['cap']:g} alpha0={sel['alpha0']:g}: CV B0 {sel['cv_b0']:.4f} "
            f"(fit - hand {d0['mean']:+.4f} [{d0['ci_lo']:+.4f}, {d0['ci_hi']:+.4f}]); CV ALC {sel['cv_alc']:.4f} "
            f"(fit - hand {d['mean']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]); benchmark effect sd {out['benchmark_effect_sd']}")
        log("  fitted coefficients: " + json.dumps(coefs))
    out_path = Path(args.out) if args.out else RESULTS / "prior_fit.json"
    out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"wrote {out_path}")
    return out


# ============================================================================= testbeds
def cmd_testbeds(args) -> dict:
    import run_experiments as R
    fit_path = Path(args.fit) if args.fit else RESULTS / "prior_fit.json"
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    if "fitted" not in fit:
        raise SystemExit("prior_fit.json has no fitted coefficients")
    params = {k: v for k, v in fit["fitted"].items()}
    out = {"fit_file": str(fit_path), "params": params, "testbeds": {}}
    datasets = {}
    import real_testbed
    raw_real = real_testbed.load_pairs()
    if raw_real:
        datasets["real"] = raw_real
    if not args.skip_synthetic:
        import simulate
        datasets["synthetic"] = simulate.generate_pairs(0, R.N_SUBJECTS, R.N_BENCHMARKS, R.MAIN_SIGMA, R.MAIN_GAMMA)
    for name, raw in datasets.items():
        s_hand = R.summarize(R.run_dataset(raw, seed=0, workers=args.workers))
        s_fit = R.summarize(R.run_dataset(raw, seed=0, params=params, workers=args.workers))
        rec = {}
        for m in ("prior", "shrink", "full"):
            rec[m] = {"hand": s_hand["methods"][m]["alc"], "fitted": s_fit["methods"][m]["alc"],
                      "paired_fit_minus_hand": harness.paired_bootstrap(s_fit["methods"][m]["per_pair_alc"],
                                                                        s_hand["methods"][m]["per_pair_alc"], N_BOOT, seed=3)}
        out["testbeds"][name] = rec
        R.headline(s_hand, f"{name} hand-set"), R.headline(s_fit, f"{name} fitted")
    real_ok = ("real" in out["testbeds"] and out["testbeds"]["real"]["full"]["paired_fit_minus_hand"]["mean"] < 0)
    out["ship_rule"] = {"cv_better_than_hand": fit.get("cv_better_than_hand"), "real_testbed_full_better": real_ok,
                        "ship": bool(fit.get("cv_better_than_hand") and real_ok)}
    path = RESULTS / "prior_fit_testbeds.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"wrote {path}; ship rule: {out['ship_rule']}")
    return out


# ============================================================================= dry run
TRUE_COEFS = {"PRIOR_INTERCEPT": -0.3, "PRIOR_YEAR_SLOPE": 0.5, "PRIOR_EFFORT_LOW": 0.1, "PRIOR_EFFORT_MID": 0.3,
              "PRIOR_EFFORT_HIGH": 0.6, "PRIOR_SIZE_SLOPE": 0.2, "PRIOR_SMALL_NAME": -0.4, "PRIOR_LARGE_NAME": 0.3,
              "PRIOR_HARNESS": 0.2, "PRIOR_TIER_HARD": -0.8, "PRIOR_TIER_EASY": 0.6, "PRIOR_TIER_NUM": 0.3,
              "PRIOR_POINTS": 0.15, "PRIOR_LENGTH": 0.2,
              "PRIOR_PROVIDER": {"acme ai": 0.4, "borealis": -0.2, "cerulean labs": 0.0, "delta systems": 0.1}}


def make_synthetic_payload(tmp: Path, n_dirs: int = 16, n_subjects: int = 60, seed: int = 0) -> tuple:
    """Synthetic tables with the measurement-db schema (columns as read by the organisers' code).

    A shared pool of subjects is evaluated on random subsets of the benchmarks (as in the release,
    where the same systems recur across benchmark directories); each benchmark has its own
    difficulty offset; labels follow a logistic model with TRUE_COEFS over the prior's own terms.
    Returns (data_dir, {dir: benchmark effect}).
    """
    import pandas as pd
    rng = random.Random(seed)
    data = tmp / "data"
    if data.exists():
        shutil.rmtree(data)
    data.mkdir(parents=True)
    providers = list(TRUE_COEFS["PRIOR_PROVIDER"])
    pool = []
    for s in range(n_subjects):
        year = rng.uniform(2023.0, 2026.5)
        size = rng.choice([None, None, 7, 32, 70, 400])
        name = f"model-{s:02d}" + (f"-{size}b" if size else rng.choice(["", "-mini", "-pro", ""]))
        pool.append({"subject_id": f"subj_{s:03d}", "normalized_name": name, "display_name": name,
                     "provider": rng.choice(providers), "release_date": f"{int(year):04d}-{int((year % 1) * 12) + 1:02d}-15",
                     "access_date": "2026-09-01", "harness": rng.choice(["default", "default", "scaffold-v1"]),
                     "harness_version": "", "reasoning_effort": rng.choice(["", "", "low", "medium", "high"]),
                     "subject_features_extra": ""})
    eligible, files, effects = [], [], {}
    for b in range(n_dirs):
        d = f"demo_bench_{b:02d}"
        (data / d).mkdir()
        ineligible = b % 7 == 6
        bench = pd.DataFrame([{"benchmark_id": d, "name": f"Synthetic benchmark {b}", "description": "dry-run tables; not measured results",
                               "domain": ["math"], "modality": ["text"], "release_date": f"202{4 + b % 3}-0{1 + b % 9}-01",
                               "n_items": 0, "n_subjects": 0, "response_type": "continuous" if ineligible and b % 2 == 0 else "binary",
                               "granularity": "aggregate" if ineligible and b % 2 == 1 else "item", "test_conditions": [""],
                               "n_trials": 1, "n_responses": 0, "saturation": None, "has_ground_truth": True}])
        n_items = rng.randint(90, 220)
        subjects = rng.sample(pool, rng.randint(12, 30))
        bench_effect = rng.gauss(0, 1.0)
        effects[d] = bench_effect
        items, responses = [], []
        for i in range(n_items):
            L = int(rng.lognormvariate(math.log(400), 0.7))
            feats = rng.choice(["", "", f"tier={rng.choice(['easy', 'medium', 'hard'])}", f"level={rng.randint(1, 5)}",
                                f"points={rng.randint(1, 6)}", "difficulty is advanced"])
            items.append({"item_id": str(i), "content": "w" * L, "item_features": feats, "interactors": "",
                          "correct_answer": "42"})
        for subj in subjects:
            zs = TRUE_COEFS["PRIOR_INTERCEPT"] + TRUE_COEFS["PRIOR_PROVIDER"][subj["provider"]] + sum(
                TRUE_COEFS[n] * v for n, v in model.subject_terms(subj)) + bench_effect
            for it in items:
                zi = sum(TRUE_COEFS[n] * v for n, v in model.item_terms({"item_content": it["content"], "item_features": it["item_features"]}))
                p = 1.0 / (1.0 + math.exp(-(zs + zi + rng.gauss(0, 0.3))))
                n_trials = 2 if b % 3 == 0 else 1
                for t in range(n_trials):
                    responses.append({"subject_id": subj["subject_id"], "item_id": int(it["item_id"]) if b % 2 else it["item_id"],
                                      "response": 1.0 if rng.random() < p else 0.0, "test_condition": "", "trial": t})
        bench.loc[0, "n_items"], bench.loc[0, "n_subjects"], bench.loc[0, "n_responses"] = n_items, len(subjects), len(responses)
        bench.to_parquet(data / d / "benchmarks.parquet", index=False)
        pd.DataFrame(subjects).to_parquet(data / d / "subjects.parquet", index=False)
        pd.DataFrame(items).to_parquet(data / d / "items.parquet", index=False)
        pd.DataFrame(responses).to_parquet(data / d / "response.parquet", index=False)
        if not ineligible:
            eligible.append(d)
            files += [f"{d}/{t}.parquet" for t in ("benchmarks", "subjects", "items", "response")]
    manifest = {"db_repo": PUBLIC_DATA_REPO, "corpus_scope": CORPUS_SCOPE, "revision": "0" * 40, "generated_unix": int(time.time()),
                "eligible_dirs": eligible, "files": files, "synthetic_dry_run": True}
    (data / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return data, effects


def cmd_dry_run(args) -> dict:
    tmp = Path(args.tmp) if args.tmp else RESULTS / "prior_fit_dry_run_payload"
    tmp.mkdir(parents=True, exist_ok=True)
    log(f"Building synthetic payload in {tmp} ...")
    data, effects = make_synthetic_payload(tmp)
    # eligibility from the benchmarks tables (as `plan` would do from the downloaded files)
    import pyarrow.parquet as pq
    resolved = []
    for d in sorted(p.name for p in data.iterdir() if p.is_dir()):
        row = pq.read_table(data / d / "benchmarks.parquet").to_pylist()[0]
        ok, why = _eligible(row)
        if ok:
            resolved.append(d)
    manifest = json.loads((data / "corpus_manifest.json").read_text(encoding="utf-8"))
    assert resolved == manifest["eligible_dirs"], (resolved, manifest["eligible_dirs"])
    ns = argparse.Namespace(payload=str(tmp), out=str(RESULTS / "prior_fit_dry_run.json"))
    pairs = load_pairs_from_payload(data)
    out = cmd_fit(ns, pairs)
    # identifiability check against the generating coefficients: a light-regularisation fit (C = 1) on all
    # rows.  (The cross-validated selection above shrinks the coefficients towards zero when benchmark
    # difficulty dominates the labels, which is a prediction question, not an identifiability one.)
    rec = {}
    if "fitted" in out:
        D = build_design(pairs, out["providers"])
        coefs, prov_beta, _ = fit_logistic(D, np.arange(len(D["y"])), 1.0, len(out["providers"]))
        coefs["PRIOR_PROVIDER"] = {p: float(b) for p, b in zip(out["providers"], prov_beta)}
        for n in SUBJECT_NAMES + ITEM_NAMES:
            rec[n] = {"true": TRUE_COEFS[n], "fitted_C1": round(coefs[n], 4), "fitted_selected": out["fitted"][n], "hand_set": out["hand_set"][n]}
        mean_effect = sum(effects[d] for d in manifest["eligible_dirs"]) / len(manifest["eligible_dirs"])
        # the provider effects are identified only up to a constant shared with the intercept
        prov_true = TRUE_COEFS["PRIOR_PROVIDER"]
        prov_fit = coefs["PRIOR_PROVIDER"]
        shift = (sum(prov_fit.get(p, 0.0) for p in prov_true) - sum(prov_true.values())) / len(prov_true)
        rec["PRIOR_INTERCEPT"] = {"true_plus_mean_benchmark_effect": round(TRUE_COEFS["PRIOR_INTERCEPT"] + mean_effect, 4),
                                  "fitted_C1_minus_provider_shift": round(coefs["PRIOR_INTERCEPT"] + shift, 4),
                                  "fitted_C1": round(coefs["PRIOR_INTERCEPT"], 4), "fitted_selected": out["fitted"]["PRIOR_INTERCEPT"]}
        rec["PRIOR_PROVIDER"] = {p: {"true": prov_true[p], "fitted_C1": round(prov_fit.get(p, 0.0), 4),
                                     "fitted_C1_centred": round(prov_fit.get(p, 0.0) - shift, 4)} for p in prov_true}
        slopes = [n for n in SUBJECT_NAMES + ITEM_NAMES]
        errs = [abs(rec[n]["fitted_C1"] - rec[n]["true"]) for n in slopes]
        out["dry_run_recovery"] = {"recovery_fit_C": 1.0, "coefficients": rec, "max_abs_error_slopes": round(max(errs), 4),
                                   "mean_abs_error_slopes": round(sum(errs) / len(errs), 4),
                                   "sign_agreement_slopes": sum(1 for n in slopes if (rec[n]["fitted_C1"] > 0) == (rec[n]["true"] > 0)),
                                   "n_slopes": len(slopes), "generator_benchmark_effect_sd": 1.0,
                                   "generator_mean_benchmark_effect": round(mean_effect, 4),
                                   "intercept_abs_error": round(abs(rec["PRIOR_INTERCEPT"]["fitted_C1_minus_provider_shift"]
                                                                   - rec["PRIOR_INTERCEPT"]["true_plus_mean_benchmark_effect"]), 4)}
        log(f"  recovery (C=1): max |fitted - true| over slopes {out['dry_run_recovery']['max_abs_error_slopes']}, "
            f"mean {out['dry_run_recovery']['mean_abs_error_slopes']}, sign agreement {out['dry_run_recovery']['sign_agreement_slopes']}/{len(slopes)}; "
            f"intercept {rec['PRIOR_INTERCEPT']}")
    out["synthetic_dry_run"] = True
    (RESULTS / "prior_fit_dry_run.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    if not args.keep_tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


# ============================================================================= main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["plan", "download", "fit", "testbeds", "all", "dry-run"])
    ap.add_argument("--payload", default=str(HERE.parents[3] / "_downloads" / "measurement_db_payload"),
                    help="directory for the downloaded tables (kept outside the repository)")
    ap.add_argument("--max-gb", type=float, default=30.0)
    ap.add_argument("--skip-items", action="store_true", help="do not download items.parquet (item coefficients stay hand-set)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fit", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--skip-synthetic", action="store_true")
    ap.add_argument("--tmp", default=None)
    ap.add_argument("--keep-tmp", action="store_true")
    args = ap.parse_args()
    if args.command == "plan":
        cmd_plan(args)
    elif args.command == "download":
        cmd_download(args)
    elif args.command == "fit":
        cmd_fit(args)
    elif args.command == "testbeds":
        cmd_testbeds(args)
    elif args.command == "dry-run":
        cmd_dry_run(args)
    elif args.command == "all":
        plan = cmd_plan(args)
        if not plan["within_cap"]:
            raise SystemExit(3)
        cmd_download(args, plan)
        out = cmd_fit(args)
        if "fitted" in out:
            cmd_testbeds(args)


if __name__ == "__main__":
    main()
