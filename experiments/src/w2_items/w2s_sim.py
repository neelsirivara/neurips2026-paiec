"""W2-sem: the W2 layer with a semantic (sentence-embedding) kernel, evaluated for many settings at once.

Setting keys:
  tau_f, share       category effects as in W2 (w2_sim.fit_w2)
  kern               "none" | "char" | "sem" | "both"
  q_c                exponent of the char-4-gram TF-IDF cosine (W2: 2)
  q_s                exponent of the sentence-embedding cosine (clipped at 0)
  tau_k              prior sd of the kernel effect: delta = sum K G / (sum K H + 1 / tau_k^2)
  topk               None or k: only the k labelled items with the largest K enter the sum (ties kept)
  m_low              multiplier on tau_k when the list's budget (median labels per pair) is <= 3
With kern "both", K = 0.5 * cos_c^q_c + 0.5 * cos_s^q_s.  kern "char" with q_c 2, topk None, m_low 1 is exactly W2
(w2_sim.eval_w2).  Per budget, the stage-1 fits are computed once per distinct tau_f and the per-block similarity
matrices once, then every setting is evaluated on them.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w2_sim import *  # noqa: F401,F403
from w2_sim import X, model, np, json, time, log, CACHE, WF, World, fit_w2, base_taus, BASE_ZS  # noqa: E402


def attach_embeddings(W, tag):
    """W.Emb: float32 (n_rows, 384) aligned with W.row_of (rows of w2_items_all.jsonl); zero rows = no embedding."""
    z = np.load(CACHE / f"w2_emb_{tag}_all.npz")
    W.Emb = np.zeros((W.G.shape[0], z["vecs"].shape[1]), dtype=np.float32)
    rows = W.row_of[z["idx"]]
    ok = rows >= 0
    W.Emb[rows[ok]] = z["vecs"][ok].astype(np.float32)
    W.emb_tag = tag
    log(f"embeddings {tag}: {int(ok.sum())} rows attached")


def _blocks(W, E, V, need_char, need_sem):
    """Per (ts, bench) block of V: target rows, labelled (block, item) aggregate indices, similarity matrices."""
    import scipy.sparse as sp
    NI1 = len(W.F.item_dir) + 1
    blkE = E["ts"] * 200 + E["bench"]
    blkV = V["ts"] * 200 + V["bench"]
    keyE = blkE * NI1 + E["item"]
    uk, invE = np.unique(keyE, return_inverse=True)
    ublk, uit = uk // NI1, uk % NI1
    orderV = np.argsort(blkV, kind="stable")
    ub_V, startV = np.unique(blkV[orderV], return_index=True)
    endV = list(startV[1:]) + [len(orderV)]
    lab_start = np.searchsorted(ublk, ub_V, side="left")
    lab_end = np.searchsorted(ublk, ub_V, side="right")
    blocks = []
    for j, b in enumerate(ub_V):
        a0, a1 = lab_start[j], lab_end[j]
        if a1 <= a0:
            continue
        rowsV = orderV[startV[j]:endV[j]]
        Lsel = np.arange(a0, a1)
        Litems = uit[Lsel]
        Lrows = W.row_of[Litems]
        okL = Lrows >= 0
        if not np.any(okL):
            continue
        Lsel, Litems, Lrows = Lsel[okL], Litems[okL], Lrows[okL]
        Tit, Tinv = np.unique(V["item"][rowsV], return_inverse=True)
        Trows = W.row_of[Tit]
        okT = Trows >= 0
        same = Tit[:, None] == Litems[None, :]
        blk = {"rowsV": rowsV, "Lsel": Lsel, "Tinv": Tinv}
        if need_char:
            RL = W.G[Lrows]
            n = RL.shape[0]
            cols_u, cinv = np.unique(RL.indices, return_inverse=True)
            df_u = np.bincount(cinv, minlength=len(cols_u)).astype(np.float64)
            XL = sp.csr_matrix((RL.data * (np.log((n + 1.0) / (df_u[cinv] + 1.0)) + 1.0), cinv, RL.indptr), shape=(n, len(cols_u)))
            nl = np.sqrt(np.asarray(XL.multiply(XL).sum(1)).ravel())
            nl[nl == 0] = 1.0
            XL = sp.diags(1.0 / nl) @ XL
            RT = W.G[np.where(okT, Trows, 0)]
            pos = np.searchsorted(cols_u, RT.indices)
            pos_c = np.clip(pos, 0, max(len(cols_u) - 1, 0))
            shared = (cols_u[pos_c] == RT.indices) if len(cols_u) else np.zeros(len(RT.indices), bool)
            dfT = np.where(shared, df_u[pos_c] if len(cols_u) else 0.0, 0.0)
            wT = RT.data * (np.log((n + 1.0) / (dfT + 1.0)) + 1.0)
            rowT = np.repeat(np.arange(RT.shape[0]), np.diff(RT.indptr))
            nt = np.sqrt(np.bincount(rowT, weights=wT * wT, minlength=RT.shape[0]))
            nt[nt == 0] = 1.0
            XT = sp.csr_matrix(((wT / nt[rowT])[shared], (rowT[shared], pos_c[shared])), shape=(RT.shape[0], len(cols_u)))
            Sc = np.clip(np.asarray((XT @ XL.T).todense()), 0.0, 1.0).astype(np.float32)
            Sc[~okT] = 0.0
            Sc[same] = 0.0
            blk["Sc"] = Sc
        if need_sem:
            Ss = np.clip(W.Emb[np.where(okT, Trows, 0)] @ W.Emb[Lrows].T, 0.0, 1.0)
            Ss[~okT] = 0.0
            Ss[same] = 0.0
            blk["Ss"] = Ss
        blocks.append(blk)
    return blocks, invE, len(uk)


def _kernel_numden(blocks, Gs, Hs, nV, s):
    """Per V row: sum_j K G_j and sum_j K H_j (0 where the row has no labelled neighbours)."""
    num = np.zeros(nV)
    den = np.zeros(nV)
    kern = s["kern"]
    for blk in blocks:
        if kern == "char":
            K = blk["Sc"] ** s["q_c"]
        elif kern == "sem":
            K = blk["Ss"] ** s["q_s"]
        else:
            K = 0.5 * blk["Sc"] ** s["q_c"] + 0.5 * blk["Ss"] ** s["q_s"]
        k = s.get("topk")
        if k is not None and K.shape[1] > k:
            thr = -np.partition(-K, k - 1, axis=1)[:, k - 1:k]
            K = np.where(K >= thr, K, 0.0)
        L = blk["Lsel"]
        a = K @ Gs[L]
        b = K @ Hs[L]
        num[blk["rowsV"]] = a[blk["Tinv"]]
        den[blk["rowsV"]] = b[blk["Tinv"]]
    return num, den


def eval_grid(W, S, ts, settings, preds_out=None):
    """{label: per-unit Brier (n_units x 6)} for every setting (label = setting['label']); with preds_out (a dict),
    also the row predictions preds_out[label][budget]."""
    taus = base_taus()

    def off(T):
        return np.clip(BASE_ZS + T["zi"], -X.CAP, X.CAP)

    V = dict(S.ev)
    V["o"] = off(S.ev)
    p0 = np.clip(X._sig(V["o"]), model.CLIP_LO, model.CLIP_HI)
    b0 = X.unit_briers(S, p0)
    out = {s["label"]: np.repeat(b0[:, None], 6, axis=1) for s in settings}
    need_char = any(s["kern"] in ("char", "both") for s in settings)
    need_sem = any(s["kern"] in ("sem", "both") for s in settings)
    for bi, b in enumerate(X.BUDGETS[1:], start=1):
        E = dict(S.acq[b])
        E["o"] = off(S.acq[b])
        low = b <= 3
        fits = {}
        for s in settings:
            tf = s["tau_f"]
            key = (round(tf, 6), s.get("share", 1.0))
            if key not in fits:
                fit = fit_w2(W, E, V, taus, tau_f=tf, share=s.get("share", 1.0))
                p = X.xp_predict_np(E, V, fit, SHIPPED_TI, True)
                pe = X._sig(fit["eta"])
                fits[key] = (p, pe)
        blocks = invE = nk = None
        if any(s["kern"] != "none" for s in settings):
            blocks, invE, nk = _blocks(W, E, V, need_char, need_sem)
        nd_cache = {}

        def _gk(s_):
            return (round(s_["tau_f"], 6), s_.get("share", 1.0), s_["kern"], s_.get("q_c") or 0, s_.get("q_s") or 0,
                    s_.get("topk") or 0)
        for s in sorted(settings, key=_gk):      # one kernel group at a time (memory)
            fk = (round(s["tau_f"], 6), s.get("share", 1.0))
            p, pe = fits[fk]
            if s["kern"] != "none" and s.get("tau_k", 0) > 0:
                tk = s["tau_k"] * (s.get("m_low", 1.0) if low else 1.0)
                gk = fk + (s["kern"], s.get("q_c"), s.get("q_s"), s.get("topk"))
                if gk not in nd_cache:
                    nd_cache.clear()
                    Gs = np.bincount(invE, weights=E["y"] - pe, minlength=nk)
                    Hs = np.bincount(invE, weights=pe * (1 - pe), minlength=nk)
                    nd_cache[gk] = _kernel_numden(blocks, Gs, Hs, len(V["ts"]), s)
                num, den = nd_cache[gk]
                d = num / (den + 1.0 / tk ** 2)
                p = np.clip(X._sig(X._logit(p) + d), model.CLIP_LO, model.CLIP_HI)
            out[s["label"]][:, bi] = X.unit_briers(S, p)
            if preds_out is not None:
                preds_out.setdefault(s["label"], {})[b] = p
        del nd_cache
        del blocks
    return out


def slabel(s):
    parts = [f"F{s['tau_f']:g}"]
    if s["kern"] != "none":
        if s["kern"] == "char":
            k = f"C{s['q_c']:g}"
        elif s["kern"] == "sem":
            k = f"S{s['q_s']:g}"
        else:
            k = f"C{s['q_c']:g}S{s['q_s']:g}"
        parts.append(f"K{s['tau_k']:g}{k}")
        if s.get("topk"):
            parts.append(f"top{s['topk']}")
    if s.get("m_low", 1.0) != 1.0:
        parts.append(f"low{s['m_low']:g}")
    return "+".join(parts)
