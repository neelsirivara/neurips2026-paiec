"""Shared helpers for the W2 item-effect workstream (reuses xpair_sim.py / deep_dive/dd_common.py unchanged)."""
import os
import sys
os.environ.setdefault("PAIEC_FIT_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "deep_dive"))
from dd_common import *  # noqa: F401,F403,E402
from dd_common import X, model, np, json, time, Path, log  # noqa: E402
import w2_feat as WF  # noqa: E402

CACHE = Path(X.DEFAULT_PAYLOAD).resolve() / "round4_cache"
RES = Path(HERE).parents[1] / "results" / "w2_items"
RES.mkdir(parents=True, exist_ok=True)
parse_features = WF.parse_features


def char_tfidf_matrix(texts, idf_from=None):
    """Row-normalised TF-IDF matrix (scipy CSR) of hashed char n-grams; idf over ``idf_from`` rows (default all)."""
    import scipy.sparse as sp
    rows, cols, vals = [], [], []
    for r, t in enumerate(texts):
        g = WF.text_grams(t)
        rows += [r] * len(g)
        cols += list(g.keys())
        vals += list(g.values())
    M = sp.csr_matrix((vals, (rows, cols)), shape=(len(texts), 1 << WF.GRAM_BITS))
    ref = M if idf_from is None else M[idf_from]
    df = np.asarray((ref > 0).sum(0)).ravel()
    n = ref.shape[0]
    idf = np.log((n + 1.0) / (df + 1.0)) + 1.0
    M = M @ sp.diags(idf)
    nr = np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel())
    nr[nr == 0] = 1.0
    return sp.diags(1.0 / nr) @ M
