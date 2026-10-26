"""Count distinct items (frame item ids) that the item-rating gate needs per scenario."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dd_common import *
F, folds = load()
rate, pyear = pair_stats(F)
elig = F.pair_n >= 80
NB = int(F.pair_dir.max()) + 1
cnt = np.bincount(F.pair_dir[elig], minlength=NB)
bm = np.bincount(F.pair_dir[elig], weights=rate[elig], minlength=NB) / np.maximum(cnt, 1)
inv = inventory_dirs(F)
mid = {F.dirs[b] for b in range(len(F.dirs)) if cnt[b] > 0 and bm[b] <= 0.50}
sub = lambda keep: [[d for d in f if d in keep] for f in folds]
allitems = set()
for name, fl in (("INV", sub(inv)), ("MID", sub(mid)), ("A", folds)):
    ts = X.make_testsets(F, fl, "A", R=300)
    S = X.Streams(F, ts)
    its = set(np.unique(np.concatenate([S.ev["item"], S.acq[31]["item"]])).tolist())
    allitems |= its
    nb = len(np.unique(F.item_dir[list(its)]))
    print(name, "distinct items", len(its), "benchmarks", nb, "cumulative", len(allitems))
