"""Audit 3: re-pick each strategy's wine from markets.csv + saved test predictions; recompute summary and CIs."""
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "audit" / "results"
P = pd.read_csv(ROOT / "outputs" / "models" / "test_predictions.csv").set_index("source_row_id")
M = pd.read_csv(ROOT / "outputs" / "sim" / "markets.csv")
C = pd.read_csv(ROOT / "outputs" / "sim" / "market_choices.csv")
clean = pd.read_csv(ROOT / "data" / "processed" / "wine_clean.csv", usecols=["source_row_id", "split"]).set_index("source_row_id")
res = {"checks": {}, "by_budget": {}}
assert P.index.is_unique
res["checks"]["all_market_rows_in_test"] = bool((clean.loc[M.source_row_id, "split"] == "test").all())

for B, g in M.groupby("budget"):
    g = g.sort_values(["market_id", "slot"])
    J = g.join(P, on="source_row_id")
    nm = J.market_id.nunique()
    sizes = J.groupby("market_id").size()
    distinct = J.groupby("market_id")["title_norm"].nunique()
    chk = {"n_markets": int(nm), "all_size_20": bool((sizes == 20).all()), "all_20_distinct_titles": bool((distinct == 20).all()),
           "all_within_budget": bool((J.price <= B).all()), "max_price_seen": float(J.price.max()),
           "pool_unique_titles_used": int(J.title_norm.nunique())}
    strat = {"Random": [], "MaxPrice": [], "Ridge": [], "Main": []}
    ties = {"MaxPrice": 0, "Ridge": 0, "Main": 0}
    saved = C[C.budget == B].set_index(["strategy", "market_id"])
    agree = {"MaxPrice": 0, "Ridge": 0, "Main": 0}
    picked_vals = {"MaxPrice": [], "Ridge": [], "Main": []}
    for mid, mm in J.groupby("market_id"):
        pts, pr = mm.points.to_numpy(float), mm.price.to_numpy(float)
        strat["Random"].append((pts.mean(), (pts >= 90).mean(), pr.mean()))
        cand = {"MaxPrice": pr == pr.max()}
        for s, col in (("Ridge", "pred_ridge_meta_price"), ("Main", "pred_xgb_meta_price")):
            p = mm[col].to_numpy(float)
            top = p == p.max()
            top &= pr == pr[top].min()
            cand[s] = top
        for s, msk in cand.items():
            ties[s] += int(msk.sum() > 1)
            # expected value under a uniform random tie-break
            strat[s].append((pts[msk].mean(), (pts[msk] >= 90).mean(), pr[msk].mean()))
            sv = saved.loc[(s, mid)]
            ok = int(sv.source_row_id) in set(mm.source_row_id[msk])
            agree[s] += ok
            picked_vals[s].append((sv.points, sv.ge90, sv.price))
    out = {"checks": chk, "markets_with_ties": ties, "saved_pick_in_my_candidate_set": agree, "strategies": {}}
    for s, v in strat.items():
        a = np.array(v)
        out["strategies"][s] = {"mean_points_EV": a[:, 0].mean(), "share_ge90_pct_EV": 100 * a[:, 1].mean(), "mean_price_EV": a[:, 2].mean()}
        if s != "Random":
            b = np.array(picked_vals[s], float)
            out["strategies"][s].update({"mean_points_saved_picks": b[:, 0].mean(), "share_ge90_pct_saved_picks": 100 * b[:, 1].mean(),
                                         "mean_price_saved_picks": b[:, 2].mean()})
    # paired bootstrap with my own seed, using the saved (realised) picks so the point estimate is identical
    rnd = np.array(strat["Random"])
    arr = {s: np.array(picked_vals[s], float) for s in picked_vals}
    rng = np.random.default_rng(12345 + B)
    idx = rng.integers(0, nm, size=(10000, nm))
    cis = {}
    for a_, b_ in (("Main", "MaxPrice"), ("Ridge", "MaxPrice"), ("Main", "Ridge")):
        for k, (j, sc) in {"ge90_pctp": (1, 100), "points": (0, 1), "price_usd": (2, 1)}.items():
            d = sc * (arr[a_][:, j] - arr[b_][:, j])
            bs = d[idx].mean(axis=1)
            cis[f"{a_}-{b_}__{k}"] = {"diff": d.mean(), "ci95": [np.percentile(bs, 2.5), np.percentile(bs, 97.5)]}
    out["bootstrap_own_seed"] = cis
    res["by_budget"][int(B)] = out

def conv(o):
    if isinstance(o, dict): return {str(k): conv(v) for k, v in o.items()}
    if isinstance(o, list): return [conv(v) for v in o]
    if isinstance(o, (np.floating, float)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    return o
res = conv(res)
json.dump(res, open(OUT / "a3_sim.json", "w"), indent=2)
print(json.dumps(res, indent=1))
