"""Audit 1: re-derive cleaning counts from raw CSV, check split sizes and leakage. Independent code."""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "audit" / "results"; OUT.mkdir(parents=True, exist_ok=True)
K = json.loads((ROOT / "outputs" / "key_numbers.json").read_text(encoding="utf-8"))
res = {}

raw = pd.read_csv(ROOT / "data" / "raw" / "winemag-data-130k-v2.csv")
res["raw_rows"] = len(raw); res["raw_cols"] = raw.shape[1]

# (b) exact duplicates ignoring the row-id column
c = [x for x in raw.columns if x != "Unnamed: 0"]
d1 = raw[~raw.duplicated(subset=c, keep="first")]
res["after_exact_dup"] = len(d1)

# (c) required fields
def ok_str(s):
    return s.notna() & (s.fillna("").astype(str).str.strip() != "")
mask = d1["price"].notna() & (d1["price"] > 0) & d1["points"].between(80, 100)
for col in ["country", "province", "variety", "title", "description"]:
    mask &= ok_str(d1[col])
d2 = d1[mask]
res["after_required"] = len(d2)
res["required_breakdown"] = {"price_missing": int(d1["price"].isna().sum()),
                             "country_missing": int((~ok_str(d1["country"])).sum()),
                             "province_missing": int((~ok_str(d1["province"])).sum()),
                             "variety_missing": int((~ok_str(d1["variety"])).sum())}

# (d) duplicate normalized descriptions
norm = lambda s: re.sub(r"\s+", " ", str(s)).strip().lower()
dn = d2["description"].map(norm)
d3 = d2[~dn.duplicated(keep="first")]
res["after_desc_dup"] = len(d3)
# sensitivity: raw (un-normalized) description duplicates
res["after_desc_dup_raw_text_only"] = int(len(d2) - d2["description"].duplicated().sum())

clean = pd.read_csv(ROOT / "data" / "processed" / "wine_clean.csv")
res["clean_csv_rows"] = len(clean)
mine = set(d3["Unnamed: 0"]); theirs = set(clean["source_row_id"])
res["row_id_set_identical"] = mine == theirs
res["row_id_sym_diff"] = len(mine ^ theirs)

# split sizes / overlap
res["split_rows"] = clean["split"].value_counts().to_dict()
tn = clean["title_norm"]
# re-derive title_norm independently and compare
tn2 = clean["title"].map(norm)
res["title_norm_rederived_mismatch"] = int((tn != tn2).sum())
S = {s: set(tn[clean.split == s]) for s in ("train", "valid", "test")}
res["unique_titles"] = {s: len(v) for s, v in S.items()}
res["title_overlap"] = {"train-valid": len(S["train"] & S["valid"]), "train-test": len(S["train"] & S["test"]),
                        "valid-test": len(S["valid"] & S["test"])}
# also check that row duplicates by (winery, title) don't cross split - covered by title
# description overlap
D = {s: set(clean.loc[clean.split == s, "description"].map(norm)) for s in S}
res["desc_overlap"] = {"train-test": len(D["train"] & D["test"]), "valid-test": len(D["valid"] & D["test"]),
                       "train-valid": len(D["train"] & D["valid"])}

# looser leakage: same title after stripping vintage years
yr = re.compile(r"(?<!\d)(19|20)\d{2}(?!\d)")
strip = lambda t: re.sub(r"\s+", " ", yr.sub(" ", t)).strip()
clean["title_novint"] = tn.map(strip)
trv = set(clean.loc[clean.split.isin(["train", "valid"]), "title_novint"])
tst = clean[clean.split == "test"]
hit_rows = tst["title_novint"].isin(trv)
uniq_test = tst.drop_duplicates("title_norm")
hit_titles = uniq_test["title_novint"].isin(trv)
res["novintage_leak"] = {"test_rows": int(len(tst)), "test_rows_sharing_wine_with_trainvalid": int(hit_rows.sum()),
                         "share_rows_pct": float(100 * hit_rows.mean()),
                         "test_unique_titles": int(len(uniq_test)),
                         "test_titles_sharing": int(hit_titles.sum()),
                         "share_titles_pct": float(100 * hit_titles.mean())}
trn = set(clean.loc[clean.split == "train", "title_novint"])
vld = clean[clean.split == "valid"]
res["novintage_leak_valid_vs_train_rows_pct"] = float(100 * vld["title_novint"].isin(trn).mean())
# how similar are the points of matched pairs? (context for how much the model could gain from memorising)
tv_mean = clean[clean.split.isin(["train", "valid"])].groupby("title_novint")["points"].mean()
m = tst[hit_rows]
res["novintage_matched_abs_points_gap_mean"] = float((m["points"] - m["title_novint"].map(tv_mean)).abs().mean())
# same winery appearing in both splits (winery is not a main-model feature; context only)
res["test_rows_winery_seen_in_trainvalid_pct"] = float(100 * tst["winery"].isin(
    set(clean.loc[clean.split.isin(["train", "valid"]), "winery"])).mean())

# re-run the documented split on my own cleaned frame to confirm assignment reproducibility
from sklearn.model_selection import GroupShuffleSplit
d3r = d3.reset_index(drop=True)
g = d3r["title"].map(norm).values
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(d3r, groups=g))
ot = d3r.iloc[tr]
tr2, va2 = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(ot, groups=ot["title"].map(norm).values))
lab = pd.Series("", index=d3r.index); lab.iloc[te] = "test"; lab.loc[ot.index[tr2]] = "train"; lab.loc[ot.index[va2]] = "valid"
mp = dict(zip(d3r["Unnamed: 0"], lab))
res["split_reproduced_mismatch_rows"] = int((clean["source_row_id"].map(mp) != clean["split"]).sum())

json.dump(res, open(OUT / "a1_clean_split.json", "w"), indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(json.dumps(res, indent=2, default=str))
