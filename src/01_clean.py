"""P1a: clean raw wine reviews, add features, and make a grouped train/valid/test split."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLEAN_CSV, EDA_DIR, RAW_CSV, SEED, ensure_dirs,  # noqa: E402
                    normalize_text, normalize_title, parse_vintage)

ensure_dirs()
df = pd.read_csv(RAW_CSV)
log = []


def record(step, df_now, prev_n, note=""):
    log.append({"step": step, "rows_remaining": len(df_now),
                "rows_dropped": (prev_n - len(df_now)) if prev_n is not None else 0, "note": note})
    return len(df_now)


n = record("a_raw", df, None, f"{df.shape[1]} columns")

# b. exact duplicates across all columns except the source row id
cols_wo_id = [c for c in df.columns if c != "Unnamed: 0"]
df = df.drop_duplicates(subset=cols_wo_id, keep="first")
n = record("b_drop_exact_duplicates", df, n, "all columns except 'Unnamed: 0'; NaN==NaN; keep first")

# c. validity requirements
def nonempty(s):
    return s.notna() & (s.astype("string").str.strip().fillna("") != "")

fail = {
    "price_invalid": ~(df["price"].notna() & np.isfinite(df["price"].astype(float)) & (df["price"] > 0)),
    "points_out_of_range": ~df["points"].between(80, 100),
}
for c in ["country", "province", "variety", "title", "description"]:
    fail[f"{c}_empty"] = ~nonempty(df[c])
keep = ~pd.concat(fail.values(), axis=1).any(axis=1)
note = "; ".join(f"{k}={int(v.sum())}" for k, v in fail.items()) + " (criteria overlap)"
df = df[keep].copy()
n = record("c_require_valid_fields", df, n, note)

# d. duplicate descriptions after normalization
df["_desc_norm"] = df["description"].map(normalize_text)
df = df.drop_duplicates(subset="_desc_norm", keep="first").copy()
n = record("d_drop_duplicate_descriptions", df, n, "strip+collapse whitespace+lowercase; keep first")

# added columns
df["source_row_id"] = df["Unnamed: 0"]
df["title_norm"] = df["title"].map(normalize_title)
df["log_price"] = np.log(df["price"].astype(float))
df["vintage"] = df["title"].map(parse_vintage)
df = df.drop(columns=["Unnamed: 0"]).reset_index(drop=True)

# split: grouped by title_norm, 80/20 outer then 80/20 inner
groups = df["title_norm"].values
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr_idx, te_idx = next(gss.split(df, groups=groups))
outer_train = df.iloc[tr_idx]
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr2, va2 = next(gss2.split(outer_train, groups=outer_train["title_norm"].values))
df["split"] = ""
df.loc[df.index[te_idx], "split"] = "test"
df.loc[outer_train.index[tr2], "split"] = "train"
df.loc[outer_train.index[va2], "split"] = "valid"
assert (df["split"] != "").all()

# checks
checks = {"row_counts": df["split"].value_counts().to_dict(),
          "n_total": int(len(df)),
          "share": (df["split"].value_counts(normalize=True).round(4)).to_dict()}
pairs = [("train", "valid"), ("train", "test"), ("valid", "test")]
title_sets = {s: set(df.loc[df.split == s, "title_norm"]) for s in ("train", "valid", "test")}
desc_sets = {s: set(df.loc[df.split == s, "_desc_norm"]) for s in ("train", "valid", "test")}
checks["title_norm_overlap"] = {f"{a}-{b}": len(title_sets[a] & title_sets[b]) for a, b in pairs}
checks["description_norm_overlap"] = {f"{a}-{b}": len(desc_sets[a] & desc_sets[b]) for a, b in pairs}
checks["unique_titles_per_split"] = {s: len(v) for s, v in title_sets.items()}
checks["unique_title_norm_total"] = int(df["title_norm"].nunique())
checks["seed"] = SEED
assert all(v == 0 for v in checks["title_norm_overlap"].values()), checks["title_norm_overlap"]
assert all(v == 0 for v in checks["description_norm_overlap"].values()), checks["description_norm_overlap"]
assert df["source_row_id"].is_unique
assert sum(checks["row_counts"].values()) == len(df)

df = df.drop(columns=["_desc_norm"])
df.to_csv(CLEAN_CSV, index=False, encoding="utf-8")
pd.DataFrame(log).to_csv(EDA_DIR / "cleaning_log.csv", index=False, encoding="utf-8")
with open(EDA_DIR / "split_checks.json", "w", encoding="utf-8") as f:
    json.dump(checks, f, indent=2, ensure_ascii=False)

print(pd.DataFrame(log).to_string(index=False))
print(json.dumps(checks, indent=2))
