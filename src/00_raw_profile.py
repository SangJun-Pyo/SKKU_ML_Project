"""Profile the raw Kaggle file: column list, missing rates, example values and how each column is used.

Outputs (outputs/eda/):
  raw_profile.csv  - one row per raw column
  raw_sample.csv   - the first 3 raw rows (key columns, description truncated) for the slides
"""
import json

import pandas as pd

from common import EDA_DIR, RAW_CSV, ensure_dirs

ensure_dirs()
raw = pd.read_csv(RAW_CSV)

# Korean description and analysis role per column (documentation, not data)
META = {
    "Unnamed: 0": ("원본 행 번호", "원본 추적용 ID"),
    "country": ("생산 국가", "입력 변수"),
    "description": ("평론가 리뷰 본문", "보조 분석(텍스트)"),
    "designation": ("와인 세부 명칭 (예: Reserve)", "미사용"),
    "points": ("평론가 점수 (80~100)", "목표 변수"),
    "price": ("병당 가격 (USD)", "입력 변수 (로그 변환)"),
    "province": ("주·세부 산지", "입력 변수"),
    "region_1": ("세부 지역 1", "미사용 (결측 많음)"),
    "region_2": ("세부 지역 2 (미국만)", "미사용 (결측 많음)"),
    "taster_name": ("평론가 이름", "평론가 효과 점검"),
    "taster_twitter_handle": ("평론가 트위터 계정", "미사용"),
    "title": ("와인 이름 (빈티지 포함)", "빈티지 추출 · 분할 기준"),
    "variety": ("포도 품종", "입력 변수"),
    "winery": ("와이너리", "추가 실험"),
}

rows = []
for col in raw.columns:
    s = raw[col]
    nonnull = int(s.notna().sum())
    example = s.dropna().iloc[0] if nonnull else None
    if isinstance(example, str) and len(example) > 40:
        example = example[:40] + "…"
    desc, role = META.get(col, ("", ""))
    rows.append({
        "column": col,
        "description_ko": desc,
        "role_ko": role,
        "dtype": str(s.dtype),
        "non_null": nonnull,
        "missing_pct": round(100 * (1 - nonnull / len(raw)), 1),
        "n_unique": int(s.nunique(dropna=True)),
        "example": example,
    })
prof = pd.DataFrame(rows)
prof.to_csv(EDA_DIR / "raw_profile.csv", index=False, encoding="utf-8")

sample_cols = ["title", "country", "province", "variety", "price", "points", "taster_name", "description"]
sample = raw.loc[:2, sample_cols].copy()
sample["description"] = sample["description"].str.slice(0, 60) + "…"
sample.to_csv(EDA_DIR / "raw_sample.csv", index=False, encoding="utf-8")

summary = {"raw_rows": int(len(raw)), "raw_cols": int(raw.shape[1]),
           "price_missing": int(raw["price"].isna().sum()),
           "exact_duplicates_excl_id": int(raw.drop(columns=["Unnamed: 0"]).duplicated().sum())}
with open(EDA_DIR / "raw_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(prof[["column", "missing_pct", "role_ko"]].to_string(index=False))
print(json.dumps(summary, ensure_ascii=False))
