# Findings & Slide Copy (story lock, v2 — 2026-10-19)

> v2: slide copy rewritten in natural Korean directly in the deck (deck is now the source of slide wording); opening quiz changed to Opus One 2006 ($195, 90) vs Montes Alpha Cabernet Sauvignon 2006 ($28, 91) — rows 28150 / 26694, key_numbers.quiz_pair. Numbers below are unchanged.

Source of every number: `outputs/key_numbers.json` (key path in brackets). Audit: `audit/audit_report.md` (0 FAIL; WARNs fixed).
Wording follows the audit's safe-wording rules: "맛" = Wine Enthusiast critic score; prices = 2017 US review prices; association, not causation.

## Headline story
가격만 보지 말고 산지·품종·빈티지를 같이 보면, 같은 예산에서 더 높은 평점의 와인을 더 싸게 고를 수 있다.
복잡한 모델(XGBoost)은 플러스알파일 뿐이고, 개선의 약 80%는 단순한 선형모델(Ridge)로도 얻어진다.
단, "나라 효과"로 보이는 것의 상당 부분은 "평론가 효과"다.

---

## Slide copy (Korean). Chart = uploaded image id.

### 01 cover
- eyebrow: 성균관대 핀테크 MBA · 머신러닝과 금융 · 조별 프로젝트
- title: 비싼 와인은 진짜 맛있을까?
- sub: 전문가 리뷰 11만 건으로 본 가격·산지·품종, 그리고 예산 안에서 고르는 법
- byline: [팀원 이름] · 2026년 10월

### 02 quiz
- title: 퀴즈 — 둘 중 더 비싼 와인은?
- card A: 미국 워싱턴 · 카베르네 소비뇽 — "밀크초콜릿·체리·향신료 향이 바로 끌린다. 맛은 매끈하고 세련됐고, 과일과 오크가 반반이다."
- card B: 미국 나파밸리 · 카베르네 소비뇽 — "즙이 많고 스파이시한 붉은 과일 중심. 어깨가 넓고, 마무리에 타닌이 강하게 치고 달콤한 담배·바닐라가 올라온다."
- hint: 힌트: 두 와인의 전문가 점수는 같다.
- notes: 리뷰 원문 요약(source_row_id 15428, 37412). 정답은 마지막에 공개.

### 03 questions
- title: 오늘 답할 질문 세 가지
- 1 가격은 전문가 점수를 얼마나 설명할까?
- 2 가격 말고 점수와 함께 움직이는 건 무엇일까?
- 3 예산이 정해져 있을 때, 모델로 고르면 "제일 비싼 것"보다 나을까?
- footnote: 이 발표에서 "맛있다" = Wine Enthusiast 평론가 점수(80–100점)

### 04 data
- title: 데이터 — Wine Enthusiast 와인 리뷰
- big numbers: 111,510건 리뷰(정제 후) [data.clean_rows] · 42개국 [eda.n_countries] · 691개 품종 [eda.n_varieties] · 가격 $4–$3,300, 중앙값 $25 [eda.price_*] · 점수 80–100, 중앙값 88 [eda.points_*]
- caveats (3 small cards): 80점 이상만 실린다 / 가격은 2017년 미국 리뷰 기준가 / 리뷰 1건 = 평론가 1명의 판단
- footer: 자료: Kaggle "Wine Reviews"(zynicide), Wine Enthusiast, 2017 수집

### 05 fair test
- title: 공정한 시험을 위해 한 일
- step 1 정제: 129,971 → 111,510 (중복 9,983 · 가격 결측 등 8,451 · 같은 리뷰 문장 27) [data.cleaning_steps]
- step 2 같은 와인은 한쪽에만: 와인 이름 기준 분할 — 학습 71,381 / 검증 17,843 / 시험 22,286, 겹침 0 [data.split_rows]
- step 3 시험셋은 마지막에 한 번: 모델·하이퍼파라미터는 검증셋으로만 선택
- step 4 독립 재계산: 분석 코드와 별도로 핵심 수치를 처음부터 다시 계산해 대조, 불일치 항목 수정
- notes: 같은 와인이 학습·시험에 섞이면 성능이 부풀려진다. 재현: python src/run_all.py (SEED 42).

### 06 EDA price vs points — chart fig01
- title: 가격이 오르면 점수도 오른다, 하지만 퍼짐이 크다
- takeaway: 로그 가격과 점수의 상관 r = 0.62 [eda.pearson_r_logprice_points] · 가격이 2배가 되면 평균 약 +2점 [value_map.points_per_price_doubling]
- notes: 원가격 상관은 0.42, 로그를 쓰면 0.62. 가격이 비슷해도 점수가 넓게 퍼져 있다는 게 그림의 핵심.

### 07 EDA buckets — chart fig02
- title: 90점 이상을 만날 확률은 가격대마다 크게 다르다
- takeaway: $10 이하 1.2% → $100 초과 89.7% · 전체 평균 37.8% [eda.price_buckets, eda.share_ge90_pct]

### 08 model design
- title: 리뷰 전에 알 수 있는 정보만으로 예측한다
- card main (메인 모델 입력 · 리뷰 전 정보): 국가 · 세부산지 · 품종 · 빈티지 · 가격(로그)
- card aux (보조분석 · 리뷰 후 정보): 리뷰 문장 — 점수를 매긴 평론가가 직접 쓴 글이라, 예측에 쓰면 답을 보고 푸는 셈
- ladder line: 평균 → Ridge(가격) → Ridge(+산지·품종·빈티지) → 랜덤포레스트 → XGBoost
- eval line: 평가 지표 MAE(점)·R² · 모델 선택은 검증셋

### 09 ladder — chart fig03
- title: 가격만으로 38%, 모든 정보를 더해도 50%
- points: 가격 하나 R² 0.38 [models.test_R2_price_only] · +산지·품종·빈티지 0.45 · XGBoost 0.50, 평균 오차 1.71점 [models.test_R2_main, test_MAE_main_points] · 나머지 절반은 이 정보로 설명되지 않는다
- small: 리뷰 문장까지 넣으면 0.77 — 사후 정보라 예측 성능이 아님

### 10 SHAP — chart fig04
- title: 모델 예측에서 가격이 차지하는 몫은 57%
- points: 가격 57% · 빈티지 15% · 품종 11% · 세부산지 10% · 국가 8% [shap.feature_share_pct]
- takeaway: 가격이 가장 크지만, 나머지 43%가 "같은 값이면 더 나은 와인"을 가르는 단서
- footnote: 비중 = 예측에 대한 평균 기여도(|SHAP|), XGBoost, 시험셋 5,000건

### 11 text — chart fig05
- title: 평론가가 높은 점수를 줄 때 쓰는 말
- points: 높은 점수와 함께: gorgeous · delicious · complex · beautiful / 낮은 점수와 함께: lacks · watery · sour · disappointing [text_aux_model.top_*]
- points: 리뷰 단어만으로 점수 변동의 70% 재현 (R² 0.70) [text_aux_model.text_only_test_R2] · 리뷰가 길수록 점수가 높다 (r = 0.54) [eda.pearson_r_review_words_points]
- footnote: 점수를 매긴 사람이 쓴 사후 정보 — 예측이 아니라 "평론가의 언어" 분석

### 12 value by country — chart fig06
- title: 같은 가격이면, 점수가 더 잘 나오는 나라
- points: 오스트리아 +1.8점 · 독일 +1.3 · 포르투갈 +1.0 / 아르헨티나 −0.7 · 칠레 −0.6 [value_map.country_top5/bottom5]
- footnote: 가격만으로 예측한 점수 대비 실제 점수 평균 · 시험셋 · 나라별 100건 이상

### 13 critic — chart fig07
- title: 그런데 나라마다 리뷰한 평론가가 다르다
- points: 아르헨티나·칠레 리뷰의 97–100%는 한 평론가(평균 86.9점, 짠 편) · 오스트리아 리뷰의 69%는 다른 한 평론가(평균 90.7점, 후한 편) [critic_check.*]
- points: 같은 평론가 안에서 비교하면 오스트리아 +1.82 → +0.22, 아르헨티나 −0.72 → −0.28, 칠레 −0.57 → −0.08
- points: 포르투갈은 +0.88로 유지(평론가 1명 기준) · 독일은 평론가에 따라 +1.78 vs −0.48로 갈림 [critic_germany_by_critic]
- takeaway: "나라 효과"의 상당 부분은 "평론가 효과"
- notes: 나라당 평론가가 1–2명뿐이라 완전한 통제는 불가능. 신뢰구간은 평론가 수준 불확실성을 반영하지 않음.

### 14 statement (accent)
- title: 그럼 예산이 정해져 있다면, 어떻게 고를까?
- setup: 시험셋에서 $30 이하 와인 20병 중 1병 고르기 × 1,000번
- strategies: 무작위 · 제일 비싼 병 · Ridge 예측 1등 · XGBoost 예측 1등
- notes: 실제 점수는 고른 뒤에만 확인. $20, $50 예산도 같은 방식.

### 15 $30 result — chart fig10
- title: $30 예산: 모델이 고른 병이 더 자주 90점을 넘는다
- points: 90점 이상 비율 41.0% → 64.7% (+23.7%p, 95% CI 20.0–27.5) · 평균 가격 $29.4 → $25.1 (−$4.2) [simulation.by_budget.usd_30]
- footnote: CI = 같은 시험셋에서 20병 묶음을 다시 뽑을 때의 변동만 반영

### 16 all budgets — chart fig09
- title: 예산이 바뀌어도 결과는 같은 방향이다
- points: $20 22.0% → 41.9% · $30 41.0% → 64.7% · $50 63.6% → 80.7% (제일 비싼 병 → XGBoost)
- points: 최고가 대비 개선분의 76–82%는 단순한 Ridge로도 얻는다 · XGBoost는 +3–5%p를 더한다 [ridge_share_of_main_gain_over_maxprice_pct, Main_minus_Ridge]
- takeaway: 핵심은 복잡한 모델이 아니라 "가격 + 산지·품종·빈티지"를 같이 보는 것

### 17 conclusion
- title: 세 질문에 대한 답
- 1 가격은 점수 변동의 약 38%를 설명한다. 비쌀수록 평균 점수는 높지만 퍼짐이 크다.
- 2 산지·품종·빈티지를 더하면 50%. 다만 나라 간 차이의 상당 부분은 평론가 차이다.
- 3 $30 예산에서 모델로 고르면 90점 이상 65% (제일 비싼 병 41%), 평균 $4 더 싸다.

### 18 quiz reveal + under-$20 examples
- title: 퀴즈 정답 — B가 5배 비싸다
- A $12 · 90점 / B $60 · 90점
- table "모델이 고른 $20 이하 예시 (시험셋, 예측 점수 순)": rank, 와인, 가격, 예측, 실제 [value_map.under20_top5]
  1 Recanati 2015 Shiraz (이스라엘) $17 91.3 92 · 2 Berger 2015 Spiegel Riesling (오스트리아) $20 90.9 90 · 3 Château Jolys 2011 Cuvée Jean (프랑스) $18 90.9 92 · 4 Winzer Krems 2016 Grüner Veltliner (오스트리아) $19 90.8 90 · 5 Geyerhof 2015 Rosensteig Grüner Veltliner (오스트리아) $20 90.8 92
- footnote: 2017년 미국 가격 · 예시일 뿐 구매 추천 아님 · 10개 중 8개가 오스트리아(평론가 영향 가능)
- notes: 두 퀴즈 와인은 서로 다른 평론가(Sean P. Sullivan / Virginie Boone)가 채점했다. $20 이하 전체에서 90점 이상은 11%인데, 모델 상위 10개는 모두 90점 이상.

### 19 limitations
- title: 이 분석이 말하지 않는 것
- 점수 ≠ 내 입맛 — 한 매체 평론가 점수, 80점 이상만 게재
- 2017년 미국 리뷰 기준가 — 지금 한국 가격이 아니다
- 상관이지 인과가 아니다 — 가격이 점수를 올리는 게 아니라 함께 움직인다
- 나라 효과와 평론가 효과는 완전히 분리되지 않는다 — 나라당 평론가 1–2명
- 신뢰구간은 같은 시험셋 안의 변동만 반영한다

### 20 team
- title: 역할 분담
- [팀원 A] 데이터 정제·탐색 분석 / [팀원 B] 모델링·예산 실험 / [팀원 C] 해석·평론가 점검·발표 — placeholders for the team to fill
- footer: 코드·재현: python src/run_all.py · README.md

### A1 appendix — variety value — chart fig08
- title: 부록 — 같은 가격이면, 점수가 더 잘 나오는 품종
- points: 그뤼너 펠트리너 +2.1 · 포르투갈 레드 +1.4 · 리슬링 +1.3 / 샴페인 블렌드 −0.9 · 메를로 −0.8 · 카베르네 프랑 −0.7

### A2 appendix — winery experiment
- title: 부록 — 와이너리 평판을 더하면?
- points: 와이너리 평균 점수(학습 데이터 안에서만 계산)를 추가하면 평균 오차 1.71 → 1.63점, R² 0.50 → 0.55 [models.winery_experiment]
- caveat: 시험셋 와인의 95%가 이미 본 와이너리 → 신규 와이너리에는 낙관적인 숫자. 메인 모델에는 넣지 않았다.

## v3 additions (2026-10-19, round 4)
- Wording: "전문가 점수" → "평점" (Wine Enthusiast 평론가 평점, 80~100점). Not "와이너리 평점": wineries do not score.
- Classical inference (src/03c_stats_ols.py, n = 89,224 train+valid, HC3 SEs):
  - OLS points ~ log_price: b = 2.90 [2.88, 2.93] → +2.01 points per price doubling; t = 234, p < 0.001; R² = adj R² = 0.382; test R² 0.381.
  - OLS + country/province/variety/vintage: log_price b = 2.97; R² 0.455, adj 0.453, test 0.447; nested F vs price-only p < 0.001.
  - Winery-clustered SE: t = 117 (price-only) → conclusion unchanged.
  - Logit (≥90) ~ log_price: odds ratio per doubling 4.61 [4.51, 4.72]; McFadden R² 0.24.
  - Diagnostics: Breusch-Pagan rejects (→ HC3), JB rejects but skew −0.35 / kurtosis 3.14, Durbin-Watson 0.78 (non-independence warning → clustered SE), cond. no. 18.8.
- Reading R² ≈ 0.45–0.50: effect is certain (p tiny, CI narrow) but most of the score variation is not explained by pre-purchase info; MAE falls from 2.53 (mean) to 1.71 (XGBoost), −32%. Adequate for screening (budget simulation), weak for single-bottle prediction.

## v4 additions (2026-10-19, round 5)
- Splits are alike (describe): points mean 88.42/88.41/88.42, ≥90 share 37.8/37.5/37.8%, median price $25 in all; KS(train vs test points) p = 0.98.
- Associations (0–1 strength): points–log price r = 0.62 (raw price 0.42, Spearman 0.61); country–province Cramér's V 0.82 → collinear (GVIF^(1/2df) 26.9 for country); log price 1.22, vintage 1.21, variety 1.09 → price effect is estimated stably. Covariance not compared across pairs (units differ) → correlation shown.
- Over-fitting: linear models train≈valid≈test (no memorisation, limited expressiveness); trees train−valid ≈ 0.09–0.10 but still best on valid/test; valid ≈ test for every model → selecting on valid did not over-fit valid.

## v5 additions (2026-10-19, round 6)
- Multicollinearity (country ⊃ province) affects attribution, not prediction: refit without country (same hyperparameters) → test R² Ridge 0.448→0.446, XGB 0.501→0.497. Kept both inputs for prediction.
- SHAP shown with country+province summed per row as one "산지" group: price 58.7%, vintage 15.1%, origin 15.0%, variety 11.2% (no-country refit: price 59.4%).
- 38% (R² of price alone) vs 59% (price share of the full model's SHAP attribution) answer different questions; both say price is the single strongest cue.
