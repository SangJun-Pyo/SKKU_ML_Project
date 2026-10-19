# 비싼 와인은 진짜 맛있을까?

성균관대학교 핀테크 MBA「머신러닝과 금융」 팀 프로젝트 (2026년 10월)

Kaggle의 Wine Enthusiast 리뷰 약 11만 건으로 두 가지를 확인했습니다.

1. 가격·산지·품종·빈티지가 평론가 평점(80~100점)을 얼마나 설명하는가
2. 예산이 정해져 있을 때 모델로 고르는 것이 "가장 비싼 와인 고르기"보다 나은가

## 결과 요약

| 질문 | 결과 (시험 데이터) |
|---|---|
| 가격만으로 평점을 얼마나 설명하나 | R² 0.38 (가격이 2배면 평점 약 +2점) |
| 산지·품종·빈티지를 더하면 | Ridge R² 0.45 → XGBoost R² 0.50, 평균 오차 1.71점 |
| 90점 이상 와인을 가려내나 | 로지스틱 AUC 0.841, XGBoost 분류 AUC 0.857 |
| $30 예산에서 20병 중 1병 고르기 | 90점 이상을 고를 확률: 최고가 선택 41% → XGBoost 65%, 평균 가격은 $29.4 → $25.1 |
| 나라별 가성비 차이 | 상당 부분이 나라마다 다른 평론가 때문 (오스트리아 +1.8점 → 같은 평론가끼리 비교 시 +0.2점) |

슬라이드에 쓴 모든 숫자는 [`outputs/key_numbers.json`](outputs/key_numbers.json)에서 나오고, 발표용 그래프는 [`outputs/figures_ko/`](outputs/figures_ko/)에 있습니다. 분석 메모는 [`reports/findings.md`](reports/findings.md)를 보세요.

---

## 실행 가이드

처음 해 보는 사람 기준으로 적었습니다. 전체 실행 시간은 노트북 기준 약 10~15분입니다.

### 1. 준비물

- **Python 3.12 이상** (3.13에서 테스트)
  - 터미널에서 `python --version`(macOS는 `python3 --version`)으로 확인하세요.
  - 없으면 https://www.python.org/downloads/ 에서 설치합니다. Windows는 설치 화면에서 **"Add python.exe to PATH"** 를 꼭 체크하세요.
- **Git** (또는 GitHub에서 ZIP으로 내려받아도 됩니다)
- **Kaggle 계정** (데이터 다운로드용, 무료)

### 2. 코드 내려받기

```bash
git clone https://github.com/SangJun-Pyo/SKKU_ML_Project.git
cd SKKU_ML_Project
```

Git이 없다면 GitHub 페이지의 초록색 **Code → Download ZIP**으로 받아 압축을 풀고, 그 폴더에서 터미널을 여세요.

### 3. 가상환경 만들기 (권장)

프로젝트 전용 파이썬 환경을 만들어 다른 프로젝트와 패키지 버전이 섞이지 않게 합니다.

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
> "스크립트를 실행할 수 없습니다" 오류가 나면 먼저 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`를 한 번 실행하세요.

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

프롬프트 앞에 `(.venv)`가 보이면 성공입니다.

### 4. 패키지 설치

```bash
pip install -r requirements.txt
```

pandas, scikit-learn, xgboost, shap, statsmodels, matplotlib 등이 버전 고정으로 설치됩니다.

### 5. 데이터 넣기 (저장소에는 포함하지 않음)

1. https://www.kaggle.com/datasets/zynicide/wine-reviews 에서 **Download**를 누릅니다.
2. 압축을 풀면 나오는 파일 중 **`winemag-data-130k-v2.csv`** 하나만 씁니다.
3. 이 파일을 `data/raw/` 폴더에 넣습니다.

```
SKKU_ML_Project/
└── data/
    └── raw/
        └── winemag-data-130k-v2.csv   ← 여기
```

> 원본 데이터는 Kaggle 배포 데이터이고 용량도 약 53MB라 저장소에 올리지 않았습니다. 이용 조건은 Kaggle 페이지를 확인하세요.

### 6. 전체 실행

```bash
python src/run_all.py
```

`src/` 안의 번호 붙은 스크립트(`00_` → `09d_`)를 순서대로 모두 실행합니다. 화면에 `===== 03_models.py =====` 같은 진행 표시가 나오고, 시간은 대부분 `03_models.py`(XGBoost 튜닝, 랜덤포레스트)에서 씁니다.

끝나면 `outputs/key_numbers.json`이 다시 만들어집니다. 시드가 고정(SEED=42)되어 있어서 같은 환경이면 저장소에 올라간 파일과 같은 숫자가 나옵니다.

### 7. 한 단계만 다시 돌리고 싶을 때

각 스크립트는 앞 단계의 결과 파일을 읽기 때문에, 한 번 전체 실행한 뒤에는 필요한 것만 따로 돌릴 수 있습니다.

```bash
python src/04_sim.py          # 예산 시뮬레이션만 다시
python src/08_key_numbers.py  # 숫자 모음만 다시
python src/09_figures_ko.py   # 그래프만 다시
```

### 문제 해결

| 증상 | 해결 |
|---|---|
| `FileNotFoundError: ... winemag-data-130k-v2.csv` | 5단계: 파일 이름과 위치(`data/raw/`)를 확인하세요. |
| `pip install`에서 버전을 찾을 수 없다는 오류 | 파이썬이 3.12 미만입니다. 3.12 이상을 설치하고 가상환경을 다시 만드세요. |
| 그래프의 한글이 네모(□)로 나옴 | Windows는 맑은 고딕, macOS는 AppleGothic을 자동으로 씁니다. Linux는 `sudo apt install fonts-noto-cjk` 후 `~/.cache/matplotlib`을 지우고 다시 실행하세요. |
| 엑셀에서 CSV 한글이 깨짐 | 파일은 UTF-8입니다. 엑셀의 "데이터 → 텍스트/CSV 가져오기"에서 UTF-8을 고르세요. |
| `python` 명령을 찾을 수 없음 (macOS) | `python3`, `pip3`로 바꿔서 입력하세요. |

---

## 폴더 구조

```
SKKU_ML_Project/
├── src/                  분석 코드 (아래 표)
├── data/raw/             원본 CSV를 넣는 곳 (직접 다운로드)
├── data/processed/       정제·분할된 데이터 (실행하면 생성)
├── outputs/
│   ├── key_numbers.json  슬라이드에 쓴 모든 숫자
│   └── figures_ko/       발표용 그래프
├── reports/findings.md   분석 결과 메모
├── audit/                결과를 처음부터 다시 계산해 대조한 독립 점검 (217개 항목)
└── requirements.txt
```

실행하면 `outputs/` 아래에 `eda/`, `models/`, `sim/`, `valuemap/`, `critic/` 폴더가 생기고 중간 결과(성능표, 예측값, 시뮬레이션 기록 등)가 저장됩니다. 이 파일들은 다시 만들 수 있어서 저장소에는 올리지 않았습니다.

## 스크립트

| 파일 | 하는 일 |
|---|---|
| `common.py` | 경로·시드·폰트·텍스트 정규화 공통 함수 |
| `00_raw_profile.py` | 원본 파일 프로필 (열 설명, 결측률, 첫 세 줄) |
| `01_clean.py` | 정제(중복·결측 제거)와 와인 이름 기준 학습/검증/시험 분할 |
| `02_eda.py` | 기초 통계, 가격대별 평점, 퀴즈 와인 확인 |
| `02b_associations.py` | 분할별 기초 통계량, 전체 변수 상관(피어슨·상관비·크래머 V), 공분산, 다중공선성(GVIF) |
| `03_models.py` | 회귀 모델 비교(평균 → Ridge/Lasso/ElasticNet → 랜덤포레스트 → XGBoost), 와이너리 실험, SHAP, 리뷰 텍스트 보조 모델 |
| `03b_classify.py` | 90점 이상 분류: 로지스틱(L1/L2) vs XGBoost 분류, AUC 차이 신뢰구간 |
| `03c_stats_ols.py` | 통계 검정(statsmodels): OLS 계수·p-value·신뢰구간·R²·F-검정·잔차 진단, 와이너리 군집 표준오차, 로지스틱 오즈비 |
| `03d_overfit_check.py` | 과적합 점검: 학습·검증·시험 성능 비교 |
| `03e_collinearity_check.py` | 다중공선성 점검: 국가를 뺀 재학습, 국가+세부 산지 합산 SHAP |
| `04_sim.py` | 예산($20/$30/$50) 와인 고르기 시뮬레이션 + 부트스트랩 신뢰구간 |
| `05_valuemap.py` | 가성비 지도 (국가·품종별 가격 대비 평점) |
| `07_critic_check.py` | 평론가 효과 점검 (같은 평론가끼리 비교) |
| `08_key_numbers.py` | 모든 결과를 `outputs/key_numbers.json` 하나로 모음 |
| `09_figures_ko.py` 외 `09b~09d` | 발표용 한글 그래프 |
| `run_all.py` | 위 스크립트를 순서대로 실행 |

## 분석 설계 메모

- **입력 변수**: 국가, 세부 산지, 품종, 로그 가격, 빈티지. 모두 와인을 사기 전에 알 수 있는 정보만 썼습니다. 리뷰 본문은 평점과 함께 쓴 사후 정보라 보조 분석에만 썼습니다.
- **분할**: 와인 이름 기준 그룹 분할(GroupShuffleSplit)로 학습 64% / 검증 16% / 시험 20%. 같은 와인이 학습과 시험에 동시에 들어가지 않습니다. 시험 데이터는 모든 선택이 끝난 뒤 한 번만 썼습니다.
- **가격은 로그 변환**: $4~$3,300로 오른쪽 꼬리가 긴 분포(왜도 18)를 종 모양(왜도 0.6)으로 바꾸고, "가격이 2배가 될 때 +2점"처럼 비율로 해석하기 위해서입니다.

## 데이터의 한계

- 평점은 한 매체(Wine Enthusiast) 평론가의 점수이며, **80점 이상만 공개**된 데이터입니다. "내 입맛"과는 다릅니다.
- 나라마다 평론가가 1~2명이라 국가 효과와 평론가 효과를 완전히 분리할 수 없습니다.
- 가격은 2017년 수집 당시 미국 가격(USD)입니다.
- 가격과 평점의 관계는 상관관계이며 인과관계가 아닙니다.

## 출처

- 데이터: [Wine Reviews, zynicide (Kaggle)](https://www.kaggle.com/datasets/zynicide/wine-reviews)
- 팀: 성균관대학교 핀테크 MBA 머신러닝과 금융 팀 프로젝트
