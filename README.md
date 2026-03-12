# 🧠 BRAIN Alpha Search

WorldQuant BRAIN 알파 수식 자동 탐색 시스템

---

## 📁 프로젝트 구조

```
brain-alpha-search/
├── alpha_search.py              # 메인 실행 파일
├── generators/
│   ├── single_factor.py         # 단일 신호 생성기
│   └── combo_factor.py          # 조합 신호 생성기
├── results/                     # 시뮬레이션 결과 저장
├── .github/workflows/
│   └── alpha_search.yml         # GitHub Actions 자동화
├── requirements.txt
└── README.md
```

---

## 🚀 로컬 실행

```bash
# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
export BRAIN_EMAIL=your@email.com
export BRAIN_PASSWORD=yourpassword

# 실행
python alpha_search.py
```

---

## ⚙️ GitHub Actions 자동화

### 1. Secrets 설정
GitHub 레포 → Settings → Secrets and variables → Actions

| Secret 이름 | 값 |
|---|---|
| `BRAIN_EMAIL` | BRAIN 계정 이메일 |
| `BRAIN_PASSWORD` | BRAIN 계정 비밀번호 |
| `SLACK_WEBHOOK` | (선택) Slack 알림 웹훅 URL |

### 2. 실행 주기
- **자동**: 매주 월요일 오전 9시 (KST)
- **수동**: Actions 탭 → "BRAIN Alpha Search" → "Run workflow"

### 3. 결과 확인
Actions 탭 → 해당 워크플로우 → Artifacts에서 CSV 다운로드

---

## 📊 필터 기준 (`alpha_search.py` 수정)

```python
FILTER = {
    "min_sharpe":   1.5,   # 최소 샤프 비율
    "min_fitness":  1.0,   # 최소 피트니스
    "min_turnover": 0.01,  # 최소 턴오버 (1%)
    "max_turnover": 0.70,  # 최대 턴오버 (70%)
}
```

---

## 🔬 실험 히스토리 및 교훈

| 신호 | TEST 결과 | 비고 |
|---|---|---|
| `operating_income / cap` | -0.16 | 2023 거의 중립 |
| `operating_income YoY` | **+0.13** | 유일한 TEST 양수 ✅ |
| `close / ts_delay(252)` | -0.80 | Subindustry에서 역방향 ❌ |
| `close / ts_delay(60)` | -2.00 | 최악 ❌ |

### 핵심 교훈
- **가격 모멘텀**: Neutralization=Subindustry 설정에서 역방향 작동
- **펀더멘털 신호**: 분기 데이터라 Turnover 구조적 한계 (~8%)
- **2023 AI 장세**: 펀더멘털 신호 전반 약화, 영업이익 YoY만 양수
- **Settings**: Subindustry > Market (이 수식 기준)

---

## ✏️ 알파 추가 방법

`generators/single_factor.py`의 `FUNDAMENTAL_SIGNALS`에 신호 추가:

```python
FUNDAMENTAL_SIGNALS = [
    "operating_income / cap",
    "your_new_signal / cap",   # 추가
    ...
]
```
