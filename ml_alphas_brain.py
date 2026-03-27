"""
ml_alphas_brain.py
──────────────────
ML 팩터 팩토리 결과를 WorldQuant Brain에 배치 시뮬레이션 제출.

ML 중요도 1~12위 팩터를 Brain 표현식으로 변환한 7개 알파를 순서대로 제출하고,
시뮬레이션 결과(Sharpe, Fitness, 자기상관)를 수집해 비교 리포트를 출력합니다.

사용법:
    python ml_alphas_brain.py --email YOU@email.com --password YOUR_PW
    python ml_alphas_brain.py --email YOU@email.com --password YOUR_PW --submit-only
    python ml_alphas_brain.py --email YOU@email.com --password YOUR_PW --collect-only
"""

import argparse
import json
import time
import sys
from pathlib import Path
from datetime import datetime

try:
    import requests
except ImportError:
    sys.exit("requests 패키지가 필요합니다: pip install requests")

# ── Brain API ─────────────────────────────────────────────────────────────────
BASE_URL = "https://api.worldquantbrain.com"

# ── 공통 시뮬레이션 파라미터 (현재 설정 기준) ─────────────────────────────────
BASE_SETTINGS = {
    "type":            "REGULAR",
    "instrumentType":  "EQUITY",
    "region":          "USA",
    "universe":        "TOP3000",
    "delay":           1,
    "decay":           0,          # 펀더멘털 저주파 신호: decay=0으로 자기상관 위험 줄임
    "neutralization":  "SUBINDUSTRY",
    "truncation":      0.08,
    "unitHandling":    "VERIFY",
    "pasteurization":  "ON",
    "nanHandling":     "OFF",
    "language":        "FASTEXPR",
    "visualization":   False,
}

# ── ML 팩터 → Brain 표현식 변환 결과 ─────────────────────────────────────────
#
#  [설계 원칙]
#  1. 모든 항은 rank()로 감싸 UNITS 오류 방지 (무차원화)
#  2. 펀더멘털 데이터는 분기 업데이트 → ts_delta / ts_mean 차분으로 동적성 주입
#  3. decay=0 사용 → 자기상관 리스크를 줄이는 대신 turnover 증가 감수
#  4. subindustry 중립화 → 섹터/산업 편향 제거
#  5. 방향이 반전되면 알파 앞에 -1 곱하기로 수정
#
#  [ML 팩터 매핑 근거]
#  #1 Accruals    → EPS 발생액 변화 (이익 품질 지표, Sloan 1996)
#  #2 Value×Qual  → ROE × (1/earnings_yield 근사) — 저평가+고퀄 교집합
#  #4 P/B 근사    → ROE 역수 (Book Value 직접 필드 없음 → ROE로 근사)
#  #9 Rev Growth  → 연간 매출 성장률 (sales YoY)
#  #12 FCF Yield  → ROA - ROE×0.6 (레버리지 조정 수익성 갭)
#  #8 Sector Mom  → Brain의 group_neutralize 대신 rank 내에서 흡수
#  복합 앙상블     → 위 4개 rank 합산 (IC 가중 단순화)

ML_ALPHAS = [
    # ─────────────────────────────────────────────────────────────────────────
    # v1: Accruals — ML 중요도 1위
    #   EPS가 과거 4분기 평균보다 '갑자기 높아지면' 이후 수익률 하락
    #   → 발생액이 클수록 short, 작을수록 long
    #   Jansen Ch.4 Sloan accruals 팩터 직접 이식
    # ─────────────────────────────────────────────────────────────────────────
    {
        "name": "v1_accruals_eps_delta",
        "description": "ML #1: Accruals — EPS 발생액 역(逆)신호. "
                       "EPS가 4분기 이동평균 대비 급등한 종목은 이후 mean-revert. "
                       "rank(-ts_delta)로 발생액 낮은 종목(현금이익) long.",
        "expression": "rank(-ts_delta(earnings_per_share_reported, 4))",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # v2: Value × Quality 상호작용 — ML 중요도 2위
    #   ROE(수익성) × EPS 성장 모멘텀 교집합
    #   단순 ROE보다 '지속적으로 성장하는 수익성'에 집중
    # ─────────────────────────────────────────────────────────────────────────
    {
        "name": "v2_value_quality_interaction",
        "description": "ML #2: Value×Quality 교집합. "
                       "ROE(Quality)와 EPS 성장(Value proxy) 동시 상위 종목 long. "
                       "두 rank 곱 → 둘 다 좋은 종목만 강하게 선별.",
        "expression": "rank(return_equity) * rank(ts_delta(earnings_per_share_reported, 4))",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # v3: P/B 근사 (ROE 역수) — ML 중요도 4위
    #   Brain 표준 계정에 book_value 직접 필드 없음
    #   → ROE = Net Income / Book Equity → 1/ROE ∝ P/B (시장가 가정 하)
    #   낮은 ROE 역수(= 높은 ROE) → 고퀄리티, 반대로 해석 주의
    #   여기서는 '저평가 가치주' 방향: 낮은 ROE = 낮은 P/B 근사 → long
    # ─────────────────────────────────────────────────────────────────────────
    {
        "name": "v3_pb_proxy_roe_inverse",
        "description": "ML #4: P/B 근사 — 1/ROE를 book-to-market 대리변수로 활용. "
                       "ROE 낮은 종목(저평가 가치주 후보) long. "
                       "단, mean-reversion 가정 하 유효 — 모멘텀 장세에서는 약화.",
        "expression": "rank(-return_equity)",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # v4: Revenue Growth YoY — ML 중요도 9위
    #   sales / ts_delay(sales, 252) - 1 = 약 1년 전 대비 매출 성장률
    #   252 거래일 ≈ 1년 (Brain delay=1 기준)
    #   성장주 팩터: 매출 고성장 종목 long
    # ─────────────────────────────────────────────────────────────────────────
    {
        "name": "v4_revenue_growth_yoy",
        "description": "ML #9: Revenue Growth YoY — 연간 매출 성장률. "
                       "sales(t) / sales(t-252) - 1 을 rank화. "
                       "고성장 종목 long. decay=0으로 분기 업데이트 충격 반영.",
        "expression": "rank(sales / ts_delay(sales, 252) - 1)",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # v5: FCF Yield 프록시 — ML 중요도 12위
    #   FCF = Operating CF - CapEx → Brain 직접 필드 제한적
    #   ROA - return_equity × 0.6 ≈ 부채 레버리지 조정 후 현금수익성 갭
    #   양수일수록 영업 효율 > 재무 레버리지 의존도 → 진짜 현금창출 기업
    # ─────────────────────────────────────────────────────────────────────────
    {
        "name": "v5_fcf_proxy_roa_roe_gap",
        "description": "ML #12: FCF Yield 프록시 — ROA와 ROE 갭으로 레버리지 제거 후 수익성 측정. "
                       "갭이 클수록 부채 의존 없는 진짜 현금창출 기업. "
                       "UNITS 오류 방지를 위해 rank() 래핑.",
        "expression": "rank(return_assets - return_equity * 0.6)",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # v6: EPS 성장 가속도 (ts_rank 버전) — ML 복합 파생
    #   단순 EPS 레벨이 아닌 '성장이 가속되는' 종목 포착
    #   ts_rank(ts_delta(eps, 4), 8) = 최근 8분기 중 EPS 성장폭 순위
    #   자기상관 위험이 낮고 동적성이 높음
    # ─────────────────────────────────────────────────────────────────────────
    {
        "name": "v6_eps_growth_acceleration",
        "description": "ML 파생: EPS 성장 가속도 — ts_rank로 분기별 EPS 델타 중 "
                       "현재 성장폭이 과거 8분기 대비 상위인 종목 long. "
                       "단순 레벨 대비 자기상관 낮고 동적 신호.",
        "expression": "ts_rank(ts_delta(earnings_per_share_reported, 4), 8)",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # v7: ML 앙상블 복합 알파 — 최종 타겟
    #   ML 중요도 상위 4개 팩터 rank 합산 (IC 가중 단순화 버전)
    #   Accruals(-), Value×Quality(+), Rev Growth(+), FCF Proxy(+)
    #   다팩터 합산으로 개별 팩터 노이즈 상쇄 기대
    #   이전 실험의 ts_rank(ts_mean(...) - ts_mean(...)) 구조에서 진화
    # ─────────────────────────────────────────────────────────────────────────
    {
        "name": "v7_ml_ensemble_composite",
        "description": "ML 앙상블 복합 알파 — 중요도 상위 4개 팩터 rank 합산. "
                       "Accruals 역신호 + Value×Quality + Rev Growth + FCF Proxy. "
                       "다팩터 결합으로 개별 노이즈 상쇄 & Sharpe 안정성 목표.",
        "expression": (
            "rank("
            "  -rank(ts_delta(earnings_per_share_reported, 4))"   # Accruals 역신호
            " + rank(return_equity)"                               # Quality
            " + rank(sales / ts_delay(sales, 252) - 1)"           # Rev Growth
            " + rank(return_assets - return_equity * 0.6)"        # FCF Proxy
            ")"
        ),
    },
]


# ── Brain API 헬퍼 ────────────────────────────────────────────────────────────

def get_session(email: str, password: str) -> requests.Session:
    """Brain API 인증 세션 생성."""
    session = requests.Session()
    session.auth = (email, password)

    resp = session.post(f"{BASE_URL}/authentication", json={"username": email, "password": password})
    if resp.status_code not in (200, 201):
        # Basic Auth만으로도 동작하는 경우가 많음
        print(f"  [인증] 로그인 응답 {resp.status_code} — Basic Auth 사용 계속")
    else:
        print(f"  [인증] ✓ Brain 로그인 성공")
    return session


def submit_alpha(session: requests.Session, alpha: dict) -> dict | None:
    """단일 알파 시뮬레이션 제출."""
    payload = {**BASE_SETTINGS, "code": alpha["expression"]}

    resp = session.post(f"{BASE_URL}/alphas", json=payload)
    if resp.status_code in (200, 201):
        data = resp.json()
        alpha_id = data.get("id") or data.get("alphaId")
        print(f"    ✓ 제출 성공 | ID: {alpha_id} | {alpha['name']}")
        return {"id": alpha_id, "name": alpha["name"], "expression": alpha["expression"],
                "description": alpha["description"], "submitted_at": datetime.now().isoformat()}
    else:
        print(f"    ✗ 제출 실패 [{resp.status_code}] | {alpha['name']}")
        print(f"      {resp.text[:200]}")
        return None


def collect_result(session: requests.Session, alpha_id: str, name: str) -> dict:
    """시뮬레이션 완료 대기 후 결과 수집."""
    max_wait = 600   # 10분 타임아웃
    interval = 30
    waited   = 0

    while waited < max_wait:
        resp = session.get(f"{BASE_URL}/alphas/{alpha_id}")
        if resp.status_code != 200:
            print(f"    ✗ 결과 조회 실패 [{resp.status_code}]")
            time.sleep(interval)
            waited += interval
            continue

        data   = resp.json()
        status = data.get("status", "").upper()

        if status in ("COMPLETE", "DONE", "FINISHED", "SUCCESS"):
            stats  = data.get("is", data.get("stats", {}))
            sharpe = stats.get("sharpe", stats.get("annualized_sharpe", None))
            fitness= stats.get("fitness", None)
            turns  = stats.get("turnover", stats.get("daily_turnover", None))
            self_c = stats.get("self_correlation", stats.get("selfcorrelation", None))
            margin = stats.get("margin", None)

            result = {
                "id":               alpha_id,
                "name":             name,
                "status":           status,
                "sharpe":           sharpe,
                "fitness":          fitness,
                "turnover":         turns,
                "self_correlation": self_c,
                "margin":           margin,
                "passes":           _check_pass(sharpe, fitness, self_c),
            }
            _print_result(result)
            return result

        elif status in ("ERROR", "FAILED", "CANCELLED"):
            print(f"    ✗ 시뮬레이션 실패: {status}")
            return {"id": alpha_id, "name": name, "status": status, "passes": False}

        else:
            print(f"    ⏳ {name} — 대기 중 ({waited}s / {max_wait}s) | status: {status}")
            time.sleep(interval)
            waited += interval

    print(f"    ⚠ 타임아웃: {name}")
    return {"id": alpha_id, "name": name, "status": "TIMEOUT", "passes": False}


def _check_pass(sharpe, fitness, self_corr) -> bool:
    """Brain 통과 기준 체크."""
    if sharpe is None or fitness is None:
        return False
    corr_ok = (self_corr is None) or (abs(self_corr) < 0.7)
    return sharpe >= 1.25 and fitness >= 1.0 and corr_ok


def _print_result(r: dict):
    """결과 출력 (컬러 없이 ASCII)."""
    passed = "✓ PASS" if r.get("passes") else "✗ FAIL"
    print(f"\n    [{passed}] {r['name']}")
    print(f"      Sharpe:      {r.get('sharpe', 'N/A')}")
    print(f"      Fitness:     {r.get('fitness', 'N/A')}")
    print(f"      Turnover:    {r.get('turnover', 'N/A')}")
    print(f"      Self-corr:   {r.get('self_correlation', 'N/A')}")
    print(f"      Margin:      {r.get('margin', 'N/A')}")


# ── 리포트 출력 ───────────────────────────────────────────────────────────────

def print_report(results: list[dict]):
    """최종 비교 리포트 출력."""
    print("\n" + "=" * 70)
    print("  ML → BRAIN 알파 시뮬레이션 결과 비교")
    print("=" * 70)
    print(f"  {'알파':<30} {'Sharpe':>8} {'Fitness':>8} {'Turnover':>9} {'Pass':>6}")
    print("  " + "-" * 66)

    passed = []
    for r in results:
        s  = r.get("sharpe")
        f  = r.get("fitness")
        t  = r.get("turnover")
        ok = "✓" if r.get("passes") else "✗"
        print(f"  {r['name']:<30} "
              f"{(f'{s:.3f}' if s else 'N/A'):>8} "
              f"{(f'{f:.3f}' if f else 'N/A'):>8} "
              f"{(f'{t:.3f}' if t else 'N/A'):>9} "
              f"{ok:>6}")
        if r.get("passes"):
            passed.append(r["name"])

    print("  " + "-" * 66)
    print(f"\n  통과 알파: {len(passed)} / {len(results)}")
    if passed:
        print(f"  → {', '.join(passed)}")

    print("\n  [다음 스텝 제안]")
    # 진단별 개선 방향
    for r in results:
        s = r.get("sharpe")
        f = r.get("fitness")
        sc = r.get("self_correlation")
        if not r.get("passes") and s is not None:
            if s < 0:
                print(f"  • {r['name']}: Sharpe 음수 → 알파 앞에 -1 곱하기")
            elif s < 1.25 and f and f >= 1.0:
                print(f"  • {r['name']}: Sharpe 부족 → ts_rank 감싸기 또는 decay 조정")
            elif sc and abs(sc) >= 0.7:
                print(f"  • {r['name']}: 자기상관 초과 → ts_delta 또는 moving avg 차분 도입")
            elif f and f < 1.0:
                print(f"  • {r['name']}: Fitness 부족 → turnover 확인, decay 줄이기")
    print("=" * 70)


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ML 알파 → Brain 배치 시뮬레이션")
    parser.add_argument("--email",    required=True,  help="Brain 계정 이메일")
    parser.add_argument("--password", required=True,  help="Brain 계정 비밀번호")
    parser.add_argument("--start",    type=int, default=0,
                        help="알파 시작 인덱스 (0-based, 기본: 0)")
    parser.add_argument("--end",      type=int, default=len(ML_ALPHAS),
                        help=f"알파 종료 인덱스 (기본: {len(ML_ALPHAS)})")
    parser.add_argument("--submit-only",  action="store_true",
                        help="제출만 하고 결과 수집 안 함 (GitHub Actions submit 단계)")
    parser.add_argument("--collect-only", action="store_true",
                        help="이전에 저장된 IDs로 결과만 수집 (collect 단계)")
    parser.add_argument("--ids-file", default="submitted_ids.json",
                        help="제출된 Alpha ID 저장/로드 파일 경로")
    parser.add_argument("--wait",     type=int, default=120,
                        help="제출 후 초기 대기 시간(초), 기본 120")
    args = parser.parse_args()

    target_alphas = ML_ALPHAS[args.start:args.end]
    print(f"\n{'='*70}")
    print(f"  ML → BRAIN 배치 제출 | {len(target_alphas)}개 알파 (#{args.start}~#{args.end-1})")
    print(f"{'='*70}\n")

    session = get_session(args.email, args.password)

    # ── 제출 단계 ────────────────────────────────────────────────────────────
    if not args.collect_only:
        print("[SUBMIT] 알파 시뮬레이션 제출 시작...\n")
        submitted = []
        for i, alpha in enumerate(target_alphas):
            print(f"  [{i+1}/{len(target_alphas)}] {alpha['name']}")
            print(f"    식: {alpha['expression'][:80]}...")
            result = submit_alpha(session, alpha)
            if result:
                submitted.append(result)
            time.sleep(3)   # API rate limit 방지

        # ID 파일 저장
        ids_path = Path(args.ids_file)
        ids_path.write_text(json.dumps(submitted, indent=2, ensure_ascii=False))
        print(f"\n  ✓ {len(submitted)}개 ID 저장 → {ids_path}")

        if args.submit_only:
            print(f"\n  [submit-only 모드] 제출 완료. --collect-only로 나중에 결과 수집.")
            print(f"  Brain 시뮬레이션 시간: 보통 5~15분 소요.")
            return

        # 시뮬레이션 완료 대기
        print(f"\n  [{args.wait}초] 시뮬레이션 완료 대기 중...")
        time.sleep(args.wait)

    # ── 수집 단계 ────────────────────────────────────────────────────────────
    ids_path = Path(args.ids_file)
    if ids_path.exists():
        submitted = json.loads(ids_path.read_text())
    else:
        print(f"  ✗ ID 파일 없음: {ids_path}")
        return

    print("\n[COLLECT] 시뮬레이션 결과 수집...\n")
    results = []
    for item in submitted:
        alpha_id = item.get("id")
        name     = item.get("name", alpha_id)
        if not alpha_id:
            continue
        print(f"  조회 중: {name} (ID: {alpha_id})")
        result = collect_result(session, alpha_id, name)
        results.append(result)

    # 결과 저장
    out_path = Path("brain_results.json")
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n  ✓ 결과 저장 → {out_path}")

    # 최종 리포트
    print_report(results)


if __name__ == "__main__":
    main()
