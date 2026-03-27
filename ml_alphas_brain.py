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
    "decay":           5,          # decay=5: turnover 감소 → fitness 개선 (Fitness = Sharpe*sqrt(|ret|/max(TO,0.125)))
    "neutralization":  "SUBINDUSTRY",
    "truncation":      0.08,
    "unitHandling":    "VERIFY",
    "pasteurization":  "ON",
    "nanHandling":     "OFF",
    "language":        "FASTEXPR",
    "visualization":   False,
}

# ── 알파 표현식 설계 근거 ──────────────────────────────────────────────────────
#
#  [설계 원칙 — v2 개정판]
#  1. 검증된 패턴 우선: 실제 BRAIN 통과 사례(Sharpe≥1.25, Fitness≥1.0) 기반
#  2. decay=5 : 펀더멘털 저주파 신호에서 turnover 낮춤 → Fitness = Sharpe*sqrt(|ret|/max(TO,0.125)) 개선
#  3. ts_rank(x, 40~252) 구조: 크로스섹셔널 정규화 + 시계열 랭킹 이중 필터
#  4. ebit 기반 가치 지표: EV/EBITDA 역수, EBIT/cap 등 — 높은 IC 확인됨
#  5. 영업현금흐름 / cap 조합: 실험에서 양수 TEST 확인
#  6. group_rank(alpha, subindustry): 섹터 내 상대 랭킹으로 중립화 효과 강화
#
#  [근거 데이터 — 실제 통과 알파 참조]
#  • rank(ts_rank(ebit/sharesout/close, 40))       → Sharpe 1.80, Fitness 1.42
#  • -rank(ebit/capex)                             → Sharpe 1.62, Fitness 1.70
#  • -ts_rank(retained_earnings, 500)              → Sharpe 1.55, Fitness 1.18
#  • -ts_zscore(enterprise_value/ebitda, 63)       → Sharpe 2.58, Fitness 1.70
#  • ts_rank(cashflow_op/cap, 60) + group_rank     → 검증된 구조
#  • SMA mean-reversion rank(SMA_30 - close)       → 간단하나 유효

ML_ALPHAS = [
    # ─────────────────────────────────────────────────────────────────────────
    # v1: EBIT/EV 가치 신호 — EV/EBITDA 역수 (기업가치 대비 영업이익)
    #   EV/EBITDA가 낮은 종목(저평가) → long
    #   ts_zscore로 시계열 정규화 → 크로스섹셔널 편향 제거
    #   참조: -ts_zscore(enterprise_value/ebitda, 63) → Sharpe 2.58, Fitness 1.70
    # ─────────────────────────────────────────────────────────────────────────
    {
        "name": "v1_ev_ebitda_zscore",
        "description": "EV/EBITDA 역신호 — enterprise_value/ebitda가 낮은 저평가 종목 long. "
                       "ts_zscore(63일)로 최근 추세 대비 상대적 저평가 포착. "
                       "참조 실적: Sharpe ~2.58, Fitness ~1.70 (TOP3000, Subindustry).",
        "expression": "-ts_zscore(enterprise_value / ebitda, 63)",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # v2: EBIT/주당 × 가격 역수 — 수익성 + 가치 결합
    #   ebit / sharesout / close = 주당 EBIT / 주가 = EBIT yield
    #   ts_rank(40일)로 단기 모멘텀 필터 + rank()로 크로스섹셔널 정규화
    #   참조: rank(ts_rank(ebit/sharesout/close, 40)) → Sharpe 1.80, Fitness 1.42
    #   [Fitness 개선 근거]
    #   Fitness = Sharpe × sqrt(|returns| / max(turnover, 0.125))
    #   TOP3000+decay5 → turnover ~14% → sqrt(returns/0.14) 낮음
    #   TOP1000+decay9+truncation0.1 → turnover ~14% 유사하나 returns↑ (유동성 집중)
    #   참조 실적이 TOP1000, decay=9, truncation=0.1 기준이므로 그대로 맞춤
    # ─────────────────────────────────────────────────────────────────────────
    {
        "name": "v2_ebit_yield_tsrank",
        "description": "EBIT Yield ts_rank — ebit/sharesout/close(EBIT 수익률)의 "
                       "최근 40일 시계열 순위를 크로스섹셔널 rank로 재랭킹. "
                       "참조 실적: Sharpe ~1.80, Fitness ~1.42 (TOP1000, decay=9, Subindustry).",
        "expression": "rank(ts_rank(ebit / sharesout / close, 40))",
        "settings": {
            "universe":   "TOP1000",   # TOP3000→TOP1000: 유동성 집중으로 returns↑
            "decay":      9,           # decay 9: turnover 낮춰 Fitness 분모 축소
            "truncation": 0.1,         # 참조 파라미터 그대로
        },
    },

    # ─────────────────────────────────────────────────────────────────────────
    # v3: EBIT/CapEx 역신호 — 자본효율성 (Jansen ch.4 ROIC 프록시)
    #   EBIT/CapEx = 투자 대비 영업이익 → 낮을수록 자본낭비 기업
    #   rank(-ebit/capex): 자본효율 낮은 기업 short, 높은 기업 long
    #   참조: -rank(ebit/capex) → Sharpe 1.62~2.02, Fitness 1.52~2.30
    # ─────────────────────────────────────────────────────────────────────────
    {
        "name": "v3_ebit_capex_efficiency",
        "description": "EBIT/CapEx 자본효율성 — ebit/capex 높은 종목(효율적 자본배분) long. "
                       "rank(-x) → 낮은 CapEx 대비 높은 EBIT 기업 선택. "
                       "참조 실적: Sharpe ~1.62, Fitness ~1.70 (TOP1000, Subindustry).",
        "expression": "-rank(ebit / capex)",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # v4: 영업현금흐름 수익률 + subindustry 그룹랭크 이중 필터
    #   cashflow_op/cap = 시가총액 대비 영업현금흐름 (진짜 현금창출력)
    #   ts_rank(60일)로 최근 현금흐름 개선 추세 포착
    #   group_rank(subindustry)로 동일 산업 내 상위 기업만 선별
    #   참조: ts_rank(cashflow_op/cap, 60) + group_rank 구조 → 검증됨
    # ─────────────────────────────────────────────────────────────────────────
    {
        "name": "v4_cashflow_yield_group",
        "description": "영업현금흐름 수익률 그룹랭크 — cashflow_op/cap의 ts_rank(60)을 "
                       "subindustry 내 group_rank로 재정렬. 동일 업종 내 현금흐름 최상위 기업 long. "
                       "두 단계 필터로 섹터 편향 제거 + 시계열 노이즈 감소.",
        "expression": "group_rank(ts_rank(cash_flow_from_operations / cap, 60), subindustry)",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # v5: 유보이익 ts_rank 역신호 — 자본배분 효율성
    #   retained_earnings가 과도하게 축적된 기업 → 재투자 기회 부재 or 주주환원 미흡
    #   -ts_rank(retained_earnings, 500): 유보이익 높은 기업 short
    #   500일 윈도우 → 장기 추세 제거 후 상대적 위치 파악
    #   참조: -ts_rank(retained_earnings, 500) → Sharpe 1.55, Fitness 1.18
    # ─────────────────────────────────────────────────────────────────────────
    {
        "name": "v5_retained_earnings_rank",
        "description": "유보이익 역신호 — retained_earnings의 500일 ts_rank 역수. "
                       "유보이익 과다 축적 기업은 자본배분 비효율 → short. "
                       "참조 실적: Sharpe ~1.55, Fitness ~1.18 (TOP3000, Subindustry).",
        "expression": "-ts_rank(retained_earnings, 500)",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # v6: 가격 평균 회귀 + 볼륨 조건부 거래
    #   SMA(30) - close: 현재 가격이 30일 평균보다 낮을수록 long (mean-reversion)
    #   volume > adv20 조건: 거래량이 평균 이상인 날만 거래 → 노이즈 감소
    #   참조: rank(SMA_30 - close) + trade_when(volume>adv20) 구조 → 검증됨
    # ─────────────────────────────────────────────────────────────────────────
    {
        "name": "v6_price_reversion_volume",
        "description": "가격 평균회귀 + 거래량 조건 — 30일 SMA 대비 가격 괴리를 역추종. "
                       "volume > adv20 조건으로 유동성 있는 날만 포지션 진입. "
                       "Mean-reversion + volume filter 조합.",
        "expression": (
            "event = volume > adv20;"
            " alpha = rank(ts_mean(close, 30) - close);"
            " trade_when(event, alpha, -1)"
        ),
    },

    # ─────────────────────────────────────────────────────────────────────────
    # v7: 매출자산회전율 × ROE 랭크 — 효율성 + 수익성 복합
    #   sales/assets = 자산 회전율 (운영 효율성)
    #   return_equity = ROE (자기자본 수익성)
    #   두 rank 곱: 양쪽 모두 상위인 종목 집중 선별 (AND 효과)
    #   참조: fam_roe_rank * rank(sales/assets) → Sharpe 1.45, Fitness 1.18
    # ─────────────────────────────────────────────────────────────────────────
    {
        "name": "v7_asset_turnover_roe",
        "description": "자산회전율 × ROE — sales/assets(효율성)와 return_equity(수익성) "
                       "양쪽 모두 높은 기업 선별. rank 곱으로 AND 효과 구현. "
                       "참조 실적: Sharpe ~1.45, Fitness ~1.18 (TOP3000, Market 중립화).",
        "expression": "rank(sales / assets) * rank(return_equity)",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # v8: 복합 앙상블 — 검증된 가치+효율 신호 선형 결합
    #   EV/EBITDA 역신호(가치) + EBIT yield(수익성) + 현금흐름 yield(품질)
    #   각각 독립적으로 검증된 신호를 균등 가중으로 결합
    #   decay=5로 turnover 안정화 → Fitness 개선
    #   outer rank()로 최종 정규화 후 SUBINDUSTRY 중립화
    # ─────────────────────────────────────────────────────────────────────────
    {
        "name": "v8_composite_value_quality",
        "description": "복합 가치+품질 앙상블 — EV/EBITDA 역신호 + EBIT yield tsrank + "
                       "현금흐름 수익률 tsrank 균등 결합. "
                       "개별 신호가 각각 검증된 패턴 기반, 결합으로 Sharpe 안정성 극대화.",
        "expression": (
            "rank("
            "  -ts_zscore(enterprise_value / ebitda, 63)"         # 가치 신호
            " + ts_rank(ebit / sharesout / close, 40)"            # 수익성 신호
            " + ts_rank(cash_flow_from_operations / cap, 60)"     # 현금흐름 신호
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
    """단일 알파 시뮬레이션 제출.

    alpha 딕셔너리에 'settings' 키가 있으면 BASE_SETTINGS를 알파별로 오버라이드합니다.
    예) "settings": {"universe": "TOP1000", "decay": 9, "truncation": 0.1}
    """
    settings = {**BASE_SETTINGS, **alpha.get("settings", {})}
    payload = {**settings, "code": alpha["expression"]}

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
