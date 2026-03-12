"""
WorldQuant BRAIN Alpha Search
==============================
알파 수식 자동 생성 → BRAIN API 제출 → 결과 수집 → 필터링

사용법:
    pip install requests pandas
    python alpha_search.py

환경변수 설정:
    export BRAIN_EMAIL=your@email.com
    export BRAIN_PASSWORD=yourpassword
"""

import os
import time
import json
import requests
import pandas as pd
from datetime import datetime
from requests.auth import HTTPBasicAuth
from generators.single_factor import generate_single_alphas
from generators.combo_factor import generate_combo_alphas

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
SETTINGS = {
    "instrumentType": "EQUITY",
    "region": "USA",
    "universe": "TOP3000",
    "delay": 1,
    "decay": 1,
    "neutralization": "SUBINDUSTRY",
    "truncation": 0.08,
    "pasteurization": "ON",
    "unitHandling": "VERIFY",
    "nanHandling": "OFF",
    "language": "FASTEXPR",
    "visualization": False,
}

# 제출 기준 필터
FILTER = {
    "min_sharpe":   1.5,
    "min_fitness":  1.0,
    "min_turnover": 0.01,
    "max_turnover": 0.70,
}

# 오늘 실험으로 파악된 블랙리스트 신호
BLACKLIST = [
    "close / ts_delay(close",   # 가격 모멘텀 → Subindustry에서 역방향
    "cap / ts_delay(cap",       # cap 모멘텀 → TEST 악화
]


# ─────────────────────────────────────────────
# 1. BRAIN API 세션
# ─────────────────────────────────────────────
def login(email: str, password: str) -> requests.Session:
    session = requests.Session()
    res = session.post(
        "https://api.worldquantbrain.com/authentication",
        auth=(email, password)
    )
    
    print(f"  로그인 응답 코드: {res.status_code}")
    
    # 200, 201 둘 다 성공으로 처리
    if res.status_code not in (200, 201):
        raise Exception(f"로그인 실패: {res.text}")
    
    # 토큰 세션에 저장
    data = res.json()
    token = data.get("token", {})
    print(f"  ✅ 로그인 성공 | user: {data.get('user', {}).get('id')}")
    
    return session

# ─────────────────────────────────────────────
# 2. 알파 제출
# ─────────────────────────────────────────────
def submit_alpha(session: requests.Session, expression: str) -> str | None:
    try:
        res = session.post(
            "https://api.worldquantbrain.com/simulations",
            json={"type": "REGULAR", "settings": SETTINGS, "regular": expression}
        )
        if res.status_code == 201:
            return res.json().get("id")
        else:
            print(f"  ⚠️  제출 실패: {res.status_code} | {expression[:40]}")
            return None
    except Exception as e:
        print(f"  ❌ 오류: {e}")
        return None


# ─────────────────────────────────────────────
# 3. 결과 조회
# ─────────────────────────────────────────────
def get_result(session: requests.Session, sim_id: str, timeout: int = 180) -> dict | None:
    for _ in range(timeout):
        try:
            res = session.get(
                f"https://api.worldquantbrain.com/simulations/{sim_id}"
            )
            data = res.json()
            status = data.get("status")

            if status == "COMPLETE":
                is_stats = data.get("is", {})
                return {
                    "sim_id":   sim_id,
                    "sharpe":   is_stats.get("sharpe"),
                    "fitness":  is_stats.get("fitness"),
                    "turnover": is_stats.get("turnover"),
                    "returns":  is_stats.get("returns"),
                    "drawdown": is_stats.get("drawdown"),
                    "margin":   is_stats.get("margin"),
                }
            elif status in ("ERROR", "FAILED"):
                return None

        except Exception:
            pass

        time.sleep(1)
    return None


# ─────────────────────────────────────────────
# 4. 배치 실행
# ─────────────────────────────────────────────
def run_batch(session: requests.Session, alphas: list[str],
              batch_size: int = 10) -> pd.DataFrame:
    results = []
    total = len(alphas)

    print(f"\n🚀 총 {total}개 알파 시뮬레이션 시작\n")

    for i in range(0, total, batch_size):
        batch = alphas[i:i + batch_size]
        sim_map = {}  # sim_id → expression

        # 배치 제출
        for expr in batch:
            sid = submit_alpha(session, expr)
            if sid:
                sim_map[sid] = expr
            time.sleep(0.5)  # rate limit 방지

        # 결과 수집
        for sid, expr in sim_map.items():
            result = get_result(session, sid)
            if result:
                result["expression"] = expr
                results.append(result)
                sharpe = result.get("sharpe", 0) or 0
                emoji = "🟢" if sharpe >= 1.5 else "🟡" if sharpe >= 1.0 else "🔴"
                print(f"  {emoji} Sharpe {sharpe:.2f} | {expr[:60]}")

        progress = min(i + batch_size, total)
        print(f"\n[{progress}/{total}] 완료 ({progress/total*100:.0f}%)\n")

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# 5. 필터링
# ─────────────────────────────────────────────
def filter_alphas(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    mask = (
        (df["sharpe"]   >= FILTER["min_sharpe"]) &
        (df["fitness"]  >= FILTER["min_fitness"]) &
        (df["turnover"] >= FILTER["min_turnover"]) &
        (df["turnover"] <= FILTER["max_turnover"])
    )
    return df[mask].sort_values("sharpe", ascending=False)


# ─────────────────────────────────────────────
# 6. 메인
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  🧠 WorldQuant BRAIN Alpha Search")
    print("=" * 60)

    # 환경변수에서 인증 정보 로드
    email    = os.environ.get("BRAIN_EMAIL")
    password = os.environ.get("BRAIN_PASSWORD")
    if not email or not password:
        raise ValueError("환경변수 BRAIN_EMAIL, BRAIN_PASSWORD를 설정하세요")

    # 로그인
    session = login(email, password)

    # 알파 생성
    print("\n📐 알파 수식 생성 중...")
    single_alphas = generate_single_alphas()
    combo_alphas  = generate_combo_alphas()
    all_alphas    = single_alphas + combo_alphas

    # 블랙리스트 필터링
    all_alphas = [a for a in all_alphas
                  if not any(b in a for b in BLACKLIST)]
    all_alphas = list(set(all_alphas))  # 중복 제거

    print(f"  단일 신호: {len(single_alphas)}개")
    print(f"  조합 신호: {len(combo_alphas)}개")
    print(f"  블랙리스트 제거 후: {len(all_alphas)}개")

    # 배치 실행
    df = run_batch(session, all_alphas, batch_size=10)

    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path  = f"results/all_results_{timestamp}.csv"
    df.to_csv(raw_path, index=False)
    print(f"\n💾 전체 결과 저장: {raw_path}")

    # 필터링
    good = filter_alphas(df)
    if not good.empty:
        good_path = f"results/good_alphas_{timestamp}.csv"
        good.to_csv(good_path, index=False)
        print(f"🏆 기준 통과 알파: {len(good)}개 → {good_path}")
        print("\n  Top 10:")
        print(good[["sharpe", "fitness", "turnover", "expression"]].head(10).to_string())
    else:
        print("😅 기준 통과 알파 없음 — 파라미터 조정 필요")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
