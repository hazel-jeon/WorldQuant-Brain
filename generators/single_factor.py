"""
단일 신호 알파 생성기
오늘 실험 결과를 반영한 유망 신호 위주로 구성
"""

# 오늘 실험으로 확인된 유망 펀더멘털 신호
FUNDAMENTAL_SIGNALS = [
    # 영업이익 기반 (오늘 TEST 양수 확인)
    "operating_income / ts_delay(operating_income, 252)",
    "operating_income / ts_delay(operating_income, 126)",
    "operating_income / ts_delay(operating_income, 63)",
    "operating_income / cap",

    # 매출 기반
    "sales / cap",
    "sales / ts_delay(sales, 252)",
    "sales / ts_delay(sales, 126)",

    # 수익성
    "earnings_per_share / close",
    "earnings_per_share / ts_delay(earnings_per_share, 252)",

    # 장부가치
    "book_to_price",
    "assets / cap",
    "assets / ts_delay(assets, 252)",

    # 현금흐름
    "cash_flow_from_operations / cap",
    "cash_flow_from_operations / ts_delay(cash_flow_from_operations, 252)",

    # 부채비율
    "debt / assets",
    "debt / cap",
]

TS_RANK_WINDOWS  = [126, 252]
GROUP_RANK_GROUPS = ["industry", "subindustry", "sector"]


def generate_single_alphas() -> list[str]:
    alphas = []

    for signal in FUNDAMENTAL_SIGNALS:
        # ts_rank 변형
        for window in TS_RANK_WINDOWS:
            alphas.append(f"ts_rank({signal}, {window})")

        # group_rank 변형
        for group in GROUP_RANK_GROUPS:
            alphas.append(f"group_rank({signal}, {group})")

        # 음수 버전 (역방향 베팅)
        for window in TS_RANK_WINDOWS:
            alphas.append(f"-ts_rank({signal}, {window})")

    return alphas
