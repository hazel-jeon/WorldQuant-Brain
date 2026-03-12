"""
조합 신호 알파 생성기
오늘 실험에서 검증된 조합 패턴 기반
"""

from itertools import combinations

# 오늘 실험으로 검증된 유망 신호 쌍
PROMISING_PAIRS = [
    # 오늘 최선 조합 (TEST -0.08)
    (
        "operating_income / ts_delay(operating_income, 252)",
        "operating_income / cap",
    ),
    # 매출 + 영업이익
    (
        "sales / ts_delay(sales, 252)",
        "operating_income / cap",
    ),
    # 영업이익 YoY + 현금흐름
    (
        "operating_income / ts_delay(operating_income, 252)",
        "cash_flow_from_operations / cap",
    ),
    # 영업이익 성장 + 장부가
    (
        "operating_income / ts_delay(operating_income, 252)",
        "book_to_price",
    ),
    # 수익성 + 밸류
    (
        "earnings_per_share / close",
        "book_to_price",
    ),
    # 영업이익 반기 + 연간
    (
        "operating_income / ts_delay(operating_income, 126)",
        "operating_income / ts_delay(operating_income, 252)",
    ),
]

# 오늘 최적으로 확인된 가중치 조합
WEIGHTS = [
    (0.75, 0.25),
    (0.60, 0.40),
    (0.50, 0.50),
    (0.70, 0.30),
]

TS_RANK_WINDOWS = [126, 252]


def generate_combo_alphas() -> list[str]:
    alphas = []

    for (sig1, sig2), (w1, w2) in [
        (pair, weight)
        for pair in PROMISING_PAIRS
        for weight in WEIGHTS
    ]:
        for window in TS_RANK_WINDOWS:
            # ts_rank 조합
            alpha = (
                f"{w1} * ts_rank({sig1}, {window})"
                f" + {w2} * ts_rank({sig2}, {window})"
            )
            alphas.append(alpha)

            # group_rank 조합
            alpha_gr = (
                f"{w1} * group_rank({sig1}, industry)"
                f" + {w2} * group_rank({sig2}, industry)"
            )
            alphas.append(alpha_gr)

            # ts_rank + group_rank 혼합
            alpha_mix = (
                f"{w1} * ts_rank({sig1}, {window})"
                f" + {w2} * group_rank({sig2}, industry)"
            )
            alphas.append(alpha_mix)

    # 3개 신호 조합 (오늘 최선 수식 변형)
    three_signal_alphas = [
        # 오늘 실험 베이스 + 추가 신호
        f"0.50 * ts_rank(operating_income / ts_delay(operating_income, 252), 252)"
        f" + 0.25 * ts_rank(operating_income / cap, 252)"
        f" + 0.25 * ts_rank(sales / cap, 252)",

        f"0.50 * ts_rank(operating_income / ts_delay(operating_income, 252), 252)"
        f" + 0.25 * ts_rank(operating_income / cap, 252)"
        f" + 0.25 * ts_rank(earnings_per_share / close, 252)",

        f"0.50 * ts_rank(operating_income / ts_delay(operating_income, 252), 252)"
        f" + 0.25 * ts_rank(operating_income / cap, 252)"
        f" + 0.25 * ts_rank(book_to_price, 252)",

        f"0.50 * ts_rank(operating_income / ts_delay(operating_income, 252), 252)"
        f" + 0.25 * ts_rank(operating_income / cap, 252)"
        f" + 0.25 * ts_rank(cash_flow_from_operations / cap, 252)",
    ]
    alphas.extend(three_signal_alphas)

    return list(set(alphas))
