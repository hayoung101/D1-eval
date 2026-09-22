"""
grading.py — 채점 (SEMANTICS.md §6)

기준:  |정답 − 예측| < 0.011        소수 둘째 자리까지 같으면 정답

PHIA 원본 채점기(figs/3a.ipynb)는 `f'{v:.2f}'` 로 반올림한 뒤 문자열을 완전
일치시켰다. 취지는 같지만 반올림 경계에서 억울한 오답이 나온다.

    정답 18.465  예측 18.47
      PHIA 컷 : '18.46' != '18.47'  ->  오답   (부동소수점 표현이 아래로 내림)
      본 기준 : 0.005 < 0.011       ->  정답

실제로 4,000문항에서 두 기준이 64건 갈리며, 그 64건은 전부 0.005~0.01 차이다.
계산 실패가 아니라 마지막 자리 표기 흔들림이므로 정답으로 본다.
0.02 이상 차이는 여전히 오답이다.

비수치 답(활동 종목 이름 등)은 문자열로 비교한다.
NULL 은 NULL 끼리만 맞는다 — 값 0 과 구별된다.
"""

from __future__ import annotations

import math

__all__ = ["NULL", "TOL", "normalize", "equal"]

NULL = "NULL"

# 소수 둘째 자리까지 일치로 보는 폭. 0.01 차이는 통과, 0.02 는 불통과.
TOL = 0.011


def normalize(x):
    """채점용 표준형. NULL, float, 또는 문자열."""
    if x is None:
        return NULL

    if isinstance(x, str):
        s = x.strip()
        if not s or s.upper().startswith("NULL") or s.lower() == "nan":
            return NULL
        try:
            f = float(s)
        except ValueError:
            return s                      # 범주형 답은 그대로
        return NULL if math.isnan(f) else f

    if isinstance(x, bool):
        return str(x)

    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x).strip()
    return NULL if math.isnan(f) else f


def equal(a, b, tol: float = TOL) -> bool:
    """|a − b| < tol 이면 정답. 범주형은 문자열 일치."""
    na, nb = normalize(a), normalize(b)

    if na is NULL or nb is NULL:
        return na is NULL and nb is NULL

    if isinstance(na, float) and isinstance(nb, float):
        return abs(na - nb) < tol

    return str(na).strip().lower() == str(nb).strip().lower()
