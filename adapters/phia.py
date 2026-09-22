"""
adapters/phia.py — PHIA 합성 사용자 로더

summary_df_{uid}.csv  ->  daily_metrics
exercise_df_{uid}.csv ->  exercise_entries

원본 정밀도를 그대로 유지한다 (SEMANTICS §6.1). 컬럼을 떨어뜨리지 않는다.
"""

from __future__ import annotations

import os

import pandas as pd

from ..core.context import Semantics, UserContext

PHIA_USERS = {
    465: "health_behavior",
    171: "inactive_insomniacs",
    333: "sedentary_sleeper",
    41: "active_achiver",
}


_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load(user_id, data_dir=None, semantics: Semantics | None = None) -> UserContext:
    data_dir = data_dir or _DATA_DIR
    sm = pd.read_csv(os.path.join(data_dir, f"summary_df_{user_id}.csv"))
    ex = pd.read_csv(os.path.join(data_dir, f"exercise_df_{user_id}.csv"))

    sm["datetime"] = pd.to_datetime(sm["datetime"])
    # 공개 exercise CSV 의 날짜 컬럼명은 'date' 다. 정답 코드는 'datetime' 을 참조하므로
    # 같은 값을 datetime 이름으로도 둔다.
    ex["datetime"] = pd.to_datetime(ex["date"] if "date" in ex.columns else ex["datetime"])

    daily_metrics = sm.set_index(pd.DatetimeIndex(sm["datetime"])).sort_index()
    exercise_entries = ex.set_index(pd.DatetimeIndex(ex["datetime"])).sort_index()

    return UserContext(
        user_id,
        {"daily_metrics": daily_metrics, "exercise_entries": exercise_entries},
        semantics,
    )
