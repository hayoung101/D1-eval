"""
context.py — 사용자 단위 실행 컨텍스트와 설정

전역 가변 상태를 두지 않는다 (SEMANTICS §8-6). 한 사용자의 표·기준일·설정은
UserContext 하나에 담기므로, 여러 사용자를 동시에 다뤄도 서로 섞이지 않는다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .frame import Table

__all__ = ["Semantics", "UserContext", "Result"]


# ---------------------------------------------------------------- 설정


@dataclass(frozen=True)
class Semantics:
    """SEMANTICS.md §7. '우리가 내린 결정'만 둔다. 버그 재현 스위치는 없다."""

    last_days_window: str = "exclude_today"   # exclude_today | include_today
    null_on_empty: bool = True
    # 집계 대상이 빈 집합일 때의 처리 (SEMANTICS §5)
    #   "keep"             pandas 기본. sum->0, mean->NaN   (일관성 없음)
    #   "null_except_count" 개수 계열만 0, 나머지는 NULL      <- 기본값
    #   "null_all"         개수 계열도 NULL
    empty_set_policy: str = "null_except_count"

    def variant(self, **kw) -> "Semantics":
        """민감도 분석용 변형본."""
        return replace(self, **kw)


# ---------------------------------------------------------------- 실행 결과


@dataclass
class Result:
    value: object = None
    status: str | None = None      # None=정상, "NULL:..."=답없음, 그 외=문항 결함
    anchor: object = None
    tables: tuple = ()

    @property
    def ok(self) -> bool:
        return self.status is None

    @property
    def is_null(self) -> bool:
        return bool(self.status) and self.status.startswith("NULL")

    @property
    def is_defect(self) -> bool:
        return bool(self.status) and not self.status.startswith("NULL")


# ---------------------------------------------------------------- 컨텍스트

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# '몇 번 / 몇 일' 을 묻는 집계. 빈 집합의 개수는 0 으로 잘 정의된다.
_COUNTING = {"count", "nunique", "size", "shape"}
_AGG = re.compile(r"\.([A-Za-z_]\w*)\s*(?:\(\s*\)|\[)")


def _final_agg(code: str) -> str:
    m = _AGG.findall(str(code))
    return m[-1] if m else ""


class UserContext:
    """한 사용자의 표들과 기준일을 들고 정답 프로그램을 실행한다."""

    def __init__(self, user_id, tables: dict[str, pd.DataFrame], semantics: Semantics | None = None):
        self.user_id = user_id
        self.sem = semantics or Semantics()
        self._tables: dict[str, Table] = {}
        self.last_day: dict[str, object] = {}

        for name, df in tables.items():
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(f"표 '{name}' 의 인덱스가 DatetimeIndex 가 아닙니다")
            t = Table(df.sort_index())
            t._tname = name
            t._sem = self.sem
            t._anchor = None
            t._sink = None
            t._base = t
            self._tables[name] = t
            self.last_day[name] = t.index.max().date() if len(t) else None

    # ------------------------------------------------------------ 기준일

    def referenced(self, code: str) -> tuple:
        """정답 프로그램이 이름으로 언급하는 표들 (실행 전 정적 판정, SEMANTICS §1.1)."""
        names = set(_IDENT.findall(str(code)))
        return tuple(n for n in self._tables if n in names)

    def anchor_for(self, code: str):
        """today = 참조 표들의 최신일 중 가장 늦은 날 (SEMANTICS §1)."""
        used = self.referenced(code)
        days = [self.last_day[n] for n in used if self.last_day[n] is not None]
        return max(days) if days else None

    # ------------------------------------------------------------ 실행

    def execute(self, code: str) -> Result:
        code = str(code).strip()
        used = self.referenced(code)
        anchor = self.anchor_for(code)
        sink: list = []

        for name in self._tables:
            t = self._tables[name]
            t._anchor = anchor
            t._sink = sink
            t._sem = self.sem

        env = {"pd": pd, "np": np}
        env.update(self._tables)

        try:
            value = eval(code, {"__builtins__": {}}, env)   # noqa: S307
        except IndexError:
            # 빈 결과에서 값을 꺼내려 함 = 대상 기록 없음
            if self.sem.null_on_empty:
                return Result(None, "NULL:대상 기록 없음", anchor, used)
            return Result(None, "IndexError: 빈 결과에서 값 추출", anchor, used)
        except Exception as exc:
            return Result(None, f"{type(exc).__name__}: {exc}", anchor, used)

        if isinstance(value, (pd.Series, pd.DataFrame, pd.Index)):
            return Result(None, f"스칼라 아님 ({type(value).__name__})", anchor, used)

        # 조건(where)을 만족하는 날이 하루도 없으면 집계할 대상이 없다 -> NULL.
        # 빈 집합의 sum 이 0 이 되는 pandas 동작을 그대로 쓰지 않는다 (SEMANTICS §5.2).
        if self.sem.null_on_empty and "empty_condition" in sink:
            return Result(None, "NULL:조건 만족일 없음", anchor, used)

        # 집계 대상이 빈 집합이었다면 (SEMANTICS §5)
        pol = self.sem.empty_set_policy
        if pol != "keep" and ("no_data" in sink or "no_match" in sink):
            counting = _final_agg(code) in _COUNTING
            if "no_data" in sink:
                # 그 기간에 기록 자체가 없다 -> 답할 수 없음
                if pol == "no_data_only" or not counting or pol == "null_all":
                    return Result(None, "NULL:기간 내 기록 없음", anchor, used)
            elif pol in ("null_except_count", "null_all"):
                # 기록은 있으나 조건에 맞는 대상이 없다
                if pol == "null_all" or not counting:
                    return Result(None, "NULL:대상 없음", anchor, used)

        try:
            if value is not None and float(value) != float(value):   # NaN
                return Result(None, "NULL:집계 결과 없음", anchor, used)
        except (TypeError, ValueError):
            pass

        return Result(value, None, anchor, used)

    # ------------------------------------------------------------ 기타

    def table(self, name) -> Table:
        return self._tables[name]

    @property
    def table_names(self) -> tuple:
        return tuple(self._tables)

    def with_semantics(self, **kw) -> "UserContext":
        """설정만 바꾼 동일 컨텍스트 (민감도 분석용)."""
        return UserContext(
            self.user_id,
            {n: pd.DataFrame(t) for n, t in self._tables.items()},
            self.sem.variant(**kw),
        )

    def __repr__(self):
        days = ", ".join(f"{n}={self.last_day[n]}" for n in self._tables)
        return f"<UserContext {self.user_id} [{days}]>"
