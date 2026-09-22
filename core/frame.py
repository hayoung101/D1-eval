"""
frame.py — 시계열 표와 기간/조건 질의

SEMANTICS.md §1~§3 의 구현.

  during(문자열)   기간 구간.  기준일(today)을 쓴다.
  on_days(날짜들)  조건 집합.  기준일을 쓰지 않는다.
  where(조건)      조건을 만족하는 날짜들을 돌려준다.

전역 상태를 두지 않는다. 기준일은 표 객체가 직접 들고 다니며(_anchor),
pandas 연산(.query(), 불리언 인덱싱 등) 뒤에도 _metadata 를 통해 전달된다.
"""

from __future__ import annotations

import datetime
import re

import pandas as pd

__all__ = ["Table"]


_LAST_DAYS = re.compile(r"^(?:last|past)\s+(\d+)\s+days?$")
_LAST_WEEK = re.compile(r"^(?:last|past)\s+week$")
_LAST_WEEKS = re.compile(r"^(?:last|past)\s+(\d+)\s+weeks?$")
_LAST_MONTH = re.compile(r"^(?:last|past)\s+month$")


class Table(pd.DataFrame):
    """DatetimeIndex 를 가진 표.

    _anchor  : 이 표에 적용할 기준일(today). 실행 직전에 주입된다.
               SEMANTICS §1 에 따라 '참조 표들의 최신일 중 max' 가 들어온다.
    _tname   : 표 이름 (진단용)
    _sem     : Semantics 설정 객체
    _sink    : 빈 조건 발생을 기록할 리스트 (중첩 호출에서도 잡히도록)
    """

    _metadata = ["_anchor", "_tname", "_sem", "_sink", "_base"]

    @property
    def _constructor(self):
        # 이게 없으면 .query() / .head() 결과가 평범한 DataFrame 이 되어
        # 이어지는 .during() 이 사라진다.
        return Table

    # ------------------------------------------------------------------ 기간

    def during(self, expr):
        """기간 구간을 자른다. 문자열만 받는다 (SEMANTICS §8-3).

        결과가 비면 원인을 구분해 기록한다.
          no_data   그 기간에 원본 표 자체에 기록이 없다   -> 답할 수 없음
          no_match  기록은 있는데 앞선 필터에 걸러졌다      -> 대상이 실제로 0
        """
        if not isinstance(expr, str):
            raise TypeError(
                "during() 은 기간 문자열만 받습니다. "
                "조건(날짜 집합)은 on_days() 를 쓰세요."
            )

        lo, hi = self._window(expr)
        if lo is None:
            return self.iloc[0:0]

        d = self.index.date
        out = self.loc[(d >= lo) & (d <= hi)]
        if len(out) == 0:
            self._note_empty(lambda b: b.loc[(b.index.date >= lo) & (b.index.date <= hi)])
        return out

    def _window(self, expr):
        """기간 표현 -> (시작일, 종료일)."""
        anchor = self._anchor
        if anchor is None:
            return None, None

        t = expr.lower().strip()
        sem = self._sem

        if t == "today":
            lo = hi = anchor
        elif t == "yesterday":
            lo = hi = anchor - datetime.timedelta(days=1)
        elif _LAST_WEEK.match(t):
            lo, hi = _calendar_weeks(anchor, 1)
        elif (m := _LAST_WEEKS.match(t)) is not None:
            n = int(m.group(1))
            if n <= 0:
                return None, None
            lo, hi = _calendar_weeks(anchor, n)
        elif _LAST_MONTH.match(t):
            lo, hi = _last_month(anchor)
        elif (m := _LAST_DAYS.match(t)) is not None:
            n = int(m.group(1))
            if n <= 0:
                return None, None
            one = datetime.timedelta(days=1)
            if sem.last_days_window == "exclude_today":
                lo, hi = anchor - n * one, anchor - one       # [today-N, today-1]
            else:
                lo, hi = anchor - (n - 1) * one, anchor       # [today-(N-1), today]
        else:
            try:
                lo = hi = pd.to_datetime(expr).date()
            except Exception as exc:
                raise ValueError(f"알 수 없는 기간 표현: {expr!r}") from exc

        return lo, hi

    def _note_empty(self, slicer):
        """빈 결과의 원인을 sink 에 기록한다."""
        if self._sink is None:
            return
        base = self._base
        if base is None:
            self._sink.append("no_data")
            return
        self._sink.append("no_match" if len(slicer(base)) else "no_data")

    # ------------------------------------------------------------------ 조건

    def where(self, condition):
        """조건을 만족하는 행들의 날짜를 돌려준다 (SEMANTICS §3.1).

        컬럼 이름('datetime')에 의존하지 않는다.
        """
        hit = super().query(condition)
        return pd.DatetimeIndex(pd.to_datetime(pd.Series(hit.index)).dt.normalize().unique())

    def on_days(self, days):
        """조건을 만족하는 날들과 교집합을 취한다 (SEMANTICS §3.2).

        기준일을 참조하지 않는다.
        """
        wanted = _as_date_set(days)
        if not wanted:
            if self._sink is not None:
                self._sink.append("empty_condition")
            return self.iloc[0:0]
        d = self.index.date
        out = self.loc[[x in wanted for x in d]]
        if len(out) == 0:
            self._note_empty(lambda b: b.loc[[x in wanted for x in b.index.date]])
        return out

    # ------------------------------------------------------------------ 진단

    @property
    def anchor(self):
        return self._anchor

    @property
    def table_name(self):
        return self._tname


# ---------------------------------------------------------------------- 보조


def _as_date_set(days):
    if days is None:
        return set()
    if isinstance(days, pd.Timestamp):
        return set() if pd.isna(days) else {days.date()}
    if isinstance(days, datetime.datetime):
        return {days.date()}
    if isinstance(days, datetime.date):
        return {days}
    if isinstance(days, str):
        return {pd.to_datetime(days).date()}
    if isinstance(days, (pd.DatetimeIndex, pd.Series, pd.Index)):
        if len(days) == 0:
            return set()
        return {d.date() for d in pd.to_datetime(pd.Series(days))}
    if isinstance(days, (list, tuple, set)):
        if not days:
            return set()
        return {d.date() if hasattr(d, "date") else pd.to_datetime(d).date() for d in days}
    raise TypeError(f"on_days() 가 받을 수 없는 형식: {type(days)}")


def _calendar_weeks(anchor, n):
    """직전 n개 달력 주 구간. 주는 일요일에 시작해 토요일에 끝난다 (SEMANTICS §2.2).

      anchor 2026-09-21(월), n=1 -> 2026-09-13(일) ~ 2026-09-19(토)
      anchor 2026-09-21(월), n=2 -> 2026-09-06(일) ~ 2026-09-19(토)
    이번 주(anchor 가 속한 주)는 포함하지 않는다.
    """
    sunday_of_this_week = anchor - datetime.timedelta(days=(anchor.weekday() + 1) % 7)
    lo = sunday_of_this_week - datetime.timedelta(days=7 * n)
    hi = sunday_of_this_week - datetime.timedelta(days=1)
    return lo, hi


def _last_month(anchor):
    """직전 달력 월 전체 (SEMANTICS §2.2).

      anchor 2023-10-31 -> 2023-09-01 ~ 2023-09-30
      anchor 2024-03-15 -> 2024-02-01 ~ 2024-02-29
    'last 30 days' 와 다르다. 이번 달(anchor 가 속한 달)은 포함하지 않는다.
    """
    first_of_this_month = anchor.replace(day=1)
    hi = first_of_this_month - datetime.timedelta(days=1)   # 지난달 말일
    lo = hi.replace(day=1)                                    # 지난달 1일
    return lo, hi
