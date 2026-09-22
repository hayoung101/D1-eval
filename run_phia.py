"""
run_phia.py — PHIA Objective Query 4,000문항의 정답을 처음부터 끝까지 생성한다.

  python run_phia.py --excel <v3 엑셀> --out <출력 폴더>

입력 (모두 wearable_bench/data/ 안)
  summary_df_{uid}.csv, exercise_df_{uid}.csv         wearable 데이터
  {persona}_final_v2.csv                              질문(엑셀본)·실행가능 code·원본 label
  <excel>                                             대조용 (선택)

출력
  ID{uid}{persona}_final_v7.csv   question / label / code / excel_row / null_reason
  answers_v7.csv                  4,000행 통합

처리 순서
  1. 어댑터로 표를 읽어 UserContext 를 만든다            (기준일 = 참조 표 최신일의 max)
  2. data/*.csv 의 code 를 그대로 실행해 정답과 상태(NULL / 결함)를 얻는다
  3. 원본 label 및 v3 엑셀과 대조한 결과를 함께 적는다

질문 교정(E3/E4/E6/E7), 표기 복구(R1~R3), on_days 변환(C2/C3) 은 모두
data/*.csv 의 question·code 컬럼에 이미 반영되어 있다. 런타임 변환은 없다.

--golden 을 주면 정답을 쓰지 않고 기존 산출물과 대조만 한다.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wearable_bench.adapters import phia                      # noqa: E402
from wearable_bench.core.context import Semantics             # noqa: E402
from wearable_bench.core.grading import NULL, equal           # noqa: E402

# persona, user_id, 엑셀 첫 행 번호 (머리글 1행)
USERS = [
    ("health_behavior", 465, 2),
    ("inactive_insomniacs", 171, 1002),
    ("sedentary_sleeper", 333, 2002),
    ("active_achiver", 41, 3002),
]


HERE = os.path.dirname(os.path.abspath(__file__))


def build(sem: Semantics,
          data_dir: str | None = None) -> pd.DataFrame:
    # 질문·code·wearable 데이터 모두 wearable_bench/data 에 있다.
    data_dir = data_dir or os.path.join(HERE, "data")
    eval_dir = data_dir

    rows = []
    for persona, uid, first_row in USERS:
        ctx = phia.load(uid, data_dir, sem)
        gt = pd.read_csv(os.path.join(eval_dir, f"{persona}_final_v2.csv"))

        for i, r in gt.iterrows():
            xrow = first_row + i
            question = r["question"]

            res = ctx.execute(r["code"])        # code 는 이미 실행 가능한 교정본
            value = (NULL if res.is_null
                     else f"ERR:{res.status}" if res.is_defect
                     else res.value)

            rows.append(dict(
                excel_row=xrow, persona=persona, user_id=uid, row=i,
                question=question,
                code=r["code"],
                label=value, null_reason=res.status or "",
                today=res.anchor, tables=",".join(res.tables),
                phia_label=r["label"],
            ))
    return pd.DataFrame(rows)


def report(d: pd.DataFrame, excel: str | None):
    n = len(d)
    nnull = int((d["label"].astype(str) == NULL).sum())
    nerr = int(d["label"].astype(str).str.startswith("ERR").sum())
    print(f"총 {n}행")
    print(f"  NULL          {nnull}")
    print(f"  수치·범주      {n - nnull - nerr}")
    print(f"  실행 오류      {nerr}")
    if nnull:
        print("\nNULL 사유:")
        print(d.loc[d["label"].astype(str) == NULL, "null_reason"]
              .value_counts().to_string())
    if excel:
        v = pd.read_excel(excel, sheet_name="Sheet1")
        final = v["수정된정답"].where(v["수정된정답"].notna(), v["기존 PHIA 정답"])
        same = sum(equal(a, b) for a, b in zip(d["label"], final))
        print(f"\nv3 엑셀과 일치: {same} / {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", default=None, help="v3 엑셀 (대조용, 선택)")
    ap.add_argument("--out", default="v7_out")
    ap.add_argument("--empty-set-policy", default="null_except_count",
                    choices=["keep", "no_data_only", "null_except_count", "null_all"])
    ap.add_argument("--last-days-window", default="exclude_today",
                    choices=["exclude_today", "include_today"])
    ap.add_argument("--golden", default=None,
                    help="기존 answers_v7.csv 와 대조만 하고 쓰지 않는다")
    args = ap.parse_args()

    sem = Semantics(empty_set_policy=args.empty_set_policy,
                    last_days_window=args.last_days_window)
    print(f"설정: {sem}\n")

    d = build(sem)
    report(d, args.excel)

    if args.golden:
        old = pd.read_csv(args.golden)
        diff = [i for i, (a, b) in enumerate(zip(old["label"], d["label"]))
                if not equal(a, b)]
        print(f"\n골든 테스트: 불일치 {len(diff)} / {len(d)}")
        if diff:
            print(d.iloc[diff][["excel_row", "question", "label"]].head(20).to_string())
        return

    os.makedirs(args.out, exist_ok=True)
    cols = ["question", "label", "code", "excel_row", "null_reason"]
    for persona, uid, _ in USERS:
        s = d[d["user_id"] == uid]
        s[cols].to_csv(os.path.join(args.out, f"ID{uid}{persona}_final_v7.csv"),
                       index=False, encoding="utf-8-sig")
    d.to_csv(os.path.join(args.out, "answers_v7.csv"), index=False, encoding="utf-8-sig")
    print(f"\n출력: {args.out}/")


if __name__ == "__main__":
    main()
