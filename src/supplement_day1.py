"""Reproduce descriptive DAY1 charging-pattern and feature-correlation tables.

Reads existing processed cells and degradation slopes. It does not fit a model,
change cohorts/splits, or use the additional batch comparisons to select features.
Run: python src/supplement_day1.py
"""
from pathlib import Path
from itertools import product
import argparse
import hashlib
import json
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["qd_cycle2", "qd_change_100_10", "qd_slope_10_100", "ir_mean_2_100",
            "ir_change_early", "tavg_mean_2_100", "chargetime_mean_2_100", "log10_deltaq_var"]
PREDICTORS = ["c_rate_stage1", "c_rate_stage2", "transition_soc_pct"]
POLICY_PATTERN = re.compile(
    r"^\s*(?P<first>\d+(?:\.\d+)?)C\((?P<soc>\d+(?:\.\d+)?)%\)-"
    r"(?P<second>\d+(?:\.\d+)?)C(?P<suffix>.*)$")
TABLE_KEYS = ("policy_correlations", "policy_groups", "feature_correlations", "strong_pairs")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def correlation_pair(x, y):
    pair = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    valid = len(pair) >= 3 and pair.x.nunique() > 1 and pair.y.nunique() > 1
    return {"n": int(len(pair)), "x_unique": int(pair.x.nunique()),
            "y_unique": int(pair.y.nunique()),
            "pearson_r": float(pair.x.corr(pair.y)) if valid else None}


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def build_supplement(root=ROOT):
    root = Path(root)
    sources = ["data/processed/cells_analysis.csv", "results/degradation_slopes.csv",
               "results/split_assignment.csv"]
    before = {name: sha256(root / name) for name in sources}
    cells = pd.read_csv(root / sources[0])
    slopes = pd.read_csv(root / sources[1])
    assert cells.cell_id.is_unique and slopes.cell_id.is_unique
    flags = cells.analysis_eligible.astype(str).str.lower()
    assert flags.isin(["true", "false"]).all()
    eligible = cells.loc[flags.eq("true")].copy()
    assert set(slopes.cell_id) == set(eligible.cell_id)
    data = eligible.merge(slopes[["batch", "cell_id", "late_slope"]],
                          on=["batch", "cell_id"], how="left", validate="one_to_one")
    data["transition_soc_pct"] = np.nan
    parsed_rows = []
    for row in data.itertuples():
        match = POLICY_PATTERN.fullmatch(str(row.policy))
        if match is None:
            parsed_rows.append({"cell_id": row.cell_id, "policy": row.policy, "parsed": False})
            continue
        first, second, soc = [float(match.group(k)) for k in ("first", "second", "soc")]
        assert np.isclose(first, row.c_rate_stage1) and np.isclose(second, row.c_rate_stage2), row.cell_id
        assert 0 <= soc <= 100, row.cell_id
        data.loc[data.cell_id.eq(row.cell_id), "transition_soc_pct"] = soc
        parsed_rows.append({"cell_id": row.cell_id, "policy": row.policy,
                            "parsed": True, "suffix": match.group("suffix")})

    policy_correlations, feature_correlations, strong_pairs = [], [], []
    for batch, group in data.groupby("batch", sort=True):
        for predictor in PREDICTORS:
            policy_correlations.append({"batch": batch, "predictor": predictor,
                "response": "late_slope", **correlation_pair(group[predictor], group.late_slope)})
        for i, j in product(range(len(FEATURES)), repeat=2):
            a, b = FEATURES[i], FEATURES[j]
            record = {"batch": batch, "feature_1": a, "feature_2": b,
                      **correlation_pair(group[a], group[b])}
            feature_correlations.append(record)
            if i < j and record["pearson_r"] is not None and abs(record["pearson_r"]) >= .85:
                strong_pairs.append(record)

    policy_groups = data.groupby(["batch", "policy"], sort=True, dropna=False).agg(
        n=("cell_id", "size"), mean_life=("cycle_life", "mean"),
        median_life=("cycle_life", "median"), std_life=("cycle_life", "std"),
        late_slope_n=("late_slope", "count"), late_slope_mean=("late_slope", "mean"),
        late_slope_median=("late_slope", "median"),
        c_rate_stage1=("c_rate_stage1", "first"), c_rate_stage2=("c_rate_stage2", "first"),
        transition_soc_pct=("transition_soc_pct", "first")).reset_index()
    soc_case = policy_groups.loc[policy_groups.batch.eq("Batch 1")
        & policy_groups.c_rate_stage1.eq(8) & policy_groups.c_rate_stage2.eq(3.6)]
    soc_case = soc_case.sort_values("transition_soc_pct")
    after = {name: sha256(root / name) for name in sources}
    assert before == after, "An input changed during the descriptive calculation."
    result = {
        "schema_version": 1,
        "scope": "DAY1 descriptive EDA for eligible Batch1/2/3 cells only; no model fitting or reselection.",
        "source_sha256": before,
        "definitions": {
            "eligible": "Existing analysis_eligible decision; exclusions and target labels are unchanged.",
            "late_slope": "Existing un-smoothed QDischarge linear slope over cycles max(101, 0.8*provided cycle_life) through provided cycle_life; >10 valid observations; Ah/cycle.",
            "late_slope_sign": "More negative means a steeper capacity decline, not a positive degradation-rate magnitude.",
            "transition_soc_pct": "The percentage in the recorded policy string: first_C(SOC%)-second_C; no reinterpretation of newstructure suffix.",
            "correlation": "Pearson r using pairwise complete finite values; no imputation; n reported per pair; undefined for n<3 or a constant variable.",
            "strong_pair_threshold": 0.85,
            "strong_pair_threshold_note": "Descriptive screening of |r| >= 0.85, not a significance test or automatic variable-removal rule.",
            "protocol_group": "One batch and exact policy string; group means give equal weight to each cell record.",
            "features": FEATURES,
            "policy_predictors": PREDICTORS,
        },
        "counts": {"eligible_cells": len(data), "by_batch": data.groupby("batch").size().to_dict(),
            "batch_policy_groups": len(policy_groups), "distinct_policy_strings": int(data.policy.nunique()),
            "singleton_groups": int(policy_groups.n.eq(1).sum()),
            "parsed_policy_cells": sum(r["parsed"] for r in parsed_rows),
            "unparsed_policy_cells": sum(not r["parsed"] for r in parsed_rows)},
        "policy_correlations": policy_correlations,
        "policy_groups": policy_groups.to_dict("records"),
        "feature_correlations": feature_correlations,
        "strong_pairs": strong_pairs,
        "same_current_soc_example": soc_case.to_dict("records"),
        "policy_parse_audit": parsed_rows,
        "limitations": [
            "첫 단계·후속 단계 전류와 전환 SOC는 개별 상관과 전체 정책별 요약으로 비교했습니다. 조건들이 함께 변하므로 각 조건의 독립 효과나 인과효과를 분리한 분석은 아닙니다.",
            "전체 정책 표는 배치×정책 40개 그룹이며 서로 다른 정책 문자열은 37개입니다. 4개 그룹은 셀이 1개뿐이므로 평균 차이를 일반화하기 어렵습니다.",
            "newstructure의 실험적 의미는 원자료만으로 확인되지 않아 해당 문자열을 보존하고 별도 정책 그룹으로 비교했습니다.",
            "후반 기울기는 수명 후반을 보는 EDA용 값입니다. 초기 100사이클 예측 입력으로 사용하지 않습니다.",
            "B1·B3는 제공된 근사 종료 라벨이고 B2는 관측 임계값 교차 라벨입니다. 공통된 종료 상태와 실험 조건을 보장하지 않습니다.",
            "B2·B3의 보완 상관 결과는 배치 차이 설명용입니다. 이 결과로 기존 후보, 분할, 전처리 또는 선택한 모델을 바꾸지 않습니다.",
        ],
        "outputs": {key: f"results/day1_supplement_{key}.csv" for key in TABLE_KEYS},
        "checks": {"source_files_unchanged": before == after,
                   "one_row_per_eligible_cell": len(data) == data.cell_id.nunique(),
                   "all_eligible_cells_have_slopes": bool(data.late_slope.notna().all()),
                   "full_feature_matrix_entries": len(feature_correlations),
                   "symmetric_matrix_expected": True},
    }
    return clean_json(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = build_supplement(args.root)
    out = args.root / "results"
    out.mkdir(parents=True, exist_ok=True)
    for key in TABLE_KEYS:
        pd.DataFrame(result[key]).to_csv(out / f"day1_supplement_{key}.csv", index=False)
    (out / "day1_supplement.json").write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                                      allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"counts": result["counts"], "policy_correlations": result["policy_correlations"],
                      "strong_pairs": result["strong_pairs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
