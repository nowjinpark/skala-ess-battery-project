"""DAY 1 descriptive analysis and prospective model design; no predictors fitted."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data/processed"
RESULTS = ROOT / "results"
FIG = RESULTS / "figures"
BATCHES = (1, 2, 3)
COLORS = {"Batch 1": "#1e6488", "Batch 2": "#db8350", "Batch 3": "#58845b"}
FEATURES = {
    "qd_cycle2": ("2사이클 방전 용량", "초기 용량 수준이 수명과 관련될 수 있다"),
    "qd_change_100_10": ("QDischarge(100) - QDischarge(10)", "초기 용량 변화가 초기 열화를 나타낼 수 있다"),
    "qd_slope_10_100": ("10~100사이클 용량의 선형 기울기", "감소 속도가 수명과 관련될 수 있다"),
    "ir_mean_2_100": ("2~100사이클 양의 내부저항 평균", "저항 수준이 배터리 상태를 반영할 수 있다"),
    "ir_change_early": ("91~100사이클 평균 IR - 2~10사이클 평균 IR", "저항 변화가 초기 열화 신호일 수 있다"),
    "tavg_mean_2_100": ("2~100사이클 평균 온도", "열적 조건이 수명과 연관될 수 있다"),
    "chargetime_mean_2_100": ("2~100사이클 유효 충전시간 평균", "충전 조건의 차이를 요약한다"),
    "log10_deltaq_var": ("log10(Var[Q100(V)-Q10(V)]), 표본분산", "전압별 용량 변화의 퍼짐이 초기 열화 신호일 수 있다"),
}
SHORT_NAMES = dict(zip(FEATURES, ["QD at cycle 2", "QD change", "QD slope", "IR mean", "IR change", "Mean temperature", "Charge time", "log Var(Delta Q)"]))


def savefig(name):
    plt.savefig(FIG / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


def native(obj):
    if isinstance(obj, dict):
        return {str(k): native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [native(x) for x in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def main():
    RESULTS.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook", font_scale=.95)
    cells = pd.concat([pd.read_csv(PROC / f"batch{i}_cells.csv") for i in BATCHES], ignore_index=True)
    summary = pd.concat([pd.read_csv(PROC / f"batch{i}_summary.csv.gz") for i in BATCHES], ignore_index=True)
    quality = pd.concat([pd.read_csv(PROC / f"batch{i}_quality.csv") for i in BATCHES], ignore_index=True)
    curves = {}
    for i in BATCHES:
        with np.load(PROC / f"batch{i}_curves.npz", allow_pickle=False) as z:
            curves.update({k: z[k] for k in z.files})
    # All Batch 1 labels equal n_summary+1. Do not falsely describe them as an
    # observed 0.88-Ah crossing. Keep near-threshold endpoints as supplied-label
    # proxies and flag their uncertainty; exclude clearly unfinished records.
    cells["cycle_life"] = cells.cycle_life_raw
    exact_eol = np.isfinite(cells.cycle_life_raw) & cells.cycle_life_raw.sub(cells.first_below_80).abs().le(1)
    b1_incomplete_ids = {"b1c0","b1c1","b1c2","b1c3","b1c4","b1c8","b1c10","b1c12","b1c13","b1c22"}
    # B1 file identity and raw endpoint pattern match the author's B1. Five
    # incomplete cells were excluded in the source; five others need a later
    # 2017-06-30 continuation that is not part of this task's three files.
    b1_incomplete = cells.cell_id.isin(b1_incomplete_ids)
    assert cells.loc[b1_incomplete,"last_qd"].gt(.885).all()
    b3_meta = json.loads((PROC / "batch3_metadata.json").read_text())
    assert b3_meta["batch_date"] == "2018-04-12" and b3_meta["n_cells"] == 46
    # Original 0-based IDs from the author Load_Data notebook. The MATLAB
    # implementation deletes from shrinking arrays; do not reuse those indices.
    b3_source_excluded = cells.cell_id.isin({"b3c37", "b3c2", "b3c23", "b3c32", "b3c42", "b3c43"})
    endpoint_proxy = (cells.batch.isin(["Batch 1", "Batch 3"]) & ~b1_incomplete & ~b3_source_excluded & cells.cycle_life_raw.eq(cells.n_summary+1)
                      & cells.last_qd.between(.88, .885, inclusive="both"))
    cells["label_status"] = np.select([exact_eol, endpoint_proxy], ["observed_80pct_crossing", "provided_near80_endpoint_proxy"], default="unresolved_or_missing")
    reason_rules = [
        (b1_incomplete, "원저자 B1 비완주 5개 또는 별도 후속기록 필요 5개"),
        (b3_source_excluded, "원저자 B3 수집 오류·비완주·잡음 6개 제외 목록"),
        (~exact_eol & ~endpoint_proxy, "확정 교차 또는 80% 근처 종료 라벨로 확인되지 않음"),
        (~np.isfinite(cells.cycle_life_raw), "수명 라벨 결측"),
        (cells.cycle_life_raw.le(100), "100사이클 예측 시점 부적합"),
        (cells.curve_issue.fillna("").ne(""), "10/100사이클 곡선 대응 확인 필요"),
        (cells.n_early_valid_qd.lt(80), "초기 유효 용량 기록 부족"),
    ]
    cells["exclusion_reason"] = ["; ".join(reason for mask,reason in reason_rules if mask.iloc[i]) for i in range(len(cells))]
    cells["analysis_eligible"] = cells.exclusion_reason.eq("")
    cells["split_role"] = "excluded"
    eligible = cells.loc[cells.analysis_eligible].copy()
    b1ids = eligible.loc[eligible.batch.eq("Batch 1"), "cell_id"]
    # Reserve a fixed internal hold-out before feature-target correlation decisions.
    dev_ids, valid_ids = train_test_split(b1ids, test_size=.2, random_state=42)
    cells.loc[cells.cell_id.isin(dev_ids), "split_role"] = "Batch1_CV_development"
    cells.loc[cells.cell_id.isin(valid_ids), "split_role"] = "Batch1_holdout"
    cells.loc[cells.analysis_eligible & cells.batch.eq("Batch 2"), "split_role"] = "Batch2_final_test"
    cells.loc[cells.analysis_eligible & cells.batch.eq("Batch 3"), "split_role"] = "Batch3_EDA_only"
    cells.to_csv(PROC / "cells_analysis.csv", index=False)
    cells[["cell_id", "batch", "analysis_eligible", "exclusion_reason", "split_role"]].to_csv(RESULTS / "split_assignment.csv", index=False)
    eligible = cells.loc[cells.analysis_eligible].copy()
    dev = cells.loc[cells.split_role.eq("Batch1_CV_development")].copy()
    kept_summary = summary.merge(eligible[["cell_id", "cycle_life"]], on="cell_id", how="inner")
    kept_summary = kept_summary.loc[kept_summary.capacity_plausible & kept_summary.cycle.ge(2)]

    # 1. Lifetimes, with raw labels explicitly separated from observed EOL labels.
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.1))
    bins = np.arange(100, max(2400, int(cells.cycle_life_raw.max()) + 150), 150)
    for batch, color in COLORS.items():
        raw = cells.loc[cells.batch.eq(batch), "cycle_life_raw"]
        clean = eligible.loc[eligible.batch.eq(batch), "cycle_life"]
        axs[0].hist(raw, bins=bins, alpha=.4, label=f"{batch} raw (n={len(raw)})", color=color)
        axs[1].hist(clean, bins=bins, alpha=.55, label=f"{batch} eligible (n={len(clean)})", color=color)
    for ax, title in zip(axs, ["Raw supplied labels (missing values omitted)", "Eligible supplied labels (B1/B3: endpoint proxies)"]):
        ax.set(title=title, xlabel="Cycle life (cycles)", ylabel="Cell count")
        ax.axvline(500, ls="--", color="gray", lw=1)
        ax.axvline(1000, ls="--", color="gray", lw=1)
        ax.legend(fontsize=8)
    fig.tight_layout()
    savefig("life_distribution")
    life_rows = []
    for batch, g in eligible.groupby("batch"):
        life_rows.append({"batch": batch, "n": len(g), "min": g.cycle_life.min(), "median": g.cycle_life.median(),
                          "mean": g.cycle_life.mean(), "std": g.cycle_life.std(), "max": g.cycle_life.max(),
                          "short_n": int(g.cycle_life.lt(500).sum()), "short_pct": 100*g.cycle_life.lt(500).mean(),
                          "long_n": int(g.cycle_life.gt(1000).sum()), "long_pct": 100*g.cycle_life.gt(1000).mean()})
    pd.DataFrame(life_rows).to_csv(RESULTS / "life_summary.csv", index=False)

    # 2. Full-life EDA is explicitly NOT an input feature source.
    norm = Normalize(eligible.cycle_life.min(), eligible.cycle_life.max())
    cmap = plt.get_cmap("viridis")
    fig, grid = plt.subplots(2, 2, figsize=(11, 7.8), sharey=True)
    axs = grid.ravel()
    degradation_rows = []
    for ax, (batch, color) in zip(axs, COLORS.items()):
        g = kept_summary.loc[kept_summary.batch.eq(batch)]
        for cid, c in g.groupby("cell_id"):
            life = float(c.cycle_life.iloc[0])
            c = c.sort_values("cycle")
            ax.plot(c.cycle, c.QDischarge.rolling(7, center=True, min_periods=1).median(), color=cmap(norm(life)), alpha=.6, lw=.8)
            early = c.loc[c.cycle.between(10, 100)]
            late = c.loc[c.cycle.between(max(101, life*.8), life)]
            se = np.polyfit(early.cycle, early.QDischarge, 1)[0] if len(early)>10 else np.nan
            sl = np.polyfit(late.cycle, late.QDischarge, 1)[0] if len(late)>10 else np.nan
            degradation_rows.append({"batch": batch, "cell_id": cid, "early_slope": se, "late_slope": sl,
                                     "late_decline_faster": bool(np.isfinite(se) and np.isfinite(sl) and sl<0 and sl<se)})
        ax.axhline(.88, color="crimson", ls="--", lw=1, label="80% of 1.1 Ah")
        ax.axvspan(0, 100, color="#d7e7f0", alpha=.6)
        ax.set(title=batch, xlabel="Cycle number", ylim=(.78, 1.16))
    axs[0].set_ylabel("Discharge capacity (Ah); rolling median 7")
    axs[2].set_ylabel("Discharge capacity (Ah); rolling median 7")
    axs[3].axis("off")
    axs[3].text(.06, .90, "Reading the curves", fontsize=14, weight="bold", transform=axs[3].transAxes)
    axs[3].text(.06, .72, "Each line = one eligible battery\nColor = supplied cycle life\nRed dashed = 0.88 Ah threshold\nBlue shading = first 100 cycles\n\nB1/B3 use supplied endpoint proxies.\nFull-life curves are for EDA only.", fontsize=11, linespacing=1.6, va="top", transform=axs[3].transAxes)
    fig.tight_layout(rect=[0, 0, .90, 1])
    cax = fig.add_axes([.93, .18, .016, .64])
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, label="Cycle life")
    savefig("capacity_degradation")
    pd.DataFrame(degradation_rows).to_csv(RESULTS / "degradation_slopes.csv", index=False)

    # 3. Difference curves share the same voltage grid within each cell.
    fig, grid = plt.subplots(2, 2, figsize=(11, 7.8))
    axs = grid.ravel()
    for ax, batch in zip(axs[:3], COLORS):
        bg = eligible.loc[eligible.batch.eq(batch)]
        for r in bg.itertuples():
            if r.cell_id + "_deltaq" in curves:
                color = "#6e3a94" if r.cycle_life<500 else "#d89513" if r.cycle_life>1000 else "#9aabb6"
                ax.plot(curves[r.cell_id+"_voltage"], curves[r.cell_id+"_deltaq"], color=color, lw=.9, alpha=.75)
        for label, color, count in [("Short <500", "#6e3a94", int(bg.cycle_life.lt(500).sum())),
                                    ("500-1000", "#9aabb6", int(bg.cycle_life.between(500,1000).sum())),
                                    ("Long >1000", "#d89513", int(bg.cycle_life.gt(1000).sum()))]:
            ax.plot([], [], color=color, label=f"{label} (n={count})")
        ax.set(title=batch, xlabel="Voltage (V)", ylabel="Q100(V) - Q10(V) (Ah)")
        ax.legend(fontsize=8, loc="lower left")
    for batch, color in COLORS.items():
        g = eligible.loc[eligible.batch.eq(batch)]
        axs[3].scatter(g.log10_deltaq_var, g.cycle_life, label=batch, s=22, alpha=.8, color=color)
    axs[3].set(xlabel="log10 Var[Delta Q(V)]", ylabel="Cycle life", title="Descriptive batch comparison")
    axs[3].legend(fontsize=8)
    fig.tight_layout()
    savefig("deltaq")

    deltaq_groups = eligible.assign(life_group=np.select(
        [eligible.cycle_life.lt(500), eligible.cycle_life.gt(1000)],
        ["short_lt500", "long_gt1000"], default="middle_500_1000"))
    deltaq_group_summary = deltaq_groups.groupby(["batch", "life_group"]).agg(
        n=("cell_id", "size"), median_log10_var=("log10_deltaq_var", "median"),
        median_deltaq_min=("deltaq_min", "median")).reset_index()
    deltaq_group_summary.to_csv(RESULTS / "deltaq_group_summary.csv", index=False)
    deltaq_group_observations = []
    for batch in COLORS:
        rows = deltaq_group_summary.loc[deltaq_group_summary.batch.eq(batch)].set_index("life_group")
        if {"short_lt500", "long_gt1000"}.issubset(rows.index):
            sh, lo = rows.loc["short_lt500"], rows.loc["long_gt1000"]
            deltaq_group_observations.append(f"{batch}: 단수명 {int(sh.n)}개의 log 분산 중앙값 {sh.median_log10_var:.2f}, 장수명 {int(lo.n)}개는 {lo.median_log10_var:.2f}. 표본 수를 고려한 기술 비교다.")
        else:
            short_n = int(rows.loc["short_lt500", "n"]) if "short_lt500" in rows.index else 0
            long_n = int(rows.loc["long_gt1000", "n"]) if "long_gt1000" in rows.index else 0
            deltaq_group_observations.append(f"{batch}: 단수명 {short_n}개, 장수명 {long_n}개로 한쪽 그룹이 없어 배치 내부 양극단 비교에는 한계가 있다.")

    # 4. Groups keep their sample counts visible. No causal conclusion is drawn.
    policy = eligible.groupby(["batch", "policy"], dropna=False).agg(n=("cycle_life", "size"),
            mean_life=("cycle_life", "mean"), std_life=("cycle_life", "std"), c_rate_stage1=("c_rate_stage1", "first")).reset_index()
    policy.to_csv(RESULTS / "policy_summary.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for batch, color in COLORS.items():
        g = eligible.loc[eligible.batch.eq(batch)]
        axs[0].scatter(g.c_rate_stage1, g.cycle_life, color=color, s=25, alpha=.75, label=batch)
    axs[0].set(xlabel="First charging stage (C-rate)", ylabel="Cycle life", title="Each dot = one eligible cell")
    axs[0].legend(fontsize=8)
    top = policy.sort_values(["n", "mean_life"], ascending=False).head(12).sort_values("mean_life")
    labels = [f"{r.batch[-1]}: {r.policy} (n={r.n})" for r in top.itertuples()]
    axs[1].barh(labels, top.mean_life, color=[COLORS[x] for x in top.batch])
    axs[1].tick_params(axis="y", labelsize=10)
    axs[1].set(xlabel="Mean cycle life", title="12 most represented batch/protocol groups")
    fig.tight_layout()
    savefig("charge_policy")

    # 5. Model-directed correlations use ONLY Batch 1's development cells.
    candidates = list(FEATURES)
    pearson = dev[candidates+["cycle_life"]].corr(method="pearson")["cycle_life"].drop("cycle_life")
    spearman = dev[candidates+["cycle_life"]].corr(method="spearman")["cycle_life"].drop("cycle_life")
    corr_table = pd.DataFrame({"feature": candidates, "pearson": pearson.reindex(candidates).to_numpy(),
                               "spearman": spearman.reindex(candidates).to_numpy(),
                               "n": [dev[[c, "cycle_life"]].dropna().shape[0] for c in candidates]})
    corr_table["abs_pearson"] = corr_table.pearson.abs()
    corr_table = corr_table.sort_values("abs_pearson", ascending=False)
    corr_table.to_csv(RESULTS / "feature_correlations.csv", index=False)
    matrix = dev[candidates].corr()
    pairs = [{"feature_a": a, "feature_b": b, "correlation": matrix.loc[a, b]}
             for i,a in enumerate(candidates) for b in candidates[i+1:]
             if np.isfinite(matrix.loc[a,b]) and abs(matrix.loc[a,b]) >= .85]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.3), gridspec_kw={"width_ratios":[1, 1.25]})
    ct = corr_table.sort_values("pearson")
    axs[0].barh(ct.feature.map(SHORT_NAMES), ct.pearson, color="#1e6488")
    axs[0].set(xlabel="Pearson r with cycle life", xlim=(-1,1), title=f"Batch 1 development only (n={len(dev)})")
    sns.heatmap(matrix.rename(index=SHORT_NAMES, columns=SHORT_NAMES), ax=axs[1], cmap="vlag", vmin=-1, vmax=1, center=0, square=True, annot=True, fmt=".2f", annot_kws={"size":9}, cbar=False)
    axs[1].set_title("Feature correlations (same development cells)")
    for ax in axs:
        ax.tick_params(labelsize=10)
    fig.tight_layout()
    savefig("correlation")

    topcorr = corr_table.iloc[0]
    dq = corr_table.set_index("feature").loc["log10_deltaq_var"]
    counts = []
    for b,g in cells.groupby("batch"):
        counts.append({"name":b, "raw_cells":len(g), "clean_cells":int(g.analysis_eligible.sum()),
                       "excluded_cells":int((~g.analysis_eligible).sum()), "continued_cells":0})
    missing_barcodes = int(cells.barcode.isna().sum() + cells.barcode.fillna("").eq("").sum() - cells.barcode.isna().sum())
    valid_barcodes = cells.loc[cells.barcode.fillna("").ne("")]
    shared_barcodes = sorted(valid_barcodes.groupby("barcode").batch.nunique().loc[lambda x: x > 1].index.tolist())
    if shared_barcodes:
        raise ValueError(f"Cross-batch barcode overlap requires review before final analysis: {shared_barcodes}")
    n_accel = sum(x["late_decline_faster"] for x in degradation_rows)
    excluded = cells.loc[~cells.analysis_eligible, ["cell_id","cycle_life_raw","n_summary","last_qd","exclusion_reason"]]
    excluded.to_csv(RESULTS / "excluded_cells.csv", index=False)
    observations_life = [f"{r['batch']}: 분석 대상 {r['n']}개, 수명 중앙값 {r['median']:.1f}, 범위 {r['min']:.0f}~{r['max']:.0f}사이클. 단수명(<500) {r['short_n']}개, 장수명(>1000) {r['long_n']}개." for r in life_rows]
    shortest = eligible.nsmallest(3,"cycle_life")
    observations_life.append("가장 짧은 제공 수명 사례: " + "; ".join(f"{r.cell_id} {r.cycle_life:.0f}사이클 ({r.policy})" for r in shortest.itertuples()) + ". 충전 첫 단계가 모두 높은 것은 아니므로 후속 충전 단계와 조건을 함께 살펴야 한다.")
    policy_correlations=[]
    for b,g in eligible.merge(pd.DataFrame(degradation_rows),on=["batch","cell_id"]).groupby("batch"):
        policy_correlations.append({"batch":b,"n":len(g),"c_rate_vs_life_r":g.c_rate_stage1.corr(g.cycle_life),
                                    "c_rate_vs_late_slope_r":g.c_rate_stage1.corr(g.late_slope)})
    b2_missing_ir=int(eligible.loc[eligible.batch.eq("Batch 2"),"ir_mean_2_100"].isna().sum())
    b3 = cells.loc[cells.batch.eq("Batch 3")]
    b3_eligible = b3.loc[b3.analysis_eligible]
    b3_observed = int(b3_eligible.label_status.eq("observed_80pct_crossing").sum())
    b3_proxy = int(b3_eligible.label_status.eq("provided_near80_endpoint_proxy").sum())
    output = {
      "meta": {"title":"초기 100사이클 기반 배터리 총수명 예측", "author":"박진원", "class":"울산 1반",
        "batches":counts, "observation_cycles":100, "nominal_capacity_Ah":1.1, "eol_capacity_Ah":.88,
        "data_source":"https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle",
        "cleaning_notes":[
          "원본 MAT와 cycle_life_raw를 보존하고, 분석 대상 여부 및 제외 사유를 별도 열로 기록했다.",
          "첫 사이클의 빈/0 기록에 영향을 받지 않도록 세 배치 모두 2~100사이클로 초기 요약을 통일했다. 용량은 0.2<QD<1.5 Ah, 온도 0~80 C, 충전시간 0~120 min 범위만 해당 특성의 요약에 사용했다. 잠정 품질 규칙으로 DAY2에서 민감도를 점검한다.",
          "Batch 1의 모든 라벨은 관측 길이+1이다. 원저자 B1 비완주 5개와 별도 후속기록이 필요한 5개를 제외했다. 나머지 36개는 제공된 근사 종료 라벨로 유지했다. 실제 80% 교차를 직접 관측한 라벨이라고 주장하지 않는다.",
          "B1 유지 36개의 종료 용량은 0.88006~0.88336 Ah, 제외 10개는 0.91312~1.04328 Ah다. 0.885 Ah는 이 차이를 확인하는 점검값이며 새로운 EOL 정의가 아니다. EOL 정의는 여전히 정격 1.1 Ah의 80%=0.88 Ah다.",
          "Batch 2는 제공 라벨과 실제 QD<0.88 Ah 최초 사이클이 일치하는 기록을 유지하고 수명 라벨이 없는 기록은 제외했다. 원본 라벨을 추정값으로 덮어쓰지 않았다.",
          f"Batch 3의 날짜(2018-04-12)와 원본 46개 셀 구조를 원저자 파일과 대조했다. 원저자 제외 ID b3c37/2/23/32/42/43을 분리했다. 남은 분석 대상 {len(b3_eligible)}개: 직접 교차 라벨 {b3_observed}개, 근사 종료 라벨 {b3_proxy}개. 라벨 값은 원본을 유지했다.",
          "과제 Batch 2(2018-02-20)와 원저자 loader의 Batch 2(2017-06-30)는 다르다. 원저자의 B2 인덱스 기반 병합·라벨 가산·삭제 규칙을 적용하지 않았다.",
          f"해석 가능한 barcode가 없는 기록 {missing_barcodes}개. 빈 식별값은 동일 셀로 판단하지 않았으며 물리적 셀 중복 여부는 확인 가능한 식별 정보의 범위로 한정한다.",
          f"Batch 2 분석 대상 중 {b2_missing_ir}개는 초기 내부저항 요약을 계산할 양의 값이 없다. 0을 정상 저항으로 평균내지 않고 결측으로 유지했다. DAY2에서 학습 fold의 중앙값으로 대체하고 IR 제외 모델과 비교한다.",
          "Delta Q는 summary.cycle에서 실제 10/100 값을 찾아 대응한 Qdlin과 셀별 Vdlin으로 계산했다. 슬롯 번호는 셀별 audit에 저장했다."],
        "cycle_records":len(summary), "label_status_counts":cells.groupby(["batch", "analysis_eligible", "label_status"]).size().reset_index(name="n").to_dict("records"),
        "quality_totals":quality.select_dtypes(include="number").sum().to_dict()},
      "life":{"summary_rows":life_rows,"observations":observations_life,
              "class_threshold":"EDA 구분: 단수명 <500, 장수명 >1000. 분류 과제의 550 기준과 별개."},
      "degradation":{"observations":[f"분석 대상 {len(degradation_rows)}개 중 {n_accel}개에서 마지막 20% 수명 구간의 용량 기울기가 초기 10~100사이클보다 더 음수였다.",
                         "초기 용량이 비슷해도 이후 열화 경로가 다를 수 있다. 초기 수준뿐 아니라 변화량을 후보로 사용한다."],
                       "knee_note":"평활화한 곡선으로 열화 가속 구간을 정성적으로 탐색했다. 정확한 knee 시점을 추정·검증한 것은 아니며, 후반부 정보와 knee를 예측 입력에 사용하지 않는다."},
      "deltaq":{"observations":[f"Batch 1 개발 표본에서 log10 Var(Delta Q)와 수명의 Pearson r={dq.pearson:.3f}, Spearman rho={dq.spearman:.3f} (n={int(dq.n)}).",
                  "변수 추가 효과는 아직 확인 전이다. DAY2에 기본 특성 집합과 Delta Q 추가 집합을 같은 검증 분할에서 비교한다."],"group_observations":deltaq_group_observations,"group_summary":deltaq_group_summary.to_dict("records"),"correlation_with_life":dq.to_dict()},
      "policy":{"observations":[f"배치별 충전 프로토콜 그룹 {len(policy)}개 중 표본 1개인 그룹이 {int(policy.n.eq(1).sum())}개다. 평균 차이를 일반적인 효과로 단정하지 않는다.",
                   "충전 첫 단계 C-rate와 프로토콜 전체를 구분했다. 온도·후속 단계·배치가 함께 달라질 수 있어 고속 충전의 인과 효과로 해석하지 않는다."] + [f"{r['batch']}: 첫 단계 C-rate와 수명 r={r['c_rate_vs_life_r']:.3f}, 수명 후반 용량 기울기와 r={r['c_rate_vs_late_slope_r']:.3f} (n={r['n']})." for r in policy_correlations],
                   "group_counts":policy.to_dict("records"),"correlations":policy_correlations},
      "correlation":{"observations":[f"Batch 1 개발 표본 {len(dev)}개에서 절대 Pearson 상관이 가장 큰 후보는 {topcorr.feature} (r={topcorr.pearson:.3f})였다.",
                         f"후보 특성 간 |r|>=0.85인 조합 {len(pairs)}개. 중복 특성의 동시 투입을 줄이고 Ridge 규제를 우선 비교한다."],
                     "top_target_correlations":corr_table.to_dict("records"),"high_feature_pairs":pairs,"vif":[]},
      "features":[{"name":k,"definition":v[0],"hypothesis":v[1],"availability":"100사이클 이내"} for k,v in FEATURES.items()],
      "model_plan":{"task":"regression", "target":"provided cycle_life (80% nominal target; B1/B3 endpoint proxies, B2 observed crossing)",
           "eda_only_n":int(eligible.batch.eq("Batch 3").sum()),"batch3_role":"EDA only; optional additional evaluation in DAY2",
           "development_n":len(dev),"holdout_n":len(valid_ids),"test_n":int(eligible.batch.eq("Batch 2").sum()),
           "random_seed":42,"cv_folds":3,"primary_metric":"MAPE (%)","secondary_metrics":["MAE (cycles)","RMSE (cycles)","R2"],
           "models":["DummyRegressor(mean)","LinearRegression", "Ridge", "RandomForestRegressor"],
           "status":"DAY 1: 모델 학습·성능 개선 수치는 아직 없음"},
      "limitations":[
         "녹음본의 DAY1 안내에 따라 Batch 1·2·3을 EDA에 포함하고 Extra는 제외했다. Batch 3는 이번 분석에서 EDA 전용이며 DAY2의 추가 성능 평가는 선택이다.",
         "소수의 배터리로 분석했으므로 상관계수·그룹 평균의 불확실성이 크다. 사이클 행 수가 독립 표본 수는 아니다.",
         "Batch 1과 Batch 3는 80% 근처에서 기록이 끝나는 제공 근사 라벨, Batch 2는 실제 임계값 교차 라벨이다. 배치별 정답 정의 차이가 있어 오차를 모델 문제만으로 설명할 수 없다. DAY2에서 근접 판정 기준 0.8825/0.885 Ah의 민감도를 확인한다.",
         "불완전 수명 기록 제외는 장수명 표본을 줄일 수 있다. 제외 전후 분포를 함께 제시하고 실제 전체 배터리 집단으로의 일반화를 제한한다.",
         "과제 요구에 따라 세 배치를 EDA했다. Batch 2는 완전히 보지 않은 외부 검증셋이라고 부르지 않으며, 모델 선택·특성 조정은 Batch 1 개발 데이터에서만 한다.",
         "Batch 1 holdout의 전체 분포도 DAY1 EDA에 포함된다. 모델별 점수·특성 선택에는 사용하지 않고 DAY2 최종 내부 검증에만 사용한다.",
         "실험실 데이터 분석이며 실제 ESS 교체 비용 절감 또는 운영 안전성 개선을 실증한 결과는 아니다.",
         "논문의 9.1%는 데이터 분할·배치 버전·특성·정제 조건이 일치하지 않으므로 참고 수치로만 기록한다."],
      "figures":{k:str((FIG/f"{k}.png").relative_to(ROOT)) for k in ["life_distribution","capacity_degradation","deltaq","charge_policy","correlation"]},
    }
    (RESULTS/"analysis.json").write_text(json.dumps(native(output), ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps(native({"counts":counts,"split":output["model_plan"],"top_correlations":corr_table.head(4).to_dict("records")}),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
