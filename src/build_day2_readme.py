"""Write a concise student report from the frozen DAY2 results."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'results/day2'


def main():
    e = json.loads((D / 'evaluation.json').read_text())
    lock = json.loads((D / 'selection_lock.json').read_text())
    cv = pd.read_csv(D / 'cv_candidates.csv').set_index('candidate_id')
    fm = pd.read_csv(D / 'final_metrics.csv')
    names = {'Dummy': '평균 예측', 'Linear': '선형회귀', 'Ridge': 'Ridge', 'RandomForest': 'Random Forest'}
    feature_names = {'basic7': '기본 7개', 'deltaq_only1': 'ΔQ 로그 분산 1개', 'no_qd_change7': '용량 변화량을 뺀 7개'}
    rows = []
    for family, cid in lock['family_winners'].items():
        a = cv.loc[cid]
        v = fm[fm.candidate_id.eq(cid) & fm.dataset.eq('Valid_B1_holdout')].iloc[0]
        t = fm[fm.candidate_id.eq(cid) & fm.dataset.eq('Test_B2')].iloc[0]
        name = names[family] + (' **(선택)**' if cid == e['winner_id'] else '')
        target = 'ln(수명)' if a.target_transform == 'log' else '원래 수명'
        rows.append(f'| {name} | {feature_names[a.feature_set]} / {target} | {a.cv_mape_pct_mean:.2f} ± {a.cv_mape_pct_sd:.2f} | {v.mape_pct:.2f} | {t.mape_pct:.2f} |')
    family_table = '\n'.join(rows)
    worst_table = '\n'.join(f'| {x["cell_id"]} | {x["actual"]:.0f} | {x["predicted"]:.1f} | {x["ape_pct"]:.1f} | {x["policy"]} |' for x in e['largest_test_errors'])
    ablation = cv[cv.family.eq('Ridge') & cv.target_transform.eq('log') & cv.params.eq('{"alpha": 10.0}')].set_index('feature_set')
    g = e['gaps_percentage_points']
    a = json.loads((ROOT / 'results/analysis.json').read_text())
    life_table = '\n'.join(
        f'| {x["batch"]} | {x["median"]:.1f} | {x["short_n"]}/{x["n"]} ({x["short_pct"]:.1f}%) | {x["long_n"]}/{x["n"]} ({x["long_pct"]:.1f}%) |'
        for x in a['life']['summary_rows']
    )
    cells = pd.read_csv(ROOT / 'data/processed/cells_analysis.csv')
    slopes = pd.read_csv(ROOT / 'results/degradation_slopes.csv').merge(
        cells[['cell_id', 'cycle_life']], on='cell_id', validate='one_to_one'
    )
    b2 = slopes[slopes.batch.eq('Batch 2')]
    short_late = b2.loc[b2.cycle_life.lt(500), 'late_slope'].median() * 1000
    long_late = b2.loc[b2.cycle_life.gt(1000), 'late_slope'].median() * 1000
    dq = {x['life_group']: x for x in a['deltaq']['group_summary'] if x['batch'] == 'Batch 2'}
    policy_table = '\n'.join(
        f'| {x["policy"]} | {x["n"]} | {x["mean_life"]:.1f} |'
        for x in a['policy']['group_counts']
        if x['batch'] == 'Batch 1' and x['policy'] in ('8C(15%)-3.6C', '8C(25%)-3.6C', '8C(35%)-3.6C')
    )
    text = f'''# ESS 배터리 수명 예측

초기 100사이클의 측정값으로 배터리 총수명을 예측하는 **회귀 과제**를 선택했습니다. 수명이 얼마나 되는지와 예측이 얼마나 빗나갔는지를 직접 비교하고, 다른 배치에도 적용할 수 있는지 확인했습니다.

## 프로젝트 개요

- **데이터셋:** [MIT-Stanford Battery Dataset](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle) (Severson et al., 2019)
- **태스크:** Regression — 제공된 총수명 `cycle_life` 예측
- **입력·분석 단위:** 초기 100사이클의 정보, 배터리 셀 1개당 표본 1개
- **수명 기준:** 정격 1.1 Ah의 80%인 0.88 Ah. 예측값의 단위는 남은 일수가 아닌 총사이클 수입니다.

| 데이터 | 원본 → 분석 | 사용한 곳 |
|---|---:|---|
| Batch 1 (2017-05-12) | 46 → 36 | 개발용 28개 / 별도 검증용 8개 |
| Batch 2 (2018-02-20) | 47 → 39 | 모델을 정한 뒤 최종 평가 |
| Batch 3 (2018-04-12) | 46 → 40 | DAY 1 EDA, 선택 사항인 추가 모델 평가는 생략 |

총 139개 중 115개를 EDA에 사용했습니다. 불완전한 기록 등 [제외 이유](results/excluded_cells.csv)를 남겼고, 원본 수명값은 바꾸지 않았습니다. B1·B3는 용량이 80% 부근에서 기록이 끝나 제공된 수명값이고, B2는 실제로 80% 아래로 내려간 시점을 확인한 값입니다. 과제 B2와 원저자 코드의 B2(2017-06-30)는 파일 날짜가 달라 셀 번호를 그대로 이어 붙이지 않았습니다.

## 파일 구조

```text
├── data/processed/          # 정제 데이터
├── notebooks/
│   ├── 01_DAY1_EDA.ipynb
│   └── 02_DAY2_Modeling.ipynb
├── src/                     # 전처리·변수 추출·학습·검증 코드
├── results/day2/            # 성능표·예측값·저장 모델
├── output/pdf/              # DAY 1 설계서
├── requirements.txt
└── README.md
```

[학습 코드](src/train_day2.py), [성능표](results/day2/final_metrics.csv), [셀별 예측값](results/day2/final_predictions.csv)에서 구현과 결과를 확인할 수 있습니다.

## 환경 설정

Python 3.12와 [requirements.txt](requirements.txt)의 버전을 사용했습니다. 아래 명령은 macOS 기준이며, 저장소를 내려받아 환경을 준비하고 저장 결과를 확인합니다.

```sh
git clone https://github.com/nowjinpark/skala-ess-battery-project.git
cd skala-ess-battery-project
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/verify_day2.py
```

[DAY 2 노트북](notebooks/02_DAY2_Modeling.ipynb)을 이 환경의 Python 커널로 실행하면 저장 모델의 예측과 지표를 확인할 수 있습니다. 저장 결과 확인에는 원본 MAT 파일이 필요하지 않습니다. 기존 결과를 보존하면서 별도 폴더에서 다시 학습하는 방법은 노트북 15절, 원본 파일명과 전처리 순서는 [데이터 안내](data/DATA_SOURCE.md)에 정리했습니다.

## EDA

### Cycle Life 분포

EDA에서는 단수명을 500사이클 미만, 장수명을 1000사이클 초과로 구분했습니다.

| 배치 | 수명 중앙값(사이클) | 단수명 <500 | 장수명 >1000 |
|---|---:|---:|---:|
{life_table}

![배치별 배터리 수명 분포](results/figures_report/life_distribution.png)

세로선은 단수명·장수명 구분에 사용한 500·1000사이클 기준입니다.

B1은 600–1000사이클에 주로 분포했고, B2는 400–500사이클에 몰리면서 긴 수명 쪽으로 분포가 이어졌습니다. B3는 800–1100사이클에 주로 분포하며 일부는 1900사이클을 넘었습니다.

**핵심 발견:** B1에 없는 단수명이 B2의 대부분이므로, B1에서 잘 맞는 모델도 B2에서는 오차가 커질 수 있다고 보았습니다.

### 열화 곡선과 knee point

![배치별 방전 용량 변화와 대표 셀의 knee 추정](results/figures_report/capacity_degradation.png)

위는 전체 셀의 열화 곡선, 아래는 대표 셀의 knee 추정 결과입니다. 분홍색은 단수명, 청록색 점선은 장수명, 회색은 중간 수명 셀입니다.

115개 셀 모두 수명 마지막 20% 구간의 용량 기울기가 초기 10–100사이클보다 더 음수였습니다. B2의 후기 기울기 중앙값은 단수명 28개가 **{short_late:.2f} mAh/사이클**, 장수명 3개가 **{long_late:.2f} mAh/사이클**로, 단수명 셀의 후기 용량 감소가 더 빨랐습니다. 초기 구간에는 용량이 증가한 셀도 있었습니다.

대표 셀 b1c11·b2c16·b3c25에서 열화가 빨라지는 전환점(knee)을 각각 약 **590·340·820사이클**로 탐색했습니다. 대표 사례의 추정값이며 전체 셀의 정확한 knee 시점으로 일반화하지 않았습니다.

**핵심 발견:** 단수명 셀의 후기 열화가 더 빨랐지만, 초기 기울기만으로 수명을 판단하기는 어려웠습니다. 후기 기울기와 knee는 미래 정보이므로 모델 입력에서 제외했습니다.

### ΔQ(V) 곡선

![수명 구간별 ΔQ 곡선과 ΔQ 로그 분산·수명의 관계](results/figures_report/deltaq.png)

세 배치의 ΔQ 곡선과 로그 분산·수명의 관계를 비교했습니다. 오른쪽 아래 산점도는 배치 차이를 살펴보기 위한 EDA입니다.

같은 전압에서 `Q100(V)-Q10(V)`를 비교했습니다. 대체로 2.8–3.1 V 부근에서 음수 방향으로 내려갔다가 3.2 V 부근에서 0에 가까워졌으며, B2 단수명 셀의 곡선이 장수명 셀보다 더 깊게 내려갔습니다. B2의 ΔQ 최솟값 중앙값은 단수명 **{dq['short_lt500']['median_deltaq_min']:.3f} Ah**, 장수명 **{dq['long_gt1000']['median_deltaq_min']:.3f} Ah**, 로그 분산 중앙값은 각각 **{dq['short_lt500']['median_log10_var']:.2f}·{dq['long_gt1000']['median_log10_var']:.2f}**로 나타났습니다. B1·B3에는 단수명 셀이 없어 같은 배치 안의 장·단수명 비교가 제한됐습니다.

B1 개발용 28개에서 ΔQ 로그 분산과 수명의 상관계수는 **r={a['deltaq']['correlation_with_life']['pearson']:.3f}**였습니다.

**핵심 발견:** ΔQ 곡선의 변화가 클수록 수명이 짧은 경향을 확인해, 분산을 수명 예측의 후보 변수로 선택했습니다.

### 충전 속도(C-rate)와 수명

![첫 단계 충전 속도와 수명 및 대표 충전 방식별 평균 수명](results/figures_report/charge_policy.png)

위는 첫 단계 C-rate와 수명, 아래는 배치별 표본 수가 많은 충전 방식 3개의 평균 수명입니다. `n`은 셀 수, `[new]`는 원본 충전 방식 이름의 `newstructure` 표기입니다.

B1에서 첫 충전 단계와 이후 전류가 같은 8C→3.6C라도, 전류를 바꾸는 충전 상태(SOC)에 따라 평균 수명이 달랐습니다.

| 충전 방식 | 셀 수 | 평균 수명(사이클) |
|---|---:|---:|
{policy_table}

**핵심 발견:** 첫 단계 C-rate만으로 수명을 설명하기 어려워 전체 충전 방식을 함께 봐야 한다고 판단했습니다. 각 조건이 2개 셀뿐이므로 충전 방식의 인과 효과로 단정하지 않았습니다. 전체 방식별 평균은 [충전 방식 비교표](results/policy_summary.csv)에 정리했습니다.

상세 그래프와 비교는 [DAY 1 설계서](output/pdf/DS-MINI-Design-울산_1반-박진원.pdf)와 [EDA 노트북](notebooks/01_DAY1_EDA.ipynb)에 정리했습니다. 배치별 비교는 설명용 분석이며, 변수와 모델은 B1 개발용 교차검증으로 선택했습니다.

## Modeling

### 피처 엔지니어링 전략

![B1 개발용 데이터의 변수·수명 상관관계와 변수 간 상관관계](results/figures_report/correlation.png)

B1 개발용 28개에서 후보 변수 8개를 비교했습니다. ΔQ 로그 분산(F8)은 수명과 음의 상관을 보였고, 용량 변화량(F2)과 기울기(F3)는 서로 높은 상관을 보여 중복을 줄이는 실험으로 이어졌습니다.

| EDA에서 확인한 내용 | 반영한 방법 |
|---|---|
| ΔQ 곡선의 변화와 수명 사이에 관련성이 있었습니다. | 분산 추가, 평균·최솟값 대체, ΔQ 단독 입력을 비교했습니다. |
| B1 개발용의 용량 변화량과 기울기는 r=0.985였습니다. | 중복 변수를 하나씩 빼 보고 Ridge도 비교했습니다. |
| 전체 충전 방식에 따라 수명이 달랐습니다. | 같은 충전 방식의 셀을 묶어 교차검증했습니다. |
| B2의 내부저항 입력값은 6개 셀에서 없었습니다. | 학습 데이터의 중앙값으로 채우고, 저항 변수를 뺀 경우도 비교했습니다. |

최종 입력은 **초기 용량 QD(2), 10-100사이클 용량 기울기, 2-100사이클 평균 내부저항·온도·충전 시간, 내부저항 변화, ΔQ 로그 분산**의 7개입니다. 내부저항 변화는 양의 내부저항 측정값으로 구한 91-100사이클 평균에서 2-10사이클 평균을 뺐습니다. ΔQ는 같은 전압에서 구한 차이의 표본분산에 log10을 적용했습니다.

### 데이터 분할과 전처리

DAY 1에서 정한 개발용 28개와 별도 검증용 8개를 유지했습니다(seed=42). 개발용의 충전 방식 18개를 묶어 **GroupKFold 3분할 교차검증**을 했고, 각 검증 부분은 10/9/9개였습니다. **각 fold의 학습·검증 사이 같은 충전 방식의 중복은 0개**였습니다. 별도 검증용 8개 중 5개는 개발용과 충전 방식이 같다는 한계가 있습니다.

결측값을 채울 중앙값과 표준화 기준은 **각 교차검증의 학습 부분에서만** 구했습니다. 선형회귀·Ridge에는 표준화를 적용했고, Random Forest에는 적용하지 않았습니다. 최종 모델의 전처리 기준은 개발용 28개에서 구해 별도 검증·B2에 그대로 적용했습니다. 전체 기록 길이, 종료 용량, 셀 ID, 100사이클 이후 측정값은 입력에 넣지 않았습니다.

충전 시간은 큰 값이 평균에 미치는 영향을 줄이기 위해 `0 < chargetime < 120`인 값만 사용했습니다. 분석 대상 셀에서 B1 3개·B2 10개의 **측정값**을 제외했으며 셀은 제외하지 않았습니다. B2의 수명과의 상관계수는 처리 전 +0.087에서 처리 후 -0.917로 달라졌습니다. 잠정 기준이며 큰 값이 발생한 원인은 확인하지 못했습니다.

### 모델 선택 및 근거

평균만 예측하는 모델을 기준으로 선형회귀·Ridge·Random Forest를 비교했습니다. 적은 데이터에서 단순한 모델부터 확인하고, Ridge로 중복 변수의 영향을 줄이며, Random Forest로 직선으로 설명하기 어려운 관계를 살펴보려 했습니다.

입력 조합 8개와 모델 설정을 미리 정해 **113개 후보**를 비교했습니다. 선형회귀·Ridge는 원래 수명과 ln(수명), Ridge의 α는 0.1/1/10/100을 비교했습니다. Random Forest는 나무 300개, 깊이 3/제한 없음, 끝 노드의 최소 표본 수 1/3, 전체 입력 변수, seed=42를 사용했습니다.

| 모델 | 계열별로 선택한 입력 / 학습할 값 | CV MAPE ± SD | 별도 검증 MAPE | B2 MAPE |
|---|---|---:|---:|---:|
{family_table}

MAPE 단위는 %이며, SD는 3개 fold 점수의 표본 표준편차입니다. 신뢰구간을 뜻하지는 않습니다.

- **최종 모델:** Ridge(α=10, C037), 입력 변수 7개, ln(수명) 학습
- **선택 이유:** B1 교차검증 평균 MAPE가 가장 작았습니다.
- **평가 방법:** 예측값을 exp로 원래 사이클 수로 되돌렸습니다. 개발용 28개로 학습한 같은 모델을 별도 검증용과 B2에 적용했습니다.

![같은 Ridge 설정에서 피처 조합별 교차검증 성능 비교](results/day2/figures/feature_ablation.png)

Ridge(α=10, ln(수명))의 입력 조합별 CV MAPE입니다. 파란 막대가 선택한 조합이며, 오차 막대는 3개 fold 점수의 표본 표준편차입니다.

같은 Ridge·α=10·ln(수명)에서 기본 7개의 CV MAPE는 {ablation.loc['basic7'].cv_mape_pct_mean:.2f}%, ΔQ 분산 추가 후 {ablation.loc['deltaq8'].cv_mape_pct_mean:.2f}%, 중복 용량 변화량 제외 후 {ablation.loc['no_qd_change7'].cv_mape_pct_mean:.2f}%였습니다. 다만 선택에 쓴 점수이므로 독립적인 성능 개선의 증거로 보지는 않았습니다. **B2에서는 선형회귀가 더 좋았지만, 테스트 결과를 보고 선택 모델을 바꾸지는 않았습니다.** 전체 설정과 비교는 [실험 계획](results/day2/experiment_plan.json)·[DAY 2 노트북](notebooks/02_DAY2_Modeling.ipynb)에 남겼습니다.

## 성능 결과

MAPE는 셀별 `|실제-예측| / 실제`를 평균 내 백분율로 표시한 값입니다. Train은 학습 데이터를 다시 예측한 점수가 아닌 **B1 교차검증 평균**입니다.

| 구분 | MAPE(%) / Gap(%p) | 계산 기준 |
|---|---:|---|
| Train (Batch1 CV) | {e['cv']['cv_mape_pct_mean']:.2f}% | 3개 검증 fold 평균 |
| Valid (Batch1 Hold-out) | {e['valid']['mape_pct']:.2f}% | B1 별도 검증용 8개 |
| Test (Batch2) | {e['test']['mape_pct']:.2f}% | B2 39개 |
| Gap (Train → Valid) | {g['valid_minus_cv']:+.2f}%p | Valid - CV |
| Gap (Valid → Test) | {g['test_minus_valid']:+.2f}%p | Test - Valid |
| Gap (Target → Test) | {g['test_minus_paper_9_1']:+.2f}%p | Test - 논문 참고값 9.1% |

화살표는 비교 방향이며, Gap은 **이후 MAPE - 기준 MAPE**로 계산했습니다. 예를 들어 Target → Test는 `55.22 - 9.10 = +46.12%p`로, 논문 목표보다 오차가 커졌다는 뜻입니다. CV와 별도 검증의 차이는 작았지만, **B2에서 {g['test_minus_valid']:.2f}%p 나빠져 다른 배치로의 일반화에 실패했습니다.** 작은 표본이므로 내부 점수가 비슷하다는 이유만으로 과적합이 없다고 보지는 않았습니다.

별도 검증의 MAE/RMSE는 {e['valid']['mae_cycles']:.2f}/{e['valid']['rmse_cycles']:.2f}사이클, R²는 {e['valid']['r2']:.3f}이었습니다. B2는 각각 {e['test']['mae_cycles']:.2f}/{e['test']['rmse_cycles']:.2f}사이클, {e['test']['r2']:.3f}이었습니다. B2의 음수 R²는 해당 배치의 실제 평균 수명을 아는 상수 예측보다 제곱오차가 컸다는 뜻입니다.

논문의 참고 MAPE 9.1%보다 {g['test_minus_paper_9_1']:.2f}%p 높았습니다. 다만 데이터 버전·정제·분할·수명값의 기준이 달라 같은 조건의 논문 재현이라고 보지는 않았습니다.

![B1 별도 검증과 B2 평가의 실제 수명·예측 수명 비교](results/day2/figures/actual_predictions.png)

점선은 실제값과 예측값이 같은 위치이고, 회색 영역은 B1 개발용 셀의 수명 범위입니다. B2의 단수명 셀은 이 범위보다 짧았지만 모델은 수명을 더 길게 예측했습니다.

## 오류 분석

상대오차(APE)가 큰 B2 셀 5개입니다. 절대오차 순위와는 다를 수 있습니다.

| 셀 | 실제 수명 | 예측 수명 | APE(%) | 충전 방식 |
|---|---:|---:|---:|---|
{worst_table}

![상대오차 상위 5개 셀과 B1 학습 범위를 벗어난 B2 입력값](results/day2/figures/error_diagnostics.png)

왼쪽은 위 5개 셀의 실제 수명과 예측값, 오른쪽은 변수별로 B1 개발용 범위를 벗어난 B2 셀 수입니다. 오른쪽 집계에서는 해당 변수의 결측값을 제외했습니다.

### 공통점과 원인 가설

다섯 셀은 모두 수명이 500 미만인데 더 길게 예측됐습니다. 개발용에 없는 충전 방식이고 초기 용량 기울기도 개발용 최댓값을 넘었습니다. B2 단수명 28개 모두를 길게 예측했으며, 이 집단의 MAPE는 **66.03%**, 평균적으로 더 길게 예측한 정도는 **295.52사이클**이었습니다.

B2의 30/39개는 개발용 최소수명 534보다 짧았습니다. 초기 용량 23개, 기울기 21개, ΔQ 로그 분산 18개도 학습 범위를 벗어났습니다. 선택 모델의 용량·기울기 계수가 양수여서 B1의 관계를 B2에 적용할 때 과대예측에 영향을 줬을 가능성이 있습니다. 다만 배치·충전 조건·측정·수명값 차이가 함께 있어 원인으로 확정하지는 않았습니다.

### 추가 확인

저항값이 없는 6개의 MAPE는 44.35%로, 값이 있는 33개의 57.19%보다 낮았습니다. 이미 본 충전 방식 7개도 MAPE가 69.44%로 높았습니다. 따라서 결측이나 새로운 충전 방식 하나로 실패를 설명하기는 어려웠습니다. 각 집단의 수명 구성도 달라 이런 비교만으로 특정 조건의 영향이 없다고 결론 내리지는 않았습니다.

### 개선 방향

같은 수명 기준을 적용한 단수명 셀과 다양한 충전 방식의 데이터를 개발용에 추가하겠습니다. 단일 ΔQ와 여러 변수를 쓰는 모델을 새 개발 배치에서 비교하고, 별도의 새 배치를 최종 평가에 남기겠습니다. 이미 확인한 B2를 보고 수정한 실험은 후속 탐색으로 구분하겠습니다.

## ESS 도메인 해석

### 활용 가능한 의사결정

추가 현장 검증을 거치면 점검·교체 검토의 우선순위를 정하는 보조 정보로 활용할 수 있다고 보았습니다. 하지만 이번 모델은 짧은 수명을 길게 예측해 교체 검토가 늦어질 수 있습니다. **현재 결과만으로 실제 ESS 교체 시점이나 충전 제어를 자동 결정하기는 어렵습니다.**

### 한계와 실제 적용을 위해 필요한 사항

현재 한계는 개발용 28개·별도 검증용 8개로 표본이 작고, 113개 후보 중 좋은 결과를 고르면서 CV가 실제보다 좋아 보일 수 있다는 점입니다. 불완전한 기록의 제외와 B1/B2 수명값 차이도 영향을 줄 수 있습니다. DAY 1에서 B2 분포를 이미 살펴봤으므로 완전히 처음 보는 테스트라고 표현하지 않았습니다.

DAY 1에서 계획했던 품질 기준·종료 용량 판정 기준 변경에 따른 모델 성능 비교는 수행하지 않았습니다. 전처리 전후의 충전 시간 상관 비교와는 구분해 남은 점검으로 두었습니다.

실험 셀의 총사이클 수를 실제 팩의 남은 사용 일수로 바로 바꿀 수는 없으며, 다른 배터리 종류·온도·운전 조건·팩 구성과 예측 불확실성도 확인해야 합니다.

## 참고문헌

- Severson et al. (2019). [Data-driven prediction of battery cycle life before capacity degradation](https://doi.org/10.1038/s41560-019-0356-8). *Nature Energy*, 4, 383–391.

## 팀 구성

- 박진원(울산 1반): 데이터 정제·EDA·피처 엔지니어링·모델 개발·성능 평가(Batch 2)·오류 해석을 진행했습니다.
'''
    (ROOT / 'README.md').write_text(text)
    print('README updated from frozen DAY2 results')


if __name__ == '__main__':
    main()
