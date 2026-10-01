from pathlib import Path
import json,pandas as pd
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'results/day2'
def main():
 e=json.loads((D/'evaluation.json').read_text());lock=json.loads((D/'selection_lock.json').read_text());cv=pd.read_csv(D/'cv_candidates.csv').set_index('candidate_id');fm=pd.read_csv(D/'final_metrics.csv')
 if not (ROOT/'docs/day1_readme.md').exists():(ROOT/'docs/day1_readme.md').write_text((ROOT/'README.md').read_text())
 rows=[]
 for fam,cid in lock['family_winners'].items():
  a=cv.loc[cid];v=fm[fm.candidate_id.eq(cid)&fm.dataset.str.startswith('Valid')].iloc[0];t=fm[fm.candidate_id.eq(cid)&fm.dataset.eq('Test_B2')].iloc[0]
  rows.append(f'| {fam}'+(' **(선택)**' if cid==e['winner_id'] else '')+f' | `{a.feature_set}` | {a.target_transform} | {a.cv_mape_pct_mean:.2f} ± {a.cv_mape_pct_sd:.2f} | {v.mape_pct:.2f} | {t.mape_pct:.2f} |')
 family='\n'.join(rows)
 worst='\n'.join(f'| {x["cell_id"]} | {x["actual"]:.0f} | {x["predicted"]:.1f} | {x["ape_pct"]:.1f} | {x["policy"]} |' for x in e['largest_test_errors'])
 ab=cv[cv.family.eq('Ridge')&cv.target_transform.eq('log')&cv.params.eq('{"alpha": 10.0}')]
 labels={'basic7':'기본 7개','deltaq8':'ΔQ log 분산 추가 8개','no_qd_change7':'중복 용량 변화량 제외 7개','no_qd_slope7':'중복 용량 기울기 제외 7개','deltaq_mean8':'ΔQ 평균으로 교체 8개','deltaq_min8':'ΔQ 최소로 교체 8개','no_ir6':'IR 두 특성 제외 6개','deltaq_only1':'ΔQ log 분산 단일 특성'}
 abl='\n'.join(f'| {labels[f]} | {ab[ab.feature_set.eq(f)].iloc[0].cv_mape_pct_mean:.2f} |' for f in labels)
 text=f'''# 초기 100사이클로 배터리 총수명 예측하기

**울산 1반 · 박진원 | 데이터 분석 미니 프로젝트 | 회귀**

초기 측정으로 배터리의 총 충방전 수명을 예측한다. DAY 1 EDA에서 정한 특성과 모델을 DAY 2에 구현했고, **B1 교차검증으로 선택한 Ridge는 내부 검증 MAPE 8.44%, B2 MAPE 55.22%**였다. 단수명 셀을 길게 예측하는 배치 일반화의 한계를 확인했다. 이 결과는 실제 ESS 운전이나 교체 시점에 바로 적용할 성능 근거가 아니다.

## 1. 데이터와 문제 정의

- 입력: 셀별 초기 100사이클 정보. 정답: 제공된 총수명 `cycle_life`.
- 수명 기준: 정격 1.1 Ah의 80%인 0.88 Ah. 잔여 일수나 팩 고장 시점이 아닌 **셀 총사이클 수**를 예측한다.
- 원본 139개 중 DAY 1의 정제 기준에 따라 115개를 EDA했다. 제외 목록과 이유는 [`results/excluded_cells.csv`](results/excluded_cells.csv)에 보존했다.
- B1은 제공된 근사 종료 라벨, B2는 직접 관측한 80% 교차 라벨이다. B1/B3와 B2의 라벨 품질을 동일하게 가정하지 않는다.
- 과제 B2 파일은 2018-02-20, 원저자 공개 전처리 코드의 B2는 2017-06-30이다. 고정 인덱스 연결·수명 가산 규칙을 복사하지 않았다.

| 데이터 | 분석 셀 수 | DAY 2 역할 |
|---|---:|---|
| Batch 1 (2017-05-12) | 36 | 개발 28 / 고정 holdout 8 |
| Batch 2 (2018-02-20) | 39 | 모델 확정 후 최종 평가 |
| Batch 3 (2018-04-12) | 40 | DAY 1 EDA만, 선택 사항인 추가 모델 평가 생략 |

개발·holdout 분할은 DAY 1의 seed=42 결과를 [`split_assignment.csv`](results/split_assignment.csv)에서 그대로 읽는다. 셀 ID 기준 중복은 없고, 한 셀의 반복 측정 행을 학습과 검증에 섞지 않는다. 해독하지 못한 원본 barcode로 물리적 셀 중복까지 입증한 것은 아니다.

## 2. EDA에서 구현으로

| DAY 1 발견 | DAY 2 구현 |
|---|---|
| B2 단수명(<500)은 28/39, B1에는 없음 | 배치별 평가를 분리하고 외삽 실패를 진단 |
| 전체 궤적에서 후반 열화가 가속 | 초기 용량·기울기만 입력, 후기 기울기·knee는 제외 |
| 초기 ΔQ 분산과 수명의 관계 | 분산 추가, 평균/최소 교체, 단일 ΔQ 특성 비교 |
| 용량 변화량-기울기 r=0.985 | 중복 변수 중 하나를 뺀 집합과 Ridge 비교 |
| 충전 조건과의 관계가 배치마다 다름 | 정확한 프로토콜을 묶는 GroupKFold로 개발 평가 |
| B2 내부저항 특성 6개 결측 | 학습 fold 중앙값 대치, IR 제외 집합도 비교 |

DAY 1의 전체 EDA·설계는 [`01_DAY1_EDA.ipynb`](notebooks/01_DAY1_EDA.ipynb)와 [`설계 보고서`](output/pdf/DS-MINI-Design-울산_1반-박진원.pdf)에 있다. 대표 셀 knee 근사는 전체 수명을 보는 보조 탐색이며 예측 입력으로 사용하지 않는다.

### 최종 입력 7개

| 입력 | 정의 |
|---|---|
| `qd_cycle2` | 2사이클 방전 용량 |
| `qd_slope_10_100` | 10-100사이클 QD의 선형 기울기 |
| `ir_mean_2_100` | 2-100사이클 양의 IR 평균 |
| `ir_change_early` | 91-100사이클 IR 평균 - 2-10사이클 IR 평균 |
| `tavg_mean_2_100` | 2-100사이클 유효 평균 온도 |
| `chargetime_mean_2_100` | 2-100사이클 유효 평균 충전 시간 |
| `log10_deltaq_var` | log10 Var[Q100(V)-Q10(V)], 표본분산 ddof=1 |

특성 추출 시 고정한 유효 범위는 용량 0.2<QD<1.5 Ah, 온도 0-80도, 충전 시간 0-120분, IR>0이다. 원본은 그대로 두고 초기 특성 계산에만 적용한다. 전압 곡선은 원본 cycle 참조와 공통 전압 격자를 확인했다. 종료 용량·전체 기록 길이·라벨 상태·셀 ID·100사이클 이후 측정치는 명시적인 입력 허용 목록 밖에 둔다.

## 3. 모델 개발과 선택

- **주 검증:** 개발 28개를 18개 충전 프로토콜로 묶은 GroupKFold 3-fold. 검증 크기 10/9/9, fold 간 동일 프로토콜 중복 0. 같은 프로토콜의 서로 다른 셀이 섞이는 것이 항상 누수라는 뜻은 아니며, 미관측 프로토콜 일반화를 더 엄격하게 확인하려는 설계다.
- **전처리:** 매 fold의 학습 부분으로만 중앙값 대치·표준화를 fit한다. 선형/Ridge는 `imputer → scaler → model`, RF는 `imputer → model`. 정해 둔 8개 특성 집합을 비교하며 전체 데이터에서 새 상관 순위를 학습하지 않는다.
- **사전 고정 탐색:** Dummy 평균 기준선 1개 + 특성 집합 8개 × (Linear raw/log 2개 + Ridge raw/log·α=0.1/1/10/100 8개 + RF 깊이3/무제한·leaf1/3 4개) = **113개 후보, 339개 fold fit**. RF는 트리300, 전체 특성, seed42.
- **선택:** 3개 검증 fold MAPE의 단순 평균 최소. 정확한 동률은 후보 순서로 결정한다. 이 CV는 후보 선택에 쓴 개발 점수이며 독립적인 최종 성능 추정치가 아니다.
- **확정 모델 C037:** 용량 변화량을 제외한 7개 특성, `Ridge(alpha=10)`, ln(수명) 학습 후 exp 역변환. 모든 지표는 원래 사이클 단위에서 산출한다. 클리핑·테스트 기반 보정 없음.
- 계열별 최선 후보까지 평가 전에 고정하고 개발 28개로 학습했다. 같은 모델로 holdout과 B2를 평가하며, holdout을 합쳐 36개로 재학습하지 않았다. [`실험 계획`](results/day2/experiment_plan.json)과 [`선택 고정 기록`](results/day2/selection_lock.json)에 설정·해시를 남겼다.

| 모델 | 계열별 CV 최선 특성 | 타깃 변환 | CV MAPE ± SD | Valid MAPE | B2 MAPE |
|---|---|---|---:|---:|---:|
{family}

단위는 %. SD는 3개 fold의 표본 표준편차이며 신뢰구간이 아니다. **B2에서 Linear가 더 낮은 오차를 냈지만, 그 결과로 선택 모델을 바꾸지 않았다.** 위 비교는 사전 고정한 계열별 후보의 최종 결과 공개다. 113개 전체 후보는 [`cv_candidates.csv`](results/day2/cv_candidates.csv)에 있다.

### 특성 비교

동일한 Ridge·ln 타깃·α=10·동일 fold를 고정한 비교다. 탐색의 공정한 대조이며, 개발 CV에서의 변화가 외부 배치 개선을 보장하지 않는다.

| 특성 집합 | CV MAPE(%) |
|---|---:|
{abl}

![특성 비교](results/day2/figures/feature_ablation.png)

일반 KFold(3, shuffle, seed42)로 확정 모델만 비교한 보조 민감도 MAPE는 **10.05 ± 2.52%**였다. 프로토콜 중복은 fold별4/3/5개였다. 이 결과로 후보를 재선택하지 않았으며, 무작위 분할이 항상 더 좋은 점수를 낸다는 주장은 하지 않는다. 고정 holdout은 알려진 프로토콜 셀5개·새 프로토콜 셀3개다.

## 4. 노션 지정 성능표와 Gap

MAPE = 평균(|실제 - 예측| / 실제) × 100. Train은 훈련 데이터 재예측 오차가 아니라 **B1 검증 fold MAPE의 평균**이다.

| 구분 | MAPE(%) / Gap(%p) | 비고 |
|---|---:|---|
| Train (Batch1 CV) | {e['cv']['cv_mape_pct_mean']:.2f}% | 3개 fold 평균, SD {e['cv']['cv_mape_pct_sd']:.2f}%p |
| Valid (Batch1 Hold-out) | {e['valid']['mape_pct']:.2f}% | B1 고정 8개 |
| Test (Batch2) | {e['test']['mape_pct']:.2f}% | B2 39개 |
| Gap (Train-Valid) | {e['gaps_percentage_points']['valid_minus_cv']:+.2f}%p | Valid - CV |
| Gap (Valid-Test) | {e['gaps_percentage_points']['test_minus_valid']:+.2f}%p | Test - Valid |
| Gap (Target-Test) | {e['gaps_percentage_points']['test_minus_paper_9_1']:+.2f}%p | Test - 논문 참고값 9.1% |

노션의 Gap 이름은 유지하되, 양수가 MAPE 악화를 뜻하도록 계산 방향을 적었다. fold별 표본 수가 달라 CV의 단순 평균과 합친 OOF MAPE는 약간 다를 수 있다.

| 보조 지표 | Valid (n=8) | Test (n=39) |
|---|---:|---:|
| MAE (사이클) | {e['valid']['mae_cycles']:.2f} | {e['test']['mae_cycles']:.2f} |
| RMSE (사이클) | {e['valid']['rmse_cycles']:.2f} | {e['test']['rmse_cycles']:.2f} |
| R² | {e['valid']['r2']:.3f} | {e['test']['r2']:.3f} |

- 내부 CV-holdout 차이 +0.21%p는 작지만 소표본·후보 선택 편향 때문에 과적합이 없다는 확정은 할 수 없다.
- B2는 +46.78%p 악화되어 배치 일반화에 실패했다. B2 R²=-0.713은 B2 평균을 아는 상수 기준보다 제곱오차가 크다는 뜻이다. B1 평균으로 학습한 Dummy와는 서로 다른 기준이다.
- 논문 참고9.1%보다 +46.12%p 높다. 데이터 버전·정제·분할·라벨 조건이 달라 같은 조건의 재현으로 주장하지 않는다. 노션에는9.1% 달성 여부만으로 점수를 결정하는 문턱이 명시돼 있지 않다.

![실제 대 예측](results/day2/figures/actual_predictions.png)

## 5. 가장 크게 틀린 셀과 원인 가설

APE가 큰 순서다. 절대오차 순서와는 다를 수 있다.

| 셀 | 실제 | 예측 | APE(%) | 프로토콜 |
|---|---:|---:|---:|---|
{worst}

**관측:** 다섯 셀 모두 단수명(<500), 미지 프로토콜, 과대예측이며 초기 용량 기울기가 학습 최댓값을 넘는다. B2 30/39개는 B1 개발 최소수명534보다 짧다. 단수명28개 전부를 길게 예측했고, 이 집단의 MAPE는66.03%, 평균 편향은+295.52사이클이었다. 전체 B2의 평균 편향은+223.18사이클이다.

**가설:** 초기 용량23/39개, 기울기21/39개, ΔQ log 분산18/39개가 B1 개발 범위를 벗어난다. 선택 모델에서 초기 용량·기울기 계수는 양수여서 B1에서 학습한 관계가 B2 과대예측에 기여했을 수 있다. 배치·충전·측정·라벨 차이가 함께 변하므로 특정 원인의 기여도를 분리하지 못했다.

**반례:** IR 결측6개 MAPE44.35% vs 비결측33개57.19%, 알려진 프로토콜7개69.44% vs 미지32개52.11%다. IR 결측이나 미지 프로토콜 하나로 실패를 설명할 수 없다. 다만 이 비조정 집단 비교가 결측의 영향이 없음을 입증하지는 않는다. newstructure 미표기30개는 학습 최소수명 미만30개와 정확히 겹쳐 표기 자체의 효과를 분리할 수 없다. B1 근사 라벨과 B2 관측 라벨도 배치와 겹치므로 라벨 차이의 기여도를 숫자로 단정하지 않는다.

![오류 진단](results/day2/figures/error_diagnostics.png)

**개선 계획:** 동일 EOL 기준의 단수명·다양한 프로토콜 개발 데이터를 확보하고, 단일 ΔQ와 다변량 모델의 배치 안정성을 새 개발 배치에서 비교한다. 새 외부 배치를 최종 평가용으로 남긴다. B2를 보고 후보·전처리·제외 기준을 바꾸는 실험은 후속 탐색으로 구분하고, 같은 B2를 미관측 테스트로 다시 주장하지 않는다.

## 6. ESS 활용과 개발 한계

추가 검증을 거친 모델은 점검·교체 검토 우선순위의 보조 정보로 활용할 수 있다. 그러나 이번 모델은 짧은 수명을 길게 예측하므로 교체 검토를 늦출 위험이 있다. 현재 결과로 실제 교체 시점·충전 제어·안전 판단을 자동 결정할 근거는 부족하다.

실험실 급속충전 셀의 총사이클 수를 실제 ESS 팩의 잔여 사용 일수로 직접 바꾸지 않는다. 현장 적용 전에 다른 화학계·온도·운전 패턴·달력 노화·팩 구성, 동일 라벨 정의, 예측 구간과 오차 비용, 범위 밖 입력 감지 및 사람의 점검 절차를 검증해야 한다.

남은 한계는 개발28/holdout8의 소표본, 113개 후보 선택에 따른 낙관성, 제외된 불완전 기록의 선택 영향, B1/B2 라벨 차이, holdout 프로토콜 일부 중복, DAY 1에서 B2 분포·라벨을 이미 EDA한 점이다. B2 점수로 모델을 조정하지 않았지만 완전히 미관찰한 블라인드 테스트라고 표현하지 않는다.

## 7. 산출물과 재현 방법

```text
README.md                       제출용 설명·성능·해석
notebooks/01_DAY1_EDA.ipynb      DAY 1 실행 노트북
notebooks/02_DAY2_Modeling.ipynb DAY 2 예측·지표 재현 노트북
src/prepare_data.py             원본 HDF5에서 필요한 초기 정보 추출
src/train_day2.py               사전 계획 → CV 선택 → 고정 모델 평가
src/verify_day2.py              분할·전처리·예측·지표·해시 독립 검증
data/processed/                 작은 추출 데이터 (원본 불변)
results/day2/                  후보·fold·모델·예측·오차 분석·검증 기록
output/pdf/                    DAY 1 설계서 / DAY 2 결과 보고서
docs/                          노션 요구·수업 정합성·검토 기록
```

Python3.12와 `requirements.txt`의 고정 버전을 사용했다. 전체 버전은 실험 계획에 기록돼 있다. **이미 실행된 프로젝트**에서는 결과를 다시 선택하지 않고 검증·노트북 재현만 실행한다.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/verify_day2.py
```

노트북은 위 환경의 Python 커널로 전체 실행하면 저장 모델 예측을 재현한다. 새 데이터 분할이나 재튜닝을 실행하지 않는다. joblib 모델은 이 프로젝트에서 직접 생성한 파일만 읽는다.

**새 실험 디렉터리에서 처음부터 재현**하려면 패키지의 `data/processed/`와 DAY 1 `results/split_assignment.csv`를 사용하고, 그 새 디렉터리에는 `results/day2/`를 복사하지 않은 상태에서 실행한다. 기존 제출 결과는 덮어쓰지 않는다.

```sh
python src/train_day2.py --phase select
python src/train_day2.py --phase evaluate
python src/verify_day2.py
python src/build_day2_figures.py
python src/build_day2_report.py
python src/build_day2_notebook.py
```

저장된 선택 결과가 있으면 `select`, 평가 결과가 있으면 `evaluate`는 중단하도록 설계했다. 명시적인 새 실험 없이 테스트 점수를 보고 재튜닝하지 않기 위해서다. 그래프·PDF의 표시는 저장 결과에서 생성된다.

원본부터 다시 추출하려면 아래 Kaggle에서 세 MAT 파일을 내려받아 `data/raw/`에 둔 뒤 `prepare_data.py → analyze.py → verify_day1.py`를 실행한다. 원본 추출 뒤에도 제외 기준·셀 ID·기존 분할을 확인한다. 수 GB의 MAT·다운로드 압축·가상환경·임시파일은 제출 ZIP에서 제외한다. 공개 저장소에는 데이터셋의 라이선스·재배포 조건에 맞는 파일만 올린다.

[`DAY 2 결과 보고서`](output/pdf/DS-MINI-Result-울산_1반-박진원.pdf) · [`DAY 2 노트북`](notebooks/02_DAY2_Modeling.ipynb) · [`노션 요구사항`](docs/notion_day2_requirements.txt) · [`최종 검토`](docs/final_day2_review.md)

## 8. 평가 기준 대응과 출처

| 노션 평가항목 | 공식 배점 | 근거 |
|---|---:|---|
| 전략 → 구현 | 20 | 2-3절 EDA·특성 집합·모델 비교 |
| Pipeline 개발 | 40 | 고정 분할, 프로토콜 CV, fold 내부 전처리, 초기 입력 제한, 검증 기록 |
| 성능 리포팅·해석 | 20 | 4절 지정 6행 표·세 Gap·논문 참고 비교 |
| 도메인 해석·한계 | 20 | 5-6절 오류 사례·반례·개선·현장 적용 한계 |

배점은 공식 평가 가중치이며 자체 채점이나 점수 보장이 아니다. 노션의 실제 제출 형식에 맞춰 [공개 GitHub 저장소](https://github.com/nowjinpark/skala-battery-cycle-life)에 게시했다. 반별 Slack thread로 링크를 전송하는 단계는 수행하지 않았다.

- [DS Mini Project 과제 안내](https://actually-war-1ea.notion.site/DS-Mini-Project-32d7f4c8669380338a27f90c471c1fcb) (2026-10-01 확인)
- [Kaggle 데이터셋](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle)
- Severson et al. (2019), [Data-driven prediction of battery cycle life before capacity degradation](https://doi.org/10.1038/s41560-019-0356-8), Nature Energy. [원저자 코드](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation)
- [scikit-learn 그룹 교차검증](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data), [Pipeline](https://scikit-learn.org/stable/modules/compose.html#pipeline-chaining-estimators), [TransformedTargetRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html)
- 수업의 Statistics·MLDL·Wrap-up·Evaluation Metrics·ML Hyperparameters와 강의 녹음. 세부 대응은 [`수업 정합성 검토`](docs/course_alignment_review.md), DAY 1 이력은 [`기존 README`](docs/day1_readme.md)에 보존했다.

작성자/팀: 울산1반 박진원. 이 저장소의 수행 범위는 데이터 정제·EDA·특성 설계·모델 비교·오류 해석·보고서 작성이다.
'''
 (ROOT/'README.md').write_text(text)
 print('README updated from frozen DAY2 results')
if __name__=='__main__':main()
