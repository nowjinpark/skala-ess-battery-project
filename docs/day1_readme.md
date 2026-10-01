# 초기 100사이클로 배터리 총수명 예측하기 - DAY 1

울산 1반 박진원. 현재 범위는 데이터 이해, EDA, 특성·모델 설계다.
예측 모델 학습 및 성능 평가는 DAY 2에 수행한다.

## 과제 범위

- 회귀: 초기 100사이클에서 계산한 정보로 80% 정격 용량에 도달하는 총수명 `cycle_life`를 예측한다.
- 강의 녹음본을 확인하여 **DAY 1 EDA는 Batch 1·2·3**, Extra는 제외한다. Batch 3의 추가 모델 성능 평가는 DAY 2 선택 사항이다.
- Batch 1 내부 개발/CV 및 holdout, Batch 2 최종 평가를 설계한다. Batch 3는 현재 EDA 전용이며 모델 선택에 사용하지 않는다.
- 데이터 수명분포, 열화곡선, Q100(V)-Q10(V), 충전조건, 초기특성 상관관계를 조사한다.
- 원본 파일은 수정하지 않으며 `sources/`의 수업 자료와 기존 실습도 수정하지 않는다.

작업 위치: `/Users/jinwon/workspace/데이터 분석 미니플젝`

강의 녹음본 315행에서 세 배치의 EDA를 안내하고, 350~376행의 Batch 3 선택 사항은 DAY 2 추가 평가에 해당한다.
이 프로젝트는 DAY 1에 필요한 정리·탐색·설계까지만 수행한다.

## 먼저 볼 파일

1. `notebooks/01_DAY1_EDA.ipynb`: 설명과 코드를 함께 읽는 학습용 노트북.
2. `output/pdf/DS-MINI-Design-울산_1반-박진원.pdf`: DAY 1 평가항목에 집중한 6쪽 설계서(본문·표 흑백, 그래프 색상).
3. `results/analysis.json`: 실제 계산한 수치와 관찰.
4. `results/excluded_cells.csv`: 제외한 배터리와 사유.
5. `results/split_assignment.csv`: DAY 2에서 그대로 사용할 고정 분할.
6. `docs/troubleshooting.md`: 확인된 문제, 판단 근거, 남은 한계.
7. `docs/requirements.md`: 녹음본과 과제 안내를 대조한 DAY 1 범위.
8. `docs/final_day1_review.md`, `docs/course_alignment_review.md`: 노션 세부 요구와 수업 개념의 최종 대조 기록.

## 실행

프로젝트 폴더에서 Python 3.12 가상환경을 활성화한 뒤 아래 순서로 실행한다.
실제 사용 버전은 `requirements.txt`에 기록한다.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/prepare_data.py
python src/analyze.py
python src/verify_day1.py
python src/build_notebook.py
python src/review_day1.py
python src/build_report_figures.py
python src/build_report.py
```

노트북에서는 프로젝트 `.venv`를 Python 커널로 선택한다.
`build_notebook.py`는 학습용 노트북을 생성한다. 결과를 갱신하려면 노트북에서 전체 셀 실행을 선택한다.
전체 MAT 파일을 다시 읽지 않아도 `data/processed/`의 작은 추출 파일로 EDA를 재실행할 수 있다.
MATLAB 7.3 파일은 HDF5 형식이므로 h5py로 필요한 필드만 읽는다.
보고서용 색상 그림은 `results/figures_report/`에 별도로 생성한다. 이전 흑백 그림은 `results/figures_bw/`에 보존했다. 기존 학습용 노트북의 색상 그림과 분석 결과는 유지한다.
보고서 한글 폰트와 사용권은 `assets/fonts/`에 포함했다.

## 데이터와 출처

[Kaggle 과제 데이터](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle)의 다음 파일을 `data/raw/`에 둔다.

- `2017-05-12_batchdata_updated_struct_errorcorrect.mat`
- `2018-02-20_batchdata_updated_struct_errorcorrect.mat`
- `2018-04-12_batchdata_updated_struct_errorcorrect.mat`

원본 MAT와 다운로드 압축 파일은 용량이 크므로 `.gitignore`에 포함했다.
출처·크기·무결성 확인 기록은 다운로드 manifest를 참고한다.
GitHub/Slack 제출은 이 작업에서 수행하지 않는다.

[과제 안내](https://actually-war-1ea.notion.site/DS-Mini-Project-32d7f4c8669380338a27f90c471c1fcb)
및 제공 `30-ESSHealth-scratch.ipynb`를 참고하여 분석 흐름을 구성했다.

Severson et al. (2019), [Data-driven prediction of battery cycle life before capacity degradation](https://doi.org/10.1038/s41560-019-0356-8).
원저자 [공개 코드](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation)는 `references/author_...`에 참고용으로 보존했다.

**버전 주의:** 원저자 BuildPkl_Batch2/LoadData.m의 Batch 2는 2017-06-30이며 과제 파일은 2018-02-20이다.
인덱스 기반 동일 배터리 연결 및 수명 가산 규칙을 확인 없이 복사하지 않는다.
검열된 수명 라벨은 원본을 보존하고 확정된 수명과 구분한다.
Batch 1 분석 대상 36개와 Batch 3 분석 대상 40개는 실제 80% 임계값 교차를 직접 관측한 라벨이 아니다.
제공된 근사 종료 라벨로 유지했으며 Batch 2의 직접 관측 라벨과 구분한다.

## 실행 결과

원본 배터리 139개 / 반복 측정 116,722행에서, 제외 사유를 기록한 뒤 배터리 115개를 EDA했다.

| 배치 | 원본 | 분석 대상 | 수명 중앙값 | 현재 역할 |
|---|---:|---:|---:|---|
| Batch 1 | 46 | 36 | 772.5사이클 | 개발 28 / 내부 holdout 8 설계 |
| Batch 2 | 47 | 39 | 472.0사이클 | DAY 2 최종 평가 설계 |
| Batch 3 | 46 | 40 | 964.5사이클 | DAY 1 EDA 전용 |

- B1·B3는 제공 근사 종료 라벨, B2는 실제 80% 교차 라벨이다. 배치 차이를 해석할 때 고려한다.
- B1 개발 28개에서 log10 Var(Delta Q)와 수명의 상관은 약 -0.782였다. 예측 성능 개선 여부는 DAY 2 검증 전이다.
- 단수명(<500) 표본은 B2에 28개 있지만 B1·B3에는 없다. 모델이 학습한 범위 밖을 예측할 때의 한계를 설계에 반영했다.
- 데이터·곡선 계산·초기 관측 범위·B3 역할·B1/B2 분할 보존 검증을 통과했다.
- 세 배치의 특성 상관은 설명용 EDA로 비교했다. 모델 선택용 개발 상관과 검증은 B1 개발 28개만 사용한다.
- 대표 3셀의 두 직선 근사는 전체 수명 곡선의 탐색적 요약이다. 수업 필수 검출법이나 초기 예측 입력으로 주장하지 않는다.
- 노트북 전체 실행과 PDF 렌더링 검수 기록은 `results/`에 저장한다.
- `output/DAY1_분석자료.zip`에는 코드·작은 추출 데이터·노트북·PDF를 묶는다. 대용량 원본과 가상환경은 포함하지 않는다.

## DAY 1을 이해하는 순서

1. 노트북 첫 부분에서 “100사이클까지 보고 총수명을 예측한다”는 문제와 데이터 단위를 확인한다.
2. 배터리 1개가 표본 1개라는 점, 원본 기록과 분석 대상이 다른 이유를 확인한다.
3. 다섯 EDA 질문을 그래프 → 관찰 → 해석 → 모델 설계 순으로 읽는다.
4. 특히 Q100(V)-Q10(V)의 분산과 로그가 왜 수명 후보 특성이 되는지 설명해 본다.
5. PDF의 후보 모델·검증 설계를 읽고, 각 모델을 비교하려는 이유를 자신의 말로 정리한다.

## DAY 2에 지킬 규칙

- 배터리 하나가 모델 입력 한 행이다. 동일 배터리의 사이클 행을 무작위로 나눠 학습/검증하지 않는다.
- 용량 80% 기준과 열화곡선 전체는 EDA 및 정답 검증에 사용한다. 모델 특성은 초기 100사이클 이내로 제한한다.
- 3-fold CV의 각 학습 부분 안에서 결측치 대체·스케일링을 학습하는 Pipeline을 사용한다.
- Dummy/선형/Ridge/Random Forest를 같은 분할에서 비교한다. 기본 특성 집합에 Delta Q를 추가하는 비교를 포함한다.
- 모델 선택은 Batch 1 개발 데이터에서 수행한다. Batch 2 점수를 보고 반복 튜닝하지 않는다.
- DAY 1 EDA에서 Batch 2 라벨과 Batch 1 holdout 분포를 관찰했다는 한계를 공개한다.
- MAPE는 백분율로 보고하고 MAE/RMSE는 사이클 단위로 보고한다. Gap은 계산 방향과 %p를 명시한다.
- 논문의 9.1%는 다른 데이터 분할/정제/배치 버전의 참고치다. 동일 조건의 재현 또는 운영 성과로 표현하지 않는다.
