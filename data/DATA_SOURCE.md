# 데이터 출처와 포함 범위

- 출처: [MIT-Stanford Dataset / itshpark](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle)
- 원 연구: Severson et al. (2019), DOI: 10.1038/s41560-019-0356-8.
- Kaggle 상세 메타데이터의 라이선스 표기: `Other (specified in description)` (2026-10-01 확인). 이 저장소는 원본 데이터에 별도의 라이선스를 부여하지 않습니다.
- `data/processed/`는 과제에서 내려받은 세 MAT 파일을 코드로 추출·정리한 수치 자료입니다. 특성·요약·선택한 초기 곡선을 포함합니다.
- 수 GB의 원본 MAT, 내려받은 압축파일, 수업 녹음·강의 원문은 포함하지 않습니다.
- 원본 파일별 다운로드 주소·크기·SHA-256 체크섬은 [source_manifest.json](source_manifest.json), 사용 배치·수명값의 기준·분석 한계는 [README](../README.md)에 정리했습니다.

## 저장된 결과 확인

[README의 환경 설정](../README.md#환경-설정)을 마친 뒤 노트북을 실행하거나 `python src/verify_day2.py`로 저장 결과를 검증할 수 있습니다. 저장소의 정제 데이터와 모델을 사용하므로 원본 MAT 파일을 내려받을 필요는 없습니다. DAY 2 재학습도 정제 데이터로 실행할 수 있으며, 별도 임시 폴더를 준비하는 절차는 [DAY 2 노트북](../notebooks/02_DAY2_Modeling.ipynb) 15절에 있습니다.

## 원본부터 데이터와 EDA 재생성

Kaggle에서 아래 세 파일을 내려받아 압축을 풀고, 저장소 최상위의 `data/raw/`에 파일명을 그대로 두세요. 날짜가 다른 Batch 2나 Extra 파일로 대체하지 않습니다.

```text
data/raw/
├── 2017-05-12_batchdata_updated_struct_errorcorrect.mat
├── 2018-02-20_batchdata_updated_struct_errorcorrect.mat
└── 2018-04-12_batchdata_updated_struct_errorcorrect.mat
```

아래 과정은 정제 데이터와 DAY 1 결과를 다시 저장하므로, 기존 결과를 보존할 별도 작업 복제본에서 실행해 주세요. 환경 설정을 마친 뒤 저장소 최상위에서 다음 순서를 따릅니다.

```sh
python src/prepare_data.py
python src/analyze.py
python src/supplement_day1.py
```

`prepare_data.py`는 세 배치의 요약·초기 특성·곡선을 추출합니다. `analyze.py`는 분석 대상과 분할을 정하고 EDA 결과를 저장하며, `supplement_day1.py`는 이 결과로 DAY 1 보완 분석을 만듭니다. DAY 2 학습 입력인 `data/processed/cells_analysis.csv`와 `results/split_assignment.csv`는 `analyze.py`에서 생성합니다. 학습은 DAY 2 노트북 15절의 별도 절차로 진행합니다.
