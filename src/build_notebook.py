"""Build the Korean DAY 1 study notebook without running analysis or training.

Run from any directory: python /path/to/project/src/build_notebook.py
Only writes notebooks/01_DAY1_EDA.ipynb; raw/processed data are read by its cells.
"""
from pathlib import Path
import json
import textwrap


ROOT = Path(__file__).resolve().parents[1]


def build_notebook():
    cells = []

    def md(source):
        cells.append({"cell_type": "markdown", "metadata": {},
                      "source": textwrap.dedent(source).strip() + "\n"})

    def code(source):
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                      "outputs": [], "source": textwrap.dedent(source).strip() + "\n"})

    md("""
    # DAY 1 — 초기 배터리 데이터로 수명을 예측하기 위한 EDA

    **목표:** 첫 100 cycle까지 관측한 정보로 최종 `cycle_life`를 예측할 준비를 한다.
    오늘은 **Batch 1·Batch 2·Batch 3의 구조, 데이터 품질, 분포, 피처 후보**를 확인한다.
    녹음의 DAY 1 안내에 따라 세 배치를 EDA에 사용하며, Extra 데이터는 제외한다.
    모델 학습과 하이퍼파라미터 탐색은 다음 단계에서 진행한다.

    - **분석 단위:** 배터리 셀 기록 한 개. 수만 개의 cycle 행은 독립 배터리 수가 아니다.
      물리 셀 식별자는 미해독 상태이므로 셀 기록 간 물리적 중복이 없다고 확정하지 않는다.
    - **X 후보:** 100 cycle 이내 용량·온도·내부저항·충전시간·방전곡선 변화.
    - **Y:** 제공 수명 라벨. 원본 `cycle_life_raw`와 분석용 `cycle_life`를 구분한다.
      Batch 1·Batch 3은 80% 근처의 **근사 종료 라벨**, Batch 2는 **80% 임계값 교차 라벨**이다.
      근사 라벨도 원본 값을 그대로 유지하며 실제 임계값 교차를 확인한 정답이라고 부르지 않는다.
    - **평가 설계:** Batch 1 개발 구간에서 교차검증·모델 선택, 고정한 Batch 1 holdout에서 내부 평가,
      Batch 2에서 배치 간 평가를 진행한다. **Batch 3은 현재 `Batch3_EDA_only`로 둔다.**
      DAY 2에서 Batch 3을 추가 성능 평가에 쓰는 것은 선택 사항이며 오늘은 모델을 평가하지 않는다.
      오늘은 holdout과 Batch 2의 분포도 확인하므로 완전히 보지 않은 테스트라고 부르지 않는다.
      모델 선택을 위한 피처·타깃 상관 분석은 개발 구간으로 제한한다.

    이 노트북은 연구 보조 도구로 생성했다. 셀을 실행하고 결과와 제외 근거를 직접 확인한 뒤,
    마지막 질문에 자신의 해석을 적는다. 실행 전에는 어떠한 성능도 달성했다고 주장하지 않는다.
    """)

    md("""
    ## 1. 환경과 데이터 준비 상태

    프로젝트 폴더 또는 `notebooks/`에서 실행할 수 있다. 원본 MAT를 다시 읽는 대신
    데이터 준비 단계에서 만든 작은 표와 cycle 10·100 곡선을 읽는다.
    필요한 파일이 없으면 준비 단계를 먼저 완료한다. 이 노트북은 데이터를 다운로드하거나 변경하지 않는다.
    """)
    code(r"""
    from pathlib import Path
    import sys
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display

    cwd = Path.cwd().resolve()
    ROOT = next((p for p in [cwd, *cwd.parents]
                 if (p / 'src' / 'prepare_data.py').is_file()
                 and (p / 'data' / 'processed').is_dir()), None)
    if ROOT is None:
        raise FileNotFoundError('프로젝트 또는 notebooks 폴더에서 실행하세요.')
    if str(ROOT / 'src') not in sys.path:
        sys.path.insert(0, str(ROOT / 'src'))
    PROCESSED = ROOT / 'data' / 'processed'
    BATCHES = {'Batch 1': 'batch1', 'Batch 2': 'batch2', 'Batch 3': 'batch3'}
    COLORS = {'Batch 1': '#2878B5', 'Batch 2': '#D87526', 'Batch 3': '#369667'}
    needed = [PROCESSED / f'{prefix}_{suffix}'
              for prefix in BATCHES.values()
              for suffix in ['cells.csv', 'summary.csv.gz', 'curves.npz']]
    missing = [str(p.relative_to(ROOT)) for p in needed if not p.is_file()]
    if missing:
        raise FileNotFoundError('데이터 준비가 필요합니다: ' + ', '.join(missing))
    cohort_paths = [PROCESSED / 'cells_analysis.csv', ROOT / 'results' / 'cells_analysis.csv']
    COHORT_PATH = next((p for p in cohort_paths if p.is_file()), None)
    if COHORT_PATH is None:
        raise FileNotFoundError('데이터 품질 검토 결과 cells_analysis.csv가 필요합니다.')
    plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False,
                         'axes.spines.right': False, 'font.size': 11,
                         'axes.titlesize': 12, 'axes.labelsize': 11})
    pd.set_option('display.max_columns', 25)
    print('프로젝트:', ROOT)
    print('분석 대상 판정표:', COHORT_PATH.relative_to(ROOT))
    """)
    code(r"""
    raw_parts, summary_parts, curves = [], [], {}
    for batch, prefix in BATCHES.items():
        part = pd.read_csv(PROCESSED / f'{prefix}_cells.csv',
                           dtype={'barcode': 'string', 'channel_id': 'string'})
        raw_parts.append(part)
        summary_parts.append(pd.read_csv(PROCESSED / f'{prefix}_summary.csv.gz'))
        with np.load(PROCESSED / f'{prefix}_curves.npz', allow_pickle=False) as archive:
            curves.update({name: archive[name].copy() for name in archive.files})
    raw = pd.concat(raw_parts, ignore_index=True)
    summary = pd.concat(summary_parts, ignore_index=True)
    decisions = pd.read_csv(COHORT_PATH)
    required = {'cell_id', 'analysis_eligible', 'exclusion_reason', 'cycle_life',
                'split_role', 'label_status'}
    assert required.issubset(decisions.columns), f'판정표 필드 확인: {required - set(decisions.columns)}'
    assert raw.cell_id.is_unique and decisions.cell_id.is_unique
    assert set(raw.batch) == set(BATCHES), '세 배치의 준비 상태를 확인하세요.'
    assert set(raw.cell_id) == set(decisions.cell_id), '원본과 판정표의 배터리 목록이 다릅니다.'
    normalized = decisions.analysis_eligible.astype(str).str.lower().str.strip()
    assert normalized.isin(['true', 'false', '1', '0']).all(), '분석 대상 판정값을 확인하세요.'
    decisions['analysis_eligible'] = normalized.isin(['true', '1'])
    # 원본 target은 덮어쓰지 않고, 별도의 분석용 target을 매핑한다.
    cells = raw.merge(decisions[list(required)], on='cell_id', validate='one_to_one')
    eligible = cells.loc[cells.analysis_eligible].copy()
    assert eligible.cycle_life.notna().all() and (eligible.cycle_life > 0).all()
    role_counts = eligible.split_role.value_counts()
    assert role_counts.get('Batch1_CV_development', 0) == 28
    assert role_counts.get('Batch1_holdout', 0) == 8
    assert role_counts.get('Batch2_final_test', 0) == 39
    assert eligible.loc[eligible.batch.eq('Batch 3'), 'split_role'].eq('Batch3_EDA_only').all()
    print('기존 seed=42 분할 유지: Batch 1 개발 28 / holdout 8, Batch 2 평가 39')
    display(cells[['batch', 'cell_id', 'policy', 'cycle_life_raw', 'cycle_life',
                   'label_status', 'analysis_eligible', 'exclusion_reason']].head(8))
    display(cells.groupby(['batch', 'split_role']).size().rename('battery_count').to_frame())
    """)

    md("""
    ## 2. 변수와 표의 단위

    | 변수 | 뜻 | 사용 시점 |
    |---|---|---|
    | `cell_id`, `barcode`, `batch` | 파일 내 ID, 물리 셀 식별자, 실험 배치 | 추적·분할용; 예측 피처로 바로 넣지 않음 |
    | `cycle` | 충전·방전 반복 번호 | 100 cycle 이내 정보만 X에 사용 |
    | `QDischarge`, `QCharge` | 방전·충전 용량(Ah) | 초기 추세·변동을 요약 |
    | `IR` | 내부저항 | 0·누락·단위·측정 조건 점검 |
    | `Tavg`, `Tmax`, `Tmin` | cycle별 온도 요약 | 초기 평균·상승 정도 후보 |
    | `chargetime` | 충전시간 | 측정 단위와 정의를 원자료와 대조 |
    | `Qdlin`, `Vdlin` | 공통 전압 격자 위의 방전 용량곡선과 전압 | cycle 10·100의 곡선 차이 계산 |
    | `cycle_life_raw` | 원본 파일의 수명 라벨 | 관측 중단·예외 여부 점검 |
    | `cycle_life` | 검토 후 유지한 제공 라벨; 배치별 근사/실측 차이 있음 | 예측할 정답 Y |
    | `label_status` | 근사 종료·80% 교차·미해결 라벨 구분 | 라벨 신뢰성과 분석 한계 설명 |

    공칭 용량 1.1Ah의 80%는 **0.88Ah**다. 초기 데이터에 0이 포함되거나 측정이 일시적으로
    흔들릴 수 있으므로, 단순히 최초로 0.88보다 작은 값만 찾고 수명이라고 단정하지 않는다.
    """)
    code(r"""
    inventory = raw.groupby('batch').agg(
        raw_cells=('cell_id', 'nunique'), cycle_rows=('n_summary', 'sum'),
        earliest_cycle=('first_cycle', 'min'), latest_cycle=('last_cycle', 'max'))
    inventory['analysis_cells'] = eligible.groupby('batch').cell_id.nunique()
    inventory['excluded_cells'] = inventory.raw_cells - inventory.analysis_cells.fillna(0)
    display(inventory)
    quality = summary.groupby('batch').agg(
        rows=('cell_id', 'size'), missing_qd=('QDischarge', lambda s: s.isna().sum()),
        zero_qd=('QDischarge', lambda s: (s == 0).sum()))
    display(quality)
    display(cells.loc[~cells.analysis_eligible,
                      ['batch', 'cell_id', 'cycle_life_raw', 'last_cycle', 'last_qd',
                       'first_below_80', 'exclusion_reason']])
    display(cells.groupby(['batch', 'label_status', 'analysis_eligible']).size()
            .rename('cell_records').to_frame())
    display(eligible.groupby('batch').agg(
        kept_cells=('cell_id', 'size'), min_final_qd=('last_qd', 'min'),
        max_final_qd=('last_qd', 'max')))
    assert np.allclose(eligible.cycle_life, eligible.cycle_life_raw), '원본 라벨 변경 여부를 확인하세요.'
    """)
    md("""
    **제외는 점수가 좋아지는 방향으로 결정하지 않는다.** 관측 중단, 수명 라벨 부재, 측정 오류처럼
    미리 설명할 수 있는 품질 기준으로 결정하고 원본은 보존한다. `capacity_plausible`은 준비 단계의
    탐색용 점검 범위(0.2~1.5Ah)이며 물리적인 판정 법칙이 아니다. 이 범위 때문에 빠진 값도 확인한다.

    **이번 데이터에서 실제 확인한 라벨 차이**

    - Batch 1의 원본 46개 모두 `cycle_life_raw = n_summary + 1`이며, 엄격한 0.88Ah 교차를
      직접 관측한 라벨이 아니다. 원저자 자료의 비완주 5개와 다른 날짜의 후속 기록 연결이 필요한 5개,
      총 10개를 제외했다.
    - 유지한 Batch 1 **36개**의 종료 용량은 약 **0.88006~0.88336Ah**다.
      제공된 근사 라벨을 그대로 사용하고 `provided_near80_endpoint_proxy`로 표시한다.
      0.885Ah는 종료 상태 점검값이며 새로운 EOL 정의가 아니다. 엄격한 EOL은 여전히 0.88Ah다.
    - Batch 2는 원본 **47개** 중 수명 라벨이 없는 8개를 제외했다. 유지한 **39개**는 제공 라벨과
      실제 최초 QD<0.88Ah cycle이 일치하며 `observed_80pct_crossing`으로 표시한다.
    - Batch 3은 원본 **46개** 중 원저자 제외 목록의 **6개**를 분리했다. 파일 날짜와 원본 셀 순서를
      확인한 후 0부터 시작하는 셀 번호 `b3c37/2/23/32/42/43`을 적용했다.
      유지한 **40개**는 모두 `cycle_life_raw = n_summary + 1`이고 엄격한 0.88Ah 교차 관측이 없다.
      종료 용량은 약 **0.880005~0.881670Ah**이며 `provided_near80_endpoint_proxy`로 표시한다.
      Batch 3은 EDA에만 사용하고 Batch 1 개발/holdout이나 Batch 2 평가에 합치지 않는다.

    세 배치의 **원본 139개 중 분석 대상은 115개(36/39/40)**다. 원본 cycle 행 **116,722개**를
    독립 배터리 표본 수로 세면 안 된다.

    따라서 모든 분석 대상을 **“실제 EOL을 확인한 셀”**이라고 묶어 부르면 부정확하다.
    라벨 정의 차이와 제외로 인한 분포 변화도 배치 간 오차의 원인이 될 수 있다.
    다음 단계에서는 근접 종료 판정값 0.8825/0.885Ah의 민감도를 점검한다.

    **데이터 출처 주의:** 이번 Batch 2 파일은 `2018-02-20`이다. 원저자의 논문용 Batch 2 로더는
    `2017-06-30`을 사용한다. 따라서 원저자 코드의 고정 ID 삭제나 배치 연결을 복사해 적용하면 안 된다.
    원논문의 9.1%는 참고 목표이며 데이터·분할 조건이 같은 재현 결과로 주장하지 않는다.
    """)
    code(r"""
    # batch_date는 기록된 값과 파일명 모두 확인한다.
    metadata_rows = []
    for batch, prefix in BATCHES.items():
        path = PROCESSED / f'{prefix}_metadata.json'
        if path.exists():
            meta = json.loads(path.read_text())
            metadata_rows.append({k: meta.get(k) for k in ['batch', 'file', 'batch_date', 'n_cells']})
    display(pd.DataFrame(metadata_rows))
    ids = cells[['batch', 'cell_id', 'barcode']].copy()
    ids['barcode'] = ids.barcode.fillna('').astype(str).str.strip()
    known_ids = ids.loc[ids.barcode.ne('')]
    repeated = known_ids.groupby('barcode').batch.nunique()
    cross_batch_barcodes = repeated[repeated > 1].index
    display(known_ids.loc[known_ids.barcode.isin(cross_batch_barcodes)])
    print('배치 간 겹치는 비어 있지 않은 barcode 수:', len(cross_batch_barcodes))
    print('barcode 누락 수:', ids.barcode.eq('').sum(), '— 누락이면 중복이 없다고 확정할 수 없습니다.')
    """)

    md("""
    ## 3. 수명 분포와 장·단수명 비율

    원본 라벨 분포와 분석 대상으로 확정한 라벨 분포를 구별한다.
    Batch 1·3 근사 라벨과 Batch 2 임계값 교차 라벨의 차이를 유지한 채 세 배치를 비교한다.
    아래의 **단수명 <500, 장수명 >1,000**은 EDA용 구간이다.
    이번 회귀의 정답을 이진 분류로 바꾸는 기준은 아니다. 비율의 분모는 각 배치의 분석 대상 배터리 수다.
    """)
    code(r"""
    display(eligible.groupby('batch').cycle_life.describe().round(1))
    life_bands = eligible[['batch', 'cell_id', 'cycle_life']].copy()
    life_bands['life_group'] = np.select(
        [life_bands.cycle_life < 500, life_bands.cycle_life > 1000],
        ['Short: <500', 'Long: >1000'], default='Middle: 500–1000')
    counts = pd.crosstab(life_bands.batch, life_bands.life_group).reindex(
        index=list(BATCHES),
        columns=['Short: <500', 'Middle: 500–1000', 'Long: >1000'], fill_value=0)
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100
    display(counts)
    display(percentages.round(1).add_suffix(' (%)'))

    all_life = pd.concat([raw.cycle_life_raw, eligible.cycle_life]).dropna()
    bins = np.linspace(0, max(1000, all_life.max()) * 1.05, 14)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for ax, batch in zip(axes.flat, BATCHES):
        raw_life = raw.loc[raw.batch.eq(batch), 'cycle_life_raw'].dropna()
        kept_life = eligible.loc[eligible.batch.eq(batch), 'cycle_life']
        ax.hist(raw_life, bins=bins, histtype='step', color='grey', linewidth=1.7,
                label=f'Raw nonmissing labels (n={len(raw_life)})')
        ax.hist(kept_life, bins=bins, alpha=.65, color=COLORS[batch],
                label=f'Analysis cohort (n={len(kept_life)})')
        ax.set(title=batch, xlabel='Cycle life', ylabel='Battery count')
        ax.legend(fontsize=9)
    percentages.plot.bar(stacked=True, ax=axes[1, 1],
                         color=['#C96F57', '#ADBDD0', '#4D8779'])
    axes[1, 1].set(title='Life bands in analysis cohort', xlabel='', ylabel='Share (%)', ylim=(0, 100))
    axes[1, 1].tick_params(axis='x', rotation=0)
    axes[1, 1].legend(fontsize=9, loc='upper center', bbox_to_anchor=(.5, -.1), ncol=1)
    plt.show()
    """)

    md("""
    ## 4. 배터리별 방전용량 곡선

    전체 수명 곡선은 데이터 이해와 라벨 점검에 사용한다. **100 cycle 이후의 값·전체 기록 길이·마지막
    용량을 X에 넣으면 미래 정보를 보는 누수**가 된다. 초기 용량이 거의 줄지 않아도 최종 수명은 다를 수 있다.
    """)
    code(r"""
    plot_data = summary.loc[summary.capacity_plausible.astype(str).str.lower().eq('true')
                            & summary.cell_id.isin(eligible.cell_id) & summary.cycle.ge(2)].copy()
    # 마지막 용량의 급락 등 점검 범위 밖 값은 raw summary에 남아 있다.
    # 3개 배치를 같은 축 범위로 비교하고 네 번째 칸에 읽는 법을 둔다.
    for early_only in [False, True]:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True,
                                 layout='constrained')
        for ax, batch in zip(axes.flat, BATCHES):
            selected = plot_data.loc[plot_data.batch.eq(batch)]
            if early_only:
                selected = selected.loc[selected.cycle.le(100)]
            for cell_id, group in selected.groupby('cell_id'):
                group = group.sort_values('cycle')
                ax.plot(group.cycle, group.QDischarge, color=COLORS[batch], alpha=.32, linewidth=.8)
            ax.axhline(.88, color='black', linestyle='--', linewidth=.9)
            title = 'first 100 cycles' if early_only else 'observed trajectories'
            ax.set(title=f'{batch}: {title} (n={selected.cell_id.nunique()})',
                   xlabel='Recorded cycle', ylabel='Discharge capacity (Ah)')
        axes[1, 1].set_axis_off()
        axes[1, 1].text(.06, .87,
            'Each line = one eligible cell record\n\n'
            'Dashed line = 0.88 Ah (80% nominal capacity)\n'
            'Batches 1 and 3: near-80% endpoint proxies\n'
            'Batch 3: EDA only\n\n'
            'Full-life trajectories are for EDA.\n'
            'Predictors must use the first 100 cycles only.',
            transform=axes[1, 1].transAxes, va='top', linespacing=1.8)
        plt.show()
    """)

    md("""
    ## 5. ΔQ(V): cycle 10과 100에서 방전곡선이 얼마나 달라졌는가?

    같은 **전압 값**에서 측정한 용량의 차이를 `ΔQ(V) = Q100(V) − Q10(V)`로 계산한다.
    이 차이곡선의 분산은 용량 총량만 보았을 때 놓칠 수 있는 곡선 모양의 변화를 요약한다.
    분산에 `log10`을 취한 값이 피처 후보 `log10_deltaq_var`다.

    배열의 인덱스와 실제 cycle 번호는 다를 수 있다. 아래에서는 추출 단계가 기록한 cycle 위치를 먼저
    보고, 곡선 길이·전압 격자·유한값·차이 계산·분산 계산이 서로 맞는지 확인한다.
    **로그를 계산할 수 없는 0 분산을 임의의 작은 상수로 몰래 바꾸지 않는다.**
    """)
    code(r"""
    available = eligible.loc[eligible.cell_id.map(lambda k: f'{k}_q10' in curves
                                                 and f'{k}_q100' in curves)].copy()
    if available.empty:
        raise ValueError('검증 가능한 cycle 10/100 곡선이 없습니다. curve_issue를 먼저 확인하세요.')
    curve_checks = []
    for row in available.itertuples():
        cid = row.cell_id
        v, q10, q100 = (curves[f'{cid}_{name}'] for name in ['voltage', 'q10', 'q100'])
        assert len(v) == len(q10) == len(q100), cid
        assert np.isfinite(np.r_[v, q10, q100]).all(), cid
        assert np.all(np.diff(v) > 0) or np.all(np.diff(v) < 0), f'{cid}: voltage grid'
        delta = q100 - q10
        assert np.allclose(delta, curves[f'{cid}_deltaq']), cid
        variance = np.var(delta, ddof=1)  # MATLAB var와 같은 표본분산
        assert variance > 0, f'{cid}: zero variance'
        assert np.isclose(np.log10(variance), row.log10_deltaq_var), cid
        curve_checks.append({'batch': row.batch, 'cell_id': cid, 'voltage_points': len(v),
                             'slot10': row.curve_slot10_zero_based,
                             'slot100': row.curve_slot100_zero_based,
                             'recomputed_log10_var': np.log10(variance)})
    checked = pd.DataFrame(curve_checks)
    display(checked.groupby('batch').agg(checked_cells=('cell_id', 'nunique'),
            voltage_points_min=('voltage_points', 'min'), voltage_points_max=('voltage_points', 'max')))
    display(checked.groupby('batch').head(1))
    print('모든 사용 가능 곡선의 격자·유한값·차이·분산 검증 통과')

    # 각 배치에서 대표 1개를 확인한다. 전체 차이곡선 비교는 다음 셀에서 한다.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for ax, batch in zip(axes.flat, BATCHES):
        sample = available.loc[available.batch.eq(batch)].iloc[0]
        cid = sample.cell_id
        v = curves[f'{cid}_voltage']
        ax.plot(v, curves[f'{cid}_q10'], label='Cycle 10', color=COLORS[batch], linestyle='--')
        ax.plot(v, curves[f'{cid}_q100'], label='Cycle 100', color=COLORS[batch])
        ax.set(title=f'{batch}: {cid} Q(V)', xlabel='Voltage (V)', ylabel='Discharge capacity (Ah)')
        ax.legend()
        axes[1, 1].plot(v, curves[f'{cid}_deltaq'], label=f'{batch}: {cid}', color=COLORS[batch])
    axes[1, 1].axhline(0, color='grey', linewidth=.7)
    axes[1, 1].set(title='Representative difference curves', xlabel='Voltage (V)',
                   ylabel='Q100(V) − Q10(V) (Ah)')
    axes[1, 1].legend(fontsize=9)
    plt.show()
    """)

    md("""
    같은 차이곡선을 세 배치 전체에서도 비교한다. 이는 과제의 기술적 EDA이며,
    Batch 2를 완전히 보지 않은 평가셋으로 주장하지 않는 이유이기도 하다.
    장수명(>1,000)·단수명(<500) 그룹의 차이곡선 분산 중앙값과 표본 수도 기술적으로 비교한다.
    표본이 적거나 한쪽 그룹이 없으면 배치별 차이를 일반화하기 어렵다.
    모델 선택용 수치 분석은 다음 절에서 개발 구간만 사용한다.
    """)
    code(r"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True, layout='constrained')
    for ax, batch in zip(axes.flat, BATCHES):
        selected = available.loc[available.batch.eq(batch)]
        for row in selected.itertuples():
            cid = row.cell_id
            color = '#703B93' if row.cycle_life < 500 else '#D69519' if row.cycle_life > 1000 else '#9CABB7'
            ax.plot(curves[f'{cid}_voltage'], curves[f'{cid}_deltaq'],
                    color=color, alpha=.5, linewidth=.8)
        for label, color, n in [
            ('Short <500', '#703B93', int(selected.cycle_life.lt(500).sum())),
            ('Middle 500–1000', '#9CABB7', int(selected.cycle_life.between(500, 1000).sum())),
            ('Long >1000', '#D69519', int(selected.cycle_life.gt(1000).sum()))]:
            ax.plot([], [], color=color, label=f'{label} (n={n})')
        ax.legend(fontsize=9)
        ax.axhline(0, color='grey', linewidth=.7)
        ax.set(title=f'{batch}: difference curves (n={len(selected)})',
               xlabel='Voltage (V)', ylabel='Q100(V) − Q10(V) (Ah)')
    axes[1, 1].set_axis_off()
    axes[1, 1].text(.06, .87,
        'Difference = Q100(V) − Q10(V)\n\n'
        'All panels use the same axes.\n'
        'Each line = one eligible cell record.\n\n'
        'Batch 3 is used for EDA only.\n'
        'Feature decisions use Batch 1 development only.',
        transform=axes[1, 1].transAxes, va='top', linespacing=1.8)
    plt.show()

    delta_groups = available[['batch', 'cell_id', 'cycle_life', 'log10_deltaq_var']].copy()
    delta_groups['life_group'] = np.select(
        [delta_groups.cycle_life < 500, delta_groups.cycle_life > 1000],
        ['Short: <500', 'Long: >1000'], default='Middle: 500–1000')
    delta_extremes = delta_groups.loc[delta_groups.life_group.ne('Middle: 500–1000')]
    group_delta_stats = delta_extremes.groupby(['batch', 'life_group']).agg(
        n=('cell_id', 'size'), valid_logvar=('log10_deltaq_var', 'count'),
        median_logvar=('log10_deltaq_var', 'median')).reindex(
            pd.MultiIndex.from_product([list(BATCHES), ['Short: <500', 'Long: >1000']],
                                       names=['batch', 'life_group']))
    group_delta_stats[['n', 'valid_logvar']] = group_delta_stats[['n', 'valid_logvar']].fillna(0).astype(int)
    display(group_delta_stats.round(4))
    print('중앙값 비교는 EDA용입니다. 그룹 표본 수를 함께 보고, 평가셋으로 피처를 고르지 않습니다.')
    """)

    md("""
    ## 6. 충전 정책별 수명과 표본 수

    평균이 높은 정책이 항상 더 좋다는 뜻은 아니다. 정책마다 배터리 수와 실험 조건이 다르고,
    배치 차이가 섞여 있을 수 있다. **평균과 n을 함께** 읽고 인과관계로 단정하지 않는다.
    """)
    code(r"""
    policy_view = eligible.copy()
    policy_view['policy'] = policy_view.policy.fillna('(missing)').replace('', '(missing)')
    policy_stats = policy_view.groupby(['batch', 'policy'], dropna=False).agg(
        n=('cell_id', 'size'), mean_life=('cycle_life', 'mean'),
        median_life=('cycle_life', 'median'), std_life=('cycle_life', 'std')).reset_index()
    display(policy_stats.sort_values(['batch', 'mean_life'], ascending=[True, False]).round(1))
    print('표본 1개뿐인 정책 그룹:', int(policy_stats.n.eq(1).sum()))
    """)

    md("""
    ## 7. 초기 피처와 수명의 관계 — Batch 1 개발 구간만 사용

    피처 후보는 예측 시점인 100 cycle 이내에서 계산한다. `qd_slope_10_100`은 초기 용량의 기울기,
    `ir_change_early`는 초기와 후반 초기구간의 내부저항 차이, `log10_deltaq_var`는 앞에서 계산한 값이다.
    여기서 상관계수와 다중공선성은 **Batch 1의 개발 구간만** 살핀다.
    `Batch1_holdout`, `Batch2_final_test`, `Batch3_EDA_only`에 속한 배터리는 이 분석에 넣지 않는다.

    Pearson은 선형 관계를, Spearman은 순위 기반의 단조 관계를 요약한다.
    두 계수 모두 인과관계나 최종 예측 성능을 확정하지 않는다. 작은 표본, 결측값, 이상치의
    영향을 함께 본다. 피처 사이 상관이 높으면 비슷한 정보를 중복으로 담을 수 있다.
    """)
    code(r"""
    FEATURES = ['qd_cycle2', 'qd_change_100_10',
                'qd_slope_10_100', 'ir_mean_2_100', 'ir_change_early',
                'tavg_mean_2_100', 'chargetime_mean_2_100', 'log10_deltaq_var']
    train_eda = eligible.loc[eligible.split_role.eq('Batch1_CV_development')].copy()
    assert train_eda.batch.eq('Batch 1').all() and len(train_eda) == 28
    display(train_eda[FEATURES].isna().sum().rename('missing_in_development').to_frame())
    relationship_rows = []
    for feature in FEATURES:
        pair = train_eda[[feature, 'cycle_life']].replace([np.inf, -np.inf], np.nan).dropna()
        enough = len(pair) >= 3 and pair[feature].nunique() > 1
        r = pair[feature].corr(pair.cycle_life) if enough else np.nan
        rho = pair[feature].corr(pair.cycle_life, method='spearman') if enough else np.nan
        relationship_rows.append({'feature': feature, 'paired_cells': len(pair),
                                  'pearson_r_with_life': r, 'spearman_rho_with_life': rho})
    relationships = pd.DataFrame(relationship_rows)
    display(relationships.sort_values('pearson_r_with_life', key=lambda s: s.abs(),
                                     ascending=False).round(3))
    """)
    code(r"""
    pair = train_eda[['log10_deltaq_var', 'cycle_life', 'cell_id']].dropna()
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.scatter(pair.log10_deltaq_var, pair.cycle_life, alpha=.8)
    ax.set(xlabel='log10 variance of Q100 − Q10', ylabel='Cycle life',
           title=f'Batch 1 development only (n={len(pair)})')
    plt.tight_layout()
    plt.show()

    corr = train_eda[FEATURES].corr(min_periods=10)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
    ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=90)
    ax.set_yticks(range(len(FEATURES)), FEATURES)
    ax.set_title('Development features (minimum 10 paired cells)')
    fig.colorbar(im, ax=ax, label='Pearson r')
    plt.tight_layout()
    plt.show()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    strong = upper.loc[upper.abs() >= .85].rename('r').reset_index()
    strong.columns = ['feature_1', 'feature_2', 'r']
    display(strong.round(3))
    """)

    md("""
    ## 8. EDA에서 다음 실험으로 연결하기

    | 오늘 확인할 근거 | 다음 단계에서 시도할 내용 |
    |---|---|
    | 수명 분포·독립 배터리 수 | 고정된 Batch 1 개발/holdout 분할과 개발 구간 CV를 사용 |
    | 초기 용량만으로 보이는 관련성 | 용량 피처만 사용하는 작은 기준 모델 |
    | ΔQ(V)의 관련성과 계산 신뢰성 | `log10_deltaq_var` 한 개 피처 모델과 비교 |
    | 온도·저항·충전시간의 추가 정보 | 소수 피처를 추가한 Ridge/ElasticNet 등의 비교 |
    | 높은 피처 간 상관·결측 | 훈련 fold에서만 대치·스케일 조정, 중복 피처 정리 또는 규제 |
    | Batch 1/2/3 분포 차이 | 실험 조건 차이·외삽 가능성을 정리하고 Batch 3은 EDA 역할로 유지 |

    **아직 모델 성능 결과는 없다.** 다음 단계에서는 Batch 1 개발 구간의 CV로 설계를 결정한 후,
    고정한 holdout과 Batch 2에서 MAPE를 평가하고 RMSE/MAE와 과대·과소 예측도 함께 본다.
    원논문의 9.1% 목표를 맞추려고 Batch 2 정답을 보고 피처나 제외 대상을 바꾸지 않는다.
    holdout·Batch 2의 분포를 EDA에서 본 사실은 최종 보고에도 밝힌다.
    Batch 3을 DAY 2 추가 평가에 사용할지는 별도로 결정하며, 오늘의 EDA가 모델 성능 평가를 뜻하지 않는다.

    전체 수명과 `last_cycle`, 기록 개수, 최종 용량, `first_below_80` 등은 품질 검토용이다.
    초기 수명 예측의 입력 피처로 넣지 않는다. 결측 대치와 표준화도 전체 데이터에서 먼저 맞추면 안 된다.
    """)

    md("""
    ## 9. 직접 답해 보는 DAY 1 마무리 질문

    1. **표본 수와 라벨:** cycle 행 수와 배터리 셀 기록 수는 각각 얼마인가?
       분석 제외 대상을 하나 골라 원본 라벨을 그대로 신뢰하기 어려운 이유를 설명해 보자.
       Batch 1·3의 유지된 라벨도 Batch 2와 같은 방식의 실측 정답이라고 부를 수 있는가?
       Batch 3을 추가했을 때 수명 분포와 장·단수명 비율은 어떻게 달라졌는가?
       **내 답:**

    2. **피처 근거:** 초기 방전용량 총량과 ΔQ(V) 피처 중 어떤 관찰이 더 설득력 있었는가?
       그래프나 실제 수치를 하나 인용하고, 상관만으로 결론을 내릴 수 없는 이유도 적어 보자.
       **내 답:**

    3. **검증과 누수:** Batch 2에서 오차가 크다면 무엇부터 점검하겠는가?
       확인할 수 있는 데이터 조건과, 테스트 정답을 보며 바꾸면 안 되는 설계를 구분해 보자.
       Batch 3의 DAY 1 EDA와 선택적인 DAY 2 추가 평가의 차이는 무엇인가?
       **내 답:**

    ### 참고와 재현 범위

    - [Severson 등, Nature Energy 2019 원논문](https://web.mit.edu/braatzgroup/Severson_NatureEnergy_2019.pdf)
    - [원저자 데이터 처리 저장소](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation)
    - 이번 분석의 Batch 2 날짜와 원논문의 Batch 2 날짜가 다르며, 배치 분할도 원논문과 다르다.
      이번 과제는 동일 조건의 논문 성능 재현이라고 표현하지 않는다.
    - 제외 판정과 원본 수명 라벨은 준비 단계의 audit 및 `cells_analysis.csv`와 대조한다.
      이 노트북은 이미 기록된 판정을 보여 주며 데이터를 자동 재라벨링하지 않는다.
    - Batch 1·3의 근사 종료 라벨과 Batch 2의 임계값 교차 라벨 차이를 최종 성능 설명에도 남긴다.
    - 수업 녹음에서는 DAY 1 EDA에 Batch 1·2·3을 사용하고 Extra를 제외한다.
      Batch 3의 모델 추가 성능 평가는 DAY 2 선택 항목이다.
    """)
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "file_extension": ".py",
                                           "mimetype": "text/x-python", "pygments_lexer": "ipython3"}},
            "nbformat": 4, "nbformat_minor": 4}


def main():
    output = ROOT / "notebooks" / "01_DAY1_EDA.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Created {output} ({len(notebook['cells'])} cells; not executed)")


if __name__ == "__main__":
    main()
