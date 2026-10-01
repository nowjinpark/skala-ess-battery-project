"""Additional descriptive checks for DAY 1 rubric coverage; no prediction model."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['qd_cycle2', 'qd_change_100_10', 'qd_slope_10_100', 'ir_mean_2_100',
            'ir_change_early', 'tavg_mean_2_100', 'chargetime_mean_2_100', 'log10_deltaq_var']


def knee_fit(x, y, life, lower=.2, upper=.9):
    """Continuous two-line visual approximation, expressly not a validated detector."""
    candidates = []
    for k in range(int(np.ceil(lower*life)), int(np.floor(upper*life))+1):
        design = np.column_stack([np.ones(len(x)), x, np.maximum(0, x-k)])
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        if beta[2] >= 0 or beta[1]+beta[2] >= 0:
            continue
        sse = float(np.square(y-design@beta).sum())
        candidates.append({'cycle': k, 'sse': sse, 'beta': beta.tolist(),
                           'before_slope': float(beta[1]), 'after_slope': float(beta[1]+beta[2])})
    if not candidates:
        raise ValueError('No declining two-line approximation')
    result = min(candidates, key=lambda v: v['sse'])
    result['at_search_boundary'] = result['cycle'] in [candidates[0]['cycle'], candidates[-1]['cycle']]
    assert not result['at_search_boundary'], 'Expand/review a boundary-limited result'
    return result


def main():
    p = ROOT/'data/processed'
    cells = pd.read_csv(p/'cells_analysis.csv')
    e = cells.loc[cells.analysis_eligible].copy()
    dev = e.loc[e.split_role.eq('Batch1_CV_development')]
    assert len(dev)==28
    summary = pd.concat([pd.read_csv(p/f'batch{i}_summary.csv.gz') for i in (1,2,3)], ignore_index=True)
    correlations, outliers, knees = [], [], []
    for batch, g in e.groupby('batch'):
        for feature in FEATURES:
            pair = g[[feature, 'cycle_life']].dropna()
            correlations.append({'batch': batch, 'feature': feature, 'n': len(pair),
                                 'pearson': float(pair[feature].corr(pair.cycle_life)),
                                 'spearman': float(pair[feature].corr(pair.cycle_life, method='spearman'))})
        q1, q3 = g.cycle_life.quantile([.25,.75])
        lower, upper = q1-1.5*(q3-q1), q3+1.5*(q3-q1)
        outliers.append({'batch': batch, 'q1': float(q1), 'q3': float(q3),
                         'lower_fence': float(lower), 'upper_fence': float(upper),
                         'short_outlier_ids': g.loc[g.cycle_life.lt(lower),'cell_id'].tolist(),
                         'upper_outlier_ids': g.loc[g.cycle_life.gt(upper),'cell_id'].tolist()})
        representative = g.assign(distance=(g.cycle_life-g.cycle_life.median()).abs()).sort_values(['distance','cell_id']).iloc[0]
        s = summary.loc[summary.cell_id.eq(representative.cell_id) & summary.capacity_plausible
                        & summary.cycle.between(10, representative.cycle_life)].sort_values('cycle')
        x = s.cycle.to_numpy()
        y = s.QDischarge.rolling(7, center=True, min_periods=1).median().to_numpy()
        result = knee_fit(x, y, representative.cycle_life)
        sensitivity = {f'{a:.1f}-{b:.2f}': knee_fit(x,y,representative.cycle_life,a,b)['cycle']
                       for a,b in [(.1,.9),(.2,.95)]}
        knees.append({'batch': batch, 'cell_id': representative.cell_id,
                      'cycle_life': float(representative.cycle_life), **result,
                      'search_range_sensitivity': sensitivity})
    b2 = e.loc[e.batch.eq('Batch 2')].copy()
    b2['newstructure_label'] = b2.policy.str.contains('newstructure', case=False, na=False)
    b2['base_policy'] = b2.policy.str.replace('-newstructure','',regex=False)
    label_groups = b2.groupby('newstructure_label').cycle_life.agg(['size','mean','median','min','max']).reset_index()
    label_groups.columns = ['newstructure_label','n','mean','median','min','max']
    paired = b2.groupby(['base_policy','newstructure_label']).cycle_life.agg(['size','mean','median']).reset_index()
    paired.columns = ['base_policy','newstructure_label','n','mean','median']
    dq_features = ['deltaq_mean','deltaq_min','log10_deltaq_var']
    dq_corr = {f: float(dev[f].corr(dev.cycle_life)) for f in dq_features}
    dq_pairs = dev[dq_features].corr()
    records = []
    groups = e.assign(life_group=np.select([e.cycle_life.lt(500),e.cycle_life.gt(1000)],['short','long'],default='middle'))
    for (batch, group), g in groups.groupby(['batch','life_group']):
        records.append({'batch':batch, 'life_group':group, 'n':len(g),
                        **{f:float(g[f].median()) for f in dq_features}})
    output = {
        'scope':'Descriptive EDA only; Batch2/3 correlations do not select or tune the predictive model',
        'batch_correlations':correlations, 'life_iqr_audit':outliers,
        'representative_knee_method':'Closest to batch median lifetime, ties by cell ID. Observed cycles 10..supplied lifetime. Rolling median7; Q=a+b*cycle+c*max(0,cycle-k). Search20..90% lifetime, choose lowest SSE with steeper negative post-slope. Exploratory curve summary, not validated detection and not a predictor.',
        'representative_knees':knees,
        'b2_newstructure_groups':label_groups.to_dict('records'),
        'b2_same_protocol_groups':paired.to_dict('records'),
        'deltaq_group_medians':records, 'deltaq_development_correlations':dq_corr,
        'deltaq_feature_correlations':dq_pairs.to_dict(),
        'nonnegative_early_slope_counts':e.assign(nonnegative=e.qd_slope_10_100.ge(0)).groupby('batch').nonnegative.sum().astype(int).to_dict(),
        'limits':['newstructure is an original text label with unverified physical meaning, not an added model feature.',
                  'An IQR flag is descriptive and does not justify deleting a valid observation.',
                  'Knee approximation uses future observations for EDA only; it is not an early-life predictive model.'],
    }
    (ROOT/'results/report_review.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps({'iqr':outliers,'knees':knees,'deltaq_dev':dq_corr},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
