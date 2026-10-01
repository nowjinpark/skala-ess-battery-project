"""Independent DAY2 audit. Never fits models, changes selection, or edits inputs.

Recomputes metrics with NumPy rather than importing training helpers. Checks the
persisted split/lock, training-fold preprocessing, early features, and predictions
from the four already-fitted family winners. Writes only validation.json.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import ast
import hashlib
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GroupKFold, KFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/day2'
CHECKS = []
BASE = ['qd_cycle2', 'qd_change_100_10', 'qd_slope_10_100',
        'ir_mean_2_100', 'ir_change_early', 'tavg_mean_2_100',
        'chargetime_mean_2_100']
FEATURES = {
    'basic7': BASE,
    'deltaq8': BASE + ['log10_deltaq_var'],
    'no_qd_change7': [x for x in BASE if x != 'qd_change_100_10'] + ['log10_deltaq_var'],
    'no_qd_slope7': [x for x in BASE if x != 'qd_slope_10_100'] + ['log10_deltaq_var'],
    'deltaq_mean8': BASE + ['deltaq_mean'],
    'deltaq_min8': BASE + ['deltaq_min'],
    'no_ir6': [x for x in BASE if not x.startswith('ir_')] + ['log10_deltaq_var'],
    'deltaq_only1': ['log10_deltaq_var'],
}
METRICS = ['mape_pct', 'mae_cycles', 'rmse_cycles', 'r2']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(name):
    return json.loads((OUT / name).read_text())


def near(actual, expected, description, rtol=1e-9, atol=1e-10):
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol,
                               equal_nan=True, err_msg=description)


def passed(name, **details):
    CHECKS.append({'check': name, 'status': 'passed', **details})


def metrics(actual, predicted):
    """Original-cycle metrics, independently expressed as elementary formulas."""
    y, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    assert len(y) == len(p) and len(y) > 1
    assert np.isfinite(y).all() and np.isfinite(p).all() and (y > 0).all()
    error = p - y
    sst = np.square(y - y.mean()).sum()
    assert sst > 0
    return {'mape_pct': float(np.mean(np.abs(error) / y) * 100),
            'mae_cycles': float(np.mean(np.abs(error))),
            'rmse_cycles': float(np.sqrt(np.mean(np.square(error)))),
            'r2': float(1 - np.square(error).sum() / sst)}


def compare_metrics(row, expected, context):
    for key, value in expected.items():
        near(row[key], value, f'{context}: {key}')


def finite_mean(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else np.nan


def verify_early_features(cells):
    audited = cells[cells.split_role.isin(
        ['Batch1_CV_development', 'Batch1_holdout', 'Batch2_final_test'])]
    n = 0
    for batch in (1, 2):
        summary = pd.read_csv(ROOT / f'data/processed/batch{batch}_summary.csv.gz')
        selected = audited[audited.batch.eq(f'Batch {batch}')]
        with np.load(ROOT / f'data/processed/batch{batch}_curves.npz') as curves:
            for _, cell in selected.iterrows():
                s = summary[summary.cell_id.eq(cell.cell_id)].reset_index(drop=True)
                # Ignore the stored early_window/capacity_plausible flags: derive
                # the input window and fixed physical-quality bounds again.
                e = s[(s.cycle >= 2) & (s.cycle <= 100)]
                assert len(e) and e.cycle.max() <= 100
                q = e[(e.QDischarge > .2) & (e.QDischarge < 1.5)]
                q_at = lambda cycle: float(q.loc[q.cycle.eq(cycle), 'QDischarge'].item())
                xq = q[(q.cycle >= 10) & (q.cycle <= 100)]
                x = xq.cycle.to_numpy(); y = xq.QDischarge.to_numpy()
                assert len(x) >= 70
                slope = np.dot(x - x.mean(), y - y.mean()) / np.square(x - x.mean()).sum()
                ir = e[e.IR > 0]
                dq = curves[cell.cell_id + '_q100'] - curves[cell.cell_id + '_q10']
                near(curves[cell.cell_id + '_deltaq'], dq, 'Q100 - Q10')
                assert np.isfinite(dq).all()
                assert s.loc[int(cell.curve_slot10_zero_based), 'cycle'] == 10
                assert s.loc[int(cell.curve_slot100_zero_based), 'cycle'] == 100
                expected = {
                    'qd_cycle2': q_at(2),
                    'qd_change_100_10': q_at(100) - q_at(10),
                    'qd_slope_10_100': slope,
                    'ir_mean_2_100': finite_mean(ir.IR),
                    'ir_change_early': finite_mean(ir.loc[ir.cycle >= 91, 'IR'])
                                       - finite_mean(ir.loc[ir.cycle <= 10, 'IR']),
                    'tavg_mean_2_100': finite_mean(e.loc[e.Tavg.between(0, 80), 'Tavg']),
                    'chargetime_mean_2_100': finite_mean(e.loc[(e.chargetime > 0) & (e.chargetime < 120), 'chargetime']),
                    'log10_deltaq_var': float(np.log10(np.square(dq - dq.mean()).sum() / (len(dq) - 1))),
                    'deltaq_mean': float(dq.mean()),
                    'deltaq_min': float(dq.min()),
                }
                for feature, value in expected.items():
                    near(cell[feature], value, f'{cell.cell_id} {feature}', atol=1e-12)
                n += 1
    assert n == 75
    passed('All ten candidate features recomputed using cycles <=100 only', cells=n,
           feature_names=sorted(set(sum(FEATURES.values(), []))),
           source='Existing processed summaries and stored Q10/Q100 curves; no full raw MAT reload')


def main():
    plan, lock, evaluation = (read_json(x) for x in
                             ['experiment_plan.json', 'selection_lock.json', 'evaluation.json'])
    cells = pd.read_csv(ROOT / 'data/processed/cells_analysis.csv')
    split = pd.read_csv(ROOT / 'results/split_assignment.csv')
    baseline = pd.read_csv(ROOT / 'references/baseline_batch12_splits.csv')
    assert cells.cell_id.is_unique and split.cell_id.is_unique and baseline.cell_id.is_unique
    roles = {'Batch1_CV_development': 28, 'Batch1_holdout': 8,
             'Batch2_final_test': 39, 'Batch3_EDA_only': 40, 'excluded': 24}
    assert cells.split_role.value_counts().to_dict() == roles
    pd.testing.assert_series_equal(cells.set_index('cell_id').split_role.sort_index(),
                                   split.set_index('cell_id').split_role.sort_index())
    columns = ['batch', 'analysis_eligible', 'exclusion_reason', 'split_role']
    pd.testing.assert_frame_equal(split.set_index('cell_id').loc[baseline.cell_id, columns],
                                  baseline.set_index('cell_id')[columns])
    dev = cells[cells.split_role.eq('Batch1_CV_development')].reset_index(drop=True)
    frames = {'Valid_B1_holdout': cells[cells.split_role.eq('Batch1_holdout')],
              'Test_B2': cells[cells.split_role.eq('Batch2_final_test')]}
    dev_ids = set(dev.cell_id)
    assert dev.policy.notna().all() and dev.policy.nunique() == 18
    assert lock['training_ids'] == dev.cell_id.tolist()
    assert not dev_ids.intersection(frames['Valid_B1_holdout'].cell_id)
    assert not dev_ids.intersection(frames['Test_B2'].cell_id)
    assert not set(frames['Valid_B1_holdout'].cell_id).intersection(frames['Test_B2'].cell_id)
    eligible = cells[cells.analysis_eligible]
    near(eligible.cycle_life, eligible.cycle_life_raw, 'Supplied labels preserved')
    assert dev.label_status.eq('provided_near80_endpoint_proxy').all()
    assert frames['Valid_B1_holdout'].label_status.eq('provided_near80_endpoint_proxy').all()
    assert frames['Test_B2'].label_status.eq('observed_80pct_crossing').all()
    passed('Original DAY1 cell IDs, roles and supplied labels preserved', role_counts=roles,
           development_policy_groups=18, baseline_rows=len(baseline))

    for relative, digest in lock['source_sha256'].items():
        assert sha(ROOT / relative) == digest == plan['source_sha256'][relative]
    assert sha(OUT / 'experiment_plan.json') == lock['plan_sha256']
    assert sha(OUT / 'cv_candidates.csv') == lock['cv_results_sha256']
    assert sha(OUT / 'selection_lock.json') == evaluation['selection_lock_sha256']
    training_source = ROOT / 'src/train_day2.py'
    assert sha(training_source) == lock['training_source_sha256'] == plan['training_source_sha256']
    assert datetime.fromisoformat(plan['created_at']) < datetime.fromisoformat(lock['locked_at'])
    assert datetime.fromisoformat(lock['locked_at']) < datetime.fromisoformat(evaluation['evaluated_at'])
    # Code audit: evaluation has no fit or fit_transform calls. No training helper
    # is imported or invoked by this verifier.
    tree = ast.parse(training_source.read_text())
    evaluate_node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'evaluate')
    assert not [x for x in ast.walk(evaluate_node) if isinstance(x, ast.Call)
                and isinstance(x.func, ast.Attribute) and x.func.attr in ['fit', 'fit_transform']]
    passed('Input, plan, training source, selection lock and timing hashes consistent',
           plan_at=plan['created_at'], locked_at=lock['locked_at'], evaluated_at=evaluation['evaluated_at'])

    assert plan['features'] == lock['feature_sets'] == FEATURES
    specs = {c['id']: c for c in plan['candidates']}
    assert len(specs) == len(plan['candidates']) == 113
    assert set(specs) == {f'C{i:03d}' for i in range(113)}
    assert Counter(c['family'] for c in specs.values()) == {'Dummy': 1, 'Linear': 16, 'Ridge': 64, 'RandomForest': 32}
    assert specs['C000']['family'] == 'Dummy' and specs['C000']['target_transform'] == 'raw'
    for name in FEATURES:
        candidates = [c for c in specs.values() if c['feature_set'] == name and c['family'] != 'Dummy']
        assert len(candidates) == 14
        for transform in ['raw', 'log']:
            assert len([c for c in candidates if c['family'] == 'Linear' and c['target_transform'] == transform]) == 1
            assert {c['params']['alpha'] for c in candidates if c['family'] == 'Ridge' and c['target_transform'] == transform} == {.1, 1., 10., 100.}
        forest = [c for c in candidates if c['family'] == 'RandomForest']
        assert {(c['params']['max_depth'], c['params']['min_samples_leaf']) for c in forest} == {(3, 1), (3, 3), (None, 1), (None, 3)}
        assert all(c['target_transform'] == 'raw' and c['params']['n_estimators'] == 300
                   and c['params']['random_state'] == 42 and c['params']['max_features'] == 1. for c in forest)
    passed('Predeclared bounded candidate grid and early-feature whitelist', candidates=113,
           families={'Dummy': 1, 'Linear': 16, 'Ridge': 64, 'RandomForest': 32}, feature_sets=8)
    verify_early_features(cells)

    cv = pd.read_csv(OUT / 'cv_candidates.csv')
    fold_metrics = pd.read_csv(OUT / 'cv_fold_metrics.csv')
    oof = pd.read_csv(OUT / 'cv_oof_predictions.csv')
    assignments = pd.read_csv(OUT / 'cv_assignments.csv')
    audit = read_json('fold_preprocessing_audit.json')
    assert len(cv) == 113 and cv.candidate_id.is_unique and set(cv.candidate_id) == set(specs)
    assert len(fold_metrics) == len(audit) == 339 and len(oof) == 3164
    assert not fold_metrics.duplicated(['candidate_id', 'fold']).any()
    assert not oof.duplicated(['candidate_id', 'cell_id']).any()
    assert not assignments.duplicated(['fold', 'cell_id']).any() and len(assignments) == 84
    assert set(assignments.role) == {'train', 'validation'} and set(assignments.fold) == {1, 2, 3}
    folds = {}
    fold_details = []
    for fold, (tr, va) in enumerate(GroupKFold(3).split(dev, groups=dev.policy), 1):
        train, valid = dev.iloc[tr], dev.iloc[va]
        for role, frame in [('train', train), ('validation', valid)]:
            observed = assignments[(assignments.fold == fold) & (assignments.role == role)]
            assert observed.cell_id.tolist() == frame.cell_id.tolist()
            assert observed.policy.tolist() == frame.policy.tolist()
        assert not set(train.policy).intersection(valid.policy)
        folds[fold] = (train, valid)
        fold_details.append({'fold': fold, 'train_n': len(train), 'validation_n': len(valid), 'policy_overlap': 0})
    assert set(assignments[assignments.role.eq('validation')].cell_id) == dev_ids
    assert assignments[assignments.role.eq('validation')].cell_id.is_unique
    passed('GroupKFold assignments reproduced; battery IDs and charging policies disjoint', folds=fold_details)

    audit_by_key = {(x['candidate_id'], x['fold']): x for x in audit}
    assert len(audit_by_key) == 339
    fm = fold_metrics.set_index(['candidate_id', 'fold'])
    cv_index = cv.set_index('candidate_id')
    for cid, spec in specs.items():
        fs = FEATURES[spec['feature_set']]
        candidate_oof = oof[oof.candidate_id.eq(cid)]
        assert set(candidate_oof.cell_id) == dev_ids and len(candidate_oof) == 28
        recomputed = []
        for fold, (train, valid) in folds.items():
            rec = audit_by_key[cid, fold]
            assert rec['train_ids'] == train.cell_id.tolist()
            assert rec['validation_ids'] == valid.cell_id.tolist() and rec['feature_names'] == fs
            x = train[fs].to_numpy(dtype=float)
            median = np.nanmedian(x, axis=0)
            near(rec['imputer_statistics'], median, f'{cid} fold {fold} medians')
            filled = np.where(np.isnan(x), median, x)
            if spec['family'] in ['Linear', 'Ridge']:
                near(rec['scaler_mean'], filled.mean(axis=0), f'{cid} fold {fold} scaler means')
            else:
                assert rec['scaler_mean'] is None
            prediction = candidate_oof[candidate_oof.fold.eq(fold)].set_index('cell_id').loc[valid.cell_id]
            assert len(prediction) == len(valid)
            near(prediction.actual, valid.cycle_life, f'{cid} fold {fold} true labels')
            m = metrics(prediction.actual, prediction.predicted)
            compare_metrics(fm.loc[cid, fold], m, f'{cid} fold {fold}')
            assert fm.loc[cid, fold]['train_n'] == len(train) and fm.loc[cid, fold]['valid_n'] == len(valid)
            if spec['family'] == 'Dummy':
                near(prediction.predicted, np.full(len(valid), train.cycle_life.mean()), 'Training-only dummy mean')
            recomputed.append(m)
        row = cv_index.loc[cid]
        assert row['family'] == spec['family'] and row['feature_set'] == spec['feature_set']
        assert row['target_transform'] == spec['target_transform'] and row['n_features'] == len(fs)
        assert json.loads(row['params']) == spec['params']
        for key in METRICS:
            values = [x[key] for x in recomputed]
            near(row[f'cv_{key}_mean'], np.mean(values), f'{cid} unweighted CV {key}')
            near(row[f'cv_{key}_sd'], np.std(values, ddof=1), f'{cid} CV sample SD {key}')
    sorted_cv = cv.sort_values(['cv_mape_pct_mean', 'candidate_id'])
    assert cv.candidate_id.tolist() == sorted_cv.candidate_id.tolist()
    assert sorted_cv.iloc[0].candidate_id == lock['winner_id'] == evaluation['winner_id']
    for family, cid in lock['family_winners'].items():
        assert sorted_cv[sorted_cv.family.eq(family)].iloc[0].candidate_id == cid
        assert specs[cid] == lock['specs'][cid]
    assert set(lock['specs']) == set(lock['family_winners'].values())
    passed('All 339 fold preprocessors use training cells only; all OOF metrics and CV selection recomputed',
           candidates=113, fold_records=339, out_of_fold_predictions=3164,
           ranking='Unweighted mean of three MAPE percentages; sample SD (ddof=1)',
           winner=lock['winner_id'], family_winners=lock['family_winners'])

    random_assign = pd.read_csv(OUT / 'random_cv_assignments.csv')
    random_scores = pd.read_csv(OUT / 'random_cv_sensitivity.csv')
    assert len(random_assign) == 84 and len(random_scores) == 3
    assert not random_assign.duplicated(['fold', 'cell_id']).any()
    overlap = []
    for fold, (tr, va) in enumerate(KFold(3, shuffle=True, random_state=42).split(dev), 1):
        train, valid = dev.iloc[tr], dev.iloc[va]
        for role, frame in [('train', train), ('validation', valid)]:
            actual_ids = random_assign[(random_assign.fold == fold) & (random_assign.role == role)].cell_id.tolist()
            assert actual_ids == frame.cell_id.tolist()
        n = len(set(train.policy).intersection(valid.policy)); overlap.append(n)
        assert random_scores.loc[random_scores.fold.eq(fold), 'policy_overlap_count'].item() == n
    near(evaluation['cv_random_sensitivity']['mean_mape_pct'], random_scores.mape_pct.mean(), 'Random CV mean')
    near(evaluation['cv_random_sensitivity']['sd_mape_pct'], random_scores.mape_pct.std(ddof=1), 'Random CV SD')
    assert evaluation['cv_random_sensitivity']['overlapping_policy_counts'] == overlap
    assert evaluation['cv_random_sensitivity']['selection_used'] is False
    passed('Random CV diagnostic split and aggregation consistent; no reselection', overlap_counts=overlap,
           scope='Individual random-CV predictions were not persisted; fold scores cannot be independently recalculated without refitting')

    final_metrics = pd.read_csv(OUT / 'final_metrics.csv')
    predictions = pd.read_csv(OUT / 'final_predictions.csv')
    final_audit = {x['candidate_id']: x for x in read_json('final_preprocessing_audit.json')}
    assert len(final_metrics) == 8 and len(predictions) == 188 and len(final_audit) == 4
    assert not final_metrics.duplicated(['candidate_id', 'dataset']).any()
    assert not predictions.duplicated(['candidate_id', 'dataset', 'cell_id']).any()
    assert set(predictions.candidate_id) == set(final_metrics.candidate_id) == set(lock['specs'])
    assert set(predictions.dataset) == set(final_metrics.dataset) == set(frames)
    assert not predictions.cell_id.str.startswith('b3').any()
    observed_model_paths = {p.stem for p in (OUT / 'models').glob('*.joblib')}
    assert observed_model_paths == set(lock['model_sha256']) == set(lock['specs'])
    models = {}; independent_final = []
    for cid, spec in lock['specs'].items():
        fs = FEATURES[spec['feature_set']]
        path = OUT / 'models' / f'{cid}.joblib'
        assert sha(path) == lock['model_sha256'][cid]
        model = joblib.load(path); models[cid] = model
        assert model.feature_names_in_.tolist() == fs
        assert list(model.named_steps) == (['imputer', 'scaler', 'model'] if spec['family'] in ['Linear', 'Ridge'] else ['imputer', 'model'])
        imputer = model.named_steps['imputer']; rec = final_audit[cid]
        assert imputer.strategy == 'median'
        median = np.nanmedian(dev[fs].to_numpy(dtype=float), axis=0)
        near(imputer.statistics_, median, f'{cid} final training medians')
        assert rec['training_ids'] == dev.cell_id.tolist() and rec['features'] == fs
        near(rec['imputer_statistics'], median, f'{cid} audit final medians')
        filled = dev[fs].fillna(dict(zip(fs, median))).to_numpy(dtype=float)
        if 'scaler' in model.named_steps:
            scaler = model.named_steps['scaler']
            assert scaler.with_mean and scaler.with_std
            assert scaler.n_samples_seen_ == 28
            near(scaler.mean_, filled.mean(axis=0), f'{cid} final scaler means')
            near(scaler.var_, filled.var(axis=0), f'{cid} final scaler population variance')
            near(scaler.scale_, np.where(filled.std(axis=0) == 0, 1., filled.std(axis=0)), f'{cid} final scaler scales')
            near(rec['scaler_mean'], filled.mean(axis=0), f'{cid} final audit scaler means')
        else:
            assert rec['scaler_mean'] is None
        est = model.named_steps['model']
        if spec['target_transform'] == 'log':
            assert isinstance(est, TransformedTargetRegressor)
            near(est.func(np.array([1., np.e, 100.])), np.log([1., np.e, 100.]), 'Natural-log target transform')
            near(est.inverse_func(np.array([0., 1., 2.])), np.exp([0., 1., 2.]), 'Exponential inverse transform')
            est = est.regressor_
        else:
            assert not isinstance(est, TransformedTargetRegressor)
        expected_class = {'Dummy': 'DummyRegressor', 'Linear': 'LinearRegression', 'Ridge': 'Ridge', 'RandomForest': 'RandomForestRegressor'}[spec['family']]
        assert type(est).__name__ == expected_class
        for key, value in spec['params'].items():
            assert est.get_params()[key] == value
        if spec['family'] == 'Dummy':
            near(est.constant_.ravel(), [dev.cycle_life.mean()], 'Final dummy uses dev labels only')
        if spec['family'] in ['Linear', 'Ridge']:
            z = (filled - model.named_steps['scaler'].mean_) / model.named_steps['scaler'].scale_
            y = np.log(dev.cycle_life.to_numpy()) if spec['target_transform'] == 'log' else dev.cycle_life.to_numpy()
            zc, yc = z - z.mean(axis=0), y - y.mean()
            if spec['family'] == 'Ridge':
                coefficient = np.linalg.solve(zc.T @ zc + spec['params']['alpha'] * np.eye(len(fs)), zc.T @ yc)
            else:
                coefficient = np.linalg.lstsq(zc, yc, rcond=None)[0]
            intercept = y.mean() - z.mean(axis=0) @ coefficient
            near(est.coef_, coefficient, f'{cid} coefficients from dev labels/declared objective')
            near(est.intercept_, intercept, f'{cid} intercept from dev labels/declared objective')
        for name, frame in frames.items():
            saved = predictions[(predictions.candidate_id == cid) & (predictions.dataset == name)].set_index('cell_id')
            assert set(saved.index) == set(frame.cell_id) and len(saved) == len(frame)
            saved = saved.loc[frame.cell_id]
            near(saved.actual, frame.cycle_life, f'{cid} {name} true labels')
            near(saved.predicted, model.predict(frame[fs]), f'{cid} {name} saved-model predictions')
            error = saved.predicted.to_numpy() - frame.cycle_life.to_numpy()
            near(saved.error_cycles, error, 'Signed errors')
            near(saved.absolute_error_cycles, abs(error), 'Absolute errors')
            near(saved.ape_pct, abs(error) / frame.cycle_life.to_numpy() * 100, 'Percentage errors')
            assert saved.family.eq(spec['family']).all()
            assert saved.policy.tolist() == frame.policy.tolist()
            assert saved.label_status.tolist() == frame.label_status.tolist()
            assert saved.ir_missing.tolist() == frame[['ir_mean_2_100', 'ir_change_early']].isna().any(axis=1).tolist()
            assert saved.policy_seen_in_dev.tolist() == frame.policy.isin(dev.policy).tolist()
            m = metrics(saved.actual, saved.predicted)
            row = final_metrics[(final_metrics.candidate_id == cid) & (final_metrics.dataset == name)].iloc[0]
            assert row['n'] == len(frame) and row['family'] == spec['family'] and row['feature_set'] == spec['feature_set']
            assert row['target_transform'] == spec['target_transform'] and bool(row['selected']) == (cid == lock['winner_id'])
            compare_metrics(row, m, f'{cid} {name}')
            independent_final.append({'candidate_id': cid, 'dataset': name, 'n': len(frame), **m})
    passed('Four frozen models preserve hashes, preprocessing and candidate specifications',
           model_sha256=lock['model_sha256'], final_training_n=28,
           extra_check='Saved Linear/Ridge coefficients independently matched to closed-form dev-only objective')
    passed('All 188 final predictions and original-cycle metrics reproduced without fitting',
           predictions=188, metric_rows=8, no_batch3_predictions=True)

    winner = lock['winner_id']
    selected = predictions[predictions.candidate_id.eq(winner)]
    for name, section in [('Valid_B1_holdout', 'valid'), ('Test_B2', 'test')]:
        group = selected[selected.dataset.eq(name)]
        compare_metrics(evaluation[section], metrics(group.actual, group.predicted), f'Evaluation {section}')
    compare_metrics({k: evaluation['cv']['cv_' + k + '_mean'] for k in METRICS},
                    {k: cv_index.loc[winner]['cv_' + k + '_mean'] for k in METRICS}, 'Evaluation CV')
    assert evaluation['winner_spec'] == specs[winner] and evaluation['features'] == FEATURES[specs[winner]['feature_set']]
    assert (evaluation['training_n'], evaluation['valid_n'], evaluation['test_n']) == (28, 8, 39)
    gap = evaluation['gaps_percentage_points']
    near(gap['valid_minus_cv'], evaluation['valid']['mape_pct'] - evaluation['cv']['cv_mape_pct_mean'], 'Valid minus CV')
    near(gap['test_minus_valid'], evaluation['test']['mape_pct'] - evaluation['valid']['mape_pct'], 'Test minus valid')
    near(gap['test_minus_paper_9_1'], evaluation['test']['mape_pct'] - 9.1, 'Test minus paper reference')
    assert evaluation['paper_reference_mape_pct'] == 9.1
    assert evaluation['batch3_model_evaluation'] is False and evaluation['post_test_tuning'] is False
    assert evaluation['label_status_counts'] == {
        'development': dev.label_status.value_counts().to_dict(),
        'holdout': frames['Valid_B1_holdout'].label_status.value_counts().to_dict(),
        'test': frames['Test_B2'].label_status.value_counts().to_dict()}
    near(evaluation['training_life_range'], [dev.cycle_life.min(), dev.cycle_life.max()], 'Training target range')
    assert evaluation['test_below_training_target_min_n'] == int((frames['Test_B2'].cycle_life < dev.cycle_life.min()).sum())
    assert evaluation['test_above_training_target_max_n'] == int((frames['Test_B2'].cycle_life > dev.cycle_life.max()).sum())
    assert evaluation['holdout_shared_protocol_cell_n'] == int(frames['Valid_B1_holdout'].policy.isin(dev.policy).sum())
    assert evaluation['negative_prediction_n'] == int((selected.predicted < 0).sum())
    largest_errors = selected[selected.dataset.eq('Test_B2')].nlargest(5, 'ape_pct').reset_index(drop=True)
    pd.testing.assert_frame_equal(pd.DataFrame(evaluation['largest_test_errors']), largest_errors,
                                  check_exact=False, rtol=1e-9, atol=1e-10)

    subgroup = pd.read_csv(OUT / 'subgroup_errors.csv')
    json_subgroup = pd.DataFrame(evaluation['subgroups'])
    pd.testing.assert_frame_equal(subgroup, json_subgroup, check_exact=False, rtol=1e-9, atol=1e-10)
    expected_groups = set()
    for name, g in selected.groupby('dataset'):
        masks = {'all': np.ones(len(g), dtype=bool), 'life_lt500': g.actual < 500,
                 'life_500_to1000': (g.actual >= 500) & (g.actual <= 1000), 'life_gt1000': g.actual > 1000,
                 'IR_missing': g.ir_missing, 'IR_present': ~g.ir_missing,
                 'known_policy': g.policy_seen_in_dev, 'unseen_policy': ~g.policy_seen_in_dev}
        for label, mask in masks.items():
            sub = g.loc[mask]
            if not len(sub):
                continue
            expected_groups.add((name, label))
            row = subgroup[(subgroup.dataset == name) & (subgroup.group == label)].iloc[0]
            assert row['n'] == len(sub)
            near(row['mape_pct'], np.mean(np.abs(sub.predicted - sub.actual) / sub.actual) * 100, 'Subgroup MAPE')
            near(row['mae_cycles'], np.mean(np.abs(sub.predicted - sub.actual)), 'Subgroup MAE')
            near(row['bias_cycles'], np.mean(sub.predicted - sub.actual), 'Subgroup bias')
    assert len(subgroup) == len(expected_groups) and set(zip(subgroup.dataset, subgroup.group)) == expected_groups
    shifts = read_json('feature_range_shift.json')
    assert len(shifts) == 2 * len(evaluation['features'])
    assert len({(r['feature'], r['dataset']) for r in shifts}) == len(shifts)
    for r in shifts:
        f, frame = r['feature'], frames[r['dataset']]
        lo, hi = dev[f].min(), dev[f].max()
        outside = frame[f].notna() & ((frame[f] < lo) | (frame[f] > hi))
        near([r['train_min'], r['train_max']], [lo, hi], 'Feature training range')
        assert r['missing_n'] == int(frame[f].isna().sum())
        assert r['outside_ids'] == frame.loc[outside, 'cell_id'].tolist()
        assert r['outside_range_n'] == int(outside.sum()) and r['total_n'] == len(frame)
    interpretation = evaluation['interpretation']
    model_est = models[winner].named_steps['model']
    if isinstance(model_est, TransformedTargetRegressor):
        model_est = model_est.regressor_
    if hasattr(model_est, 'coef_'):
        expected_kind = 'standardized_feature_coefficients_in_log_cycles' if specs[winner]['target_transform'] == 'log' else 'standardized_feature_coefficients_in_cycles'
        assert interpretation['kind'] == expected_kind
        near(interpretation['intercept'], model_est.intercept_, 'Reported intercept')
        assert list(interpretation['values']) == evaluation['features']
        near(list(interpretation['values'].values()), model_est.coef_, 'Reported coefficients')
    passed('Evaluation summary, gaps, subgroup errors, label counts, range shifts and interpretation consistent',
           subgroup_rows=len(subgroup), feature_range_rows=len(shifts))

    inputs = ['experiment_plan.json', 'selection_lock.json', 'evaluation.json', 'cv_candidates.csv',
              'cv_assignments.csv', 'cv_fold_metrics.csv', 'cv_oof_predictions.csv',
              'fold_preprocessing_audit.json', 'final_metrics.csv', 'final_predictions.csv',
              'final_preprocessing_audit.json', 'random_cv_sensitivity.csv', 'random_cv_assignments.csv',
              'subgroup_errors.csv', 'feature_range_shift.json']
    return {'status': 'passed', 'passed': True, 'validated_at': datetime.now(timezone.utc).isoformat(),
            'checks': CHECKS, 'check_count': len(CHECKS), 'winner_id': winner,
            'independent_final_metrics': independent_final,
            'source_sha256': lock['source_sha256'],
            'artifact_sha256': {name: sha(OUT / name) for name in inputs},
            'model_sha256': lock['model_sha256'], 'verifier_sha256': sha(__file__),
            'no_models_fitted_or_reselected': True,
            'audit_limits': [
                'This checks saved artifacts and reviewed code; it cannot prove that no unrecorded earlier experiments occurred.',
                'Group-CV metrics were recomputed from stored OOF predictions, not from a second training run. Fold scaler variance was not persisted, so fold audit checks training IDs, imputer statistics and scaler means.',
                'Random-CV diagnostic predictions were not saved; only its assignments, policy overlap and score aggregation are verified.',
                'Processed early features are independently recomputed from existing summary/curve files; this audit does not repeat the raw MATLAB extraction.',
                'Cell-ID disjointness does not establish physical-cell identity; opaque barcode/channel fields remain unresolved.',
                'Correct calculations do not remove small-sample selection optimism, observed DAY1 test EDA, or batch/label-definition differences.'
            ]}


if __name__ == '__main__':
    try:
        result = main()
    except Exception as exc:
        result = {'status': 'failed', 'passed': False, 'validated_at': datetime.now(timezone.utc).isoformat(),
                  'checks_completed': CHECKS, 'error': f'{type(exc).__name__}: {exc}',
                  'verifier_sha256': sha(__file__), 'no_models_fitted_or_reselected': True}
        (OUT / 'validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        raise
    (OUT / 'validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'checks': result['check_count'],
                      'winner_id': result['winner_id'], 'output': str(OUT / 'validation.json')}, ensure_ascii=False))
