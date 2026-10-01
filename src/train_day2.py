"""DAY 2: predeclared grouped CV selection, then locked-model final evaluation.
Run --phase select before --phase evaluate. No test-driven candidate changes.
"""
from pathlib import Path
import argparse, hashlib, json, platform, sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib, sklearn
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/day2'
BASE=['qd_cycle2','qd_change_100_10','qd_slope_10_100','ir_mean_2_100','ir_change_early','tavg_mean_2_100','chargetime_mean_2_100']
FEATURE_SETS={
 'basic7':BASE,
 'deltaq8':BASE+['log10_deltaq_var'],
 'no_qd_change7':[x for x in BASE if x!='qd_change_100_10']+['log10_deltaq_var'],
 'no_qd_slope7':[x for x in BASE if x!='qd_slope_10_100']+['log10_deltaq_var'],
 'deltaq_mean8':BASE+['deltaq_mean'],
 'deltaq_min8':BASE+['deltaq_min'],
 'no_ir6':[x for x in BASE if x not in ['ir_mean_2_100','ir_change_early']]+['log10_deltaq_var'],
 'deltaq_only1':['log10_deltaq_var'],
}
METRICS=['mape_pct','mae_cycles','rmse_cycles','r2']

def stamp(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def native(x):
 if isinstance(x,dict): return {str(k):native(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)): return [native(v) for v in x]
 if isinstance(x,np.generic): return native(x.item())
 if isinstance(x,float) and not np.isfinite(x): return None
 return x

def dump(path,obj): Path(path).write_text(json.dumps(native(obj),ensure_ascii=False,indent=2)+'\n')
def metric(y,p):
 assert np.isfinite(p).all()
 return dict(zip(METRICS,[mean_absolute_percentage_error(y,p)*100,mean_absolute_error(y,p),np.sqrt(mean_squared_error(y,p)),r2_score(y,p)]))

def candidates():
 c=[{'id':'C000','family':'Dummy','feature_set':'basic7','params':{},'target_transform':'raw'}]
 for name in FEATURE_SETS:
  for target in ['raw','log']:
   c.append({'family':'Linear','feature_set':name,'params':{},'target_transform':target})
   for a in [.1,1.,10.,100.]: c.append({'family':'Ridge','feature_set':name,'params':{'alpha':a},'target_transform':target})
  for depth in [3,None]:
   for leaf in [1,3]:
    c.append({'family':'RandomForest','feature_set':name,'params':{'n_estimators':300,'max_depth':depth,'min_samples_leaf':leaf,'max_features':1.,'random_state':42,'n_jobs':1},'target_transform':'raw'})
 for i,x in enumerate(c): x['id']=f'C{i:03d}'
 return c

def pipeline(c):
 if c['family']=='Dummy': estimator=DummyRegressor(strategy='mean')
 elif c['family']=='Linear': estimator=LinearRegression()
 elif c['family']=='Ridge': estimator=Ridge(**c['params'])
 else: estimator=RandomForestRegressor(**c['params'])
 if c['target_transform']=='log': estimator=TransformedTargetRegressor(regressor=estimator,func=np.log,inverse_func=np.exp)
 steps=[('imputer',SimpleImputer(strategy='median'))]
 if c['family'] in ['Linear','Ridge']: steps.append(('scaler',StandardScaler()))
 return Pipeline(steps+[('model',estimator)])

def load_data():
 d=pd.read_csv(ROOT/'data/processed/cells_analysis.csv')
 s=pd.read_csv(ROOT/'results/split_assignment.csv')
 assert d.cell_id.is_unique and s.cell_id.is_unique
 assert d.set_index('cell_id').split_role.equals(s.set_index('cell_id').split_role)
 assert d.groupby('split_role').size().to_dict()=={'Batch1_CV_development':28,'Batch1_holdout':8,'Batch2_final_test':39,'Batch3_EDA_only':40,'excluded':24}
 return d

def sources(): return {str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'data/processed/cells_analysis.csv', ROOT/'results/split_assignment.csv']}

def check_sources(lock): assert lock['source_sha256']==sources(), 'Inputs differ from selection lock'

def select():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'selection_lock.json').exists(): raise RuntimeError('Selection is already locked. Inspect results; do not retune after test evaluation.')
 d=load_data(); dev=d[d.split_role.eq('Batch1_CV_development')].reset_index(drop=True)
 folds=list(GroupKFold(n_splits=3).split(dev,groups=dev.policy))
 specs=candidates()
 plan={'created_at':stamp(),'source_sha256':sources(),'scope':'DAY2 regression; B1 development 28 / fixed B1 holdout 8 / B2 test 39. B3 excluded from fitting and evaluation.',
 'target':'provided cycle_life. Linear/Ridge raw vs natural-log target (exp inverse) are predeclared. Metrics always in original cycles; no clipping, bias correction or post-hoc calibration',
 'features':FEATURE_SETS,'candidates':specs,'selection':'Minimum unweighted mean of 3 validation-fold MAPE percentages; exact ties resolved by candidate order. Never use holdout/B2 scores to choose.',
 'primary_cv':'GroupKFold(3), exact charging policy groups, 18 unique policies; no shared policy across training/validation within a fold.',
 'secondary_cv':'Locked winner only, shuffled KFold(3, random_state=42), sensitivity diagnostic; not a selection criterion.',
 'preprocessing':'Fit median imputation and (Linear/Ridge only) StandardScaler inside each training fold. Fit final pipelines only on 28 development cells.',
 'planned_reporting':'All four family winners evaluated once on B1 holdout and B2 after selection is locked. CV-only feature ablations, errors, missing-IR and lifetime subgroups, and train-range shift are descriptive.',
 'known_limits':['All-batch EDA was seen in DAY1; test is not completely unseen.', 'B1 endpoint proxy vs B2 observed crossing labels.', 'Fixed holdout is cell-disjoint but 5 of its 8 cells share a protocol with development.', 'Grouped development CV used to select candidates is optimistic as a selected development score, not nested unbiased performance.', 'Paper 9.1% is not a like-for-like reproduction.'],
 'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':sklearn.__version__},'training_source_sha256':sha(__file__)}
 dump(OUT/'experiment_plan.json',plan)
 assignments=[]
 for k,(a,b) in enumerate(folds,1):
  assert not set(dev.iloc[a].cell_id)&set(dev.iloc[b].cell_id)
  assert not set(dev.iloc[a].policy)&set(dev.iloc[b].policy)
  assignments.extend({'fold':k,'role':role,'cell_id':dev.iloc[j].cell_id,'policy':dev.iloc[j].policy} for idx,role in [(a,'train'),(b,'validation')] for j in idx)
 pd.DataFrame(assignments).to_csv(OUT/'cv_assignments.csv',index=False)
 results=[]; details=[]; audit=[]; oof_rows=[]
 for i,c in enumerate(specs):
  fs=FEATURE_SETS[c['feature_set']]; vals=[]
  for k,(a,b) in enumerate(folds,1):
   tr,va=dev.iloc[a],dev.iloc[b]
   model=pipeline(c).fit(tr[fs],tr.cycle_life)
   pred=model.predict(va[fs]); m=metric(va.cycle_life,pred); vals.append(m)
   train_mape=metric(tr.cycle_life,model.predict(tr[fs]))['mape_pct']
   details.append({'candidate_id':c['id'],'family':c['family'],'feature_set':c['feature_set'],'fold':k,'train_n':len(a),'valid_n':len(b),'train_fit_mape_pct':train_mape,**m})
   audit.append({'candidate_id':c['id'],'fold':k,'train_ids':tr.cell_id.tolist(),'validation_ids':va.cell_id.tolist(),'feature_names':fs,'imputer_statistics':model.named_steps['imputer'].statistics_.tolist(),'scaler_mean':model.named_steps['scaler'].mean_.tolist() if 'scaler' in model.named_steps else None})
   oof_rows.extend({'candidate_id':c['id'],'fold':k,'cell_id':cell,'actual':y,'predicted':p} for cell,y,p in zip(va.cell_id,va.cycle_life,pred))
  row={'candidate_id':c['id'],'family':c['family'],'feature_set':c['feature_set'],'n_features':len(fs),'target_transform':c['target_transform'],'params':json.dumps(c['params'],sort_keys=True)}
  for key in METRICS: row[f'cv_{key}_mean']=float(np.mean([v[key] for v in vals])); row[f'cv_{key}_sd']=float(np.std([v[key] for v in vals],ddof=1))
  results.append(row)
  if i%13==0: print(f'Grouped CV {i+1}/{len(specs)} complete',flush=True)
 leaderboard=pd.DataFrame(results).sort_values(['cv_mape_pct_mean','candidate_id']).reset_index(drop=True)
 leaderboard.to_csv(OUT/'cv_candidates.csv',index=False); pd.DataFrame(details).to_csv(OUT/'cv_fold_metrics.csv',index=False); pd.DataFrame(oof_rows).to_csv(OUT/'cv_oof_predictions.csv',index=False)
 dump(OUT/'fold_preprocessing_audit.json',audit)
 best=leaderboard.iloc[0].candidate_id
 winners={family:leaderboard[leaderboard.family.eq(family)].iloc[0].candidate_id for family in ['Dummy','Linear','Ridge','RandomForest']}
 selected={c['id']:c for c in specs if c['id'] in winners.values()}
 models=OUT/'models'; models.mkdir(exist_ok=True)
 for cid,c in selected.items(): joblib.dump(pipeline(c).fit(dev[FEATURE_SETS[c['feature_set']]],dev.cycle_life),models/f'{cid}.joblib')
 c=next(c for c in specs if c['id']==best); sensitivity=[]; random_assign=[]
 for k,(a,b) in enumerate(KFold(3,shuffle=True,random_state=42).split(dev),1):
  tr,va=dev.iloc[a],dev.iloc[b]; model=pipeline(c).fit(tr[FEATURE_SETS[c['feature_set']]],tr.cycle_life)
  sensitivity.append({'fold':k,'policy_overlap_count':len(set(tr.policy)&set(va.policy)),**metric(va.cycle_life,model.predict(va[FEATURE_SETS[c['feature_set']]]))})
  random_assign.extend({'fold':k,'role':role,'cell_id':dev.iloc[j].cell_id} for idx,role in [(a,'train'),(b,'validation')] for j in idx)
 pd.DataFrame(sensitivity).to_csv(OUT/'random_cv_sensitivity.csv',index=False);pd.DataFrame(random_assign).to_csv(OUT/'random_cv_assignments.csv',index=False)
 lock={'locked_at':stamp(),'source_sha256':sources(),'plan_sha256':sha(OUT/'experiment_plan.json'),'cv_results_sha256':sha(OUT/'cv_candidates.csv'),'winner_id':best,'family_winners':winners,'specs':selected,'feature_sets':FEATURE_SETS,'model_sha256':{cid:sha(models/f'{cid}.joblib') for cid in selected},'training_ids':dev.cell_id.tolist(),'training_source_sha256':sha(__file__),'selection_basis':'GroupKFold validation mean MAPE only, before any DAY2 holdout/B2 predictions'}
 dump(OUT/'selection_lock.json',lock)
 print('SELECTION LOCKED:',json.dumps({'winner':best,'spec':selected[best],'cv':leaderboard.iloc[0].to_dict()},ensure_ascii=False),flush=True)

def evaluate():
 if (OUT/'evaluation.json').exists(): raise RuntimeError('Final evaluation already recorded. Verify/review existing predictions rather than retuning.')
 lock=json.loads((OUT/'selection_lock.json').read_text());check_sources(lock)
 assert lock['plan_sha256']==sha(OUT/'experiment_plan.json') and lock['cv_results_sha256']==sha(OUT/'cv_candidates.csv')
 d=load_data(); dev=d[d.split_role.eq('Batch1_CV_development')].copy(); h=d[d.split_role.eq('Batch1_holdout')].copy(); t=d[d.split_role.eq('Batch2_final_test')].copy()
 ids=[set(v.cell_id) for v in [dev,h,t]]; assert not(ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2])
 metrics=[];preds=[];pre=[]
 cv=pd.read_csv(OUT/'cv_candidates.csv').set_index('candidate_id')
 for cid,c in lock['specs'].items():
  path=OUT/'models'/f'{cid}.joblib'; assert sha(path)==lock['model_sha256'][cid]
  model=joblib.load(path);fs=FEATURE_SETS[c['feature_set']]
  assert np.allclose(model.named_steps['imputer'].statistics_,dev[fs].median().values)
  pre.append({'candidate_id':cid,'features':fs,'training_ids':dev.cell_id.tolist(),'imputer_statistics':model.named_steps['imputer'].statistics_.tolist(),'scaler_mean':model.named_steps['scaler'].mean_.tolist() if 'scaler' in model.named_steps else None})
  for role,frame in [('Valid_B1_holdout',h),('Test_B2',t)]:
   pred=model.predict(frame[fs]);m=metric(frame.cycle_life,pred)
   metrics.append({'candidate_id':cid,'family':c['family'],'feature_set':c['feature_set'],'target_transform':c['target_transform'],'dataset':role,'n':len(frame),**m,'selected':cid==lock['winner_id']})
   for (_,row),v in zip(frame.iterrows(),pred):
    preds.append({'candidate_id':cid,'family':c['family'],'dataset':role,'cell_id':row.cell_id,'actual':row.cycle_life,'predicted':v,'error_cycles':v-row.cycle_life,'absolute_error_cycles':abs(v-row.cycle_life),'ape_pct':abs(v-row.cycle_life)/row.cycle_life*100,'policy':row.policy,'label_status':row.label_status,'ir_missing':bool(pd.isna(row.ir_mean_2_100) or pd.isna(row.ir_change_early)),'policy_seen_in_dev':row.policy in set(dev.policy)})
 pd.DataFrame(metrics).to_csv(OUT/'final_metrics.csv',index=False); pr=pd.DataFrame(preds);pr.to_csv(OUT/'final_predictions.csv',index=False);dump(OUT/'final_preprocessing_audit.json',pre)
 winner=lock['winner_id']; wp=pr[pr.candidate_id.eq(winner)].copy();wc=lock['specs'][winner];fs=FEATURE_SETS[wc['feature_set']]
 cm=cv.loc[winner];valid=next(x for x in metrics if x['candidate_id']==winner and x['dataset'].startswith('Valid'));test=next(x for x in metrics if x['candidate_id']==winner and x['dataset']=='Test_B2')
 subgroup=[]
 for dataset,g in wp.groupby('dataset'):
  masks={'all':np.ones(len(g),dtype=bool),'life_lt500':g.actual.lt(500),'life_500_to1000':g.actual.between(500,1000),'life_gt1000':g.actual.gt(1000),'IR_missing':g.ir_missing,'IR_present':~g.ir_missing,'known_policy':g.policy_seen_in_dev,'unseen_policy':~g.policy_seen_in_dev}
  for name,mask in masks.items():
   z=g.loc[mask]
   if len(z): subgroup.append({'dataset':dataset,'group':name,'n':len(z),'mape_pct':z.ape_pct.mean(),'mae_cycles':z.absolute_error_cycles.mean(),'bias_cycles':z.error_cycles.mean()})
 pd.DataFrame(subgroup).to_csv(OUT/'subgroup_errors.csv',index=False)
 shifts=[]
 for f in fs:
  lo,hi=dev[f].min(),dev[f].max()
  for role,frame in [('Valid_B1_holdout',h),('Test_B2',t)]:
   nonmissing=frame[f].notna();outside=nonmissing & ~frame[f].between(lo,hi)
   shifts.append({'feature':f,'dataset':role,'train_min':lo,'train_max':hi,'missing_n':int((~nonmissing).sum()),'outside_range_n':int(outside.sum()),'total_n':len(frame),'outside_ids':frame.loc[outside,'cell_id'].tolist()})
 dump(OUT/'feature_range_shift.json',shifts)
 model=joblib.load(OUT/'models'/f'{winner}.joblib');params=model.named_steps['model'];interpret={}
 if isinstance(params,TransformedTargetRegressor):params=params.regressor_
 if hasattr(params,'coef_'):interpret={'kind':'standardized_feature_coefficients_in_log_cycles' if wc['target_transform']=='log' else 'standardized_feature_coefficients_in_cycles','intercept':float(params.intercept_),'values':dict(zip(fs,params.coef_.tolist())),'caution':'Associations conditional on other features, not causal. Coefficients can remain unstable with small sample and correlated inputs.'}
 elif hasattr(params,'feature_importances_'):interpret={'kind':'training_impurity_importance','values':dict(zip(fs,params.feature_importances_.tolist())),'caution':'Training impurity importance is not causal or independent validation; correlated features divide importance.'}
 randomcv=pd.read_csv(OUT/'random_cv_sensitivity.csv')
 info={'evaluated_at':stamp(),'selection_lock_sha256':sha(OUT/'selection_lock.json'),'winner_id':winner,'winner_spec':wc,'features':fs,'training_n':len(dev),'valid_n':len(h),'test_n':len(t),'cv':cm.to_dict(),'valid':valid,'test':test,'gaps_percentage_points':{'valid_minus_cv':valid['mape_pct']-cm.cv_mape_pct_mean,'test_minus_valid':test['mape_pct']-valid['mape_pct'],'test_minus_paper_9_1':test['mape_pct']-9.1},'paper_reference_mape_pct':9.1,'cv_random_sensitivity':{'mean_mape_pct':randomcv.mape_pct.mean(),'sd_mape_pct':randomcv.mape_pct.std(ddof=1),'overlapping_policy_counts':randomcv.policy_overlap_count.tolist(),'selection_used':False},'label_status_counts':{role:frame.label_status.value_counts().to_dict() for role,frame in [('development',dev),('holdout',h),('test',t)]},'training_life_range':[dev.cycle_life.min(),dev.cycle_life.max()],'test_below_training_target_min_n':int(t.cycle_life.lt(dev.cycle_life.min()).sum()),'test_above_training_target_max_n':int(t.cycle_life.gt(dev.cycle_life.max()).sum()),'holdout_shared_protocol_cell_n':int(h.policy.isin(dev.policy).sum()),'negative_prediction_n':int(wp.predicted.lt(0).sum()),'subgroups':subgroup,'largest_test_errors':wp[wp.dataset.eq('Test_B2')].nlargest(5,'ape_pct').to_dict('records'),'interpretation':interpret,'batch3_model_evaluation':False,'post_test_tuning':False,'limitations':json.loads((OUT/'experiment_plan.json').read_text())['known_limits']}
 dump(OUT/'evaluation.json',info)
 print(json.dumps({k:info[k] for k in ['winner_id','valid','test','gaps_percentage_points']},ensure_ascii=False,indent=2))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['select','evaluate'],required=True);args=parser.parse_args()
 (select if args.phase=='select' else evaluate)()
