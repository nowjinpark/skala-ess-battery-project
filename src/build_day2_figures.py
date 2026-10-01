"""Publication figures for frozen DAY 2 results (no model selection)."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,fontManager
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/day2';FIG=OUT/'figures'
BLUE='#0072B2';ORANGE='#D55E00';TEAL='#009E73';GRAY='#838B92'
LABELS={'basic7':'기본 7개','deltaq8':'ΔQ 분산 추가 8개','no_qd_change7':'용량 변화량 제외 7개','no_qd_slope7':'용량 기울기 제외 7개','deltaq_mean8':'ΔQ 평균으로 교체 8개','deltaq_min8':'ΔQ 최소로 교체 8개','no_ir6':'IR 두 특성 제외 6개','deltaq_only1':'ΔQ 분산 단일 특성'}
SHORT={'qd_cycle2':'초기 용량','qd_slope_10_100':'용량 기울기','ir_mean_2_100':'평균 IR','ir_change_early':'IR 변화','tavg_mean_2_100':'평균 온도','chargetime_mean_2_100':'평균 충전 시간','log10_deltaq_var':'log 분산(ΔQ)'}
def style():
 fontManager.addfont(str(ROOT/'assets/fonts/NanumGothic-Regular.ttf'))
 plt.rcParams.update({'font.family':'NanumGothic','font.size':10,'axes.titlesize':12,'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'figure.facecolor':'white','savefig.facecolor':'white'})
def save(fig,name):
 fig.savefig(FIG/f'{name}.png',dpi=220,bbox_inches='tight');plt.close(fig)
def main():
 FIG.mkdir(exist_ok=True);style();ev=json.loads((OUT/'evaluation.json').read_text());cv=pd.read_csv(OUT/'cv_candidates.csv');pr=pd.read_csv(OUT/'final_predictions.csv');win=pr[pr.candidate_id.eq(ev['winner_id'])]
 # Same family, target transform, alpha and folds make this comparison interpretable.
 ab=cv[cv.family.eq('Ridge') & cv.target_transform.eq('log') & cv.params.eq('{"alpha": 10.0}')].copy()
 ab['order']=ab.feature_set.map({k:i for i,k in enumerate(LABELS)});ab=ab.sort_values('order')
 fig,ax=plt.subplots(figsize=(9.5,3.65));y=np.arange(len(ab));colors=[BLUE if x=='no_qd_change7' else '#A3AEB7' for x in ab.feature_set]
 ax.barh(y,ab.cv_mape_pct_mean,color=colors,height=.68,edgecolor='white');ax.errorbar(ab.cv_mape_pct_mean,y,xerr=ab.cv_mape_pct_sd,fmt='none',ecolor='#383838',capsize=3,lw=1)
 ax.set_yticks(y,[LABELS[x] for x in ab.feature_set]);ax.invert_yaxis();ax.set_xlabel('B1 Group CV MAPE (%) · 낮을수록 좋음');ax.set_xlim(0,19);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
 for yi,(_,r) in zip(y,ab.iterrows()):ax.text(r.cv_mape_pct_mean+r.cv_mape_pct_sd+.25,yi,f'{r.cv_mape_pct_mean:.2f}',va='center',fontsize=9)
 fig.tight_layout();save(fig,'feature_ablation')
 fig,axes=plt.subplots(1,2,figsize=(9.5,3.85))
 for ax,role,title in zip(axes,['Valid_B1_holdout','Test_B2'],['B1 holdout (n=8)','B2 test (n=39)']):
  g=win[win.dataset.eq(role)];lo,hi=ev['training_life_range'];ax.axvspan(lo,hi,color='#E8ECEF',zorder=0,label='B1 학습 타깃 범위')
  for mask,label,color,marker in [(g.actual.lt(500),'단수명 <500',ORANGE,'^'),(g.actual.ge(500),'수명 ≥500',BLUE,'o')]:
   z=g[mask]
   if len(z):ax.scatter(z.actual,z.predicted,label=label,color=color,marker=marker,s=40,alpha=.8,edgecolors='white',linewidth=.4)
  ax.plot([300,1250],[300,1250],'--',color='#333333',lw=1,label='정답 = 예측');ax.set(xlim=(300,1250),ylim=(300,1250),title=title,xlabel='실제 수명 (사이클)',ylabel='예측 수명 (사이클)');ax.set_aspect('equal');ax.grid(alpha=.16)
 axes[1].legend(fontsize=8,loc='upper left');fig.tight_layout();save(fig,'actual_predictions')
 fig,axes=plt.subplots(1,2,figsize=(9.5,3.5),gridspec_kw={'width_ratios':[1,1.25]})
 worst=win[win.dataset.eq('Test_B2')].nlargest(5,'ape_pct');y=np.arange(5)
 axes[0].barh(y-.18,worst.actual,height=.34,color=BLUE,label='실제');axes[0].barh(y+.18,worst.predicted,height=.34,color=ORANGE,label='예측');axes[0].set_yticks(y,worst.cell_id);axes[0].invert_yaxis();axes[0].set_xlabel('수명 (사이클)');axes[0].set_title('APE가 큰 5개 셀');axes[0].legend(fontsize=9);axes[0].grid(axis='x',alpha=.18);axes[0].set_axisbelow(True)
 shifts=pd.read_json(OUT/'feature_range_shift.json');s=shifts[shifts.dataset.eq('Test_B2')];y=np.arange(len(s));axes[1].barh(y,s.outside_range_n,color=TEAL);axes[1].set_yticks(y,[SHORT[f] for f in s.feature]);axes[1].invert_yaxis();axes[1].set_title('B1 특성 범위 밖인 B2 셀 수');axes[1].set_xlim(0,29);axes[1].set_xlabel('셀 수 (B2 전체 39개, 결측 제외)');axes[1].grid(axis='x',alpha=.18);axes[1].set_axisbelow(True)
 for yi,(_,r) in zip(y,s.iterrows()):axes[1].text(r.outside_range_n+.4,yi,str(r.outside_range_n),va='center',fontsize=9)
 fig.tight_layout(w_pad=3);save(fig,'error_diagnostics')
 fig,ax=plt.subplots(figsize=(9.5,3.2));co=ev['interpretation']['values'];keys=sorted(co,key=co.get);vals=[co[k] for k in keys];y=np.arange(len(keys));ax.barh(y,vals,color=[BLUE if v<0 else ORANGE for v in vals]);ax.set_yticks(y,[SHORT[k] for k in keys]);ax.axvline(0,color='#444',lw=.8);ax.set_xlabel('표준화한 특성 1단위당 ln(예측 수명) 계수');ax.grid(axis='x',alpha=.18);fig.tight_layout();save(fig,'coefficients')
 (OUT/'figure_manifest.json').write_text(json.dumps({'style':'black/white text, colored graphs; consistent blue/orange; no predictive retraining','files':[str(p.relative_to(ROOT)) for p in FIG.glob('*.png')]},ensure_ascii=False,indent=2))
 print('4 DAY2 figures saved')
if __name__=='__main__':main()
