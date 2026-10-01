"""Concise DAY 2 report, derived only from frozen results."""
from pathlib import Path
import json
from xml.sax.saxutils import escape
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import Color
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'results/day2';OUT=ROOT/'output/pdf/DS-MINI-Result-울산_1반-박진원.pdf'
W,H=A4;L=38;WIDTH=W-2*L;BLACK=Color(.08,.08,.08);GRAY=Color(.34,.34,.34);LIGHT=Color(.94,.94,.94);RULE=Color(.72,.72,.72);PAGES=5
class Report:
 def __init__(self):
  self.c=canvas.Canvas(str(OUT),pagesize=A4);self.c.setTitle('DAY 2 모델 구현·성능 분석 | 배터리 수명 예측');self.c.setAuthor('울산 1반 박진원');self.page=0;self.y=0
 def p(self,text,size=9.6,leading=14,gap=7,color=BLACK):
  p=Paragraph(text,ParagraphStyle('p',fontName='Nanum',fontSize=size,leading=leading,wordWrap='CJK',textColor=color));_,h=p.wrap(WIDTH,1000)
  if self.y-h<48:raise RuntimeError(f'Overflow page{self.page}: {text[:60]}')
  p.drawOn(self.c,L,self.y-h);self.y-=h+gap
 def start(self,label,title):
  if self.page:self.c.showPage()
  self.page+=1;c=self.c;c.setFillColor(BLACK);c.setFont('NanumBold',9);c.drawString(L,H-32,'DAY 2 모델 구현·성능 분석');c.setFont('Nanum',8);c.drawRightString(W-L,H-32,'울산 1반 · 박진원');c.setStrokeColor(BLACK);c.setLineWidth(.7);c.line(L,H-41,W-L,H-41);self.y=H-59;self.p(f'<b>{label} | {title}</b>',17,22,12);c.setStrokeColor(RULE);c.line(L,35,W-L,35);c.setFont('Nanum',8);c.setFillColor(GRAY);c.drawString(L,22,'ESS 배터리 수명 예측 · 회귀');c.drawRightString(W-L,22,f'{self.page} / {PAGES}')
 def h(self,text):self.p(f'<b>{text}</b>',11.2,16,6)
 def table(self,headers,rows,widths,size=9,leading=12.5):
  cells=[]
  for i,row in enumerate([headers]+rows):
   style=ParagraphStyle('cell',fontName='NanumBold' if i==0 else 'Nanum',fontSize=size,leading=leading,wordWrap='CJK',textColor=BLACK)
   cells.append([Paragraph(escape(str(x)).replace('\n','<br/>'),style) for x in row])
  t=Table(cells,colWidths=widths);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),LIGHT),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEABOVE',(0,0),(-1,0),.6,BLACK),('LINEBELOW',(0,0),(-1,0),.5,BLACK),('LINEBELOW',(0,1),(-1,-1),.3,RULE),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]));_,h=t.wrap(WIDTH,1000)
  if self.y-h<48:raise RuntimeError(f'Table overflow p{self.page}')
  t.drawOn(self.c,L,self.y-h);self.y-=h+9
 def fig(self,name,height,caption):
  img=ImageReader(str(D/'figures'/f'{name}.png'));w,h=img.getSize();s=min(WIDTH/w,height/h);w*=s;h*=s
  if self.y-h<48:raise RuntimeError('Figure overflow')
  self.c.drawImage(img,L+(WIDTH-w)/2,self.y-h,width=w,height=h);self.y-=h+5;self.p(caption,8.2,11.5,9,GRAY)
 def save(self):assert self.page==PAGES;self.c.save()
def main():
 for name,file in [('Nanum','NanumGothic-Regular.ttf'),('NanumBold','NanumGothic-Bold.ttf')]:pdfmetrics.registerFont(TTFont(name,str(ROOT/'assets/fonts'/file)))
 pdfmetrics.registerFontFamily('Nanum',normal='Nanum',bold='NanumBold')
 e=json.loads((D/'evaluation.json').read_text());lock=json.loads((D/'selection_lock.json').read_text());cv=pd.read_csv(D/'cv_candidates.csv').set_index('candidate_id');fm=pd.read_csv(D/'final_metrics.csv');r=Report()
 r.start('전략 → 구현','초기 100사이클로 총수명 예측')
 r.p('<b>목표:</b> 초기 100사이클의 측정값으로 배터리의 총수명 cycle_life를 예측한다. 정격 1.1 Ah의 80%인 0.88 Ah를 수명 기준으로 삼으며 셀 1개가 입력 1행이다.')
 r.table(['구분','표본·용도','구현'],[['B1 개발','28개 / 모델·특성·파라미터 선택','프로토콜 단위 GroupKFold 3-fold'],['B1 holdout','8개 / 고정 내부 검증','DAY 1의 seed=42 분할 유지'],['B2 test','39개 / 다른 배치 최종 평가','모델 확정 후 평가; 재튜닝 없음'],['B3','DAY 1 EDA만 수행','선택 사항인 추가 모델 평가는 생략']],[86,193,WIDTH-279])
 r.h('DAY 1 관찰을 실제 구현으로 연결')
 r.table(['EDA 근거','실제 구현'],[['용량 변화량-기울기 r=0.985','중복 두 변수 중 하나를 뺀 특성 집합 비교'],['Delta Q의 수명 관계와 요약값 중복','log 분산 추가·평균/최소 교체·단일 특성 비교'],['B2 내부저항 결측 6개','학습 fold 중앙값 대치, IR 제외 집합도 사전 비교'],['소표본·특성 상관·배치 차이','Dummy, Linear, Ridge, Random Forest 비교']],[216,WIDTH-216],8.8,12.5)
 r.h('최종 선택된 입력 7개')
 r.table(['특성','계산 정의 (100사이클 이내)'],[['초기 용량','QD(2)'],['초기 용량 기울기','10-100사이클 QD의 선형 기울기'],['평균 내부저항','2-100사이클 양의 IR 평균'],['내부저항 변화','91-100사이클 IR 평균 - 2-10사이클 IR 평균'],['평균 온도 / 충전 시간','2-100사이클 유효 온도 평균 / 충전 시간 평균'],['Delta Q의 log 분산','log10 Var[Q100(V)-Q10(V)], 표본분산 ddof=1']],[132,WIDTH-132],8.6,12)
 r.p('<b>학습 순서:</b> fold별 중앙값 대치 → 표준화 → ln(수명)에 Ridge 학습 → exp로 원단위 복원. 평가 지표는 복원한 사이클 수로 계산한다. 전체 수명 궤적, 후기 기울기, knee, 종료 용량, 셀 ID는 입력에서 제외했다.',9,13)
 r.p('<b>라벨 조건:</b> B1은 제공된 근사 종료 라벨, B2는 실제 80% 교차 라벨이다. 원본 139개 중 DAY 1에 정한 제외 기준과 분할을 그대로 사용했다. B2 날짜는 원저자 공개 코드의 B2와 달라 임의로 셀을 연결하지 않았다.',8.6,12.5,0,GRAY)

 r.start('모델 선택','개발 데이터에서만 후보 확정')
 r.p('<b>사전 고정한 탐색:</b> 특성 집합 8개. Linear/Ridge는 raw·ln 타깃을 비교하고 Ridge alpha는 0.1/1/10/100. RF는 트리 300개, 깊이 3/무제한, leaf 최소 1/3, 전체 특성, seed=42. Dummy를 포함해 113개 후보를 같은 3개 fold에서 비교했다.',9.3,13.5)
 r.p('<b>선택 기준:</b> 검증 fold MAPE의 단순 평균이 가장 작은 후보. fold 크기는 10/9/9개이며 학습·검증 사이 프로토콜 중복은 0개다. 선택 후 개발 28개로 학습한 모델을 고정하고 holdout·B2에 동일하게 적용했다.',9.3,13.5)
 rows=[]
 for fam,cid in lock['family_winners'].items():
  a=cv.loc[cid];v=fm[fm.candidate_id.eq(cid)&fm.dataset.str.startswith('Valid')].iloc[0];t=fm[fm.candidate_id.eq(cid)&fm.dataset.eq('Test_B2')].iloc[0]
  rows.append([fam+(' (선택)' if cid==e['winner_id'] else ''),a.feature_set+(' / ln y' if a.target_transform=='log' else ' / raw y'),f'{a.cv_mape_pct_mean:.2f} ± {a.cv_mape_pct_sd:.2f}',f'{v.mape_pct:.2f}',f'{t.mape_pct:.2f}'])
 r.table(['모델','계열별 CV 최선 후보','CV MAPE','Valid','Test'],rows,[89,182,99,68,WIDTH-438],8.5,12)
 r.p('모든 MAPE 단위는 %. CV의 ±는 3개 fold의 표본 표준편차이며 신뢰구간이 아니다. 계열별 최선 후보도 평가 전에 고정했다. 선택된 Ridge는 alpha=10, 용량 변화량을 제외한 7개 특성, ln 타깃을 사용한다.',8.6,12.5)
 r.fig('feature_ablation',216,'그림 1. Ridge·ln 타깃·alpha=10·동일 fold를 고정한 특성 비교. 막대=CV 평균, 오차선=fold 표준편차. 선택에 쓴 개발 점수다.')
 r.p('<b>해석:</b> 기본 7개에 Delta Q 분산을 더하고 중복 용량 변화량을 제거한 구성이 개발 CV에서 유리했다. 독립적인 성능 개선의 확정 증거는 아니다. B2에서 Linear 단일 특성이 28.57%로 더 낮았지만 이 결과로 최종 모델을 교체하지 않았다.',9.1,13.5)
 r.p(f'<b>검증 한계:</b> 후보 113개에 비해 개발 표본은 28개로 작아 선택 편향이 가능하다. 같은 확정 모델의 일반 KFold 민감도 MAPE는 {e["cv_random_sensitivity"]["mean_mape_pct"]:.2f}%였다. 무작위 분할이 항상 낙관적이라는 뜻은 아니다. 기존 holdout은 알려진 프로토콜 5개·새 프로토콜 3개 셀을 포함한다.',8.8,13,0)

 r.start('필수 성능표','내부 검증과 배치 일반화의 차이')
 g=e['gaps_percentage_points'];r.table(['노션 구분','MAPE / Gap','계산·의미'],[['Train (Batch1 CV)',f'{e["cv"]["cv_mape_pct_mean"]:.2f}%','3개 검증 fold MAPE의 평균'],['Valid (Batch1 Hold-out)',f'{e["valid"]["mape_pct"]:.2f}%','고정한 B1 8개'],['Test (Batch2)',f'{e["test"]["mape_pct"]:.2f}%','B2 39개, 선택 모델 고정'],['Gap (Train-Valid)',f'{g["valid_minus_cv"]:+.2f}%p','Valid - CV'],['Gap (Valid-Test)',f'{g["test_minus_valid"]:+.2f}%p','Test - Valid'],['Gap (Target-Test)',f'{g["test_minus_paper_9_1"]:+.2f}%p','Test - 논문 참고값 9.1%']],[181,102,WIDTH-283],9,13)
 r.p('MAPE = 평균(|실제-예측| / 실제) × 100. 노션의 Gap 명칭을 유지하되 양수가 오차 증가를 뜻하도록 식을 명시했다. Train은 학습 데이터에 다시 예측한 적합 오차가 아니다.',8.6,12.5)
 r.table(['보조 지표','Valid (n=8)','Test (n=39)'],[['MAE (사이클)',f'{e["valid"]["mae_cycles"]:.2f}',f'{e["test"]["mae_cycles"]:.2f}'],['RMSE (사이클)',f'{e["valid"]["rmse_cycles"]:.2f}',f'{e["test"]["rmse_cycles"]:.2f}'],['R²',f'{e["valid"]["r2"]:.3f}',f'{e["test"]["r2"]:.3f}']],[181,150,WIDTH-331],8.8,12)
 r.fig('actual_predictions',230,'그림 2. 대각선 위는 수명 과대예측. 회색 영역은 B1 개발의 실제 수명 범위(534-1074사이클)다.')
 r.p('<b>Gap 해석:</b> 내부 CV와 holdout 차이는 +0.21%p로 작지만, 이것만으로 과적합이 없다고 확정할 수 없다. B2에서는 +46.78%p 악화되어 배치 일반화에 실패했다. B2 R²=-0.713은 B2 평균 수명을 정답으로 아는 상수 기준보다 제곱오차가 큼을 뜻한다.',9,13.5)
 r.p('<b>논문 비교:</b> 9.1% 대비 +46.12%p로 참고 목표에 못 미쳤다. 배치 버전·정제·분할·라벨 조건이 달라 동일 조건의 논문 재현으로 해석하지 않는다.',8.8,13,0)

 r.start('오류 분석','짧은 수명을 길게 예측하는 문제')
 r.table(['셀','실제 → 예측 (사이클)','APE','프로토콜'],[[x['cell_id'],f'{x["actual"]:.0f} → {x["predicted"]:.1f}',f'{x["ape_pct"]:.1f}%',x['policy']] for x in e['largest_test_errors']],[60,150,61,WIDTH-271],8.7,12.5)
 r.fig('error_diagnostics',218,'그림 3. 왼쪽: 최대 상대오차 사례. 오른쪽: 학습 특성의 최솟값~최댓값 밖인 B2 셀 수; 같은 셀이 여러 특성에 중복될 수 있다.')
 r.p('<b>공통점:</b> 최대 APE 5개는 모두 실제 수명 500 미만이고 과대예측했다. B1 개발에 없는 프로토콜이며, 초기 용량 기울기도 B1 개발 최댓값을 넘는다. B2 단수명 28개 전부가 과대예측되었고 평균 편향은 +295.52사이클, MAPE는 66.03%다.',9.2,13.5)
 r.p('<b>원인 가설:</b> B2의 30/39개는 B1 개발 최소수명 534보다 짧다. 초기 용량 23개, 기울기 21개도 학습 특성 범위를 벗어난다. 선택 모델의 용량·기울기 계수가 양수여서 B1의 관계를 B2로 옮길 때 긴 수명 예측에 기여했을 수 있다. 실험 집단·측정·라벨 차이와 얽혀 있어 원인으로 확정하지 않는다.',9.1,13.5)
 r.p('<b>반례도 확인:</b> IR 결측 6개 MAPE 44.35%, 비결측 33개 57.19%이므로 결측만으로 실패를 설명할 수 없다. 알려진 프로토콜 7개도 MAPE 69.44%로 높다. newstructure 표기 여부는 짧은 수명 집단과 겹쳐 독립 효과를 분리할 수 없다.',9.1,13.5)
 r.p('<b>다음 실험:</b> 같은 정의의 수명 라벨과 짧은 수명·다양한 프로토콜을 포함한 개발 데이터를 추가한다. 단일 Delta Q와 다변량 모델의 배치 안정성을 새 개발 배치에서 비교하고, 새로 확보한 별도 배치를 최종 평가에 남긴다. 이번 B2 결과를 보고 재튜닝한 성능은 별도 후속 실험으로 구분해야 한다.',9.1,13.5,0)

 r.start('도메인 해석 · 재현','활용 조건과 남은 한계')
 r.h('ESS/BESS 관점의 의미')
 r.p('초기 측정으로 수명을 가늠하는 모델은 추가 현장 검증을 거친 뒤 배터리 검사·교체 검토의 우선순위를 정하는 보조 정보로 활용할 수 있다. 그러나 이번 모델은 짧은 수명을 과대평가하므로 교체 검토가 늦어질 위험이 있다. 현재 성능으로 실제 교체 시점이나 충전 제어를 자동 결정하기에는 근거가 부족하다.')
 r.p('예측 대상은 실험 셀의 총 충방전 사이클 수다. 실제 설비의 남은 사용 일수·달력 수명이나 팩 전체의 고장 시점을 직접 예측한 것이 아니다. 실험실 급속충전 셀에서 확인한 관계를 다른 화학계·온도·충전 운전·팩 구성으로 옮기려면 별도 검증이 필요하다.')
 r.h('개발 한계와 개선의 우선순위')
 r.table(['한계','후속 검증 / 대응'],[['개발 28개·holdout 8개의 소표본','더 많은 독립 셀·프로토콜·수집 배치를 확보하고 반복/중첩 검증으로 선택 안정성 확인'],['단수명 표본·배치 분포 차이','학습 범위를 넓히고 범위 밖 입력을 감지해 추가 검사 대상으로 분리'],['B1 근사 종료 / B2 관측 라벨','동일한 EOL 정의와 측정 종료 기준을 적용한 데이터셋으로 재검증'],['DAY 1에서 전체 배치 EDA 관찰','B2는 완전히 미관찰한 테스트가 아님을 공개하고 새 외부 검증 배치 확보'],['불확실성과 운영 비용 미검증','예측 구간의 보정·과대예측 비용·현장 점검 절차를 검증한 뒤 의사결정 지원 범위 설정']],[161,WIDTH-161],9,13)
 r.h('재현 가능한 산출물')
 r.p('README.md에 과제 지정 형식의 결과와 해석을 정리했다. 02_DAY2_Modeling.ipynb는 저장 모델의 예측·지표를 재현하며, train_day2.py는 새 실행 환경에서 select → evaluate 순서로 동일한 실험을 수행한다. 기존 선택 결과가 있으면 재튜닝을 막도록 실행을 중단한다.',9.1,13.5)
 r.p('실험 계획, fold별 셀·프로토콜, 전처리 통계, 113개 CV 결과, 선택 고정 기록, 모델, 개별 예측, 평가 지표와 무결성 검증 기록을 results/day2/에 보존했다. 원본 MAT 파일과 가상환경은 제출용 압축에 포함하지 않는다.',9.1,13.5)
 r.h('출처')
 r.p('DS Mini Project 노션의 DAY 2 안내·README 예시·평가표(2026-10-01 확인). Kaggle itshpark/data-driven-prediction-of-battery-cycle. 제공 Statistics·MLDL·Wrap-up·Evaluation Metrics·ML Hyperparameters 자료 및 강의 녹음.',8.4,12)
 r.p('Severson et al. (2019), Data-driven prediction of battery cycle life before capacity degradation. Nature Energy. DOI: 10.1038/s41560-019-0356-8. scikit-learn 공식 문서: GroupKFold, Pipeline, TransformedTargetRegressor. 상세 링크·버전·명령은 README와 실험 계획에 기록했다.',8.4,12,0,GRAY)
 r.save();print(OUT)
if __name__=='__main__':main()
