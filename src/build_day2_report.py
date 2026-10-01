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
 r.start('분석 목표와 구현','초기 100사이클로 총수명 예측')
 r.p('<b>목표:</b> 초기 100사이클 측정값으로 총수명 cycle_life를 예측했습니다. 수명이 몇 사이클인지와 오차 크기를 직접 비교하기 위해 회귀를 선택했습니다. 기준은 정격 1.1 Ah의 80%인 0.88 Ah이며, 셀 1개를 데이터 1개로 보았습니다.',9.3,13.5)
 r.table(['구분','표본과 역할','사용 방법'],[['B1 개발용','28개 / 입력·모델·설정 선택','같은 충전 방식을 묶어 3분할 교차검증'],['B1 별도 검증','8개 / 내부 성능 확인','DAY 1의 seed=42 분할 유지'],['B2 테스트','39개 / 다른 배치에서 평가','모델을 정한 뒤 평가'],['B3','DAY 1 EDA에 사용','선택 사항인 추가 모델 평가는 생략']],[88,193,WIDTH-281],8.7,12)
 r.h('DAY 1에서 확인한 내용을 반영했습니다')
 r.table(['확인한 내용','모델에 반영한 방법'],[['용량 변화량과 기울기 r=0.985','중복되는 변수 하나씩 제외해 비교했습니다.'],['Delta Q의 분산·평균·최솟값과 수명의 관계','분산 추가·다른 통계값 대체·단일 입력을 비교했습니다.'],['충전 방식과 수명의 관계가 배치별로 달랐습니다.','교차검증에서 같은 충전 방식이 학습·검증에 섞이지 않게 했습니다.'],['B2 내부저항 입력값이 6개 셀에서 없었습니다.','학습 데이터의 중앙값을 사용하고 저항 제외도 비교했습니다.']],[216,WIDTH-216],8.5,12)
 r.h('최종 입력 변수 7개')
 r.table(['변수','계산 방법: 모두 100사이클 이내'],[['초기 용량','2사이클 방전 용량 QD(2)'],['초기 용량 기울기','10-100사이클 QD의 선형 기울기'],['평균 내부저항','2-100사이클 양의 IR 평균'],['내부저항 변화','91-100사이클 IR 평균 - 2-10사이클 IR 평균'],['평균 온도 / 충전 시간','2-100사이클 유효 온도 평균 / 충전 시간 평균'],['Delta Q의 로그 분산','log10 Var[Q100(V)-Q10(V)], 표본분산 ddof=1']],[132,WIDTH-132],8.5,11.8)
 r.p('<b>전처리:</b> 교차검증의 각 학습 부분에서만 중앙값과 표준화 기준을 구했습니다. 최종 모델은 개발용 28개의 기준을 그대로 적용했습니다. 전체 기록 길이·종료 용량·후기 기울기·knee·셀 ID는 입력에서 제외했습니다.',8.8,12.5)
 r.p('<b>수명값의 차이:</b> B1은 80% 부근에서 기록이 끝나 제공된 값이고, B2는 실제로 80% 아래로 내려간 시점을 확인한 값입니다. DAY 1의 제외 기준·분할과 원본 값을 유지했습니다. 과제 B2는 원저자 코드와 날짜가 달라 셀 번호를 이어 붙이지 않았습니다.',8.4,12,0,GRAY)

 r.start('모델 비교와 선택','개발용 데이터에서 후보를 골랐습니다')
 r.p('<b>비교 방법:</b> 평균 예측을 기준으로 선형회귀·Ridge·Random Forest를 비교했습니다. 입력 조합 8개를 정하고, 선형회귀·Ridge에서는 원래 수명과 ln(수명)을 비교했습니다. Ridge alpha=0.1/1/10/100 등을 미리 정해 총 113개 후보를 비교했습니다. RF는 나무 300개, 깊이 3/제한 없음, 끝 노드의 최소 표본 수 1/3, 전체 입력 변수, seed=42를 사용했습니다.',9.1,13.3)
 r.p('<b>선택 기준:</b> B1 개발용 28개를 충전 방식 18개로 묶어 GroupKFold 교차검증을 했습니다. 각 검증 부분은 10/9/9개이며 각 fold의 학습·검증 사이 같은 충전 방식의 중복은 0개입니다. 3개 검증 MAPE의 단순 평균이 가장 작은 후보를 선택했습니다.',9.1,13.3)
 names={'Dummy':'평균 예측','Linear':'선형회귀','Ridge':'Ridge','RandomForest':'Random Forest'}
 fsnames={'basic7':'기본 7개','deltaq_only1':'Delta Q 로그 분산 1개','no_qd_change7':'용량 변화량을 뺀 7개'}
 rows=[]
 for fam,cid in lock['family_winners'].items():
  a=cv.loc[cid];v=fm[fm.candidate_id.eq(cid)&fm.dataset.str.startswith('Valid')].iloc[0];t=fm[fm.candidate_id.eq(cid)&fm.dataset.eq('Test_B2')].iloc[0]
  rows.append([names[fam]+(' (선택)' if cid==e['winner_id'] else ''),fsnames[a.feature_set]+(' / ln(수명)' if a.target_transform=='log' else ' / 원래 수명'),f'{a.cv_mape_pct_mean:.2f} ± {a.cv_mape_pct_sd:.2f}',f'{v.mape_pct:.2f}',f'{t.mape_pct:.2f}'])
 r.table(['모델','계열별로 선택한 입력 / 학습할 값','CV MAPE','Valid','Test'],rows,[85,179,101,68,WIDTH-433],8.3,11.7)
 r.p('<b>최종 모델:</b> C037, Ridge(alpha=10), 입력 7개입니다. ln(수명)을 학습한 뒤 exp로 원래 사이클 수로 되돌려 모든 지표를 계산했습니다. 개발용 28개로 학습한 같은 모델을 별도 검증·B2에 적용했습니다. 표의 MAPE 단위는 %이며, ±는 fold 표준편차로 신뢰구간은 아닙니다.',8.7,12.5)
 r.fig('feature_ablation',208,'그림 1. Ridge·ln(수명)·alpha=10과 분할을 고정한 입력 비교입니다. 막대는 CV 평균, 오차선은 fold 표준편차입니다.')
 r.p('<b>결과 해석:</b> Delta Q 분산을 더하고 중복 용량 변화량을 뺀 조합이 개발 CV에서 유리했습니다. 다만 선택에 쓴 점수이므로 독립적인 성능 개선의 증거로 보지는 않았습니다. B2에서는 단일 Delta Q 선형회귀가 28.57%로 더 좋았지만, 테스트 점수를 보고 모델을 바꾸지는 않았습니다.',9,13)
 r.p(f'<b>검증 한계:</b> 표본 28개로 113개 후보를 비교해 좋은 점수가 우연히 선택됐을 가능성이 있습니다. 별도 검증용 8개 중 5개는 개발용과 충전 방식이 같았습니다. 일반 KFold로 같은 선택 모델을 확인한 보조 MAPE는 {e["cv_random_sensitivity"]["mean_mape_pct"]:.2f}%였고, 모델 재선택에는 쓰지 않았습니다.',8.6,12.5,0)

 r.start('성능과 논문 비교','B2에서는 예측 오차가 커졌습니다')
 g=e['gaps_percentage_points'];r.table(['과제 지정 구분','MAPE / Gap','계산 기준'],[['Train (Batch1 CV)',f'{e["cv"]["cv_mape_pct_mean"]:.2f}%','3개 검증 fold MAPE의 평균'],['Valid (Batch1 Hold-out)',f'{e["valid"]["mape_pct"]:.2f}%','B1 별도 검증용 8개'],['Test (Batch2)',f'{e["test"]["mape_pct"]:.2f}%','B2 39개, 선택 모델 고정'],['Gap (Train-Valid)',f'{g["valid_minus_cv"]:+.2f}%p','Valid - CV'],['Gap (Valid-Test)',f'{g["test_minus_valid"]:+.2f}%p','Test - Valid'],['Gap (Target-Test)',f'{g["test_minus_paper_9_1"]:+.2f}%p','Test - 논문 참고값 9.1%']],[181,102,WIDTH-283],9,13)
 r.p('MAPE는 |실제-예측| / 실제의 평균에 100을 곱한 값입니다. Gap은 양수가 오차 증가를 뜻하도록 계산 방향을 적었습니다. Train은 학습 데이터를 다시 예측한 점수가 아닌 교차검증 평균입니다.',8.8,12.5)
 r.table(['보조 지표','Valid (n=8)','Test (n=39)'],[['MAE (사이클)',f'{e["valid"]["mae_cycles"]:.2f}',f'{e["test"]["mae_cycles"]:.2f}'],['RMSE (사이클)',f'{e["valid"]["rmse_cycles"]:.2f}',f'{e["test"]["rmse_cycles"]:.2f}'],['R²',f'{e["valid"]["r2"]:.3f}',f'{e["test"]["r2"]:.3f}']],[181,150,WIDTH-331],8.8,12)
 r.fig('actual_predictions',225,'그림 2. 대각선 위는 수명을 더 길게 예측한 경우입니다. 회색은 B1 개발용 수명 범위인 534-1074사이클입니다.')
 r.p('<b>Gap 해석:</b> CV와 별도 검증의 차이는 +0.21%p였지만, 표본이 작아 과적합이 없다고 확정하지는 않았습니다. B2에서는 +46.78%p 나빠져 다른 배치로의 일반화에 실패했습니다. B2의 R²=-0.713은 해당 배치의 실제 평균 수명을 아는 상수 예측보다 제곱오차가 컸다는 뜻입니다.',9,13.5)
 r.p('<b>논문 비교:</b> MAPE 9.1%보다 +46.12%p 높아 참고 목표에 못 미쳤습니다. 데이터 버전·정제·분할·수명값의 기준이 달라 같은 조건의 논문 재현이라고 보지는 않았습니다.',8.9,13,0)

 r.start('오류 사례','짧은 수명을 길게 예측했습니다')
 r.table(['셀','실제 → 예측 (사이클)','APE','충전 방식'],[[x['cell_id'],f'{x["actual"]:.0f} → {x["predicted"]:.1f}',f'{x["ape_pct"]:.1f}%',x['policy']] for x in e['largest_test_errors']],[60,150,61,WIDTH-271],8.7,12.5)
 r.fig('error_diagnostics',205,'그림 3. 왼쪽은 상대오차가 큰 사례, 오른쪽은 학습 입력 범위를 벗어난 B2 셀 수입니다. 한 셀이 여러 변수에 포함될 수 있습니다.')
 r.p('<b>공통점:</b> 상대오차가 큰 5개 모두 실제 수명은 500 미만인데 더 길게 예측됐습니다. 개발용에 없는 충전 방식이며 초기 용량 기울기도 개발용 최댓값을 넘었습니다. 단수명 28개 전부를 길게 예측했고, 이 집단의 MAPE는 66.03%, 평균적으로 더 길게 예측한 정도는 295.52사이클이었습니다.',9.1,13.3)
 r.p('<b>원인으로 생각한 점:</b> B2의 30/39개는 개발용 최소수명 534보다 짧았습니다. 초기 용량 23개, 기울기 21개도 학습 범위를 벗어났습니다. 모델의 용량·기울기 계수가 양수여서 B1의 관계를 B2에 적용한 것이 과대예측에 영향을 줬을 수 있습니다. 다만 실험 집단·측정·수명값 차이가 함께 있어 원인으로 확정하지는 않았습니다.',9,13.2)
 r.p('<b>반례도 확인했습니다:</b> 저항값이 없는 6개의 MAPE는 44.35%로, 값이 있는 33개의 57.19%보다 낮았습니다. 이미 본 충전 방식 7개도 69.44%로 높았습니다. 따라서 결측이나 새로운 충전 방식 하나로 실패를 설명하기는 어려웠습니다. 집단 구성도 달라 특정 조건의 영향이 없다고 결론 내리지는 않았습니다.',9,13.2)
 r.p('<b>개선 방향:</b> 같은 기준의 수명값과 단수명·다양한 충전 방식의 개발 데이터를 추가하겠습니다. 단일 Delta Q와 여러 변수를 쓴 모델은 새 개발 배치에서 비교하고, 별도의 새 배치로 최종 평가하겠습니다. B2를 보고 바꾸는 실험은 후속 탐색으로 구분하겠습니다.',9,13.2,0)

 r.start('활용과 한계','ESS에 적용하기 전에 확인할 점')
 r.h('점검·교체 검토를 돕는 정보로 보았습니다')
 r.p('추가 현장 검증을 거치면 점검·교체 검토의 우선순위를 정하는 보조 정보로 활용할 수 있다고 보았습니다. 하지만 이번 모델은 짧은 수명을 길게 예측해 교체 검토가 늦어질 수 있습니다. 현재 결과만으로 실제 교체 시점이나 충전 제어를 자동 결정하기는 어렵습니다.')
 r.p('예측 대상은 실험 셀의 총 충방전 사이클 수입니다. 실제 ESS 팩의 남은 사용 일수나 고장 시점으로 바로 바꿀 수는 없습니다. 다른 배터리 종류·온도·운전 조건·팩 구성에서도 맞는지 확인해야 합니다.')
 r.h('남은 한계와 다음에 할 일')
 r.table(['한계','보완할 점'],[['개발용 28개·별도 검증용 8개로 표본이 작습니다.','여러 셀·충전 방식·배치를 확보하고, 분할을 바꿔도 선택이 안정적인지 확인하겠습니다.'],['113개 후보 중 좋은 점수를 선택했습니다.','선택에 쓴 CV를 독립적인 최종 성능으로 보지 않겠습니다.'],['B1 근사 수명값과 B2 관측 수명값이 다릅니다.','같은 수명·측정 종료 기준을 적용한 데이터로 다시 확인하겠습니다.'],['DAY 1에서 B2 분포도 살펴봤습니다.','완전히 처음 보는 테스트라고 하지 않고, 새로운 평가 배치를 확보하겠습니다.'],['예측 불확실성과 운영 비용을 확인하지 못했습니다.','과대예측 비용과 범위 밖 입력을 점검하고, 현장 검토 절차를 마련하겠습니다.']],[180,WIDTH-180],8.9,12.7)
 r.p('불완전한 기록을 제외한 영향과 별도 검증용의 충전 방식 일부 중복도 남은 한계입니다. Batch 3는 DAY 1 분석에 사용했고 이번 모델의 추가 평가에는 사용하지 않았습니다.',9,13)
 r.h('실행 방법과 산출물')
 r.p('README에 환경 준비와 실행 방법을 정리했습니다. DAY 2 노트북은 저장 모델의 예측과 지표를 확인할 수 있게 만들었습니다. 처음부터 학습하는 절차, 입력 변수 정의와 자세한 검증은 노트북과 실험 계획에 남겼습니다.',9.1,13.5)
 r.h('출처와 작성자')
 r.p('DS Mini Project의 DAY 2 안내·README 예시, 제공 Statistics·MLDL·Wrap-up·Evaluation Metrics·ML Hyperparameters 자료와 강의 녹음을 참고했습니다. 데이터는 Kaggle itshpark/data-driven-prediction-of-battery-cycle을 사용했습니다.',8.6,12.3)
 r.p('Severson et al. (2019), Data-driven prediction of battery cycle life before capacity degradation. Nature Energy. DOI: 10.1038/s41560-019-0356-8. 원저자 코드와 scikit-learn 공식 문서의 링크는 README에 정리했습니다.',8.4,12,5,GRAY)
 r.p('울산 1반 박진원: 데이터 정제·EDA·입력 변수 설계·모델 비교·오류 해석·보고서 작성을 진행했습니다.',8.6,12,0)
 r.save();print(OUT)
if __name__=='__main__':main()
