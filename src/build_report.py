"""Build the concise DAY 1 submission from audited results."""
from pathlib import Path
import argparse
import json
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import Color
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.utils import ImageReader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/pdf/DS-MINI-Design-울산_1반-박진원.pdf'
W, H = A4
L = 43
WIDTH = W - 2 * L
BLACK = Color(0, 0, 0)
GRAY = Color(.35, .35, .35)
LIGHT = Color(.94, .94, .94)
RULE = Color(.65, .65, .65)
WHITE = Color(1, 1, 1)
PAGES = 6


def esc(value):
    return escape(str(value)).replace('\n', '<br/>')


class Report:
    def __init__(self, output):
        self.c = canvas.Canvas(str(output), pagesize=A4)
        self.c.setTitle('DAY 1 분석·모델 설계서 | 배터리 수명 예측')
        self.c.setAuthor('박진원 | 울산 1반')
        self.page = 0
        self.y = 0

    def paragraph(self, text, size=10, leading=15, gap=7, color=BLACK):
        style = ParagraphStyle('p', fontName='Nanum', fontSize=size, leading=leading,
                               wordWrap='CJK', textColor=color)
        p = Paragraph(text, style)
        _, height = p.wrap(WIDTH, 1000)
        if self.y-height < 49:
            raise RuntimeError(f'Page {self.page} overflows: {text[:70]}')
        p.drawOn(self.c, L, self.y-height)
        self.y -= height+gap

    def start(self, label, title):
        if self.page:
            self.c.showPage()
        self.page += 1
        self.c.setFillColor(BLACK)
        self.c.setFont('NanumBold', 9)
        self.c.drawString(L, H-32, 'DAY 1 분석·모델 설계서')
        self.c.setFont('Nanum', 8)
        self.c.drawRightString(W-L, H-32, '울산 1반 · 박진원')
        self.c.setStrokeColor(BLACK)
        self.c.setLineWidth(.7)
        self.c.line(L, H-41, W-L, H-41)
        self.y = H-59
        self.paragraph(f'<b>{label} | {title}</b>', size=17, leading=22, gap=12)
        self.c.setStrokeColor(RULE)
        self.c.setLineWidth(.4)
        self.c.line(L, 35, W-L, 35)
        self.c.setFillColor(GRAY)
        self.c.setFont('Nanum', 8)
        self.c.drawString(L, 22, 'ESS 배터리 수명 예측 · 회귀')
        self.c.drawRightString(W-L, 22, f'{self.page} / {PAGES}')

    def heading(self, text):
        self.paragraph(f'<b>{text}</b>', size=11.2, leading=16, gap=6)

    def finding(self, label, text):
        self.paragraph(f'<b>{label}</b>  {text}', size=10, leading=15.5, gap=8)

    def table(self, headers, rows, widths, size=9, leading=13):
        cells = []
        for i, row in enumerate([headers]+rows):
            style = ParagraphStyle('cell', fontName='NanumBold' if i == 0 else 'Nanum',
                                   fontSize=size, leading=leading, wordWrap='CJK', textColor=BLACK)
            cells.append([Paragraph(esc(x), style) for x in row])
        table = Table(cells, colWidths=widths)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), LIGHT),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LINEABOVE', (0, 0), (-1, 0), .6, BLACK),
            ('LINEBELOW', (0, 0), (-1, 0), .5, BLACK),
            ('LINEBELOW', (0, 1), (-1, -1), .3, RULE),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        _, height = table.wrap(WIDTH, 1000)
        if self.y-height < 49:
            raise RuntimeError(f'Table overflows page {self.page}')
        table.drawOn(self.c, L, self.y-height)
        self.y -= height+10

    def figure(self, name, height, caption):
        path = ROOT / 'results/figures_report' / f'{name}.png'
        img = ImageReader(str(path))
        w, h = img.getSize()
        scale = min(WIDTH/w, height/h)
        w, h = w*scale, h*scale
        if self.y-h < 49:
            raise RuntimeError(f'Figure overflows page {self.page}')
        self.c.drawImage(img, L+(WIDTH-w)/2, self.y-h, width=w, height=h)
        self.y -= h+5
        self.paragraph(caption, size=8.5, leading=12, gap=11, color=GRAY)

    def save(self):
        assert self.page == PAGES
        self.c.save()


def build(data, output):
    for name, filename in [('Nanum', 'NanumGothic-Regular.ttf'), ('NanumBold', 'NanumGothic-Bold.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(ROOT/'assets/fonts'/filename)))
    pdfmetrics.registerFontFamily('Nanum', normal='Nanum', bold='NanumBold')
    output.parent.mkdir(parents=True, exist_ok=True)
    r = Report(output)
    life = data['life']['summary_rows']
    dq = data['deltaq']['correlation_with_life']
    pair = data['correlation']['high_feature_pairs'][0]
    plan = data['model_plan']
    review = json.loads((ROOT/'results/report_review.json').read_text())

    r.start('문제 정의 · EDA 1', '수명 분포와 배치별 차이')
    r.paragraph('<b>분석 목표:</b> 배터리의 초기 100사이클 정보로 전체 수명(cycle_life)을 예측합니다. '
                '수명 기준은 정격 용량 1.1 Ah의 80%인 0.88 Ah로 정했습니다. 배터리 셀 1개를 데이터 1개로 봅니다.')
    counts = {b['name']: b for b in data['meta']['batches']}
    r.table(['배치', '원본 → 분석', '평균 ± 표준편차', '중앙값', '수명 범위', '<500 n(%)', '>1000 n(%)'],
            [[v['batch'], f"{counts[v['batch']]['raw_cells']} → {v['n']}", f"{v['mean']:.1f} ± {v['std']:.1f}",
              f"{v['median']:.1f}", f"{v['min']:.0f}-{v['max']:.0f}",
              f"{v['short_n']} ({v['short_pct']:.1f})", f"{v['long_n']} ({v['long_pct']:.1f})"] for v in life],
            [52, 68, 110, 53, 85, 70, WIDTH-438], size=8.5, leading=12)
    r.paragraph('수명 단위는 사이클이며, n은 셀 개수입니다. 기록이 불완전하거나 후속 기록이 필요한 B1 10개, '
                '수명값이 없는 B2 8개, 원저자 제외 목록에 있는 B3 6개를 빼고 총 115개를 분석했습니다. '
                '<b>B1·B3는 용량이 80% 부근에서 기록이 끝나 제공된 수명값이고, B2는 용량이 80% 아래로 내려간 시점을 확인한 값입니다.</b> '
                '원본 값은 바꾸지 않았습니다. 과제 B2(2018-02-20)는 원저자 코드의 B2(2017-06-30)와 달라 셀 번호를 그대로 연결하지 않았습니다.',
                size=9, leading=13.2)
    r.figure('life_distribution', 190, '그림 1. 세 배치의 수명 분포를 150-2300사이클 범위에서 비교했습니다. 단수명은 500 미만, 장수명은 1000 초과입니다.')
    r.finding('확인한 내용', 'B2는 39개 중 28개(71.8%)가 단수명인 반면, B1·B3에는 단수명 셀이 없었습니다. '
              'B3는 40개 중 19개(47.5%)가 장수명이었습니다.')
    r.finding('이상치와 원인 검토', '배치별 1.5×IQR 기준으로 확인한 결과, 수명이 짧은 쪽의 이상치는 없었습니다. '
              '긴 쪽에는 B2 9개(b2c7 등), B3 4개(b3c38 등)가 있었지만 유효한 기록이므로 유지했습니다. '
              '가장 짧은 b2c19(392), b2c6(393)도 이 기준의 이상치는 아닙니다. '
              'B2의 newstructure 표시가 없는 30개와 있는 9개의 수명 중앙값은 각각 451/904사이클이었습니다. '
              '집단 차이는 보이지만 표시의 정확한 의미를 확인하지 못해 원인으로 단정하지 않았습니다.')
    r.finding('모델에 반영할 내용', 'B1으로 학습한 모델은 B2의 더 짧은 수명도 예측해야 합니다. '
              'Random Forest는 학습한 수명 범위 밖의 값을 예측할 수 없으므로, 선형회귀·Ridge와 비교하겠습니다.')

    r.start('EDA 2', '용량이 줄어드는 양상')
    r.figure('capacity_degradation', 330,
             '그림 2. 위는 전체 셀의 용량 변화, 아래는 각 배치의 수명 중앙값에 가장 가까운 셀입니다. '
             '검은 점선은 변화 추세를 나타낸 두 직선이고, 세로선은 두 직선이 만나는 지점입니다.')
    r.finding('확인한 내용', '분석한 115개 셀 모두에서 마지막 20% 수명 구간의 용량 기울기가 '
              '초기 10-100사이클 구간보다 더 작았습니다. 후반에는 용량 감소가 더 가파르게 나타났습니다.')
    r.finding('급격히 줄어드는 지점', '용량이 더 빠르게 줄기 시작하는 지점(knee)을 살펴봤습니다. '
              'b1c11/b2c16/b3c25의 변화 추세를 두 직선으로 나눠 보니, 만나는 지점은 약 590/340/820사이클이었습니다. '
              '10사이클부터 기록을 사용하고, 각 지점 주변 7사이클의 중앙값으로 일시적인 변동을 줄였습니다. '
              '뒤쪽 직선이 더 가파르게 내려가는 경우만 비교했습니다. 수명의 20-90% 구간에서 변동을 줄인 값과 두 직선의 차이를 각각 제곱해 더한 값이 가장 작은 지점을 골랐습니다. '
              '범위를 10-90%, 20-95%로 바꿔도 결과는 같았습니다. 대략적인 위치를 살펴본 것으로, 정확한 knee라고 확정하지는 않았습니다.')
    r.finding('해석', '초기 용량 기울기가 0 이상인 셀은 B1 7/36, B2 22/39, B3 1/40개였습니다. '
              '초기에는 용량이 늘어나는 경우도 있어 변화량을 모두 용량 감소로 해석하지 않았습니다. '
              '후반 기울기가 더 작다는 사실만으로 knee의 위치를 알 수 있는 것도 아닙니다.')
    r.finding('모델에 반영할 내용', '초기 용량과 함께 10-100사이클의 변화량·기울기를 입력 후보로 사용하겠습니다. '
              '예측 시점에 알 수 없는 100사이클 이후 용량, 후반 기울기, knee 위치는 입력에서 제외하겠습니다.')

    r.start('EDA 3', '전압별 용량 차이와 수명의 관계')
    r.figure('deltaq', 325,
             '그림 3. Delta Q(V)는 같은 전압에서 100사이클 용량에서 10사이클 용량을 뺀 값입니다. '
             '곡선은 수명 그룹별로, 산점도는 배치별로 색과 모양을 구분했습니다.')
    groups = [x for x in review['deltaq_group_medians'] if x['life_group'] in ['short','long']]
    r.table(['배치·그룹(n)', 'Delta Q 평균', 'Delta Q 최솟값', 'log10(Delta Q 분산)'],
            [[f"{x['batch']} {'단' if x['life_group']=='short' else '장'}({x['n']})", f"{x['deltaq_mean']:.4f}",
              f"{x['deltaq_min']:.4f}", f"{x['log10_deltaq_var']:.2f}"] for x in groups],
            [125, 116, 116, WIDTH-357], size=8.8, leading=12)
    r.paragraph('셀마다 구한 평균·최솟값·로그 분산을 그룹별 중앙값으로 정리했습니다. '
                '평균과 최솟값의 단위는 Ah입니다. B1·B3에는 단수명 셀이 없었습니다.', size=8.5, leading=12)
    r.finding('확인한 내용과 해석', 'B2의 단수명 셀은 장수명 셀보다 Delta Q의 평균·최솟값이 더 작고 분산이 컸습니다. '
              'B1·B3의 장수명 셀에서도 작은 분산을 확인했지만, 단수명 셀이 없어 두 그룹을 직접 비교할 수는 없었습니다.')
    r.finding('입력 변수 선택 이유', 'B1 개발용 28개의 수명과 상관계수는 Delta Q 평균 +0.723, 최솟값 +0.764, 로그 분산 -0.782였습니다. '
              '세 변수끼리도 |r|=0.966-0.990으로 높아 비슷한 정보를 담고 있습니다. '
              '따라서 로그 분산을 우선 사용하고, 평균·최솟값으로 바꾼 경우도 교차검증으로 비교하겠습니다. '
              '분산은 표본분산(ddof=1)을 쓰고 log10을 적용합니다. 상관계수가 조금 높다고 예측 성능도 더 좋다고 단정하지는 않았습니다.')

    r.start('EDA 4', '충전 조건과 수명의 관계')
    r.figure('charge_policy', 325,
             '그림 4. 첫 충전 단계의 C-rate와 수명, 배치별 주요 충전 방식의 평균 수명을 비교했습니다. '
             '셀 개수가 많은 방식 3개씩을 골랐고, 개수가 같으면 이름순으로 정렬했습니다(n은 셀 개수입니다).')
    r.table(['배치', 'C-rate와 수명 r', 'C-rate와 후반 기울기 r'],
            [[x['batch'], f"{x['c_rate_vs_life_r']:.3f}", f"{x['c_rate_vs_late_slope_r']:.3f}"]
             for x in data['policy']['correlations']], [73, 170, WIDTH-243], size=9)
    r.finding('확인한 내용과 해석', 'B1에서 첫 전류가 같은 8C여도, 전류를 바꾸는 충전 상태(SOC)가 15/25/35%일 때 '
              '평균 수명은 1008.5/676.5/607.5사이클로 달랐습니다(각 2개). '
              '첫 전류뿐 아니라 바꾸는 시점과 이후 충전 방식도 함께 봐야 합니다. '
              'B3에서는 C-rate가 높을수록 후반 용량 감소가 더 가파른 경향이 있었지만, 원인과 결과로 단정하지는 않았습니다. '
              '전체 40개 충전 방식 중 4개는 셀이 1개뿐이어서 해석에 주의가 필요합니다.')
    r.finding('모델에 반영할 내용', '초기 충전 시간처럼 실제 측정한 조건을 입력 후보로 검토하겠습니다. '
              '첫 전류만 보고 수명을 판단하지 않고, 이후 충전 방식과 배치 차이도 고려하겠습니다. '
              '표본이 적은 그룹의 결과를 전체 셀에 그대로 적용하지 않겠습니다.')

    r.start('EDA 5', '초기 변수와 수명의 관계')
    r.figure('correlation', 225,
             '그림 5. B1 개발용 28개로 계산한 피어슨 상관계수입니다. 음수는 파랑, 양수는 주황으로 표시하고 각 칸에 값을 적었습니다.')
    labels = ['초기 용량', '용량 변화량', '용량 기울기', '평균 저항', '저항 변화', '평균 온도', '평균 충전 시간', 'Delta Q의 로그 분산']
    fs = [x['name'] for x in data['features']]
    corr = {(x['batch'],x['feature']):x for x in review['batch_correlations']}
    r.table(['배치별 상관관계 비교', 'B1 r', 'B2 r', 'B3 r'],
            [[label]+[f"{corr[(f'Batch {i}',feature)]['pearson']:+.3f}" for i in (1,2,3)]
             for label,feature in zip(labels,fs)], [WIDTH-240,80,80,80], size=8.7, leading=11.5)
    r.paragraph('B1은 36개, B3는 40개를 비교했습니다. B2의 저항 평균·변화는 33개, 나머지는 39개입니다. '
                '이 표는 배치 차이를 살펴보기 위한 것으로, B2·B3를 보고 모델을 조정하지는 않겠습니다.', size=8.3, leading=11.5)
    r.finding('확인한 내용과 해석', '8개 후보 중 B1·B3는 Delta Q의 로그 분산(-0.827/-0.742), '
              'B2는 충전 시간(-0.917)이 수명과 가장 강한 상관관계를 보였습니다. '
              '충전 시간의 상관계수는 배치에 따라 부호가 달라, 모든 배치에서 같은 관계가 나타난다고 보기 어렵습니다.')
    r.finding('모델에 반영할 내용', f"모델 선택에 사용할 B1 개발용 28개에서는 로그 분산과 수명의 r={dq['pearson']:.3f}, "
              f"용량 변화량과 기울기 사이의 r={pair['correlation']:.3f}이었습니다. "
              '비슷한 정보를 담은 변수들이 겹치는 문제(다중공선성)를 고려해 변수를 줄인 경우와 Ridge를 비교하겠습니다. '
              '최종 변수·모델 선택과 전처리는 B1 개발용 데이터의 교차검증 안에서 진행하겠습니다.')

    r.start('모델 설계', '분석 결과를 반영한 모델 설계')
    r.heading('입력 변수 후보와 전처리')
    r.table(['입력 후보', '계산 방법 / 사용하려는 이유'], [
        ['초기 용량', 'QD(2)를 사용해 처음 용량 수준을 비교합니다.'],
        ['초기 용량 변화량', 'QD(100)-QD(10)으로 초기 변화를 봅니다. 증가도 가능해 모두 용량 감소로 해석하지는 않습니다.'],
        ['초기 용량 기울기', '10-100사이클 QD의 변화 속도를 봅니다. 변화량과 정보가 겹쳐 하나를 뺀 경우도 비교합니다.'],
        ['평균 내부저항', '2-100사이클 양의 IR 평균으로 초기 저항 수준을 봅니다.'],
        ['내부저항 변화', '91-100사이클 IR 평균에서 2-10사이클 평균을 뺍니다. 평균 계산에는 양의 IR 측정값만 사용합니다.'],
        ['평균 온도', '2-100사이클의 유효한 Tavg 평균으로 초기 온도 조건을 봅니다.'],
        ['평균 충전 시간', '2-100사이클의 유효한 충전 시간 평균을 사용합니다. 배치별로 수명과의 관계가 다를 수 있습니다.'],
        ['Delta Q의 로그 분산', 'log10(Var[Q100(V)-Q10(V)]), ddof=1을 사용해 전압별 용량 변화가 얼마나 퍼져 있는지 봅니다.'],
    ], [128, WIDTH-128], size=8.7, leading=12)
    r.paragraph('기본 7개 변수에 로그 분산을 더한 경우, 중복 변수를 줄인 경우, Delta Q 평균·최솟값으로 바꾼 경우를 비교하겠습니다. '
                '결측값 대체와 표준화 기준은 교차검증의 각 학습 부분에서 구하겠습니다. B2의 저항값이 없는 6개는 최종 학습에 쓰는 개발용 28개의 중앙값으로 채우겠습니다.', size=9, leading=13)
    r.heading('후보 모델과 선택 이유')
    r.table(['후보', '비교하려는 이유'], [
        ['평균 예측 모델', 'DummyRegressor로 평균값만 예측해 다른 모델과 비교할 기준으로 삼겠습니다.'],
        ['선형회귀 / Ridge', '적은 데이터에서 먼저 비교할 모델입니다. 학습 범위 밖 수명도 예측해 보고, Ridge로 중복 변수에 따른 계수 변동을 줄이겠습니다.'],
        ['Random Forest', '직선으로 설명하기 어려운 관계와 변수들의 조합을 살펴보겠습니다. 학습 수명 범위 밖을 예측하지 못하는 한계도 확인하겠습니다.'],
    ], [128, WIDTH-128], size=9, leading=12.5)
    r.heading('검증과 평가 계획')
    r.paragraph(f"<b>데이터 분할:</b> B1을 개발용 {plan['development_n']}개와 별도 검증용 {plan['holdout_n']}개로 고정했습니다(seed=42). "
                f"개발용 데이터에서 3분할 교차검증(CV)으로 모델을 고른 뒤, B2 {plan['test_n']}개로 최종 평가하겠습니다. "
                'B3는 DAY 1 분석에 사용하며, 추가 모델 평가는 선택 사항입니다.', size=9, leading=13)
    r.paragraph('<b>평가 지표:</b> 논문과 비교할 MAPE(%)를 주지표로 쓰겠습니다. 실제 사이클 오차는 MAE·RMSE로 보고, R²도 함께 확인하겠습니다. '
                'DAY 2에서 MAPE 차이(Gap)는 별도 검증-CV, B2-별도 검증, B2-논문 참고값 9.1%로 계산하고 %p로 표시하겠습니다. '
                'DAY 1에서는 분석과 설계까지 진행했습니다.', size=9, leading=13)
    r.paragraph('<b>결과를 볼 때 주의할 점:</b> 세 배치의 분포는 이미 살펴봤지만 B2 점수로 모델을 조정하지는 않겠습니다. '
                'CV는 개발 과정의 점수로 해석하겠습니다. B1·B3의 수명값, 불완전한 기록의 제외, 논문과 다른 데이터 조건도 함께 밝히겠습니다.', size=9, leading=13)
    r.paragraph('<b>출처:</b> Kaggle, itshpark/data-driven-prediction-of-battery-cycle 및 원저자 전처리 코드. '
                'Severson et al. (2019), Nature Energy, DOI: 10.1038/s41560-019-0356-8. '
                'DS Mini Project의 DAY 1 안내와 강의 녹음 내용을 참고했습니다.', size=8, leading=11.5, gap=0, color=GRAY)
    r.save()
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', type=Path, default=ROOT/'results/analysis.json')
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    print(build(json.loads(args.analysis.read_text()), args.output))


if __name__ == '__main__':
    main()
