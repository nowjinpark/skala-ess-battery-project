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

    r.start('문제 정의 · EDA 1', '수명 분포와 배치 차이')
    r.paragraph('<b>회귀 과제:</b> 배터리의 초기 100사이클 정보로 제공된 총수명 cycle_life를 예측한다. '
                '수명 기준은 정격 1.1 Ah의 80%(0.88 Ah)이며, 배터리 셀 1개를 표본 1개로 분석한다.')
    counts = {b['name']: b for b in data['meta']['batches']}
    r.table(['배치', '원본 → 분석', '평균 ± 표준편차', '중앙값', '수명 범위', '<500 n(%)', '>1000 n(%)'],
            [[v['batch'], f"{counts[v['batch']]['raw_cells']} → {v['n']}", f"{v['mean']:.1f} ± {v['std']:.1f}",
              f"{v['median']:.1f}", f"{v['min']:.0f}-{v['max']:.0f}",
              f"{v['short_n']} ({v['short_pct']:.1f})", f"{v['long_n']} ({v['long_pct']:.1f})"] for v in life],
            [52, 68, 110, 53, 85, 70, WIDTH-438], size=8.5, leading=12)
    r.paragraph('수명 단위: 사이클. 원본 139개 중 B1 비완주·후속 기록 필요 10개, B2 수명 결측 8개, '
                'B3 원저자 제외 목록 6개를 분리해 115개를 분석했다. '
                '<b>B1·B3는 제공 근사 종료 라벨, B2는 실제 80% 교차 라벨</b>이며 원본 값은 보존했다. '
                '과제 B2(2018-02-20)는 원저자 B2(2017-06-30)와 달라 고정 인덱스 연결 규칙을 복사하지 않았다.',
                size=9, leading=13.2)

    r.figure('life_distribution', 190, '그림 1. 노션의 150-2300사이클 범위에서 비교한 수명 분포. 단수명 <500, 장수명 >1000.')
    r.finding('관찰', 'B2는 39개 중 28개(71.8%)가 단수명인 반면 B1·B3에는 단수명 표본이 없다. '
              'B3는 40개 중 19개(47.5%)가 장수명이다.')
    r.finding('이상치·원인 가설', '배치별 1.5×IQR 규칙의 하단 이상치는 없다. 상단에는 B2 9개(b2c7 등), B3 4개(b3c38 등)가 있으나 유효 기록으로 유지했다. '
              '최단 사례 b2c19(392), b2c6(393)은 하단 이상치가 아니다. B2의 newstructure 미표기 30개와 표기 9개의 중앙값은 451/904사이클이다. '
              '실험 집단의 차이가 관련될 수 있지만 표기의 물리적 의미는 미확인이므로 원인으로 확정하지 않는다.')
    r.finding('설계 시사점', 'B1으로 학습한 모델은 B2의 짧은 수명으로 예측 범위를 넓혀야 한다. '
              '학습 타깃 범위 밖으로 외삽하지 못하는 Random Forest의 한계를 고려해 선형·Ridge와 비교한다.')

    r.start('EDA 2', '용량 저하와 열화 가속')
    r.figure('capacity_degradation', 350,
             '그림 2. 위: 전체 셀의 용량 궤적. 아래: 중앙수명에 가장 가까운 대표 셀. 검은 점선은 두 직선 근사, 세로선은 근사 전환점이다.')
    r.finding('관찰', esc(data['degradation']['observations'][0]))
    r.finding('Knee 탐색', '대표 셀 b1c11/b2c16/b3c25의 두 직선 근사 전환점은 약 590/340/820사이클이다. '
              '10사이클~제공 수명 관측을 7사이클 중앙값으로 평활화하고 수명의 20~90% 구간에서 잔차제곱합이 가장 작은 전환점을 찾았다. '
              '탐색 범위를 10~90%, 20~95%로 바꿔도 결과는 같았다. 설명용 근사이며 정확한 knee를 검증한 결과는 아니다.')
    r.finding('해석', '초기 기울기가 음수가 아닌 셀은 B1 7/36, B2 22/39, B3 1/40개다. '
              '초기 용량 증가도 있으므로 초기 변화량을 모두 열화량이라고 해석하지 않는다. 후기 기울기 비교 자체가 knee 검출을 의미하지도 않는다.')
    r.finding('설계 시사점', '초기 용량 수준 외에 10-100사이클 용량 변화량과 기울기를 후보로 사용한다. '
              '100사이클 이후 용량, 후반 기울기, knee 위치는 예측 입력에 넣지 않는다.')

    r.start('EDA 3', '전압별 용량 차이와 장·단수명')
    r.figure('deltaq', 340,
             '그림 3. Delta Q(V)=Q100(V)-Q10(V). 곡선의 색·선 모양은 수명 그룹, 산점도의 색·점 모양은 배치를 나타낸다.')
    groups = {(x['batch'], x['life_group']): x for x in data['deltaq']['group_summary']}
    short = groups[('Batch 2', 'short_lt500')]
    long = groups[('Batch 2', 'long_gt1000')]
    groups = [x for x in review['deltaq_group_medians'] if x['life_group'] in ['short','long']]
    r.table(['배치·그룹(n)', 'mean(Delta Q)', 'min(Delta Q)', 'log10 Var(Delta Q)'],
            [[f"{x['batch']} {'단' if x['life_group']=='short' else '장'}({x['n']})", f"{x['deltaq_mean']:.4f}",
              f"{x['deltaq_min']:.4f}", f"{x['log10_deltaq_var']:.2f}"] for x in groups],
            [125, 116, 116, WIDTH-357],size=8.8,leading=12)
    r.paragraph('각 값은 셀별 곡선 통계의 그룹 중앙값이며 평균·최솟값의 단위는 Ah다. B1·B3는 단수명 표본이 없다.',
                size=8.5,leading=12)
    r.finding('관찰·해석', 'B2 단수명은 장수명보다 평균·최소 변화량이 더 음수이고 분산이 크다. '
              'B1·B3의 장수명에서도 작은 분산이 관찰되지만 단수명과 직접 비교할 수는 없다.')
    r.finding('선택 근거', 'B1 개발 28개의 수명 상관은 Delta Q 평균 +0.723, 최소 +0.764, log 분산 -0.782다. '
              '세 통계끼리 |r|=0.966~0.990으로 겹쳐 log 분산을 대표 후보로 두고 평균·최소로 교체한 집합을 CV에서 비교한다. '
              '표본분산(ddof=1)에 log10을 적용하며 작은 상관 차이를 성능 우위로 단정하지 않는다.')

    r.start('EDA 4', '충전 조건과 수명의 관계')
    r.figure('charge_policy', 340,
             '그림 4. 첫 단계 C-rate와 수명 및 배치별 표본 수 상위 3개 프로토콜의 평균. 동률은 프로토콜명 순으로 선정했다(n 표기).')
    r.table(['배치', 'C-rate ↔ 수명 r', 'C-rate ↔ 후반 기울기 r'],
            [[x['batch'], f"{x['c_rate_vs_life_r']:.3f}", f"{x['c_rate_vs_late_slope_r']:.3f}"]
             for x in data['policy']['correlations']], [73, 170, WIDTH-243], size=9)
    r.finding('관찰·해석', 'B1에서 같은 8C라도 전환 SOC 15/25/35%의 평균 수명은 1008.5/676.5/607.5사이클(각 n=2)이다. '
              '첫 단계 전류뿐 아니라 전환 시점·후속 충전 패턴을 함께 봐야 한다. '
              'B3의 높은 C-rate는 더 음수인 후반 기울기와 연관되지만 인과 효과는 아니다. 전체 40개 그룹 중 4개는 n=1이다.')
    r.finding('설계 시사점', '초기 충전 시간 등 실제 관측되는 조건을 후보로 검토한다. '
              '첫 단계 전류만으로 수명을 단정하지 않고 후속 충전 단계·배치 차이와 작은 집단의 불확실성을 함께 해석한다.')

    r.start('EDA 5', '초기 특성의 상관과 중복 정보')
    r.figure('correlation', 230,
             '그림 5. B1 개발 28개에서 계산한 Pearson 상관. 음의 상관은 파랑, 양의 상관은 주황이며 각 칸에 수치를 표시했다.')
    labels = ['초기 용량', '용량 변화량', '용량 기울기', '평균 저항', '저항 변화', '평균 온도', '평균 충전 시간', 'log 분산(Delta Q)']
    fs = [x['name'] for x in data['features']]
    corr = {(x['batch'],x['feature']):x for x in review['batch_correlations']}
    r.table(['전체 배치의 설명용 비교', 'B1 r', 'B2 r', 'B3 r'],
            [[label]+[f"{corr[(f'Batch {i}',feature)]['pearson']:+.3f}" for i in (1,2,3)]
             for label,feature in zip(labels,fs)], [WIDTH-240,80,80,80], size=8.7,leading=11.5)
    r.paragraph('B1 n=36, B3 n=40. B2의 저항 평균·변화는 n=33, 나머지는 n=39. 이 비교는 설명용이며 B2/B3를 모델 튜닝에 쓰지 않는다.',
                size=8.3,leading=11.5)
    r.finding('관찰·해석', '8개 후보 중 절대 상관은 B1·B3에서 log 분산(-0.827/-0.742), B2에서는 충전 시간(-0.917)이 가장 크다. '
              '충전 시간의 부호가 배치별로 바뀌므로 모든 배치에서 같은 관계라고 가정하기 어렵다.')
    r.finding('설계 시사점', f"모델 선택용 B1 개발 28개에서는 log 분산 r={dq['pearson']:.3f}, 용량 변화량-기울기 r={pair['correlation']:.3f}다. "
              '중복 특성을 줄인 집합과 Ridge를 비교하며, 데이터에 따른 최종 선택과 전처리는 B1 개발 구간의 CV 안에서 수행한다.')

    r.start('모델 설계', 'EDA 근거를 특성과 모델에 연결')
    r.heading('핵심 특성 및 전처리')
    r.table(['후보 특성', '계산 정의 / 선정 이유'], [
        ['초기 용량', 'QD(2); 초기 용량 수준을 비교하는 기준 특성.'],
        ['초기 용량 변화량', 'QD(100)-QD(10); 용량 변화의 크기. 증가도 가능하므로 열화량으로 단정하지 않음.'],
        ['초기 용량 기울기', '10-100사이클 QD의 선형 기울기; 변화 속도, 변화량과 중복되어 교체·제외 비교.'],
        ['평균 내부저항', '2-100사이클 양의 IR 평균; 초기 저항 상태.'],
        ['내부저항 변화', '91-100사이클 IR 평균 - 2-10사이클 IR 평균; 초기 저항 변화(양의 값만 사용).'],
        ['평균 온도', '2-100사이클 유효 Tavg 평균; 초기 열적 조건.'],
        ['평균 충전 시간', '2-100사이클 유효 충전 시간 평균; 충전 조건 요약, 배치별 관계 차이에 유의.'],
        ['log 분산(Delta Q)', 'log10(Var[Q100(V)-Q10(V)]), ddof=1; 초기 전압별 변화의 퍼짐을 요약.'],
    ], [128, WIDTH-128], size=8.7, leading=12)
    r.paragraph('기본 7개 → log 분산 추가 8개 → 중복 축소 집합을 같은 분할에서 비교하고 Delta Q 평균·최소로 대체하는 비교도 한다. '
                'B2의 IR 평균·변화 결측 6개는 학습 fold 중앙값으로 대체한다. 표준화도 학습 fold에서만 수행한다.', size=9, leading=13)

    r.heading('후보 모델과 선정 이유')
    r.table(['후보', 'EDA에 근거한 역할'], [
        ['평균 예측 기준선', 'DummyRegressor로 평균만 예측해 다른 모델이 개선하는지 비교.'],
        ['Linear / Ridge', '소표본의 단순 기준선. B2 단수명으로 외삽 가능성을 비교하고, Ridge로 중복 특성에 따른 계수 불안정을 완화.'],
        ['Random Forest', '비선형·특성 간 상호작용 비교. 학습 수명 범위를 벗어나는 예측은 불가능하므로 외삽 한계를 점검.'],
    ], [128, WIDTH-128], size=9, leading=12.5)
    r.heading('검증 설계')
    r.paragraph(f"<b>분할:</b> 배터리 단위로 B1 개발 {plan['development_n']}개 / holdout {plan['holdout_n']}개를 고정"
                f"(seed=42). 개발 구간의 3-fold CV로 후보를 선택하고, B2 {plan['test_n']}개에서 최종 평가한다. "
                'B3는 DAY 1 EDA에만 사용하며 추가 모델 평가는 선택이다.', size=9.2, leading=13.5)
    r.paragraph('<b>평가:</b> 논문의 참고 목표와 비교하는 MAPE(%)를 주지표로, 실제 사이클 오차와 큰 오차를 확인하는 MAE·RMSE 및 R²를 보조로 사용한다. '
                'DAY 2에는 MAPE Gap을 holdout-CV, B2-holdout, B2-논문 참고 9.1%로 계산해 %p로 설명한다. '
                '모델 학습과 성능 산출은 이번 DAY 1 범위에 포함하지 않았다.', size=9.2, leading=13.5)
    r.paragraph('<b>해석 조건:</b> 세 배치의 분포를 EDA에서 관찰했다. B2 점수로 튜닝하지 않으며, '
                '개발 EDA를 보고 후보를 정했으므로 CV는 개발 과정의 점수다. B1·B3 근사 라벨, 불완전 기록 제외 및 논문과 다른 데이터 조건을 평가에 명시한다.',
                size=9.2, leading=13.5)
    r.paragraph('<b>출처:</b> Kaggle, itshpark/data-driven-prediction-of-battery-cycle 및 원저자 공개 전처리 코드. '
                'Severson et al. (2019), Nature Energy, DOI: 10.1038/s41560-019-0356-8. '
                '과제 DS Mini Project의 DAY 1 안내와 제공 강의 녹음본을 기준으로 구성했다.',
                size=8, leading=11.5, gap=0, color=GRAY)
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
