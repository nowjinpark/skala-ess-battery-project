"""Draw the five DAY 1 report figures with accessible color encoding from existing analysis data.

Only writes results/figures_report. Does not change data, splits, analysis or notebooks.
Run: .venv/bin/python src/build_report_figures.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data' / 'processed'
RESULTS = ROOT / 'results'
OUT = RESULTS / 'figures_report'
BATCHES = ['Batch 1', 'Batch 2', 'Batch 3']
BATCH_STYLE = {
    'Batch 1': {'marker': 'o', 'face': '#0072B2', 'hatch': ''},
    'Batch 2': {'marker': '^', 'face': '#D55E00', 'hatch': '///'},
    'Batch 3': {'marker': 's', 'face': '#009E73', 'hatch': 'xx'},
}
FEATURES = [
    'qd_cycle2', 'qd_change_100_10', 'qd_slope_10_100', 'ir_mean_2_100',
    'ir_change_early', 'tavg_mean_2_100', 'chargetime_mean_2_100', 'log10_deltaq_var',
]
FEATURE_LABELS = [
    'QD at cycle 2', 'QD change', 'QD slope', 'IR mean',
    'IR change', 'Mean temperature', 'Charge time', 'log Var(Delta Q)',
]
GROUP_STYLE = {
    'short': {'color': '#A23B72', 'linestyle': '-', 'linewidth': .8},
    'middle': {'color': '#A8B0B8', 'linestyle': '-', 'linewidth': .65},
    'long': {'color': '#007A87', 'linestyle': (0, (4, 2.5)), 'linewidth': .8},
}


def life_group(life):
    return 'short' if life < 500 else 'long' if life > 1000 else 'middle'


def scatter_batches(ax, eligible, x, y, size=23):
    for batch in BATCHES:
        g = eligible.loc[eligible.batch.eq(batch)]
        style = BATCH_STYLE[batch]
        ax.scatter(g[x], g[y], marker=style['marker'], facecolors=style['face'],
                   edgecolors='black', linewidths=.65, s=size, label=batch, zorder=3)


def save(fig, name):
    # Fixed canvas sizes keep every figure below 360 pt at its nominal print size.
    path = OUT / f'{name}.png'
    fig.savefig(path, dpi=220, facecolor='white')
    width, height = fig.get_size_inches()
    plt.close(fig)
    return {'file': str(path.relative_to(ROOT)), 'width_pt': round(width * 72, 1),
            'height_pt': round(height * 72, 1)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9.4, 'axes.titlesize': 10.5,
        'axes.labelsize': 9.6, 'xtick.labelsize': 8.6, 'ytick.labelsize': 8.6,
        'legend.fontsize': 8.1, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '0.2', 'axes.labelcolor': 'black', 'text.color': 'black',
        'xtick.color': 'black', 'ytick.color': 'black', 'grid.color': '0.88',
        'grid.linewidth': .5, 'axes.axisbelow': True, 'hatch.linewidth': .45,
        'figure.constrained_layout.h_pad': .045, 'figure.constrained_layout.w_pad': .045,
    })
    cells = pd.read_csv(PROC / 'cells_analysis.csv')
    eligible = cells.loc[cells.analysis_eligible].copy()
    dev = eligible.loc[eligible.split_role.eq('Batch1_CV_development')].copy()
    assert eligible.groupby('batch').size().to_dict() == {'Batch 1': 36, 'Batch 2': 39, 'Batch 3': 40}
    assert len(dev) == 28 and dev.batch.eq('Batch 1').all()
    summary = pd.concat([pd.read_csv(PROC / f'batch{i}_summary.csv.gz') for i in (1, 2, 3)], ignore_index=True)
    curves = {}
    for i in (1, 2, 3):
        with np.load(PROC / f'batch{i}_curves.npz', allow_pickle=False) as archive:
            curves.update({k: archive[k].copy() for k in archive.files})
    manifest = {}
    review = json.loads((RESULTS / 'report_review.json').read_text())

    # 1. Eligible supplied lifetimes: same bins, same axes, one batch per panel.
    upper = 2300
    bins = np.arange(150, 2401, 150)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharex=True, sharey=True, layout='constrained')
    for ax, batch in zip(axes, BATCHES):
        life = eligible.loc[eligible.batch.eq(batch), 'cycle_life']
        ax.hist(life, bins=bins, color=BATCH_STYLE[batch]['face'], alpha=.8, edgecolor='black', linewidth=.7,
                hatch=BATCH_STYLE[batch]['hatch'])
        ax.axvline(500, color='0.25', linestyle=':', linewidth=.9)
        ax.axvline(1000, color='0.25', linestyle='--', linewidth=.9)
        ax.set(title=f'{batch} (n={len(life)})\nmedian = {life.median():.1f}',
               xlabel='Cycle life', xlim=(150, upper), xticks=[500, 1500, 2300])
        ax.grid(axis='y')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    axes[0].set_ylabel('Cell records')
    manifest['life_distribution'] = save(fig, 'life_distribution')

    # 2. Observed QD trajectories, smoothed for display only; never used as X.
    retained = summary.loc[summary.capacity_plausible & summary.cycle.ge(2)
                           & summary.cell_id.isin(eligible.cell_id)].copy()
    lives = eligible.set_index('cell_id').cycle_life
    fig, grid = plt.subplots(2, 3, figsize=(7.2, 4.9), sharey=True, layout='constrained')
    axes = grid[0]
    for ax, batch in zip(axes, BATCHES):
        records = list(retained.loc[retained.batch.eq(batch)].groupby('cell_id'))
        records.sort(key=lambda item: life_group(lives.loc[item[0]]) != 'middle')
        for cid, group in records:
            group = group.sort_values('cycle')
            smoothed = group.QDischarge.rolling(7, center=True, min_periods=1).median()
            ax.plot(group.cycle, smoothed, alpha=.72, **GROUP_STYLE[life_group(lives.loc[cid])])
        ax.axhline(.88, color='black', linestyle=':', linewidth=1)
        ax.axvspan(0, 100, facecolor='#EDF3F7', edgecolor='0.6', hatch='////', linewidth=0, zorder=0)
        ax.set(title=f'{batch} (n={len(records)})', xlabel='Cycle',
               xlim=(0, upper + 75), ylim=(.8, 1.15), xticks=[0, 1000, 2000])
        ax.grid(axis='y')
    axes[0].set_ylabel('Discharge capacity (Ah)')
    for ax, item in zip(grid[1], review['representative_knees']):
        g = retained.loc[retained.cell_id.eq(item['cell_id']) & retained.cycle.between(10, item['cycle_life'])].sort_values('cycle')
        x = g.cycle.to_numpy()
        y = g.QDischarge.rolling(7, center=True, min_periods=1).median().to_numpy()
        a, b, c = item['beta']
        fit = a+b*x+c*np.maximum(0,x-item['cycle'])
        ax.plot(x, y, color=BATCH_STYLE[item['batch']]['face'], linewidth=1.2)
        ax.plot(x, fit, color='black', linestyle='--', linewidth=.8)
        ax.axvline(item['cycle'], color='0.25', linestyle=':', linewidth=.8)
        ax.axhline(.88, color='0.5', linestyle=':', linewidth=.6)
        rounded = int(round(item['cycle']/10)*10)
        ax.set(title=f"{item['cell_id']}: approx. {rounded} cycles", xlabel='Cycle', ylim=(.8,1.15))
        ax.grid(axis='y')
    grid[1,0].set_ylabel('Representative capacity (Ah)')
    legend = [Line2D([], [], label='Short <500', **GROUP_STYLE['short']),
              Line2D([], [], label='Middle 500-1000', **GROUP_STYLE['middle']),
              Line2D([], [], label='Long >1000', **GROUP_STYLE['long']),
              Line2D([], [], color='black', linestyle=':', label='0.88 Ah reference'),
              Patch(facecolor='#EDF3F7', edgecolor='0.6', hatch='////', label='First 100 cycles')]
    fig.legend(handles=legend, loc='outside lower center', ncol=3, frameon=False,
               handlelength=2.8, columnspacing=1.2)
    manifest['capacity_degradation'] = save(fig, 'capacity_degradation')

    # 3. Lifetime-group differences; the fourth panel is descriptive across batches.
    all_delta = np.concatenate([curves[f'{cid}_deltaq'] for cid in eligible.cell_id])
    lower, higher = float(all_delta.min()), float(all_delta.max())
    delta_pad = (higher - lower) * .08
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.95), layout='constrained')
    for ax, batch in zip(axes.flat, BATCHES):
        g = eligible.loc[eligible.batch.eq(batch)].copy()
        g['_layer'] = g.cycle_life.map(lambda v: 0 if life_group(v) == 'middle' else 1)
        for row in g.sort_values('_layer').itertuples():
            ax.plot(curves[f'{row.cell_id}_voltage'], curves[f'{row.cell_id}_deltaq'],
                    alpha=.7, **GROUP_STYLE[life_group(row.cycle_life)])
        handles = []
        for group, label, count in [
                ('short', 'Short <500', g.cycle_life.lt(500).sum()),
                ('middle', 'Middle 500-1000', g.cycle_life.between(500, 1000).sum()),
                ('long', 'Long >1000', g.cycle_life.gt(1000).sum())]:
            handles.append(Line2D([], [], label=f'{label} (n={count})', **GROUP_STYLE[group]))
        ax.legend(handles=handles, loc='lower left', framealpha=.96, borderpad=.35,
                  handlelength=2.6, fontsize=7.6)
        ax.axhline(0, color='0.55', linewidth=.6)
        ax.set(title=batch, xlabel='Voltage (V)', ylabel='Q100(V) - Q10(V) (Ah)',
               xlim=(1.95, 3.55), ylim=(lower - delta_pad, higher + delta_pad))
    scatter_batches(axes[1, 1], eligible, 'log10_deltaq_var', 'cycle_life', size=18)
    axes[1, 1].set(title='Descriptive comparison', xlabel='log10 Var[Delta Q(V)]', ylabel='Cycle life')
    axes[1, 1].legend(loc='upper right', framealpha=.96)
    axes[1, 1].grid(linewidth=.4)
    manifest['deltaq'] = save(fig, 'deltaq')

    # 4. Balanced descriptive coverage: 3 high-count protocol groups per batch.
    policy = eligible.groupby(['batch', 'policy'], dropna=False).agg(
        n=('cell_id', 'size'), mean_life=('cycle_life', 'mean')).reset_index()
    top = pd.concat([g.sort_values(['n','policy'], ascending=[False,True]).head(3) for _,g in policy.groupby('batch')]).sort_values(['batch','mean_life'])
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.95), layout='constrained',
                             gridspec_kw={'height_ratios': [1, 1.65]})
    scatter_batches(axes[0], eligible, 'c_rate_stage1', 'cycle_life', size=21)
    axes[0].set(title='First charging stage and supplied lifetime',
                xlabel='First-stage C-rate', ylabel='Cycle life')
    axes[0].legend(ncol=3, loc='upper right', framealpha=.95)
    axes[0].grid(linewidth=.4)
    labels = []
    for i, row in enumerate(top.itertuples()):
        protocol = str(row.policy).replace('-newstructure', ' [new]')
        labels.append(f'B{row.batch[-1]} | {protocol} | n={row.n}')
        axes[1].barh(i, row.mean_life, color=BATCH_STYLE[row.batch]['face'], alpha=.8, edgecolor='black', linewidth=.6,
                     hatch=BATCH_STYLE[row.batch]['hatch'], height=.72)
        axes[1].text(row.mean_life + 25, i, f'{row.mean_life:.0f}', va='center', fontsize=7.9)
    axes[1].set_yticks(range(len(top)), labels)
    axes[1].tick_params(axis='y', labelsize=7.9)
    axes[1].set(title='3 groups per batch, ranked by n ([new] = newstructure)',
                xlabel='Mean cycle life', xlim=(0, top.mean_life.max() * 1.13))
    axes[1].grid(axis='x')
    manifest['charge_policy'] = save(fig, 'charge_policy')

    # 5. No Batch 2/3 or B1 holdout enters model-directed correlations.
    pearson = dev[FEATURES + ['cycle_life']].corr()['cycle_life'].drop('cycle_life')
    matrix = dev[FEATURES].corr(min_periods=10)
    ids = {feature: f'F{i + 1}' for i, feature in enumerate(FEATURES)}
    labels = {feature: f'{ids[feature]}  {label}' for feature, label in zip(FEATURES, FEATURE_LABELS)}
    ordered = pearson.sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9), layout='constrained',
                             gridspec_kw={'width_ratios': [1, 1.25]})
    for i, (feature, value) in enumerate(ordered.items()):
        axes[0].barh(i, value, color='#D55E00' if value >= 0 else '#0072B2',
                     edgecolor='black', linewidth=.6, hatch='' if value >= 0 else '///')
        axes[0].text(value + (.045 if value >= 0 else -.045), i, f'{value:+.2f}',
                     ha='left' if value >= 0 else 'right', va='center', fontsize=8)
    axes[0].set_yticks(range(len(ordered)), [labels[x] for x in ordered.index])
    axes[0].tick_params(axis='y', labelsize=8.1)
    axes[0].axvline(0, color='black', linewidth=.6)
    axes[0].set(title='Feature vs. life', xlabel='Pearson r', xlim=(-1.15, 1.05))
    axes[0].set_xticks([-1, 0, 1])
    axes[0].grid(axis='x')
    cmap = LinearSegmentedColormap.from_list('signed_correlation', ['#00649D', '#FFFFFF', '#C35400'])
    axes[1].imshow(matrix, cmap=cmap, vmin=-1, vmax=1, aspect='equal')
    axes[1].set_xticks(range(8), [ids[x] for x in FEATURES])
    axes[1].set_yticks(range(8), [ids[x] for x in FEATURES])
    axes[1].tick_params(length=0, labelsize=8.2)
    axes[1].set_title('Feature correlations')
    for i in range(8):
        for j in range(8):
            value = matrix.iloc[i, j]
            axes[1].text(j, i, f'{value:+.2f}', ha='center', va='center', fontsize=7.5,
                         color='white' if abs(value) > .6 else 'black')
    axes[1].set_xticks(np.arange(-.5, 8, 1), minor=True)
    axes[1].set_yticks(np.arange(-.5, 8, 1), minor=True)
    axes[1].grid(which='minor', color='white', linewidth=.6)
    axes[1].tick_params(which='minor', bottom=False, left=False)
    fig.suptitle('Batch 1 development only (n=28)', fontsize=11)
    manifest['correlation'] = save(fig, 'correlation')

    (OUT / 'manifest.json').write_text(json.dumps({
        'scope': 'Five report figures with color, line styles and markers; no model fitting',
        'analysis_cells': len(eligible), 'development_cells_for_correlation': len(dev),
        'figures': manifest,
    }, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
