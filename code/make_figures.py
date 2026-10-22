"""Generate the manuscript figures from the result registries.

Usage:
  python make_figures.py            # all figures
  python make_figures.py fig4 fig7  # selected figures
"""

import csv
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 8,
    'axes.spines.top': True, 'axes.spines.right': True,
    'figure.dpi': 120, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
RESULTS = os.path.join(ROOT, 'results')
PREDICTIONS = os.path.join(ROOT, 'predictions')
FIGURES = os.path.join(ROOT, 'figures')
os.makedirs(FIGURES, exist_ok=True)

BLUE, ORANGE, GREEN, RED, GRAY, PURPLE = (
    '#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8C8C8C', '#8172B3')
LIGHTGRAY = '#d9d9d9'


def load(name):
    return json.load(open(os.path.join(RESULTS, name)))


def seed_avg_preds(tag='main', name='TempAlignEEGNet'):
    files = sorted(glob.glob(os.path.join(
        PREDICTIONS, f'preds_{tag}_{name}_f*_s*.npz')))
    if not files:
        return None
    per_seed = {}
    for f in files:
        seed = int(f.split('_s')[1].split('.')[0])
        per_seed.setdefault(seed, []).append(f)
    seeds = sorted(per_seed)
    merged = {}
    for s in seeds:
        arrs = [np.load(f) for f in sorted(per_seed[s])]
        merged[s] = {k: np.concatenate([a[k] for a in arrs])
                     for k in ('pids', 'targets', 'preds')}
    ref = merged[seeds[0]]
    preds = np.mean([merged[s]['preds'] for s in seeds], axis=0)
    return {'pids': ref['pids'], 'targets': ref['targets'], 'preds': preds}


def panel_caption(ax, text, y=-0.28):
    ax.text(0.5, y, text, transform=ax.transAxes, ha='center', va='top',
            fontsize=8)


def fig1():
    rows = list(csv.DictReader(open(os.path.join(
        RESULTS, 'participant_demographics.csv'))))
    groups = [('ADHD', 'yes', '#2CA02C'), ('Control', 'no', '#FF7F0E'),
              ('Undetermined', 'undetermined', '#7F7F7F')]
    counts = [sum(1 for r in rows if r['diagnosed'] == g)
              for _, g, _ in groups]
    ages = [[float(r['age']) for r in rows if r['diagnosed'] == g]
            for _, g, _ in groups]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6),
                             gridspec_kw={'width_ratios': [1.1, 1]})
    tick = ['ADHD', 'Control', 'Undet.']
    axes[0].bar(range(3), counts, color=[c for _, _, c in groups])
    for i, c in enumerate(counts):
        axes[0].text(i, c + 1, str(c), ha='center', fontsize=8)
    axes[0].set_xticks(range(3))
    axes[0].set_xticklabels(tick, fontsize=6.5, rotation=30, ha='right')
    axes[0].set_ylabel('Participants')
    axes[0].set_xlabel(f'(a) Diagnostic group (n={sum(counts)})', fontsize=7)

    bp = axes[1].boxplot(ages, patch_artist=True, widths=0.5)
    for patch, (_, _, c) in zip(bp['boxes'], groups):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    axes[1].set_xticks(range(1, 4))
    axes[1].set_xticklabels(tick, fontsize=6.5, rotation=30, ha='right')
    axes[1].set_ylabel('Age (years)')
    axes[1].set_xlabel('(b) Age by group', fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'Fig1.pdf'))
    plt.close(fig)


def fig2():
    src = os.path.join(FIGURES, 'Fig2_source.jpg')
    if os.path.exists(src):
        im = plt.imread(src)
        h, w = im.shape[:2]
        dpi = 300
        fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(im, interpolation='none')
        ax.set_axis_off()
        fig.savefig(os.path.join(FIGURES, 'Fig2.pdf'), pad_inches=0)
        plt.close(fig)
        return

    blue, orange = '#4C72B0', '#DD8452'
    gray_fill, gray_edge = '#f2f2f2', '#9a9a9a'

    def panel(ax, x, y, w, h, fc=gray_fill):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.06',
            fc=fc, ec=gray_edge, lw=1.1))

    def wbox(ax, x, y, w, h, text, ec='#444444', fs=7.2, lw=1.2, fc='white'):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle='round,pad=0.015,rounding_size=0.05',
            fc=fc, ec=ec, lw=lw))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=fs)

    def arrow(ax, p1, p2, color='k', lw=1.1, ms=11):
        ax.add_patch(FancyArrowPatch(
            p1, p2, arrowstyle='-|>', color=color, lw=lw, mutation_scale=ms,
            shrinkA=0, shrinkB=0))

    def squiggles(ax, x, y, w, h, n=8, seed=0):
        rng = np.random.RandomState(seed)
        xs = np.linspace(x + 0.05, x + w - 0.05, 120)
        for i in range(n):
            base = y + h * (i + 0.6) / (n + 0.2)
            amp = h * 0.035
            sig = (amp * np.sin(np.linspace(0, 8 * np.pi, 120)
                                + rng.rand() * 6)
                   + amp * 0.6 * np.sin(np.linspace(0, 21 * np.pi, 120)
                                        + rng.rand() * 6)
                   + rng.randn(120) * amp * 0.25)
            ax.plot(xs, base + sig, color='#333333', lw=0.5, zorder=3)

    def gaze_dots(ax, x, y, w, h, seed=1):
        rng = np.random.RandomState(seed)
        cx, cy = x + w * 0.5, y + h * 0.5
        ax.scatter(cx + rng.randn(45) * w * 0.13,
                   cy + rng.randn(45) * h * 0.16,
                   s=4, color=blue, edgecolor='none', zorder=3)

    def mask_icon(ax, x, y, s, n=5):
        cell = s / n
        for r in range(n):
            for c in range(n):
                fc = '#555555' if c <= r else 'white'
                ax.add_patch(Rectangle(
                    (x + c * cell, y + (n - 1 - r) * cell), cell, cell,
                    fc=fc, ec='#bbbbbb', lw=0.4, zorder=3))

    def xy_icon(ax, x, y, w, h):
        ax.add_patch(Rectangle((x, y), w, h, fc='white', ec='#444444', lw=1.0))
        for k in range(1, 4):
            ax.plot([x + k * w / 4, x + k * w / 4], [y, y + h],
                    color='#cccccc', lw=0.5)
            ax.plot([x, x + w], [y + k * h / 4, y + k * h / 4],
                    color='#cccccc', lw=0.5)
        arrow(ax, (x + 0.12 * w, y + 0.25 * h), (x + 0.75 * w, y + 0.7 * h),
              lw=0.9, ms=8)

    fig, ax = plt.subplots(figsize=(7.9, 4.0))
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 6.2)
    ax.axis('off')

    ax.text(6.0, 5.85, 'CausalEEGNet', ha='center', va='center',
            fontsize=13, weight='bold')

    ax.text(0.85, 5.42, 'EEG Input', ha='center', fontsize=8.5)
    ax.add_patch(Rectangle((0.15, 3.55), 1.4, 1.65, fc='white',
                           ec='#444444', lw=1.0))
    squiggles(ax, 0.15, 3.55, 1.4, 1.65)
    arrow(ax, (1.55, 4.35), (2.02, 4.35))

    panel(ax, 2.02, 3.30, 2.55, 2.20)
    ax.text(3.30, 5.28, 'Multi-Scale EEG Encoder', ha='center', fontsize=7.8)
    wbox(ax, 2.14, 4.55, 1.36, 0.36, 'Conv1D (k=3)', fs=6.8)
    wbox(ax, 2.14, 4.08, 1.36, 0.36, 'Conv1D (k=7)', fs=6.8)
    wbox(ax, 2.14, 3.61, 1.36, 0.36, 'Conv1D (k=15)', fs=6.8)
    ax.add_patch(FancyBboxPatch(
        (3.55, 4.02), 0.40, 1.02, boxstyle='round,pad=0.015,rounding_size=0.05',
        fc='white', ec='#444444', lw=1.1))
    ax.text(3.75, 4.53, 'Concat', ha='center', va='center', fontsize=6.4,
            rotation=90)
    for yy in (4.73, 4.26, 3.79):
        arrow(ax, (3.50, yy), (3.55, yy), lw=0.9, ms=7)
    wbox(ax, 4.01, 4.26, 0.55, 0.48, 'Linear', fs=6.0)
    arrow(ax, (3.95, 4.53), (4.01, 4.50), lw=0.9, ms=7)
    ax.text(4.74, 4.70, r'$X_{\mathrm{EEG}}$', fontsize=9, style='italic')

    ax.text(0.85, 3.02, 'Eye-tracking Input', ha='center', fontsize=8.5)
    ax.text(0.85, 1.02, '50 resampled points', ha='center', va='top',
            fontsize=6.2)
    ax.add_patch(Rectangle((0.15, 1.15), 1.4, 1.55, fc='white',
                           ec='#444444', lw=1.0))
    gaze_dots(ax, 0.15, 1.15, 1.4, 1.55)
    arrow(ax, (1.55, 1.92), (2.02, 1.92))

    panel(ax, 2.02, 1.20, 2.55, 1.48)
    ax.text(3.30, 2.48, 'Gaze Encoder', ha='center', fontsize=7.8)
    wbox(ax, 2.16, 1.32, 2.28, 0.92,
         'Linear $\\rightarrow$ GRU\n(2-layer, unidirectional)')
    arrow(ax, (4.57, 1.92), (4.62, 1.92))
    ax.text(4.64, 2.16, r'$X_{\mathrm{gaze}}$', fontsize=9, style='italic')

    panel(ax, 5.55, 1.40, 2.25, 2.55)
    ax.text(6.68, 3.82, 'Temporally Constrained\nCross-Attention',
            ha='center', va='center', fontsize=7.2)
    ax.text(6.68, 3.40, '($\\times$2 layers)', ha='center', fontsize=7.0)
    ax.text(5.68, 3.12, 'Q', fontsize=8, color=blue)
    ax.text(5.68, 2.38, 'K', fontsize=8, color=orange)
    ax.text(5.68, 2.02, 'V', fontsize=8, color=orange)
    mask_icon(ax, 6.10, 1.72, 0.86)
    ax.text(6.68, 1.56, '$M$ (temporal mask)', ha='center', fontsize=7.0)

    ax.add_patch(FancyArrowPatch(
        (4.56, 4.50), (5.62, 3.20), arrowstyle='-|>', color=blue, lw=1.0,
        mutation_scale=11, connectionstyle='angle,angleA=0,angleB=90,rad=0',
        shrinkA=0, shrinkB=0))
    arrow(ax, (4.95, 1.92), (5.55, 2.38), color=orange, lw=1.0)
    arrow(ax, (4.95, 1.92), (5.55, 2.02), color=orange, lw=1.0)

    arrow(ax, (7.80, 2.75), (7.98, 2.75))
    panel(ax, 7.98, 2.05, 0.82, 1.40)
    ax.text(8.39, 2.75, 'Temporal\nAttention\nPooling', ha='center',
            va='center', fontsize=6.8, rotation=90)
    arrow(ax, (8.80, 2.75), (8.98, 2.75))
    panel(ax, 8.98, 2.20, 0.62, 1.05)
    ax.text(9.29, 2.72, 'MLP', ha='center', va='center', fontsize=7.4,
            rotation=90)

    arrow(ax, (9.60, 2.95), (9.90, 3.10))
    arrow(ax, (9.60, 2.55), (9.90, 2.55))
    ax.text(9.95, 3.10, '$\\Delta$col', fontsize=7.6, va='center')
    ax.text(9.95, 2.55, '$\\Delta$row', fontsize=7.6, va='center')
    arrow(ax, (10.55, 3.05), (10.78, 2.95))
    arrow(ax, (10.55, 2.60), (10.78, 2.60))
    xy_icon(ax, 10.80, 2.15, 1.00, 1.00)
    ax.text(11.30, 3.22, '(x,y)', ha='center', fontsize=7.0)

    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(FIGURES, 'Fig2.pdf'))
    plt.close(fig)


def fig3():
    d = load('training_history.json')
    h = d['history']
    ep = np.arange(1, len(h['train_loss']) + 1)
    best = d['best_epoch']

    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.2))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.30,
                        wspace=0.38)

    ax = axes[0]
    ax.plot(ep, h['train_loss'], color=BLUE, lw=1.3, label='Training')
    ax.plot(ep, h['inner_loss'], color=ORANGE, lw=1.3, label='Inner validation')
    ax.axvline(best, color=GRAY, ls=':', lw=0.9)
    ax.set_ylabel('Huber loss')
    ax.legend(fontsize=6.5, loc='center left', bbox_to_anchor=(0.01, 0.56),
              frameon=True, framealpha=0.9, edgecolor='none')
    panel_caption(ax, '(a) Training and validation loss', y=-0.30)

    ax = axes[1]
    ax.plot(ep, h['inner_rmse'], color=ORANGE, lw=1.3)
    ax.axvline(best, color=GRAY, ls=':', lw=0.9)
    ax.set_ylabel('RMSE (z units)')
    panel_caption(ax, '(b) Validation RMSE', y=-0.30)

    ax = axes[2]
    ax.plot(ep, h['inner_corr_col'], color=BLUE, lw=1.3, label=r'$\Delta$col')
    ax.plot(ep, h['inner_corr_row'], color=ORANGE, lw=1.3, label=r'$\Delta$row')
    ax.axvline(best, color=GRAY, ls=':', lw=0.9)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylabel('Pearson $r$')
    ax.legend(fontsize=6.0, loc='lower left', frameon=False)
    panel_caption(ax, '(c) Prediction correlation over epochs', y=-0.30)

    for ax in axes:
        ax.set_xlim(0, len(ep) + 1)
        ax.set_xlabel('Epoch', fontsize=7)
        ax.tick_params(labelsize=6.5, pad=2)
        ax.grid(lw=0.3, alpha=0.35)
        ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(True)

    fig.savefig(os.path.join(FIGURES, 'Fig3.pdf'))
    plt.close(fig)


def model_stats():
    s = load('summary.json')
    order = ['TempAlignEEGNet', 'GazeOnly', 'EEGOnly', 'LearnedMaskNet',
             'LSTM', 'MaskedLSTM', 'Transformer', 'MaskedTransformer', 'LTC',
             'WaveletTransformer', 'RidgeGaze', 'PersistenceTrend',
             'MeanPredictor']
    labels = {'LSTM': 'LSTM', 'MaskedLSTM': 'M-LSTM',
              'Transformer': 'Transformer',
              'MaskedTransformer': 'M-Transf.',
              'LTC': 'LTC', 'WaveletTransformer': 'Wavelet',
              'GazeOnly': 'Gaze', 'EEGOnly': 'EEG',
              'TempAlignEEGNet': 'CausalEEGNet',
              'LearnedMaskNet': 'L-mask', 'RidgeGaze': 'Ridge',
              'PersistenceTrend': 'Trend', 'MeanPredictor': 'Mean'}
    names = [labels[n] for n in order if n in s['models']]
    rmse = [s['models'][n]['rmse_mean'] for n in order if n in s['models']]
    rmse_ci = [s['models'][n]['rmse_ci95'] for n in order if n in s['models']]
    rc = [s['models'][n]['r_col_mean'] for n in order if n in s['models']]
    rc_ci = [s['models'][n]['r_col_ci95'] for n in order if n in s['models']]
    rr = [s['models'][n]['r_row_mean'] for n in order if n in s['models']]
    rr_ci = [s['models'][n]['r_row_ci95'] for n in order if n in s['models']]
    colors = [GREEN, BLUE, PURPLE, '#F2C14E', GRAY, '#9DC3E6', GRAY,
              '#C5B4E3', ORANGE, RED, GRAY, GRAY, LIGHTGRAY][:len(names)]
    return names, rmse, rmse_ci, rc, rc_ci, rr, rr_ci, colors


def bars3(axs, names, series, colors, ylabels, captions, value_labels):
    x = np.arange(len(names))
    for ax, (vals, cis), ylab, cap, show_lab in zip(
            axs, series, ylabels, captions, value_labels):
        ax.bar(x, vals, yerr=cis, color=colors, edgecolor='k', lw=0.5,
               capsize=2.5, width=0.78,
               error_kw={'lw': 0.8, 'capthick': 0.8})
        lo = min(0.0, min(v - e for v, e in zip(vals, cis)))
        hi = max(v + e for v, e in zip(vals, cis))
        span = max(hi - lo, 1e-3)
        ax.set_ylim(lo - 0.08 * span, hi + 0.30 * span)
        if show_lab:
            for i, (v, e) in enumerate(zip(vals, cis)):
                dy = 3 if i % 2 == 0 else 11
                ax.annotate(f'{v:.3f}', xy=(i, hi), xytext=(0, dy),
                            textcoords='offset points', ha='center',
                            va='bottom', fontsize=5.0, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=6.4, rotation=45, ha='right')
        ax.set_ylabel(ylab, fontsize=8.5)
        ax.set_xlabel(cap, fontsize=9, labelpad=6)
        ax.grid(axis='y', lw=0.35, alpha=0.4)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7.5, pad=2)


def fig4():
    names, rmse, rmse_ci, rc, rc_ci, rr, rr_ci, colors = model_stats()
    fig = plt.figure(figsize=(7.0, 5.3))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 0.95], hspace=1.05,
                          wspace=0.9, left=0.06, right=0.99, top=0.95,
                          bottom=0.10)
    axs = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]),
           fig.add_subplot(gs[0, 4:6])]
    bars3(axs, names, [(rmse, rmse_ci), (rc, rc_ci), (rr, rr_ci)], colors,
          ['RMSE', 'Pearson $r$', 'Pearson $r$'],
          ['(a) Prediction RMSE', r'(b) $\Delta$col correlation',
           r'(c) $\Delta$row correlation'],
          [True, True, True])

    sa = seed_avg_preds('main', 'TempAlignEEGNet')
    if sa is not None:
        t, p = sa['targets'], sa['preds']
        pooled = load('summary.json')['models']['TempAlignEEGNet']['pooled']
        for k, (span, lab, tag) in enumerate(
                [(slice(0, 3), 'col', 'd'), (slice(3, 6), 'row', 'e')]):
            ax = fig.add_subplot(gs[1, span])
            q = np.percentile(np.abs(t[:, k]), 99.5)
            m = np.abs(t[:, k]) <= q
            qy = np.percentile(np.abs(p[m, k]), 99.5)
            qmax = max(q, qy)
            ax.scatter(t[m, k], p[m, k], s=2.4, alpha=0.30, color='0.55',
                       edgecolors='none', rasterized=True)
            ax.axhline(0, color='0.75', lw=0.6, zorder=0)
            ax.plot([-qmax, qmax], [-qmax, qmax], 'r--', lw=1.0,
                    label='Perfect prediction')
            ax.set_xlim(-qmax, qmax)
            ax.set_ylim(-qmax, qmax)
            rk = 'r_col' if lab == 'col' else 'r_row'
            r2k = 'R2_col' if lab == 'col' else 'R2_row'
            ax.set_title(f"$\\Delta${lab}: $r$ = {pooled[rk]:.3f}, "
                         f"$R^2$ = {pooled[r2k]:.3f}", fontsize=8)
            ax.grid(lw=0.3, alpha=0.35)
            ax.set_axisbelow(True)
            ax.legend(fontsize=6, loc='upper left', frameon=True,
                      handlelength=1.4)
            ax.set_xlabel(f'Actual $\\Delta${lab} (z)\n'
                          + f'({tag}) ' + ('Horizontal' if lab == 'col'
                                           else 'Vertical')
                          + ' held-out predictions', fontsize=8)
            ax.set_ylabel(rf'Predicted $\Delta${lab} (z)', fontsize=8.5)
            ax.tick_params(labelsize=7.5, pad=2)

    fig.savefig(os.path.join(FIGURES, 'Fig4.pdf'))
    plt.close(fig)


def fig5():
    abl = load('results_ablation.json')
    ver = load('mask_verification.json')
    order = [('TempAlignEEGNet', 'Full model'), ('NoMask', 'No mask'),
             ('FrontalExcluded', 'Frontal\nremoved'),
             ('TempStructNet', 'Learned\nmask')]
    names, rmse, rmse_ci, rc, rc_ci, rr, rr_ci = [], [], [], [], [], [], []
    summ = load('summary.json')['models']['TempAlignEEGNet']
    for key, lab in order:
        if key == 'TempAlignEEGNet':
            names.append(lab)
            rmse.append(summ['rmse_mean'])
            rmse_ci.append(summ['rmse_ci95'])
            rc.append(summ['r_col_mean'])
            rc_ci.append(summ['r_col_ci95'])
            rr.append(summ['r_row_mean'])
            rr_ci.append(summ['r_row_ci95'])
            continue
        runs = abl.get(key, {}).get('runs', [])
        if not runs:
            continue
        names.append(lab)
        n = len(runs)
        rmse.append(np.mean([r['rmse'] for r in runs]))
        rmse_ci.append(1.96 * np.std([r['rmse'] for r in runs], ddof=1)
                       / np.sqrt(n))
        rc.append(np.mean([r['r_col'] for r in runs]))
        rc_ci.append(1.96 * np.std([r['r_col'] for r in runs], ddof=1)
                     / np.sqrt(n))
        rr.append(np.mean([r['r_row'] for r in runs]))
        rr_ci.append(1.96 * np.std([r['r_row'] for r in runs], ddof=1)
                     / np.sqrt(n))
    colors = [RED, BLUE, '#2CA25F', '#B0B0B0'][:len(names)]

    fig = plt.figure(figsize=(7.0, 4.7))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.0], hspace=0.7,
                          wspace=0.9, left=0.06, right=0.99, top=0.95,
                          bottom=0.10)
    axs = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]),
           fig.add_subplot(gs[0, 4:6])]
    bars3(axs, names, [(rmse, rmse_ci), (rc, rc_ci), (rr, rr_ci)], colors,
          ['RMSE', 'Pearson $r$', 'Pearson $r$'],
          ['(a) RMSE', r'(b) $\Delta$col correlation',
           r'(c) $\Delta$row correlation'],
          [True, True, True])

    T, S = 256, 50
    i = np.arange(T)[:, None]
    j = np.arange(S)[None, :]
    allowed = (j <= np.floor(i * S / T)).astype(float)
    d = ver['learned_mask']
    soft = 1.0 / (1.0 + np.exp(-(((i / T - d['delay_normalized']) - (j / S))
                                 / d['bandwidth'])))
    ax = fig.add_subplot(gs[1, 0:3])
    im = ax.imshow(allowed, origin='lower', aspect='auto', cmap='Blues',
                   extent=[0, 30, 0, 30], vmin=0, vmax=1.2)
    ax.plot([0, 30], [0, 30], 'r--', lw=0.9)
    ax.set_xlabel('Gaze position $j$ (normalized)\n'
                  '(d) Fixed temporal mask', fontsize=8)
    ax.set_ylabel('EEG position $i$ (normalized)', fontsize=8.5)
    ax.tick_params(labelsize=7.5, pad=2)

    ax = fig.add_subplot(gs[1, 3:6])
    im = ax.imshow(soft, origin='lower', aspect='auto', cmap='Greens',
                   extent=[0, 30, 0, 30], vmin=0, vmax=1)
    ax.plot([0, 30], [0, 30], 'r--', lw=0.9)
    ax.set_xlabel('Gaze position $j$ (normalized)\n'
                  '(e) Learned soft mask', fontsize=8)
    ax.set_ylabel('EEG position $i$ (normalized)', fontsize=8.5)
    ax.tick_params(labelsize=7.5, pad=2)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    fig.savefig(os.path.join(FIGURES, 'Fig5.pdf'))
    plt.close(fig)


def fig6():
    profile_path = os.path.join(RESULTS, 'attention_profiles.json')
    if os.path.exists(profile_path):
        d = json.load(open(profile_path))
        attn = np.array(d['mean_attention'])
        samples = np.array(d.get('sample_profiles_max',
                                 d.get('sample_profiles')))
        mean_prof = np.array(d.get('mean_profile_max',
                                   d.get('mean_profile')))
    else:
        attn = np.array(load('mask_verification.json')['fixed_mask']['mean_attention'])
        samples = None
        mean_prof = attn.max(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7),
                             gridspec_kw={'width_ratios': [1.4, 1]})
    ax = axes[0]
    vmax = np.percentile(attn[attn > 0], 99)
    im = ax.imshow(attn, aspect='auto', origin='upper', cmap='hot',
                   extent=[0, 50, 256, 0], vmin=0, vmax=vmax)
    ax.set_xlabel('Gaze position $j$ (time)\n'
                  '(a) Mean cross-attention weights', fontsize=7.5)
    ax.set_ylabel('EEG position $i$ (time)')
    cb = plt.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label('Attention weight', fontsize=5.8)
    cb.ax.tick_params(labelsize=6)

    ax = axes[1]
    epos = np.arange(256)
    if samples is not None:
        for k, row in enumerate(samples):
            ax.plot(epos, row, color=plt.cm.tab10(k), lw=0.8, alpha=0.7,
                    label=f'Sample {k + 1}')
    ax.plot(epos, mean_prof, color='k', lw=1.3, label='Mean')
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('EEG position $i$ (time)\n'
                  '(b) Per-sample max attention', fontsize=7.5)
    ax.set_ylabel('Max attention weight')
    ax.grid(lw=0.3, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(fontsize=5.6, loc='upper right', frameon=False,
              handlelength=1.4, labelspacing=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'Fig6.pdf'))
    plt.close(fig)


def fig7():
    d = load('channel_zeroout.json')
    ch = d['channels']
    mean = np.array(d['mean_delta_mse'])
    std = np.array(d['std_delta_mse'])
    order = np.argsort(-mean)

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))
    ax = axes[0]
    ax.barh(range(len(ch)), mean[order][::-1], xerr=std[order][::-1],
            color=BLUE, capsize=2)
    ax.set_yticks(range(len(ch)))
    ax.set_yticklabels([ch[i] for i in order][::-1], fontsize=7)
    ax.set_xlabel('MSE increase (training mean; z-score$^2$)\n'
                  '(a) Per-channel contribution (gaze changes)', fontsize=7.5)

    ax = axes[1]
    pos = {'AF3': (-0.35, 0.9), 'AF4': (0.35, 0.9), 'F7': (-0.7, 0.6),
           'F8': (0.7, 0.6), 'F3': (-0.3, 0.65), 'F4': (0.3, 0.65),
           'FC5': (-0.6, 0.3), 'FC6': (0.6, 0.3), 'T7': (-0.85, 0.0),
           'T8': (0.85, 0.0), 'P7': (-0.6, -0.5), 'P8': (0.6, -0.5),
           'O1': (-0.3, -0.85), 'O2': (0.3, -0.85)}
    vals = np.array(d['mean_delta_mse'])
    norm = TwoSlopeNorm(vmin=float(vals.min()), vcenter=0.0,
                        vmax=float(vals.max()))
    cmap = plt.cm.RdBu_r
    for c, (cx, cy) in pos.items():
        i = ch.index(c)
        rgba = cmap(norm(vals[i]))
        lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        ax.scatter(cx, cy, s=520, color=rgba, edgecolor='k', lw=0.5,
                   zorder=2)
        ax.text(cx, cy, c, ha='center', va='center', fontsize=6.0,
                color='white' if lum < 0.5 else 'black', zorder=3)
    ax.add_patch(plt.Circle((0, 0), 1.20, fill=False, ec='k', lw=1.0))
    ax.plot([-0.18, 0, 0.18], [1.16, 1.34, 1.16], 'k-', lw=1.0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label('MSE increase (z-score$^2$)', fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    ax.set_xlim(-1.30, 1.30)
    ax.set_ylim(-1.30, 1.42)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.text(0.5, -0.10, '(b) Topographic importance map',
            transform=ax.transAxes, ha='center', va='top', fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'Fig7.pdf'))
    plt.close(fig)


def fig8():
    ver = load('mask_verification.json')
    abl = load('results_ablation.json')

    delta = ver['learned_mask']['delay_normalized']
    beta = ver['learned_mask']['bandwidth']

    T, S = 256, 50
    i = np.arange(T)[:, None]
    j = np.arange(S)[None, :]
    allowed = (j <= np.floor(i * S / T)).astype(float)
    soft = 1.0 / (1.0 + np.exp(-(((i / T - delta) - (j / S)) / beta)))

    runs = abl.get('TempStructNet', {}).get('runs', [])
    delays = np.array([r['delay'] for r in runs if r.get('delay') is not None])

    syn = ver.get('synthetic', [])
    offsets = sorted({s['target_offset'] for s in syn})

    history_path = os.path.join(RESULTS, 'learned_structure_history.json')
    history = json.load(open(history_path)) if os.path.exists(history_path) else None

    fig = plt.figure(figsize=(6.6, 4.6))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.42,
                          left=0.07, right=0.97, top=0.96, bottom=0.13)

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(soft, origin='lower', aspect='auto', cmap='RdYlGn',
                   extent=[0, 30, 0, 30], vmin=0, vmax=1)
    ax.plot([0, 30], [0, 30], 'k--', lw=0.9,
            label=rf'Learned boundary ($\delta$={delta:.3f})')
    ax.set_xlabel('Gaze position $j$ (normalized)\n(a) Learned soft mask',
                  fontsize=7.5)
    ax.set_ylabel('EEG position $i$ (normalized)')
    ax.legend(fontsize=5.5, loc='upper center', framealpha=0.85)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label('Attention weight', fontsize=5.8)
    cb.ax.tick_params(labelsize=6)

    ax = fig.add_subplot(gs[0, 1])
    diff = soft - allowed
    im = ax.imshow(diff, origin='lower', aspect='auto', cmap='RdBu_r',
                   extent=[0, 30, 0, 30], vmin=-0.4, vmax=0.4)
    ax.plot([0, 30], [0, 30], 'k--', lw=0.8)
    ax.set_xlabel('Gaze position $j$ (normalized)\n'
                  '(b) Learned minus fixed mask', fontsize=7.5)
    ax.set_ylabel('EEG position $i$ (normalized)')
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label('Difference', fontsize=5.8)
    cb.ax.tick_params(labelsize=6)

    ax = fig.add_subplot(gs[1, 0])
    if history is not None:
        n = int(np.argmin(history['inner_loss'])) + 1
        ep = np.array(history['epoch'][:-1])
        curve = np.array(history['delay'][:-1])
        ax.plot(ep, curve, color='tab:blue', lw=1.2, label='Learned delay')
        ax.axvline(n, color='0.55', ls=':', lw=0.9, label='Selected epoch')
        ax.axhline(history['delay'][-1], color='tab:red', ls='--', lw=1.0,
                   alpha=0.6,
                   label=f'Selected delay = {history["delay"][-1]:.3f}')
        ax2 = ax.twinx()
        ax2.plot(ep, history['temperature'][:-1], color='tab:green',
                 ls='--', lw=1.0, alpha=0.7)
        ax2.set_ylabel('Temperature', color='tab:green', fontsize=7)
        ax2.tick_params(axis='y', colors='tab:green', labelsize=6.5)
        ax.legend(fontsize=5.8, loc='upper right', frameon=False)
    else:
        ax.hist(delays, bins=8, color='tab:blue', alpha=0.85)
        ax.axvline(0.15, color='tab:red', lw=1.1, ls=':')
        ax.axvline(delta, color='tab:red', lw=1.1)
    ax.grid(lw=0.3, alpha=0.35)
    ax.set_axisbelow(True)
    if history is not None:
        ax.set_xlabel('Epoch\n(c) Delay evolution during training',
                      fontsize=7.2)
    else:
        ax.set_xlabel(r'Learned delay $\delta$ (normalized)'
                      + '\n(c) Real data', fontsize=7.2)
    ax.set_ylabel(r'Learned $\delta$')

    ax = fig.add_subplot(gs[1, 1])
    if history is not None:
        curve = np.array(history['bandwidth'][:-1])
        ax.plot(ep, curve, color='tab:blue', lw=1.2,
                label='Learned bandwidth')
        ax.axvline(n, color='0.55', ls=':', lw=0.9, label='Selected epoch')
        ax.axhline(history['bandwidth'][-1], color='tab:red', ls='--', lw=1.0,
                   alpha=0.6,
                   label=f'Selected bandwidth = {history["bandwidth"][-1]:.3f}')
        ax.legend(fontsize=5.8, loc='lower right', frameon=False)
    else:
        means = [np.mean([s['delta_learned'] for s in syn
                          if s['target_offset'] == o]) for o in offsets]
        stds = [np.std([s['delta_learned'] for s in syn
                        if s['target_offset'] == o]) for o in offsets]
        ax.bar(range(len(offsets)), means, yerr=stds, color=ORANGE, capsize=2)
        ax.set_xticks(range(len(offsets)))
        ax.set_xticklabels([f'{o:.2f}' for o in offsets])
        ax.set_ylim(0, 0.2)
    ax.grid(lw=0.3, alpha=0.35)
    ax.set_axisbelow(True)
    if history is not None:
        ax.set_xlabel('Epoch\n(d) Bandwidth evolution during training',
                      fontsize=7.2)
    else:
        ax.set_xlabel('Generative target offset\n(d) Synthetic control',
                      fontsize=7.5)
    ax.set_ylabel(r'Learned $\beta$')

    for ax in fig.axes:
        ax.tick_params(labelsize=7, pad=2)

    fig.savefig(os.path.join(FIGURES, 'Fig8.pdf'))
    plt.close(fig)


def fig9():
    res = load('lagged_analysis.json')
    imp1 = np.array([r['eeg_to_gaze_imp'] for r in res])
    p1 = np.array([r['eeg_to_gaze_p'] for r in res])
    imp2 = np.array([r['gaze_to_eeg_imp'] for r in res])
    p2 = np.array([r['gaze_to_eeg_p'] for r in res])

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
    ax = axes[0]
    bp = ax.boxplot([imp1, imp2], patch_artist=True, widths=0.5,
                    showfliers=False)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['EEG$\\rightarrow$gaze', 'Gaze$\\rightarrow$EEG'])
    for patch, c in zip(bp['boxes'], [BLUE, ORANGE]):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_ylabel('Relative MSE improvement (CV)', fontsize=8)
    ax.set_xlabel(f"(a) Lagged predictability (L=4, 16 Hz)\n"
                  f"sig={int((p1<0.05).sum())}/{len(p1)} vs "
                  f"{int((p2<0.05).sum())}/{len(p2)}", fontsize=7.5,
                  labelpad=6)

    ax = axes[1]
    bins = np.linspace(0, 1, 21)
    ax.axvspan(0, 0.05, color=RED, alpha=0.08, lw=0)
    ax.hist(p1, bins=bins, alpha=0.65, color=BLUE, edgecolor='black',
            linewidth=0.35, label='EEG$\\rightarrow$gaze')
    ax.hist(p2, bins=bins, alpha=0.55, color=ORANGE, edgecolor='black',
            linewidth=0.35, label='Gaze$\\rightarrow$EEG')
    ax.axvline(0.05, color=RED, ls='--', lw=1, label='p=0.05')
    ax.set_xlabel('Permutation p-value\n(b) Null distribution check',
                  fontsize=7.5, labelpad=4)
    ax.set_ylabel('Participants')
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'Fig9.pdf'))
    plt.close(fig)


def fig10():
    res = load('lagged_analysis.json')
    imp_f = 100.0 * np.array([r['eeg_to_gaze_imp'] for r in res])
    imp_r = 100.0 * np.array([r['gaze_to_eeg_imp'] for r in res])
    p_f = np.array([r['eeg_to_gaze_p'] for r in res])
    p_r = np.array([r['gaze_to_eeg_p'] for r in res])
    n = len(res)
    med = [float(np.median(imp_f)), float(np.median(imp_r))]
    sig = [int((p_f < 0.05).sum()), int((p_r < 0.05).sum())]
    pmed = [float(np.median(p_f)), float(np.median(p_r))]
    labels = ['EEG$\\to$gaze', 'Gaze$\\to$EEG']
    colors = ['#E06C5A', '#B7B7B7']

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.95, bottom=0.24,
                        wspace=0.45)
    x = np.arange(2)

    ax = axes[0]
    ax.bar(x, med, color=colors, edgecolor='black', linewidth=0.5, width=0.6)
    ax.axhline(0, color='k', lw=0.6)
    for i, v in enumerate(med):
        ax.annotate(f'{v:+.1f}', (i, v), xytext=(0, -4),
                    textcoords='offset points', ha='center', va='top',
                    fontsize=6.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.8)
    ax.set_ylabel('Median improvement (%)', fontsize=7.5)
    ax.set_xlabel('(a) Relative MSE improvement', fontsize=7.5, labelpad=5)
    ax.set_ylim(min(med) - 0.45, 0.15)

    ax = axes[1]
    ax.bar(x, sig, color=colors, edgecolor='black', linewidth=0.5, width=0.6)
    for i, v in enumerate(sig):
        ax.annotate(f'{v}', (i, v), xytext=(0, 3),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=6.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.8)
    ax.set_ylabel(f'Participants (of {n})', fontsize=7.5)
    ax.set_xlabel('(b) Significant ($p<0.05$)', fontsize=7.5, labelpad=5)
    ax.set_ylim(0, max(sig) * 1.28)

    ax = axes[2]
    ax.bar(x, pmed, color=colors, edgecolor='black', linewidth=0.5, width=0.6)
    for i, v in enumerate(pmed):
        ax.annotate(f'{v:.3f}', (i, v), xytext=(0, 3),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=6.2)
    ax.axhline(0.05, color=RED, ls='--', lw=1, label='$p=0.05$')
    ax.legend(fontsize=6, loc='upper right', frameon=False,
              handlelength=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.8)
    ax.set_ylabel('Median permutation $p$', fontsize=7.5)
    ax.set_xlabel('(c) Permutation $p$-value', fontsize=7.5, labelpad=5)
    ax.set_ylim(0, 0.8)

    for ax in axes:
        ax.tick_params(labelsize=7, pad=2)

    fig.savefig(os.path.join(FIGURES, 'Fig10.pdf'))
    plt.close(fig)


def fig11():
    s = load('summary.json')
    keys = [('BandTheta', 'Theta\n(4--8 Hz)'), ('BandAlpha', 'Alpha\n(8--13 Hz)'),
            ('BandBeta', 'Beta\n(13--30 Hz)')]
    colors = ['#E24A33', '#348ABD', '#2CA25F']
    rmse = [s['models'][k]['rmse_mean'] for k, _ in keys]
    rc = [s['models'][k]['r_col_mean'] for k, _ in keys]
    rr = [s['models'][k]['r_row_mean'] for k, _ in keys]
    labels = [lab for _, lab in keys]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    panels = [(rmse, 'RMSE', '(a) RMSE'),
              (rc, r'$r_{\Delta\mathrm{col}}$', r'(b) $\Delta$col correlation'),
              (rr, r'$r_{\Delta\mathrm{row}}$', r'(c) $\Delta$row correlation')]
    x = np.arange(len(keys))
    for ax, (vals, ylab, cap) in zip(axes, panels):
        ax.bar(x, vals, color=colors, edgecolor='k', lw=0.5, width=0.72)
        top = max(vals)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02 * top, f'{v:.3f}', ha='center',
                    va='bottom', fontsize=6.4)
        ax.set_ylim(0, top * 1.18)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.0)
        ax.set_ylabel(ylab, fontsize=8)
        ax.set_xlabel(cap, fontsize=8)
        ax.grid(axis='y', lw=0.3, alpha=0.35)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=6.8, pad=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'Fig11.pdf'))
    plt.close(fig)


def fig12():
    s = load('summary.json')
    names = [n for n in ['TempAlignEEGNet', 'PermutedEEG', 'PermutedTarget']
             if n in s['models']]
    if len(names) < 2:
        return
    labels = ['Full', 'EEG\nscrambled', 'Target\nshuffled']
    colors = ['#E06C5A', '#B7B7B7', '#9DC3E6']
    x = np.arange(len(names))

    def fold_mean(m, key):
        return float(np.mean([f[key] for f in m['per_fold'].values()]))

    rmse = [s['models'][n]['rmse_mean'] for n in names]
    rmse_ci = [s['models'][n]['rmse_ci95'] for n in names]
    rc = [s['models'][n]['r_col_mean'] for n in names]
    rc_ci = [s['models'][n]['r_col_ci95'] for n in names]
    rr = [fold_mean(s['models'][n], 'r_row') for n in names]

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.7))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.25,
                        wspace=0.55)

    panels = [(rmse, rmse_ci, 'RMSE', '(a) RMSE'),
              (rc, rc_ci, r'$r_{\Delta\mathrm{col}}$',
               r'(b) $\Delta$col correlation'),
              (rr, None, r'$r_{\Delta\mathrm{row}}$',
               r'(c) $\Delta$row correlation')]
    for ax, (vals, errs, ylab, cap) in zip(axes[:3], panels):
        ax.bar(x, vals, yerr=errs, color=colors, edgecolor='black',
               linewidth=0.45, width=0.66, capsize=1.8)
        top = max(v + (e or 0) for v, e in zip(vals, errs or [0] * len(vals)))
        for i, (v, e) in enumerate(zip(vals, errs or [0] * len(vals))):
            ax.text(i, v + e + 0.02 * top, f'{v:.3f}', ha='center',
                    va='bottom', fontsize=5.8)
        low = min([0.0] + [v - (e or 0) for v, e in
                            zip(vals, errs or [0] * len(vals))])
        ax.set_ylim(low * 1.15 if low < 0 else 0, top * 1.22)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=5.8)
        ax.set_ylabel(ylab, fontsize=7.2)
        ax.set_xlabel(cap, fontsize=6.8, labelpad=5)
        ax.grid(axis='y', lw=0.3, alpha=0.35)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=6.2, pad=2)

    ax = axes[3]
    full = np.array([r['r_col'] for r in
                     s['models']['TempAlignEEGNet']['per_participant']])
    ctrl = np.array([r['r_col'] for r in
                     s['models']['PermutedTarget']['per_participant']])
    bins = np.linspace(-0.30, 0.65, 24)
    ax.hist(ctrl, bins=bins, color='#70B6E5', edgecolor='black',
            linewidth=0.35, alpha=0.9, label='Target-shuffled')
    ax.axvline(full.mean(), color='#E06C5A', lw=1.4, label='Full mean')
    ax.axvline(ctrl.mean(), color='0.35', ls='--', lw=1, label='Control mean')
    ax.set_xlim(-0.30, 0.65)
    ax.set_xlabel(r'Per-participant $r_{\Delta\mathrm{col}}$' '\n'
                  '(d) Control distribution', fontsize=6.8, labelpad=5)
    ax.set_ylabel('Participants', fontsize=7.2)
    ax.legend(fontsize=5.2, frameon=False, loc='upper right',
              handlelength=1.4)
    ax.tick_params(labelsize=6.2, pad=2)

    fig.savefig(os.path.join(FIGURES, 'Fig12.pdf'))
    plt.close(fig)


def fig13():
    rows = list(csv.DictReader(open(os.path.join(
        RESULTS, 'per_participant_metrics.csv'))))
    rc = np.array([float(r['r_col']) for r in rows])
    rr = np.array([float(r['r_row']) for r in rows])
    pooled = load('summary.json')['models']['TempAlignEEGNet']['pooled']
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.25,
                        wspace=0.28)
    for ax, v, lab, tag, pk in [(axes[0], rc, 'col', 'a', 'r_col'),
                                (axes[1], rr, 'row', 'b', 'r_row')]:
        ax.hist(v, bins=18, color=BLUE, alpha=0.85, edgecolor='white')
        ax.axvline(v.mean(), color=RED, lw=1.2,
                   label=f'Mean = {v.mean():.3f}')
        ax.axvline(pooled[pk], color=GREEN, ls='--', lw=1.2,
                   label=f'Overall = {pooled[pk]:.3f}')
        ax.set_xlabel(rf'Per-participant $r_{{\Delta\mathrm{{{lab}}}}}$' '\n'
                      rf'({tag}) $r_{{\Delta\mathrm{{{lab}}}}}$ distribution '
                      f'(n = {len(v)})', fontsize=7.5, labelpad=3)
        ax.set_ylabel('Count', fontsize=7.5)
        ax.grid(lw=0.3, alpha=0.35)
        ax.set_axisbelow(True)
        ax.legend(fontsize=6.2, frameon=True, loc='upper right')
        ax.tick_params(labelsize=6.8, pad=2)
    fig.savefig(os.path.join(FIGURES, 'Fig13.pdf'))
    plt.close(fig)


FIGURES_BY_NAME = {
    'fig1': fig1, 'fig2': fig2, 'fig3': fig3, 'fig4': fig4, 'fig5': fig5,
    'fig6': fig6, 'fig7': fig7, 'fig8': fig8, 'fig9': fig9, 'fig10': fig10,
    'fig11': fig11, 'fig12': fig12, 'fig13': fig13,
}


FONT_SIZE = {'fig1': 8, 'fig3': 8, 'fig4': 8, 'fig5': 8, 'fig13': 8}


def main():
    wanted = sys.argv[1:] or list(FIGURES_BY_NAME)
    for name in wanted:
        key = name.lower()
        fn = FIGURES_BY_NAME.get(key)
        if fn is None:
            print('unknown figure:', name)
            continue
        with plt.rc_context({'font.size': FONT_SIZE.get(key, 9)}):
            fn()
        print('saved', name)


if __name__ == '__main__':
    main()
