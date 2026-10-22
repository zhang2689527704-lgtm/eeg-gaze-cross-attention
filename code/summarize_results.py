"""Aggregate all experiment JSONs into a single numbers registry
(summary.json) used by every table, figure, and number in the manuscript.

Contents:
  - per-model fold-level metrics (mean +/- std across folds, 95% t-CI)
  - pooled held-out metrics with participant-bootstrap 95% CI
  - paired participant-level tests between TempAlignEEGNet and each other
    model (Wilcoxon + paired t on per-participant r_col and R2_col)
"""

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import (evaluate_predictions, per_participant_metrics, mean_ci,
                     paired_participant_test, bootstrap_participant_ci, pearson,
                     predictive_r2)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
RESULTS = os.path.join(ROOT, 'results')
PREDICTIONS = os.path.join(ROOT, 'predictions')
TAGS = ['main', 'ablation']
PRIMARY_SEED = 42


def load_preds(pattern):
    files = sorted(glob.glob(os.path.join(PREDICTIONS, pattern)))
    if not files:
        return None
    per_seed = {}
    for f in files:
        seed = int(f.split('_s')[1].split('.')[0])
        per_seed.setdefault(seed, []).append(f)
    merged = {}
    for seed, fs in per_seed.items():
        arrs = [np.load(f) for f in sorted(fs)]
        merged[seed] = {
            'pids': np.concatenate([a['pids'] for a in arrs]),
            'targets': np.concatenate([a['targets'] for a in arrs]),
            'preds': np.concatenate([a['preds'] for a in arrs]),
        }
    return merged


def seed_averaged(per_seed):
    """Average predictions across seeds per window (same window order per seed
    is not guaranteed across folds; instead we align by (fold file, index) —
    files sorted identically across seeds, and within a fold the dataloader
    order is deterministic, so concatenating sorted-by-fold files aligns."""
    seeds = sorted(per_seed)
    ref = per_seed[seeds[0]]
    preds = np.mean([per_seed[s]['preds'] for s in seeds], axis=0)
    return {'pids': ref['pids'], 'targets': ref['targets'], 'preds': preds}


def model_summary(name, runs, merged, params=None):
    folds = sorted({r['fold'] for r in runs})
    per_fold = {}
    for f in folds:
        fr = [r for r in runs if r['fold'] == f]
        per_fold[f] = {
            'r_col': float(np.mean([r['r_col'] for r in fr])),
            'r_row': float(np.mean([r['r_row'] for r in fr])),
            'R2_col': float(np.mean([r['R2_col'] for r in fr])),
            'R2_row': float(np.mean([r['R2_row'] for r in fr])),
            'rmse': float(np.mean([r['rmse'] for r in fr])),
            'n_seeds': len(fr),
        }
    if params is None:
        params = runs[0].get('params')
    out = {'per_fold': per_fold, 'params': params}
    for key in ['r_col', 'r_row', 'R2_col', 'R2_row', 'rmse']:
        vals = [per_fold[f][key] for f in folds]
        m, h = mean_ci(vals)
        out[f'{key}_mean'] = float(m)
        out[f'{key}_ci95'] = float(h) if np.isfinite(h) else 0.0
        out[f'{key}_std'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    if merged is not None:
        sa = seed_averaged(merged)
        pooled = evaluate_predictions(sa['targets'], sa['preds'])
        out['pooled'] = {k: float(v) for k, v in pooled.items()}
        out['ci_r_col'] = bootstrap_participant_ci(
            sa['targets'], sa['preds'], sa['pids'],
            lambda t, p: pearson(t[:, 0], p[:, 0]))
        out['ci_r_row'] = bootstrap_participant_ci(
            sa['targets'], sa['preds'], sa['pids'],
            lambda t, p: pearson(t[:, 1], p[:, 1]))
        out['ci_R2_col'] = bootstrap_participant_ci(
            sa['targets'], sa['preds'], sa['pids'],
            lambda t, p: predictive_r2(t[:, 0], p[:, 0]))
        out['per_participant'] = per_participant_metrics(
            sa['targets'], sa['preds'], sa['pids'])
    return out


def main():
    summary = {'models': {}, 'paired': {}, 'paired_aux': {}, 'config': {
        'primary_seed': PRIMARY_SEED, 'n_folds': 5,
        'outer_split_seed': 42}}

    for tag in TAGS:
        path = os.path.join(RESULTS, f'results_{tag}.json')
        if not os.path.exists(path):
            continue
        results = json.load(open(path))
        for name, v in results.items():
            runs = v.get('runs', [])
            if name == 'StatBaselines':
                for bname in sorted({r['name'] for r in runs}):
                    bruns = [r for r in runs if r['name'] == bname]
                    files = sorted(glob.glob(os.path.join(
                        PREDICTIONS,
                        f'preds_{tag}_StatBaselines_{bname}_f*.npz')))
                    merged = None
                    if files:
                        arrs = [np.load(f) for f in files]
                        merged = {0: {
                            'pids': np.concatenate([a['pids'] for a in arrs]),
                            'targets': np.concatenate([a['targets'] for a in arrs]),
                            'preds': np.concatenate([a['preds'] for a in arrs])}}
                    summary['models'][bname] = model_summary(
                        bname, bruns, merged, params=None)
                continue
            if not runs:
                continue
            merged = load_preds(f'preds_{tag}_{name}_f*_s*.npz')
            summary['models'][name] = model_summary(
                name, runs, merged, params=v.get('params'))
            summary['models'][name]['tag'] = tag

    # masked LSTM / Transformer baselines (separate runner)
    extra_path = os.path.join(RESULTS, 'extra_models_cv.json')
    if os.path.exists(extra_path):
        extra = json.load(open(extra_path))
        for name, blk in extra.items():
            s = model_summary(name, blk['runs'], None, params=blk['params'])
            for f in s['per_fold'].values():
                f.pop('n_seeds', None)
            summary['models'][name] = s

    # Table 1 alias: the learned-mask ablation is reported as LearnedMaskNet
    if 'TempStructNet' in summary['models']:
        lm = json.loads(json.dumps(summary['models']['TempStructNet']))
        for k in ('per_participant', 'pooled', 'tag',
                  'ci_r_col', 'ci_r_row', 'ci_R2_col'):
            lm.pop(k, None)
        for f in lm['per_fold'].values():
            f.pop('n_seeds', None)
        lm['params'] = None
        summary['models']['LearnedMaskNet'] = lm

    # paired tests vs the proposed model
    ref_name = 'TempAlignEEGNet'
    if ref_name in summary['models'] and 'per_participant' in summary['models'][ref_name]:
        ref = summary['models'][ref_name]['per_participant']
        for other in summary['models']:
            if other == ref_name or 'per_participant' not in summary['models'][other]:
                continue
            oth = summary['models'][other]['per_participant']
            summary['paired'][other] = {
                'r_col': paired_participant_test(ref, oth, 'r_col'),
                'R2_col': paired_participant_test(ref, oth, 'R2_col'),
                'rmse': paired_participant_test(
                    [{**r, 'rmse': -r['rmse']} for r in ref],
                    [{**r, 'rmse': -r['rmse']} for r in oth], 'rmse'),
            }

    # auxiliary paired tests: each comparison against the gaze-only variant
    aux = {'FrontalVsGazeOnly': 'FrontalExcluded',
           'TempStructVsGazeOnly': 'TempStructNet',
           'PermEEGVsGazeOnly': 'PermutedEEG',
           'LSTMvsGazeOnly': 'LSTM',
           'TransformerVsGazeOnly': 'Transformer',
           'LTCvsGazeOnly': 'LTC',
           'WaveletVsGazeOnly': 'WaveletTransformer',
           'RidgeVsGazeOnly': 'RidgeGaze',
           'NoMaskVsGazeOnly': 'NoMask'}
    gaze = summary['models'].get('GazeOnly', {})
    if 'per_participant' in gaze:
        ref = gaze['per_participant']
        for key, model in aux.items():
            other = summary['models'].get(model, {})
            if 'per_participant' in other:
                summary['paired_aux'][key] = paired_participant_test(
                    other['per_participant'], ref, 'r_col')

    with open(os.path.join(RESULTS, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)

    print(f"{'Model':<22} {'r_col':>18} {'R2_col':>18} {'rmse':>16}")
    for name, m in summary['models'].items():
        print(f"{name:<22} {m['r_col_mean']:+.3f}±{m['r_col_ci95']:.3f}   "
              f"{m['R2_col_mean']:+.3f}±{m['R2_col_ci95']:.3f}   "
              f"{m['rmse_mean']:.3f}±{m['rmse_ci95']:.3f}")
    print('\npaired vs', ref_name)
    for other, p in summary['paired'].items():
        pc = p['r_col']
        print(f"  {other:<22} Δr={pc['mean_diff']:+.3f} "
              f"[{pc['ci_lo']:+.3f},{pc['ci_hi']:+.3f}] p_w={pc['wilcoxon_p']:.2e}")


if __name__ == '__main__':
    main()
