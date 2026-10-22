"""Evaluation metrics, uncertainty, and paired tests.

All metrics are computed on held-out participant folds.
  - r          : Pearson correlation (per dimension)
  - R2_pred    : out-of-sample predictive R^2 = 1 - SSE/SST, where SST is
                 about the mean of the held-out targets (NOT Pearson r^2)
  - RMSE       : in z-score units of the training-fold target standardization
"""

import numpy as np
from scipy import stats


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def predictive_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - y_true.mean()) ** 2)
    if sst < 1e-12:
        return 0.0
    return float(1.0 - sse / sst)


def evaluate_predictions(targets, preds):
    """targets/preds: (N, 2) arrays. Returns dict of metrics."""
    out = {}
    for i, name in enumerate(['col', 'row']):
        out[f'r_{name}'] = pearson(targets[:, i], preds[:, i])
        out[f'R2_{name}'] = predictive_r2(targets[:, i], preds[:, i])
    out['rmse'] = float(np.sqrt(np.mean((targets - preds) ** 2)))
    return out


def per_participant_metrics(targets, preds, pids, min_windows=30):
    """Per-participant r and predictive R^2 on held-out windows.

    Returns dict with arrays keyed by pid for participants with at least
    min_windows held-out windows.
    """
    rows = []
    for pid in np.unique(pids):
        m = pids == pid
        if m.sum() < min_windows:
            continue
        t, p = targets[m], preds[m]
        rows.append({
            'pid': int(pid), 'n': int(m.sum()),
            'r_col': pearson(t[:, 0], p[:, 0]),
            'r_row': pearson(t[:, 1], p[:, 1]),
            'R2_col': predictive_r2(t[:, 0], p[:, 0]),
            'rmse': float(np.sqrt(np.mean((t - p) ** 2))),
        })
    return rows


def bootstrap_participant_ci(targets, preds, pids, metric_fn, n_boot=2000, seed=0):
    """Bootstrap 95% CI resampling participants with replacement."""
    rng = np.random.RandomState(seed)
    unique = np.unique(pids)
    idx_by_pid = {pid: np.where(pids == pid)[0] for pid in unique}
    stats_boot = []
    for _ in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([idx_by_pid[p] for p in sampled])
        stats_boot.append(metric_fn(targets[idx], preds[idx]))
    stats_boot = np.array(stats_boot)
    return {
        'mean': float(stats_boot.mean()),
        'lo': float(np.percentile(stats_boot, 2.5)),
        'hi': float(np.percentile(stats_boot, 97.5)),
    }


def paired_participant_test(per_part_a, per_part_b, key='r_col'):
    """Paired comparison across participants (a vs b), by pid.

    Returns Wilcoxon signed-rank and paired t-test results plus the mean
    difference and its bootstrap CI.
    """
    da = {r['pid']: r[key] for r in per_part_a}
    db = {r['pid']: r[key] for r in per_part_b}
    common = sorted(set(da) & set(db))
    a = np.array([da[p] for p in common])
    b = np.array([db[p] for p in common])
    diff = a - b
    if len(common) < 5 or np.allclose(diff, 0):
        return {'n': len(common), 'mean_diff': float(diff.mean()) if len(diff) else 0.0,
                'wilcoxon_p': 1.0, 'ttest_p': 1.0, 'ci_lo': np.nan, 'ci_hi': np.nan}
    w = stats.wilcoxon(a, b)
    t = stats.ttest_rel(a, b)
    rng = np.random.RandomState(0)
    boots = [diff[rng.randint(0, len(diff), len(diff))].mean() for _ in range(2000)]
    return {
        'n': len(common),
        'mean_diff': float(diff.mean()),
        'ci_lo': float(np.percentile(boots, 2.5)),
        'ci_hi': float(np.percentile(boots, 97.5)),
        'wilcoxon_p': float(w.pvalue),
        'ttest_p': float(t.pvalue),
    }


def mean_ci(values, axis=0):
    """Mean and 95% CI of fold-level values (t distribution)."""
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    m = v.mean(axis=axis)
    if n < 2:
        return m, np.nan
    se = v.std(axis=axis, ddof=1) / np.sqrt(n)
    h = se * stats.t.ppf(0.975, n - 1)
    return m, h
