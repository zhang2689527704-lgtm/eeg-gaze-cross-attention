"""Lagged directional predictability analysis.

Design (per participant, sample level):
  - series resampled to 16 Hz: EEG (14 channels) and gaze displacement
    (Delta col, Delta row at the 62.5 ms grid).
  - restricted model: predict the target at time t from its own past L=4 lags.
  - unrestricted model: additionally include the other modality's past L lags.
  - MSE estimated by blocked 5-fold cross-validation along time within the
    participant (contiguous blocks avoid leakage from serial correlation).
  - significance: circular time-shift permutation of the predictor modality
    (100 shifts, each >= 8 s) preserves autocorrelation structure.

Two directions are tested symmetrically:
  EEG -> gaze  (does past EEG improve prediction of future gaze change?)
  gaze -> EEG  (does past gaze improve prediction of future EEG?)
Both are lagged *predictability* tests (Wiener-Granger sense), not causal
intervention claims.
"""

import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_main import load_cache

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', 'results')

FS = 16.0
L_LAGS = 4
N_PERM = 100
RIDGE_ALPHA = 10.0


def resample_series(x, t_src, t_dst):
    return np.stack([np.interp(t_dst, t_src, x[:, c]) for c in range(x.shape[1])], axis=1)


def make_lagged(X, y, L):
    """Returns (Xt, yt) with rows of L lagged copies aligned to y[L:]."""
    n, f = X.shape
    rows = np.concatenate([X[L - k - 1: n - k - 1] for k in range(L)], axis=1)
    return rows, y[L:]


def cv_mse(Xt, yt, alpha=RIDGE_ALPHA, n_blocks=5):
    n = len(yt)
    bounds = np.linspace(0, n, n_blocks + 1).astype(int)
    sse, tot = 0.0, 0
    for b in range(n_blocks):
        te = np.zeros(n, dtype=bool)
        te[bounds[b]:bounds[b + 1]] = True
        if te.sum() == 0 or (~te).sum() == 0:
            continue
        m = Ridge(alpha=alpha).fit(Xt[~te], yt[~te])
        pred = m.predict(Xt[te])
        sse += np.sum((yt[te] - pred) ** 2)
        tot += te.sum()
    return sse / max(tot, 1)


def rel_improvement(X_rest, X_full, y):
    Xr, yr = make_lagged(X_rest, y, L_LAGS)
    Xf, _ = make_lagged(X_full, y, L_LAGS)
    mse_r = cv_mse(Xr, yr)
    mse_f = cv_mse(Xf, yr)
    return (mse_r - mse_f) / mse_r, mse_r, mse_f


def analyze_participant(payload):
    pid, eeg16, dgaze16, rng_seed = payload
    rng = np.random.RandomState(rng_seed)
    n = min(len(eeg16), len(dgaze16))
    eeg16, dgaze16 = eeg16[:n], dgaze16[:n]
    if n < 16 * 60:  # need >= 60 s
        return None

    out = {'pid': int(pid), 'n_samples': int(n)}
    # EEG -> gaze: target = dgaze (2 dims) predicted from own lags [+ eeg lags]
    Xg = dgaze16
    obs_imp, mse_r, mse_f = rel_improvement(Xg, np.concatenate([Xg, eeg16], axis=1), dgaze16)
    null = []
    for _ in range(N_PERM):
        shift = rng.randint(int(8 * FS), n - int(8 * FS))
        eeg_shift = np.roll(eeg16, shift, axis=0)
        imp, _, _ = rel_improvement(Xg, np.concatenate([Xg, eeg_shift], axis=1), dgaze16)
        null.append(imp)
    null = np.array(null)
    out['eeg_to_gaze_imp'] = float(obs_imp)
    out['eeg_to_gaze_p'] = float((np.sum(null >= obs_imp) + 1) / (N_PERM + 1))
    out['eeg_to_gaze_mse_restricted'] = float(mse_r)
    out['eeg_to_gaze_mse_full'] = float(mse_f)

    # gaze -> EEG: target = EEG channels predicted from own lags [+ gaze lags]
    obs_imp2, mse_r2, mse_f2 = rel_improvement(eeg16, np.concatenate([eeg16, Xg], axis=1), eeg16)
    null2 = []
    for _ in range(N_PERM):
        shift = rng.randint(int(8 * FS), n - int(8 * FS))
        g_shift = np.roll(Xg, shift, axis=0)
        imp, _, _ = rel_improvement(eeg16, np.concatenate([eeg16, g_shift], axis=1), eeg16)
        null2.append(imp)
    null2 = np.array(null2)
    out['gaze_to_eeg_imp'] = float(obs_imp2)
    out['gaze_to_eeg_p'] = float((np.sum(null2 >= obs_imp2) + 1) / (N_PERM + 1))
    out['gaze_to_eeg_mse_restricted'] = float(mse_r2)
    out['gaze_to_eeg_mse_full'] = float(mse_f2)
    return out


def build_participant_series(cache):
    """Downsample cached continuous EEG + gaze displacement series per participant."""
    eeg_sr = 128.0
    out = {}
    offsets = cache['offsets']
    meta = cache['meta']
    # gaze series are not stored continuously in the cache; rebuild quickly from raw
    from data_pipeline import BalladeerData
    d = BalladeerData()

    for p in meta['participants']:
        pid, user = p['pid'], p['user']
        # EEG continuous
        lo, hi = offsets[pid], offsets[pid + 1]
        eeg = cache['eeg'][lo:hi]
        n_eeg = len(eeg)
        t_eeg = np.arange(n_eeg) / eeg_sr
        n16 = int(np.floor(t_eeg[-1] * FS))
        t16 = np.arange(n16) / FS
        eeg16 = resample_series(eeg.astype(np.float64), t_eeg, t16)

        arr = d.participant_arrays(user)
        if arr is None:
            continue
        eeg_t_abs = arr['eeg_t']
        gt_abs = arr['gaze_t'] - eeg_t_abs[0]
        col = arr['gaze_col']
        row = arr['gaze_row']
        # gaze displacement at the 16 Hz grid
        t16_g = t16
        col16 = np.interp(t16_g, gt_abs, col)
        row16 = np.interp(t16_g, gt_abs, row)
        dcol = np.diff(col16, prepend=col16[0])
        drow = np.diff(row16, prepend=row16[0])
        dgaze16 = np.stack([dcol, drow], axis=1)
        out[pid] = (eeg16.astype(np.float32), dgaze16.astype(np.float32))
    return out


def main():
    cache = load_cache()
    print('building per-participant series ...')
    series = build_participant_series(cache)
    print(f'{len(series)} participants')
    payloads = [(pid, e, g, 12345 + pid) for pid, (e, g) in sorted(series.items())]
    with Pool(8) as pool:
        results = [r for r in pool.map(analyze_participant, payloads) if r is not None]
    with open(os.path.join(RESULTS, 'lagged_analysis.json'), 'w') as f:
        json.dump(results, f, indent=1)

    import numpy as np
    imp1 = np.array([r['eeg_to_gaze_imp'] for r in results])
    p1 = np.array([r['eeg_to_gaze_p'] for r in results])
    imp2 = np.array([r['gaze_to_eeg_imp'] for r in results])
    p2 = np.array([r['gaze_to_eeg_p'] for r in results])
    print(f"EEG->gaze: median imp={np.median(imp1):.5f}, "
          f"sig(p<.05)={np.sum(p1 < 0.05)}/{len(p1)}")
    print(f"gaze->EEG: median imp={np.median(imp2):.5f}, "
          f"sig(p<.05)={np.sum(p2 < 0.05)}/{len(p2)}")


if __name__ == '__main__':
    main()
