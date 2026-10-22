"""Per-channel importance via zero-out (replace channel by training mean),
computed on trained TempAlign models across the 5 outer folds (seed 42).
"""

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_v2 import TempAlignEEGNet
from run_main import load_cache, outer_folds, train_one, CFG, WindowDS

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', 'results')
CHALL = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']


@torch.no_grad()
def evaluate_with_zeroed(model, ds, device, zero_ch=None):
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    model.eval()
    se, n = 0.0, 0
    for b in loader:
        eeg = b['eeg'].to(device).clone()
        if zero_ch is not None:
            eeg[:, zero_ch, :] = 0.0  # normalized mean
        out = model(eeg, b['gaze'].to(device), b['mask'].to(device))
        diff = out['pred'].cpu() - b['target']
        se += float((diff ** 2).sum())
        n += diff.numel()
    return se / n


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache = load_cache()
    folds = outer_folds(cache['pid'], seed=42)
    results = {'channels': CHALL, 'folds': []}

    for fold in range(5):
        tr_users, te_users = folds[fold]
        tr_idx_all = np.where(np.isin(cache['pid'], tr_users))[0]
        te_idx = np.where(np.isin(cache['pid'], te_users))[0]
        rng = np.random.RandomState(1000 + fold)
        inner_users = rng.permutation(tr_users)[:max(1, int(0.2 * len(tr_users)))]
        sel = np.isin(cache['pid'][tr_idx_all], inner_users)
        inner_idx, tr_idx = tr_idx_all[sel], tr_idx_all[~sel]

        tr_ds = WindowDS(cache, tr_idx, augment=True)
        em, es, tm, ts = tr_ds.stats()
        inner_ds = WindowDS(cache, inner_idx, em, es, tm, ts)
        te_ds = WindowDS(cache, te_idx, em, es, tm, ts)

        t0 = time.time()
        model = TempAlignEEGNet(hidden_dim=48, dropout=0.2)
        model, ep, best, _ = train_one(model, tr_ds, inner_ds, CFG, device, 42)
        base_mse = evaluate_with_zeroed(model, te_ds, device, None)
        row = {'fold': fold, 'base_mse': base_mse, 'epochs': ep,
               'delta_mse': []}
        for c in range(14):
            m = evaluate_with_zeroed(model, te_ds, device, c)
            row['delta_mse'].append(m - base_mse)
        results['folds'].append(row)
        print(f"fold {fold}: base_mse={base_mse:.4f} ep={ep} ({time.time()-t0:.0f}s)")

    arr = np.array([r['delta_mse'] for r in results['folds']])
    results['mean_delta_mse'] = arr.mean(axis=0).tolist()
    results['std_delta_mse'] = arr.std(axis=0).tolist()
    results['ranking'] = [CHALL[i] for i in np.argsort(-arr.mean(axis=0))]
    with open(os.path.join(RESULTS, 'channel_zeroout.json'), 'w') as f:
        json.dump(results, f, indent=1)
    print('ranking:', results['ranking'])


if __name__ == '__main__':
    main()
