"""Main experiment runner: nested participant-level 5-fold cross-validation.

Protocol (identical for every model):
  - outer 5-fold split by participant (seed 42); held-out fold never used for
    model selection
  - inner split: 20% of training participants held out for early stopping
    (nested CV; the reported test folds are untouched)
  - normalization statistics computed on inner-train windows only
  - AdamW lr=5e-4 wd=1e-3, Huber(delta=1), batch 64, warmup 3 epochs +
    cosine annealing, grad clip 1.0, EEG Gaussian-noise augmentation
    (sigma=0.05), early stopping patience 8 (min 5 epochs), max 40 epochs
  - metrics: r, out-of-sample predictive R^2, RMSE (z units) on outer fold

Statistical baselines (no training): mean predictor, persistence (no change),
persistence (last-step trend), ridge regression on the gaze history.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import RidgeCV

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_v2 import (TempAlignEEGNet, TempAlignGazeOnly, TempAlignEEGOnly,
                       count_params)
from baselines_v2 import (build_lstm_dual, build_transformer_dual,
                          build_ltc_dual, build_wavelet_dual)
from metrics import evaluate_predictions

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
CACHE = os.path.join(ROOT, 'cache')
RESULTS = os.path.join(ROOT, 'results')
PREDICTIONS = os.path.join(ROOT, 'predictions')
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(PREDICTIONS, exist_ok=True)

SEEDS = [42, 43, 44]
N_FOLDS = 5


def load_cache():
    z = np.load(os.path.join(CACHE, 'windows.npz'))
    meta = json.load(open(os.path.join(CACHE, 'meta.json')))
    qc = z['qc_pass']
    keep = np.where(qc)[0]
    return {
        'eeg': z['eeg'], 'offsets': np.array(meta['eeg_offsets'], dtype=np.int64),
        'pid': z['win_pid'][keep], 'start': z['win_start'][keep],
        'targets': z['targets'][keep], 'gaze50': z['gaze50'][keep],
        'meta': meta, 'n_total': len(z['win_pid']), 'n_qc': int(qc.sum()),
    }


class WindowDS(Dataset):
    def __init__(self, cache, idx, eeg_mean=None, eeg_std=None,
                 tgt_mean=None, tgt_std=None, augment=False, noise=0.05):
        self.c = cache
        self.idx = idx
        self.augment = augment
        self.noise = noise
        eegs = np.stack([self._eeg(i) for i in idx])  # (N, 256, 14)
        if eeg_mean is None:
            self.eeg_mean = eegs.mean(axis=(0, 1)).astype(np.float32)
            self.eeg_std = (eegs.std(axis=(0, 1)) + 1e-8).astype(np.float32)
            self.tgt_mean = cache['targets'][idx].mean(axis=0).astype(np.float32)
            self.tgt_std = (cache['targets'][idx].std(axis=0) + 1e-8).astype(np.float32)
        else:
            self.eeg_mean, self.eeg_std = eeg_mean, eeg_std
            self.tgt_mean, self.tgt_std = tgt_mean, tgt_std
        self.eegs = eegs

    def _eeg(self, i):
        off = self.c['offsets'][self.c['pid'][i]]
        s = self.c['start'][i]
        return self.c['eeg'][off + s: off + s + 256]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = self.idx[k]
        eeg = (self.eegs[k] - self.eeg_mean) / self.eeg_std
        if self.augment:
            eeg = eeg + np.random.randn(*eeg.shape).astype(np.float32) * self.noise
        eeg = torch.from_numpy(eeg.T.copy())  # (14, 256)
        gaze = torch.from_numpy(self.c['gaze50'][i])  # (50, 2)
        mask = torch.ones(50, dtype=torch.bool)
        t = (self.c['targets'][i] - self.tgt_mean) / self.tgt_std
        return {'eeg': eeg, 'gaze': gaze, 'mask': mask,
                'target': torch.from_numpy(t.astype(np.float32)),
                'pid': int(self.c['pid'][i])}

    def stats(self):
        return self.eeg_mean, self.eeg_std, self.tgt_mean, self.tgt_std


def outer_folds(pid, seed=42, n_folds=N_FOLDS):
    rng = np.random.RandomState(seed)
    users = np.unique(pid)
    rng.shuffle(users)
    folds = []
    size = len(users) // n_folds
    for f in range(n_folds):
        lo = f * size
        hi = lo + size if f < n_folds - 1 else len(users)
        val = set(users[lo:hi].tolist())
        tr = [u for u in users if u not in val]
        folds.append((np.array(tr), np.array(sorted(val))))
    return folds


def train_one(model, tr_ds, inner_ds, cfg, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)
    criterion = nn.HuberLoss(delta=1.0)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    tr_loader = DataLoader(tr_ds, batch_size=cfg['batch'], shuffle=True, drop_last=True)
    in_loader = DataLoader(inner_ds, batch_size=128, shuffle=False)

    warm, total = cfg['warmup'], cfg['epochs']
    base_lrs = [pg['lr'] for pg in opt.param_groups]

    best = float('inf')
    best_state = None
    wait = 0
    hist = []
    for ep in range(1, total + 1):
        scale = (ep / warm) if ep <= warm else 0.5 * (1 + np.cos(np.pi * (ep - warm) / max(1, total - warm)))
        for pg, blr in zip(opt.param_groups, base_lrs):
            pg['lr'] = max(1e-6, blr * scale)
        if hasattr(model, '_on_epoch'):
            model._on_epoch(model, ep)

        model.train()
        for b in tr_loader:
            eeg = b['eeg'].to(device)
            gaze = b['gaze'].to(device)
            mask = b['mask'].to(device)
            tgt = b['target'].to(device)
            opt.zero_grad()
            out = model(eeg, gaze, mask)
            loss = criterion(out['pred'], tgt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        vl, n = 0.0, 0
        with torch.no_grad():
            for b in in_loader:
                out = model(b['eeg'].to(device), b['gaze'].to(device), b['mask'].to(device))
                vl += criterion(out['pred'], b['target'].to(device)).item()
                n += 1
        vl /= max(n, 1)
        hist.append(vl)
        if vl < best - 1e-5:
            best = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if ep >= cfg['min_epochs'] and wait >= cfg['patience']:
            break
    model.load_state_dict(best_state)
    return model, ep, best, hist


@torch.no_grad()
def predict(model, ds, device):
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    model.eval()
    preds, tars, pids = [], [], []
    for b in loader:
        out = model(b['eeg'].to(device), b['gaze'].to(device), b['mask'].to(device))
        preds.append(out['pred'].cpu().numpy())
        tars.append(b['target'].numpy())
        pids.append(b['pid'].numpy())
    return np.concatenate(tars), np.concatenate(preds), np.concatenate(pids)


def stat_baselines(cache, tr_idx, te_idx):
    """Mean / persistence / ridge baselines in z-units, fit only on tr_idx."""
    tgt_tr = cache['targets'][tr_idx]
    mu, sd = tgt_tr.mean(axis=0), tgt_tr.std(axis=0) + 1e-8
    tgt_te = (cache['targets'][te_idx] - mu) / sd

    out = {}
    # mean predictor (raw train mean)
    pred_mean = np.tile(mu, (len(te_idx), 1))
    out['MeanPredictor'] = ((pred_mean - mu) / sd, tgt_te)
    # persistence: no change
    pred_zero = np.zeros((len(te_idx), 2))
    out['PersistenceZero'] = ((pred_zero - mu) / sd, tgt_te)
    # persistence: last-step trend (40 ms) in gaze space
    g = cache['gaze50'][te_idx]
    step = g[:, -1, :] - g[:, -2, :]
    out['PersistenceTrend'] = ((step - mu) / sd, tgt_te)
    # ridge on the full gaze history
    Xtr = cache['gaze50'][tr_idx].reshape(len(tr_idx), -1)
    Xte = cache['gaze50'][te_idx].reshape(len(te_idx), -1)
    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]).fit(Xtr, tgt_tr)
    pred_r = ridge.predict(Xte)
    out['RidgeGaze'] = ((pred_r - mu) / sd, tgt_te)
    return out


MODEL_FNS = {
    'TempAlignEEGNet': lambda: TempAlignEEGNet(hidden_dim=48, n_attn_layers=2, n_heads=4, dropout=0.2),
    'GazeOnly': lambda: TempAlignGazeOnly(hidden_dim=48, n_attn_layers=2, n_heads=4, dropout=0.2),
    'EEGOnly': lambda: TempAlignEEGOnly(hidden_dim=48, n_attn_layers=2, n_heads=4, dropout=0.2),
    'LSTM': lambda: build_lstm_dual(hidden_dim=48),
    'Transformer': lambda: build_transformer_dual(hidden_dim=48),
    'LTC': lambda: build_ltc_dual(hidden_dim=48),
    'WaveletTransformer': lambda: build_wavelet_dual(hidden_dim=48),
}

CFG = {'lr': 5e-4, 'wd': 1e-3, 'batch': 64, 'epochs': 40, 'warmup': 3,
       'patience': 8, 'min_epochs': 5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default=','.join(MODEL_FNS.keys()))
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--seeds', default=','.join(map(str, SEEDS)))
    ap.add_argument('--tag', default='main')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    cache = load_cache()
    print(f"windows: total={cache['n_total']} qc_pass={cache['n_qc']} "
          f"participants={len(np.unique(cache['pid']))}")

    folds = outer_folds(cache['pid'], seed=42)
    fold_ids = [int(x) for x in args.folds.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]
    model_names = args.models.split(',')

    results_path = os.path.join(RESULTS, f'results_{args.tag}.json')
    results = json.load(open(results_path)) if os.path.exists(results_path) else {}

    for name in model_names:
        if name == 'StatBaselines':
            continue  # handled after the trained models (see block below)
        results.setdefault(name, {'runs': [], 'params': None})
        for fold in fold_ids:
            tr_users, te_users = folds[fold]
            tr_mask_all = np.isin(cache['pid'], tr_users)
            te_mask = np.isin(cache['pid'], te_users)
            tr_idx_all = np.where(tr_mask_all)[0]
            te_idx = np.where(te_mask)[0]

            # inner split: 20% of training participants for early stopping
            rng = np.random.RandomState(1000 + fold)
            inner_users = rng.permutation(tr_users)[: max(1, int(0.2 * len(tr_users)))]
            inner_mask = np.isin(cache['pid'][tr_idx_all], inner_users)
            inner_idx = tr_idx_all[inner_mask]
            tr_idx = tr_idx_all[~inner_mask]

            for seed in seeds:
                if any(r['fold'] == fold and r['seed'] == seed for r in results[name]['runs']):
                    print(f'  skip {name} fold{fold} seed{seed}')
                    continue
                t0 = time.time()
                tr_ds = WindowDS(cache, tr_idx, augment=True)
                em, es, tm, ts = tr_ds.stats()
                inner_ds = WindowDS(cache, inner_idx, em, es, tm, ts)
                te_ds = WindowDS(cache, te_idx, em, es, tm, ts)
                model = MODEL_FNS[name]()
                if results[name]['params'] is None:
                    results[name]['params'] = count_params(model)
                model, ep, best, hist = train_one(model, tr_ds, inner_ds, CFG, device, seed)
                targets, preds, pids = predict(model, te_ds, device)
                m = evaluate_predictions(targets, preds)
                m.update({'fold': fold, 'seed': seed, 'epochs': ep,
                          'inner_best': best, 'time': time.time() - t0})
                results[name]['runs'].append(m)
                print(f"  {name} f{fold} s{seed}: r_col={m['r_col']:.3f} "
                      f"R2_col={m['R2_col']:.3f} rmse={m['rmse']:.3f} "
                      f"ep={ep} ({m['time']:.0f}s)")

                np.savez_compressed(
                    os.path.join(PREDICTIONS,
                                 f'preds_{args.tag}_{name}_f{fold}_s{seed}.npz'),
                    targets=targets, preds=preds, pids=pids)
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=1)

    # statistical baselines for every fold (stored under 'StatBaselines')
    if 'StatBaselines' in model_names:
        results.setdefault('StatBaselines', {'runs': [], 'params': 0})
        for fold in fold_ids:
            if any(r['fold'] == fold for r in results['StatBaselines']['runs']):
                continue
            tr_users, te_users = folds[fold]
            tr_idx = np.where(np.isin(cache['pid'], tr_users))[0]
            te_idx = np.where(np.isin(cache['pid'], te_users))[0]
            for bname, (pred, tgt) in stat_baselines(cache, tr_idx, te_idx).items():
                m = evaluate_predictions(tgt, pred)
                m.update({'fold': fold, 'seed': 0, 'name': bname})
                results['StatBaselines']['runs'].append(m)
                np.savez_compressed(
                    os.path.join(PREDICTIONS,
                                 f'preds_{args.tag}_StatBaselines_{bname}_f{fold}.npz'),
                    targets=tgt, preds=pred, pids=cache['pid'][te_idx])
                print(f"  StatBaselines {bname} f{fold}: r_col={m['r_col']:.3f} "
                      f"R2_col={m['R2_col']:.3f}")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=1)
    print('done')


if __name__ == '__main__':
    main()
