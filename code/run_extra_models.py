"""Masked LSTM / Transformer baselines under the nested CV protocol.

Each baseline keeps its own LSTM or Transformer EEG tower and replaces the
unmasked fusion with CausalEEGNet's temporally masked cross-attention, so the
comparison isolates the EEG encoder while holding the mask fixed.

Usage:
  python run_extra_models.py --models MaskedLSTM,MaskedTransformer \
      --folds 0,1,2,3,4 --seeds 42,43,44
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from run_main import (CFG, WindowDS, load_cache, outer_folds, predict,
                      train_one)
from models_v2 import (AttentionBlock, GazeStateEncoder,
                       TemporalAttentionHead, count_params)
from baselines_v2 import LSTMTower, TransformerTower
from metrics import evaluate_predictions

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
RESULTS = os.path.join(ROOT, 'results')
PREDICTIONS = os.path.join(ROOT, 'predictions')
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(PREDICTIONS, exist_ok=True)


class MaskedFusionBaseline(nn.Module):
    """EEG tower + gaze GRU + temporally masked cross-attention + head."""

    def __init__(self, tower, hidden_dim=48, n_heads=4, n_layers=2,
                 dropout=0.2):
        super().__init__()
        self.eeg_tower = tower
        self.gaze_encoder = GazeStateEncoder(2, hidden_dim)
        self.blocks = nn.ModuleList([
            AttentionBlock(hidden_dim, n_heads, dropout, mode='cross',
                           use_mask=True)
            for _ in range(n_layers)])
        self.pred_head = TemporalAttentionHead(hidden_dim, dropout=dropout)

    def forward(self, eeg, gaze_context, gaze_mask=None):
        x = self.eeg_tower(eeg)
        gaze_feat = self.gaze_encoder(gaze_context, gaze_mask)
        for blk in self.blocks:
            x, _ = blk(x, gaze_feat, gaze_mask)
        return {'pred': self.pred_head(x)}


MODELS = {
    'MaskedLSTM': lambda: MaskedFusionBaseline(LSTMTower(14, 48, 2, 0.2)),
    'MaskedTransformer': lambda: MaskedFusionBaseline(
        TransformerTower(14, 48, 4, 2, 0.2)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--seeds', default='42,43,44')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    cache = load_cache()
    print(f"windows: qc_pass={cache['n_qc']} "
          f"participants={len(np.unique(cache['pid']))}")

    folds = outer_folds(cache['pid'], seed=42)
    fold_ids = [int(x) for x in args.folds.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]
    names = args.models.split(',')

    path = os.path.join(RESULTS, 'extra_models_cv.json')
    results = json.load(open(path)) if os.path.exists(path) else {}

    for name in names:
        results.setdefault(name, {'runs': [], 'params': None})
        for fold in fold_ids:
            tr_users, te_users = folds[fold]
            tr_idx_all = np.where(np.isin(cache['pid'], tr_users))[0]
            te_idx = np.where(np.isin(cache['pid'], te_users))[0]

            rng = np.random.RandomState(1000 + fold)
            inner_users = rng.permutation(tr_users)[
                :max(1, int(0.2 * len(tr_users)))]
            inner_mask = np.isin(cache['pid'][tr_idx_all], inner_users)
            inner_idx = tr_idx_all[inner_mask]
            tr_idx = tr_idx_all[~inner_mask]

            for seed in seeds:
                if any(r['fold'] == fold and r['seed'] == seed
                       for r in results[name]['runs']):
                    print(f'  skip {name} f{fold} s{seed}')
                    continue
                t0 = time.time()
                tr_ds = WindowDS(cache, tr_idx, augment=True)
                em, es, tm, ts = tr_ds.stats()
                inner_ds = WindowDS(cache, inner_idx, em, es, tm, ts)
                te_ds = WindowDS(cache, te_idx, em, es, tm, ts)

                model = MODELS[name]()
                if results[name]['params'] is None:
                    results[name]['params'] = count_params(model)
                model, ep, best, hist = train_one(model, tr_ds, inner_ds, CFG,
                                                  device, seed)
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
                                 f'preds_main_{name}_f{fold}_s{seed}.npz'),
                    targets=targets, preds=preds, pids=pids)
                with open(path, 'w') as f:
                    json.dump(results, f, indent=1)
    print('done')


if __name__ == '__main__':
    main()
