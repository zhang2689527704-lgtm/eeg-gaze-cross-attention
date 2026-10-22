"""Record the learned delay and bandwidth at every training epoch for one
representative learned-mask run (fold 0, seed 42). Used for Fig. 8(c,d).

Usage:
  python run_structure_history.py
  python run_structure_history.py --epochs 50 --out learned_structure_history_50.json
"""

import argparse
import json
import os

import numpy as np
import torch

from run_main import CFG, WindowDS, load_cache, outer_folds, train_one
from models_v2 import TempStructNet

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', 'results')
os.makedirs(RESULTS, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--out', default='learned_structure_history.json')
    args = ap.parse_args()

    cfg = dict(CFG)
    if args.epochs:
        cfg['epochs'] = args.epochs
        cfg['patience'] = args.epochs
        cfg['min_epochs'] = args.epochs

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache = load_cache()
    folds = outer_folds(cache['pid'], seed=42)
    tr_users, _ = folds[0]
    tr_idx_all = np.where(np.isin(cache['pid'], tr_users))[0]
    rng = np.random.RandomState(1000)
    inner_users = rng.permutation(tr_users)[
        :max(1, int(0.2 * len(tr_users)))]
    inner_sel = np.isin(cache['pid'][tr_idx_all], inner_users)
    inner_idx = tr_idx_all[inner_sel]
    tr_idx = tr_idx_all[~inner_sel]

    tr_ds = WindowDS(cache, tr_idx, augment=True)
    em, es, tm, ts = tr_ds.stats()
    inner_ds = WindowDS(cache, inner_idx, em, es, tm, ts)

    model = TempStructNet(hidden_dim=48, dropout=0.2)
    hist = {'epoch': [], 'delay': [], 'bandwidth': [], 'temperature': []}

    def on_epoch(m, ep):
        m.set_temperature(max(0.1, 1.0 * (0.1 / 1.0) ** (ep / cfg['epochs'])))
        hist['epoch'].append(int(ep))
        hist['delay'].append(float(m.delay.detach()))
        hist['bandwidth'].append(float(m.bandwidth.detach()))
        hist['temperature'].append(float(m.temperature))

    model._on_epoch = on_epoch
    model, ep, best, loss_hist = train_one(
        model, tr_ds, inner_ds, cfg, device, seed=42)

    hist['epoch'].append(int(ep) + 1)
    hist['delay'].append(float(model.delay.detach()))
    hist['bandwidth'].append(float(model.bandwidth.detach()))
    hist['temperature'].append(float(model.temperature))
    hist['final_epoch'] = int(ep)
    hist['inner_loss'] = [float(x) for x in loss_hist]

    path = os.path.join(RESULTS, args.out)
    with open(path, 'w') as f:
        json.dump(hist, f, indent=1)
    print('saved', path)
    print('final delay %.4f bandwidth %.4f' % (
        hist['delay'][-1], hist['bandwidth'][-1]))


if __name__ == '__main__':
    main()
