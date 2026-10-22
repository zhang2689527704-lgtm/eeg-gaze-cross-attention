"""Ablation and control experiments (same nested CV protocol as run_main).

Variants:
  NoMask           : TempAlignEEGNet with the temporal mask disabled
  FrontalExcluded  : 8 posterior-centric channels (AF3/AF4/F7/F8/F3/F4 removed)
  TempStructNet    : learned mask (corrected Gumbel math), records delta/beta
  BandTheta/Alpha/Beta : single-band EEG (bandpassed continuous signal)
  PermutedEEG      : training EEG windows scrambled across windows (train only)
  PermutedTarget   : training targets scrambled (train only)
  LTC_param_matched: LTC tower with hidden=64 (~ parameter matched)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from scipy import signal
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_v2 import TempAlignEEGNet, TempStructNet, count_params
from baselines_v2 import FusionBaseline, LTCTower
from metrics import evaluate_predictions
from run_main import load_cache, outer_folds, train_one, predict, CFG

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
RESULTS = os.path.join(ROOT, 'results')
PREDICTIONS = os.path.join(ROOT, 'predictions')

CH_ALL = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
CH_FRONTAL_EXCLUDED = ['FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6']
CH_IDX = {c: i for i, c in enumerate(CH_ALL)}


class AblationDS(Dataset):
    def __init__(self, cache, idx, eeg_src=None, channels=None, stats=None,
                 eeg_perm=None, tgt_perm=None, augment=False, noise=0.05):
        self.c = cache
        self.idx = idx
        self.eeg_src = eeg_src if eeg_src is not None else cache['eeg']
        self.ch = [CH_IDX[c] for c in channels] if channels else None
        self.eeg_perm = eeg_perm
        self.tgt_perm = tgt_perm
        self.augment = augment
        self.noise = noise
        eegs = np.stack([self._eeg(k) for k in range(len(idx))])
        if stats is None:
            self.eeg_mean = eegs.mean(axis=(0, 1)).astype(np.float32)
            self.eeg_std = (eegs.std(axis=(0, 1)) + 1e-8).astype(np.float32)
            tgt = cache['targets'][idx if tgt_perm is None else tgt_perm]
            self.tgt_mean = tgt.mean(axis=0).astype(np.float32)
            self.tgt_std = (tgt.std(axis=0) + 1e-8).astype(np.float32)
        else:
            self.eeg_mean, self.eeg_std, self.tgt_mean, self.tgt_std = stats
        self.eegs = eegs

    def _eeg(self, k):
        i = self.idx[k]
        if self.eeg_perm is not None:
            i = self.eeg_perm[k]
        off = self.c['offsets'][self.c['pid'][i]]
        s = self.c['start'][i]
        win = self.eeg_src[off + s: off + s + 256]
        if self.ch is not None:
            win = win[:, self.ch]
        return win

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = self.idx[k]
        eeg = (self.eegs[k] - self.eeg_mean) / self.eeg_std
        if self.augment:
            eeg = eeg + np.random.randn(*eeg.shape).astype(np.float32) * self.noise
        eeg = torch.from_numpy(eeg.T.copy())
        gaze = torch.from_numpy(self.c['gaze50'][i])
        tgt_i = i if self.tgt_perm is None else self.tgt_perm[k]
        t = (self.c['targets'][tgt_i] - self.tgt_mean) / self.tgt_std
        return {'eeg': eeg, 'gaze': gaze,
                'mask': torch.ones(50, dtype=torch.bool),
                'target': torch.from_numpy(t.astype(np.float32)),
                'pid': int(self.c['pid'][i])}

    def stats(self):
        return self.eeg_mean, self.eeg_std, self.tgt_mean, self.tgt_std


def band_filter_cache(cache, band):
    lo, hi = band
    fs = 128.0
    b, a = signal.butter(4, [lo / (fs / 2), hi / (fs / 2)], btype='band')
    out = np.empty_like(cache['eeg'])
    offs = cache['offsets']
    for k in range(len(offs) - 1):
        seg = cache['eeg'][offs[k]:offs[k + 1]]
        out[offs[k]:offs[k + 1]] = signal.filtfilt(
            b, a, seg.astype(np.float64), axis=0).astype(np.float32)
    return out


def make_folds(cache):
    return outer_folds(cache['pid'], seed=42)


def run_variant(name, model_fn, cache, folds, fold_ids, seeds, tag,
                channels=None, eeg_src=None, perm_eeg=False, perm_target=False,
                on_epoch=None):
    results_path = os.path.join(RESULTS, f'results_{tag}.json')
    results = json.load(open(results_path)) if os.path.exists(results_path) else {}
    results.setdefault(name, {'runs': [], 'params': None})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for fold in fold_ids:
        tr_users, te_users = folds[fold]
        tr_idx_all = np.where(np.isin(cache['pid'], tr_users))[0]
        te_idx = np.where(np.isin(cache['pid'], te_users))[0]
        rng = np.random.RandomState(1000 + fold)
        inner_users = rng.permutation(tr_users)[:max(1, int(0.2 * len(tr_users)))]
        inner_sel = np.isin(cache['pid'][tr_idx_all], inner_users)
        inner_idx = tr_idx_all[inner_sel]
        tr_idx = tr_idx_all[~inner_sel]

        for seed in seeds:
            if any(r['fold'] == fold and r['seed'] == seed for r in results[name]['runs']):
                print(f'  skip {name} f{fold} s{seed}')
                continue
            t0 = time.time()

            # scrambled source indices (train only)
            tr_eeg_perm = None
            tr_tgt_perm = None
            if perm_eeg:
                rng2 = np.random.RandomState(7000 + fold * 10 + seed)
                tr_eeg_perm = tr_idx[rng2.permutation(len(tr_idx))]
            if perm_target:
                rng2 = np.random.RandomState(8000 + fold * 10 + seed)
                tr_tgt_perm = tr_idx[rng2.permutation(len(tr_idx))]

            tr_ds = AblationDS(cache, tr_idx, eeg_src=eeg_src, channels=channels,
                               eeg_perm=tr_eeg_perm, tgt_perm=tr_tgt_perm,
                               augment=True)
            stats = tr_ds.stats()
            inner_ds = AblationDS(cache, inner_idx, eeg_src=eeg_src, channels=channels,
                                  stats=stats)
            te_ds = AblationDS(cache, te_idx, eeg_src=eeg_src, channels=channels,
                               stats=stats)

            model = model_fn()
            if results[name]['params'] is None:
                results[name]['params'] = count_params(model)

            if on_epoch is not None:
                model._on_epoch = on_epoch
            model, ep, best, hist = train_one(model, tr_ds, inner_ds, CFG, device, seed)
            targets, preds, pids = predict(model, te_ds, device)
            m = evaluate_predictions(targets, preds)
            m.update({'fold': fold, 'seed': seed, 'epochs': ep,
                      'inner_best': best, 'time': time.time() - t0})
            if hasattr(model, 'delay'):
                m['delay'] = float(model.delay.detach().cpu())
                m['bandwidth'] = float(model.bandwidth.detach().cpu())
            results[name]['runs'].append(m)
            print(f"  {name} f{fold} s{seed}: r_col={m['r_col']:.3f} "
                  f"R2_col={m['R2_col']:.3f} rmse={m['rmse']:.3f} "
                  f"delay={m.get('delay')} ep={ep} ({m['time']:.0f}s)")

            np.savez_compressed(
                os.path.join(PREDICTIONS,
                             f'preds_{tag}_{name}_f{fold}_s{seed}.npz'),
                targets=targets, preds=preds, pids=pids)
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', default='NoMask,FrontalExcluded,TempStructNet,PermutedEEG,PermutedTarget')
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--seeds', default='42,43,44')
    ap.add_argument('--tag', default='ablation')
    args = ap.parse_args()

    cache = load_cache()
    folds = make_folds(cache)
    fold_ids = [int(x) for x in args.folds.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]
    variants = args.variants.split(',')
    print(f'cache: {cache["n_qc"]} qc-passed windows, '
          f'{len(np.unique(cache["pid"]))} participants')

    for v in variants:
        print(f'=== {v} ===')
        if v == 'NoMask':
            run_variant(v, lambda: TempAlignEEGNet(use_mask=False, hidden_dim=48,
                                                   dropout=0.2),
                        cache, folds, fold_ids, seeds, args.tag)
        elif v == 'FrontalExcluded':
            run_variant(v, lambda: TempAlignEEGNet(n_channels=len(CH_FRONTAL_EXCLUDED),
                                                   hidden_dim=48, dropout=0.2),
                        cache, folds, fold_ids, seeds, args.tag,
                        channels=CH_FRONTAL_EXCLUDED)
        elif v == 'TempStructNet':
            def ts_fn():
                m = TempStructNet(hidden_dim=48, dropout=0.2)

                def on_epoch(model, ep):
                    tau = max(0.1, 1.0 * (0.1 / 1.0) ** (ep / CFG['epochs']))
                    model.set_temperature(tau)
                m._on_epoch = on_epoch
                return m
            # temperature hook: train_one calls m._on_epoch at each epoch start
            run_variant(v, ts_fn, cache, folds, fold_ids, seeds, args.tag)
        elif v in ('BandTheta', 'BandAlpha', 'BandBeta'):
            bands = {'BandTheta': (4, 8), 'BandAlpha': (8, 13), 'BandBeta': (13, 30)}
            src = band_filter_cache(cache, bands[v])
            run_variant(v, lambda: TempAlignEEGNet(hidden_dim=48, dropout=0.2),
                        cache, folds, fold_ids, seeds, args.tag, eeg_src=src)
        elif v == 'PermutedEEG':
            run_variant(v, lambda: TempAlignEEGNet(hidden_dim=48, dropout=0.2),
                        cache, folds, fold_ids, seeds, args.tag, perm_eeg=True)
        elif v == 'PermutedTarget':
            run_variant(v, lambda: TempAlignEEGNet(hidden_dim=48, dropout=0.2),
                        cache, folds, fold_ids, seeds, args.tag, perm_target=True)
        elif v == 'LTC_param_matched':
            run_variant(v, lambda: FusionBaseline(LTCTower(14, 64), 64),
                        cache, folds, fold_ids, seeds, args.tag)
        else:
            raise ValueError(v)


if __name__ == '__main__':
    main()
