"""Verification that prohibited (future) gaze positions receive zero attention,
for both the fixed proportional mask (TempAlignEEGNet) and the learned binary
mask (TempStructNet).

Also saves the mean attention weights of a trained fold-0 TempAlign model for
figure regeneration, and runs the synthetic delay-recovery check for
TempStructNet.
"""

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_v2 import TempAlignEEGNet, TempStructNet, _proportional_mask
from run_main import load_cache, outer_folds, train_one, CFG
from run_ablations import AblationDS

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', 'results')


def fold0_datasets(cache, channels=None):
    folds = outer_folds(cache['pid'], seed=42)
    tr_users, te_users = folds[0]
    tr_idx_all = np.where(np.isin(cache['pid'], tr_users))[0]
    te_idx = np.where(np.isin(cache['pid'], te_users))[0]
    rng = np.random.RandomState(1000)
    inner_users = rng.permutation(tr_users)[:max(1, int(0.2 * len(tr_users)))]
    inner_sel = np.isin(cache['pid'][tr_idx_all], inner_users)
    inner_idx = tr_idx_all[inner_sel]
    tr_idx = tr_idx_all[~inner_sel]
    tr_ds = AblationDS(cache, tr_idx, channels=channels, augment=True)
    stats = tr_ds.stats()
    inner_ds = AblationDS(cache, inner_idx, channels=channels, stats=stats)
    te_ds = AblationDS(cache, te_idx, channels=channels, stats=stats)
    return tr_ds, inner_ds, te_ds


@torch.no_grad()
def check_fixed_mask(model, ds, device, n_batches=8):
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    blocked_ref = _proportional_mask(256, 50, device).unsqueeze(0).unsqueeze(0)
    max_blocked, total_mass, n = 0.0, 0.0, 0
    mean_attn = None
    sample_profiles = []
    profile_sum = np.zeros(256, dtype=np.float64)
    n_prof = 0
    for bi, b in enumerate(loader):
        out = model(b['eeg'].to(device), b['gaze'].to(device), b['mask'].to(device))
        w = out['attn_weights'][-1]  # (B, H, 256, 50)
        blocked = blocked_ref.expand_as(w)
        max_blocked = max(max_blocked, float(w[blocked].max()) if blocked.any() else 0.0)
        total_mass += float(w.sum(dim=-1).mean()) * w.shape[0]
        n += w.shape[0]
        mw = w.mean(dim=(0, 1)).cpu().numpy()
        mean_attn = mw if mean_attn is None else mean_attn + mw
        prof = w.max(dim=-1).values.mean(dim=1).cpu().numpy()  # (B, 256)
        for row in prof:
            if len(sample_profiles) < 5:
                sample_profiles.append([float(x) for x in row])
        profile_sum += prof.sum(axis=0)
        n_prof += prof.shape[0]
        if bi + 1 >= n_batches:
            break
    mean_attn /= max(1, min(n_batches, bi + 1))
    return {'max_attention_on_blocked': max_blocked,
            'mean_row_mass': total_mass / max(n, 1),
            'mean_attention': mean_attn.tolist(),
            'sample_profiles': sample_profiles,
            'mean_profile': (profile_sum / max(n_prof, 1)).tolist()}


@torch.no_grad()
def check_learned_mask(model, ds, device, n_batches=8):
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    max_blocked = 0.0
    for bi, b in enumerate(loader):
        out = model(b['eeg'].to(device), b['gaze'].to(device), b['mask'].to(device),
                    return_mask=True)
        w = out['attn_weights'][-1]
        g = out['mask']  # (256, 50) or (B?) - penalty applied uniformly
        blocked = (g < 0.5).unsqueeze(0).unsqueeze(0).expand_as(w)
        if blocked.any():
            max_blocked = max(max_blocked, float(w[blocked].max()))
        if bi + 1 >= n_batches:
            break
    return {'max_attention_on_blocked': max_blocked,
            'delay_normalized': float(model.delay.detach().cpu()),
            'bandwidth': float(model.bandwidth.detach().cpu())}


def synthetic_recovery(seeds=(0, 1, 2), target_offsets=(1.05, 1.15, 1.25)):
    """Synthetic generative test structurally mirroring the real task.

    Each sample: a 2.6 s EEG series (mixture of theta/alpha/beta sinusoids +
    noise); the gaze curve is the channel-mean EEG; the gaze context covers
    the first 2 s (50 uniform samples) and the target is the gaze change from
    just inside the window (x=0.99) to a time BEYOND the window whose offset
    varies across conditions (x in {1.05, 1.15, 1.25}). If the learned delay
    parameter tracked the generative timing it should differ across
    conditions; the experiment tests exactly that.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    S, n_ch, n_ext = 50, 14, 416  # 2.6 s @ 160 Hz-equivalent grid
    n_train, n_val = 2000, 500
    nt = n_train + n_val

    t = np.arange(n_ext) / 128.0
    freqs = [5.0, 10.0, 20.0]
    e_ext = np.zeros((nt, n_ch, n_ext), dtype=np.float32)
    for c in range(n_ch):
        for f in freqs:
            e_ext[:, c, :] += rng.randn(nt, 1) * np.sin(
                2 * np.pi * f * t + rng.rand(nt, 1) * 2 * np.pi)
    e_ext += 0.5 * rng.randn(*e_ext.shape)
    g_curve = e_ext.mean(axis=1)  # (nt, n_ext) clean gaze curve
    gnoisy = g_curve + 0.3 * rng.randn(*g_curve.shape)

    win_eeg = e_ext[:, :, :320]  # "2 s window" of the 2.6 s series
    idx_ctx = (np.arange(S) / S * 319).astype(int)
    gaze_ctx = gnoisy[:, idx_ctx]

    results = []
    for off in target_offsets:
        i_hi = int(off / 1.3 * (n_ext - 1))
        i_lo = int(0.99 / 1.3 * (n_ext - 1))
        targets = (gnoisy[:, i_hi] - gnoisy[:, i_lo])[:, None]
        targets = np.concatenate([targets, np.zeros_like(targets)], axis=1)
        targets = targets.astype(np.float32)

        class SynDS(torch.utils.data.Dataset):
            def __init__(self, eeg, gaze, tgt):
                self.eeg, self.gaze, self.tgt = eeg, gaze, tgt
                self.em = eeg.mean(axis=(0, 2), keepdims=True)
                self.es = eeg.std(axis=(0, 2), keepdims=True) + 1e-8
                self.tm = tgt.mean(axis=0)
                self.ts = tgt.std(axis=0) + 1e-8

            def __len__(self):
                return len(self.tgt)

            def __getitem__(self, k):
                e = ((self.eeg[k] - self.em[0]) / self.es[0]).astype(np.float32)
                g = np.stack([self.gaze[k], self.gaze[k]], -1).astype(np.float32)
                t = ((self.tgt[k] - self.tm) / self.ts).astype(np.float32)
                return {'eeg': torch.from_numpy(e), 'gaze': torch.from_numpy(g),
                        'mask': torch.ones(S, dtype=torch.bool),
                        'target': torch.from_numpy(t)}

        tr_ds = SynDS(win_eeg[:n_train], gaze_ctx[:n_train], targets[:n_train])
        in_ds = SynDS(win_eeg[n_train:], gaze_ctx[n_train:], targets[n_train:])
        for seed in seeds:
            model = TempStructNet(hidden_dim=48, n_heads=4, dropout=0.1)

            def on_epoch(m, ep):
                m.set_temperature(max(0.1, 1.0 * (0.1 / 1.0) ** (ep / 40)))
            model._on_epoch = on_epoch
            torch.manual_seed(seed)
            model, ep, best, _ = train_one(model, tr_ds, in_ds, CFG, device, seed)
            results.append({'target_offset': off, 'seed': seed,
                            'delta_learned': float(model.delay.detach().cpu()),
                            'bandwidth': float(model.bandwidth.detach().cpu()),
                            'epochs': ep})
            print(f"  offset={off} seed={seed} learned_delta={model.delay.item():.3f}")
    return results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache = load_cache()
    report = {}

    print('training TempAlign fold0 for mask verification ...')
    tr_ds, inner_ds, te_ds = fold0_datasets(cache)
    model = TempAlignEEGNet(hidden_dim=48, dropout=0.2)
    t0 = time.time()
    model, ep, best, hist = train_one(model, tr_ds, inner_ds, CFG, device, 42)
    report['fixed_mask'] = check_fixed_mask(model, te_ds, device)
    report['fixed_mask']['train_epochs'] = ep
    report['fixed_mask']['train_time'] = time.time() - t0
    report['fixed_mask']['inner_loss_history'] = [float(x) for x in hist]
    print('fixed mask max attention on blocked positions:',
          report['fixed_mask']['max_attention_on_blocked'])

    print('training TempStructNet fold0 ...')
    ts = TempStructNet(hidden_dim=48, dropout=0.2)

    def on_epoch(m, ep_):
        m.set_temperature(max(0.1, 1.0 * (0.1 / 1.0) ** (ep_ / CFG['epochs'])))
    ts._on_epoch = on_epoch
    ts, ep2, best2, _ = train_one(ts, tr_ds, inner_ds, CFG, device, 42)
    report['learned_mask'] = check_learned_mask(ts, te_ds, device)
    report['learned_mask']['delay_ms'] = report['learned_mask']['delay_normalized'] * 2000.0
    print('learned mask max attention on blocked positions:',
          report['learned_mask']['max_attention_on_blocked'])

    print('synthetic delay recovery ...')
    report['synthetic'] = synthetic_recovery()

    profiles = {k: report['fixed_mask'][k] for k in
                ('max_attention_on_blocked', 'mean_row_mass',
                 'mean_attention', 'sample_profiles', 'mean_profile',
                 'train_epochs')}
    with open(os.path.join(RESULTS, 'attention_profiles.json'), 'w') as f:
        json.dump(profiles, f, indent=1)
    print('saved', os.path.join(RESULTS, 'attention_profiles.json'))

    with open(os.path.join(RESULTS, 'mask_verification.json'), 'w') as f:
        json.dump(report, f, indent=1)
    print('saved', os.path.join(RESULTS, 'mask_verification.json'))


if __name__ == '__main__':
    main()
