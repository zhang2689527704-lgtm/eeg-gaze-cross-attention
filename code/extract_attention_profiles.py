"""Extract per-head attention profiles for Fig. 6(b).

Trains the fold-0 fixed-mask model with the same protocol and seed used for
mask verification, then records, for five held-out samples, the maximum
attention weight over gaze positions for each head and the max over heads.
"""

import json
import os

import numpy as np
import torch

from run_main import CFG, train_one
from verify_mask import fold0_datasets
from run_main import load_cache
from models_v2 import TempAlignEEGNet

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', 'results')
os.makedirs(RESULTS, exist_ok=True)


@torch.no_grad()
def profiles(model, ds, device, n_batches=8):
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    model.eval()
    sample_max, sample_heads = [], []
    mean_attn = None
    mean_max = np.zeros(256, dtype=np.float64)
    n_max = 0
    for bi, b in enumerate(loader):
        out = model(b['eeg'].to(device), b['gaze'].to(device),
                    b['mask'].to(device))
        w = out['attn_weights'][-1]  # (B, H, 256, 50)
        mw = w.mean(dim=(0, 1)).cpu().numpy()
        mean_attn = mw if mean_attn is None else mean_attn + mw
        ph = w.max(dim=-1).values.cpu().numpy()  # (B, H, 256)
        pm = ph.max(axis=1)  # (B, 256)
        for j in range(pm.shape[0]):
            if len(sample_max) < 5:
                sample_max.append([float(x) for x in pm[j]])
                sample_heads.append([[float(x) for x in ph[j, h]]
                                     for h in range(ph.shape[1])])
        mean_max += pm.sum(axis=0)
        n_max += pm.shape[0]
        if bi + 1 >= n_batches:
            break
    mean_attn /= max(1, min(n_batches, bi + 1))
    return {
        'mean_attention': mean_attn.tolist(),
        'sample_profiles_max': sample_max,
        'sample_profiles_heads': sample_heads,
        'mean_profile_max': (mean_max / max(n_max, 1)).tolist(),
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache = load_cache()
    tr_ds, inner_ds, te_ds = fold0_datasets(cache)
    model = TempAlignEEGNet(hidden_dim=48, dropout=0.2)
    model, ep, best, hist = train_one(model, tr_ds, inner_ds, CFG, device, 42)
    out = profiles(model, te_ds, device)
    out['train_epochs'] = int(ep)
    path = os.path.join(RESULTS, 'attention_profiles.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print('saved', path, 'epochs', ep)


if __name__ == '__main__':
    main()
