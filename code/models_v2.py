"""CausalEEGNet and its variants.

CausalEEGNet is implemented as TempAlignEEGNet (fixed proportional temporal
mask); TempStructNet is the learned-mask variant. The learned mask is trained
with a straight-through Gumbel-Softmax sample and evaluated as
G = 1[soft_mask > 0.5]; prohibited positions receive an additive -1e9 penalty,
which underflows to exactly zero attention after softmax. The earliest gaze
sample remains attendable in every row, so no query row is fully masked.
The delay parameter delta is reported in normalized window units; its
conversion to milliseconds is delta * window_sec * 1000.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

MASK_NEG = -1e9


class MultiScaleEEGEncoder(nn.Module):
    def __init__(self, n_channels=14, hidden_dim=48, kernels=(3, 7, 15)):
        super().__init__()
        self.scales = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(n_channels, hidden_dim, k, padding=k // 2),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
            ) for k in kernels
        ])
        self.fusion = nn.Linear(hidden_dim * len(kernels), hidden_dim)

    def forward(self, x):
        outs = [s(x) for s in self.scales]
        multi = torch.cat(outs, dim=1).permute(0, 2, 1)
        return self.fusion(multi)


class GazeStateEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=48, n_layers=2):
        super().__init__()
        self.pos_embed = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, num_layers=n_layers,
                              batch_first=True, bidirectional=False)

    def forward(self, gaze, mask=None):
        x = self.pos_embed(gaze)
        out, _ = self.encoder(x)
        if mask is not None:
            out = out * mask.unsqueeze(-1).float()
        return out


def _proportional_mask(q_len, kv_len, device):
    """Boolean (q_len, kv_len); True = blocked.

    EEG position i (normalized time i/q_len) may attend to gaze position j
    (normalized time j/kv_len) iff j/kv_len <= i/q_len, i.e. j <= i*kv_len/q_len.
    """
    i = torch.arange(q_len, device=device, dtype=torch.float32)
    j = torch.arange(kv_len, device=device, dtype=torch.float32)
    boundary = (i * (kv_len / q_len)).floor().unsqueeze(1)
    return j.unsqueeze(0) > boundary


def _causal_self_mask(q_len, device):
    """Standard causal mask for self-attention over a single stream."""
    m = torch.triu(torch.ones(q_len, q_len, device=device), diagonal=1).bool()
    return m


class AttentionBlock(nn.Module):
    """Pre-norm attention block usable for self- or cross-attention.

    mode='cross': queries from stream1 (EEG), keys/values from stream2 (gaze),
                  masked by the proportional temporal mask.
    mode='self' : queries/keys/values from stream1; causal option for the
                  gaze stream.
    """

    def __init__(self, hidden_dim=48, n_heads=4, dropout=0.1,
                 mode='cross', causal=False, use_mask=True):
        super().__init__()
        assert mode in ('cross', 'self')
        self.mode = mode
        self.causal = causal
        self.use_mask = use_mask
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, feats, kv_feats=None, kv_mask=None):
        B, Q, _ = feats.shape
        kv = feats if self.mode == 'self' else kv_feats
        KV = kv.shape[1]

        q = self.q_proj(self.norm1(feats)).view(B, Q, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(B, KV, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, KV, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        if self.mode == 'cross' and self.use_mask:
            blocked = _proportional_mask(Q, KV, attn.device)
            attn = attn.masked_fill(blocked.unsqueeze(0).unsqueeze(0), MASK_NEG)
        elif self.causal:
            blocked = _causal_self_mask(Q, attn.device)
            attn = attn.masked_fill(blocked.unsqueeze(0).unsqueeze(0), MASK_NEG)

        if kv_mask is not None:
            pad = ~kv_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(pad, MASK_NEG)

        w = F.softmax(attn, dim=-1)
        w = self.dropout(w)
        out = torch.matmul(w, v).transpose(1, 2).contiguous().view(B, Q, self.hidden_dim)
        out = self.out_proj(out)
        out = feats + out
        out = out + self.ffn(self.norm2(out))
        return out, w


class TemporalAttentionHead(nn.Module):
    def __init__(self, hidden_dim=48, n_layers=2, dropout=0.1):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_dim, 2))
        self.mlp = nn.Sequential(*layers)
        self.temporal_attn = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        weights = F.softmax(self.temporal_attn(features).squeeze(-1), dim=-1)
        agg = torch.bmm(weights.unsqueeze(1), features).squeeze(1)
        return self.mlp(agg)


class TempAlignEEGNet(nn.Module):
    """EEG+gaze cross-attention with a fixed proportional temporal mask."""

    def __init__(self, n_channels=14, hidden_dim=48, n_attn_layers=2,
                 n_heads=4, dropout=0.1, use_mask=True):
        super().__init__()
        self.eeg_encoder = MultiScaleEEGEncoder(n_channels, hidden_dim)
        self.gaze_encoder = GazeStateEncoder(2, hidden_dim)
        self.blocks = nn.ModuleList([
            AttentionBlock(hidden_dim, n_heads, dropout, mode='cross',
                           use_mask=use_mask)
            for _ in range(n_attn_layers)
        ])
        self.pred_head = TemporalAttentionHead(hidden_dim, dropout=dropout)

    def forward(self, eeg, gaze_context, gaze_mask=None):
        eeg_f = self.eeg_encoder(eeg)
        gaze_f = self.gaze_encoder(gaze_context, gaze_mask)
        attn_list = []
        x = eeg_f
        for blk in self.blocks:
            x, w = blk(x, gaze_f, gaze_mask)
            attn_list.append(w)
        return {'pred': self.pred_head(x), 'attn_weights': attn_list}


class TempAlignGazeOnly(nn.Module):
    """Same architecture applied to the gaze stream alone (causal self-attention)."""

    def __init__(self, hidden_dim=48, n_attn_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.gaze_encoder = GazeStateEncoder(2, hidden_dim)
        self.blocks = nn.ModuleList([
            AttentionBlock(hidden_dim, n_heads, dropout, mode='self', causal=True)
            for _ in range(n_attn_layers)
        ])
        self.pred_head = TemporalAttentionHead(hidden_dim, dropout=dropout)

    def forward(self, eeg, gaze_context, gaze_mask=None):
        x = self.gaze_encoder(gaze_context, gaze_mask)
        for blk in self.blocks:
            x, _ = blk(x)
        return {'pred': self.pred_head(x)}


class TempAlignEEGOnly(nn.Module):
    """Same architecture applied to the EEG stream alone (self-attention)."""

    def __init__(self, n_channels=14, hidden_dim=48, n_attn_layers=2,
                 n_heads=4, dropout=0.1):
        super().__init__()
        self.eeg_encoder = MultiScaleEEGEncoder(n_channels, hidden_dim)
        self.blocks = nn.ModuleList([
            AttentionBlock(hidden_dim, n_heads, dropout, mode='self')
            for _ in range(n_attn_layers)
        ])
        self.pred_head = TemporalAttentionHead(hidden_dim, dropout=dropout)

    def forward(self, eeg, gaze_context, gaze_mask=None):
        x = self.eeg_encoder(eeg)
        for blk in self.blocks:
            x, _ = blk(x)
        return {'pred': self.pred_head(x)}


class TempStructNet(nn.Module):
    """Learned temporal mask (corrected math).

    The mask probability p_ij = sigmoid(((i/T - delta) - j/S) / beta) is fully
    determined by the two interpretable parameters delta (temporal delay,
    normalized to the window) and beta (boundary bandwidth). During training a
    hard straight-through Gumbel-Softmax sample G ~ Bernoulli(p_ij) is drawn;
    at eval time G = 1[p_ij > 0.5]. The additive penalty is
    M = (1 - G) * (-1e9), guaranteeing zero attention on blocked positions in
    exact arithmetic.
    """

    def __init__(self, n_channels=14, hidden_dim=48, n_layers=2, n_heads=4,
                 dropout=0.1, max_delay=0.3, delay_init=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.max_delay = max_delay
        self.temperature = 1.0

        self.eeg_encoder = MultiScaleEEGEncoder(n_channels, hidden_dim)
        self.gaze_encoder = GazeStateEncoder(2, hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim))
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.pred_head = TemporalAttentionHead(hidden_dim, dropout=dropout)

        self.log_delay = nn.Parameter(torch.tensor(delay_init, dtype=torch.float32))
        self.log_bandwidth = nn.Parameter(torch.tensor(0.0))

    @property
    def delay(self):
        return torch.sigmoid(self.log_delay) * self.max_delay

    @property
    def bandwidth(self):
        return F.softplus(self.log_bandwidth) + 0.1

    def set_temperature(self, t):
        self.temperature = float(t)

    def soft_mask(self, q_len, kv_len, device):
        eeg_pos = torch.arange(q_len, device=device, dtype=torch.float32) / q_len
        gaze_pos = torch.arange(kv_len, device=device, dtype=torch.float32) / kv_len
        boundary = (eeg_pos - self.delay).unsqueeze(1)
        return torch.sigmoid((boundary - gaze_pos.unsqueeze(0)) / self.bandwidth)

    def binary_mask(self, q_len, kv_len, device):
        eps = 1e-6
        p = self.soft_mask(q_len, kv_len, device).clamp(eps, 1 - eps)
        if self.training:
            logits = torch.stack([torch.log(1 - p), torch.log(p)], dim=-1)
            y = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)
            g = y[..., 1]
        else:
            g = (p > 0.5).float()
        # Keep the earliest gaze sample attendable in every row; otherwise a
        # fully blocked row degenerates to a uniform softmax over prohibited
        # positions.
        force_first = torch.zeros_like(g)
        force_first[:, 0] = 1.0
        return torch.maximum(g, force_first)

    def forward(self, eeg, gaze_context, gaze_mask=None, return_mask=False):
        B, _, _ = eeg.shape
        eeg_f = self.eeg_encoder(eeg)
        gaze_f = self.gaze_encoder(gaze_context, gaze_mask)
        Q, KV = eeg_f.shape[1], gaze_f.shape[1]

        g = self.binary_mask(Q, KV, eeg.device)
        penalty = (1.0 - g) * MASK_NEG

        q = self.q_proj(self.norm1(eeg_f)).view(B, Q, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(gaze_f).view(B, KV, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(gaze_f).view(B, KV, self.n_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn + penalty.unsqueeze(0).unsqueeze(0)
        if gaze_mask is not None:
            pad = ~gaze_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(pad, MASK_NEG)
        w = F.softmax(attn, dim=-1)

        out = torch.matmul(w, v).transpose(1, 2).contiguous().view(B, Q, self.hidden_dim)
        out = self.out_proj(out)
        out = eeg_f + out
        out = out + self.ffn(self.norm2(out))
        pred = self.pred_head(out)
        result = {'pred': pred, 'attn_weights': [w],
                  'delay': float(self.delay.detach().cpu()),
                  'bandwidth': float(self.bandwidth.detach().cpu())}
        if return_mask:
            result['mask'] = g.detach()
        return result


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
