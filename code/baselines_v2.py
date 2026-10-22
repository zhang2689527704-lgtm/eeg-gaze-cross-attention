"""Neural baselines with identical inputs (EEG + gaze), fusion module
(unmasked cross-attention), and prediction head. Differences therefore reflect
the EEG encoder and the temporal mask rather than input availability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models_v2 import GazeStateEncoder, TemporalAttentionHead


class FusionBaseline(nn.Module):
    """EEG tower (architecture-specific) + gaze GRU + unmasked cross-attention
    fusion + shared prediction head."""

    def __init__(self, eeg_tower, hidden_dim=48, n_heads=4, dropout=0.2):
        super().__init__()
        self.eeg_tower = eeg_tower
        self.gaze_encoder = GazeStateEncoder(2, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, dropout=dropout,
            batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim))
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.pred_head = TemporalAttentionHead(hidden_dim, dropout=dropout)

    def forward(self, eeg, gaze_context, gaze_mask=None):
        eeg_feat = self.eeg_tower(eeg)  # (B, T', hidden)
        gaze_feat = self.gaze_encoder(gaze_context, gaze_mask)

        kpm = None
        if gaze_mask is not None:
            kpm = ~gaze_mask
        attn_out, _ = self.cross_attn(eeg_feat, gaze_feat, gaze_feat,
                                      key_padding_mask=kpm)
        x = self.norm1(eeg_feat + attn_out)
        x = self.norm2(x + self.ffn(x))
        return {'pred': self.pred_head(x)}


class LSTMTower(nn.Module):
    def __init__(self, n_channels=14, hidden_dim=48, n_layers=2, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(n_channels, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, n_layers,
                            batch_first=True, dropout=dropout,
                            bidirectional=False)

    def forward(self, eeg):
        x = self.proj(eeg.transpose(1, 2))  # (B, T, hidden)
        out, _ = self.lstm(x)
        return out


class TransformerTower(nn.Module):
    def __init__(self, n_channels=14, hidden_dim=48, n_heads=4, n_layers=2,
                 dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(n_channels, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, eeg):
        x = self.proj(eeg.transpose(1, 2))
        return self.encoder(x)


class LTCTower(nn.Module):
    """Liquid time-constant inspired tower (conv + GRU with learnable tau)."""

    def __init__(self, n_channels=14, hidden_dim=48, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, hidden_dim, 7, stride=2, padding=3),
            nn.BatchNorm1d(hidden_dim), nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 7, stride=2, padding=3),
            nn.BatchNorm1d(hidden_dim), nn.GELU())
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.log_tau = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, eeg):
        x = self.conv(eeg).permute(0, 2, 1)
        out, _ = self.gru(x)
        return out * torch.sigmoid(self.log_tau)


class WaveletTower(nn.Module):
    """Multi-resolution strided-conv decomposition + Transformer encoder."""

    def __init__(self, n_channels=14, hidden_dim=48, n_heads=4, n_layers=2,
                 dropout=0.2):
        super().__init__()
        self.c1 = nn.Conv1d(n_channels, hidden_dim, 8, stride=2, padding=3)
        self.c2 = nn.Conv1d(hidden_dim, hidden_dim, 8, stride=2, padding=3)
        self.c3 = nn.Conv1d(hidden_dim, hidden_dim, 8, stride=2, padding=3)
        self.b1 = nn.BatchNorm1d(hidden_dim)
        self.b2 = nn.BatchNorm1d(hidden_dim)
        self.b3 = nn.BatchNorm1d(hidden_dim)
        self.fusion = nn.Linear(hidden_dim * 3, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, eeg):
        x1 = F.gelu(self.b1(self.c1(eeg)))
        x2 = F.gelu(self.b2(self.c2(x1)))
        x3 = F.gelu(self.b3(self.c3(x2)))
        x1d = F.adaptive_avg_pool1d(x1, x3.shape[2])
        x2d = F.adaptive_avg_pool1d(x2, x3.shape[2])
        multi = torch.cat([x1d, x2d, x3], dim=1).permute(0, 2, 1)
        x = self.fusion(multi)
        return self.transformer(x)


def build_lstm_dual(n_channels=14, hidden_dim=48, **kw):
    return FusionBaseline(LSTMTower(n_channels, hidden_dim), hidden_dim)


def build_transformer_dual(n_channels=14, hidden_dim=48, **kw):
    return FusionBaseline(TransformerTower(n_channels, hidden_dim), hidden_dim)


def build_ltc_dual(n_channels=14, hidden_dim=48, **kw):
    return FusionBaseline(LTCTower(n_channels, hidden_dim), hidden_dim)


def build_wavelet_dual(n_channels=14, hidden_dim=48, **kw):
    return FusionBaseline(WaveletTower(n_channels, hidden_dim), hidden_dim)
