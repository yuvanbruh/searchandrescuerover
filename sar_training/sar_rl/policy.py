"""
Candidate-scoring policy: a shared encoder scores every candidate goal, a small
self-attention stack lets candidates see each other, and a masked softmax picks
one. Handles a variable number of candidates (padded + masked).
Features are already normalised by the simulator, so deployment needs no
observation scaling.
"""
import torch
import torch.nn as nn


class CandidatePolicy(nn.Module):
    def __init__(self, n_cand_feat=13, n_glob_feat=10, d=64, heads=4, layers=2):
        super().__init__()
        self.cand_enc = nn.Sequential(nn.Linear(n_cand_feat, d), nn.ReLU(), nn.Linear(d, d))
        self.glob_enc = nn.Sequential(nn.Linear(n_glob_feat, d), nn.ReLU(), nn.Linear(d, d))
        self.fuse = nn.Linear(2 * d, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=2 * d,
                                           dropout=0.0, batch_first=True)
        self.attn = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.logit_head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        self.value_head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, cand, mask, glob):
        """cand (B,K,F) float, mask (B,K) bool (True = valid), glob (B,G) float."""
        mask = mask.clone()
        empty = ~mask.any(dim=1)
        if empty.any():                      # terminal obs with no candidates: avoid NaNs
            mask[empty, 0] = True
        g = self.glob_enc(glob)
        h = self.cand_enc(cand)
        h = self.fuse(torch.cat([h, g[:, None, :].expand(-1, h.size(1), -1)], dim=-1))
        h = self.attn(h, src_key_padding_mask=~mask)
        logits = self.logit_head(h).squeeze(-1).masked_fill(~mask, -1e9)
        m = mask.float().unsqueeze(-1)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        value = self.value_head(torch.cat([pooled, g], dim=-1)).squeeze(-1)
        return logits, value
