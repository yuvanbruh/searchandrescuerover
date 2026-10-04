"""
Pure-NumPy forward pass of the trained CandidatePolicy (no torch needed inside
ROS). Weights come from policy.npz made by export_policy.py.
"""
import numpy as np


def _relu(x):
    return np.maximum(x, 0.0)


def _lin(x, w, b):
    return x @ w.T + b


def _ln(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def _softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


class NumpyCandidatePolicy:
    def __init__(self, path):
        z = np.load(path)
        self.p = {k: z[k] for k in z.files}
        self.heads = int(self.p.get("heads", 4))
        n = 0
        while f"attn.layers.{n}.linear1.weight" in self.p:
            n += 1
        self.n_layers = n

    def _layer(self, x, mask, i):
        p, pre = self.p, f"attn.layers.{i}."
        K, d = x.shape
        H = self.heads
        hd = d // H
        qkv = _lin(x, p[pre + "self_attn.in_proj_weight"], p[pre + "self_attn.in_proj_bias"])
        q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]
        q = q.reshape(K, H, hd).transpose(1, 0, 2)
        k = k.reshape(K, H, hd).transpose(1, 0, 2)
        v = v.reshape(K, H, hd).transpose(1, 0, 2)
        scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(hd)
        scores[:, :, ~mask] = -1e9
        att = _softmax(scores) @ v
        att = att.transpose(1, 0, 2).reshape(K, d)
        att = _lin(att, p[pre + "self_attn.out_proj.weight"], p[pre + "self_attn.out_proj.bias"])
        x = _ln(x + att, p[pre + "norm1.weight"], p[pre + "norm1.bias"])
        ff = _lin(_relu(_lin(x, p[pre + "linear1.weight"], p[pre + "linear1.bias"])),
                  p[pre + "linear2.weight"], p[pre + "linear2.bias"])
        return _ln(x + ff, p[pre + "norm2.weight"], p[pre + "norm2.bias"])

    def logits(self, cand, mask, glob):
        p = self.p
        cand = np.asarray(cand, dtype=np.float64)
        glob = np.asarray(glob, dtype=np.float64)
        mask = np.asarray(mask, dtype=bool).copy()
        if not mask.any():
            mask[0] = True
        g = _lin(_relu(_lin(glob, p["glob_enc.0.weight"], p["glob_enc.0.bias"])),
                 p["glob_enc.2.weight"], p["glob_enc.2.bias"])
        h = _lin(_relu(_lin(cand, p["cand_enc.0.weight"], p["cand_enc.0.bias"])),
                 p["cand_enc.2.weight"], p["cand_enc.2.bias"])
        h = _lin(np.concatenate([h, np.broadcast_to(g, (h.shape[0], g.shape[0]))], -1),
                 p["fuse.weight"], p["fuse.bias"])
        for i in range(self.n_layers):
            h = self._layer(h, mask, i)
        out = _lin(_relu(_lin(h, p["logit_head.0.weight"], p["logit_head.0.bias"])),
                   p["logit_head.2.weight"], p["logit_head.2.bias"])[:, 0]
        return np.where(mask, out, -1e9)

    def act(self, cand, mask, glob):
        lg = self.logits(cand, mask, glob)
        pr = _softmax(lg)
        return int(np.argmax(lg)), pr
