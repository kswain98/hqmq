"""Hadamard / random-orthogonal pre-rotation wrapper for any KVQuantizer.

Modern open LLMs (Qwen2.5, Llama-3) have heavy-tailed K-channel distributions
post-RoPE. Per-token quantization can't handle the outliers. VecInfer (2025),
QuaRot (2024), and SpinQuant (2024) all show that applying a *fixed orthogonal
rotation* to K (and V) before quantization spreads the outlier mass across
channels, making quantization friendly. The rotation preserves attention scores
exactly: $\\text{softmax}(Q (R K)^\\top) = \\text{softmax}((Q R) K^\\top)$, so as
long as we undo the rotation on read, the attention math is unchanged modulo
the quantization noise on the rotated K.

This wrapper applies a single fixed orthogonal matrix per (layer, head) before
calling the inner quantizer's `quantize_K` / `quantize_V`. The rotation is
the Hadamard matrix (Sylvester recursion) when `d_head` is a power of 2;
otherwise a random orthonormal matrix.
"""

import math

import torch

from .base import KVQuantizer


def hadamard_matrix(n: int, dtype=torch.float32) -> torch.Tensor:
    """Sylvester Hadamard H_n / sqrt(n). n must be a power of 2."""
    assert n > 0 and (n & (n - 1) == 0), f"n={n} must be a power of 2"
    H = torch.tensor([[1.0]], dtype=dtype)
    while H.size(0) < n:
        H = torch.cat([torch.cat([H, H], dim=1),
                       torch.cat([H, -H], dim=1)], dim=0)
    H = H / math.sqrt(n)
    return H


def random_orthogonal(n: int, seed: int = 0, dtype=torch.float32) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(n, n, generator=g, dtype=dtype)
    Q, _ = torch.linalg.qr(M)
    return Q


class HadamardKVQuantizer(KVQuantizer):
    """Wraps an inner KVQuantizer with a fixed orthogonal pre-rotation on K and V."""

    def __init__(self, inner: KVQuantizer, d_head: int, kind: str = "auto",
                 device: str = "cuda"):
        self.inner = inner
        self.d_head = d_head
        if kind == "auto":
            kind = "hadamard" if (d_head > 0 and d_head & (d_head - 1) == 0) else "random"
        if kind == "hadamard":
            self._R = hadamard_matrix(d_head)
        elif kind == "random":
            self._R = random_orthogonal(d_head, seed=42)
        else:
            raise ValueError(kind)
        self._R = self._R.to(device=device)
        self._R_t = self._R.t().contiguous()
        self._cached = {}

    def _get_R(self, ref):
        key = (ref.device, ref.dtype)
        if key not in self._cached:
            self._cached[key] = (self._R.to(device=ref.device, dtype=ref.dtype),
                                  self._R_t.to(device=ref.device, dtype=ref.dtype))
        return self._cached[key]

    def _rotate(self, x):
        # x: (B, H, T, d_head). R: (d_head, d_head). Rotate last dim.
        R, _ = self._get_R(x)
        return torch.einsum("bhtc,cd->bhtd", x, R)

    def _unrotate(self, x):
        _, Rt = self._get_R(x)
        return torch.einsum("bhtc,cd->bhtd", x, Rt)

    def quantize_K(self, k, layer_idx):
        k_rot = self._rotate(k)
        k_rot_q = self.inner.quantize_K(k_rot, layer_idx)
        return self._unrotate(k_rot_q)

    def quantize_V(self, v, layer_idx):
        v_rot = self._rotate(v)
        v_rot_q = self.inner.quantize_V(v_rot, layer_idx)
        return self._unrotate(v_rot_q)

    def bits_per_value(self):
        return self.inner.bits_per_value()

    @property
    def name(self):
        return f"Hadamard_{self.inner.name}"

    # Compat with calibration plumbing
    def trainable_parameters(self):
        if hasattr(self.inner, "trainable_parameters"):
            return self.inner.trainable_parameters()
        return []

    def set_training(self, training, tau=None):
        if hasattr(self.inner, "set_training"):
            self.inner.set_training(training, tau=tau)

    def _refresh_rotations(self):
        if hasattr(self.inner, "_refresh_rotations"):
            self.inner._refresh_rotations()
