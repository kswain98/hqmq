"""Hurwitz Quaternion Multiplicative Quantization (HQMQ).

Each 4-element chunk of K or V is treated as a quaternion. The unit direction
is quantized to a product of two quaternions:

    direction_q = q_p · q_s

where:
  - q_p ∈ 2T = the 24-element Hurwitz quaternion group (24-cell vertices). Fixed.
  - q_s ∈ a learned secondary codebook of `S` unit quaternions per (layer, head, K-or-V).

Storage per chunk: log2(24 · S) bits for the (primary, secondary) index pair,
plus radius bits.

The contribution vs existing KV-quant work: the codebook composition rule is
*quaternion multiplication* (group algebra), not vector addition (CommVQ) or
free VQ over a flat learned codebook (VPTQ). The Hurwitz primary set gives the
structure; the learned secondary set gives expressivity.

Training: soft codebook over 24*S joint products + STE-to-hard forward.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from .base import KVQuantizer
from .spherical import _make_24cell_codebook


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamiltonian product. q1, q2: (..., 4) with (w, x, y, z). Returns (..., 4)."""
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


class HQMQQuantizer(KVQuantizer):
    """Hurwitz Quaternion Multiplicative Quantizer for KV cache.

    Direction of each 4-chunk is quantized as q_p · q_s. q_p ∈ 24-cell (fixed),
    q_s ∈ learned per-(layer, head, K|V) codebook of size `secondary_size`.
    """

    def __init__(self, n_layers: int, n_heads: int,
                 secondary_size: int = 24,
                 radius_bits: int = 4,
                 jl_dim: int = 0,
                 magnitude_mode: str = "unit",   # "unit" | "free"
                 robust_radius_scale: bool = False,
                 log_radius: bool = False,       # geometric (log-spaced) radius levels
                 device: str = "cuda",
                 init: str = "random",
                 seed: int = 42):
        self.chunk_dim = 4
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.secondary_size = secondary_size
        self.radius_bits = radius_bits
        self.jl_dim = jl_dim
        self.magnitude_mode = magnitude_mode
        self.robust_radius_scale = robust_radius_scale
        self.log_radius = log_radius
        if magnitude_mode not in ("unit", "free"):
            raise ValueError(f"magnitude_mode={magnitude_mode}")
        self.training = False
        self.tau = 1.0
        self._buf_cache: dict = {}

        # Primary codebook: 24 Hurwitz unit quaternions (24-cell).
        self._primary = _make_24cell_codebook()  # (24, 4) fp32

        # Optional JL residual projection (orthonormal rows)
        if jl_dim > 0:
            g_jl = torch.Generator().manual_seed(seed + 1)
            M = torch.randn(max(jl_dim, 4), 4, generator=g_jl)
            Q, _ = torch.linalg.qr(M)
            self._R = Q[:jl_dim]  # (jl_dim, 4)
        else:
            self._R = None

        # Secondary codebook: learnable raw params (will be normalized on use).
        # Shape: (n_layers, n_heads, 2, secondary_size, 4) — 2 for K, V
        g = torch.Generator().manual_seed(seed)
        if init == "random":
            raw = torch.randn(n_layers, n_heads, 2, secondary_size, 4, generator=g)
        elif init == "hurwitz_perturbed":
            # Init close to 24-cell + small noise. If secondary_size != 24, repeat/trim.
            base = self._primary  # (24, 4)
            if secondary_size <= 24:
                base = base[:secondary_size]
            else:
                base = base.repeat((secondary_size + 23) // 24, 1)[:secondary_size]
            base = base.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, S, 4)
            raw = base.expand(n_layers, n_heads, 2, secondary_size, 4).clone()
            raw = raw + 0.1 * torch.randn_like(raw)
        else:
            raise ValueError(init)
        self.secondary_raw = nn.Parameter(raw.to(device=device, dtype=torch.float32))

    def trainable_parameters(self):
        return [self.secondary_raw]

    def set_training(self, training: bool, tau: Optional[float] = None):
        self.training = training
        if tau is not None:
            self.tau = tau

    def _refresh_rotations(self):
        # Compatible with calibration.py's plumbing.
        pass

    def _get_codebook(self, layer_idx: int, kv: int, ref: torch.Tensor) -> torch.Tensor:
        """Compute the 24*S joint quaternions q_p·q_s for one (layer, head, kv).

        Returns shape (n_heads, 24*secondary_size, 4) — flattened joint codebook per head.
        Cached per (device, dtype) for the primary; per call for the secondary.
        """
        key = ("primary", ref.device, ref.dtype)
        if key not in self._buf_cache:
            self._buf_cache[key] = self._primary.to(device=ref.device, dtype=ref.dtype)
        primary = self._buf_cache[key]                                  # (24, 4)
        secondary_raw = self.secondary_raw[layer_idx, :, kv]            # (H, S, 4) fp32
        if self.magnitude_mode == "unit":
            # Normalize to unit quaternions — codebook lies on S^3
            secondary = secondary_raw / secondary_raw.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        else:  # "free"
            # Let secondary have free magnitudes — codeword magnitude embedded.
            secondary = secondary_raw
        secondary = secondary.to(ref.dtype)
        # Products: p has shape (24, 4), s has shape (H, S, 4).
        # Broadcast to (H, 24, S, 4) and multiply.
        p_exp = primary.unsqueeze(0).unsqueeze(2).expand(secondary.shape[0], 24, secondary.shape[1], 4)  # (H, 24, S, 4)
        s_exp = secondary.unsqueeze(1).expand(secondary.shape[0], 24, secondary.shape[1], 4)             # (H, 24, S, 4)
        products = quat_mul(p_exp, s_exp)                                                                # (H, 24, S, 4)
        joint = products.reshape(secondary.shape[0], 24 * secondary.shape[1], 4)                         # (H, 24*S, 4)
        return joint

    def _qdq(self, x: torch.Tensor, layer_idx: int, kv: int) -> torch.Tensor:
        B, H, T, d_head = x.shape
        if d_head % 4 != 0:
            raise ValueError(f"d_head={d_head} not divisible by 4")
        n_chunks = d_head // 4
        x_chunked = x.view(B, H, T, n_chunks, 4)

        radius = x_chunked.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (B, H, T, n_chunks, 1)

        joint = self._get_codebook(layer_idx, kv, x)                    # (H, K, 4), K=24*S
        if self.magnitude_mode == "unit":
            # Match unit direction against unit codewords — inner-product equivalent to L2.
            direction = x_chunked / radius                              # (B, H, T, n_chunks, 4)
            sims = torch.einsum("bhtnc,hkc->bhtnk", direction, joint)
        else:  # "free"
            # L2-optimal nearest neighbor: sims = <x, c> - 0.5 * ||c||^2
            inner = torch.einsum("bhtnc,hkc->bhtnk", x_chunked, joint)
            cb_norm2 = (joint * joint).sum(dim=-1)                       # (H, K)
            sims = inner - 0.5 * cb_norm2.unsqueeze(0).unsqueeze(2).unsqueeze(2)
        idx = sims.argmax(dim=-1)                                       # (B, H, T, n_chunks)
        # Gather the hard codeword from `joint` per (head, idx).
        joint_b = joint.unsqueeze(0).unsqueeze(2).unsqueeze(3)           # (1, H, 1, 1, K, 4)
        idx_gather = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 1, 4)   # (B, H, T, n_chunks, 1, 4)
        joint_b_exp = joint_b.expand(B, H, T, n_chunks, joint.shape[1], 4)
        direction_q_hard = joint_b_exp.gather(dim=-2, index=idx_gather).squeeze(-2)  # (B, H, T, n_chunks, 4)

        if self.training:
            weights = torch.softmax(sims / self.tau, dim=-1)             # (B, H, T, n_chunks, K)
            codeword_soft = torch.einsum("bhtnk,hkc->bhtnc", weights, joint)
            codeword_q = codeword_soft + (direction_q_hard - codeword_soft).detach()
        else:
            codeword_q = direction_q_hard

        if self.magnitude_mode == "unit":
            # Apply quantized radius to the unit codeword.
            if self.radius_bits >= 16:
                radius_q = radius
            elif self.radius_bits == 0:
                radius_q = radius.mean(dim=-2, keepdim=True).expand_as(radius)
            elif self.log_radius:
                # Log-spaced (geometric) radius quantization. Handles 100s× dynamic
                # range gracefully — both bulk and outliers get distinct codes.
                # log(r) is quantized linearly over [log(r_min), log(r_max)].
                qmax = (1 << self.radius_bits) - 1
                log_r = torch.log(radius.clamp(min=1e-8))
                log_r_max = log_r.amax(dim=-2, keepdim=True)
                log_r_min = log_r.amin(dim=-2, keepdim=True)
                log_range = (log_r_max - log_r_min).clamp(min=1e-8)
                # Quantize log_r to [0, qmax]
                log_q = ((log_r - log_r_min) / log_range * qmax).round().clamp(0, qmax)
                # Dequantize
                log_dq = log_r_min + (log_q / qmax) * log_range
                radius_q_hard = torch.exp(log_dq)
                if self.training:
                    radius_q = radius + (radius_q_hard - radius).detach()
                else:
                    radius_q = radius_q_hard
            else:
                if self.robust_radius_scale:
                    r_med = radius.median(dim=-2, keepdim=True).values
                    r_mad = (radius - r_med).abs().median(dim=-2, keepdim=True).values
                    r_max = (r_med + 3.0 * r_mad).clamp(min=1e-8)
                else:
                    r_max = radius.amax(dim=-2, keepdim=True).clamp(min=1e-8)
                qmax = (1 << self.radius_bits) - 1
                scale = r_max / qmax
                radius_q_hard = (radius / scale).round().clamp(0, qmax) * scale
                if self.training:
                    radius_q = radius + (radius_q_hard - radius).detach()
                else:
                    radius_q = radius_q_hard
            primary_out = radius_q * codeword_q
        else:
            # Free-magnitude codewords already include magnitude info.
            # Apply a single per-token global scale (free; 1 fp16 per token).
            # This rescales the codebook to fit the token's overall magnitude.
            x_norm = radius.mean(dim=-2, keepdim=True)                   # (B, H, T, 1, 1)
            cb_norm_mean = (joint * joint).sum(dim=-1).sqrt().mean()     # scalar
            per_token_scale = x_norm / cb_norm_mean.clamp(min=1e-8)
            primary_out = codeword_q * per_token_scale

        # Optional JL residual refinement
        if self.jl_dim > 0:
            # Cache the JL matrix on this device/dtype
            key_R = ("R", x.device, x.dtype)
            if key_R not in self._buf_cache:
                self._buf_cache[key_R] = self._R.to(device=x.device, dtype=x.dtype)
            R = self._buf_cache[key_R]                                   # (jl_dim, 4)
            residual = x_chunked - primary_out                           # (B, H, T, n_chunks, 4)
            y = torch.einsum("bhtnc,jc->bhtnj", residual, R)             # (B, H, T, n_chunks, jl_dim)
            signs_hard = torch.sign(y)
            if self.training:
                signs_soft = torch.tanh(y / self.tau)
                signs = signs_soft + (signs_hard - signs_soft).detach()
            else:
                signs = signs_hard
            res_norm = residual.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            alpha = res_norm / (self.jl_dim ** 0.5)
            y_q = signs * alpha
            residual_q = torch.einsum("bhtnj,jc->bhtnc", y_q, R)
            out = primary_out + residual_q
        else:
            out = primary_out

        return out.view(B, H, T, d_head)

    def quantize_K(self, k, layer_idx):
        return self._qdq(k, layer_idx, kv=0)

    def quantize_V(self, v, layer_idx):
        return self._qdq(v, layer_idx, kv=1)

    def bits_per_value(self):
        # log2(24 * S) for direction + radius_bits (or 0 for free magnitude) + jl_dim
        # per-token scale overhead (16/d_head ≈ 0.06 for d_head=256) is ignored.
        bits_dir = math.log2(24 * self.secondary_size)
        if self.magnitude_mode == "free":
            radius_contribution = 0
        else:
            radius_contribution = self.radius_bits
        return (bits_dir + radius_contribution + self.jl_dim) / 4.0

    @property
    def name(self) -> str:
        parts = [f"HQMQ_s{self.secondary_size}_r{self.radius_bits}"]
        if self.jl_dim > 0:
            parts.append(f"jl{self.jl_dim}")
        if self.magnitude_mode != "unit":
            parts.append(self.magnitude_mode)
        if self.log_radius:
            parts.append("logr")
        if self.robust_radius_scale:
            parts.append("robust")
        return "_".join(parts)
