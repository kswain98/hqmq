"""Output-aware learned-rotation wrapper around any KVQuantizer.

The wrapper applies per-(layer, head) chunk-dim × chunk-dim rotations R_lh
to K and V before quantization and the inverse R_lh^T after dequantization.
R_lh is initialized to identity and trained on a calibration set with the
objective of minimizing attention-output MSE (Q @ K^T-style outputs), not
K/V reconstruction.

The key conceptual move: we let the model "reshape" the K,V distribution
so that whatever fixed codebook the underlying quantizer uses is a better
fit. Trained jointly with the output-aware loss, this should let a fixed
spherical codebook (e.g., 24-cell) achieve quality comparable to a much
larger/learned codebook.

Calibration training happens once before evaluation; the rotations are
then frozen and applied at inference time.
"""

from typing import Optional

import torch
import torch.nn as nn

from .base import KVQuantizer


class QuaternionRotationKVQuantizer(KVQuantizer):
    """Per-(layer, head) chunk-dim-4 rotation parameterized as a unit quaternion.

    For chunk_dim=4 (where K and V are chunked into 4-tuples), each chunk's
    rotation can be parameterized as left-multiplication by a unit quaternion
    q = (w, x, y, z). This gives a 4-parameter (per (layer, head)) rotation
    matrix L_q ∈ SO(4) of the left-isoclinic family, rather than the 6-parameter
    full-SO(4) rotation. The space is smaller and the parameterization is
    natively unit-norm-constrained, so calibration optimization is more stable.

    Connection to the existing 24-cell codebook: those 24 codewords are
    themselves unit Hurwitz quaternions; quaternion-parameterized rotation
    keeps the entire pipeline in the quaternion algebra.

    For chunk_dim != 4, falls back to error (use LearnedRotationKVQuantizer).
    """

    def __init__(self, inner_quantizer: KVQuantizer, n_layers: int, n_heads: int,
                 chunk_dim: int = 4, device: str = "cuda"):
        assert chunk_dim == 4, "QuaternionRotationKVQuantizer only supports chunk_dim=4"
        self.inner = inner_quantizer
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.chunk_dim = 4
        # Parameterize each quaternion as 4 raw params, normalized on use.
        # Init at (1, 0, 0, 0) so the rotation matrix is identity.
        init = torch.zeros(n_layers, n_heads, 4, dtype=torch.float32, device=device)
        init[..., 0] = 1.0
        self.q_K_raw = nn.Parameter(init.clone())
        self.q_V_raw = nn.Parameter(init.clone())
        self._cached_R_K = None
        self._cached_R_V = None

    def trainable_parameters(self):
        return [self.q_K_raw, self.q_V_raw]

    @staticmethod
    def _left_mult_matrix(q: torch.Tensor) -> torch.Tensor:
        """q: (..., 4) unit quaternion (w, x, y, z). Returns rotation (..., 4, 4)."""
        w, x, y, z = q.unbind(dim=-1)
        zero = torch.zeros_like(w)
        # Left-multiplication-by-quaternion matrix (acts on a 4-vector treated as quaternion).
        # row 0: [w, -x, -y, -z]
        # row 1: [x,  w, -z,  y]
        # row 2: [y,  z,  w, -x]
        # row 3: [z, -y,  x,  w]
        r0 = torch.stack([w, -x, -y, -z], dim=-1)
        r1 = torch.stack([x,  w, -z,  y], dim=-1)
        r2 = torch.stack([y,  z,  w, -x], dim=-1)
        r3 = torch.stack([z, -y,  x,  w], dim=-1)
        R = torch.stack([r0, r1, r2, r3], dim=-2)
        return R

    def _refresh_rotations(self):
        # Normalize raw params to unit quaternion, then build rotation matrix.
        q_K = self.q_K_raw / self.q_K_raw.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        q_V = self.q_V_raw / self.q_V_raw.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self._cached_R_K = self._left_mult_matrix(q_K)  # (L, H, 4, 4)
        self._cached_R_V = self._left_mult_matrix(q_V)

    def _apply_rotation(self, x, R_all, layer_idx):
        B, H, T, d_head = x.shape
        n_chunks = d_head // self.chunk_dim
        x_chunked = x.view(B, H, T, n_chunks, self.chunk_dim)
        R = R_all[layer_idx].to(x.dtype)  # (H, 4, 4)
        x_rot = torch.einsum("bhtnc,hcd->bhtnd", x_chunked, R)
        return x_rot.view(B, H, T, d_head)

    def _apply_inverse_rotation(self, x, R_all, layer_idx):
        B, H, T, d_head = x.shape
        n_chunks = d_head // self.chunk_dim
        x_chunked = x.view(B, H, T, n_chunks, self.chunk_dim)
        R = R_all[layer_idx].to(x.dtype)
        # Inverse of a quaternion rotation is the conjugate quaternion's matrix,
        # which equals the transpose since rotations are orthogonal.
        x_rot = torch.einsum("bhtnc,hdc->bhtnd", x_chunked, R)
        return x_rot.view(B, H, T, d_head)

    def quantize_K(self, k, layer_idx):
        if self._cached_R_K is None:
            self._refresh_rotations()
        k_rot = self._apply_rotation(k, self._cached_R_K, layer_idx)
        k_rot_q = self.inner.quantize_K(k_rot, layer_idx)
        k_rot_ste = k_rot + (k_rot_q - k_rot).detach()
        return self._apply_inverse_rotation(k_rot_ste, self._cached_R_K, layer_idx)

    def quantize_V(self, v, layer_idx):
        if self._cached_R_V is None:
            self._refresh_rotations()
        v_rot = self._apply_rotation(v, self._cached_R_V, layer_idx)
        v_rot_q = self.inner.quantize_V(v_rot, layer_idx)
        v_rot_ste = v_rot + (v_rot_q - v_rot).detach()
        return self._apply_inverse_rotation(v_rot_ste, self._cached_R_V, layer_idx)

    def bits_per_value(self):
        return self.inner.bits_per_value()

    @property
    def name(self) -> str:
        return f"QR_{self.inner.name}"


class LearnedScaleKVQuantizer(KVQuantizer):
    """AWQ-style per-(layer, head, channel) learnable scale around an inner quantizer.

    Applies element-wise multiplicative scaling to K, V (per d_head channel,
    per layer, per head) before quantization, and inverse-scales after.
    Calibrated on a small calibration set with output-MSE objective. The
    scales push outlier channels into a quantizable range — same principle
    that makes AWQ effective for weight quantization.

    Initialization: scale = 1.0 everywhere (so untrained behaviour matches inner).
    """

    def __init__(self, inner_quantizer: KVQuantizer, n_layers: int, n_heads: int,
                 d_head: int, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.inner = inner_quantizer
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_head
        # Log-scale parameterization: scale = exp(log_scale), init log_scale=0 → scale=1.
        # Log-space keeps scales positive and stabilizes gradient.
        self.log_scale_K = nn.Parameter(torch.zeros(n_layers, n_heads, d_head, dtype=torch.float32, device=device))
        self.log_scale_V = nn.Parameter(torch.zeros(n_layers, n_heads, d_head, dtype=torch.float32, device=device))

    def trainable_parameters(self):
        return [self.log_scale_K, self.log_scale_V]

    # No-op set_training (compatible with the soft-codebook plumbing in calibration.py).
    def _refresh_rotations(self):
        pass

    def quantize_K(self, k, layer_idx):
        s = self.log_scale_K[layer_idx].exp().to(k.dtype)  # (H, d_head)
        s_b = s.unsqueeze(0).unsqueeze(2)                   # broadcast: (1, H, 1, d_head)
        k_scaled = k * s_b
        k_scaled_q = self.inner.quantize_K(k_scaled, layer_idx)
        k_scaled_ste = k_scaled + (k_scaled_q - k_scaled).detach()
        return k_scaled_ste / s_b

    def quantize_V(self, v, layer_idx):
        s = self.log_scale_V[layer_idx].exp().to(v.dtype)
        s_b = s.unsqueeze(0).unsqueeze(2)
        v_scaled = v * s_b
        v_scaled_q = self.inner.quantize_V(v_scaled, layer_idx)
        v_scaled_ste = v_scaled + (v_scaled_q - v_scaled).detach()
        return v_scaled_ste / s_b

    def bits_per_value(self):
        return self.inner.bits_per_value()

    @property
    def name(self) -> str:
        return f"LS_{self.inner.name}"


class LearnedRotationKVQuantizer(KVQuantizer):
    """Wraps an inner KVQuantizer with per-(layer, head) chunk_dim×chunk_dim rotations.

    `inner_quantizer` is the spherical (or any) quantizer applied between
    forward and inverse rotation.

    Forward:
        x' = R @ x   (chunked, per-(layer, head))
        x'_q = inner_quantizer(x')
        x_q = R^T @ x'_q

    R is parameterized as `expm(skew)` where skew is a learnable skew-symmetric
    matrix per (layer, head). This keeps R orthogonal during training.
    """

    def __init__(self, inner_quantizer: KVQuantizer, n_layers: int, n_heads: int,
                 chunk_dim: int, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.inner = inner_quantizer
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.chunk_dim = chunk_dim
        self.device = device
        self.param_dtype = torch.float32  # train rotations in fp32 for stability

        # skew[layer, head] is (chunk_dim, chunk_dim) skew-symmetric
        # We parameterize the upper triangular part; expm gives the rotation.
        n_skew_params = chunk_dim * (chunk_dim - 1) // 2
        # Shape: (n_layers, n_heads, n_skew_params), init to zero (R = I)
        self.skew_params_K = nn.Parameter(torch.zeros(n_layers, n_heads, n_skew_params, dtype=self.param_dtype, device=device))
        self.skew_params_V = nn.Parameter(torch.zeros(n_layers, n_heads, n_skew_params, dtype=self.param_dtype, device=device))

        self._cached_R_K: Optional[torch.Tensor] = None
        self._cached_R_V: Optional[torch.Tensor] = None

    def _params_to_rotation(self, skew_params: torch.Tensor) -> torch.Tensor:
        """skew_params: (..., n_skew_params). Returns rotation (..., chunk_dim, chunk_dim)."""
        d = self.chunk_dim
        n_skew = d * (d - 1) // 2
        assert skew_params.shape[-1] == n_skew
        flat_shape = skew_params.shape[:-1]
        skew = torch.zeros(*flat_shape, d, d, dtype=skew_params.dtype, device=skew_params.device)
        # Fill upper triangle
        idx = 0
        for i in range(d):
            for j in range(i + 1, d):
                skew[..., i, j] = skew_params[..., idx]
                skew[..., j, i] = -skew_params[..., idx]
                idx += 1
        # Matrix exponential of skew-symmetric gives orthogonal
        R = torch.matrix_exp(skew)
        return R

    def _refresh_rotations(self):
        self._cached_R_K = self._params_to_rotation(self.skew_params_K)  # (L, H, d, d)
        self._cached_R_V = self._params_to_rotation(self.skew_params_V)

    def trainable_parameters(self):
        return [self.skew_params_K, self.skew_params_V]

    def _apply_rotation(self, x: torch.Tensor, R_all: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """x: (B, H, T, d_head). R_all: (L, H, chunk_dim, chunk_dim).
        Apply R_all[layer_idx, head] to each chunk of x's last dim."""
        B, H, T, d_head = x.shape
        n_chunks = d_head // self.chunk_dim
        x_chunked = x.view(B, H, T, n_chunks, self.chunk_dim)
        R = R_all[layer_idx].to(x.dtype)  # (H, chunk_dim, chunk_dim)
        # einsum: (B, H, T, n_chunks, chunk_dim) × (H, chunk_dim, chunk_dim') -> (B, H, T, n_chunks, chunk_dim')
        x_rot = torch.einsum("bhtnc,hcd->bhtnd", x_chunked, R)
        return x_rot.view(B, H, T, d_head)

    def _apply_inverse_rotation(self, x: torch.Tensor, R_all: torch.Tensor, layer_idx: int) -> torch.Tensor:
        B, H, T, d_head = x.shape
        n_chunks = d_head // self.chunk_dim
        x_chunked = x.view(B, H, T, n_chunks, self.chunk_dim)
        R = R_all[layer_idx].to(x.dtype)
        # Inverse rotation: R^T
        x_rot = torch.einsum("bhtnc,hdc->bhtnd", x_chunked, R)
        return x_rot.view(B, H, T, d_head)

    def quantize_K(self, k, layer_idx):
        if self._cached_R_K is None:
            self._refresh_rotations()
        k_rot = self._apply_rotation(k, self._cached_R_K, layer_idx)
        k_rot_q = self.inner.quantize_K(k_rot, layer_idx)
        # Straight-through estimator: gradient flows through quantization as identity.
        # Without this, argmax/round/sign break the gradient chain and rotation params get zero gradient.
        k_rot_ste = k_rot + (k_rot_q - k_rot).detach()
        k_q = self._apply_inverse_rotation(k_rot_ste, self._cached_R_K, layer_idx)
        return k_q

    def quantize_V(self, v, layer_idx):
        if self._cached_R_V is None:
            self._refresh_rotations()
        v_rot = self._apply_rotation(v, self._cached_R_V, layer_idx)
        v_rot_q = self.inner.quantize_V(v_rot, layer_idx)
        v_rot_ste = v_rot + (v_rot_q - v_rot).detach()
        v_q = self._apply_inverse_rotation(v_rot_ste, self._cached_R_V, layer_idx)
        return v_q

    def bits_per_value(self):
        # Rotation matrices are calibration overhead, not per-token. Amortized to ~0
        # over a long generation; we report just the inner quantizer's bits.
        return self.inner.bits_per_value()

    @property
    def name(self) -> str:
        return f"LR_{self.inner.name}"
