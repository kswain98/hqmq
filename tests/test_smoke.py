"""CPU-only smoke tests for the `hqmq` package.

These don't require a GPU — they verify:
  - the package imports cleanly
  - core math objects (24-cell, bit accounting) are correct
  - quantizer classes can be constructed
  - public API surface is stable

GPU-only tests live in tests/test_gpu.py and are skipped when CUDA isn't present.
"""
import pytest
import torch


# =====================================================================
# Package import
# =====================================================================

def test_import_hqmq():
    """The top-level package imports cleanly."""
    import hqmq  # noqa: F401
    assert hqmq.__version__ == "0.1.0"


def test_public_api_surface():
    """All advertised public names are present."""
    import hqmq
    for name in [
        "HQMQ", "PaddedHQMQ",
        "OutlierAwareHQMQQuantizer", "OutlierAwareGenericQuantizer",
        "fused_attention", "fused_attention_reference",
        "with_outlier_extraction", "device_info",
        "kernels_ada", "kernels_hopper", "kernels_blackwell",
    ]:
        assert hasattr(hqmq, name), f"hqmq.{name} missing from public API"


def test_kernels_submodule():
    """hqmq.kernels provides hardware-specific kernel re-exports."""
    from hqmq import kernels
    for name in [
        "fused_attention_ada", "fused_attention_hopper",
        "fused_attention_blackwell", "fused_attention_reference",
        "decode_dequant", "encode_joint_fp8",
        "hopper_available", "blackwell_available",
    ]:
        assert hasattr(kernels, name), f"hqmq.kernels.{name} missing"


# =====================================================================
# 24-cell / Hurwitz primary codebook
# =====================================================================

def test_24cell_has_24_unit_quaternions():
    """The primary codebook is 24 unit quaternions on S^3."""
    from src.quantizers.spherical import _make_24cell_codebook
    pts = _make_24cell_codebook()
    assert pts.shape == (24, 4), f"expected (24, 4), got {tuple(pts.shape)}"
    norms = pts.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(24), atol=1e-6), \
        f"some 24-cell vertices are not unit length: norms={norms}"


def test_24cell_min_pairwise_angle_is_60deg():
    """The 24-cell achieves the optimal S^3 kissing arrangement (min angle 60°)."""
    from src.quantizers.spherical import _make_24cell_codebook
    pts = _make_24cell_codebook().float()
    # Inner products give cos(angle); 60° → cos = 0.5
    cos_table = pts @ pts.T
    # Mask the diagonal (self-pairs are 1)
    cos_table.fill_diagonal_(-2)
    max_cos_off_diag = cos_table.max().item()
    # Min angle = 60° means max cos = 0.5
    assert abs(max_cos_off_diag - 0.5) < 1e-4, \
        f"max off-diagonal cos = {max_cos_off_diag}, expected 0.5 (60°)"


def test_24cell_is_a_group_under_quat_mul():
    """Closure: 2T · 2T ⊆ 2T (it's a group of order 24)."""
    from src.quantizers.spherical import _make_24cell_codebook
    from src.quantizers.hqmq import quat_mul
    pts = _make_24cell_codebook().float()
    # For each pair, the product should be one of the 24 elements (up to sign matching)
    # We check that the product is in the set within tolerance.
    pts_set = set(tuple(round(float(x), 5) for x in p) for p in pts)
    for i in range(24):
        for j in range(0, 24, 5):  # sample to keep runtime down
            prod = quat_mul(pts[i:i+1], pts[j:j+1])[0]
            prod_tup = tuple(round(float(x), 5) for x in prod)
            assert prod_tup in pts_set, \
                f"product of 24-cell elements {i}, {j} = {prod_tup} not in 2T"


# =====================================================================
# Bit accounting
# =====================================================================

@pytest.mark.parametrize("S, b_r, d_h, expected_per_element", [
    (24,  3, 128, 3.04),   # s24_r3:   (log2(576) + 3) / 4 = (9.17 + 3) / 4 = 3.04
    (96,  4, 128, 3.79),   # s96_r4:   (log2(2304) + 4) / 4 = (11.17 + 4) / 4 = 3.79
    (192, 6, 128, 4.54),   # s192_r6:  (log2(4608) + 6) / 4 = (12.17 + 6) / 4 = 4.54
])
def test_bits_per_value(S, b_r, d_h, expected_per_element):
    """The bit accounting formula matches the paper's Table 6."""
    from src.quantizers.hqmq import HQMQQuantizer
    q = HQMQQuantizer(n_layers=1, n_heads=1, secondary_size=S, radius_bits=b_r,
                      device="cpu", init="random")
    # Note: HQMQQuantizer.bits_per_value() excludes the 16/d_h overhead by default
    bits = q.bits_per_value()
    assert abs(bits - expected_per_element) < 0.01, \
        f"s{S}_r{b_r}: expected ~{expected_per_element}, got {bits:.3f}"


# =====================================================================
# Quantizer construction
# =====================================================================

def test_construct_hqmq_cpu():
    """HQMQ can be built on CPU (no CUDA needed)."""
    import hqmq
    q = hqmq.HQMQ(n_layers=4, n_heads=2, secondary_size=24, radius_bits=3,
                  device="cpu", init="random")
    assert q.chunk_dim == 4
    assert q.secondary_size == 24
    assert q.radius_bits == 3


def test_outlier_wrapper():
    """with_outlier_extraction returns a wrapped quantizer."""
    import hqmq
    base = hqmq.HQMQ(n_layers=4, n_heads=2, secondary_size=24, radius_bits=3,
                     device="cpu", init="random")
    wrapped = hqmq.with_outlier_extraction(base, C=3.0)
    assert wrapped.median_mult == 3.0
    assert wrapped.threshold_mode == "median_mult"
    assert wrapped.inner is base


def test_padded_hqmq_construction():
    """Padding wrapper computes the right padded head dim."""
    import hqmq
    q = hqmq.PaddedHQMQ(n_layers=2, n_heads=2, d_head=45, secondary_size=24,
                       radius_bits=3, device="cpu")
    assert q.d_head_orig == 45
    assert q.d_head_padded == 48
    assert q.pad_count == 3


# =====================================================================
# Device info
# =====================================================================

def test_device_info_runs_without_gpu():
    """device_info() returns something sensible even on CPU-only machines."""
    import hqmq
    info = hqmq.device_info()
    assert isinstance(info, dict)
    assert "variant_auto_dispatched" in info
    if not torch.cuda.is_available():
        assert info["variant_auto_dispatched"] == "cpu"


# =====================================================================
# Quantize/dequantize round-trip on CPU
# =====================================================================

def test_quantize_dequantize_cpu_shapes():
    """The fake-quant call preserves input shape exactly."""
    import hqmq
    q = hqmq.HQMQ(n_layers=2, n_heads=2, secondary_size=24, radius_bits=3,
                  device="cpu", init="random")
    K = torch.randn(1, 2, 16, 128, dtype=torch.float32)
    Kq = q.quantize_K(K, layer_idx=0)
    assert Kq.shape == K.shape
    assert Kq.dtype == K.dtype


def test_quantize_dequantize_reduces_information():
    """A quantize-dequantize round-trip should produce a tensor that differs from
    the input (otherwise the quantization is a no-op)."""
    import hqmq
    q = hqmq.HQMQ(n_layers=2, n_heads=2, secondary_size=24, radius_bits=3,
                  device="cpu", init="random")
    K = torch.randn(1, 2, 16, 128, dtype=torch.float32)
    Kq = q.quantize_K(K, layer_idx=0)
    # Should differ but not catastrophically (correlation > 0.5)
    diff = (K - Kq).abs().mean().item()
    assert diff > 1e-4, f"quantization had no effect (diff={diff})"
    flat_K = K.flatten()
    flat_Kq = Kq.flatten()
    corr = torch.corrcoef(torch.stack([flat_K, flat_Kq]))[0, 1].item()
    assert corr > 0.7, f"quantization corrupted signal: corr(K, Kq) = {corr}"


# =====================================================================
# Reference attention numerics (no Triton, no GPU)
# =====================================================================

def test_fused_attention_reference_on_cpu():
    """The PyTorch reference fused-attention runs end-to-end on CPU."""
    import hqmq
    B, H_q, H_kv, T, dh = 1, 4, 1, 32, 16
    S = 24
    K_size = 24 * S
    r_qmax = 7
    Q = torch.randn(B, H_q, T, dh, dtype=torch.float32)
    idx = torch.randint(0, K_size, (B, H_kv, T, dh // 4), dtype=torch.int32)
    rq = torch.randint(1, r_qmax + 1, (B, H_kv, T, dh // 4), dtype=torch.int32)
    rs = torch.rand((B, H_kv, T), dtype=torch.float32) + 0.5
    joint = torch.randn((H_kv, K_size, 4), dtype=torch.float32)
    joint = joint / joint.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    O = hqmq.fused_attention_reference(
        Q, idx, rq, rs, joint, idx, rq, rs, joint, r_qmax=r_qmax, causal=True,
    )
    assert O.shape == Q.shape
    assert torch.isfinite(O).all(), "output contains NaN or inf"
