"""GPU smoke tests for the `hqmq` package.

Skipped entirely when CUDA isn't available, so safe to include in CI.
"""
import pytest
import torch

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="requires CUDA")


@cuda_only
def test_quantize_on_gpu():
    """Construct a quantizer on the GPU and quantize a tensor."""
    import hqmq
    q = hqmq.HQMQ(n_layers=2, n_heads=2, secondary_size=24, radius_bits=3,
                  device="cuda", init="random")
    K = torch.randn(1, 2, 16, 128, dtype=torch.bfloat16, device="cuda")
    Kq = q.quantize_K(K, layer_idx=0)
    assert Kq.shape == K.shape
    assert Kq.device == K.device


@cuda_only
def test_fused_attention_matches_reference_fp32():
    """The Triton fused kernel matches the PyTorch reference in fp32."""
    import hqmq
    B, H_q, H_kv, T, dh = 1, 4, 1, 64, 32
    S = 24
    K_size = 24 * S
    r_qmax = 7

    torch.manual_seed(0)
    Q = torch.randn(B, H_q, T, dh, dtype=torch.float32, device="cuda") * 0.3
    idx = torch.randint(0, K_size, (B, H_kv, T, dh // 4), dtype=torch.int32, device="cuda")
    rq = torch.randint(1, r_qmax + 1, (B, H_kv, T, dh // 4), dtype=torch.int32, device="cuda")
    rs = torch.rand((B, H_kv, T), dtype=torch.float32, device="cuda") + 0.5
    joint = torch.randn((H_kv, K_size, 4), dtype=torch.float32, device="cuda")
    joint = joint / joint.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    O_ref = hqmq.fused_attention_reference(
        Q, idx, rq, rs, joint, idx, rq, rs, joint, r_qmax=r_qmax, causal=True,
    )
    O_tri = hqmq.fused_attention(
        Q, idx, rq, rs, joint, idx, rq, rs, joint, r_qmax=r_qmax, causal=True,
    )
    max_abs_diff = (O_ref - O_tri).abs().max().item()
    # fp32 tolerance: ~5e-4 (matches the published correctness number)
    assert max_abs_diff < 1e-3, f"fp32 kernel diverges from reference: {max_abs_diff}"


@cuda_only
def test_device_info_reports_real_gpu():
    """device_info() returns the real GPU info on CUDA."""
    import hqmq
    info = hqmq.device_info()
    assert info["gpu"] is not None
    assert info["variant_auto_dispatched"] in {"ada", "ada (older)", "hopper", "blackwell"}


@cuda_only
def test_fp8_codebook_fallback_on_pre_hopper():
    """On Ada the use_fp8_codebook=True flag silently falls through to fp16 kernel."""
    import hqmq
    info = hqmq.device_info()
    if info["fp8_codebook_supported"]:
        pytest.skip("This test verifies the Ada fallback; current GPU is Hopper+")

    B, H_q, H_kv, T, dh = 1, 4, 1, 32, 32
    S = 24
    K_size = 24 * S
    r_qmax = 7
    torch.manual_seed(0)
    Q = torch.randn(B, H_q, T, dh, dtype=torch.float16, device="cuda") * 0.3
    idx = torch.randint(0, K_size, (B, H_kv, T, dh // 4), dtype=torch.int32, device="cuda")
    rq = torch.randint(1, r_qmax + 1, (B, H_kv, T, dh // 4), dtype=torch.int32, device="cuda")
    rs = torch.rand((B, H_kv, T), dtype=torch.float16, device="cuda") + 0.5
    joint = torch.randn((H_kv, K_size, 4), dtype=torch.float16, device="cuda")
    joint = joint / joint.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    O_default = hqmq.fused_attention(
        Q, idx, rq, rs, joint, idx, rq, rs, joint, r_qmax=r_qmax, causal=True,
    )
    O_fp8_flag = hqmq.fused_attention(
        Q, idx, rq, rs, joint, idx, rq, rs, joint, r_qmax=r_qmax, causal=True,
        use_fp8_codebook=True,
    )
    # On Ada the flag is ignored, so outputs should be identical
    assert torch.equal(O_default, O_fp8_flag), \
        "use_fp8_codebook should silently fall through on non-Hopper GPUs"


@cuda_only
def test_encode_joint_fp8():
    """FP8 codebook encoding produces float8_e4m3fn tensors."""
    from hqmq.kernels import encode_joint_fp8
    joint = torch.randn(2, 24 * 24, 4, dtype=torch.float16, device="cuda")
    joint = joint / joint.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    joint_fp8 = encode_joint_fp8(joint)
    assert joint_fp8.dtype == torch.float8_e4m3fn
    assert joint_fp8.shape == joint.shape
