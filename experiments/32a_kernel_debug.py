"""Minimal correctness check for the Triton HQMQ decode kernel."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.quantizers.hqmq_kernel import hqmq_decode_triton


def pytorch_decode(idx, radius_q, radius_scale, joint, r_qmax):
    B, H, T, n_chunks = idx.shape
    K = joint.shape[1]
    d_head = n_chunks * 4
    joint_b = joint.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    idx_gather = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 1, 4)
    joint_b_exp = joint_b.expand(B, H, T, n_chunks, K, 4)
    codeword = joint_b_exp.gather(dim=-2, index=idx_gather.long()).squeeze(-2)
    radius = (radius_q.to(joint.dtype) * radius_scale.unsqueeze(-1)) / r_qmax
    return (codeword * radius.unsqueeze(-1)).view(B, H, T, d_head)


B, H, T, d_head, S = 1, 1, 4, 8, 4
n_chunks = d_head // 4
K = 24 * S
r_qmax = 15

torch.manual_seed(0)
dev = "cuda"
dtype = torch.float32  # Use fp32 for clear correctness check

idx = torch.randint(0, K, (B, H, T, n_chunks), dtype=torch.int32, device=dev)
radius_q = torch.full((B, H, T, n_chunks), r_qmax, dtype=torch.int32, device=dev)  # all 1.0
radius_scale = torch.ones((B, H, T), dtype=dtype, device=dev)
joint = torch.arange(H * K * 4, dtype=dtype, device=dev).view(H, K, 4)

out_pt = pytorch_decode(idx, radius_q, radius_scale, joint, r_qmax)
out_tt = hqmq_decode_triton(idx, radius_q, radius_scale, joint, r_qmax)
print("idx:", idx)
print("pt shape:", out_pt.shape, "tt shape:", out_tt.shape)
print("pt[0,0,0,:8]:", out_pt[0, 0, 0, :8])
print("tt[0,0,0,:8]:", out_tt[0, 0, 0, :8])
print("pt[0,0,1,:8]:", out_pt[0, 0, 1, :8])
print("tt[0,0,1,:8]:", out_tt[0, 0, 1, :8])
print("max abs err:", (out_pt - out_tt).abs().max().item())
