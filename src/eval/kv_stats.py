"""KV-cache statistics probe.

Hooks the attention layers' KV cache update to capture the K and V tensors
being written to the cache (i.e., post-RoPE). Returns per-layer statistics:

- Per-channel mean-absolute values (which channels are outliers?)
- Per-chunk norm distribution (max-to-median ratio per (head))
- Per-(layer, head) chunk-norm quantiles

Used to diagnose why HQMQ works on Mistral-7B but not Qwen2.5-7B.
"""

from typing import Dict

import torch
from transformers import DynamicCache


class StatsCapturingCache(DynamicCache):
    """DynamicCache that records statistics about every (K, V) update without modification."""

    def __init__(self):
        super().__init__()
        self.stats: Dict[int, Dict[str, Dict]] = {}

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        self._record(key_states, "K", layer_idx)
        self._record(value_states, "V", layer_idx)
        return super().update(key_states, value_states, layer_idx, cache_kwargs)

    def _record(self, x: torch.Tensor, name: str, layer_idx: int):
        # x: (B, n_kv_heads, T, d_head). All ops in fp32 for stability.
        x = x.detach().to(torch.float32)
        B, H, T, d_head = x.shape
        chunk_dim = 4
        n_chunks = d_head // chunk_dim
        x_chunked = x.view(B, H, T, n_chunks, chunk_dim)
        chunk_norms = x_chunked.norm(dim=-1)  # (B, H, T, n_chunks)
        per_head_chunk_norms = chunk_norms.permute(1, 0, 2, 3).reshape(H, -1)  # (H, B*T*n_chunks)

        per_head_max = per_head_chunk_norms.amax(dim=-1)
        per_head_med = per_head_chunk_norms.median(dim=-1).values
        per_head_min = per_head_chunk_norms.amin(dim=-1)
        per_head_ratio = per_head_max / per_head_med.clamp(min=1e-8)

        # Channel-wise: which dimensions are outliers?
        channel_abs_mean = x.abs().mean(dim=(0, 1, 2))                     # (d_head,)
        channel_abs_max = x.abs().amax(dim=(0, 1, 2))                      # (d_head,)
        max_channel_ratio = channel_abs_max.amax() / channel_abs_mean.mean().clamp(min=1e-8)

        # Quantiles
        quantiles = torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99, 0.999], device=x.device)
        chunk_norm_q = torch.quantile(chunk_norms.flatten(), quantiles)

        self.stats.setdefault(layer_idx, {})[name] = {
            "per_head_max_to_median": per_head_ratio.cpu().tolist(),
            "channel_abs_max_to_mean": float(max_channel_ratio),
            "chunk_norm_quantiles": chunk_norm_q.cpu().tolist(),
            "chunk_norm_quantile_labels": [0.01, 0.1, 0.5, 0.9, 0.99, 0.999],
            "global_max": float(chunk_norms.amax()),
            "global_median": float(chunk_norms.median()),
        }


@torch.no_grad()
def probe_kv_stats(model, tokenizer, sample_text: str, device: str = "cuda"):
    cache = StatsCapturingCache()
    ids = tokenizer(sample_text, return_tensors="pt").input_ids.to(device)
    # Truncate to model's max length
    if ids.size(1) > 2048:
        ids = ids[:, :2048]
    _ = model(ids, past_key_values=cache, use_cache=True)
    return cache.stats
