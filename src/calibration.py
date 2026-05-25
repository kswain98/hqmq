"""Calibration training for output-aware quantizer parameters.

Given a quantizer with trainable parameters (e.g., LearnedRotationKVQuantizer),
trains them on a small calibration set to minimize ATTENTION-OUTPUT MSE
(not K,V reconstruction MSE).

Mechanism: for each calibration sequence, run the model TWICE:
  1. fp16 forward, capture attention output (or final logits) per layer
  2. quantized forward with current rotations, compute the same outputs
  3. backprop the MSE between (2) and (1) into the rotation parameters

For tractability, we compare FINAL LOGITS only (one tensor per sequence)
rather than per-layer attention outputs — this is a coarser proxy but
keeps the implementation simple. End-to-end logit MSE is the right target
anyway for downstream perplexity / accuracy.
"""

from typing import Iterable, List

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from .model_utils import QuantizedCache


def calibrate_output_aware(
    model,
    quantizer,
    calibration_token_ids: torch.Tensor,   # (n_seqs, seq_len)
    *,
    steps: int = 200,
    lr: float = 1e-3,
    batch_size: int = 2,
    log_every: int = 20,
    device: str = "cuda",
):
    """Train `quantizer.trainable_parameters()` to minimize fp16 vs quantized
    logits MSE on the calibration data.

    Model is frozen; only the quantizer's parameters are trained.
    """
    if not hasattr(quantizer, "trainable_parameters"):
        raise ValueError("quantizer has no trainable_parameters()")
    # Freeze model — we only train rotation params.
    for p in model.parameters():
        p.requires_grad_(False)
    params = quantizer.trainable_parameters()
    opt = torch.optim.Adam(params, lr=lr)
    # Cosine LR schedule from `lr` down to lr/100 over the full run.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr / 100)
    n_seqs, seq_len = calibration_token_ids.shape

    # Enable soft-codebook training mode on the inner quantizer if supported
    # (Walk the .inner chain in case of nested wrappers.)
    deep = quantizer
    while hasattr(deep, "inner"):
        deep = deep.inner
    if hasattr(deep, "set_training"):
        deep.set_training(True, tau=1.0)
        print(f"  Enabled soft-codebook training (tau=1.0) on {type(deep).__name__}")

    # Pre-compute fp16 reference logits (no quantization) — these are the target.
    print("  Pre-computing fp16 reference logits ...")
    ref_logits_all = []
    with torch.no_grad():
        for i in range(0, n_seqs, batch_size):
            batch = calibration_token_ids[i : i + batch_size].to(device)
            cache = DynamicCache()
            out = model(batch, past_key_values=cache, use_cache=True)
            ref_logits_all.append(out.logits.detach())
    ref_logits = torch.cat(ref_logits_all, dim=0)  # (n_seqs, seq_len, vocab_size)

    print("  Training rotations ...")
    losses: List[float] = []
    for step in range(steps):
        # Sample a batch
        ids = torch.randperm(n_seqs)[:batch_size]
        batch = calibration_token_ids[ids].to(device)
        ref = ref_logits[ids]
        # Quantized forward (with grad on rotations)
        quantizer._cached_R_K = None  # force recompute on each step
        quantizer._cached_R_V = None
        cache = QuantizedCache(quantizer)
        out = model(batch, past_key_values=cache, use_cache=True)
        # Cast to fp32 for stable loss
        loss = F.mse_loss(out.logits.float(), ref.float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = sum(p.grad.norm().item() if p.grad is not None else 0.0 for p in params)
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if (step + 1) % log_every == 0 or step == 0:
            cur_lr = sched.get_last_lr()[0]
            print(f"    step {step + 1}/{steps}  loss = {loss.item():.6f}  "
                  f"|grad| = {grad_norm:.4f}  lr = {cur_lr:.2e}")
    # Disable training mode so subsequent eval uses hard argmax
    if hasattr(deep, "set_training"):
        deep.set_training(False)
        print(f"  Disabled soft-codebook training (hard eval)")
    # Final refresh so subsequent eval uses trained rotations (if applicable)
    if hasattr(quantizer, "_refresh_rotations"):
        quantizer._refresh_rotations()
    return losses


def sample_calibration_data(tokenizer, n_seqs: int = 16, seq_len: int = 512) -> torch.Tensor:
    """Tokens drawn from WikiText-103 train split."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    text = "\n\n".join(ds["text"][:5000])
    encodings = tokenizer(text, return_tensors="pt")
    ids = encodings.input_ids[0]  # (T_total,)
    n_total = ids.size(0)
    starts = torch.randperm(n_total - seq_len)[:n_seqs]
    return torch.stack([ids[s : s + seq_len] for s in starts])
