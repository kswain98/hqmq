#!/usr/bin/env bash
# Experiments queued for execution on bigger hardware (H100 / H200 / B200).
#
# These were blocked on the 24 GB RTX 4090 by either:
#   (a) model weights not fitting (Llama-3-70B at bf16 = 141 GB),
#   (b) eager-attention QK^T tile exceeding GPU memory at long context, or
#   (c) HQMQ codebook gather intermediate growing with d_h (Gemma-2-9B).
#
# Required: PyTorch >= 2.10, transformers >= 5.3, triton >= 3.4, ~$GPU_MEM GB.
#
# Usage:
#   GPU=H100  bash experiments/big_gpu_queue.sh           # ~80 GB, runs all of tier-A
#   GPU=H200  bash experiments/big_gpu_queue.sh           # 141 GB, runs all of tier-A + tier-B
#   GPU=B200  bash experiments/big_gpu_queue.sh           # 192 GB, runs everything

set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${GPU:-H100}
echo "=== Running queued experiments for ${GPU} ==="

# ============================================================
# Tier A — fits on H100 (80 GB)
# ============================================================
echo ""
echo "[Tier A.1] Gemma-2-9B full HQMQ sweep (50w x 2048, all s24-s192 configs)"
python experiments/43_gemma_sweep.py --max-windows 50 2>&1 | tee runs/H_43_gemma_50w.log

echo ""
echo "[Tier A.2] Llama-3-8B s192 configs at 50w (removes the footnote in main paper)"
python experiments/31_llama_s192_only.py --max-windows 50 2>&1 | tee runs/H_31_llama_s192_50w.log

echo ""
echo "[Tier A.3] Qwen3-8B 16k + 32k needle (HQMQ + Med3x vs naive int4 vs fp16)"
python experiments/50_needle_qwen3_long.py --seq-lens 16384 32768 \
    --n-trials-per-depth 5 2>&1 | tee runs/H_50_needle_qwen3_long.log

echo ""
echo "[Tier A.4] MMLU 5-shot on Qwen3-8B (paper-grade, n=2000 examples)"
python experiments/48_mmlu_mistral.py --model Qwen/Qwen3-8B --max-examples 2000 \
    2>&1 | tee runs/H_48_mmlu_qwen3.log

echo ""
echo "[Tier A.5] Disentanglement ablation on Qwen3-8B (matches Qwen2.5 story)"
python experiments/30_disentangle_outlier.py --model Qwen/Qwen3-8B \
    --max-windows 50 2>&1 | tee runs/H_30_disentangle_qwen3.log

# ============================================================
# Tier B — needs H200 (141 GB) or A100 80GB with offload
# ============================================================
if [[ "${GPU}" == "H200" || "${GPU}" == "B200" || "${GPU}" == "A100-80GB-OFFLOAD" ]]; then
  echo ""
  echo "[Tier B.1] Llama-3-70B HQMQ Pareto sweep (NousResearch mirror)"
  python experiments/29_llama_sweep.py --model NousResearch/Meta-Llama-3-70B \
      --max-windows 20 2>&1 | tee runs/H_29_llama_70b_sweep.log

  echo ""
  echo "[Tier B.2] Llama-3-70B at 128k context: KV memory + needle"
  python experiments/50_needle_qwen3_long.py --model NousResearch/Meta-Llama-3-70B \
      --seq-lens 32768 65536 131072 \
      --n-trials-per-depth 3 2>&1 | tee runs/H_50_needle_llama70b_128k.log

  echo ""
  echo "[Tier B.3] Qwen2.5-72B paper-grade sweep"
  python experiments/09_hqmq_sweep.py --model Qwen/Qwen2.5-72B \
      --seq-len 2048 --max-windows 30 2>&1 | tee runs/H_09_qwen72b_sweep.log
fi

# ============================================================
# Tier C — B200-only frontier models
# ============================================================
if [[ "${GPU}" == "B200" ]]; then
  echo ""
  echo "[Tier C.1] gpt-oss-120b (130 GB in MXFP4)"
  python experiments/40_gptoss_padded.py --model openai/gpt-oss-120b \
      --max-windows 10 2>&1 | tee runs/H_40_gptoss120b.log

  echo ""
  echo "[Tier C.2] DeepSeek-V2-Lite MLA experimental quantization"
  # Requires patching modeling_deepseek.py for transformers 5.3+ compatibility first
  echo "  (MLA adapter implementation pending; skipping)"
fi

# ============================================================
# Fused-kernel benchmarks on the new GPU
# ============================================================
echo ""
echo "[Bench 1] Re-run single-variant fused kernel benchmark"
python experiments/44_fused_attn_test.py 2>&1 | tee runs/H_44_fused_kernel_${GPU}.log

echo ""
echo "[Bench 2] Side-by-side comparison: Ada vs Hopper(FP8) vs Blackwell auto-dispatch"
python experiments/52_bench_all_variants.py 2>&1 | tee runs/H_52_bench_all_${GPU}.log

echo ""
echo "[Bench 3] Per-GPU block-size autotune (use to update _pick_device_config defaults)"
python experiments/51_kernel_autotune.py 2>&1 | tee runs/H_51_autotune_${GPU}.log

echo ""
echo "=== All queued experiments complete ==="
echo "Results saved in runs/H_*.log and runs/*.json"
echo ""
echo "Next steps:"
echo "  1. Compare runs/H_52_bench_all_${GPU}.log to the paper's Mistral-prefill numbers"
echo "  2. If H_51_autotune found a config different from _pick_device_config defaults,"
echo "     update src/quantizers/hqmq_attention.py::_pick_device_config for ${GPU}"
echo "  3. Update paper Table 7 (fused-kernel speedup) with the new GPU's numbers"
